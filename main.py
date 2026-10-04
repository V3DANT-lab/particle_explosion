"""
Particle Explosion Ball - main entry point.

What it does
------------
1. Opens the webcam.
2. Runs MediaPipe hand tracking and draws 21 green landmark dots plus a
   red palm-center dot ON TOP OF THE CAMERA PIP (bottom-right).
3. Renders glowing ember particles arranged as a 3D ball using PyOpenGL.
   Each particle is shaded as a real lit sphere (normal reconstructed from
   gl_PointCoord, Lambert + Blinn-Phong + Fresnel) with depth testing so
   the ball genuinely occludes itself, and a second additive pass adds the
   glow halo. A faint reference floor grid anchors the scene in space.
4. The webcam is drawn as a small picture-in-picture in the bottom-right
   corner (NOT as the full background). The 21 green landmark dots and
   the red palm dot are drawn on top of that inset.
5. Watches your gesture: **open your palm and the ball detonates
   immediately** - there is no charging step. The embers burst apart and
   hang in the air (no gravity, no floor), held apart only by air drag
   until they cool and fade. Close your hand into a fist (or move it out
   of frame) to rebuild the ball and go again.

Controls
--------
  SPACE      - trigger the explosion manually
  R          - reset the ball
  F1         - toggle the on-screen FPS / state HUD
  S          - save a screenshot to ./screenshots/
  Q / ESC    - quit

Run it
------
    python main.py
    python main.py --camera 1
    python main.py --width 960 --height 720
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

from hand_tracker import HandTracker, HandState
from particles import ParticleSystem
from renderer import Renderer


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class GameState:
    """High-level state machine.

    States:
      IDLE      - the ball is intact and follows your hand. An OPEN PALM
                  detonates it immediately (no charging step).
      EXPLODED  - embers burst apart and hang in the air. Ignores further
                  gestures until you re-arm.
      COOLING   - the embers have faded out; still waiting for a re-arm.

    Re-arming: close your hand into a fist (held for a moment) or move the
    hand out of frame, and the ball is rebuilt.
    """

    IDLE      = "IDLE"
    EXPLODED  = "EXPLODED"
    COOLING   = "COOLING"


# ---------------------------------------------------------------------------
# Camera source - tries to open the webcam, falls back to synthetic frames
# ---------------------------------------------------------------------------
class CameraSource:
    """OpenCV webcam wrapper with a graceful fallback.

    The requested index is tried first; if it cannot be opened (or fails to
    deliver a frame) we scan the other indices and use the first camera that
    actually works. This matters on machines where the webcam is not /dev/video0
    - without it the app silently shows a synthetic frame instead of the
    camera and it looks like "the camera window is missing".

    If no webcam is available at all we emit a synthetic frame so the rest of
    the app still runs (you'll see the particles + the HUD, and the app falls
    back to an automatic demo cycle so the explosion is still visible).
    """

    SCAN_INDICES = (0, 1, 2, 3, 4, 5)

    def __init__(self, camera_index: int, width: int, height: int) -> None:
        self.cap = None
        self.fallback = False
        self.index = camera_index
        self.resolution = None

        order = [camera_index] + [i for i in self.SCAN_INDICES
                                  if i != camera_index]
        errors = []
        for idx in order:
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if not cap.isOpened():
                cap.release()
                errors.append(f"index {idx}: cannot open")
                continue
            # Ask for the preferred resolution - many cameras accept it.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                errors.append(f"index {idx}: open but no frames")
                continue
            self.cap = cap
            self.index = idx
            self.resolution = (frame.shape[1], frame.shape[0])
            if idx != camera_index:
                print(f"[info] camera index {camera_index} unusable; "
                      f"using camera index {idx} instead.", file=sys.stderr)
            break

        if self.cap is None:
            print(f"[warn] no usable webcam found ({'; '.join(errors)}); "
                  "running with a synthetic background and no hand tracking.",
                  file=sys.stderr)
            self.fallback = True
        else:
            print(f"[info] camera {self.index} streaming at "
                  f"{self.resolution[0]}x{self.resolution[1]}", file=sys.stderr)

        # Synthetic fallback frame state (regenerated a few times a second
        # so we are not spending a full CPU core on mgrid every frame).
        self._synth_w = max(160, min(640, width))
        self._synth_h = max(120, min(480, height))
        self._synth_frame = self._make_synth_frame(0.0)
        self._synth_next = 0.0

    def _make_synth_frame(self, t: float) -> np.ndarray:
        h, w = self._synth_h, self._synth_w
        yy, xx = np.mgrid[0:h, 0:w]
        frame = np.stack([
            ((np.sin(xx / 40 + t) * 0.5 + 0.5) * 30 + 15).astype(np.uint8),
            ((np.sin(yy / 60 + t * 0.7) * 0.5 + 0.5) * 30 + 15).astype(np.uint8),
            np.full((h, w), 25, dtype=np.uint8),
        ], axis=-1)
        return np.ascontiguousarray(frame)

    def read(self):
        if self.fallback:
            now = time.monotonic()
            if now >= self._synth_next:
                self._synth_frame = self._make_synth_frame(now)
                self._synth_next = now + 1.0 / 12.0
            return True, self._synth_frame

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Camera died mid-session - switch to the synthetic source
            # rather than tearing the whole app down.
            print("[warn] camera stream ended; switching to synthetic frames.",
                  file=sys.stderr)
            self.cap.release()
            self.cap = None
            self.fallback = True
            return True, self._synth_frame
        # Mirror the camera so it feels natural
        frame = cv2.flip(frame, 1)
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class ParticleExplosionApp:
    # Auto-demo timings (seconds) used when there is no hand tracking.
    AUTO_IDLE_DELAY     = 1.8
    AUTO_RESET_DELAY    = 3.2

    # How long after losing your hand the ball still re-arms itself.
    HAND_LOSS_GRACE     = 0.6

    def __init__(self, args: argparse.Namespace) -> None:
        self.width = args.width
        self.height = args.height
        self.fps_target = 60
        self.show_hud = True

        # ---- pygame + OpenGL context --------------------------------
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        # Request a *compatibility* profile.
        #
        # This is important: a core-profile context has no default vertex
        # array object, so every glVertexAttribPointer() call fails with
        # GL_INVALID_OPERATION and nothing is ever drawn - the window comes
        # up blank even though everything "succeeded". The shaders are also
        # GLSL 1.20, which a core profile may reject outright.
        try:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_PROFILE_MASK,
                pygame.GL_CONTEXT_PROFILE_COMPATIBILITY)
        except Exception:
            # Older pygame builds may not have these constants; fall back to
            # whatever context SDL gives us.
            pass

        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.DOUBLEBUF | pygame.OPENGL,
        )
        pygame.display.set_caption("Particle Explosion Ball - PyOpenGL + MediaPipe")

        self.renderer = Renderer(self.width, self.height)
        self.particles = ParticleSystem(num_particles=1000)

        # Hand tracking is optional: if the model is missing or the driver
        # refuses to start it we still want a working window.
        self.tracker = None
        try:
            self.tracker = HandTracker()
        except Exception as exc:
            print(f"[warn] hand tracking unavailable: {exc}", file=sys.stderr)
        self.tracking_ok = bool(self.tracker is not None
                                and self.tracker.available)

        self.camera = CameraSource(args.camera, self.width, self.height)

        # Auto-demo drives the state machine when there is nothing to
        # gesture at, so the explosion is always demonstrable.
        self.auto_demo = bool(args.auto_demo and
                              (self.camera.fallback or not self.tracking_ok))
        if self.auto_demo:
            print("[info] auto-demo enabled (no hand input available).",
                  file=sys.stderr)

        # State
        self.state = GameState.IDLE
        self.flash_alpha = 0.0
        self.shockwave = None
        self.last_time = time.monotonic()
        self.fps_smooth = 60.0
        self._pending_screenshot = False
        self._auto_timer = 0.0
        self._last_hand_seen = -1e9
        self._last_palm_world = np.array([0.0, 0.4, 0.0], dtype=np.float32)

        # Optional: a tiny font for the HUD
        self.font = pygame.font.SysFont("DejaVu Sans, Arial", 16)
        self._hud_cache = None
        self._hud_key = None

        # Screenshot output directory
        self.shot_dir = Path("./screenshots")
        self.shot_dir.mkdir(parents=True, exist_ok=True)
        self.shot_counter = 0

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False
                        elif event.key == pygame.K_r:
                            self.reset_ball()
                            self._auto_timer = 0.0
                        elif event.key == pygame.K_SPACE:
                            self._manual_explosion()
                        elif event.key == pygame.K_F1:
                            self.show_hud = not self.show_hud
                        elif event.key == pygame.K_s:
                            # Defer to AFTER render so we capture the
                            # current frame, not the previous one.
                            self._pending_screenshot = True

                dt = clock.tick(self.fps_target) / 1000.0
                if dt <= 0.0:
                    continue
                dt = min(dt, 1.0 / 30.0)

                # Smooth FPS reading
                inst_fps = 1.0 / dt
                self.fps_smooth = self.fps_smooth * 0.9 + inst_fps * 0.1

                self.step(dt)

                # Deferred screenshot AFTER render but BEFORE flip so we
                # capture the freshly rendered back buffer.
                if self._pending_screenshot:
                    self._pending_screenshot = False
                    self.take_screenshot()

                pygame.display.flip()
        finally:
            self.shutdown()

    # ------------------------------------------------------------------ step
    def step(self, dt: float) -> None:
        """Run one frame."""
        # 1. Read camera + run hand tracking
        ok, frame_bgr = self.camera.read()
        hand_state = HandState()
        if ok and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.renderer.update_camera_frame(frame_rgb)
            if (self.tracking_ok and not self.camera.fallback
                    and frame_bgr is not None):
                hand_state = self.tracker.process(frame_bgr)
        else:
            self.camera.fallback = True

        # 2. Update game state machine based on hand gesture
        self._update_state(hand_state, dt)

        # EXPLODED -> COOLING once the debris has come to rest
        if self.state == GameState.EXPLODED and self.particles.is_settled:
            self.state = GameState.COOLING

        # 3. Update particles. The ball follows the hand while it is armed
        # or idle, but never mid-explosion.
        target_center = None
        if hand_state.hand_present and hand_state.palm is not None:
            target_center = self._hand_to_world(hand_state.palm)
            # Remember it so a re-arm rebuilds the ball where your hand is.
            self._last_palm_world = target_center

        if self.state in (GameState.EXPLODED, GameState.COOLING):
            target_center = None

        self.particles.update(dt, target_center=target_center)

        # 4. Update flash + shockwave
        if self.flash_alpha > 0.0:
            self.flash_alpha = max(0.0, self.flash_alpha - dt * 4.2)
        if self.shockwave is not None:
            self.shockwave["radius"] += dt * 600.0   # px per second
            self.shockwave["alpha"] = max(0.0,
                                          self.shockwave["alpha"] - dt * 1.4)
            if self.shockwave["alpha"] <= 0.01:
                self.shockwave = None

        # 5. Render (the HUD is composited by OpenGL, not by pygame)
        hud = self._draw_hud(hand_state) if self.show_hud else None
        self.renderer.render(
            particle_system=self.particles,
            hand_state=hand_state,
            flash_alpha=self.flash_alpha,
            shockwave=self.shockwave,
            time_sec=time.monotonic(),
            hud_surface=hud,
        )

    # -------------------------------------------------------- state machine
    def _hand_seen_recently(self) -> bool:
        """True if a hand was detected within HAND_LOSS_GRACE seconds."""
        return (time.monotonic() - self._last_hand_seen) < self.HAND_LOSS_GRACE

    @staticmethod
    def _hand_to_world(palm) -> np.ndarray:
        """Map normalized, mirrored image coords into the visible world box."""
        hx, hy = float(palm[0]), float(palm[1])
        wx = (hx - 0.5) * 6.0
        wy = 0.5 + (0.5 - hy) * 3.0
        # Clamp so the ball can never wander off screen.
        wx = max(-2.7, min(2.7, wx))
        wy = max(-0.7, min(1.9, wy))
        return np.array([wx, wy, 0.0], dtype=np.float32)

    def _update_state(self, hand_state: HandState, dt: float) -> None:
        g = hand_state.gesture

        if hand_state.hand_present:
            self._last_hand_seen = time.monotonic()

        # ---- automatic demo cycle (no camera / no hand tracking) ----
        if self.auto_demo:
            self._auto_timer += dt
            if self.state == GameState.IDLE and \
                    self._auto_timer >= self.AUTO_IDLE_DELAY:
                self._auto_timer = 0.0
                self._trigger_explosion(hand_state)
            elif self.state in (GameState.EXPLODED, GameState.COOLING) \
                    and self._auto_timer >= self.AUTO_RESET_DELAY:
                self._auto_timer = 0.0
                self.reset_ball()
            return

        # ---- gesture driven ----

        # Losing the hand re-arms the ball, so the app can never get stuck
        # with an exploded ball you cannot reset.
        #
        # This only applies if a hand was actually visible a moment ago.
        # Without that guard, pressing SPACE with no hand in frame would
        # explode and then instantly undo itself on the very next frame.
        if not hand_state.hand_present:
            if self.state != GameState.IDLE and self._hand_seen_recently():
                self.reset_ball()
            return

        # A closed fist (held long enough to be unambiguous) rebuilds the
        # ball. Requiring the full fist hold prevents the classifier
        # flickering between "fist" and "open" from machine-gunning
        # explosions.
        if g == "fist":
            if (self.state != GameState.IDLE
                    and hand_state.fist_progress >= 1.0):
                self.reset_ball()
            return

        # An OPEN PALM detonates immediately - there is no charging step.
        if g == "open":
            if (self.state == GameState.IDLE
                    and self.particles.state == "ball"):
                self._trigger_explosion(hand_state)
            return

    def _trigger_explosion(self, hand_state: HandState) -> None:
        # Choose explosion center: project the 3D ball's world-space center to
        # screen space so the shockwave + flash overlay land exactly where the
        # 3D ball appears on screen (NOT at the hand's image position, which
        # lives inside the small PIP).
        ball_world = self.particles.center
        cx, cy = self.renderer.project_world_to_screen(ball_world)

        # Particle explosion in 3D world space (use ball center)
        self.particles.explode()

        # Screen-space flash + shockwave - centered on the 3D ball
        self.flash_alpha = 0.72
        self.shockwave = {
            "center": (cx, cy),
            "radius": 10.0,
            "alpha": 1.0,
            "color": (1.0, 0.55, 0.15),
        }

        self.state = GameState.EXPLODED
        self._auto_timer = 0.0

    def _manual_explosion(self) -> None:
        """SPACE - detonate without needing a hand gesture."""
        if self.state == GameState.IDLE and self.particles.state == "ball":
            self._trigger_explosion(HandState())

    # ---------------------------------------------------------------- reset
    def reset_ball(self) -> None:
        self.particles.reset_ball(center=self._last_palm_world)
        self.state = GameState.IDLE
        self.flash_alpha = 0.0
        self.shockwave = None
        self._auto_timer = 0.0

    # ------------------------------------------------------------------ HUD
    def _draw_hud(self, hand_state: HandState):
        """Compose the HUD text and hand it to the renderer as a texture.

        Note: pygame cannot blit onto an OPENGL surface, so this returns an
        RGBA surface which renderer._draw_hud() uploads and draws itself.
        """
        lines = [
            f"FPS: {self.fps_smooth:5.1f}",
            f"State: {self.state}",
            f"Gesture: {hand_state.gesture}"
            + (f"  re-arm={hand_state.fist_progress*100:3.0f}%"
               if hand_state.gesture == "fist" else ""),
            f"Particles: {len(self.particles.positions)}",
            f"Hand tracking: {'on' if self.tracking_ok else 'off'}",
            "[SPACE] explode  [R] reset  [F1] HUD  [S] screenshot  [Q] quit",
        ]
        hint = ""
        if self.state == GameState.IDLE:
            hint = "Open your palm to detonate  (or press SPACE)"
        elif self.state in (GameState.EXPLODED, GameState.COOLING):
            hint = "BOOM! Close your fist to rebuild the ball"
        if hint:
            lines.append("")

        key = tuple(lines)
        if key != self._hud_key:
            self._hud_key = key
            self._hud_cache = self._compose_hud(lines, hint)

        return self._hud_cache

    def _compose_hud(self, lines, hint: str):
        """Render HUD text into a tight RGBA surface."""
        pad = 8
        line_h = 18
        rendered = [self.font.render(s, True, (255, 255, 255)) for s in lines]
        widths = [s.get_width() for s in rendered]
        hint_surf = self.font.render(hint, True, (255, 220, 100)) if hint else None

        w = (max(widths) if widths else 0) + pad * 2
        h = len(lines) * line_h + pad * 2
        if hint_surf is not None:
            h += 10 + line_h

        surface = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
        y = pad
        for surf in rendered:
            surface.blit(_shadow(surf), (pad + 1, y + 1))   # drop shadow
            surface.blit(surf, (pad, y))
            y += line_h

        if hint_surf is not None:
            y += 10
            surface.blit(_shadow(hint_surf), (pad + 1, y + 1))
            surface.blit(hint_surf, (pad, y))

        return surface

    # ----------------------------------------------------------- screenshot
    def take_screenshot(self) -> None:
        # Read pixels from the GL back buffer. glFinish() first so the read
        # cannot race the rasteriser (otherwise we can capture a stale or
        # partially-drawn frame).
        from OpenGL.GL import glFinish, glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
        glFinish()
        data = glReadPixels(0, 0, self.width, self.height,
                            GL_RGB, GL_UNSIGNED_BYTE)
        # GL stores bottom row first; flip vertically.
        arr = np.frombuffer(data, dtype=np.uint8).reshape(
            (self.height, self.width, 3))
        arr = np.flipud(arr)
        # Convert to BGR for OpenCV
        arr_bgr = np.ascontiguousarray(arr[..., ::-1])
        path = self.shot_dir / f"shot_{self.shot_counter:04d}.png"
        cv2.imwrite(str(path), arr_bgr)
        self.shot_counter += 1
        print(f"[screenshot] saved {path}", file=sys.stderr)

    # ------------------------------------------------------------- shutdown
    def shutdown(self) -> None:
        if self.tracker is not None:
            try:
                self.tracker.close()
            except Exception:
                pass
        self.camera.release()
        pygame.quit()


def _shadow(surf):
    """Return a black silhouette of `surf` for the HUD drop shadow."""
    shadow = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    shadow.blit(surf, (0, 0))
    shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
    return shadow


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0,
                   help="OpenCV camera index (default 0).")
    p.add_argument("--width", type=int, default=1280,
                   help="Window width in pixels (default 1280).")
    p.add_argument("--height", type=int, default=720,
                   help="Window height in pixels (default 720).")
    p.add_argument("--auto-demo", dest="auto_demo", action="store_true",
                   default=True,
                   help="Run the automatic demo cycle when there is no "
                        "camera / no hand tracking (default: on).")
    p.add_argument("--no-auto-demo", dest="auto_demo", action="store_false",
                   help="Disable the automatic demo cycle.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = ParticleExplosionApp(args)
    app.run()


if __name__ == "__main__":
    main()

"""
Functional test for the paths the frame-capture harness cannot reach:

  1. no-camera fallback (synthetic frames) + auto-demo cycle
  2. hand-landmark overlay rendering (the PIP dots)
  3. HUD build + F1 toggle
  4. screenshot writing (S key path)
  5. reset / state-machine transitions
  6. clean shutdown
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pygame
from OpenGL.GL import GL_NO_ERROR, glGetError, glReadPixels, GL_RGB, \
    GL_UNSIGNED_BYTE, glFinish

import main as app_main
from hand_tracker import HandState

FAILS = []


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}{('  -> ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(label)


def grab(app):
    """Return an (h, w, 3) float RGB array, top-left origin, 0..255."""
    glFinish()
    d = glReadPixels(0, 0, app.width, app.height, GL_RGB, GL_UNSIGNED_BYTE)
    a = np.frombuffer(d, np.uint8).reshape(app.height, app.width, 3)[::-1].copy()
    return a.astype(np.float32)


def luma(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def build(width=800, height=600, camera=0):
    sys.argv = ["x", "--width", str(width), "--height", str(height),
                "--camera", str(camera)]
    return app_main.ParticleExplosionApp(app_main.parse_args())


# ---------------------------------------------------------------- test 1
print("\n1. No-camera fallback + auto-demo")
# Make camera discovery fail so we exercise the synthetic path.
app_main.CameraSource.SCAN_INDICES = ()
app = build()
check("camera fell back to synthetic frames", app.camera.fallback)
check("auto-demo enabled when there is no input", app.auto_demo)
check("synthetic frame produced",
      app.camera.read()[0] and app.camera.read()[1] is not None)

seen = set()
for i in range(400):
    app.step(1 / 60.0)
    seen.add(app.state)
check("auto-demo cycled through states",
      {"IDLE", "EXPLODED"} <= seen, f"states seen: {sorted(seen)}")
check("particles actually exploded at some point",
      app.particles.state in ("exploding", "settled", "ball"))
check("no GL errors", glGetError() == GL_NO_ERROR)
app.shutdown()
app_main.CameraSource.SCAN_INDICES = (0, 1, 2, 3, 4, 5)

# ---------------------------------------------------------------- test 2
print("\n2. Hand landmark overlay + HUD + screenshot")
app = build()
lm = np.array([
    [0.50, 0.60], [0.46, 0.55], [0.43, 0.51], [0.40, 0.48], [0.38, 0.45],
    [0.47, 0.52], [0.45, 0.45], [0.44, 0.40], [0.43, 0.36],
    [0.50, 0.50], [0.50, 0.42], [0.50, 0.36], [0.50, 0.31],
    [0.53, 0.51], [0.54, 0.44], [0.55, 0.39], [0.55, 0.35],
    [0.56, 0.53], [0.58, 0.48], [0.59, 0.44], [0.60, 0.41],
], dtype=np.float32)
hs = HandState(landmarks=lm, palm=lm[[0, 5, 9, 13, 17]].mean(axis=0),
               gesture="open", hand_present=True)

ok, frame = app.camera.read()
app.renderer.update_camera_frame(
    __import__("cv2").cvtColor(frame, __import__("cv2").COLOR_BGR2RGB))

pip_before = luma(grab(app))[
    app.renderer.height - app.renderer._pip_size()[1] - 20:
    app.renderer.height - 30,
    app.renderer.width - app.renderer._pip_size()[0] - 20:
    app.renderer.width - 30].mean()

hud = app._draw_hud(hs)
check("HUD surface built with non-zero size",
      hud is not None and hud.get_width() > 50 and hud.get_height() > 20,
      f"{hud.get_width()}x{hud.get_height()}")

app.renderer.render(particle_system=app.particles, hand_state=hs,
                    flash_alpha=0.0, shockwave=None, time_sec=1.0,
                    hud_surface=hud)
L = luma(grab(app))
check("rendered frame is not black", L.max() > 60, f"max luma {L.max():.1f}")
check("no GL errors after overlay render", glGetError() == GL_NO_ERROR)

# The green landmark dots should be present inside the PIP.
pw, ph = app.renderer._pip_size()
px = app.renderer.width - pw - app.renderer.PIP_MARGIN
py = app.renderer.height - ph - app.renderer.PIP_MARGIN
sub = L[py:py + ph, px:px + pw]
rgb_green = ((sub > 60)).sum()
check("hand landmark dots rendered inside the PIP", rgb_green > 50,
      f"{rgb_green} bright px in PIP")

# ---------------------------------------------------------------- test 3
print("\n3. Screenshot + reset")
shots_before = len(list(app.shot_dir.glob("*.png")))
app.take_screenshot()
shots_after = len(list(app.shot_dir.glob("*.png")))
check("screenshot written", shots_after == shots_before + 1,
      f"{shots_before} -> {shots_after}")
newest = sorted(app.shot_dir.glob("*.png"))[-1]
import cv2
im = cv2.imread(str(newest))
check("screenshot is a valid non-empty image",
      im is not None and im.shape[0] == app.height and im.shape[1] == app.width,
      f"{None if im is None else im.shape}")

app.state = app_main.GameState.EXPLODED
app.flash_alpha = 0.3
app.reset_ball()
check("reset restores IDLE + full lives + ball state",
      app.state == app_main.GameState.IDLE and app.flash_alpha == 0.0
      and app.shockwave is None and app.particles.state == "ball"
      and float(app.particles.lives.min()) == 1.0)

app.show_hud = False
seen_hud = []
orig_render = app.renderer.render


def spy(**kw):
    seen_hud.append(kw.get("hud_surface"))
    return orig_render(**kw)


app.renderer.render = spy
app.step(1 / 60.0)
app.show_hud = True
app.step(1 / 60.0)
check("F1 toggle: no HUD texture when hidden, HUD texture when shown",
      seen_hud[0] is None and seen_hud[1] is not None,
      f"hidden={seen_hud[0]} shown={'surface' if seen_hud[1] else None}")
app.renderer.render = orig_render

app.shutdown()

# ---------------------------------------------------------------- test 4
print("\n4. Gesture mechanism: open palm detonates instantly, fist re-arms")
app = build()
ok, frame = app.camera.read()
app.renderer.update_camera_frame(
    __import__("cv2").cvtColor(frame, __import__("cv2").COLOR_BGR2RGB))


def fake_state(gesture, progress=1.0, present=True):
    return HandState(landmarks=lm, palm=lm[[0, 5, 9, 13, 17]].mean(axis=0),
                     gesture=gesture, fist_progress=progress,
                     hand_present=present)


app._last_palm_world = app._hand_to_world(lm[[0, 5, 9, 13, 17]].mean(axis=0))
app.state = app_main.GameState.IDLE
app.particles.reset_ball(center=app._last_palm_world)

# An open palm must detonate on the very first frame - no charging gate.
app._update_state(fake_state("open"), 1 / 60.0)
check("open palm detonates immediately (no charging step)",
      app.state == app_main.GameState.EXPLODED
      and app.particles.state == "exploding")

# While exploded, gestures must not re-trigger.
app._update_state(fake_state("open"), 1 / 60.0)
check("already-exploded state ignores further open palms",
      app.state == app_main.GameState.EXPLODED)

# A half-closed fist must NOT re-arm (avoids machine-gunning).
app._update_state(fake_state("fist", progress=0.4), 1 / 60.0)
check("half-closed fist does not re-arm", app.state == app_main.GameState.EXPLODED)

# A fully closed fist re-arms.
app._update_state(fake_state("fist", progress=1.0), 1 / 60.0)
check("fully closed fist re-arms the ball",
      app.state == app_main.GameState.IDLE
      and app.particles.state == "ball")

# Losing the hand must also re-arm, so it can never get stuck exploded.
app._update_state(fake_state("open"), 1 / 60.0)
app._update_state(fake_state("none", present=False), 1 / 60.0)
check("losing the hand re-arms",
      app.state == app_main.GameState.IDLE
      and app.particles.state == "ball")

# ---------------------------------------------------------------- test 5
print("\n5. Embers float: no gravity, no floor, no falling")
app.particles.reset_ball(center=np.array([0.0, 0.4, 0.0], dtype=np.float32))
app.particles.explode()
y0 = float(app.particles.positions[:, 1].mean())

# Sample the whole burst: the mean height must not trend downward.
heights = [y0]
for i in range(150):
    app.particles.update(1 / 60.0)
    heights.append(float(app.particles.positions[:, 1].mean()))

drift = heights[-1] - heights[0]
check("cloud does not sink (no gravity)",
      abs(drift) < 0.35, f"mean-y drift over 2.5s = {drift:+.3f} units")

from particles import FLOOR_Y
below_grid = int((app.particles.positions[:, 1] < FLOOR_Y).sum())
check("cloud never drops below the reference floor grid",
      below_grid == 0,
      f"{below_grid} particles below y={FLOOR_Y} "
      f"(cloud min y = {float(app.particles.positions[:, 1].min()):.2f})")

app.particles.reset_ball(center=np.array([0.0, 0.4, 0.0], dtype=np.float32))
spread_start = float(np.linalg.norm(
    app.particles.positions - app.particles.center, axis=1).mean())
app.particles.explode()
for _ in range(60):                       # 1 second after the burst
    app.particles.update(1 / 60.0)
spread_later = float(np.linalg.norm(
    app.particles.positions - app.particles.center, axis=1).mean())
check("embers actually separate from each other",
      spread_later > spread_start * 1.5,
      f"mean radius {spread_start:.2f} -> {spread_later:.2f}")

zmax = float(app.particles.positions[:, 2].max())
check("embers stay in front of the camera (z < 2.6)",
      zmax <= 2.6 + 1e-3, f"max z = {zmax:.2f}")

# ---------------------------------------------------------------- test 6
print("\n6. Landmark skeleton + dots visible in the PIP")
app.state = app_main.GameState.IDLE
app.particles.reset_ball()
hs_open = fake_state("open")
app.renderer.render(particle_system=app.particles, hand_state=hs_open,
                    flash_alpha=0.0, shockwave=None, time_sec=1.0,
                    hud_surface=app._draw_hud(hs_open))
rgb = grab(app)
L = luma(rgb)
pw, ph = app.renderer._pip_size()
px = app.renderer.width - pw - app.renderer.PIP_MARGIN
py = app.renderer.height - ph - app.renderer.PIP_MARGIN
sub = L[py:py + ph, px:px + pw]
check("landmark overlay renders bright pixels in the PIP",
      sub.max() > 100, f"PIP max luma {sub.max():.1f}")
greenish = int(((rgb[..., 1] > 90) & (rgb[..., 1] > rgb[..., 0] + 25)).sum())
reddish = int(((rgb[..., 0] > 90) & (rgb[..., 0] > rgb[..., 1] + 35)).sum())
check("green landmark/bone pixels present", greenish > 150,
      f"{greenish} green px")
check("red palm pixels present", reddish > 40, f"{reddish} red px")
check("no GL errors", glGetError() == GL_NO_ERROR)
app.shutdown()

print("\n" + "=" * 62)
if FAILS:
    print(f"FAILURES ({len(FAILS)}):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("ALL FUNCTIONAL CHECKS PASSED")

"""
OpenGL renderer for the particle explosion demo.

Owns
-----
  * Shader compilation + uniform binding helpers
  * Nine shader programs: background / floor_grid / camera_pip /
    particle_core / particle_glow / overlay / flash / blit / shockwave
  * VBOs for the particles, background quad, floor grid, camera PIP quad,
    hand-overlay points, flash quads and the shockwave ring
  * A VAO (vertex array object) so the renderer works even when the driver
    hands us a *core profile* context
  * Camera frame upload as a GL texture each tick
  * Camera math (perspective matrix, look-at)
  * Drawing pipeline per frame:
        1. Dark vignette background
        2. 3D reference floor grid (depth written)
        3. Particles as lit 3D spheres (depth tested + written)
        4. Additive glow halo pass (depth tested, no depth write)
        5. Shockwave ring
        6. Camera picture-in-picture (bottom-right) + frame bars
        7. Hand landmark dots + palm dot, inside the PIP
        8. HUD text layer (uploaded as a texture; pygame cannot blit onto
           an OPENGL surface)
        9. Flash (additive white quad)

Context notes
-------------
The app must request a *compatibility* profile. In a core profile there is
no default vertex array object, so glVertexAttribPointer() fails with
GL_INVALID_OPERATION and every draw produces nothing - i.e. a black window.
We create a VAO when one is available so we also survive a core context.
"""

from __future__ import annotations

import ctypes
from typing import Optional, Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DYNAMIC_DRAW,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LEQUAL,
    GL_LESS,
    GL_LINEAR,
    GL_LINE_LOOP,
    GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_PROGRAM_POINT_SIZE,
    GL_RGBA,
    GL_RGB,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glAttachShader,
    glBindBuffer,
    glBindTexture,
    glBlendFunc,
    glBufferData,
    glBufferSubData,
    glClear,
    glClearColor,
    glClearDepth,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteShader,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glDisableVertexAttribArray,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenTextures,
    glGetProgramInfoLog,
    glGetShaderInfoLog,
    glGetUniformLocation,
    glLineWidth,
    glLinkProgram,
    glPointSize,
    glShaderSource,
    glTexImage2D,
    glTexParameteri,
    glUniform1f,
    glUniform1i,
    glUniform2f,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)

from particles import FLOOR_Y, NUM_PARTICLES
from shaders import SHADER_SOURCES
from hand_tracker import HAND_CONNECTIONS


# ---------------------------------------------------------------------------
# Shader helpers
# ---------------------------------------------------------------------------
def _compile_shader(src: str, shader_type) -> int:
    """Compile a single GLSL shader.

    Only *errors* raise. Some drivers emit warnings (unused varyings,
    implicit conversions, ...) through the info log even for a perfectly
    valid shader, and turning those into exceptions makes the app fail to
    start on perfectly healthy drivers.
    """
    shader = glCreateShader(shader_type)
    glShaderSource(shader, [src])
    glCompileShader(shader)
    log = glGetShaderInfoLog(shader)
    if log:
        if isinstance(log, bytes):
            log = log.decode("utf-8", errors="replace")
        low = log.lower()
        if "error" in low or "failed" in low:
            raise RuntimeError(f"Shader compile failed:\n{log}")
        print(f"[gl] shader warning (ignored): {log.strip()}")
    return shader


def _link_program(vert_src: str, frag_src: str) -> int:
    vs = _compile_shader(vert_src, GL_VERTEX_SHADER)
    fs = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    log = glGetProgramInfoLog(prog)
    if log:
        if isinstance(log, bytes):
            log = log.decode("utf-8", errors="replace")
        low = log.lower()
        if "error" in low or "failed" in low:
            raise RuntimeError(f"Program link failed:\n{log}")
        print(f"[gl] program warning (ignored): {log.strip()}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


# ---------------------------------------------------------------------------
# Camera math
# ---------------------------------------------------------------------------
def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    fov = np.radians(fov_deg)
    f = 1.0 / np.tan(fov / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye, target, up) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -s.dot(eye)
    m[1, 3] = -u.dot(eye)
    m[2, 3] = f.dot(eye)
    return m


def _project_world_to_screen(world_pos, mvp, width, height):
    """Project a 3D world-space point to 2D screen pixel coords.

    Returns (px, py) in pixel space (origin top-left). If the point is
    behind the camera, returns the screen center as a fallback.
    """
    p = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0],
                 dtype=np.float32)
    clip = mvp @ p
    if clip[3] <= 0.0:
        return width / 2.0, height / 2.0
    ndc = clip[:3] / clip[3]
    px = (ndc[0] * 0.5 + 0.5) * width
    py = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    return float(px), float(py)


# Shared camera configuration (eye, target, up, fov) - kept here so the
# explosion code can project the ball center to screen space without
# duplicating the camera math.
CAMERA_EYE    = np.array([0.0, 0.5, 5.0], dtype=np.float32)
CAMERA_TARGET = np.array([0.0, 0.4, 0.0], dtype=np.float32)
CAMERA_UP     = np.array([0.0, 1.0, 0.0], dtype=np.float32)
CAMERA_FOV    = 60.0

# Key light direction in *view space* (z toward the camera). Placed up and
# to the left so spheres get a readable highlight and a shaded lower-right.
LIGHT_DIR_VIEW = np.array([-0.45, 0.62, 0.64], dtype=np.float32)

# Extent of the reference floor grid, in world units.
FLOOR_EXTENT = 14.0


class Renderer:
    """Encapsulates all OpenGL state for the demo."""

    SHOCKWAVE_SEGMENTS = 96
    PIP_WIDTH   = 320     # px (the height is derived from camera aspect)
    PIP_MARGIN  = 24      # px from the bottom-right corner

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        # ---- Compile all programs once -------------------------------
        self.prog_background   = _link_program(*SHADER_SOURCES["background"])
        self.prog_floor        = _link_program(*SHADER_SOURCES["floor_grid"])
        self.prog_camera_pip   = _link_program(*SHADER_SOURCES["camera_pip"])
        self.prog_particle_core = _link_program(*SHADER_SOURCES["particle_core"])
        self.prog_particle_glow = _link_program(*SHADER_SOURCES["particle_glow"])
        self.prog_overlay      = _link_program(*SHADER_SOURCES["overlay"])
        self.prog_flash        = _link_program(*SHADER_SOURCES["flash"])
        self.prog_blit         = _link_program(*SHADER_SOURCES["blit"])
        self.prog_shockwave    = _link_program(*SHADER_SOURCES["shockwave"])

        # ---- Uniform locations ---------------------------------------
        self._u_bg_brightness = glGetUniformLocation(self.prog_background,
                                                     "u_brightness")
        self._u_bg_time       = glGetUniformLocation(self.prog_background,
                                                     "u_time")

        self._u_fl_mvp    = glGetUniformLocation(self.prog_floor, "u_mvp")
        self._u_fl_y      = glGetUniformLocation(self.prog_floor, "u_floor_y")
        self._u_fl_color  = glGetUniformLocation(self.prog_floor, "u_color")
        self._u_fl_fade   = glGetUniformLocation(self.prog_floor, "u_fade")

        self._u_pip_resolution = glGetUniformLocation(self.prog_camera_pip,
                                                      "u_resolution")
        self._u_pip_texture    = glGetUniformLocation(self.prog_camera_pip,
                                                      "u_texture")
        self._u_pip_alpha      = glGetUniformLocation(self.prog_camera_pip,
                                                      "u_alpha")
        self._u_pip_uv_min     = glGetUniformLocation(self.prog_camera_pip,
                                                      "u_uv_min")
        self._u_pip_uv_max     = glGetUniformLocation(self.prog_camera_pip,
                                                      "u_uv_max")

        self._u_pc_mvp         = glGetUniformLocation(self.prog_particle_core,
                                                      "u_mvp")
        self._u_pc_resolution  = glGetUniformLocation(self.prog_particle_core,
                                                      "u_resolution")
        self._u_pc_scale       = glGetUniformLocation(self.prog_particle_core,
                                                      "u_point_scale")
        self._u_pc_time        = glGetUniformLocation(self.prog_particle_core,
                                                      "u_time")
        self._u_pc_light       = glGetUniformLocation(self.prog_particle_core,
                                                      "u_light_dir")
        self._u_pc_alpha_scale = glGetUniformLocation(self.prog_particle_core,
                                                      "u_alpha_scale")

        self._u_pg_mvp    = glGetUniformLocation(self.prog_particle_glow,
                                                 "u_mvp")
        self._u_pg_res    = glGetUniformLocation(self.prog_particle_glow,
                                                 "u_resolution")
        self._u_pg_scale  = glGetUniformLocation(self.prog_particle_glow,
                                                 "u_point_scale")
        self._u_pg_time   = glGetUniformLocation(self.prog_particle_glow,
                                                 "u_time")
        self._u_pg_grow   = glGetUniformLocation(self.prog_particle_glow,
                                                 "u_grow")
        self._u_pg_str    = glGetUniformLocation(self.prog_particle_glow,
                                                 "u_glow_strength")

        self._u_o_resolution = glGetUniformLocation(self.prog_overlay,
                                                    "u_resolution")
        self._u_o_offset     = glGetUniformLocation(self.prog_overlay,
                                                    "u_offset")
        self._u_o_scale      = glGetUniformLocation(self.prog_overlay,
                                                    "u_scale")
        self._u_o_alpha      = glGetUniformLocation(self.prog_overlay,
                                                    "u_alpha")

        self._u_fl_resolution = glGetUniformLocation(self.prog_flash,
                                                     "u_resolution")
        self._u_fl_alpha     = glGetUniformLocation(self.prog_flash,
                                                    "u_alpha")

        self._u_bl_resolution = glGetUniformLocation(self.prog_blit,
                                                     "u_resolution")
        self._u_bl_texture    = glGetUniformLocation(self.prog_blit,
                                                     "u_texture")
        self._u_bl_alpha      = glGetUniformLocation(self.prog_blit,
                                                     "u_alpha")

        self._u_sw_resolution = glGetUniformLocation(self.prog_shockwave,
                                                     "u_resolution")
        self._u_sw_alpha      = glGetUniformLocation(self.prog_shockwave,
                                                     "u_alpha")

        # ---- Vertex array object -------------------------------------
        # A core-profile context has no default VAO, which makes every
        # glVertexAttribPointer() call fail. Create one when available and
        # bind it before each draw; fall back silently on GL 2.1 drivers
        # where VAO 0 is valid anyway.
        self._vao = None
        self._has_vao = False
        try:
            from OpenGL.GL import (glGenVertexArrays, glBindVertexArray)
            arr = glGenVertexArrays(1)
            if arr:
                self._vao = int(arr)
                self._has_vao = True
        except Exception as exc:                     # pragma: no cover
            print(f"[gl] no VAO support ({exc}); using legacy attribute "
                  "path (requires a compatibility context).")

        # ---- Full-screen background quad (NDC) ----------------------
        bg_quad = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
        ], dtype=np.float32)
        self._bg_vbo = self._make_vbo(bg_quad, GL_STATIC_DRAW)

        # ---- Floor grid quad (world x/z extent) ----------------------
        e = FLOOR_EXTENT
        floor_quad = np.array([
            -e, -e,   e, -e,  -e,  e,
            -e,  e,   e, -e,   e,  e,
        ], dtype=np.float32)
        self._floor_vbo = self._make_vbo(floor_quad, GL_STATIC_DRAW)

        # ---- Camera PIP quad (pixel-space, regenerated each frame) ---
        # 6 verts * (2 pos + 2 uv) * 4 bytes = 96 bytes
        self._pip_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._pip_vbo)
        glBufferData(GL_ARRAY_BUFFER, 96, None, GL_DYNAMIC_DRAW)

        # Camera texture (clamped so we never wrap-sample at the edges)
        self._cam_tex = self._make_texture()

        # ---- Particle buffers ----------------------------------------
        self._p_pos_vbo  = self._make_empty(NUM_PARTICLES * 12)
        self._p_col_vbo  = self._make_empty(NUM_PARTICLES * 12)
        self._p_size_vbo = self._make_empty(NUM_PARTICLES * 4)
        self._p_life_vbo = self._make_empty(NUM_PARTICLES * 4)

        # ---- Overlay (hand landmarks + palm) -------------------------
        # 6 floats per vertex (2 pos + 3 col + 1 pad) * 4 bytes = 24 bytes
        self._ov_vbo = self._make_empty(256 * 24)

        # ---- Shockwave ring VBO --------------------------------------
        self._sw_vbo = self._make_empty(self.SHOCKWAVE_SEGMENTS * 20)

        # ---- Solid quads (flash, PIP frame bars, hand skeleton bones) ----
        # 6 floats per vertex (x, y, r, g, b, pad). The hand skeleton alone
        # is 21 bones * 6 verts = 126 vertices.
        self._solid_vbo = self._make_empty(512 * 24)

        # ---- HUD texture (RGBA) --------------------------------------
        self._hud_tex = self._make_texture()

        # ---- GL state we want to keep set across frames --------------
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPointSize(1.0)

        # Honour gl_PointSize from the vertex shader.
        #
        # This is not a given: some drivers/contexts come up with
        # GL_PROGRAM_POINT_SIZE disabled, in which case every particle is
        # rasterised as a 1x1 pixel no matter what the shader asks for and
        # the whole ball disappears (the sphere-disc 'discard' then throws
        # away that single fragment entirely). Without this the particles
        # are effectively invisible.
        try:
            glEnable(GL_PROGRAM_POINT_SIZE)
        except Exception as exc:                     # pragma: no cover
            print(f"[gl] could not enable GL_PROGRAM_POINT_SIZE ({exc}); "
                  "particles may render at 1px.")

        # Cached MVP - populated each frame by _draw_particles(); used by
        # the app to project the ball center to screen space when
        # triggering the explosion.
        self._cached_mvp: Optional[np.ndarray] = None

        # Latest camera frame, kept here so render() can be called
        # independently of update_camera_frame().
        self._cam_frame_rgb: Optional[np.ndarray] = None

    # ------------------------------------------------------------ helpers
    def _make_vbo(self, data: np.ndarray, usage) -> int:
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, usage)
        return vbo

    def _make_empty(self, nbytes: int) -> int:
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
        return vbo

    def _make_texture(self) -> int:
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        return tex

    def _bind_vao(self) -> None:
        """Bind our VAO (no-op when the driver/context has none)."""
        if not self._has_vao:
            return
        from OpenGL.GL import glBindVertexArray
        glBindVertexArray(self._vao)

    # ------------------------------------------------------------ public API
    def project_world_to_screen(self, world_pos) -> Tuple[float, float]:
        """Project a 3D world-space point to 2D screen pixel coords.

        Uses the cached MVP from the most recent _draw_particles() call.
        Returns (px, py) in pixel space (origin top-left). If the MVP
        hasn't been computed yet (first frame), returns the screen center.
        """
        if self._cached_mvp is None:
            return self.width / 2.0, self.height / 2.0
        return _project_world_to_screen(world_pos, self._cached_mvp,
                                        self.width, self.height)

    def update_particle_buffers(self, positions: np.ndarray, colors: np.ndarray,
                                sizes: np.ndarray, lives: np.ndarray) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, self._p_pos_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, positions.nbytes, positions.tobytes())
        glBindBuffer(GL_ARRAY_BUFFER, self._p_col_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, colors.nbytes, colors.tobytes())
        glBindBuffer(GL_ARRAY_BUFFER, self._p_size_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizes.nbytes, sizes.tobytes())
        glBindBuffer(GL_ARRAY_BUFFER, self._p_life_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, lives.nbytes, lives.tobytes())

    def update_camera_frame(self, frame_rgb: np.ndarray) -> None:
        """Upload the latest webcam frame as the camera texture."""
        self._cam_frame_rgb = frame_rgb
        if frame_rgb is None:
            return
        h, w = frame_rgb.shape[:2]
        if w == 0 or h == 0:
            return
        glBindTexture(GL_TEXTURE_2D, self._cam_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE,
                     np.ascontiguousarray(frame_rgb).tobytes())

    def render(self,
               particle_system,
               hand_state,
               flash_alpha: float,
               shockwave: Optional[dict],
               time_sec: float,
               hud_surface=None) -> None:
        """Draw a full frame."""
        glViewport(0, 0, self.width, self.height)

        # The depth mask must be ON while clearing, otherwise glClear skips
        # the depth buffer entirely and the uninitialised contents (0.0 on
        # some drivers) reject every depth-tested fragment - which silently
        # hides the particles and the floor grid.
        glDepthMask(True)
        glClearDepth(1.0)
        glDisable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthMask(False)

        # 1. Dark background (no depth interaction)
        self._draw_background(particle_system, time_sec)

        # 2/3/4. Particles: lit spheres (depth written) + additive glow.
        self._draw_particles(particle_system)

        # 5. Shockwave ring (screen space, no depth)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(False)
        if shockwave is not None and shockwave.get("alpha", 0.0) > 0.01:
            self._draw_shockwave(shockwave)

        # 6. Camera PIP (bottom-right corner) + frame
        pip_rect = self._draw_camera_pip()

        # 7. Hand landmarks inside the PIP
        if hand_state.hand_present and hand_state.landmarks is not None:
            self._draw_hand_overlay(hand_state, pip_rect)

        # 8. HUD text layer
        if hud_surface is not None:
            self._draw_hud(hud_surface)

        # 9. Flash
        if flash_alpha > 0.0:
            self._draw_flash(flash_alpha)

    # --------------------------------------------------------- private draws
    def _draw_background(self, particle_system, time_sec: float) -> None:
        glUseProgram(self.prog_background)
        # Dim the background when exploding so the particles glow more.
        brightness = 0.55 if particle_system.state == "exploding" else 1.0
        glUniform1f(self._u_bg_brightness, brightness)
        glUniform1f(self._u_bg_time, time_sec)

        self._bind_vao()
        glBindBuffer(GL_ARRAY_BUFFER, self._bg_vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0)

    def _draw_floor(self, mvp: np.ndarray) -> None:
        """Faint reference grid on the virtual floor - the main depth cue."""
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(True)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(self.prog_floor)
        glUniformMatrix4fv(self._u_fl_mvp, 1, True, mvp)
        glUniform1f(self._u_fl_y, float(FLOOR_Y))
        glUniform3f(self._u_fl_color, 0.30, 0.42, 0.62)
        glUniform1f(self._u_fl_fade, 1.0)

        self._bind_vao()
        glBindBuffer(GL_ARRAY_BUFFER, self._floor_vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0)

    def _bind_particle_attribs(self) -> None:
        """Bind the four per-particle attribute streams (locations 0..3)."""
        self._bind_vao()

        glBindBuffer(GL_ARRAY_BUFFER, self._p_pos_vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self._p_col_vbo)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self._p_size_vbo)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, False, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self._p_life_vbo)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, False, 0, None)

    def _unbind_particle_attribs(self) -> None:
        for i in range(4):
            glDisableVertexAttribArray(i)

    def _draw_particles(self, particle_system) -> None:
        """Two passes: lit 3D spheres, then an additive glow halo."""
        view = _look_at(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP)
        proj = _perspective(CAMERA_FOV, self.width / max(1, self.height),
                           0.1, 100.0)
        mvp = (proj @ view).astype(np.float32)

        # Cache the MVP for the explosion code (so the shockwave center
        # can be projected to screen space without re-computing the
        # camera matrices).
        self._cached_mvp = mvp

        # Physically correct perspective point scale:
        #   pixels = world_radius * proj[1][1] * viewport_height / (2 * dist)
        point_scale = float(proj[1, 1]) * self.height * 0.5

        self.update_particle_buffers(
            particle_system.positions,
            particle_system.colors,
            particle_system.sizes,
            particle_system.lives,
        )

        # ---- floor reference grid (writes depth) ----
        self._draw_floor(mvp)

        # ---- pass 1: lit spheres (depth tested + written) ----
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDepthMask(True)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(self.prog_particle_core)
        glUniformMatrix4fv(self._u_pc_mvp, 1, True, mvp)
        glUniform2f(self._u_pc_resolution, float(self.width), float(self.height))
        glUniform1f(self._u_pc_scale, point_scale)
        glUniform1f(self._u_pc_time, particle_system.time)
        glUniform3f(self._u_pc_light, *LIGHT_DIR_VIEW.tolist())
        glUniform1f(self._u_pc_alpha_scale, 1.0)

        self._bind_particle_attribs()
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES)
        self._unbind_particle_attribs()

        # ---- pass 2: additive glow (depth tested, no depth write) ----
        glDepthMask(False)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        glUseProgram(self.prog_particle_glow)
        glUniformMatrix4fv(self._u_pg_mvp, 1, True, mvp)
        glUniform2f(self._u_pg_res, float(self.width), float(self.height))
        glUniform1f(self._u_pg_scale, point_scale)
        glUniform1f(self._u_pg_time, particle_system.time)
        glUniform1f(self._u_pg_grow, 2.4)
        glUniform1f(self._u_pg_str, 0.55)

        self._bind_particle_attribs()
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES)
        self._unbind_particle_attribs()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _pip_size(self) -> Tuple[int, int]:
        """PIP size in pixels, matching the camera frame's aspect ratio."""
        w = max(160, min(self.PIP_WIDTH, self.width // 4))
        aspect = 4.0 / 3.0
        if self._cam_frame_rgb is not None:
            h, cw = self._cam_frame_rgb.shape[:2]
            if cw:
                aspect = float(cw) / float(max(1, h))
                aspect = max(0.5, min(2.5, aspect))
        h = int(round(w / aspect))
        # Never let the PIP grow taller than a quarter of the window.
        max_h = max(80, self.height // 4)
        if h > max_h:
            h = max_h
            w = int(round(h * aspect))
        return w, h

    def _draw_camera_pip(self) -> Tuple[int, int, int, int]:
        """Draw the webcam as a small inset in the bottom-right corner.

        Returns the pixel-space rectangle (x, y, w, h) of the PIP so the
        caller can position the hand-overlay dots correctly.
        """
        pip_w, pip_h = self._pip_size()
        pip_x = self.width - pip_w - self.PIP_MARGIN
        pip_y = self.height - pip_h - self.PIP_MARGIN

        # Build the PIP quad in pixel-space. Each vertex has (x, y, u, v).
        x0, y0 = pip_x, pip_y
        x1, y1 = pip_x + pip_w, pip_y + pip_h
        # The camera texture is already mirrored in main.py via cv2.flip,
        # so we use uv (0,0)->(1,1) directly.
        quad = np.array([
            x0, y0, 0.0, 0.0,
            x1, y0, 1.0, 0.0,
            x0, y1, 0.0, 1.0,
            x0, y1, 0.0, 1.0,
            x1, y0, 1.0, 0.0,
            x1, y1, 1.0, 1.0,
        ], dtype=np.float32)

        glUseProgram(self.prog_camera_pip)
        glUniform2f(self._u_pip_resolution, float(self.width), float(self.height))
        glUniform1i(self._u_pip_texture, 0)
        glUniform1f(self._u_pip_alpha, 1.0)
        glUniform2f(self._u_pip_uv_min, 0.0, 0.0)
        glUniform2f(self._u_pip_uv_max, 1.0, 1.0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._cam_tex)

        glBindBuffer(GL_ARRAY_BUFFER, self._pip_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, quad.nbytes, quad.tobytes())

        self._bind_vao()
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 16, None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, False, 16, ctypes.c_void_p(8))

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

        # Thin bright frame so the inset stands out.
        self._draw_pip_frame(pip_x, pip_y, pip_w, pip_h)

        return pip_x, pip_y, pip_w, pip_h

    def _draw_solid_quads(self, quads, alpha: float) -> None:
        """Draw axis-aligned solid quads in pixel space.

        `quads` is a sequence of (x, y, w, h, (r, g, b)).
        """
        verts = []
        for x, y, w, h, col in quads:
            r, g, b = col
            verts.extend([
                x,     y,     r, g, b, 1.0,
                x + w, y,     r, g, b, 1.0,
                x,     y + h, r, g, b, 1.0,
                x,     y + h, r, g, b, 1.0,
                x + w, y,     r, g, b, 1.0,
                x + w, y + h, r, g, b, 1.0,
            ])
        self._draw_solid_verts(verts, alpha)

    def _draw_solid_bones(self, segments, thickness: float,
                          color, alpha: float) -> None:
        """Draw thick line segments (any angle) in pixel space.

        Drawn as quads rather than GL_LINES because a core profile clamps
        glLineWidth to 1.0, so line thickness would otherwise vary by
        driver.

        `segments` is a sequence of ((x0, y0), (x1, y1)).
        """
        half = max(0.5, thickness * 0.5)
        r, g, b = color
        verts = []
        for (x0, y0), (x1, y1) in segments:
            dx, dy = x1 - x0, y1 - y0
            length = (dx * dx + dy * dy) ** 0.5
            if length < 1e-4:
                continue
            # Unit normal, scaled by half the thickness.
            nx, ny = -dy / length * half, dx / length * half
            ax, ay = x0 + nx, y0 + ny
            bx, by = x1 + nx, y1 + ny
            cx, cy = x1 - nx, y1 - ny
            dx2, dy2 = x0 - nx, y0 - ny
            verts.extend([
                ax, ay, r, g, b, 1.0,
                bx, by, r, g, b, 1.0,
                cx, cy, r, g, b, 1.0,
                cx, cy, r, g, b, 1.0,
                bx, by, r, g, b, 1.0,
                dx2, dy2, r, g, b, 1.0,
            ])
        self._draw_solid_verts(verts, alpha)

    def _draw_solid_verts(self, verts, alpha: float) -> None:
        """Upload a flat (x, y, r, g, b, pad) vertex array and draw it."""
        if not verts:
            return
        data = np.array(verts, dtype=np.float32)

        glUseProgram(self.prog_flash)
        glUniform2f(self._u_fl_resolution, float(self.width), float(self.height))
        glUniform1f(self._u_fl_alpha, alpha)

        glBindBuffer(GL_ARRAY_BUFFER, self._solid_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data.tobytes())

        self._bind_vao()
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 24, None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 24, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, False, 24, ctypes.c_void_p(20))

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawArrays(GL_TRIANGLES, 0, len(verts) // 6)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)

    def _draw_pip_frame(self, x: int, y: int, w: int, h: int) -> None:
        """2px frame around the PIP, drawn as four solid bars.

        Drawn as bars rather than GL_LINES because a core profile clamps
        glLineWidth to 1.0, so a line-based frame would come out at a
        different thickness depending on the driver.
        """
        t = 2
        col = (0.55, 0.72, 0.92)
        self._draw_solid_quads([
            (x - t, y - t, w + 2 * t, t, col),          # top
            (x - t, y + h, w + 2 * t, t, col),           # bottom
            (x - t, y, t, h, col),                        # left
            (x + w, y, t, h, col),                        # right
        ], alpha=0.95)

    def _draw_hand_overlay(self, hand_state, pip_rect) -> None:
        """Draw the hand skeleton and landmarks over the camera inset.

        * 21 green landmark dots connected by bright green bones (with a
          dark underlay so the green reads on any background)
        * 1 red palm-centroid dot inside a red ring
        * a yellow box around the hand extent while the palm reads as
          "open" - i.e. while it is the trigger for the explosion

        The landmarks are in normalized [0, 1] image space; the dots are
        mapped into the PIP by the OVERLAY vertex shader (offset + scale)
        while the bones are built directly in pixel space.
        """
        pip_x, pip_y, pip_w, pip_h = pip_rect

        lm = hand_state.landmarks
        palm = hand_state.palm.reshape(1, 2)
        palm_xy = (float(palm[0, 0]), float(palm[0, 1]))

        # Normalized -> pixel position inside the PIP.
        def to_px(p):
            return (pip_x + float(p[0]) * pip_w,
                    pip_y + float(p[1]) * pip_h)

        # ---- bones: draw them as thick quads, not GL_LINES ----
        segments = [(to_px(lm[a]), to_px(lm[b]))
                    for a, b in HAND_CONNECTIONS]
        # Dark underlay first so the green reads on any background.
        self._draw_solid_bones(segments, thickness=5.0,
                               color=(0.02, 0.10, 0.02), alpha=0.55)
        self._draw_solid_bones(segments, thickness=2.4,
                               color=(0.25, 1.0, 0.35), alpha=0.95)

        # ---- dots: 21 green landmarks + 1 red palm centroid ----
        verts = []
        for p in lm:
            verts.append([p[0], p[1], 0.10, 1.0, 0.25, 1.0])
        verts.append([palm[0, 0], palm[0, 1], 1.0, 0.15, 0.15, 1.0])

        # Point sizes: fingertips a touch larger so the hand reads clearly.
        size = np.full((len(verts), 1), 9.0, dtype=np.float32)
        for tip in (4, 8, 12, 16, 20):
            size[tip] = 12.0
        size[-1, 0] = 18.0                     # the red palm dot
        data = np.hstack([np.array(verts, dtype=np.float32), size])

        glUseProgram(self.prog_overlay)
        glUniform2f(self._u_o_resolution, float(self.width), float(self.height))
        glUniform2f(self._u_o_scale, float(pip_w), float(pip_h))
        glUniform2f(self._u_o_offset, float(pip_x), float(pip_y))
        glUniform1f(self._u_o_alpha, 1.0)

        glBindBuffer(GL_ARRAY_BUFFER, self._ov_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data.tobytes())

        self._bind_vao()
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 24, None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 24, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, False, 24, ctypes.c_void_p(20))

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawArrays(GL_POINTS, 0, len(verts))

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)

        # ---- red ring around the palm centroid + "open palm" box ----
        # Ring radius follows the hand's own size so it hugs the palm
        # whether you hold your hand far from the camera or close to it.
        xs_px = [float(p[0]) * pip_w for p in lm]
        ys_px = [float(p[1]) * pip_h for p in lm]
        extent = max(max(xs_px) - min(xs_px), max(ys_px) - min(ys_px))
        ring_r = float(np.clip(0.55 * extent, 12.0, 42.0))

        ring = []
        steps = 28
        cx, cy = to_px(palm_xy)
        for i in range(steps):
            a0 = 2.0 * np.pi * i / steps
            a1 = 2.0 * np.pi * (i + 1) / steps
            ring.append(((cx + ring_r * np.cos(a0), cy + ring_r * np.sin(a0)),
                         (cx + ring_r * np.cos(a1), cy + ring_r * np.sin(a1))))
        self._draw_solid_bones(ring, thickness=3.0,
                               color=(1.0, 0.2, 0.2), alpha=0.95)

        if hand_state.gesture == "open":
            # Highlight the hand extent: this box is what makes an open
            # palm visible as the trigger for the explosion.
            xs = [float(p[0]) for p in lm]
            ys = [float(p[1]) for p in lm]
            pad_x, pad_y = 0.02, 0.02
            x0 = pip_x + max(0.0, min(xs) - pad_x) * pip_w
            x1 = pip_x + min(1.0, max(xs) + pad_x) * pip_w
            y0 = pip_y + max(0.0, min(ys) - pad_y) * pip_h
            y1 = pip_y + min(1.0, max(ys) + pad_y) * pip_h
            self._draw_solid_quads([
                (x0 - 3, y0 - 3, x1 - x0 + 6, 3, (1.0, 0.85, 0.2)),
                (x0 - 3, y1,     x1 - x0 + 6, 3, (1.0, 0.85, 0.2)),
                (x0 - 3, y0 - 3, 3, y1 - y0 + 6, (1.0, 0.85, 0.2)),
                (x1,     y0 - 3, 3, y1 - y0 + 6, (1.0, 0.85, 0.2)),
            ], alpha=0.95)

    def _draw_shockwave(self, sw: dict) -> None:
        cx, cy = sw["center"]
        radius = sw["radius"]
        alpha = sw["alpha"]
        color = sw.get("color", (1.0, 0.6, 0.2))

        theta = np.linspace(0, 2 * np.pi,
                            self.SHOCKWAVE_SEGMENTS, endpoint=False)
        xs = cx + radius * np.cos(theta)
        ys = cy + radius * np.sin(theta)
        cols = np.tile(color, (self.SHOCKWAVE_SEGMENTS, 1))
        verts = np.column_stack([xs, ys, cols]).astype(np.float32)

        glUseProgram(self.prog_shockwave)
        glUniform2f(self._u_sw_resolution, float(self.width), float(self.height))
        glUniform1f(self._u_sw_alpha, alpha)

        glBindBuffer(GL_ARRAY_BUFFER, self._sw_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts.tobytes())

        self._bind_vao()
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 20, None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 20, ctypes.c_void_p(8))

        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glPointSize(3.5)
        glDrawArrays(GL_POINTS, 0, self.SHOCKWAVE_SEGMENTS)
        glPointSize(1.0)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

    def _draw_hud(self, surface, x: int = 8, y: int = 8) -> None:
        """Blit an RGBA pygame surface to the screen via a GL texture.

        pygame refuses to blit 2D pixels onto an OPENGL surface, so the
        HUD has to go through a texture to be visible at all.
        """
        import pygame

        w, h = surface.get_size()
        if w <= 0 or h <= 0:
            return

        data = pygame.image.tobytes(surface, "RGBA")
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data)

        quad = np.array([
            x,     y,     0.0, 0.0,
            x + w, y,     1.0, 0.0,
            x,     y + h, 0.0, 1.0,
            x,     y + h, 0.0, 1.0,
            x + w, y,     1.0, 0.0,
            x + w, y + h, 1.0, 1.0,
        ], dtype=np.float32)

        glUseProgram(self.prog_blit)
        glUniform2f(self._u_bl_resolution, float(self.width), float(self.height))
        glUniform1i(self._u_bl_texture, 0)
        glUniform1f(self._u_bl_alpha, 1.0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)

        glBindBuffer(GL_ARRAY_BUFFER, self._solid_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, quad.nbytes, quad.tobytes())

        self._bind_vao()
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 16, None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, False, 16, ctypes.c_void_p(8))

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

    def _draw_flash(self, alpha: float) -> None:
        """Full-screen white quad with given alpha (additive)."""
        glUseProgram(self.prog_flash)
        glUniform2f(self._u_fl_resolution, float(self.width), float(self.height))
        glUniform1f(self._u_fl_alpha, alpha)

        w, h = self.width, self.height
        quad = np.array([
            0, 0, 1.0, 1.0, 1.0, 1.0,
            w, 0, 1.0, 1.0, 1.0, 1.0,
            0, h, 1.0, 1.0, 1.0, 1.0,
            0, h, 1.0, 1.0, 1.0, 1.0,
            w, 0, 1.0, 1.0, 1.0, 1.0,
            w, h, 1.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self._solid_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, quad.nbytes, quad.tobytes())

        self._bind_vao()
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 24, None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 24, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, False, 24, ctypes.c_void_p(20))

        glBlendFunc(GL_SRC_ALPHA, GL_ONE)          # additive for the flash
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)

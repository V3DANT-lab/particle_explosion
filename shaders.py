"""
GLSL shaders for the particle explosion demo.

Shader programs
---------------
  * background    - dark vignette gradient behind the particles (no webcam)
  * floor_grid    - a faint 3D reference grid on the virtual floor plane;
                    this is what makes the scene read as a 3D space instead
                    of a flat particle blob
  * camera_pip    - draws the webcam frame as a small picture-in-picture
                    in the bottom-right corner of the window
  * particle_core - lit 3D spheres (depth tested, depth written). The point
                    sprite's disc is shaded as a real sphere using a normal
                    reconstructed from gl_PointCoord, giving Lambert +
                    Blinn-Phong + Fresnel shading. This is what makes the
                    embers look like solid 3D objects.
  * particle_glow - additive soft halo drawn on top (no depth write) so
                    hot particles bleed light into the air around them
  * overlay       - hand-tracking dots and other screen-space points
  * flash         - solid-colour screen-space quads (explosion flash and
                    the PIP frame bars)
  * blit          - draws an RGBA texture (used for the HUD text layer,
                    which cannot be blitted with pygame onto an OPENGL
                    surface)
  * shockwave     - explosion shockwave ring

GLSL version
------------
Everything targets ``#version 120`` (OpenGL 2.1). That is deliberate: the
app requests a *compatibility* profile context, which every desktop driver
supports, and 120 is the highest version guaranteed there.

NOTE: do not request a core profile in main.py. A core profile has no
default vertex-array object, so every glVertexAttribPointer() call fails
with GL_INVALID_OPERATION and the window comes up completely blank.
"""

# ---------------------------------------------------------------------------
# Background - dark vignette gradient (no webcam, the camera is in PIP)
# ---------------------------------------------------------------------------
BACKGROUND_VERT = """
#version 120
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
    // a_position is in NDC [-1, 1]. Convert to uv [0, 1].
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

BACKGROUND_FRAG = """
#version 120
varying vec2 v_uv;
uniform float u_brightness;
uniform float u_time;

void main() {
    // Dark blueish gradient: lighter at top, darker at bottom.
    float top = mix(0.10, 0.02, v_uv.y);
    vec3 base = vec3(0.04, 0.05, 0.08) + vec3(top * 0.6, top * 0.7, top);

    // Vignette: darken corners
    vec2 c = v_uv - 0.5;
    float vig = 1.0 - smoothstep(0.35, 0.75, length(c) * 1.2);
    base *= mix(0.55, 1.0, vig);

    // Subtle pulsing in the center (where the ball lives)
    float pulse = 0.5 + 0.5 * sin(u_time * 0.6);
    base += vec3(0.06, 0.04, 0.02) * pulse
          * (1.0 - smoothstep(0.0, 0.4, length(c)));

    gl_FragColor = vec4(base * u_brightness, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Floor grid - a perspective-correct grid on the y = u_floor_y plane.
# Provides the depth cue that turns the particle cloud into a 3D scene.
# ---------------------------------------------------------------------------
FLOOR_VERT = """
#version 120
attribute vec2 a_position;      // (world x, world z) extent of the grid
uniform mat4  u_mvp;
uniform float u_floor_y;
varying vec3  v_world;
void main() {
    vec3 w = vec3(a_position.x, u_floor_y, a_position.y);
    v_world = w;
    gl_Position = u_mvp * vec4(w, 1.0);
}
"""

FLOOR_FRAG = """
#version 120
varying vec3 v_world;
uniform vec3  u_color;
uniform float u_fade;

// Grid lines computed analytically (no derivative extension needed, which
// keeps this working on plain OpenGL 2.1 drivers).
void main() {
    vec2 gcoord = v_world.xz * 0.5;
    vec2 g = abs(fract(gcoord) - 0.5);

    // Widen the line with distance so it stays roughly 1px on screen and
    // avoids nasty moire in the far field.
    float d = length(v_world.xz);
    float w = 0.015 + 0.004 * d;
    float line = 1.0 - smoothstep(0.0, w, min(g.x, g.y));

    // Fade out with distance from the origin / the camera.
    float fade = exp(-d * 0.22);

    // Slight highlight right under the ball for depth anchoring.
    float near = 1.0 - smoothstep(0.0, 3.0, d);

    float a = line * (fade * 0.85 + near * 0.25) * u_fade;
    if (a < 0.004) discard;
    gl_FragColor = vec4(u_color, a);
}
"""

# ---------------------------------------------------------------------------
# Camera picture-in-picture
# ---------------------------------------------------------------------------
CAMERA_PIP_VERT = """
#version 120
attribute vec2 a_position;     // pixel-space coords of the PIP rect
attribute vec2 a_texcoord;    // texture uv in [0, 1]
uniform vec2 u_resolution;
varying vec2 v_texcoord;
void main() {
    v_texcoord = a_texcoord;
    vec2 ndc = (a_position / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

CAMERA_PIP_FRAG = """
#version 120
varying vec2 v_texcoord;
uniform sampler2D u_texture;
uniform float u_alpha;
uniform vec2  u_uv_min;
uniform vec2  u_uv_max;
void main() {
    vec2 uv = mix(u_uv_min, u_uv_max, v_texcoord);
    vec3 c = texture2D(u_texture, uv).rgb;
    // Slightly boost saturation so the small image reads better.
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(lum), c, 1.18);
    // Inner vignette so the PIP looks framed.
    vec2 d = v_texcoord - 0.5;
    float vig = 1.0 - smoothstep(0.30, 0.65, length(d) * 1.3);
    c *= mix(0.85, 1.0, vig);
    gl_FragColor = vec4(c, u_alpha);
}
"""

# ---------------------------------------------------------------------------
# IMPORTANT: sprite-relative coordinates
# ---------------------------------------------------------------------------
# The point-sprite shaders do NOT use gl_PointCoord. Some drivers (notably
# Mesa/radeonsi on certain AMD chips) return gl_PointCoord == (0,0) for
# every fragment, which turns any radial falloff or sphere-disc clip into
# either "fully discarded" or "uniformly flat". Instead each vertex shader
# passes the sprite's window-space centre and half-size as varyings, and the
# fragment shader reconstructs the sprite-relative coordinate from
# gl_FragCoord. That is driver-independent and verified working.

# ---------------------------------------------------------------------------
# Particle pass 1 - lit 3D spheres (the "real looking" pass)
#
# a_size is a *world-space radius*. u_point_scale already contains the
# projection / viewport term, so:
#     gl_PointSize = radius * u_point_scale / view_distance
# which is physically correct perspective attenuation. (The previous code
# treated a_size as pixels and multiplied by an extra height/1.2 factor,
# which pushed every particle past the clamp and turned the whole ball
# into a wall of 120px blobs.)
# ---------------------------------------------------------------------------
PARTICLE_CORE_VERT = """
#version 120
attribute vec3  a_position;
attribute vec3  a_color;
attribute float a_size;       // world-space radius
attribute float a_life;

uniform mat4  u_mvp;
uniform vec2  u_resolution;
uniform float u_point_scale; // proj[1][1] * viewport_height * 0.5
uniform float u_time;

varying vec3  v_color;
varying float v_life;
varying vec2  v_center_px;
varying float v_half_px;

void main() {
    vec4 clip = u_mvp * vec4(a_position, 1.0);
    gl_Position = clip;

    // For the standard perspective matrix clip.w == -view_z == distance
    // along the view axis (always > 0 for visible geometry).
    float w = clip.w;

    // Sprite-relative coordinate reconstruction (see note above).
    vec2 ndc = clip.xy / max(abs(w), 1e-4);
    v_center_px = (ndc * 0.5 + 0.5) * u_resolution;

    if (w < 0.60) {
        // Debris that flies past (or right up to) the camera would
        // otherwise blow up to a screen-filling sprite and white out the
        // whole image. Park it outside the clip volume instead.
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_half_px = 0.5;
        v_color = vec3(0.0);
        v_life  = 0.0;
        gl_PointSize = 1.0;
        return;
    }

    // Two-frequency flicker so the shimmer never looks like a single
    // obvious periodic pulse.
    float wobble = 0.94
                 + 0.045 * sin(u_time * 4.2 + a_position.x * 2.7
                                        + a_position.y * 1.9)
                 + 0.025 * sin(u_time * 7.3 + a_position.z * 3.1);

    float size_px = clamp(a_size * u_point_scale * wobble / max(w, 0.60),
                          1.0, 64.0);
    gl_PointSize = size_px;

    v_half_px = size_px * 0.5;
    v_color   = a_color;
    v_life    = a_life;
}
"""

PARTICLE_CORE_FRAG = """
#version 120
varying vec3  v_color;
varying float v_life;
varying vec2  v_center_px;
varying float v_half_px;

uniform vec3  u_light_dir;    // light direction, view space
uniform float u_alpha_scale;

void main() {
    // Reconstruct a sphere normal from the sprite disc. This is what turns
    // a flat blob into a shaded 3D ball of light.
    vec2 uv = (gl_FragCoord.xy - v_center_px) / max(v_half_px, 0.5);
    float r = length(uv);

    // Soft rim rather than a hard 'discard' at r == 1: when a sprite is only
    // 1-2px across the reconstructed coordinate spans nearly the whole
    // sprite, and a hard cut leaves visible little squares instead of round
    // embers.
    float edge = 1.0 - smoothstep(0.82, 1.0, r);
    if (edge <= 0.002) discard;

    vec3 n = normalize(vec3(uv, sqrt(max(1.0 - r * r, 0.0))));

    vec3 L = normalize(u_light_dir);
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L + V);

    float ndl  = max(dot(n, L), 0.0);
    float ndh  = max(dot(n, H), 0.0);
    float spec = pow(ndh, 42.0);
    float fres = pow(1.0 - n.z, 3.0);

    // Facing ratio -> the centre of the sphere reads hotter than the rim.
    float facing = pow(max(n.z, 0.0), 0.55);

    float life = clamp(v_life, 0.0, 1.0);

    vec3 base = v_color * (0.30 + 0.95 * facing);
    vec3 col  = base * (0.22 + 0.90 * ndl)
              + vec3(1.0, 0.93, 0.82) * spec * 0.85
              + v_color * fres * 0.65;

    // Hot core tint right in the middle.
    col += vec3(1.0, 0.85, 0.55) * pow(facing, 8.0) * 0.35;

    float a = life * u_alpha_scale * edge;
    if (a < 0.004) discard;
    gl_FragColor = vec4(col, a);
}
"""

# ---------------------------------------------------------------------------
# Particle pass 2 - additive glow halo.
# Drawn larger than the sphere with additive blending and depth writes off,
# so hot particles bleed light into the surrounding air (bloom).
# ---------------------------------------------------------------------------
PARTICLE_GLOW_VERT = """
#version 120
attribute vec3  a_position;
attribute vec3  a_color;
attribute float a_size;
attribute float a_life;

uniform mat4  u_mvp;
uniform vec2  u_resolution;
uniform float u_point_scale;
uniform float u_time;
uniform float u_grow;        // halo radius multiplier

varying vec3  v_color;
varying float v_life;
varying vec2  v_center_px;
varying float v_half_px;

void main() {
    vec4 clip = u_mvp * vec4(a_position, 1.0);
    gl_Position = clip;
    float w = clip.w;

    vec2 ndc = clip.xy / max(abs(w), 1e-4);
    v_center_px = (ndc * 0.5 + 0.5) * u_resolution;

    if (w < 0.60) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_half_px = 0.5;
        v_color = vec3(0.0);
        v_life  = 0.0;
        gl_PointSize = 1.0;
        return;
    }

    float wobble = 0.94
                 + 0.045 * sin(u_time * 4.2 + a_position.x * 2.7
                                        + a_position.y * 1.9)
                 + 0.025 * sin(u_time * 7.3 + a_position.z * 3.1);

    float size_px = clamp(a_size * u_point_scale * wobble * u_grow
                          / max(w, 0.60), 1.0, 72.0);
    gl_PointSize = size_px;

    v_half_px = size_px * 0.5;
    v_color   = a_color;
    v_life    = a_life;
}
"""

PARTICLE_GLOW_FRAG = """
#version 120
varying vec3  v_color;
varying float v_life;
varying vec2  v_center_px;
varying float v_half_px;

uniform float u_glow_strength;

void main() {
    vec2 uv = (gl_FragCoord.xy - v_center_px) / max(v_half_px, 0.5);
    float d = length(uv);                    // 0 centre .. 1 edge

    // Wide soft falloff - no hard edge so the halo blends smoothly.
    float halo  = pow(max(1.0 - d, 0.0), 2.6);
    float bloom = pow(max(1.0 - d, 0.0), 6.0) * 0.6;

    float life = clamp(v_life, 0.0, 1.0);
    float a = (halo * 0.55 + bloom * 0.5) * life * u_glow_strength;
    if (a < 0.003) discard;

    vec3 col = v_color + vec3(0.35, 0.18, 0.05) * bloom;
    gl_FragColor = vec4(col * a, a);
}
"""

# ---------------------------------------------------------------------------
# Overlay shaders - hand-tracking dots (screen space)
# ---------------------------------------------------------------------------
OVERLAY_VERT = """
#version 120
attribute vec2 a_position;
attribute vec3 a_color;
attribute float a_size;
uniform vec2 u_resolution;
uniform vec2 u_offset;       // pixel-space offset (used to translate the
                             //   hand overlay into the PIP region)
uniform vec2 u_scale;        // scale applied before offset
varying vec3 v_color;
varying vec2 v_center_px;
varying float v_half_px;
void main() {
    vec2 pos = a_position * u_scale + u_offset;
    vec2 ndc = (pos / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = a_size;

    vec2 wnd = vec2(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5) * u_resolution;
    v_center_px = wnd;
    v_half_px   = a_size * 0.5;

    v_color = a_color;
}
"""

OVERLAY_FRAG = """
#version 120
varying vec3 v_color;
varying vec2 v_center_px;
varying float v_half_px;
uniform float u_alpha;
void main() {
    vec2 uv = (gl_FragCoord.xy - v_center_px) / max(v_half_px, 0.5);
    float d = length(uv);
    // Filled disc with a soft rim and bright center.
    float fill = 1.0 - smoothstep(0.55, 1.0, d);
    float core = 1.0 - smoothstep(0.0, 0.4, d);
    vec3 col = v_color * (0.6 + 0.4 * core);
    gl_FragColor = vec4(col, fill * u_alpha);
}
"""

# ---------------------------------------------------------------------------
# Solid-colour screen-space quads: explosion flash + PIP frame bars.
# The vertex layout is (x, y, r, g, b, pad) - the trailing pad keeps the
# 24-byte stride shared with the overlay buffer layout.
# ---------------------------------------------------------------------------
FLASH_VERT = """
#version 120
attribute vec2 a_position;
attribute vec3 a_color;
attribute float a_size;       // unused but kept for shared VBO layout
uniform vec2 u_resolution;
varying vec3 v_color;
void main() {
    vec2 ndc = (a_position / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = a_color;
}
"""

FLASH_FRAG = """
#version 120
varying vec3 v_color;
uniform float u_alpha;
void main() {
    gl_FragColor = vec4(v_color, u_alpha);
}
"""

# ---------------------------------------------------------------------------
# Texture blit - draws an RGBA texture as an axis-aligned pixel-space quad.
# Used for the HUD: pygame cannot blit 2D pixels onto an OPENGL surface, so
# the text is rendered into a plain surface and uploaded as a texture.
# ---------------------------------------------------------------------------
BLIT_VERT = """
#version 120
attribute vec2 a_position;
attribute vec2 a_texcoord;
uniform vec2 u_resolution;
varying vec2 v_texcoord;
void main() {
    v_texcoord = a_texcoord;
    vec2 ndc = (a_position / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

BLIT_FRAG = """
#version 120
varying vec2 v_texcoord;
uniform sampler2D u_texture;
uniform float u_alpha;
void main() {
    vec4 c = texture2D(u_texture, v_texcoord);
    gl_FragColor = vec4(c.rgb, c.a * u_alpha);
}
"""

# ---------------------------------------------------------------------------
# Shockwave ring
# ---------------------------------------------------------------------------
SHOCKWAVE_VERT = """
#version 120
attribute vec2 a_position;
attribute vec3 a_color;
uniform vec2 u_resolution;
varying vec3 v_color;
void main() {
    vec2 ndc = (a_position / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = a_color;
}
"""

SHOCKWAVE_FRAG = """
#version 120
varying vec3 v_color;
uniform float u_alpha;
void main() {
    gl_FragColor = vec4(v_color, u_alpha);
}
"""

SHADER_SOURCES = {
    "background":   (BACKGROUND_VERT,   BACKGROUND_FRAG),
    "floor_grid":   (FLOOR_VERT,        FLOOR_FRAG),
    "camera_pip":   (CAMERA_PIP_VERT,   CAMERA_PIP_FRAG),
    "particle_core": (PARTICLE_CORE_VERT, PARTICLE_CORE_FRAG),
    "particle_glow": (PARTICLE_GLOW_VERT, PARTICLE_GLOW_FRAG),
    "overlay":      (OVERLAY_VERT,      OVERLAY_FRAG),
    "flash":        (FLASH_VERT,        FLASH_FRAG),
    "blit":         (BLIT_VERT,         BLIT_FRAG),
    "shockwave":    (SHOCKWAVE_VERT,    SHOCKWAVE_FRAG),
}

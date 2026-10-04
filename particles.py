"""
Particle system for the explosion demo.

Holds 1000 particles and provides:
  * reset_ball()      - arrange particles in a uniform-density sphere
  * update(dt, ...)   - advance the simulation one step
  * explode(center)   - kick off the explosion from a given 3D center
  * buffer accessors  - positions / colors / sizes / lives for the renderer

States:
  * "ball"      - particles arranged in a slowly rotating sphere
  * "exploding" - the embers burst apart and hang in the air. There is NO
                  gravity and NO floor collision: air drag bleeds off the
                  burst velocity, soft world bounds keep the cloud in
                  frame, and the particles drift while they cool and fade
  * "settled"   - every ember has faded out; stays here until reset_ball()

Units
-----
Positions, velocities and *radii* are in world units. The renderer turns a
radius into pixels with a proper perspective divide, so a particle's on
screen size shrinks as it moves away from the camera - that is what sells
the 3D depth. (`sizes` must therefore hold world-space radii, not pixels.)

All heavy lifting is done with NumPy so 1000 particles update in a single
vectorized pass per frame.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants - tune these to taste
# ---------------------------------------------------------------------------
NUM_PARTICLES       = 1000
BALL_RADIUS         = 1.0          # world units

# The embers do NOT fall: there is no gravity and no floor collision.
# They burst apart, decelerate against air drag, and hang there drifting
# until their light fades out.
GRAVITY             = 0.0          # deliberately zero - embers float
AIR_DRAG            = 1.8          # per second; bleeds off the burst
                                   # velocity within ~1s so the cloud
                                   # separates and then stops, hanging
                                   # in the air
FLOOR_Y             = -4.2         # height of the reference floor grid.
                                   # Purely a visual depth cue - it sits
                                   # well below the cloud, which never
                                   # drops below about y = -3.6
BOUND_DAMPING       = 0.45         # energy retained when a particle is
                                   # turned back by a soft boundary
EXPLOSION_SPEED_MIN = 3.4
EXPLOSION_SPEED_MAX = 6.6
LIFE_DECAY_PER_SEC  = 0.26         # how fast a particle cools off
ROTATION_SPEED      = 0.45         # rad/s, slow spin while in ball state

# Soft world bounds. They keep the drifting cloud inside the view and, more
# importantly, keep it in front of the camera (which sits at z = +5, so the
# +z limit is much tighter than the others). These are generous enough that
# the burst normally stops before it ever reaches them.
BOUND_X = 5.0
BOUND_Y_MIN, BOUND_Y_MAX = -3.6, 4.2
BOUND_Z_MIN, BOUND_Z_MAX = -5.0, 2.6

# Particle radius range, in world units. With a 60 deg vertical FOV and a
# 720px-tall viewport, a radius of 0.03 is roughly 3px across at the
# camera's default distance - i.e. a small glowing ember.
PARTICLE_RADIUS_MIN = 0.020
PARTICLE_RADIUS_MAX = 0.065


class ParticleSystem:
    """Vectorized particle simulation backed by NumPy."""

    def __init__(self, num_particles: int = NUM_PARTICLES) -> None:
        self.n = num_particles

        # Per-particle arrays
        self.positions  = np.zeros((self.n, 3), dtype=np.float32)
        self.velocities = np.zeros((self.n, 3), dtype=np.float32)
        self.colors     = np.zeros((self.n, 3), dtype=np.float32)
        self.sizes      = np.zeros((self.n, 1), dtype=np.float32)
        self.lives      = np.ones(self.n, dtype=np.float32)

        # Per-particle offsets used for the slow rotation in ball state
        self._base_positions = np.zeros((self.n, 3), dtype=np.float32)
        self._phase          = np.zeros(self.n, dtype=np.float32)

        self.center = np.array([0.0, 0.4, 0.0], dtype=np.float32)
        self.time   = 0.0
        self.state   = "ball"

        self.reset_ball()

    # ------------------------------------------------------------------ setup
    def reset_ball(self, center=None) -> None:
        """Arrange the particles in a uniformly-filled sphere."""
        if center is not None:
            self.center = np.asarray(center, dtype=np.float32)

        n = self.n

        # Uniform sampling inside a ball: r = R * cbrt(u) gives constant
        # volume density, and a uniform cosine for the polar angle.
        u = np.random.rand(n)
        r = (BALL_RADIUS * np.cbrt(u)).astype(np.float32)

        cos_theta = (1.0 - 2.0 * np.random.rand(n)).astype(np.float32)
        cos_theta = np.clip(cos_theta, -0.999, 0.999)
        sin_theta = np.sqrt(1.0 - cos_theta ** 2).astype(np.float32)
        phi = (2.0 * np.pi * np.random.rand(n)).astype(np.float32)

        x = (r * sin_theta * np.cos(phi)).astype(np.float32)
        y = (r * cos_theta).astype(np.float32)
        z = (r * sin_theta * np.sin(phi)).astype(np.float32)

        base = np.stack([x, y, z], axis=1).astype(np.float32)
        self._base_positions = base
        self.positions = base + self.center

        # Random phase so the flicker is uncorrelated between particles.
        self._phase = (np.random.rand(n) * (2.0 * np.pi)).astype(np.float32)

        # Colors: warm ember palette (deep red -> orange -> pale yellow).
        # We pick a temperature per particle; hotter = brighter/yellower.
        temps = np.random.rand(n).astype(np.float32)
        r_ch = 0.85 + 0.15 * temps
        g_ch = 0.20 + 0.70 * temps
        b_ch = 0.05 + 0.55 * np.power(temps, 2.5)
        self.colors = np.stack([r_ch, g_ch, b_ch], axis=1).astype(np.float32)

        # Sizes: WORLD-SPACE radii (see module docstring).
        self.sizes = (PARTICLE_RADIUS_MIN
                      + (PARTICLE_RADIUS_MAX - PARTICLE_RADIUS_MIN)
                      * np.power(np.random.rand(n, 1), 1.6)).astype(np.float32)

        self.velocities.fill(0.0)
        self.lives[:] = 1.0
        self.time = 0.0
        self.state = "ball"

    # ----------------------------------------------------------------- update
    def update(self, dt: float, target_center=None) -> None:
        """Advance the simulation by `dt` seconds."""
        self.time += dt

        if self.state == "ball":
            self._update_ball(dt, target_center)
        elif self.state == "exploding":
            self._update_exploding(dt)
        # "settled" -> nothing to do until reset_ball()

    def _update_ball(self, dt: float, target_center) -> None:
        # Smoothly track a target center (e.g. hand position) if given.
        if target_center is not None:
            target = np.asarray(target_center, dtype=np.float32)
            # Critically-damped follow so the ball doesn't lag wildly.
            self.center += (target - self.center) * min(1.0, 5.0 * dt)

        # Slow rotation around the Y axis.
        ang = self.time * ROTATION_SPEED
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        bx = self._base_positions[:, 0]
        by = self._base_positions[:, 1]
        bz = self._base_positions[:, 2]
        # Also a gentle wobble in y so the ball looks alive.
        wobble = 0.04 * np.sin(self.time * 1.7 + self._phase * 0.5)
        self.positions[:, 0] = self.center[0] + bx * cos_a + bz * sin_a
        self.positions[:, 1] = self.center[1] + by + wobble
        self.positions[:, 2] = self.center[2] - bx * sin_a + bz * cos_a

    def _update_exploding(self, dt: float) -> None:
        """Burst apart and hang in the air.

        There is deliberately no gravity and no floor collision here: the
        embers separate radially, air drag bleeds off the burst velocity,
        and the particles then float in place while they cool and fade.
        """
        # Air drag only. No gravity term.
        self.velocities *= max(0.0, 1.0 - AIR_DRAG * dt)
        self.positions += self.velocities * dt

        # Soft world bounds: turn the particle back with heavy damping
        # rather than bouncing it off a hard wall, so the cloud eases to a
        # stop instead of sloshing. Keeping +z tight also guarantees the
        # embers stay in front of the camera.
        pos, vel = self.positions, self.velocities
        bounds = (
            (0, -BOUND_X, BOUND_X),
            (1, BOUND_Y_MIN, BOUND_Y_MAX),
            (2, BOUND_Z_MIN, BOUND_Z_MAX),
        )
        for axis, lo, hi in bounds:
            p = pos[:, axis]
            v = vel[:, axis]

            below = p < lo
            if np.any(below):
                p[below] = lo
                # Only reflect if still heading outwards.
                v[below] = np.where(v[below] < 0.0, -v[below] * BOUND_DAMPING,
                                    v[below])

            above = p > hi
            if np.any(above):
                p[above] = hi
                v[above] = np.where(v[above] > 0.0, -v[above] * BOUND_DAMPING,
                                    v[above])

        # Cool down + fade
        self.lives -= dt * LIFE_DECAY_PER_SEC
        self.lives = np.clip(self.lives, 0.0, 1.0)

        # Color shift: hotter (yellow) -> cooler (deep red) as life drops
        cool = (1.0 - self.lives).reshape(-1, 1)
        self.colors[:, 0] = 0.55 + 0.45 * (1.0 - cool[:, 0])  # red stays high
        self.colors[:, 1] = np.clip(self.colors[:, 1] * (1.0 - 0.6 * cool[:, 0]),
                                    0.05, 1.0)
        self.colors[:, 2] = np.clip(self.colors[:, 2] * (1.0 - 0.85 * cool[:, 0]),
                                    0.0, 1.0)

        if np.all(self.lives <= 0.0):
            self.state = "settled"

    # --------------------------------------------------------------- explosion
    def explode(self, center=None) -> None:
        """Trigger an outward explosion from the current ball center."""
        if self.state != "ball":
            return
        if center is not None:
            self.center = np.asarray(center, dtype=np.float32)

        # Direction from center to each particle
        dirs = self.positions - self.center
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        # Particles sitting exactly at the centre get a random direction so
        # they still fly somewhere sensible instead of staying put.
        degenerate = (norms[:, 0] < 1e-3)
        if np.any(degenerate):
            dirs[degenerate] = np.random.normal(size=(int(degenerate.sum()), 3))
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-3)
        dirs = dirs / norms

        # Random outward speed
        speeds = (EXPLOSION_SPEED_MIN
                  + (EXPLOSION_SPEED_MAX - EXPLOSION_SPEED_MIN)
                  * np.random.rand(self.n, 1)).astype(np.float32)
        self.velocities = dirs * speeds

        # NO upward kick and NO gravity: the burst is purely radial so the
        # cloud expands symmetrically around the explosion point and then
        # hangs there. An upward bias here would make the whole cloud
        # slowly drift up out of frame.

        # Small swirl component, applied perpendicular to the burst
        # direction so it does not bias the overall expansion.
        swirl = 0.35
        self.velocities[:, 0] += -dirs[:, 1] * swirl * np.random.rand(self.n)
        self.velocities[:, 2] +=  dirs[:, 1] * swirl * np.random.rand(self.n)

        self.state = "exploding"
        self.lives[:] = 1.0

    @property
    def is_settled(self) -> bool:
        return self.state == "settled"

    # ----------------------------------------------------------- buffer writers
    def positions_bytes(self) -> bytes:
        return self.positions.astype(np.float32).tobytes()

    def colors_bytes(self) -> bytes:
        return self.colors.astype(np.float32).tobytes()

    def sizes_bytes(self) -> bytes:
        return self.sizes.astype(np.float32).tobytes()

    def lives_bytes(self) -> bytes:
        return self.lives.astype(np.float32).tobytes()

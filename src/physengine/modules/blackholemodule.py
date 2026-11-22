"""Black hole: a ray-traced Schwarzschild black hole, its accretion disc and bodies orbiting it.

Written by Claude under a one-off override of CLAUDE.md (2026-11-20, module
3). Review the physics.

Units: G = c = 1, lengths and times in units of the hole's mass M (for
Sagittarius A*, 4.3 million solar masses, M = 6.4 million km and 21 s). The
hole sits at the origin, the disc lies in the xz-plane and spins about +y.
The camera is a static observer at its distance from the hole (it cannot
hover inside the horizon, r = 2M; there the view is black).

- The image: schwarzschild.py (formulas, references), blackhole_renderer.py
  (the ray tracer).
- Bodies: exact geodesics, geodesics.py; their lensed and redshifted images,
  blackhole_images.py; scenes, blackhole_scenes.py.
"""

from dataclasses import replace

import numpy as np

from physengine.core.module import BaseModule
from physengine.core.spec import Choice, SliderFloat
from physengine.modules import kerr
from physengine.modules.blackhole_editor import (
    BodiesWindow,
    added_bodies,
    forget_added,
)
from physengine.modules.blackhole_editor import reset_settings as reset_editor_settings
from physengine.modules.blackhole_images import Images, RayTable, body_images
from physengine.modules.blackhole_renderer import (
    LOG_T_RANGE,
    BlackHoleRenderer,
    TraceSettings,
    blackbody_table,
)
from physengine.modules.blackhole_scenes import (
    DEFAULT_HOLE_MASS,
    DEFAULT_STAR,
    GM_SUN_KM,
    RING_ASPECT,
    RING_RADIUS,
    SCENES,
    STAR_SCENES,
    STARS,
    TDE_L,
    TDE_START,
    gas_ring,
    infalling,
    scene_options,
    star_scene,
)
from physengine.modules.blackhole_sph import BETA_COOL, Gas
from physengine.modules.blackhole_star import IntactStar
from physengine.modules.geodesics import BodySystem, History, energy
from physengine.modules.kerr_bodies import (
    KerrBodySystem,
    kerr_capture_L,
    kerr_circular_state,
    kerr_images,
    local_velocity_from_schwarzschild,
    states_from_local,
)
from physengine.modules.schwarzschild import R_HORIZON, R_ISCO, R_PHOTON, shadow_angular_radius

REDSHIFT_MODES = {"Doppler + gravity": 0, "Gravity only": 1, "None": 2}
SKY_MODES = {"Stars": 0, "Checkerboard": 1}
RESOLUTIONS = {"Full": 1.0, "Half": 0.5}
DEFAULT_SCENE = "Precessing orbit"

SECONDS_PER_M_SUN = 4.925491e-6  # G M_sun / c^3
# Mean density inside the horizon, M / (4/3 pi (2 G M / c^2)^3), for one
# solar mass [kg/m^3]; it falls as 1 / M^2.
HOLE_DENSITY_SUN = 1.845e19


def _duration(seconds):
    for unit, size in (("yr", 3.156e7), ("d", 86400.0), ("h", 3600.0), ("min", 60.0)):
        if seconds >= size:
            return f"{seconds / size:.2f} {unit}"
    return f"{seconds:.3g} s"


# With spin, images of up to this many bodies are found exactly in Kerr
# (kerr_bodies.kerr_images, ~0.2 ms each); more (a torn-apart star) use the
# Schwarzschild solver as an approximation, as do trail points.
KERR_EXACT_MAX = 50
TRAIL_SPAN = 250.0  # [M] of past path drawn behind each body
MIN_SPRITE_RADIUS = 1.0  # pixels; smaller images keep their total light
GAS_DRAW_MAX = 0.4  # [M] largest drawn radius of a gas particle
GAS_GLOW_MAX = 3.0  # hottest drawn gas, in star surface temperatures


class BlackHoleModule(BaseModule):
    dt = 1.0 / 60.0  # wall seconds per physics step; `speed` M of coordinate time each

    def __init__(self) -> None:
        self.scene_name = DEFAULT_SCENE
        self.spin = 0.0
        self.hole_mass_log = float(np.log10(DEFAULT_HOLE_MASS))
        self.star = DEFAULT_STAR
        self.gas_cooling = "Fast"
        self.gas_viscosity = 1.0
        self.speed = 40.0
        self.trails = "Shown"
        self.light_delay = "On"
        self.body_brightness = 1.0
        self.disc = "Shown"
        self.disc_outer = 20.0
        self.disc_temperature = 7000.0
        self.redshift = "Doppler + gravity"
        self.sky = "Stars"
        self.exposure = 0.6
        self.resolution = "Full"
        self.specs = (
            Choice(
                attr_name="scene_name",
                label="Scene",
                default=DEFAULT_SCENE,
                options=tuple(SCENES),
                need_reset=True,
            ),
            SliderFloat(
                attr_name="spin", label="Spin a/M", default=0.0, min=0.0, max=0.998, need_reset=True
            ),
            # In units of M the hole always looks the same; its mass sets how
            # large and heavy a star is next to it (star scenes) and the clock.
            SliderFloat(
                attr_name="hole_mass_log",
                label="Hole mass [log10 M_sun]",
                default=float(np.log10(DEFAULT_HOLE_MASS)),
                min=4.0,
                max=9.0,
                need_reset=True,
            ),
            Choice(
                attr_name="star",
                label="Star (star scenes)",
                default=DEFAULT_STAR,
                options=tuple(STARS),
                need_reset=True,
            ),
            Choice(
                attr_name="gas_cooling",
                label="Gas cooling (gas scene)",
                default="Fast",
                options=("Fast", "None"),
            ),
            SliderFloat(
                attr_name="gas_viscosity",
                label="Gas viscosity (gas scenes)",
                default=1.0,
                min=0.2,
                max=5.0,
            ),
            SliderFloat(attr_name="speed", label="Speed [M/s]", default=40.0, min=5.0, max=400.0),
            Choice(
                attr_name="trails", label="Trails", default="Shown", options=("Shown", "Hidden")
            ),
            Choice(
                attr_name="light_delay",
                label="Light travel time",
                default="On",
                options=("On", "Off"),
            ),
            SliderFloat(
                attr_name="body_brightness", label="Body brightness", default=1.0, min=0.1, max=20.0
            ),
            Choice(
                attr_name="disc",
                label="Accretion disc",
                default="Shown",
                options=("Shown", "Hidden"),
            ),
            SliderFloat(
                attr_name="disc_outer", label="Disc outer edge [M]", default=20.0, min=8.0, max=40.0
            ),
            SliderFloat(
                attr_name="disc_temperature",
                label="Disc temperature [K]",
                default=7000.0,
                min=2000.0,
                max=20000.0,
            ),
            Choice(
                attr_name="redshift",
                label="Redshift",
                default="Doppler + gravity",
                options=tuple(REDSHIFT_MODES),
            ),
            Choice(attr_name="sky", label="Sky", default="Stars", options=tuple(SKY_MODES)),
            SliderFloat(attr_name="exposure", label="Exposure", default=0.6, min=0.1, max=8.0),
            Choice(
                attr_name="resolution",
                label="Resolution",
                default="Full",
                options=tuple(RESOLUTIONS),
            ),
        )
        self.renderer = BlackHoleRenderer()
        self.window = BodiesWindow(self)
        self.camera_distance = 40.0  # last seen, for diagnostics
        self.pending_view = True  # set the camera on the first frame
        self.rays = None  # RayTable for the current camera distance
        self.bb_table = blackbody_table()
        self.bodies = None
        self.intact_star = None
        self.gas = None
        self.kerr_guess = None  # last frame's image directions, Kerr only
        self.last_delay = None  # last frame's light-travel times, one per image

    # ---------------------------------------------------------------- bodies

    def load(self):
        """Bodies of the current scene, plus the ones added to it in the Bodies window."""
        self.hole_mass = 10.0**self.hole_mass_log  # solar masses
        star_info = None
        specs = list(SCENES[self.scene_name])
        if self.scene_name in STAR_SCENES:
            # With spin the capture limit moves (below 4 for this mostly
            # prograde orbit): stay the same fraction under the Kerr one.
            plunge_L = TDE_L
            if self.spin > 0.0:
                plunge_L = TDE_L / 4.0 * kerr_capture_L(self.spin, 25.0, TDE_START)
            specs, star_info = star_scene(self.scene_name, self.hole_mass, self.star, plunge_L)
        if self.scene_name == "Gas ring":
            specs = gas_ring()[0]
        specs = specs + added_bodies(self.scene_name)
        self.names = [s.name for s in specs]
        self.radius = np.array([s.radius for s in specs], dtype=float)
        self.temperature = np.array([s.temperature for s in specs], dtype=float)
        self.options = scene_options(self.scene_name, self.hole_mass, self.star)
        x = np.array([s.x for s in specs]).reshape(-1, 3)
        w = np.array([s.w for s in specs]).reshape(-1, 3)
        mass = [s.mass for s in specs]
        if self.spin > 0.0:
            # Same positions and local velocities, now in the Kerr spacetime.
            v = np.array(
                [local_velocity_from_schwarzschild(xi, wi) for xi, wi in zip(x, w, strict=True)]
            )
            states, consts = (
                states_from_local(x, v, self.spin)
                if len(x)
                else (np.zeros((0, 5)), np.zeros((0, 3)))
            )
            if self.scene_name == "Plunging body":
                # Frame dragging moves the capture threshold (for a mostly
                # prograde orbit, below L = 4): start just under the Kerr one.
                L = 0.999 * kerr_capture_L(self.spin, 20.0, 30.0)
                spec = infalling("Star", 30.0, L, inclination_deg=20.0, temperature=9000.0)
                v0 = local_velocity_from_schwarzschild(spec.x, spec.w)
                states[0], consts[0] = (
                    q[0] for q in states_from_local(spec.x[None], v0[None], self.spin)
                )
            if self.scene_name == "Inside and outside the ISCO":
                # Just outside the Kerr isco, and between it and the prograde photon orbit.
                a = self.spin
                r_isco = kerr.isco(a)
                r_photon = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(-a)))
                states[0], consts[0] = kerr_circular_state(r_isco + 1.0, a, np.radians(12.0))
                states[1], consts[1] = kerr_circular_state(
                    r_photon + 0.6 * (r_isco - r_photon), a, np.radians(-12.0), nudge=1e-4
                )
            self.bodies = KerrBodySystem(
                states, consts, self.spin, mass, eps=self.options.softening
            )
        else:
            self.bodies = BodySystem(x, w, mass=mass, eps=self.options.softening)
        self.bodies.OUTER_STEP = self.options.outer_step
        self.intact_star = None if star_info is None else IntactStar(star_info, self.bodies)
        self.gas = Gas(self.bodies.n) if self.options.gas else None
        if self.gas is not None and self.options.gas_energy is not None:
            # Gas from the start (the ring), with shear viscosity as in a disc.
            k = len(self.options.gas_energy)
            self.gas.start(np.arange(k), self.options.gas_energy, self.options.gas_spacing)
            self.gas.shear_viscosity = True
        self.bodies.hydro = self.gas
        spacing, length = self.options.history
        self.history = History(self.bodies, spacing=spacing, length=length)

    def add_body(self, spec):
        """Add a body now, at the current time (its past is filled in by the history)."""
        self.names.append(spec.name)
        self.radius = np.append(self.radius, spec.radius)
        self.temperature = np.append(self.temperature, spec.temperature)
        if self.spin > 0.0:
            v = local_velocity_from_schwarzschild(spec.x, spec.w)
            st, c = states_from_local(np.reshape(spec.x, (1, 3)), v[None], self.spin)
            self.bodies.add_state(st[0], c[0], spec.mass)
        else:
            self.bodies.add(spec.x, spec.w, spec.mass)

    def step(self, dt):
        b = self.bodies
        massive = b.alive & (b.mass > 0.0)
        if b.n > 20 and massive.sum() > 1:
            # Kick interval for the star's self-gravity: 1% of the dynamical
            # time of the cloud as it now is, sqrt(R^3 / m). It grows as the
            # debris spreads and self-gravity stops mattering.
            x = b.x[massive]
            size = np.sqrt(np.mean(np.sum((x - x.mean(axis=0)) ** 2, axis=1)))
            t_dyn = np.sqrt(size**3 / b.mass[massive].sum())
            step = 0.01 * t_dyn
            if self.gas is not None:
                self.gas.cooling = BETA_COOL if self.gas_cooling == "Fast" else 0.0
                self.gas.viscosity = float(self.gas_viscosity)
                if self.options.gas_energy is not None:
                    # The ring: a thin disc that radiates its heat, or not.
                    self.gas.cooling = 0.0
                    self.gas.disc_aspect = RING_ASPECT if self.gas_cooling == "Fast" else 0.0
                step = min(step, self.gas.time_step)  # the gas's Courant time
            b.OUTER_STEP = float(np.clip(step, self.options.outer_step, 2.0))
        b.advance(dt * self.speed)
        if self.intact_star is not None:
            self.intact_star.update(b, dt * self.speed)

    # ------------------------------------------------------------- rendering

    def settings(self):
        return replace(
            TraceSettings(),
            show_disc=self.disc == "Shown" and not self.options.hide_disc,
            r_in=kerr.isco(self.spin) if self.spin > 0.0 else R_ISCO,
            spin=float(self.spin),
            r_out=float(self.disc_outer),
            t_peak=float(self.disc_temperature),
            redshift_mode=REDSHIFT_MODES[self.redshift],
            sky_mode=SKY_MODES[self.sky],
        )

    def _colour(self, temperature):
        """Blackbody colour of luminance 1, from the same table as the disc."""
        lo, hi = LOG_T_RANGE
        x = (np.log10(np.maximum(temperature, 1.0)) - lo) / (hi - lo)
        idx = np.clip(x, 0.0, 1.0) * (len(self.bb_table) - 1)
        grid = np.arange(len(self.bb_table))
        return np.stack([np.interp(idx, grid, self.bb_table[:, c]) for c in range(3)], 1)

    def _images(self, cam_pos, st):
        """Both images of every body: one row per (body, order).

        Returns (body index, order, directions, visible, solid, g, position,
        emission time). With light-travel time on, each image shows the body
        where it was when the light left: t_e = t - delay(position at t_e),
        solved by fixed-point iteration (the delay changes much more slowly
        than t, so three passes settle it).
        """
        b = self.bodies
        n = b.n
        which = np.concatenate([np.arange(n), np.arange(n)])
        order = np.repeat([0, 1], n)
        rows = np.arange(2 * n)
        x = np.concatenate([b.x, b.x])
        w = np.concatenate([b.w, b.w])
        t_e = np.full(2 * n, b.t)
        known = np.concatenate([b.alive, b.alive])
        passes = 3 if self.light_delay == "On" else 0
        if passes and self.last_delay is not None and len(self.last_delay) == 2 * n:
            # Start from last frame's emission times: the delays change slowly,
            # so the positions (and the Kerr solver's starting directions) are
            # already nearly right.
            t_e = b.t - self.last_delay
            x, w, known = self.history.state_at(b, which, t_e)
        exact_kerr = self.spin > 0.0 and n <= KERR_EXACT_MAX
        for it in range(passes + 1):
            if exact_kerr:
                im = self._kerr_images(cam_pos, st, x, w, which, order)
            else:
                ws = self._schwarzschild_w(x, w)
                im = body_images(
                    self.rays, cam_pos, x, ws, energy(x, ws), st.r_in, st.r_out, st.show_disc
                )
                im = im._replace(
                    dirs=im.dirs[rows, order],
                    visible=im.visible[rows, order],
                    solid=im.solid[rows, order],
                    g=im.g[rows, order],
                    delay=im.delay[rows, order],
                )
            vis = im.visible
            if it == passes:
                break
            t_e = b.t - np.where(vis, im.delay, 0.0)
            x, w, known = self.history.state_at(b, which, t_e)
        self.last_delay = np.where(vis, b.t - t_e, 0.0) if passes else None
        # Light from after a body's capture never arrives (it is frozen at the horizon anyway).
        fell = np.isfinite(b.captured_at[which]) & (t_e >= b.captured_at[which])
        vis = vis & known & ~fell
        return which, order, im.dirs, vis, im.solid, im.g, x, t_e

    def _schwarzschild_w(self, x, w):
        """w = dx/dtau for the Schwarzschild image solver: as stored, or (Kerr, approximate) gamma dx/dt."""
        if self.spin == 0.0:
            return w
        v2 = np.minimum(np.sum(w * w, axis=1), 0.999)
        return w / np.sqrt(1.0 - v2)[:, None]

    def _kerr_images(self, cam_pos, st, x, w, which, order):
        """Exact Kerr images (kerr_bodies.kerr_images), started from the last answer.

        An image that was not found (hidden, or its start too far off after a
        long time hidden) starts again from the Schwarzschild image next time.
        """
        n2 = len(x)
        if self.kerr_guess is None or len(self.kerr_guess) != n2:
            self.kerr_guess = np.full((n2, 3), np.nan)
        fresh = ~np.all(np.isfinite(self.kerr_guess), axis=1)
        if fresh.any():
            ws = self._schwarzschild_w(x, w)
            sim = body_images(self.rays, cam_pos, x, ws, energy(x, ws), st.r_in, st.r_out, False)
            start = sim.dirs[np.arange(n2), order]
            self.kerr_guess[fresh] = start[fresh]
        d, vis, solid, g, delay = kerr_images(
            cam_pos, x, w, self.kerr_guess, self.spin, st.r_in, st.r_out, st.show_disc
        )
        self.kerr_guess = np.where(vis[:, None], d, np.nan)
        return Images(d, vis, solid, g, delay)

    def _sprites(self, camera, cam_pos, H):
        """Images of the bodies (both orders) and of their trails (direct image only)."""
        st = self.settings()
        if self.bodies.n == 0:
            return None
        r0 = float(np.linalg.norm(cam_pos))
        if self.rays is None or self.rays.r0 != r0:
            self.rays = RayTable(r0)
        px_per_rad = H / (2.0 * np.tan(camera.fov / 2.0))
        dirs, rgbs, radii = [], [], []

        which, order, d, vis, solid, g, x, t_e = self._images(cam_pos, st)
        if st.redshift_mode == 2:
            g = np.ones_like(g)
        elif st.redshift_mode == 1:  # as if at rest where it is (Kerr: at rest in the ZAMO frame)
            if self.spin > 0.0:
                a = self.spin
                rb = np.array([kerr.from_world(p, a)[:2] for p in x]).reshape(-1, 2)
                rc, thc, _ = kerr.from_world(cam_pos, a)
                enu_b = np.array([kerr.metric(q[0], q[1], a)[4] for q in rb])
                g = enu_b / kerr.metric(rc, thc, a)[4]
            else:
                r = np.linalg.norm(x, axis=1)
                g = np.sqrt(1.0 - 2.0 / r) / np.sqrt(1.0 - 2.0 / r0)
        temp = self.temperature[which]
        brightness_temp = temp
        if self.gas is not None and self.gas.u_start > 0.0 and len(self.gas.u) == self.bodies.n:
            # Display: gas heated by shocks glows hotter. Its drawn temperature
            # rises as the fourth root of its thermal energy, from the star's
            # surface temperature (the true gas temperature, millions of
            # kelvin, is that of its optically thick interior).
            heat = np.clip((self.gas.u / self.gas.u_start) ** 0.25, 0.6, GAS_GLOW_MAX)
            if self.options.gas_energy is not None:
                # The ring: coloured like a disc, whose surface temperature
                # falls outward as r^-3/4 (Shakura & Sunyaev 1973).
                r = np.maximum(self.bodies.r(), 1.0)
                heat = np.clip((r / RING_RADIUS) ** -0.75, 0.5, GAS_GLOW_MAX)
            heat = np.where(self.bodies.mass > 0.0, heat, 1.0)[which]
            brightness_temp = temp * np.minimum(heat, 1.4)  # T^4 would wash the picture out
            temp = temp * heat
        # Surface brightness relative to the disc's peak, times g^4; colour of g T.
        boost = self.body_brightness * self.options.brightness  # display only
        surface = boost * (brightness_temp / st.t_peak) ** 4 * g**4
        colour = self._colour(temp * g)
        radius = self.radius
        soft = np.zeros(len(radius), dtype=bool)
        star = self.intact_star
        if star is not None:
            # A star's particles are drawn as soft blobs about as wide as
            # their spacing, so they merge into one glowing body; once they
            # are gas, as wide as their smoothing length (display only;
            # capped so thin outlying gas stays small).
            spacing = star.info.radius / np.cbrt(star.info.n)
            wide = np.full(len(radius), spacing)
            if self.gas is not None and len(self.gas.h) == len(radius):
                is_gas = self.bodies.mass > 0.0
                wide = np.where(is_gas, np.clip(0.6 * self.gas.h, spacing, GAS_DRAW_MAX), wide)
            radius = radius.copy()
            radius[star.idx] = wide[star.idx]
            soft[star.idx] = True
        elif self.gas is not None and len(self.gas.h) == len(radius):
            # Gas without a star (the ring): the same soft blobs.
            is_gas = self.bodies.mass > 0.0
            spacing = self.options.gas_spacing
            wide = np.clip(0.6 * self.gas.h, 0.5 * spacing, GAS_DRAW_MAX)
            radius = np.where(is_gas, wide, radius)
            soft = is_gas
        rad_px = radius[which] * np.sqrt(solid) * px_per_rad
        shown = vis & (g > 0.0)
        drawn = np.maximum(rad_px, MIN_SPRITE_RADIUS)
        keep_light = (rad_px / drawn) ** 2  # same total light when drawn larger
        # Blobs drawn wider than the star's particle: the same total light.
        keep_light = keep_light * ((self.radius / radius) ** 2)[which]
        dirs.append(d[shown])
        rgbs.append((colour * (surface * keep_light)[:, None])[shown])
        radii.append(np.where(soft[which], -drawn, drawn)[shown])  # negative: a soft blob

        if self.trails == "Shown" and self.options.trails:
            h = self.history
            for i in range(self.bodies.n):
                # Up to the time the direct image shows.
                t_show = t_e[i] if vis[i] else self.bodies.t
                k_hi = min(int(np.floor(t_show / h.spacing)), h.k_last)
                k_lo = max(k_hi - int(TRAIL_SPAN / h.spacing), h.k_last - h.length + 1)
                ks = np.arange(k_lo, k_hi + 1)
                pts = h.x[ks % h.length, i]
                pts = pts[np.all(np.isfinite(pts), axis=1) & (np.linalg.norm(pts, axis=1) > 2.05)]
                if len(pts) < 2:
                    continue
                tim = body_images(
                    self.rays,
                    cam_pos,
                    pts,
                    np.zeros_like(pts),
                    np.ones(len(pts)),
                    st.r_in,
                    st.r_out,
                    st.show_disc,
                )
                keep = tim.visible[:, 0]
                fade = np.linspace(0.03, 0.5, len(pts))[keep]
                c = self._colour(np.array([self.temperature[i]]))[0]
                dirs.append(tim.dirs[keep, 0])
                rgbs.append(fade[:, None] * c[None, :])
                radii.append(np.full(int(keep.sum()), 0.7))

        return np.vstack(dirs), np.vstack(rgbs), np.concatenate(radii)

    def setup(self, ctx):
        self.load()
        self.renderer.setup(ctx)

    def render(self, ctx, camera):
        if self.pending_view:
            camera.distance, camera.pitch = self.options.view
            self.pending_view = False
        view = camera.view()
        cam_pos = -view[:3, :3].T @ view[:3, 3]
        self.camera_distance = float(np.linalg.norm(cam_pos))
        sprites = None
        if self.camera_distance > R_HORIZON:
            sprites = self._sprites(camera, cam_pos, ctx.fbo.viewport[3])
        self.renderer.render(
            ctx,
            camera,
            self.settings(),
            exposure=self.exposure,
            scale=RESOLUTIONS[self.resolution],
            sprites=sprites,
        )

    # -------------------------------------------------------------------- UI

    def draw_ui(self):
        self.window.update()

    def discard_edits(self):
        forget_added()
        reset_editor_settings()

    def diagnostics(self):
        r = self.camera_distance
        if r <= R_HORIZON:
            where = " Inside the horizon: no static observer, view black"
        else:
            where = f" Shadow radius (exact): {np.degrees(shadow_angular_radius(r)):.2f} deg"
        t = self.bodies.t
        b = self.bodies
        extra = ""
        if b.n > 20:
            e = energy(b.x[b.alive], b.w[b.alive])
            star = self.intact_star
            if star is not None:
                m_sun, radius_km, _ = STARS[self.star]
                extra = (
                    f" {self.star} star: {m_sun:g} M_sun, radius {star.info.radius:.3g} M"
                    f" ({radius_km:.3g} km): {star.state}\n"
                )
            if self.gas is not None and (b.alive & (b.mass > 0.0)).any():
                gas = b.alive & (b.mass > 0.0)
                # T = (gamma - 1) mu m_p u / k_B with mu = 0.6 (ionised, solar mix).
                kelvin = 2.0 / 3.0 * 0.6 * 1.0888e13 * np.median(self.gas.u[gas])
                extra += f" Gas (SPH): median temperature {kelvin:.2g} K\n"
            extra += (
                f" {self.options.info}\n"
                f" Particles: {b.n}, fallen in {int((~b.alive).sum())}, "
                f"bound {int((e < 1.0).sum())}, escaping {int((e >= 1.0).sum())}\n"
            )
        if self.spin > 0.0:
            a = self.spin
            extra += (
                f" Spin a = {a:.3f}: horizon {kerr.horizon(a):.3f} M, ISCO {kerr.isco(a):.3f} M\n"
            )
            if b.n > KERR_EXACT_MAX:
                extra += " (Images of this many particles use Schwarzschild lensing, approximate)\n"
        return (
            f" Hole: {self.hole_mass:.3g} M_sun, M = {GM_SUN_KM * self.hole_mass:.3g} km"
            f" = {_duration(SECONDS_PER_M_SUN * self.hole_mass)},"
            f" mean density {HOLE_DENSITY_SUN / self.hole_mass**2:.3g} kg/m^3\n"
            + extra
            + f" Time: {t:.0f} M ({_duration(t * SECONDS_PER_M_SUN * self.hole_mass)})\n"
            f" Camera distance: {r:.2f} M\n"
            + where
            + (
                ""
                if self.spin > 0.0
                else f"\n Horizon {R_HORIZON:.0f} M, photon sphere {R_PHOTON:.0f} M,"
                f" ISCO {R_ISCO:.0f} M (disc inner edge)"
            )
            + "\n"
            f" Last trace: {self.renderer.trace_ms:.1f} ms (GPU), redone only when the view changes"
        )

    def teardown(self):
        self.renderer.release()

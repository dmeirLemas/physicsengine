"""Interactive scene editing for the N-body module. UI only.

Mouse controls in the view:
    Ctrl + left press      place a new body there; drag to set its velocity
    (keep Ctrl held)       scroll to change its mass; releasing Ctrl adds it
    right click            cancel the body being placed
    Ctrl + right click     remove the body under the cursor
    left click             select the body under the cursor (empty space deselects)

The selected body's values are edited live in the "Scene editor" window, and
its predicted path is drawn. After an edit that changes the motion (adding,
removing, mass, position, velocity, Make orbit), the system as it is at that
moment becomes the scene's restart point: Restart returns there. Full reset
forgets all edits and reloads the scene from its definition.

Units are those of the N-body module: AU, years, solar masses, AU/yr.
"""

import time
from copy import deepcopy

import numpy as np
from imgui_bundle import imgui

from physengine.modules.nbody_scenes import BLACK_HOLE, Scene, rocky_radius, star_radius
from physengine.modules.postnewtonian import schwarzschild_radius

FOLLOW_NONE = "None"
FOLLOW_COM = "Centre of mass"
MAX_TRAIL_YEARS = 5.0
FLASH_SECONDS = 0.9

# Kept at module level so they survive the N-body module being rebuilt on
# Restart.
settings = {
    # Length of the orbit trails in years of simulated time. 0 hides them.
    "trail_years": 1.0,
    # What the camera follows: FOLLOW_NONE, FOLLOW_COM or a body name.
    "follow": FOLLOW_NONE,
    # Velocity of a placed body per AU of drag, in 1/yr. The default makes a
    # 1 AU drag give 2 pi AU/yr, Earth's orbital speed.
    "speed_per_au": 2.0 * np.pi,
    # Length of the predicted path drawn for the selected or placed body.
    "lookahead_years": 2.0,
    # Starting values for placed bodies.
    "new_mass": 1.0e-6,
    "new_radius": 0.02,
    "new_color": [0.8, 0.8, 0.8],
    # "Body", or "Black hole": then its physical radius is the horizon,
    # 2 G M / c^2, and bodies that touch it are absorbed.
    "new_kind": "Body",
}
DEFAULT_SETTINGS = deepcopy(settings)


def reset_settings():
    """Back to DEFAULT_SETTINGS (Full reset). Updates the dict in place, so imports of it stay valid."""
    settings.clear()
    settings.update(deepcopy(DEFAULT_SETTINGS))


MASS_STEP = 10.0**0.25  # mass factor per scroll notch while placing


def new_body_radius(kind, mass, G):
    """Physical radius of a placed body: a black hole's horizon, a main-sequence
    star's radius, or that of a body of Earth's density."""
    if kind == "Black hole":
        return schwarzschild_radius(mass, G)
    if kind == "Star":
        return star_radius(mass)
    return rocky_radius(mass)


CLICK_SLOP_PX = 4.0  # a press and release closer than this is a click
PICK_SLOP_PX = 6.0  # bodies are clickable this far outside their disc
WINDOW_MARGIN = 10.0
FIELD_WIDTH = 160.0

HELP = (
    "Ctrl + left drag: place a body (drag = velocity)\n"
    "  keep Ctrl held and scroll to change its mass,\n"
    "  release Ctrl to add it; right click cancels\n"
    "Ctrl + right click: remove a body\n"
    "Left click: select a body"
)


# ------------------------------------------------------------ restart point

# The state Restart returns to after edits, for one scene at a time. Kept at
# module level so it survives the module being rebuilt.
_restart_point = None


def save_restart_point(scene_name, state, names, colors, radii, view_distance):
    """Make `state` (with these names and looks) what Restart of `scene_name` loads."""
    global _restart_point
    _restart_point = {
        "scene_name": scene_name,
        "state": deepcopy(state),
        "names": list(names),
        "colors": list(colors),
        "radii": list(radii),
        "view_distance": view_distance,
    }


def load_restart_point(scene_name):
    """A fresh Scene from the restart point of `scene_name`, or None if it has none."""
    p = _restart_point
    if p is None or p["scene_name"] != scene_name:
        return None
    return Scene(
        state=deepcopy(p["state"]),
        names=list(p["names"]),
        colors=list(p["colors"]),
        radii=list(p["radii"]),
        view_distance=p["view_distance"],
    )


def clear_restart_point():
    global _restart_point
    _restart_point = None


# --------------------------------------------------------------- orbit helper

# Bodies are only placed in orbit within this fraction of the target's Hill
# radius. Prograde orbits are stable out to about 0.49 Hill radii (Domingos,
# Winter & Yokoyama 2006, MNRAS 373, 1227); a third leaves a margin.
HILL_FRACTION = 1.0 / 3.0


def hill_limited_position(nb, source, target):
    """New position for `source` if it is too far from `target` to orbit it.

    The target's primary is the more massive body pulling hardest on it (for
    a moon's planet, the star). Within the Hill radius
        r_H = a (m / (3 M))^(1/3),
    with a the target-primary distance, m the target's mass and M the
    primary's, the target's gravity dominates the primary's tide. Outside it,
    an orbit around the target is pulled apart. If the source is further than
    HILL_FRACTION r_H, returns the point at that distance in the same
    direction; otherwise (or with no primary) returns None.
    """
    m_t = nb.mass[target]
    candidates = [k for k in range(nb.N) if k not in (source, target) and nb.mass[k] > m_t]
    if not candidates:
        return None

    def pull(k):
        return nb.mass[k] / np.sum((nb.pos[k] - nb.pos[target]) ** 2)

    primary = max(candidates, key=pull)
    a = np.linalg.norm(nb.pos[target] - nb.pos[primary])
    limit = HILL_FRACTION * a * (m_t / (3.0 * nb.mass[primary])) ** (1.0 / 3.0)

    rel = nb.pos[source] - nb.pos[target]
    r = np.linalg.norm(rel)
    if r <= limit or r == 0.0:
        return None
    return nb.pos[target] + rel / r * limit


def circular_orbit_velocity(G, pos, mass, vel, central_pos, central_vel, central_mass):
    """Velocity for a circular orbit around a central body at the current separation.

    Speed sqrt(G (M + m) / r) relative to the central body. The orbit keeps
    the plane and direction of the current relative motion; with no sideways
    relative motion it goes counter-clockwise in the xy-plane seen from +z
    (the xz-plane if the separation is along z). Returns None if the bodies
    coincide or have no mass.
    """
    rel = pos - central_pos
    r = np.linalg.norm(rel)
    total_mass = central_mass + mass
    if r == 0.0 or total_mass <= 0.0:
        return None

    rel_vel = vel - central_vel
    normal = np.cross(rel, rel_vel)
    if np.linalg.norm(normal) <= 1e-9 * r * np.linalg.norm(rel_vel):
        # No sideways motion (or none at all): pick a default plane.
        normal = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(np.cross(normal, rel)) < 1e-9 * r:
            normal = np.array([0.0, -1.0, 0.0])

    direction = np.cross(normal, rel)
    direction /= np.linalg.norm(direction)
    return central_vel + np.sqrt(G * total_mass / r) * direction


# ------------------------------------------------------------ screen helpers


def project(camera, points, W, H):
    """Pixel x, y of world points (M, 3), and whether each is in front of the camera."""
    points = np.atleast_2d(points)
    clip = np.hstack([points, np.ones((len(points), 1))]) @ camera.view_proj().T
    in_front = clip[:, 3] > 0.0
    w = np.where(in_front, clip[:, 3], 1.0)
    return (clip[:, 0] / w + 1.0) * 0.5 * W, (1.0 - clip[:, 1] / w) * 0.5 * H, in_front


def pixel_radii(camera, pos, radii, H, min_radius_px):
    """On-screen radius in pixels of each body disc, as the body renderer draws it."""
    pos = np.atleast_2d(pos)
    view = camera.view()
    depth = -(pos @ view[2, :3] + view[2, 3])
    f = 1.0 / np.tan(camera.fov / 2)
    px = np.asarray(radii, dtype=float) * f * H / (2.0 * np.maximum(depth, 1e-30))
    return np.maximum(px, min_radius_px)


def pick_body(camera, W, H, pos, radii, min_radius_px, mouse):
    """Index of the body under the mouse (nearest centre wins), or None."""
    x, y, in_front = project(camera, pos, W, H)
    dist = np.hypot(x - mouse[0], y - mouse[1])
    hit = in_front & (dist <= pixel_radii(camera, pos, radii, H, min_radius_px) + PICK_SLOP_PX)
    if not hit.any():
        return None
    return int(np.argmin(np.where(hit, dist, np.inf)))


def draw_path(draw_list, camera, W, H, points, color):
    """Draw a world-space path as a line that fades toward its far end."""
    if points is None or len(points) < 2:
        return
    x, y, in_front = project(camera, points, W, H)
    r, g, b = color
    n = len(points) - 1
    for k in range(n):
        if in_front[k] and in_front[k + 1]:
            col = imgui.get_color_u32((r, g, b, 0.9 - 0.7 * k / n))
            draw_list.add_line((x[k], y[k]), (x[k + 1], y[k + 1]), col, 1.5)


def _edit_vec3(label, values):
    """Float3 input that only changes the components the user edited.

    The widget works in float32; writing all three back would round the
    untouched components. Returns the new values, or None if unchanged.
    """
    shown = [float(np.float32(v)) for v in values]
    imgui.set_next_item_width(FIELD_WIDTH * 1.6)
    changed, new = imgui.input_float3(label, shown, "%.6f")
    if not changed:
        return None
    return [float(n) if n != s else float(v) for n, s, v in zip(new, shown, values)]


# ------------------------------------------------------------------- editor


class SceneEditor:
    """The Scene editor window, the mouse controls in the view, and their overlays.

    Works on an NBodyModule `m` through its editing methods: add_body,
    remove_body, rename_body, state_edited, style_edited,
    anchor_state, predict_path.
    """

    def __init__(self, module):
        self.m = module
        self.selected = None  # index of the selected body
        self.orbit_source = None  # body waiting for an orbit target to be clicked
        self.press_pos = None  # where a plain left press started, for click detection
        self.placing = None  # body being placed with Ctrl, see _update_placing

    def update(self, camera, G):
        """Draw the window and handle the mouse. Call once per ImGui frame."""
        io = imgui.get_io()
        W, H = io.display_size.x, io.display_size.y
        if self.selected is not None and self.selected >= self.m.nb.N:
            self.selected = None

        self._draw_window(W, H, G)
        if camera is None or W <= 0 or H <= 0:
            return

        draw_list = imgui.get_background_draw_list()
        self._handle_mouse(camera, G, W, H, draw_list)
        self._draw_selection(camera, W, H, draw_list)
        self._draw_flashes(camera, W, H, draw_list)

    def _draw_flashes(self, camera, W, H, draw_list):
        """Collisions and mergers: a bright burst and an expanding ring (display only)."""
        m = self.m
        now = time.perf_counter()
        m.flashes[:] = [f for f in m.flashes if now - f[3] < FLASH_SECONDS]
        for pos, (r, g, b), size, start in m.flashes:
            x, y, in_front = project(camera, pos, W, H)
            if not in_front[0]:
                continue
            f = (now - start) / FLASH_SECONDS
            px = max(pixel_radii(camera, pos, [size], H, 12.0)[0], 12.0)
            centre = (x[0], y[0])
            burst = (1.0 - f) ** 3
            draw_list.add_circle_filled(
                centre, px * 0.35 * (0.4 + f), imgui.get_color_u32((1.0, 1.0, 1.0, 0.8 * burst))
            )
            draw_list.add_circle_filled(
                centre, px * 0.7 * (0.4 + f), imgui.get_color_u32((r, g, b, 0.35 * burst))
            )
            ring = imgui.get_color_u32((r, g, b, 0.9 * (1.0 - f) ** 1.5))
            draw_list.add_circle(centre, px * (0.3 + 1.7 * f), ring, 0, 2.5)

    def _ring(self, draw_list, camera, W, H, i, rgba):
        """Circle around body i's disc."""
        m = self.m
        x, y, in_front = project(camera, m.nb.pos[i], W, H)
        if not in_front[0]:
            return
        radius = pixel_radii(camera, m.nb.pos[i], [m.scene.radii[i]], H, m.renderer.min_radius_px)[
            0
        ]
        draw_list.add_circle((x[0], y[0]), radius + 4.0, imgui.get_color_u32(rgba), 0, 1.5)

    # ------------------------------------------------------------- window

    def _draw_window(self, W, H, G):
        m = self.m
        imgui.set_next_window_pos(
            (W - WINDOW_MARGIN, WINDOW_MARGIN), imgui.Cond_.always, (1.0, 0.0)
        )
        imgui.set_next_window_size_constraints(
            (0.0, 0.0), (max(W * 0.45, 1.0), max(H - 2 * WINDOW_MARGIN, 1.0))
        )
        flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move
        if not imgui.begin("Scene editor", None, flags)[0]:
            imgui.end()  # collapsed
            return

        imgui.text_disabled("Controls (?)")
        imgui.set_item_tooltip(HELP)

        follow_options = [FOLLOW_NONE, FOLLOW_COM, *m.scene.names]
        if settings["follow"] not in follow_options:
            settings["follow"] = FOLLOW_NONE
        imgui.set_next_item_width(FIELD_WIDTH)
        changed, idx = imgui.combo(
            "Follow", follow_options.index(settings["follow"]), follow_options
        )
        if changed:
            settings["follow"] = follow_options[idx]

        imgui.set_next_item_width(FIELD_WIDTH)
        _, settings["trail_years"] = imgui.slider_float(
            "Trail", settings["trail_years"], 0.0, MAX_TRAIL_YEARS, "%.1f yr"
        )

        imgui.set_next_item_width(FIELD_WIDTH)
        _, settings["lookahead_years"] = imgui.slider_float(
            "Look-ahead", settings["lookahead_years"], 0.0, 10.0, "%.1f yr"
        )
        imgui.set_item_tooltip("Predicted path length for the selected or placed body")

        if imgui.collapsing_header("New body"):
            self._draw_new_body_settings()

        if imgui.collapsing_header("Bodies", imgui.TreeNodeFlags_.default_open):
            for i, name in enumerate(m.scene.names):
                if m.nb.debris[i]:
                    continue  # listed as a count below; still selectable in the view
                r, g, b = m.scene.colors[i]
                imgui.color_button(
                    f"##swatch{i}", (r, g, b, 1.0), imgui.ColorEditFlags_.no_tooltip, (12, 12)
                )
                imgui.same_line()
                if imgui.selectable(f"{name}##{i}", self.selected == i)[0]:
                    self.selected = None if self.selected == i else i

            if m.nb.debris.any():
                imgui.text_disabled(f"Debris: {int(m.nb.debris.sum())} pieces")

        if self.selected is not None:
            imgui.separator()
            self._draw_body_panel(self.selected, G)

        imgui.end()

    def _draw_new_body_settings(self):
        kinds = ["Body", "Star", "Black hole"]
        imgui.set_next_item_width(FIELD_WIDTH)
        changed, k = imgui.combo("Kind##new", kinds.index(settings["new_kind"]), kinds)
        if changed:
            settings["new_kind"] = kinds[k]
            if kinds[k] == "Black hole":
                settings["new_color"] = list(BLACK_HOLE)
                settings["new_mass"] = max(settings["new_mass"], 1.0)
            if kinds[k] == "Star":
                settings["new_color"] = [1.0, 0.85, 0.55]
                settings["new_mass"] = max(settings["new_mass"], 0.1)
        imgui.set_item_tooltip(
            "Body: the size of a rocky body of that mass. Star: the size of a\n"
            "main-sequence star (the Sun: 0.0047 AU), far easier to tear apart.\n"
            "Black hole: swallows what falls into it (decided on the exact orbit\n"
            "inside 6 G M / c^2) and tears apart what comes inside its tidal\n"
            "radius, R (M_hole / m)^(1/3). While you place or select a body,\n"
            "that radius is drawn round each hole (orange), with the hole's\n"
            "capture radius (red)."
        )
        if settings["new_kind"] == "Black hole":
            # A stellar-mass hole tears and swallows only at distances far
            # too small to see in a planetary view; a million solar masses
            # does it at a fraction of an AU.
            for label, value in (("10", 10.0), ("1e6", 1.0e6), ("1e9", 1.0e9)):
                if imgui.small_button(f"{label} M_sun##hole"):
                    settings["new_mass"] = value
                imgui.same_line()
            imgui.text_disabled("presets")
        imgui.set_next_item_width(FIELD_WIDTH)
        changed, mass = imgui.input_double("Mass##new", settings["new_mass"], 0.0, 0.0, "%.3e")
        if changed:
            settings["new_mass"] = max(mass, 0.0)
        imgui.set_item_tooltip("Starting mass; scroll while placing to change it")

        imgui.set_next_item_width(FIELD_WIDTH)
        changed, radius = imgui.input_double(
            "Draw size##new", settings["new_radius"], 0.0, 0.0, "%.4f"
        )
        if changed:
            settings["new_radius"] = max(radius, 0.0)
        imgui.set_item_tooltip(
            "Drawn radius in AU (display only). The physical radius is\nthat of a body of Earth's density with the placed mass."
        )

        imgui.set_next_item_width(FIELD_WIDTH)
        _, color = imgui.color_edit3("Colour##new", settings["new_color"])
        settings["new_color"] = [float(c) for c in color]

        imgui.set_next_item_width(FIELD_WIDTH)
        _, settings["speed_per_au"] = imgui.slider_float(
            "Speed##new",
            settings["speed_per_au"],
            0.1,
            100.0,
            "%.2f",
            imgui.SliderFlags_.logarithmic,
        )
        imgui.set_item_tooltip("AU/yr of velocity per AU dragged")

    def _draw_body_panel(self, i, G):
        m = self.m
        nb, scene = m.nb, m.scene

        imgui.set_next_item_width(FIELD_WIDTH)
        changed, name = imgui.input_text("Name", scene.names[i])
        if changed:
            m.rename_body(i, name)

        imgui.set_next_item_width(FIELD_WIDTH)
        changed, mass = imgui.input_double("Mass", float(nb.mass[i]), 0.0, 0.0, "%.4e")
        if changed:
            nb.mass[i] = max(mass, 0.0)
            if nb.black_hole[i]:
                nb.radius[i] = schwarzschild_radius(nb.mass[i], G)
            m.state_edited()

        new_pos = _edit_vec3("Pos", nb.pos[i])
        if new_pos is not None:
            nb.pos[i] = new_pos
            m.state_edited(moved=i)

        new_vel = _edit_vec3("Vel", nb.vel[i])
        if new_vel is not None:
            nb.vel[i] = new_vel
            m.state_edited()

        imgui.set_next_item_width(FIELD_WIDTH)
        # Use the widget's changed flag: comparing values would see the
        # widget's float32 rounding as an edit on every frame.
        changed, color = imgui.color_edit3("Colour", list(scene.colors[i]))
        if changed:
            scene.colors[i] = tuple(float(c) for c in color)
            m.style_edited()

        imgui.set_next_item_width(FIELD_WIDTH)
        changed, radius = imgui.input_double("Radius", float(nb.radius[i]), 0.0, 0.0, "%.4e")
        if changed:
            nb.radius[i] = max(radius, 0.0)
            m.state_edited()
        imgui.set_item_tooltip("Physical radius in AU, for collisions (0 = never collides)")

        changed, hole = imgui.checkbox("Black hole", bool(nb.black_hole[i]))
        if changed:
            nb.black_hole[i] = hole
            nb.radius[i] = schwarzschild_radius(nb.mass[i], G) if hole else rocky_radius(nb.mass[i])
            m.state_edited()
        imgui.set_item_tooltip(
            "Radius = horizon, 2 G M / c^2. Absorbs whatever comes within\n"
            "3 horizon radii, and is what remains of any merger."
        )

        imgui.set_next_item_width(FIELD_WIDTH)
        changed, radius = imgui.input_double("Draw size", float(scene.radii[i]), 0.0, 0.0, "%.6f")
        if changed:
            scene.radii[i] = max(radius, 0.0)
            m.style_edited()
        imgui.set_item_tooltip("Drawn radius in AU (display only)")

        imgui.begin_disabled(nb.N < 2)
        if imgui.button("Make orbit"):
            self.orbit_source = None if self.orbit_source == i else i
        imgui.end_disabled()
        imgui.set_item_tooltip(
            "Then click the body to orbit: sets a circular-orbit velocity,\n"
            "and moves this body closer if it is too far out to stay bound"
        )

        imgui.same_line()
        following = settings["follow"] == scene.names[i]
        if imgui.button("Unfollow" if following else "Follow"):
            settings["follow"] = FOLLOW_NONE if following else scene.names[i]

        # Removed last, after everything that reads body i has been drawn.
        imgui.same_line()
        remove = imgui.button("Remove")

        if self.orbit_source == i:
            imgui.text_colored(
                (1.0, 0.8, 0.3, 1.0), "Click the body to orbit (right click cancels)"
            )

        if remove:
            self._remove(i)

    # -------------------------------------------------------------- mouse

    def _handle_mouse(self, camera, G, W, H, draw_list):
        io = imgui.get_io()
        m = self.m
        nb, scene = m.nb, m.scene
        ctrl = io.key_ctrl
        mouse = (io.mouse_pos.x, io.mouse_pos.y)

        # While Ctrl is held or a body is being placed, take the mouse from
        # the app so it does not orbit, pan or zoom the camera.
        if ctrl or self.placing is not None:
            imgui.set_next_frame_want_capture_mouse(True)

        if self.placing is not None:
            self._update_placing(camera, G, W, H, draw_list, ctrl, mouse)
            return

        over_ui = (
            imgui.is_window_hovered(imgui.HoveredFlags_.any_window) or imgui.is_any_item_active()
        )
        if over_ui:
            if imgui.is_mouse_clicked(0):
                self.press_pos = None
            return

        hover = pick_body(camera, W, H, nb.pos, scene.radii, m.renderer.min_radius_px, mouse)

        if ctrl:
            # The app does not see the wheel while we hold the mouse, so zoom here.
            if io.mouse_wheel != 0.0:
                camera.zoom(io.mouse_wheel)
            if hover is not None:
                self._ring(draw_list, camera, W, H, hover, (1.0, 0.3, 0.3, 0.9))

            if imgui.is_mouse_clicked(0):
                self._start_placing(camera, W, H, mouse)
            elif imgui.is_mouse_clicked(1) and hover is not None:
                self._remove(hover)
            return

        if self.orbit_source is not None:
            if hover is not None and hover != self.orbit_source:
                self._ring(draw_list, camera, W, H, hover, (1.0, 0.8, 0.3, 0.9))
            draw_list.add_text(
                (mouse[0] + 14.0, mouse[1] + 10.0),
                imgui.get_color_u32((1.0, 0.8, 0.3, 1.0)),
                f"{scene.names[self.orbit_source]} orbits ...",
            )
            if imgui.is_mouse_clicked(1):
                self.orbit_source = None

        # A plain left click (press and release without dragging the camera).
        if imgui.is_mouse_clicked(0):
            self.press_pos = mouse
        if imgui.is_mouse_released(0) and self.press_pos is not None:
            moved = np.hypot(mouse[0] - self.press_pos[0], mouse[1] - self.press_pos[1])
            self.press_pos = None
            if moved <= CLICK_SLOP_PX:
                self._click(G, hover)

    def _click(self, G, idx):
        if self.orbit_source is None:
            self.selected = idx
            return
        if idx is None or idx == self.orbit_source:
            return

        m = self.m
        nb = m.nb
        i = self.orbit_source
        self.orbit_source = None

        new_pos = hill_limited_position(nb, i, idx)
        pos = nb.pos[i] if new_pos is None else new_pos
        vel = circular_orbit_velocity(
            G, pos, nb.mass[i], nb.vel[i], nb.pos[idx], nb.vel[idx], nb.mass[idx]
        )
        if vel is None:
            return
        nb.pos[i] = pos
        nb.vel[i] = vel
        m.state_edited(moved=None if new_pos is None else i)

    def _remove(self, i):
        self.m.remove_body(i)

    def on_body_removed(self, i, replacement=None):
        """Keep indices right after body i is deleted. `replacement` (an
        index from before the deletion) takes over i's selection, as when i
        merged into it."""

        def shift(idx):
            if idx is None:
                return None
            if idx == i:
                idx = replacement
                if idx is None:
                    return None
            return idx - 1 if idx > i else idx

        self.selected = shift(self.selected)
        self.orbit_source = shift(self.orbit_source)

    # ----------------------------------------------------------- placing

    def _anchor(self):
        """Position and velocity placed bodies are relative to: the followed
        body or centre of mass, or the origin at rest."""
        return self.m.anchor_state()

    def _start_placing(self, camera, W, H, mouse):
        cursor = camera.screen_to_plane(mouse[0], mouse[1], W, H)
        if cursor is None:
            return
        anchor_pos, _ = self._anchor()
        self.placing = {
            "start_rel": cursor - anchor_pos,  # position relative to the anchor
            "vel_rel": np.zeros(3),  # velocity relative to the anchor
            "mass": settings["new_mass"],
            "dragging": True,  # left button still down
        }

    def _update_placing(self, camera, G, W, H, draw_list, ctrl, mouse):
        """The left drag sets the velocity. Scrolling changes the mass. The body
        is added once both the button and Ctrl are up. Right click cancels."""
        p = self.placing
        io = imgui.get_io()

        if imgui.is_mouse_clicked(1):
            self.placing = None
            return

        if io.mouse_wheel != 0.0:
            p["mass"] *= MASS_STEP**io.mouse_wheel

        anchor_pos, anchor_vel = self._anchor()
        start = p["start_rel"] + anchor_pos
        speed = settings["speed_per_au"]

        if p["dragging"]:
            cursor = camera.screen_to_plane(mouse[0], mouse[1], W, H)
            if cursor is not None:
                p["vel_rel"] = (cursor - start) * speed
            if not imgui.is_mouse_down(0):
                p["dragging"] = False

        body = {
            "mass": p["mass"],
            "pos": start,
            "vel": p["vel_rel"] + anchor_vel,
            "color": tuple(settings["new_color"]),
            "radius": settings["new_radius"],
            # Physical radius for collisions: a body of Earth's density, or
            # for a black hole its horizon.
            "phys_radius": new_body_radius(settings["new_kind"], p["mass"], G),
            "black_hole": settings["new_kind"] == "Black hole",
        }

        if not p["dragging"] and not ctrl:
            self.placing = None
            self.m.add_body(body)
            self.selected = self.m.nb.N - 1
            return

        self._draw_placing(camera, G, W, H, draw_list, body, start + p["vel_rel"] / speed)

    def _draw_placing(self, camera, G, W, H, draw_list, body, drag_end):
        m = self.m
        nb = m.nb

        draw_path(draw_list, camera, W, H, m.predict_path(extra=body), body["color"])
        if not body["black_hole"]:
            self._draw_hole_zones(camera, W, H, draw_list, body["mass"], body["phys_radius"])

        x, y, in_front = project(camera, np.array([body["pos"], drag_end]), W, H)
        if not in_front[0]:
            return
        white = imgui.get_color_u32((1.0, 1.0, 1.0, 0.8))
        r, g, b = body["color"]
        if in_front[1]:
            draw_list.add_line((x[0], y[0]), (x[1], y[1]), white, 1.5)
        draw_list.add_circle_filled((x[0], y[0]), 4.0, imgui.get_color_u32((r, g, b, 1.0)))

        # Speed relative to the most massive body, next to the circular-orbit
        # speed around it at this distance, sqrt(G (M + m) / r).
        text = f"m = {body['mass']:.2e}"
        if nb.N == 0:
            text += f"\nv = {np.linalg.norm(body['vel']):.3f} AU/yr"
        else:
            heavy = int(np.argmax(nb.mass))
            speed = np.linalg.norm(body["vel"] - nb.vel[heavy])
            text += f"\nv = {speed:.3f} AU/yr"
            dist = np.linalg.norm(body["pos"] - nb.pos[heavy])
            total_mass = nb.mass[heavy] + body["mass"]
            if dist > 0.0 and total_mass > 0.0:
                text += f"  (circular: {np.sqrt(G * total_mass / dist):.3f})"

        # Label on the side facing the screen centre, so it is not cut off.
        size = imgui.calc_text_size(text)
        tx = x[0] + 12.0 if x[0] < W / 2 else x[0] - 12.0 - size.x
        draw_list.add_text((tx, y[0] + 10.0), white, text)

    # ---------------------------------------------------------- selection

    def _draw_selection(self, camera, W, H, draw_list):
        i = self.selected
        if i is None or self.placing is not None:
            return
        m = self.m
        draw_path(draw_list, camera, W, H, m.predict_path(idx=i), m.scene.colors[i])
        self._ring(draw_list, camera, W, H, i, (1.0, 1.0, 1.0, 0.8))
        nb = m.nb
        if not nb.black_hole[i] and not nb.debris[i]:
            self._draw_hole_zones(camera, W, H, draw_list, nb.mass[i], nb.radius[i])

    def _draw_hole_zones(self, camera, W, H, draw_list, mass, radius):
        """Round every black hole: its capture radius (red) and the radius inside
        which its tide tears a body of this mass and radius apart (orange)."""
        nb = self.m.nb
        if mass <= 0.0 or radius <= 0.0:
            return
        for h in np.flatnonzero(nb.black_hole):
            # Shown: the capture radius of a body falling from far away,
            # 8 G M / c^2 = 4 horizon radii in these coordinates' Newtonian
            # terms (pericentre of the L = 4 G M / c parabola).
            capture = 4.0 * nb.radius[h]
            tidal = radius * (nb.mass[h] / mass) ** (1.0 / 3.0)
            x, y, in_front = project(camera, nb.pos[h], W, H)
            if not in_front[0]:
                continue
            for size, rgba, label in (
                (tidal, (1.0, 0.6, 0.2, 0.8), "torn apart inside"),
                (capture, (1.0, 0.25, 0.25, 0.8), "swallowed inside"),
            ):
                if size == tidal and tidal <= capture:
                    continue  # the hole swallows this body whole
                px = pixel_radii(camera, nb.pos[h], [size], H, 0.0)[0]
                col = imgui.get_color_u32(rgba)
                if px < 3.0:
                    # Too small to draw at this zoom: say how small.
                    if size == tidal:
                        draw_list.add_text(
                            (x[0] + 14.0, y[0] - 22.0),
                            col,
                            f"torn apart only inside {tidal:.2g} AU (too small to see here)",
                        )
                    continue
                draw_list.add_circle((x[0], y[0]), px, col, 64, 1.0)
                draw_list.add_text((x[0] + px * 0.71 + 3.0, y[0] - px * 0.71 - 12.0), col, label)

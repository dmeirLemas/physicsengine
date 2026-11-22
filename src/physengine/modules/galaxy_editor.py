"""The Galaxy setup window: edit the galaxy models and the encounter. UI only.

The window edits a draft copy of the scene. Apply stores the draft as the
scene's edited version and rebuilds the module from it. Restart keeps the
edits, Full reset returns to the preset (galaxy_encounters.SCENES).

Units: kpc, 1e10 solar masses, degrees.
"""

from copy import deepcopy

from imgui_bundle import imgui

from physengine.modules.galaxy_encounters import SCENES, apocentre

WINDOW_MARGIN = 10.0
FIELD_WIDTH = 170.0

HELP = (
    "Edit the scene, then Apply to rebuild it from the new values.\n"
    "Restart keeps the edits; Full reset (Shift+R) returns to the preset.\n"
    "Masses in 1e10 solar masses, lengths in kpc."
)
ANGLES_HELP = (
    "Toomre & Toomre (1972) angles of each disc, describing the\n"
    "companion's orbit with the disc as the reference plane.\n"
    "Inclination: tilt between disc and orbit (0 prograde, 180 retrograde).\n"
    "Argument: angle in the orbit plane from where the companion\n"
    "crosses the disc plane (going to the spin side) to the pericentre."
)

# Lowest start distance offered [kpc], unless the orbit's apocentre is closer.
MIN_START = 10.0
MAX_START = 400.0


# ----------------------------------------------------------- edited scene

# The edited scene, for one scene name at a time. Kept at module level so it
# survives the galaxy module being rebuilt.
_edited = None


def scene_for(scene_name):
    """A fresh copy of the scene to build: the edited one if there is one, else the preset."""
    if _edited is not None and _edited[0] == scene_name:
        return deepcopy(_edited[1])
    return deepcopy(SCENES[scene_name])


def save_edits(scene_name, scene):
    global _edited
    _edited = (scene_name, deepcopy(scene))


def clear_edits():
    global _edited
    _edited = None


# ---------------------------------------------------------------- widgets


def _float(label, obj, attr, lo, hi, fmt="%.2f", log=False, tooltip=None):
    imgui.set_next_item_width(FIELD_WIDTH)
    flags = imgui.SliderFlags_.logarithmic if log else 0
    changed, value = imgui.slider_float(label, getattr(obj, attr), lo, hi, fmt, flags)
    if tooltip:
        imgui.set_item_tooltip(tooltip)
    if changed:
        setattr(obj, attr, float(min(max(value, lo), hi)))
    return changed


def _int(label, obj, attr, lo, hi, tooltip=None):
    imgui.set_next_item_width(FIELD_WIDTH)
    flags = imgui.SliderFlags_.logarithmic
    changed, value = imgui.slider_int(label, getattr(obj, attr), lo, hi, "%d", flags)
    if tooltip:
        imgui.set_item_tooltip(tooltip)
    if changed:
        setattr(obj, attr, int(min(max(value, lo), hi)))
    return changed


def _hernquist(comp, tag, n_range, mass_range, a_range, r_max_range):
    _int(f"Particles##{tag}", comp, "n", *n_range)
    _float(f"Mass##{tag}", comp, "mass", *mass_range, log=True, tooltip="1e10 solar masses")
    _float(f"Scale radius##{tag}", comp, "a", *a_range, fmt="%.2f kpc")
    _float(f"Cut-off##{tag}", comp, "r_max", *r_max_range, fmt="%.1f kpc")
    comp.r_max = max(comp.r_max, 2.0 * comp.a)  # keep the cut-off outside the core


def _model(model, g):
    """Widgets for one GalaxyModel; g tags the widget ids."""
    if imgui.collapsing_header(f"Dark halo##{g}"):
        _hernquist(model.halo, f"halo{g}", (500, 20000), (5.0, 200.0), (5.0, 60.0), (50.0, 600.0))
    if imgui.collapsing_header(f"Bulge##{g}"):
        _hernquist(model.bulge, f"bulge{g}", (100, 5000), (0.1, 5.0), (0.3, 3.0), (2.0, 30.0))
    if imgui.collapsing_header(f"Disc##{g}"):
        d, tag = model.disc, f"disc{g}"
        _int(f"Particles##{tag}", d, "n", 500, 20000)
        _float(f"Mass##{tag}", d, "mass", 0.5, 20.0, log=True, tooltip="1e10 solar masses")
        _float(f"Scale length##{tag}", d, "R_d", 1.0, 8.0, fmt="%.2f kpc")
        _float(f"Scale height##{tag}", d, "z0", 0.1, 1.0, fmt="%.2f kpc")
        _float(f"Cut-off##{tag}", d, "R_max", 5.0, 50.0, fmt="%.1f kpc")
        d.R_max = max(d.R_max, 2.0 * d.R_d)
        _float(
            f"Toomre Q##{tag}",
            d,
            "Q",
            0.8,
            3.0,
            tooltip="Stability of the disc at 2.5 scale lengths.\n"
            "Below 1 it breaks into clumps; 1.5 gives spiral structure;\n"
            "higher is hotter and smoother.",
        )
    imgui.text_disabled(f"Total mass {model.total_mass():.1f}, {model.particle_count()} particles")


def _orbit(enc):
    _float(
        "Pericentre",
        enc,
        "pericentre",
        0.0,
        60.0,
        fmt="%.1f kpc",
        tooltip="Closest approach of the two centres on the first\n"
        "passage (the simulated one comes out 1-3 kpc closer);\n0 is head-on",
    )
    if enc.pericentre == 0.0:
        enc.eccentricity = 1.0  # a radial orbit is set up as parabolic only
    imgui.begin_disabled(enc.pericentre == 0.0)
    _float(
        "Eccentricity",
        enc,
        "eccentricity",
        0.3,
        2.0,
        tooltip="Below 1: bound, (apocentre - pericentre) / (apocentre + pericentre);\n"
        "falls back and merges sooner.\n"
        "1: parabolic. Above 1: unbound, faster flyby.",
    )
    imgui.end_disabled()

    r_a = apocentre(enc.pericentre, enc.eccentricity)
    hi = min(r_a, MAX_START)
    lo = min(max(enc.pericentre, MIN_START), hi)
    enc.start_distance = min(max(enc.start_distance, lo), hi)
    _float("Start distance", enc, "start_distance", lo, hi, fmt="%.1f kpc")

    if enc.eccentricity < 1.0:
        imgui.text_disabled(f"Bound orbit, apocentre {r_a:.1f} kpc")
    elif enc.eccentricity == 1.0:
        imgui.text_disabled("Parabolic orbit")
    else:
        imgui.text_disabled("Hyperbolic orbit")


def _angles(enc):
    for g, disc in enumerate(enc.discs):
        imgui.text(f"Galaxy {g + 1}")
        _float(f"Inclination##{g}", disc, "inclination", 0.0, 180.0, fmt="%.0f deg")
        _float(f"Argument##{g}", disc, "argument", 0.0, 360.0, fmt="%.0f deg")


# ----------------------------------------------------------------- window


class SetupEditor:
    """The Galaxy setup window. Works on a GalaxyModule `m` (m.scene_name, m.scene)."""

    def __init__(self, module):
        self.m = module
        self.draft = None  # edited copy of m.scene, made on the first frame

    def update(self):
        m = self.m
        if m.scene is None:
            return
        if self.draft is None:
            self.draft = deepcopy(m.scene)

        W, H = imgui.get_io().display_size
        pos = (W - WINDOW_MARGIN, WINDOW_MARGIN)
        imgui.set_next_window_pos(pos, imgui.Cond_.always, (1.0, 0.0))
        # Taller than the screen: scroll instead of running off the bottom.
        max_size = (max(W * 0.45, 1.0), max(H - 2 * WINDOW_MARGIN, 1.0))
        imgui.set_next_window_size_constraints((0.0, 0.0), max_size)
        flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move
        if not imgui.begin("Galaxy setup", None, flags)[0]:
            imgui.end()  # collapsed
            return

        imgui.text_disabled(f"{m.scene_name} (?)")
        imgui.set_item_tooltip(HELP)

        draft = self.draft
        enc = draft.encounter
        if enc is not None:
            if imgui.collapsing_header("Orbit", imgui.TreeNodeFlags_.default_open):
                _orbit(enc)
            if imgui.collapsing_header("Disc orientation", imgui.TreeNodeFlags_.default_open):
                imgui.text_disabled("Toomre angles (?)")
                imgui.set_item_tooltip(ANGLES_HELP)
                _angles(enc)

        if len(draft.galaxies) == 1:
            _model(draft.galaxies[0], 0)
        elif imgui.begin_tab_bar("galaxies"):
            for g, model in enumerate(draft.galaxies):
                if imgui.begin_tab_item(f"Galaxy {g + 1}")[0]:
                    _model(model, g)
                    other = 1 - g
                    if imgui.button(f"Copy to galaxy {other + 1}"):
                        draft.galaxies[other] = deepcopy(model)
                    imgui.end_tab_item()
            imgui.end_tab_bar()

        imgui.separator()
        changed = draft != m.scene
        imgui.begin_disabled(not changed)
        if imgui.button("Apply"):
            save_edits(m.scene_name, draft)
            m.reset_requested = True
        imgui.set_item_tooltip("Rebuild the scene from these values")
        imgui.same_line()
        if imgui.button("Revert"):
            self.draft = deepcopy(m.scene)
        imgui.set_item_tooltip("Back to the values the scene was built with")
        imgui.end_disabled()
        imgui.same_line()
        if imgui.button("Preset"):
            self.draft = deepcopy(SCENES[m.scene_name])
        imgui.set_item_tooltip("Load the preset's values (Apply to use them)")
        if changed:
            imgui.text_colored((1.0, 0.75, 0.3, 1.0), "Not applied yet")

        imgui.end()

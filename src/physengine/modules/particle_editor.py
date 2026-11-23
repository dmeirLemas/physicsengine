"""The "Custom event" window of the particle-physics module. UI only.

It edits a draft of a custom setup (particle_custom.Setup): a list of
entries, each a particle that is fired through the detector or a parent that
is made to decay. Every frame the draft is checked (particle_custom.check);
a valid draft that differs from the setup that runs is handed to the module
(ParticleModule.apply_custom), which shows an event of it. An invalid draft
is not applied: the last valid setup keeps running and the window says what
is wrong.

The draft is kept at module level, so that it survives the particle module
being rebuilt (Restart); Full reset forgets it.

Units shown: GeV, degrees, metres, tesla.
"""

import math
from copy import deepcopy

from imgui_bundle import imgui

from physengine.modules import particle_custom as custom
from physengine.modules import scenario_files
from physengine.modules.particle_custom import Entry
from physengine.modules.particle_data import ABOUT, PARTICLES

WINDOW_MARGIN = 10.0
FIELD_WIDTH = 150.0
WINDOW_SIZE = (380.0, 620.0)  # default width, height; the content scrolls inside
WRAP = 330.0  # pixels: width of the window's text
ERROR = (1.0, 0.5, 0.4, 1.0)
WARNING = (1.0, 0.8, 0.35, 1.0)
GOOD = (0.55, 0.9, 0.6, 1.0)
ETA_SLIDER = 5.0  # the pseudorapidity slider's range (typed values may go to ETA_MAX)
START_RANGE = 3.0  # m: the range of the starting point's fields
PANEL_ABOVE = 330.0  # pixels left free at the top for the app's Simulation panel

QUANTITY_LABELS = ("Momentum p", "Total energy E", "Kinetic energy T")
ANGLE_LABELS = ("Polar angle theta", "Pseudorapidity eta")
WHERE_LABELS = ("After its lifetime (flies first)", "At its starting point (forced)")
FACE_CAPTION = "Particles pass through each other: they do not interact in this simulation."
COLLIDE_CAPTION = (
    "Collide: the pair forms a Z where the two meet, with probability given by the"
    " Breit-Wigner at their sqrt(s) (1 at the peak, 1/2 at half a width from it), and"
    " only within 10 widths of the Z mass; otherwise they pass through. Stated"
    " approximations: the rate is not computed from a cross section (at the peak a"
    " pair always forms the Z), and the Z decays isotropically (the direction of the"
    " incoming pair is ignored for the decay angle)."
)
# The selected entry's sections and whether each starts open.
SECTIONS = (("What", True), ("Where", False), ("How many", False), ("Face each other", False))
ABOUT_TEXT = (
    "Place particles by hand and watch what the detector does with them. This is"
    " not a collision: nothing here says what a collision would produce. The one"
    " exception is a pair set to collide (e+ e- or mu+ mu-, under 'Face each other'),"
    " which can form a Z."
)
RANDOM_MODE = "Random, by branching fraction"
TYPED_MODE = "Other: type the daughters"

# ------------------------------------------------------------------ draft

_draft = None  # the Setup being edited
_selected = 0  # index of the entry whose controls are shown
_search = ""  # the particle-name search box
_partner = 0  # index of the entry chosen to face the selected one
MAX_MATCHES = 15  # most names the search lists


def draft():
    global _draft
    if _draft is None:
        _draft = deepcopy(custom.current())
    return _draft


def set_draft(setup):
    """Edit a copy of `setup` from now on (a preset, a loaded file)."""
    global _draft, _selected
    _draft = deepcopy(setup)
    _selected = 0


def forget_draft():
    """Drop the draft: the window starts again from the setup that runs (Full reset)."""
    global _draft, _selected
    _draft = None
    _selected = 0


# ---------------------------------------------------------------- helpers


def _name_label(name):
    return f"{name}   ({ABOUT[name][0]})" if name in ABOUT else name


def matching_names(text, limit=MAX_MATCHES):
    """Names of the particle table containing `text` (case-insensitive), at most `limit`."""
    needle = text.strip().lower()
    return [n for n in PARTICLES if needle in n.lower()][:limit]


# Separation put between two entries that share one start point (metres).
# Chosen as 1.0 m: small against a detector, large enough to see two tracks.
PAIR_SEPARATION = 1.0


def face_each_other(a, b):
    """Point entries a and b at each other's starting point (by theta and phi).
    Starts that coincide are first pulled apart along the beam (z) axis by
    PAIR_SEPARATION: both at the origin gives z = -0.5 m and +0.5 m; otherwise
    a keeps its start and b moves to 1 m above it. Returns a one-line caption."""
    d = [float(q) - float(p) for p, q in zip(a.start, b.start)]
    length = math.sqrt(sum(c * c for c in d))
    moved = ""
    if length < 1.0e-9:
        half = 0.5 * PAIR_SEPARATION
        if all(abs(float(c)) < 1.0e-9 for c in a.start):
            a.start = (0.0, 0.0, -half)
            b.start = (0.0, 0.0, half)
        else:
            b.start = (float(a.start[0]), float(a.start[1]), float(a.start[2]) + PAIR_SEPARATION)
        d = [float(q) - float(p) for p, q in zip(a.start, b.start)]
        length = math.sqrt(sum(c * c for c in d))
        moved = f"Starts {PAIR_SEPARATION:g} m apart on the beam axis. "
    theta = math.degrees(math.acos(max(-1.0, min(1.0, d[2] / length))))
    phi = math.degrees(math.atan2(d[1], d[0]))
    a.angles, a.theta, a.phi = "theta", theta, phi
    b.angles, b.theta = "theta", 180.0 - theta
    b.phi = phi - 180.0 if phi > 0.0 else phi + 180.0
    return f"{moved}Aimed at each other: A theta {a.theta:.1f}, phi {a.phi:.1f}; B theta {b.theta:.1f}, phi {b.phi:.1f} deg."


def controls_for(kind):
    """Names of the control groups shown for an entry of kind "fire" or "decay"."""
    groups = ["kind", "particle", "momentum", "direction", "start", "count", "spread", "face"]
    if kind == "decay":
        groups[2:2] = ["decay mode", "daughters", "where it decays"]
    return tuple(groups)


def _percent(fraction):
    return f"{100.0 * fraction:.4g}%"


def mode_labels(name):
    """The choices of the decay-mode box for parent `name`, and the daughters
    string each stands for (None for the typed one)."""
    labels, values = [RANDOM_MODE], [""]
    for fraction, daughters in custom.modes_of(name):
        text = " ".join(daughters)
        shown = "Z Z* -> 4 leptons (e, mu)" if tuple(daughters) == custom.HIGGS_ZZ else text
        labels.append(f"{shown}   {_percent(fraction)}")
        values.append(text)
    labels.append(TYPED_MODE)
    values.append(None)
    return labels, values


def _speed(beta):
    if beta < 0.9999:
        return f"{beta:.4f} c"
    return f"c less {(1.0 - beta) * 299792458.0:.3g} m/s"


def _length(metres):
    if metres >= 1000.0:
        return f"{metres / 1000.0:.3g} km"
    if metres >= 1.0:
        return f"{metres:.3g} m"
    if metres >= 0.01:
        return f"{metres * 100.0:.3g} cm"
    if metres >= 1.0e-5:
        return f"{metres * 1000.0:.3g} mm"
    return f"{metres * 1.0e6:.3g} micrometres"


def readout(entry, field_strength):
    """The derived quantities of an entry as lines of text (empty if it has none)."""
    try:
        kin = custom.kinematics(entry, field_strength)
    except ValueError:
        return []
    particle = PARTICLES[entry.name]
    lines = [
        f"p = {kin.momentum:.5g} GeV   E = {kin.energy:.5g} GeV   T = {kin.kinetic:.5g} GeV",
    ]
    if particle.mass > 0.0:
        lines.append(f"speed beta = {_speed(kin.beta)}   gamma = {kin.gamma:.5g}")
    else:
        lines.append("speed: that of light (no mass)")
    eta = f"{kin.eta:.3f}" if abs(kin.eta) != float("inf") else ("+inf" if kin.eta > 0 else "-inf")
    lines.append(f"p_T = {kin.pt:.5g} GeV   theta = {kin.theta:.2f} deg   eta = {eta}")
    if particle.width > 0.0:
        lines.append(f"decays at once (width {particle.width:.4g} GeV)")
    elif kin.decay_length is not None:
        lines.append(f"mean decay length = {_length(kin.decay_length)}   (beta gamma c tau)")
        lines.append(f"   with c tau = {_length(particle.ctau)}")
    else:
        lines.append("does not decay here (stable in this table)")
    if particle.charge == 0:
        lines.append("no charge: not bent by the magnet")
    elif kin.radius is None:
        lines.append("no momentum across the field: not bent")
    else:
        lines.append(
            f"bending radius = {_length(kin.radius)} in {field_strength:.2f} T   (p_T / 0.3 q B)"
        )
    return lines


# ----------------------------------------------------------------- window


class CustomWindow:
    def __init__(self, module):
        self.m = module
        self.pair_message = ""
        self.message = ""  # why the draft could not be applied, if apply itself failed

    # -- the pieces

    def _presets(self):
        imgui.set_next_item_width(FIELD_WIDTH)
        changed, k = imgui.combo("Preset", 0, ["Choose a preset..."] + list(custom.PRESETS))
        if changed and k > 0:
            set_draft(custom.preset(custom.PRESETS[k - 1]))

    def _entry_list(self, setup, checks):
        global _selected
        total = custom.placed(setup)
        imgui.text(f"Entries: {len(setup.entries)} of at most {custom.MAX_ENTRIES}")
        imgui.same_line()
        imgui.text_disabled(f"({total} particles placed, at most {custom.MAX_PLACED})")
        for i, entry in enumerate(setup.entries):
            what = "Fire" if entry.action == "fire" else "Decay"
            mark = "  [!]" if checks[i][0] else ""
            if imgui.selectable(
                f"{i + 1}. {what}: {custom.describe(entry)}{mark}##entry{i}", i == _selected
            )[0]:
                _selected = i
        full = len(setup.entries) >= custom.MAX_ENTRIES
        imgui.begin_disabled(full)
        if imgui.button("Add: fire a particle"):
            setup.entries.append(Entry("fire", "mu-", value=10.0))
            _selected = len(setup.entries) - 1
        imgui.same_line()
        if imgui.button("Add: decay a parent"):
            setup.entries.append(Entry("decay", "K0S", value=5.0, daughters="pi+ pi-"))
            _selected = len(setup.entries) - 1
        imgui.end_disabled()
        has = 0 <= _selected < len(setup.entries)
        imgui.begin_disabled(full or not has)
        if imgui.button("Duplicate"):
            copy = deepcopy(setup.entries[_selected])
            copy.collide = ""  # a collision is between two entries, not three
            setup.entries.insert(_selected + 1, copy)
            _selected += 1
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(not has)
        if imgui.button("Remove"):
            custom.unpair(setup, _selected)  # its partner no longer collides
            del setup.entries[_selected]
            _selected = max(min(_selected, len(setup.entries) - 1), 0)
        imgui.end_disabled()

    def _entry_controls(self, entry, setup=None):
        """The selected entry's controls, in collapsing sections (SECTIONS).
        Decay-only controls appear only for a decay entry."""
        for title, opened in SECTIONS:
            imgui.set_next_item_open(opened, imgui.Cond_.appearing)
            if not imgui.collapsing_header(title):
                continue
            if title == "What":
                self._what(entry)
            elif title == "Where":
                self._where(entry)
            elif title == "How many":
                self._how_many(entry)
            elif title == "Face each other":
                self._face(entry, setup)

    def _what(self, entry):
        global _search
        field_strength = self.m.field
        fire = entry.action == "fire"
        if imgui.radio_button("Fire it", fire):
            entry.action = "fire"
        imgui.set_item_tooltip("Fire it through the detector")
        imgui.same_line()
        if imgui.radio_button("Decay it", not fire):
            entry.action = "decay"
        imgui.set_item_tooltip("Make it a parent that decays")
        names = list(PARTICLES)
        imgui.set_next_item_width(FIELD_WIDTH)
        _search = imgui.input_text("Search", _search)[1][:40]
        if _search.strip():
            hits = matching_names(_search)
            if not hits:
                imgui.text_disabled("No particle matches")
            for n in hits:
                if imgui.selectable(f"{_name_label(n)}##hit{n}", n == entry.name)[0]:
                    entry.name = n
                    if not entry.typed:
                        entry.daughters = ""
            total = sum(1 for n in PARTICLES if _search.strip().lower() in n.lower())
            if total > len(hits):
                imgui.text_disabled(f"... {total - len(hits)} more: type more letters")
        imgui.set_next_item_width(FIELD_WIDTH)
        index = names.index(entry.name) if entry.name in names else 0
        changed, index = imgui.combo("Particle", index, [_name_label(n) for n in names])
        if changed:
            entry.name = names[index]
            if not entry.typed:
                entry.daughters = ""  # the old parent's mode means nothing for the new one
        if entry.action == "decay":
            self._decay_controls(entry)

        try:
            kin = custom.kinematics(entry, field_strength)
        except ValueError:
            kin = None
        imgui.set_next_item_width(FIELD_WIDTH)
        q = custom.QUANTITIES.index(entry.quantity) if entry.quantity in custom.QUANTITIES else 0
        changed, q = imgui.combo("Given", q, list(QUANTITY_LABELS))
        if changed:
            entry.quantity = custom.QUANTITIES[q]
            if kin is not None:  # keep the momentum: show it as the new quantity
                entry.value = (kin.momentum, kin.energy, kin.kinetic)[q]
        imgui.set_next_item_width(FIELD_WIDTH)
        flags = imgui.SliderFlags_.logarithmic
        changed, value = imgui.slider_float(
            f"{QUANTITY_LABELS[q]} [GeV]", float(entry.value), 0.0, custom.P_MAX, "%.4g", flags
        )
        if changed:
            entry.value = float(value)
        imgui.set_item_tooltip("Drag, or Ctrl+click to type a value")
        imgui.same_line()
        if imgui.small_button("At rest"):
            entry.value = PARTICLES[entry.name].mass if entry.quantity == "energy" else 0.0

        imgui.set_next_item_width(FIELD_WIDTH)
        a = custom.ANGLES.index(entry.angles) if entry.angles in custom.ANGLES else 0
        changed, a = imgui.combo("Direction by", a, list(ANGLE_LABELS))
        if changed:
            entry.angles = custom.ANGLES[a]
            if kin is not None:  # keep the direction
                if entry.angles == "eta":
                    entry.eta = float(max(min(kin.eta, custom.ETA_MAX), -custom.ETA_MAX))
                else:
                    entry.theta = float(kin.theta)
        imgui.set_next_item_width(FIELD_WIDTH)
        if entry.angles == "eta":
            changed, value = imgui.slider_float(
                "Pseudorapidity eta", float(entry.eta), -ETA_SLIDER, ETA_SLIDER, "%.2f"
            )
            if changed:
                entry.eta = float(value)
            imgui.set_item_tooltip(
                "eta = -ln tan(theta / 2): 0 is across the beam, +-5 almost along it"
            )
        else:
            changed, value = imgui.slider_float(
                "Polar angle theta [deg]", float(entry.theta), 0.0, 180.0, "%.1f"
            )
            if changed:
                entry.theta = float(value)
            imgui.set_item_tooltip(
                "From the beam (z) axis: 90 is straight across, 0 and 180 along the beam"
            )
        imgui.set_next_item_width(FIELD_WIDTH)
        changed, value = imgui.slider_float(
            "Azimuth phi [deg]", float(entry.phi), -180.0, 180.0, "%.0f"
        )
        if changed:
            entry.phi = float(value)

    def _where(self, entry):
        imgui.set_next_item_width(FIELD_WIDTH)
        changed, value = imgui.drag_float3(
            "Start x, y, z [m]",
            [float(v) for v in entry.start],
            0.01,
            -START_RANGE,
            START_RANGE,
            "%.2f",
        )
        if changed:
            entry.start = tuple(float(v) for v in value)
        imgui.set_item_tooltip(
            "Where the particle starts. The beam runs along z; the tracker reaches\n"
            "1.2 m from it and 2.8 m along it. Drag, or Ctrl+click to type."
        )

    def _how_many(self, entry):
        imgui.set_next_item_width(FIELD_WIDTH)
        changed, value = imgui.slider_int("Count", int(entry.count), 1, custom.MAX_COUNT)
        if changed:
            entry.count = int(value)
        imgui.set_next_item_width(FIELD_WIDTH)
        changed, value = imgui.slider_float(
            "Momentum spread [%]", float(entry.spread_p), 0.0, custom.SPREAD_P_MAX, "%.0f"
        )
        if changed:
            entry.spread_p = float(value)
        imgui.set_item_tooltip(
            "Each particle's momentum is drawn from a Gaussian of this relative width"
        )
        imgui.set_next_item_width(FIELD_WIDTH)
        changed, value = imgui.slider_float(
            "Angle spread [deg]",
            float(entry.spread_angle),
            0.0,
            custom.SPREAD_ANGLE_MAX,
            "%.1f",
        )
        if changed:
            entry.spread_angle = float(value)
        imgui.set_item_tooltip("Each particle's direction is turned by Gaussian angles of this width")

    def _face(self, entry, setup):
        global _partner
        if setup is None or len(setup.entries) < 2:
            imgui.text_disabled("Needs a second entry.")
            return
        others = [i for i in range(len(setup.entries)) if i != _selected]
        if _partner not in others:
            _partner = others[0]
        imgui.set_next_item_width(FIELD_WIDTH)
        k = others.index(_partner)
        changed, k = imgui.combo(
            "Partner", k, [f"{i + 1}. {setup.entries[i].name}" for i in others]
        )
        if changed:
            _partner = others[k]
        imgui.same_line()
        if imgui.button("Face each other##pair"):
            self.pair_message = face_each_other(entry, setup.entries[_partner])
        imgui.set_item_tooltip(
            "Aim both entries at each other's start point (by polar angle and azimuth).\n"
            "Entries sharing one start are first set 1 m apart on the beam axis"
        )
        if self.pair_message:
            imgui.text_disabled(self.pair_message)
        self._collide(entry, setup)

    def _collide(self, entry, setup):
        """The "Collide" checkbox for the selected entry and its partner, and
        what the collision did in the event shown."""
        other = setup.entries[_partner]
        paired = custom.collision_partner(setup, _selected) == _partner
        problem = custom.collision_problem(entry, other)
        imgui.push_text_wrap_pos(0.0)  # wrap at the window's edge
        if problem and not paired:
            imgui.text_disabled(FACE_CAPTION)
            imgui.text_disabled(f"Collide is not offered for this pair: {problem}.")
            mate = custom.collision_partner(setup, _selected)
            if mate is not None:
                imgui.text_disabled(f"(This entry is set to collide with entry {mate + 1}.)")
            imgui.pop_text_wrap_pos()
            return
        changed, on = imgui.checkbox("Collide##pair", paired)
        if changed:
            if on:
                custom.pair_up(setup, _selected, _partner)
            else:
                custom.unpair(setup, _selected)
            paired = on
        imgui.set_item_tooltip(
            "The two form a Z where they meet, if their sqrt(s) is near the Z mass"
        )
        if not paired:
            imgui.text_disabled(FACE_CAPTION)
            imgui.pop_text_wrap_pos()
            return
        imgui.text_disabled(COLLIDE_CAPTION)
        event = self.m.event
        records = [] if event is None or not event.custom else event.collisions
        pair = tuple(sorted((_selected, _partner)))
        names = tuple(setup.entries[k].name for k in pair)
        shown = [r for r in records if tuple(r["entries"]) == pair and tuple(r["names"]) == names]
        if shown and setup == self.m.custom:
            imgui.text(custom.collision_caption(shown[0]))
        else:
            root_s = custom.nominal_sqrt_s(entry, other)
            if root_s is not None:
                imgui.text_disabled(
                    f"sqrt(s) at the set momenta = {root_s:.4f} GeV (no event drawn yet)"
                )
        imgui.pop_text_wrap_pos()

    def _decay_controls(self, entry):
        labels, values = mode_labels(entry.name)
        if entry.typed:
            index = len(labels) - 1
        elif entry.daughters in values:
            index = values.index(entry.daughters)
        else:
            index = 0
        imgui.set_next_item_width(FIELD_WIDTH + 60.0)
        changed, index = imgui.combo("Decay mode", index, labels)
        if changed:
            entry.typed = values[index] is None
            if not entry.typed:
                entry.daughters = values[index]
        if entry.typed:
            imgui.set_next_item_width(FIELD_WIDTH + 60.0)
            changed, text = imgui.input_text("Daughters", entry.daughters)
            if changed:
                entry.daughters = text[:200]
            imgui.set_item_tooltip(
                "Names from the particle table, separated by spaces, for example:\n"
                "pi+ pi- pi0. Known: " + " ".join(PARTICLES)
            )
        if PARTICLES[entry.name].width > 0.0:
            imgui.text_disabled("Decays at once where it is put (a resonance)")
        elif custom.modes_of(entry.name):
            imgui.set_next_item_width(FIELD_WIDTH + 60.0)
            w = custom.WHERE.index(entry.where) if entry.where in custom.WHERE else 0
            changed, w = imgui.combo("It decays", w, list(WHERE_LABELS))
            if changed:
                entry.where = custom.WHERE[w]

    # -- the window

    def update(self):
        global _selected
        m = self.m
        io = imgui.get_io()
        W, H = io.display_size.x, io.display_size.y
        # Bottom left, growing upwards: the Simulation panel is top left, the
        # Event window top right. Taller than the room below the Simulation
        # panel: it scrolls.
        imgui.set_next_window_pos(
            (WINDOW_MARGIN, H - WINDOW_MARGIN), imgui.Cond_.always, (0.0, 1.0)
        )
        room = max(H - PANEL_ABOVE - WINDOW_MARGIN, 0.45 * H, 1.0)
        size = (min(WINDOW_SIZE[0], 0.45 * W), min(WINDOW_SIZE[1], room))
        imgui.set_next_window_size(size, imgui.Cond_.appearing)
        imgui.set_next_window_size_constraints((300.0, 200.0), (max(W * 0.6, 301.0), room))
        if not imgui.begin("Custom event", None, imgui.WindowFlags_.no_move)[0]:
            imgui.end()  # collapsed
            self._apply(draft())
            return

        setup = draft()
        self._presets()
        scenario_files.draw(
            "particles", m.scenario, m.load_scenario, note="Saves the setup that is running"
        )
        imgui.separator()
        checks = [custom.check(entry, m.field) for entry in setup.entries]
        self._entry_list(setup, checks)
        _selected = max(min(_selected, len(setup.entries) - 1), 0)
        # (The list may just have changed.)
        checks = [custom.check(entry, m.field) for entry in setup.entries]

        # What is wrong, or that all is well: straight under the list, so that
        # it stays in view when the window has to scroll.
        refused = custom.problems(setup, m.field)
        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
        if refused:
            imgui.text_colored(
                ERROR, "Not shown: the last valid setup keeps running until this is put right:"
            )
            for text in refused[:4]:
                imgui.text_colored(ERROR, f"Not possible: {text}")
            if len(refused) > 4:
                imgui.text_colored(ERROR, f"... and {len(refused) - 4} more")
        elif self.message:
            imgui.text_colored(ERROR, self.message)
        elif m.custom_dirty or m.custom_waiting:
            imgui.text_disabled("Making an event of this setup...")
        else:
            imgui.text_colored(
                GOOD, "This setup is running. Every new event redraws the random parts."
            )
        if setup.entries:
            for text in checks[_selected][1]:
                imgui.text_colored(WARNING, f"Note: {text}")
        imgui.pop_text_wrap_pos()

        if setup.entries:
            entry = setup.entries[_selected]
            imgui.separator_text(f"Entry {_selected + 1}")
            imgui.push_id(_selected)
            self._entry_controls(entry, setup)
            imgui.pop_id()
            lines = readout(entry, m.field)
            if lines:
                imgui.separator_text("Follows from these values")
                for line in lines:
                    imgui.text(line)
        if imgui.collapsing_header("About"):
            imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
            imgui.text_disabled(ABOUT_TEXT)
            imgui.pop_text_wrap_pos()
        imgui.end()
        self._apply(setup)

    def _apply(self, setup):
        """Hand a valid draft that differs from the running setup to the module."""
        m = self.m
        if setup == m.custom or custom.problems(setup, m.field):
            return
        try:
            m.apply_custom(setup)
            self.message = ""
        except Exception as error:  # noqa: BLE001 (shown in the window; must not end the app)
            self.message = f"Could not use this setup: {error}"

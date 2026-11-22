"""The Bodies window of the black-hole module: list the bodies, add new ones. UI only.

Added bodies are remembered per scene (module level, so they survive the
module being rebuilt): Restart starts them again from where they were added,
Full reset forgets them.
"""

from copy import deepcopy

import numpy as np
from imgui_bundle import imgui

from physengine.modules.blackhole_scenes import at_rest, orbiting

WINDOW_MARGIN = 10.0
FIELD_WIDTH = 150.0

# ------------------------------------------------------------ added bodies

_added = None  # (scene name, [BodySpec])


def added_bodies(scene_name):
    if _added is not None and _added[0] == scene_name:
        return deepcopy(_added[1])
    return []


def remember(scene_name, spec):
    global _added
    if _added is None or _added[0] != scene_name:
        _added = (scene_name, [])
    _added[1].append(deepcopy(spec))


def forget_added():
    global _added
    _added = None


# ----------------------------------------------------------------- window

settings = {"r_p": 8.0, "r_a": 20.0, "inclination": 30.0, "temperature": 7000.0}
DEFAULT_SETTINGS = dict(settings)


def reset_settings():
    settings.clear()
    settings.update(DEFAULT_SETTINGS)


class BodiesWindow:
    def __init__(self, module):
        self.m = module
        self.message = ""

    def update(self):
        m = self.m
        io = imgui.get_io()
        W, H = io.display_size.x, io.display_size.y
        imgui.set_next_window_pos(
            (W - WINDOW_MARGIN, WINDOW_MARGIN), imgui.Cond_.always, (1.0, 0.0)
        )
        imgui.set_next_window_size_constraints((0.0, 0.0), (max(W * 0.45, 1.0), max(H - 20.0, 1.0)))
        flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move
        if not imgui.begin("Bodies", None, flags)[0]:
            imgui.end()
            return

        bodies = m.bodies
        r = bodies.r()
        if bodies.n == 0:
            imgui.text_disabled("No bodies")
        if bodies.n > 20:  # a star of particles: a summary, not a list
            imgui.text(f"{bodies.n} particles, {int((~bodies.alive).sum())} fallen in")
            if m.options.info:
                imgui.text_disabled(m.options.info)
        for i in range(bodies.n if bodies.n <= 20 else 0):
            if bodies.alive[i]:
                imgui.text(f"{m.names[i]}: r = {r[i]:.2f} M")
            else:
                imgui.text_disabled(f"{m.names[i]}: fell in at t = {bodies.captured_at[i]:.0f} M")

        if imgui.collapsing_header("Add a body"):
            for key, label, lo, hi, fmt in (
                ("r_p", "Pericentre", 4.0, 60.0, "%.2f M"),
                ("r_a", "Apocentre", 4.0, 150.0, "%.2f M"),
                ("inclination", "Inclination", -90.0, 90.0, "%.0f deg"),
                ("temperature", "Temperature", 2500.0, 30000.0, "%.0f K"),
            ):
                imgui.set_next_item_width(FIELD_WIDTH)
                _, settings[key] = imgui.slider_float(label, settings[key], lo, hi, fmt)
            imgui.set_item_tooltip("Colour and brightness of the body")
            if imgui.button("Add orbit"):
                self._add("orbit")
            imgui.set_item_tooltip(
                "Bound orbit between pericentre and apocentre,\nstarting at the apocentre"
            )
            imgui.same_line()
            if imgui.button("Drop from rest"):
                self._add("drop")
            imgui.set_item_tooltip("Let go at rest at the apocentre distance: falls straight in")
            if self.message:
                imgui.text_colored((1.0, 0.6, 0.3, 1.0), self.message)
            imgui.text_disabled(
                "Stable circular orbits need r > 6 M (ISCO);\nbelow it bodies spiral in."
            )
        imgui.end()

    def _add(self, kind):
        m = self.m
        r_p, r_a = sorted((settings["r_p"], settings["r_a"]))
        name = f"Body {m.bodies.n + 1}"
        try:
            if kind == "orbit":
                spec = orbiting(
                    name,
                    r_p,
                    r_a,
                    inclination_deg=settings["inclination"],
                    temperature=settings["temperature"],
                )
            else:
                incl = np.radians(settings["inclination"])
                spec = at_rest(
                    name,
                    r_a,
                    direction=(np.cos(incl), np.sin(incl), 0.0),
                    temperature=settings["temperature"],
                )
        except ValueError as error:
            self.message = str(error)
            return
        self.message = ""
        m.add_body(spec)
        remember(m.scene_name, spec)

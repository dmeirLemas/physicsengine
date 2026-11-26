"""The panels of the atom module: each scene's controls, numbers and plots. UI only.

Written by Claude (2026-11-23, atom module). No physics is computed here
beyond calling atom_orbitals, atom_decay, atom_reactions and atom_data for
the numbers shown.
"""

import numpy as np
from imgui_bundle import imgui

from physengine.modules import atom_decay, atom_nucleus, atom_orbitals, atom_reactions
from physengine.modules import atom_scenes as scenes
from physengine.modules.atom_data import (
    NAMES,
    SYMBOLS,
    format_energy,
    format_time,
    label,
    semf_binding,
    semf_terms,
)

WRAP = 390.0  # pixels: width of the panel's text and plots
GREY = (0.6, 0.63, 0.7, 1.0)
GOLD = (1.0, 0.8, 0.3, 1.0)
WARN = (1.0, 0.6, 0.35, 1.0)


def wrapped(text, color=None):
    imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
    if color is None:
        imgui.text_wrapped(text)
    else:
        imgui.push_style_color(imgui.Col_.text, color)
        imgui.text_wrapped(text)
        imgui.pop_style_color()
    imgui.pop_text_wrap_pos()


def note(text):
    imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
    imgui.text_disabled(text)
    imgui.pop_text_wrap_pos()


def swatch(color, text):
    imgui.color_button(f"##{text}", (*color, 1.0), imgui.ColorEditFlags_.no_tooltip, (12, 12))
    imgui.same_line()
    imgui.text(text)


def u32(color, alpha=1.0):
    return imgui.get_color_u32((float(color[0]), float(color[1]), float(color[2]), alpha))


class Plot:
    """A small line plot drawn with ImGui's draw list."""

    def __init__(self, height, x_range, y_range, xlog=False, ylog=False, width=WRAP):
        self.xlog, self.ylog = xlog, ylog
        self.x0, self.x1 = self._t(np.array(x_range, dtype=float), xlog)
        self.y0, self.y1 = self._t(np.array(y_range, dtype=float), ylog)
        if self.x1 <= self.x0:
            self.x1 = self.x0 + 1.0
        if self.y1 <= self.y0:
            self.y1 = self.y0 + 1.0
        imgui.dummy((width, height))
        low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.left, self.top, self.right, self.bottom = low.x, low.y, high.x, high.y
        self.hovered = imgui.is_item_hovered()
        self.draw = imgui.get_window_draw_list()
        self.draw.add_rect_filled((low.x, low.y), (high.x, high.y), u32((0.03, 0.04, 0.07), 0.9))
        self.draw.add_rect((low.x, low.y), (high.x, high.y), u32((0.3, 0.33, 0.4)))

    @staticmethod
    def _t(values, log):
        return np.log10(np.maximum(values, 1e-300)) if log else values

    def px(self, x, y):
        x = (self._t(np.asarray(x, dtype=float), self.xlog) - self.x0) / (self.x1 - self.x0)
        y = (self._t(np.asarray(y, dtype=float), self.ylog) - self.y0) / (self.y1 - self.y0)
        return (
            self.left + x * (self.right - self.left),
            self.bottom - y * (self.bottom - self.top),
        )

    def line(self, xs, ys, color, thickness=1.5, alpha=1.0):
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if self.ylog:
            ok &= ys > 0.0
        if self.xlog:
            ok &= xs > 0.0
        if ok.sum() < 2:
            return
        px, py = self.px(xs[ok], ys[ok])
        py = np.clip(py, self.top, self.bottom)
        px = np.clip(px, self.left, self.right)
        points = [imgui.ImVec2(float(a), float(b)) for a, b in zip(px, py, strict=True)]
        self.draw.add_polyline(points, u32(color, alpha), thickness, 0)

    def bars(self, edges, heights, color, alpha=0.6):
        for a, b, h in zip(edges[:-1], edges[1:], heights, strict=True):
            if h <= 0.0:
                continue
            (xa, ya), (xb, yb) = self.px(a, h), self.px(b, self.y_data(0.0))
            self.draw.add_rect_filled(
                (float(xa), float(max(ya, self.top))), (float(xb), float(yb)), u32(color, alpha)
            )

    def y_data(self, fraction):
        value = self.y0 + fraction * (self.y1 - self.y0)
        return 10.0**value if self.ylog else value

    def vline(self, x, color, alpha=0.9):
        px, _ = self.px(x, self.y_data(0.0))
        if self.left <= px <= self.right:
            self.draw.add_line(
                (float(px), self.top), (float(px), self.bottom), u32(color, alpha), 1.0
            )

    def marker(self, x, y, color, radius=4.0):
        px, py = self.px(x, y)
        if self.left <= px <= self.right and self.top - 1 <= py <= self.bottom + 1:
            self.draw.add_circle_filled((float(px), float(py)), radius, u32(color))
            self.draw.add_circle((float(px), float(py)), radius + 1.5, u32((1.0, 1.0, 1.0)), 0, 1.0)

    def text(self, x, y, text, color=GREY, dx=3.0, dy=-14.0):
        px, py = self.px(x, y)
        self.draw.add_text((float(px) + dx, float(py) + dy), u32(color), text)

    def corner_text(self, text, color=GREY, right=False, bottom=False):
        size = imgui.calc_text_size(text)
        x = self.right - size.x - 4.0 if right else self.left + 4.0
        y = self.bottom - size.y - 2.0 if bottom else self.top + 2.0
        self.draw.add_text((x, y), u32(color), text)


def axis_labels(left, right, middle=""):
    imgui.text_disabled(left)
    if middle:
        imgui.same_line(WRAP * 0.5 - imgui.calc_text_size(middle).x * 0.5)
        imgui.text_disabled(middle)
    imgui.same_line(WRAP - imgui.calc_text_size(right).x)
    imgui.text_disabled(right)


def nuclide_chooser(key, presets, current, state):
    """A combo of presets and a text box for any other nuclide. Returns the
    name chosen this frame, or None."""
    chosen = None
    items = list(presets)
    index = items.index(current) if current in items else -1
    imgui.set_next_item_width(130.0)
    changed, index = imgui.combo(f"##preset{key}", index, items)
    if changed and index >= 0:
        chosen = items[index]
    imgui.same_line()
    imgui.set_next_item_width(90.0)
    _, state[key] = imgui.input_text(f"##own{key}", state.get(key, ""))
    imgui.same_line()
    if imgui.button(f"Show##{key}") and state.get(key, "").strip():
        chosen = state[key].strip()
    imgui.set_item_tooltip("Any nuclide, written like U-238, Pa-234m, Co-60 or n")
    return chosen


# ===================================================================== orbitals


def draw_orbital(scene, state):
    s = scene.s
    changed_z, z = imgui.slider_int("Nuclear charge Z", s["Z"], 1, 10)
    if changed_z:
        s["Z"] = z
        scene.rebuild()
        scene.say(scene.describe())
    imgui.same_line()
    imgui.text_disabled(scene.species())
    n, l, m = s["n"], s["l"], s["m"]  # noqa: E741
    c1, n2 = imgui.slider_int("n (shell)", n, 1, atom_orbitals.N_MAX)
    c2, l2 = imgui.slider_int("l (shape)", l, 0, max(n2 - 1, 0))
    l2 = min(l2, n2 - 1)
    c3, m2 = imgui.slider_int("m (orientation)", m, -l2, l2)
    m2 = int(np.clip(m2, -l2, l2))
    if c1 or c2 or c3:
        scene.set_orbital(n2, l2, m2)
    changed, real = imgui.checkbox("Real orbitals (p_x, d_xy ...)", s["real"])
    if changed:
        s["real"] = real
        scene.rebuild()
        scene.say(scene.describe())
    name = atom_orbitals.orbital_name(s["n"], s["l"], s["m"], s["real"])
    imgui.text_colored(GOLD, f"{name}    E = {scene.energy():.4f} eV")
    radius = atom_orbitals.mean_radius(s["n"], s["l"], s["Z"])
    imgui.text(f"mean radius {radius:.4g} a0 = {radius * 0.0529177:.4g} nm")
    imgui.text(f"nodes: {s['n'] - s['l'] - 1} radial, {s['l']} angular")
    if s["real"]:
        swatch(scenes.POSITIVE, "psi > 0")
        imgui.same_line()
        swatch(scenes.NEGATIVE, "psi < 0")
    else:
        note("Colour: the phase of psi, which goes round m times with the angle about the axis.")

    imgui.separator()
    imgui.text("Energy levels: click one to jump there")
    _level_diagram(scene)
    line = scene.last_line
    if line is not None:
        air = f", {line['air']:.5g} nm in air" if line["air"] else ""
        verb = "emitted" if line["emitted"] else "absorbed"
        wrapped(
            f"Last photon ({verb}): n = {line['upper']} <-> {line['lower']},"
            f" {line['energy']:.5g} eV, {line['vacuum']:.6g} nm in vacuum{air} ({line['band']}).",
            (*scene.photon["rgb"], 1.0) if scene.photon else None,
        )
    note(
        "Levels drawn equally spaced (energies in eV at the left). Green: allowed"
        " electric-dipole transitions, l changing by 1. Wavelengths from the Rydberg formula"
        " with the reduced mass; the fine structure (0.01 nm in H-alpha) is not included."
    )


def _level_diagram(scene):
    s = scene.s
    n_max = atom_orbitals.N_MAX
    height, width = 178.0, WRAP
    imgui.dummy((width, height))
    low = imgui.get_item_rect_min()
    draw = imgui.get_window_draw_list()
    draw.add_rect_filled(
        (low.x, low.y), (low.x + width, low.y + height), u32((0.03, 0.04, 0.07), 0.9)
    )
    left = low.x + 96.0
    column = (width - 102.0) / n_max
    mouse = imgui.get_mouse_pos()
    clicked = imgui.is_item_hovered() and imgui.is_mouse_clicked(0)
    hovered = None

    def level_y(n):
        # Equally spaced, not to scale: the true spacing goes as 1/n^2 and
        # would crowd every level above n = 3 into a few pixels.
        return low.y + height - 30.0 - (height - 44.0) * (n - 1) / (n_max - 1)

    for n in range(1, n_max + 1):
        y = level_y(n)
        draw.add_text((low.x + 3.0, y - 7.0), u32(GREY), f"n={n} {scene.energy(n):9.3f}")
        for l in range(n):  # noqa: E741
            x0, x1 = left + l * column + 3.0, left + (l + 1) * column - 3.0
            current = (n, l) == (s["n"], s["l"])
            allowed = abs(l - s["l"]) == 1 and n != s["n"]
            color = GOLD if current else (0.35, 0.9, 0.5) if allowed else (0.38, 0.42, 0.5)
            over = x0 - 2 <= mouse.x <= x1 + 2 and abs(mouse.y - y) <= 4.5
            draw.add_line((x0, y), (x1, y), u32(color), 4.0 if (current or over) else 2.0)
            if over:
                hovered = (n, l)
    for l in range(n_max):  # noqa: E741
        draw.add_text(
            (left + (l + 0.5) * column - 3.0, low.y + height - 16.0),
            u32(GREY),
            atom_orbitals.L_LETTERS[l],
        )
    draw.add_text((low.x + 3.0, low.y + height - 16.0), u32(GREY), "eV")
    if hovered is not None and hovered != (s["n"], s["l"]):
        n2, l2 = hovered
        if n2 != s["n"]:
            upper, lower = max(n2, s["n"]), min(n2, s["n"])
            photon, vacuum = atom_orbitals.transition(upper, lower, s["Z"], scene.nuclear_mass)
            rule = "allowed" if abs(l2 - s["l"]) == 1 else "forbidden: l must change by 1"
            imgui.set_tooltip(
                f"to {n2}{atom_orbitals.L_LETTERS[l2]}: {photon:.4g} eV, {vacuum:.5g} nm ({rule})"
            )
        if clicked:
            scene.jump(n2, l2)


def draw_atom(scene, state):
    s = scene.s
    changed, z = imgui.slider_int("Element Z", s["Z"], 1, atom_orbitals.Z_MAX_CONFIGURATION)
    if changed:
        s["Z"] = z
        scene.rebuild()
    Z = s["Z"]
    imgui.text_colored(GOLD, f"{NAMES[Z].capitalize()}  ({SYMBOLS[Z]}, Z = {Z})")
    imgui.text(atom_orbitals.configuration_text(Z))
    if Z in atom_orbitals.EXCEPTIONS:
        note("One of the exceptions to the simple filling order.")
    changed, magnify = imgui.checkbox("Magnify the nucleus (display only)", s["magnify"])
    if changed:
        s["magnify"] = magnify
    if imgui.button("Zoom to the nucleus"):
        state["zoom_to"] = 5.0 * scene.true_radius * scene.magnification
    imgui.same_line()
    if imgui.button("Zoom to the atom"):
        state["zoom_to"] = scene.view[0]
    ratio = scene.extent / 1.9 / scene.true_radius
    note(
        f"True sizes: nucleus {atom_nucleus.radius(scene.A):.2f} fm, outer shell about"
        f" {scene.extent / 1.9 * 52917.7:,.0f} fm: {ratio:,.0f} times larger. Unmagnified, the"
        " nucleus is far below one pixel until you zoom in a long way (mouse wheel or button)."
    )
    flags = imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.sizing_fixed_fit
    if imgui.begin_table("shells", 5, flags):
        for head in ("shell", "electrons", "Z_eff", "energy [eV]", "radius [a0]"):
            imgui.table_setup_column(head)
        imgui.table_headers_row()
        for row in scene.rows:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_colored((*scenes.SHELL_COLORS[row["n"] - 1], 1.0), row["name"])
            imgui.table_next_column()
            imgui.text(f"{row['electrons']}")
            imgui.table_next_column()
            imgui.text(f"{row['z_eff']:.2f}")
            imgui.table_next_column()
            imgui.text(f"{row['energy']:.4g}")
            imgui.table_next_column()
            imgui.text(f"{row['radius']:.3g}")
        imgui.end_table()
    note(
        "Z_eff and energies by Slater's rules; radii of hydrogen-like orbitals with that"
        " charge. Rough values: an illustration of shells, not a Hartree-Fock calculation."
        " 'Thin slice' under Cloud view shows the shells as rings."
    )


# ====================================================================== nucleus

NUCLEUS_PRESETS = (
    "H-2",
    "He-4",
    "C-12",
    "O-16",
    "Fe-56",
    "Ni-62",
    "Sn-120",
    "Pb-208",
    "U-235",
    "U-238",
)


def draw_nucleus(scene, state):
    s = scene.s
    table = scene.table
    c1, z = imgui.slider_int("Protons Z", s["Z"], 1, len(SYMBOLS) - 1)
    c2, n = imgui.slider_int("Neutrons N", s["A"] - s["Z"], 0, 180)
    if c1 or c2:
        s["Z"], s["A"] = z, z + n
        scene.rebuild()
    for k, name in enumerate(NUCLEUS_PRESETS):
        if k % 5:
            imgui.same_line()
        if imgui.small_button(name):
            s["Z"], s["A"], _ = scenes.parse_label(name)
            scene.rebuild()
    Z, A = s["Z"], s["A"]
    imgui.text_colored(GOLD, f"{label(Z, A)}  ({NAMES[Z]}):  {Z} protons + {A - Z} neutrons")
    binding = table.binding(Z, A)
    imgui.text(f"Radius 1.2 fm x A^(1/3) = {scene.R:.3f} fm")
    imgui.text(f"Binding energy {binding:.3f} MeV = {binding / A:.4f} MeV per nucleon")
    note(f"from: {table.source(Z, A)}")
    mass = table.mass(Z, A)
    imgui.text(f"Atomic mass {mass / 931.49410242:.6f} u = {mass:.3f} MeV")
    nuclide = table.get(Z, A)
    if nuclide is None:
        wrapped("Not in the tables: probably cannot exist even briefly.", WARN)
    elif nuclide.stable:
        extra = (
            f", {nuclide.abundance * 100:.4g}% of natural {NAMES[Z]}" if nuclide.abundance else ""
        )
        imgui.text(f"Stable{extra}")
    else:
        imgui.text(f"Half-life {format_time(nuclide.half_life)}")
        for b in atom_decay.branches(table, Z, A)[:4]:
            q = "" if np.isnan(b.q) else f", {format_energy(b.q)}"
            imgui.bullet_text(
                f"{atom_decay.MODE_TEXT[b.mode]} {b.fraction * 100:.3g}% -> {b.daughter_name}{q}"
            )

    if imgui.collapsing_header("Semi-empirical mass formula"):
        terms = semf_terms(Z, A)
        total = float(semf_binding(Z, A))
        for name, value in terms.items():
            imgui.text(f"  {name:10s} {float(value):+10.2f} MeV")
        imgui.text(f"  {'sum':10s} {total:+10.2f} MeV   (table {binding:.2f})")
        note(
            "Volume: each nucleon binds to its neighbours. Surface: those at the surface have"
            " fewer. Coulomb: protons repel. Asymmetry: unequal numbers of protons and neutrons"
            " cost energy. Pairing: even numbers bind better. Coefficients fitted to AME2020"
            " (3 MeV rms); no shell effects."
        )

    imgui.separator()
    imgui.text("Binding energy per nucleon [MeV]")
    if "curve" not in state:
        state["curve"] = atom_nucleus.binding_curve(table)
    masses, measured, formula = state["curve"]
    plot = Plot(150.0, (0.0, 270.0), (0.0, 9.2))
    plot.line(masses, formula, (0.45, 0.6, 1.0), 1.2, 0.8)
    plot.line(masses[np.isfinite(measured)], measured[np.isfinite(measured)], (1.0, 0.75, 0.3), 1.6)
    plot.marker(A, binding / A, (0.3, 1.0, 0.6))
    plot.text(A, binding / A, label(Z, A), (0.6, 1.0, 0.75), dx=6.0, dy=2.0)
    plot.corner_text("9", GREY)
    plot.corner_text("0", GREY, bottom=True)
    axis_labels("A = 0", "270", "mass number A")
    swatch((1.0, 0.75, 0.3), "measured (most bound nuclide of each A)")
    swatch((0.45, 0.6, 1.0), "semi-empirical formula")
    note(
        "The peak near iron and nickel is why fusing light nuclei and splitting heavy ones"
        " both release energy."
    )


# ======================================================================== decay


def draw_decay(scene, state):
    s = scene.s
    chosen = nuclide_chooser("decay", scenes.DECAY_PRESETS, s["nuclide"], state)
    if chosen:
        scene.set_nuclide(chosen)
    _, s["continuous"] = imgui.checkbox("One after another", s["continuous"])
    imgui.same_line()
    if imgui.button("Next decay"):
        scene.new_event()
    nuclide = scene.nuclide
    title = s["nuclide"]
    if nuclide is not None and nuclide.half_life is not None:
        title += f"   half-life {format_time(nuclide.half_life)}"
    imgui.text_colored(GOLD, title)
    flags = imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.sizing_fixed_fit
    if scene.options and imgui.begin_table("branches", 4, flags):
        for head in ("decay", "share", "to", "energy"):
            imgui.table_setup_column(head)
        imgui.table_headers_row()
        for b in scene.options[:6]:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(atom_decay.MODE_TEXT[b.mode].split(" (")[0])
            imgui.table_next_column()
            imgui.text(f"{b.fraction * 100:.4g}%")
            imgui.table_next_column()
            imgui.text(b.daughter_name)
            imgui.table_next_column()
            imgui.text("~200 MeV" if np.isnan(b.q) else format_energy(b.q))
        imgui.end_table()
    beta = [b for b in scene.options if b.mode in ("beta-", "beta+")]
    if beta:
        b = beta[0]
        positron = b.mode == "beta+"
        what = "positron" if positron else "electron"
        imgui.separator()
        imgui.text(f"Energy of the {what} in each decay so far")
        data = np.array(scene.energies.get(b.mode, []))
        edges = np.linspace(0.0, b.q, 31)
        T = np.linspace(0.0, b.q, 200)
        theory = atom_decay.beta_spectrum(T, b.q, b.daughter[0], positron)
        theory = theory / max(theory.max(), 1e-300)
        counts, _ = np.histogram(data, edges)
        # Scale the histogram so that both have the same area.
        area = np.sum(theory) * (T[1] - T[0])
        scale = area / max(counts.sum() * (edges[1] - edges[0]), 1e-300)
        plot = Plot(110.0, (0.0, b.q), (0.0, 1.25))
        plot.bars(edges, counts * scale, (0.3, 0.8, 0.65))
        plot.line(T, theory, (1.0, 0.8, 0.3), 1.6)
        axis_labels("0", format_energy(b.q), "kinetic energy")
        mean = atom_decay.beta_mean_energy(b.q, b.daughter[0], positron)
        seen = f"; mean of these {format_energy(float(data.mean()))}" if len(data) else ""
        imgui.text(f"{len(data)} decays{seen}")
        note(
            f"Line: the allowed spectrum (mean {format_energy(mean)}). The rest of the"
            f" {format_energy(b.q)} goes to the {'neutrino' if positron else 'antineutrino'}."
        )
        if imgui.button("Add 2000 decays"):
            scene.fill(2000)
        if scene.to_fill:
            imgui.same_line()
            imgui.text_disabled(f"{scene.to_fill} to go")
    elif scene.options and scene.options[0].mode == "alpha":
        note(
            "Alpha decay has two products, so the alpha particle always gets the same"
            " energy: Q times (daughter mass) / (total mass)."
        )


# ================================================================ fission, fusion


def _energy_bar(text, value, total, color):
    imgui.text(f"{text:22s}{value:7.1f} MeV")
    imgui.same_line(250.0)
    low = imgui.get_cursor_screen_pos()
    width = 130.0 * max(value, 0.0) / max(total, 1e-9)
    draw = imgui.get_window_draw_list()
    draw.add_rect_filled((low.x, low.y + 3.0), (low.x + width, low.y + 13.0), u32(color))
    imgui.new_line()


def draw_fission(scene, state):
    s = scene.s
    systems = list(atom_reactions.FISSILE)
    names = ["U-235 + slow neutron", "Pu-239 + slow neutron", "Cf-252 (spontaneous)"]
    imgui.set_next_item_width(190.0)
    changed, index = imgui.combo("##system", systems.index(s["system"]), names)
    if changed:
        scene.set_system(systems[index])
    _, s["continuous"] = imgui.checkbox("One after another", s["continuous"])
    imgui.same_line()
    if imgui.button("Next fission"):
        scene.new_event()
    event = scene.animation.event
    (zl, al, _, _), (zh, ah, _, _) = event.fragments
    nu = len(event.neutrons)
    imgui.text_colored(GOLD, f"{label(zl, al)} + {label(zh, ah)} + {nu} n")
    total = event.q + event.later
    _energy_bar("fragments (motion)", event.fragment_energy, total, (0.95, 0.45, 0.3))
    _energy_bar("prompt neutrons", event.neutron_energy, total, scenes.NEUTRON)
    _energy_bar("prompt gamma rays", event.gamma_energy, total, scenes.PHOTON)
    _energy_bar("later: beta decays", event.later, total, scenes.ELECTRON)
    imgui.text(f"{'total':22s}{total:7.1f} MeV")
    if event.estimated_masses:
        note("A fragment's mass here is an estimate (AME systematics or the mass formula).")
    note(
        "The fragments' motion is the Coulomb repulsion of two charged pieces a few fm apart."
        " Of the later energy, roughly a third leaves with antineutrinos."
    )
    if scene.totals:
        data = np.array(scene.totals)
        imgui.separator()
        imgui.text(f"Mean of {len(data)} fissions:")
        imgui.text(
            f"  {data[:, 0].mean() + data[:, 4].mean():.1f} MeV in all;"
            f" fragments {data[:, 1].mean():.1f};  {data[:, 5].mean():.2f} neutrons"
        )
    imgui.text("Fragment masses (before neutron emission)")
    system = atom_reactions.FISSILE[s["system"]]
    masses, yields = atom_reactions.mass_yield(system, np.arange(66, 176))
    top = yields.max() * 1.25
    plot = Plot(110.0, (66.0, 175.0), (0.0, top))
    both = np.array(scene.light_masses + scene.heavy_masses)
    if len(both):
        counts = np.bincount(both.astype(int), minlength=177)[66:176]
        plot.bars(
            np.arange(66, 177) - 0.5, counts / max(len(scene.light_masses), 1), (0.4, 0.6, 0.9)
        )
    plot.line(masses, yields, (1.0, 0.8, 0.3), 1.5)
    for a in event.primary_masses:
        plot.vline(a, (0.3, 1.0, 0.6))
    plot.corner_text(f"{top * 100:.1f}%", GREY)
    axis_labels("A = 66", "175", "mass number")
    note(
        "Line: the yield parametrisation (two humps: equal halves are rare). Bars: the"
        " fissions so far; green: this one."
    )
    if imgui.button("Add 2000 fissions"):
        scene.fill(2000)
    if scene.to_fill:
        imgui.same_line()
        imgui.text_disabled(f"{scene.to_fill} to go")


def draw_chain_reaction(scene, state):
    s = scene.s
    sim = scene.sim
    c1, s["R"] = imgui.slider_float("radius [mean free paths]", s["R"], 0.5, 4.0, "%.2f")
    c2, s["p"] = imgui.slider_float("fission probability p", s["p"], 0.1, 1.0, "%.2f")
    c3, s["start"] = imgui.slider_int("neutrons at the start", int(s["start"]), 1, 200)
    if c1 or c2 or c3 or imgui.button("Start again"):
        scene.restart()
        sim = scene.sim
    imgui.same_line()
    if imgui.button("Add 20 neutrons"):
        scene.inject(20)
    k_inf = sim.k_infinity
    uniform = k_inf * (1.0 - float(atom_reactions.escape_probability(s["R"])))
    measured = sim.measured_k()
    imgui.text(f"k without leaks: {sim.nu_bar:.2f} x {s['p']:.2f} = {k_inf:.2f}")
    imgui.text(f"k estimate with leaks (uniform source): {uniform:.2f}")
    if measured is None:
        imgui.text("k measured: (waiting for 20 neutrons to end)")
    else:
        state_text = (
            "supercritical: grows"
            if measured > 1.03
            else ("subcritical: dies out" if measured < 0.97 else "about critical")
        )
        imgui.text_colored(GOLD, f"k measured: {measured:.3f}   ({state_text})")
    imgui.text(f"neutrons in flight: {sim.count}    time {sim.time:.1f}")
    imgui.text(f"fissions {sim.fissions}   captured {sim.captures}   escaped {sim.leaks}")
    if sim.saturated:
        wrapped("Stopped: more neutrons than the scene follows. 'Start again'.", WARN)
    history = np.array(sim.history)
    top = max(float(history[:, 1].max()), 10.0)
    span = max(float(history[-1, 0]), 5.0)
    plot = Plot(120.0, (0.0, span), (0.7, top * 1.5), ylog=True)
    plot.line(history[:, 0], history[:, 1], (0.55, 0.85, 1.0), 2.0)
    plot.corner_text(f"{top:.0f}", GREY)
    plot.corner_text("1", GREY, bottom=True)
    axis_labels("0", f"{span:.0f}", "time [mean free flight times]; neutrons, log scale")
    note(
        "k is the number of neutrons that follow each neutron. A straight rising line on"
        " this scale is exponential growth. Change the radius: a small sphere leaks too many"
        " neutrons through its surface to sustain a chain."
    )


def draw_fusion(scene, state):
    s = scene.s
    table = scene.table
    choices = list(scenes.FUSION_CHOICES)
    imgui.set_next_item_width(270.0)
    changed, index = imgui.combo("##reaction", choices.index(s["reaction"]), choices)
    if changed:
        scene.set_choice(choices[index])
    _, s["continuous"] = imgui.checkbox("One after another", s["continuous"])
    imgui.same_line()
    if imgui.button("Next"):
        scene.new_event()
    reaction = scene.reaction
    event = scene.event
    imgui.text_colored(GOLD, f"{reaction.name}    Q = {format_energy(event.q)}")
    wrapped(reaction.note)
    for name, T in event.kinetic_energies():
        imgui.bullet_text(f"{name}: {format_energy(T)}")
    if len(event.products) == 2:
        note("Two products: the lighter one takes the larger share, in the ratio of the masses.")
    if scene.is_chain:
        summary = atom_reactions.pp_chain_summary(table)
        imgui.separator()
        imgui.text("The chain as a whole: 4 p -> He-4 + 2 e+ + 2 nu")
        for k, (name, times, q, extra) in enumerate(summary["steps"]):
            marker = ">" if k == scene.chain_step % 3 else " "
            more = f" + {extra:.3f}" if extra else ""
            imgui.text(f"{marker} {times} x  {name}:  {q:.3f}{more} MeV")
        imgui.text(f"  total {summary['total']:.3f} MeV, of which the neutrinos")
        imgui.text(f"  carry off {summary['neutrinos']:.2f} MeV on average")
        note(
            "0.7% of the four protons' mass becomes energy. Steps 1 and 2 must each happen"
            " twice to supply the two He-3 of step 3."
        )

    imgui.separator()
    imgui.text("Getting through the Coulomb barrier")
    _, s["log_temperature"] = imgui.slider_float(
        "log10 temperature [K]", s["log_temperature"], 6.0, 10.0, "%.2f"
    )
    T = scene.temperature
    kT = atom_reactions.thermal_energy_kev(T)
    (z1, a1), (z2, a2) = reaction.reactants
    e_gamow = (
        atom_reactions.gamow_energy(z1, z2, atom_reactions.reduced_mass(table, reaction)) * 1e3
    )
    height = atom_reactions.barrier(reaction) * 1e3
    e0, width = atom_reactions.gamow_peak(e_gamow, kT)
    imgui.text(f"T = {T:.3g} K: typical energy kT = {kT:.3g} keV")
    imgui.text(f"barrier height {height:.0f} keV = {height / kT:.3g} kT")
    imgui.text(
        f"tunnelling probability at kT:      {float(atom_reactions.tunnelling(kT, e_gamow)):.2e}"
    )
    imgui.text(
        f"at the Gamow peak ({e0:.3g} keV): {float(atom_reactions.tunnelling(e0, e_gamow)):.2e}"
    )
    span = min(max(e0 + 2.0 * width, 6.0 * kT), 20.0 * e0)
    energies = np.linspace(span / 400.0, span, 300)
    window = atom_reactions.gamow_window(energies, e_gamow, kT)
    plot = Plot(120.0, (0.0, span), (0.0, 1.12))
    plot.line(energies, np.exp(-energies / kT), (0.4, 0.65, 1.0), 1.3)
    tunnel = atom_reactions.tunnelling(energies, e_gamow)
    plot.line(energies, tunnel / max(tunnel.max(), 1e-300), (1.0, 0.5, 0.35), 1.3)
    plot.line(energies, window / max(window.max(), 1e-300), (1.0, 0.85, 0.3), 2.2)
    plot.vline(e0, (1.0, 0.85, 0.3), 0.6)
    axis_labels("0", f"{span:.3g} keV", "energy of the pair")
    swatch((0.4, 0.65, 1.0), "pairs with this energy: exp(-E / kT)")
    swatch((1.0, 0.5, 0.35), "tunnelling: exp(-sqrt(E_G / E))")
    swatch((1.0, 0.85, 0.3), "product: the Gamow peak")
    note(
        f"Each curve is scaled to its own maximum. E_G = {e_gamow:.4g} keV. Fusion happens in"
        " the narrow window where there are still pairs and tunnelling is no longer hopeless."
        " How often it happens also needs the nuclear cross-section (S-factor), which is not"
        " included here."
    )


PANELS = {
    "Hydrogen-like orbital": draw_orbital,
    "Atom: shells and nucleus": draw_atom,
    "Nucleus and binding energy": draw_nucleus,
    "Decay of one nucleus": draw_decay,
    "Fission of one nucleus": draw_fission,
    "Chain reaction": draw_chain_reaction,
    "Fusion": draw_fusion,
}

LEGENDS = {
    "Hydrogen-like orbital": (
        ((1.0, 0.95, 0.85), "nucleus (a marker: far too small to see)"),
        (scenes.POSITIVE, "cloud: where the electron can be found"),
    ),
    "Atom: shells and nucleus": tuple(
        (color, f"shell n = {n + 1} ({'KLMNOPQ'[n]})")
        for n, color in enumerate(scenes.SHELL_COLORS[:6])
    )
    + ((scenes.PROTON, "proton"), (scenes.NEUTRON, "neutron")),
    "nuclear": (
        (scenes.PROTON, "proton"),
        (scenes.NEUTRON, "neutron"),
        (scenes.ELECTRON, "electron"),
        (scenes.POSITRON, "positron"),
        (scenes.NEUTRINO, "neutrino or antineutrino"),
        (scenes.PHOTON, "gamma ray (wavy line)"),
    ),
    "Chain reaction": (
        ((0.55, 0.85, 1.0), "neutron"),
        ((1.0, 0.75, 0.3), "fission"),
        ((0.30, 0.36, 0.5), "the material (dots only mark it)"),
    ),
}

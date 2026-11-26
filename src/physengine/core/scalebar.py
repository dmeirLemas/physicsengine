"""A map-style scale bar drawn over the 3D view with ImGui.

The bar is exact only at the distance of the point the camera looks at (the
orbit target): that is where one pixel is height / (2 d tan(fov / 2)) world
units. Closer things are larger, farther ones smaller. Display only.

The pure helpers (pixels_per_unit, nice_length, length_text, label_text) need
no ImGui context; draw_scale_bar is the only function that draws.
"""

import math

from imgui_bundle import imgui

AU_M = 1.495978707e11
LIGHT_YEAR_M = 9.4607304725808e15
KPC_M = 3.0856775814913673e19

MAX_BAR_PIXELS = 160.0
MARGIN = 20.0
TICK = 6.0

# (smallest length in metres shown in this unit, unit name, metres per unit)
_LENGTH_UNITS = (
    (KPC_M * 0.1, "kpc", KPC_M),
    (LIGHT_YEAR_M * 0.1, "light-years", LIGHT_YEAR_M),
    (AU_M * 0.1, "AU", AU_M),
    (1e3, "km", 1e3),
    (0.1, "m", 1.0),
    (1e-3, "mm", 1e-3),
    (1e-6, "um", 1e-6),
    (1e-9, "nm", 1e-9),
    (1e-12, "pm", 1e-12),
)


def number_text(value):
    """Three significant digits, without an exponent for everyday sizes (2000, not 2e+03)."""
    if value != 0.0 and not 1e-3 <= abs(value) < 1e9:
        return f"{value:.3g}"
    rounded = float(f"{value:.3g}")
    return f"{rounded:,.0f}" if abs(rounded) >= 1e4 else f"{rounded:g}"


def length_text(metres):
    """A length in SI or astronomical units with a prefix that keeps the number readable."""
    for size, name, per_unit in _LENGTH_UNITS:
        if metres >= size:
            return f"{number_text(metres / per_unit)} {name}"
    return f"{metres / 1e-15:.2g} fm"


def pixels_per_unit(distance, fov, height):
    """Pixels per world unit at `distance` from the eye, for a perspective view
    with vertical field of view `fov` (radians) on a window `height` pixels tall."""
    return height / (2.0 * distance * math.tan(fov / 2.0))


def nice_length(max_length):
    """The largest 1, 2 or 5 times a power of ten that is at most max_length."""
    if not (max_length > 0.0) or not math.isfinite(max_length):
        return 0.0
    power = 10.0 ** math.floor(math.log10(max_length))
    for step in (5.0, 2.0, 1.0):
        if step * power <= max_length * (1.0 + 1e-12):
            return step * power
    return power  # not reached; kept for float safety


def label_text(length, unit_name="", unit_metres=None):
    """Label of a bar `length` world units long.

    The natural unit first, then the real length when unit_metres is known;
    the second part is dropped when it would say the same thing again.
    """
    own = f"{number_text(length)} {unit_name}".strip()
    if not unit_metres:
        return own
    real = length_text(length * unit_metres)
    if real.split()[-1] == unit_name or not unit_name:
        return real
    return f"{own} = {real}"


def draw_scale_bar(
    camera,
    width,
    height,
    unit_metres=None,
    unit_name="",
    note="",
    distance=None,
    corner="left",
):
    """Draw the bar in a bottom corner. Call once per ImGui frame.

    distance: distance from the eye at which the bar is exact; the camera's
    orbit distance when None. corner: "left" or "right".
    """
    d = camera.distance if distance is None else distance
    if width <= 0 or height <= 0 or not (d > 0.0):
        return
    per_unit = pixels_per_unit(d, camera.fov, height)
    length = nice_length(MAX_BAR_PIXELS / per_unit) if per_unit > 0.0 else 0.0
    if length <= 0.0:
        return
    pixels = length * per_unit

    text = label_text(length, unit_name, unit_metres)
    text_size = imgui.calc_text_size(text)
    span = max(pixels, text_size.x)
    if corner == "right":
        x0 = width - MARGIN - span
    else:
        x0 = MARGIN
    y = height - MARGIN
    colour = imgui.get_color_u32((0.85, 0.9, 1.0, 0.9))
    shadow = imgui.get_color_u32((0.0, 0.0, 0.0, 0.6))
    draw_list = imgui.get_background_draw_list()
    for dx, dy, c in ((1.0, 1.0, shadow), (0.0, 0.0, colour)):
        draw_list.add_line((x0 + dx, y + dy), (x0 + pixels + dx, y + dy), c, 2.0)
        draw_list.add_line((x0 + dx, y - TICK + dy), (x0 + dx, y + dy), c, 2.0)
        draw_list.add_line((x0 + pixels + dx, y - TICK + dy), (x0 + pixels + dx, y + dy), c, 2.0)
        draw_list.add_text((x0 + dx, y - TICK - text_size.y - 2.0 + dy), c, text)
        if note:  # on its own line above, so that the bar's corner stays narrow
            draw_list.add_text((x0 + dx, y - TICK - 2.0 * text_size.y - 5.0 + dy), c, note)


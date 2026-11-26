"""A guided tour: the app runs through a list of stops on its own. Plumbing only.

A stop names a module, the spec values it starts with (its scene), how long
it is shown and a caption; the camera turns slowly round what the module
itself framed. The tour is the app's `director` (see Window): the app calls
update() once per frame and passes it the keys and mouse moves first.

Keys while a tour runs: Space holds it (and lets go again), Right and Left
skip to the next or previous stop, Escape ends it where it is. Dragging or
scrolling in the view holds it too, so the view can be explored. F5 starts
the tour from anywhere; so does request_start() (the home screen's button).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from imgui_bundle import imgui

SETTLE_FRAMES = 3  # frames a module gets to set its own view before the tour moves it
MAX_FRAME = 0.1  # s: a slow frame (a module being built) does not eat a stop's time


@dataclass
class Stop:
    module: str  # registry name
    title: str
    caption: str
    seconds: float = 20.0
    params: dict = field(default_factory=dict)  # spec values the module starts with
    time_scale: float = 1.0
    orbit: float = 0.06  # rad/s the camera turns
    zoom: float = 1.0  # camera distance at the end, as a fraction of the start
    pitch: float | None = None  # pitch reached at the end; None leaves it
    move: float = 1.0  # fraction of the stop during which the camera moves; then it rests
    prepare: Callable[[], None] | None = None  # called before the module is built
    start: Callable | None = None  # called with the module once it runs


_requested = False


def request_start():
    """Ask the tour to start on the next frame (for modules, which do not see the app)."""
    global _requested
    _requested = True


def smoothstep(f):
    return f * f * (3.0 - 2.0 * f)


class Tour:
    def __init__(self, stops, home="Home", collapse=()):
        self.stops = list(stops)
        self.home = home
        # Names of ImGui windows folded away while the tour runs (control
        # panels that would cover the picture); unfolded when it ends.
        self.collapse = tuple(collapse)
        self.unfold = False
        self.active = False
        self.index = 0
        self.elapsed = 0.0
        self.holding = False
        self.frames = 0  # since the stop's module was built
        self.base = None  # camera (distance, yaw, pitch) the stop's motion starts from

    # ------------------------------------------------------------- control

    def start(self, win):
        if self.stops:
            self.active = True
            self.go(win, 0)

    def end(self):
        self.active = False
        self.unfold = True

    def go(self, win, index):
        """Show stop `index`; past the last one the tour ends on the home screen."""
        if index >= len(self.stops):
            self.end()
            if self.home in win.registry:
                win.request_switch(self.home)
            return
        self.index = max(index, 0)
        stop = self.stops[self.index]
        self.elapsed, self.holding, self.frames, self.base = 0.0, False, 0, None
        if stop.prepare is not None:
            stop.prepare()
        win.request_switch(stop.module, stop.params)

    def _rebase(self, camera, stop):
        """Make the stop's camera motion pass through the view as it is now."""
        moved, s = self._progress(stop)
        pitch = camera.pitch
        if stop.pitch is not None and s < 1.0:
            pitch = (camera.pitch - stop.pitch * s) / (1.0 - s)
        self.base = (camera.distance / stop.zoom**s, camera.yaw - stop.orbit * moved, pitch)

    def _progress(self, stop):
        """(seconds the camera has moved, eased fraction of its motion done)."""
        span = max(stop.seconds * stop.move, 1e-6)
        moved = min(self.elapsed, span)
        return moved, smoothstep(moved / span)

    # --------------------------------------------------------------- input

    def on_key(self, win, key, action, modifiers):
        """True if the tour used the key."""
        keys = win.window.keys
        if action != keys.ACTION_PRESS:
            return False
        if not self.active:
            if key == keys.F5:
                self.start(win)
                return True
            return False
        if key == keys.SPACE:
            self.holding = not self.holding
            if not self.holding and self.base is not None:
                self._rebase(win.camera, self.stops[self.index])
        elif key == keys.RIGHT:
            self.go(win, self.index + 1)
        elif key == keys.LEFT:
            self.go(win, self.index - 1)
        elif key == keys.ESCAPE or key == keys.F5:
            self.end()
        else:
            return False
        return True

    def on_view_input(self):
        """The user moved the camera: hold the tour so that they can look around."""
        if self.active:
            self.holding = True

    # -------------------------------------------------------------- frames

    def update(self, win, frame_time):
        """Once per frame, inside the ImGui frame."""
        global _requested
        if _requested:
            _requested = False
            self.start(win)
        if self.unfold and not self.active:
            self.unfold = False
            for name in self.collapse:
                imgui.set_window_collapsed(name, False)
        if not self.active:
            return
        stop = self.stops[self.index]
        if win.pending_module_name is None:
            if win.active_module_name != stop.module:
                self.end()  # the user went somewhere else
                return
            self._advance(win, stop, min(frame_time, MAX_FRAME))
        if self.active:
            self._draw_caption(self.stops[self.index])

    def _advance(self, win, stop, dt):
        camera = win.camera
        win.pause_flag = False
        win.time_scale = stop.time_scale
        self.frames += 1
        if self.frames < SETTLE_FRAMES:
            return
        if self.base is None:
            self.base = (camera.distance, camera.yaw, camera.pitch)
            for name in self.collapse:
                imgui.set_window_collapsed(name, True)
            if stop.start is not None:
                stop.start(win.active_module)
        if self.holding:
            return
        self.elapsed += dt
        moved, s = self._progress(stop)
        distance, yaw, pitch = self.base
        camera.distance = distance * stop.zoom**s
        camera.yaw = yaw + stop.orbit * moved
        camera.pitch = pitch if stop.pitch is None else pitch + (stop.pitch - pitch) * s
        if self.elapsed >= stop.seconds:
            self.go(win, self.index + 1)

    def _draw_caption(self, stop):
        io = imgui.get_io()
        W, H = io.display_size.x, io.display_size.y
        state = "held: Space to go on" if self.holding else "Space: hold"
        hint = (
            f"{self.index + 1} / {len(self.stops)}    {state}    Left / Right: skip    Esc: end"
        )
        lines = (
            (stop.title, (1.0, 0.9, 0.55, 1.0)),
            (stop.caption, (0.92, 0.95, 1.0, 1.0)),
            (hint, (0.6, 0.65, 0.75, 1.0)),
        )
        sizes = [imgui.calc_text_size(text) for text, _ in lines]
        width = max(size.x for size in sizes) + 28.0
        height = sum(size.y for size in sizes) + 8.0 * (len(lines) - 1) + 20.0
        x0, y0 = (W - width) / 2.0, H - height - 18.0
        draw_list = imgui.get_foreground_draw_list()
        draw_list.add_rect_filled(
            (x0, y0), (x0 + width, y0 + height), imgui.get_color_u32((0.03, 0.04, 0.07, 0.78)), 6.0
        )
        y = y0 + 10.0
        for (text, color), size in zip(lines, sizes, strict=True):
            draw_list.add_text(((W - size.x) / 2.0, y), imgui.get_color_u32(color), text)
            y += size.y + 8.0

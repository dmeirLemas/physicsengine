import time

import moderngl_window as mglw
from imgui_bundle import imgui
from moderngl_window.timers.clock import Timer

from physengine.core.camera import OrbitCamera
from physengine.core.imgui_renderer import ImGuiRenderer
from physengine.core.module import BaseModule
from physengine.core.recorder import Recorder


class Window:
    def __init__(self, W, H, registry: dict, title="TEST", director=None) -> None:
        self.pause_flag = True
        self.W = W
        self.H = H

        if not registry:
            raise Exception("Empty registry")

        self.registry = registry

        self.combo_box_items = list(registry.keys())

        self.active_module: BaseModule
        self.active_module_name: str | None = None

        self.pending_module_name: str | None = None
        # True when the pending switch is a reset that keeps the spec values.
        self.pending_keep_params = False
        # Spec values the pending module starts with (request_switch).
        self.pending_params: dict = {}
        # An optional object that runs the app by itself, such as a guided
        # tour (core/tour.py). Every frame the app calls its
        # update(window, frame_time); keys go to its on_key(window, key,
        # action, modifiers) first (True: the key was used), and camera moves
        # by the user are reported to its on_view_input().
        self.director = director

        self.time_scale = 1.0
        self.step_count = 0
        self.frame_counter = 0

        self.accumulator = 0.0
        self.max_accumulator = 0.25
        # Wall-clock seconds of physics per frame. When the steps cost more
        # than this, the simulation runs slower than the time scale asks for
        # instead of dragging the frame rate down.
        self.frame_budget = 0.020
        # Simulated time per wall second, measured over the last speed_window
        # seconds as (time asked for - time dropped) / wall time. Counting
        # dropped time, not steps taken, keeps it steady when a step only
        # comes every few frames.
        self.achieved_speed = 1.0
        self.speed_window = 0.5
        self.window_wall = 0.0
        self.window_kept = 0.0

        window_cls = mglw.get_local_window_cls("glfw")
        self.window = window_cls(size=(W, H), title=title, gl_version=(4, 6), vsync=True)
        mglw.activate_context(window=self.window)
        self.ctx = self.window.ctx

        imgui.create_context()
        self.imgui_ = ImGuiRenderer(self.window)

        self.window.resize_func = self.on_resize
        self.window.key_event_func = self.on_key

        self.window.mouse_drag_event_func = self.on_mouse_drag
        self.window.mouse_scroll_event_func = self.on_mouse_scroll
        self.window.mouse_position_event_func = self.on_mouse_position
        self.window.mouse_press_event_func = self.on_mouse_press
        self.window.mouse_release_event_func = self.on_mouse_release
        self.window.unicode_char_entered_func = self.on_char

        self.camera = OrbitCamera(W, H)

        # Records the 3D view, without the ImGui windows, to an MP4 (F9).
        self.recorder = Recorder()
        self.record_for_seconds = False
        self.record_seconds = 10.0
        # The Recording window, opened from the Simulation window.
        self.show_recording_window = False
        self.confirm_discard = False
        # Bottom-left corner and width of the Simulation window, to place it below.
        self.sim_window_rect = (10.0, 10.0, 0.0)

    def on_mouse_drag(self, x, y, dx, dy):
        self.imgui_.mouse_drag_event(x, y, dx, dy)
        if not imgui.get_io().want_capture_mouse:
            # Left drag orbits; right or middle drag pans.
            if self.director is not None:
                self.director.on_view_input()
            states = self.window.mouse_states
            if states.right or states.middle:
                self.camera.pan(dx, dy)
            else:
                self.camera.orbit(dx, dy)

    def on_mouse_scroll(self, x, y):
        self.imgui_.mouse_scroll_event(x, y)
        if not imgui.get_io().want_capture_mouse:
            if self.director is not None:
                self.director.on_view_input()
            self.camera.zoom(y)

    def on_mouse_position(self, x, y, dx, dy):
        self.imgui_.mouse_position_event(x, y, dx, dy)

    def on_mouse_press(self, x, y, button):
        self.imgui_.mouse_press_event(x, y, button)

    def on_mouse_release(self, x, y, button):
        self.imgui_.mouse_release_event(x, y, button)

    def on_char(self, char):
        self.imgui_.unicode_char_entered(char)

    def on_resize(self, W, H):
        self.W = W
        self.H = H

        self.imgui_.resize(W, H)
        self.camera.on_resize(W, H)

    def on_key(self, key, action, modifiers):
        ACTION_PRESS = self.window.keys.ACTION_PRESS
        SPACE = self.window.keys.SPACE
        ESC = self.window.keys.ESCAPE
        R = self.window.keys.R
        F9 = self.window.keys.F9

        self.imgui_.key_event(key, action, modifiers)
        if not imgui.get_io().want_capture_keyboard:
            director = self.director
            if director is not None and director.on_key(self, key, action, modifiers):
                return
            if key == SPACE and action == ACTION_PRESS:
                self.pause_flag = not self.pause_flag

            # Reset: rebuild the active module from scratch on the next frame.
            if key == R and action == ACTION_PRESS:
                if modifiers.shift:
                    self.request_full_reset()
                else:
                    self.request_reset()

            if key == F9 and action == ACTION_PRESS:
                self.toggle_recording()

            if key == ESC and action == ACTION_PRESS:
                self.window.close()

    def run(self):
        try:
            self.timer = Timer()
            self.timer.start()

            first_key = next(iter(self.registry))
            self.switch(first_key)

            while not self.window.is_closing:
                if self.pending_module_name is not None:
                    self.switch(
                        self.pending_module_name, self.pending_keep_params, self.pending_params
                    )
                    self.pending_module_name = None
                    self.pending_keep_params = False
                    self.pending_params = {}

                curr_time, frame_time = self.timer.next_frame()

                if not self.pause_flag:
                    wanted = frame_time * self.time_scale
                    self.accumulator += wanted
                    dropped = max(self.accumulator - self.max_accumulator, 0.0)
                    self.accumulator -= dropped

                    budget_start = time.perf_counter()
                    while self.accumulator >= self.dt:
                        self.physics_step()
                        self.accumulator -= self.dt
                        if time.perf_counter() - budget_start > self.frame_budget:
                            # Can't keep up: drop the whole steps still owed,
                            # but keep the fraction of a step, which is not
                            # debt (dropping it too would lose time after
                            # every step that is slower than the budget).
                            owed = self.accumulator // self.dt * self.dt
                            self.accumulator -= owed
                            dropped += owed
                            break

                    self.window_wall += frame_time
                    self.window_kept += wanted - dropped
                    if self.window_wall >= self.speed_window:
                        self.achieved_speed = max(self.window_kept, 0.0) / self.window_wall
                        self.window_wall = self.window_kept = 0.0
                else:
                    self.achieved_speed = self.time_scale
                    self.window_wall = self.window_kept = 0.0

                self.window.clear(0.1, 0.2, 0.3)
                self.active_module.render(self.ctx, self.camera)
                # Read the frame now, before the ImGui windows are drawn over it.
                self.recorder.capture(self.window.buffer_size, frame_time)

                self.imgui_.begin_frame(frame_time)

                imgui.set_next_window_pos((10.0, 10.0), imgui.Cond_.always)
                imgui.begin(
                    "Simulation",
                    None,
                    imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move,
                )

                idx = self.combo_box_items.index(self.active_module_name)

                changed, new_idx = imgui.combo("Modules", idx, self.combo_box_items)

                if changed:
                    self.pending_module_name = self.combo_box_items[new_idx]
                    self.pending_keep_params = False

                imgui.text(f"Step {self.step_count}")
                _, self.pause_flag = imgui.checkbox("Pause", self.pause_flag)
                imgui.same_line()
                if imgui.button("Restart"):
                    self.request_reset()
                imgui.set_item_tooltip("Back to the start (R). Keeps the scene and its edits.")
                imgui.same_line()
                if imgui.button("Full reset"):
                    self.request_full_reset()
                imgui.set_item_tooltip(
                    "Back to the starting state: default settings, no edits (Shift+R)"
                )
                _, self.time_scale = imgui.slider_float("Time Scale", self.time_scale, 0.1, 10.0)
                if not self.pause_flag and self.achieved_speed < 0.9 * self.time_scale:
                    imgui.text_colored(
                        (1.0, 0.75, 0.3, 1.0),
                        f"CPU-limited: running at {self.achieved_speed:.2f} of {self.time_scale:.2f}",
                    )

                imgui.separator()
                for spec in self.active_module.specs:
                    if spec.draw(self.active_module) and spec.need_reset:
                        self.request_reset()

                self.draw_record_launcher()

                diagnostics = self.active_module.diagnostics()
                if diagnostics and imgui.tree_node("Diagnostics"):
                    imgui.text(diagnostics)
                    imgui.tree_pop()
                pos, size = imgui.get_window_pos(), imgui.get_window_size()
                self.sim_window_rect = (pos.x, pos.y + size.y, size.x)
                imgui.end()

                self.draw_recording_window()

                # The module's own windows, if it has any.
                self.active_module.draw_ui()
                if self.active_module.reset_requested:
                    self.request_reset()
                wanted_module = self.active_module.switch_requested
                if wanted_module is not None and wanted_module in self.registry:
                    self.pending_module_name = wanted_module
                    self.pending_keep_params = False

                if self.director is not None:
                    self.director.update(self, frame_time)

                self.imgui_.end_frame()
                self.window.swap_buffers()

                self.frame_counter += 1

        finally:
            self.recorder.stop()
            if self.active_module_name is not None:
                self.active_module.teardown()
            self.imgui_.shutdown()
            self.window.destroy()

    def toggle_recording(self):
        """Start recording the 3D view, or stop and save the recording."""
        recorder = self.recorder
        if recorder.recording:
            recorder.stop()
        else:
            recorder.duration = self.record_seconds if self.record_for_seconds else None
            recorder.start(self.window.buffer_size)

    def recording_status(self):
        """(text, colour) of the recorder's state, such as "REC 0:12.3"."""
        recorder = self.recorder
        if not recorder.recording:
            return "Not recording", (0.6, 0.6, 0.6, 1.0)
        elapsed = f"{int(recorder.elapsed // 60)}:{recorder.elapsed % 60:04.1f}"
        if recorder.duration is not None:
            total = recorder.duration
            elapsed += f" / {int(total // 60)}:{total % 60:04.1f}"
        if recorder.paused:
            return f"PAUSED {elapsed}", (1.0, 0.8, 0.3, 1.0)
        return f"REC {elapsed}", (1.0, 0.3, 0.3, 1.0)

    def draw_record_launcher(self):
        """The Simulation window's line for recording: a button that opens the
        Recording window, and the recorder's state while it records."""
        label = "Hide recording" if self.show_recording_window else "Recording..."
        if imgui.button(label):
            self.show_recording_window = not self.show_recording_window
        imgui.set_item_tooltip("Record the 3D view to a video (F9 starts and stops)")
        if self.recorder.recording:
            imgui.same_line()
            text, color = self.recording_status()
            imgui.text_colored(color, text)

    def draw_recording_window(self):
        """Controls for the recorder: what to record, start, pause, stop, discard."""
        if not self.show_recording_window:
            return
        recorder = self.recorder
        left, bottom, sim_width = self.sim_window_rect
        H = imgui.get_io().display_size.y
        # Below the Simulation window, or beside it when there is no room below.
        if bottom + 10.0 + 200.0 <= H:
            imgui.set_next_window_pos((left, bottom + 10.0), imgui.Cond_.always)
        else:
            imgui.set_next_window_pos((left + sim_width + 10.0, 10.0), imgui.Cond_.always)
        flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move
        expanded, self.show_recording_window = imgui.begin("Recording", True, flags)
        if not expanded:
            imgui.end()
            return

        text, color = self.recording_status()
        imgui.text_colored(color, text)
        if recorder.recording and recorder.duration is not None:
            imgui.progress_bar(recorder.elapsed / recorder.duration, (220.0, 0.0), "")

        imgui.separator()
        imgui.begin_disabled(recorder.recording)
        if imgui.radio_button("Until stopped", not self.record_for_seconds):
            self.record_for_seconds = False
        imgui.same_line()
        if imgui.radio_button("For", self.record_for_seconds):
            self.record_for_seconds = True
        imgui.same_line()
        imgui.set_next_item_width(90.0)
        _, seconds = imgui.input_float("seconds", self.record_seconds, 1.0, 10.0, "%.1f")
        self.record_seconds = max(seconds, 0.5)
        imgui.set_next_item_width(90.0)
        fps_choices = (24, 30, 60)
        index = fps_choices.index(recorder.fps) if recorder.fps in fps_choices else 2
        changed, index = imgui.combo("Frame rate", index, [str(f) for f in fps_choices])
        if changed:
            recorder.fps = fps_choices[index]
        bw, bh = self.window.buffer_size
        scales = (1.0, 0.5)
        labels = [f"Full ({bw}x{bh})", f"Half ({int(bw * 0.5)}x{int(bh * 0.5)})"]
        index = scales.index(recorder.scale) if recorder.scale in scales else 0
        imgui.set_next_item_width(170.0)
        changed, index = imgui.combo("Resolution", index, labels)
        if changed:
            recorder.scale = scales[index]
        imgui.set_item_tooltip("Half: 4x fewer pixels to copy and encode, lighter on slow modules.")
        imgui.end_disabled()

        imgui.separator()
        if not recorder.recording:
            self.confirm_discard = False
            if imgui.button("Start"):
                self.toggle_recording()
            imgui.set_item_tooltip("F9")
        elif self.confirm_discard:
            imgui.text("Delete this recording?")
            if imgui.button("Yes, discard"):
                recorder.discard()
                self.confirm_discard = False
            imgui.same_line()
            if imgui.button("Keep recording"):
                self.confirm_discard = False
        else:
            if imgui.button("Resume" if recorder.paused else "Pause"):
                recorder.toggle_pause()
            imgui.set_item_tooltip(
                "While paused, nothing is recorded; the video carries on from here on resume."
            )
            imgui.same_line()
            if imgui.button("Stop and save"):
                recorder.stop()
            imgui.set_item_tooltip("F9")
            imgui.same_line()
            if imgui.button("Discard"):
                self.confirm_discard = True

        if recorder.message:
            imgui.text_disabled(recorder.message)
        imgui.text_disabled("Only the 3D view is recorded, not these windows.")
        imgui.end()

    def request_reset(self):
        """Rebuild the active module on the next frame, keeping its spec values.

        Does nothing if a switch to another module is already pending.
        """
        if self.pending_module_name is None:
            self.pending_module_name = self.active_module_name
            self.pending_keep_params = True

    def request_full_reset(self):
        """Rebuild the active module as it starts: no edits, every setting at its default.

        The module first discards the edits it keeps across rebuilds; the new
        instance then starts from its own defaults, as when it is first chosen.
        """
        if self.pending_module_name is None:
            self.active_module.discard_edits()
            self.pending_module_name = self.active_module_name
            self.pending_keep_params = False

    def request_switch(self, class_name, params=None):
        """Switch to a module on the next frame (also to the active one, rebuilt from
        its defaults). params: spec values it starts with, set before its setup()."""
        if class_name not in self.registry:
            raise Exception("Class to switch does not exist in the registry.")
        self.pending_module_name = class_name
        self.pending_keep_params = False
        self.pending_params = dict(params or {})

    def switch(self, class_name, keep_params=False, params=None):
        if class_name not in self.registry:
            raise Exception("Class to switch does not exist in the registry.")

        kept = {}
        if self.active_module_name is not None:
            if keep_params and class_name == self.active_module_name:
                kept = {
                    spec.attr_name: getattr(self.active_module, spec.attr_name)
                    for spec in self.active_module.specs
                }
            self.active_module.teardown()
        kept.update(params or {})

        self.active_module = self.registry[class_name]()
        self.camera.reset()

        # Carried-over values are set before setup(), so setup() sees them.
        for attr_name, value in kept.items():
            setattr(self.active_module, attr_name, value)

        if self.active_module.dt > self.max_accumulator:
            raise Exception("dt of new class > max_accumulator")

        self.active_module.setup(self.ctx)

        for spec in self.active_module.specs:
            if not hasattr(self.active_module, spec.attr_name):
                raise Exception(f"{self.active_module} missing {spec.attr_name} required in {spec}")

        self.dt = self.active_module.dt

        self.step_count = 0
        self.frame_counter = 0

        self.time_scale = 1.0
        self.accumulator = 0.0
        self.pause_flag = True
        self.active_module_name = class_name

    def physics_step(self):
        self.active_module.step(self.active_module.dt)
        self.step_count += 1

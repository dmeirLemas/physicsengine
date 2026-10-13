import moderngl_window as mglw
from imgui_bundle import imgui
from moderngl_window.timers.clock import Timer

from physengine.core.camera import OrbitCamera
from physengine.core.imgui_renderer import ImGuiRenderer
from physengine.core.module import BaseModule


class Window:
    def __init__(self, W, H, registry: dict, title="TEST") -> None:
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

        self.time_scale = 1.0
        self.step_count = 0
        self.frame_counter = 0

        self.accumulator = 0.0
        self.max_accumulator = 0.25

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

    def on_mouse_drag(self, x, y, dx, dy):
        self.imgui_.mouse_drag_event(x, y, dx, dy)
        if not imgui.get_io().want_capture_mouse:
            self.camera.orbit(dx, dy)

    def on_mouse_scroll(self, x, y):
        self.imgui_.mouse_scroll_event(x, y)
        if not imgui.get_io().want_capture_mouse:
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

        self.imgui_.key_event(key, action, modifiers)
        if not imgui.get_io().want_capture_keyboard:
            if key == SPACE and action == ACTION_PRESS:
                self.pause_flag = not self.pause_flag

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
                    self.switch(self.pending_module_name)
                    self.pending_module_name = None

                curr_time, frame_time = self.timer.next_frame()

                if not self.pause_flag:
                    self.accumulator += frame_time * self.time_scale
                    self.accumulator = min(self.max_accumulator, self.accumulator)

                    while self.accumulator >= self.dt:
                        self.physics_step()
                        self.accumulator -= self.dt

                self.window.clear(0.1, 0.2, 0.3)
                self.active_module.render(self.ctx, self.camera)

                self.imgui_.begin_frame(frame_time)

                imgui.begin("Simulation")

                idx = self.combo_box_items.index(self.active_module_name)

                changed, new_idx = imgui.combo("Modules", idx, self.combo_box_items)

                if changed:
                    self.pending_module_name = self.combo_box_items[new_idx]

                imgui.text(f"{self.step_count, self.frame_counter}")
                _, self.pause_flag = imgui.checkbox("Pause", self.pause_flag)
                _, self.time_scale = imgui.slider_float("Time Scale", self.time_scale, 0.1, 10.0)

                imgui.text("Specs")
                for spec in self.active_module.specs:
                    spec.draw(self.active_module)

                imgui.text(f"{self.active_module.diagnostics()}")
                imgui.end()

                self.imgui_.end_frame()
                self.window.swap_buffers()

                self.frame_counter += 1

        finally:
            if self.active_module_name is not None:
                self.active_module.teardown()
            self.imgui_.shutdown()
            self.window.destroy()

    def switch(self, class_name):
        if self.active_module_name is not None:
            for spec in self.active_module.specs:
                if not hasattr(self.active_module, spec.attr_name):
                    raise Exception(
                        f"{self.active_module} missing {spec.attr_name} required in {spec}"
                    )

            self.active_module.teardown()
        if class_name not in self.registry:
            raise Exception("Class to switch does not exist in the registry.")

        self.active_module = self.registry[class_name]()
        self.active_module.setup(self.ctx)
        self.dt = self.active_module.dt

        self.step_count = 0
        self.frame_counter = 0

        self.accumulator = 0.0
        self.pause_flag = True
        self.active_module_name = class_name

    def physics_step(self):
        self.active_module.step(self.active_module.dt)
        self.step_count += 1

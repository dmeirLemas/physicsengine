import moderngl_window as mglw
from moderngl_window.timers.clock import Timer


class Window:
    def __init__(self, W, H, module, title="TEST") -> None:
        self.pause_flag = False
        self.W = W
        self.H = H

        self.active_module = module
        self.time_scale = 1.0
        self.step_count = 0
        self.frame_counter = 0

        self.accumulator = 0.0
        self.dt = self.active_module.dt if self.active_module is not None else 1.0 / 240.0
        self.max_accumulator = 0.25

        window_cls = mglw.get_local_window_cls("glfw")
        self.window = window_cls(size=(W, H), title=title, gl_version=(4, 6), vsync=True)
        mglw.activate_context(window=self.window)
        self.ctx = self.window.ctx

        self.window.resize_func = self.on_resize
        self.window.key_event_func = self.on_key

    def on_resize(self, W, H):
        self.W = W
        self.H = H

    def on_key(self, key, action, modifiers):
        ACTION_PRESS = self.window.keys.ACTION_PRESS
        SPACE = self.window.keys.SPACE
        ESC = self.window.keys.ESCAPE

        if key == SPACE and action == ACTION_PRESS:
            self.pause_flag = not self.pause_flag

        if key == ESC and action == ACTION_PRESS:
            self.window.close()

    def run(self):
        try:
            self.timer = Timer()
            self.timer.start()
            print_time = 0

            if self.active_module is not None:
                self.active_module.setup(self.ctx)

            while not self.window.is_closing:
                curr_time, frame_time = self.timer.next_frame()
                if curr_time - print_time >= 1.0:
                    print_time = curr_time
                    print(self.step_count, self.frame_counter)

                if not self.pause_flag:
                    self.accumulator += frame_time * self.time_scale
                    self.accumulator = min(self.max_accumulator, self.accumulator)

                    while self.accumulator >= self.dt:
                        self.physics_step()
                        self.accumulator -= self.dt

                self.window.clear(0.1, 0.2, 0.3)
                if self.active_module is not None:
                    self.active_module.render(self.ctx, self.W, self.H)
                self.window.swap_buffers()

                self.frame_counter += 1

        finally:
            if self.active_module is not None:
                try:
                    self.active_module.teardown()
                finally:
                    self.window.destroy()

    def physics_step(self):
        if self.active_module is not None:
            self.active_module.step(self.active_module.dt)
            self.step_count += 1

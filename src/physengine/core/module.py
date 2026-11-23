from abc import ABC, abstractmethod


class BaseModule(ABC):
    dt = 1.0 / 240.0
    specs = tuple()
    # A module sets this to True to ask the app to rebuild it on the next
    # frame, keeping its spec values (the same as the Reset button).
    reset_requested = False
    # A module sets this to the registry name of another module to ask the
    # app to switch to it on the next frame (the home screen's buttons).
    switch_requested = None

    @abstractmethod
    def setup(self, ctx): ...

    @abstractmethod
    def step(self, dt): ...

    @abstractmethod
    def render(self, ctx, camera): ...

    @abstractmethod
    def teardown(self): ...

    def draw_ui(self):
        return

    def discard_edits(self):
        """Called just before a full reset rebuilds the module.

        A module that keeps state across rebuilds (for example user edits that
        Restart returns to, or settings stored outside its specs) drops it
        here, so the rebuilt module starts from its definition. Spec values
        go back to their defaults anyway: a full reset builds a fresh instance.
        """
        return

    def diagnostics(self):
        return ""

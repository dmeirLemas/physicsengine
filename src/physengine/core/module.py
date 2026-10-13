from abc import ABC, abstractmethod


class BaseModule(ABC):
    dt = 1.0 / 240.0
    specs = tuple()

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

    def diagnostics(self):
        return ""

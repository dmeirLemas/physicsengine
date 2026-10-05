from abc import ABC, abstractmethod


class BaseModule(ABC):
    dt = 1.0 / 240.0

    @abstractmethod
    def setup(self, ctx): ...

    @abstractmethod
    def step(self, dt): ...

    @abstractmethod
    def render(self, ctx, width, height): ...

    @abstractmethod
    def draw_ui(self): ...

    @abstractmethod
    def teardown(self): ...

    def diagnostics(self):
        return ""

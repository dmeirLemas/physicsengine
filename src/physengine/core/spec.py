from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from imgui_bundle import imgui


@dataclass(kw_only=True)
class Spec(ABC):
    attr_name: str
    label: str
    default: Any

    @abstractmethod
    def draw(self, module): ...


@dataclass(kw_only=True)
class SliderFloat(Spec):
    min: float
    max: float
    need_reset: bool = False

    def draw(self, module):
        val = getattr(module, self.attr_name)
        _, val = imgui.slider_float(self.label, val, self.min, self.max)
        setattr(module, self.attr_name, val)

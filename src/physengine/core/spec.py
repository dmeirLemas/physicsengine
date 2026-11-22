from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from imgui_bundle import imgui


@dataclass(kw_only=True)
class Spec(ABC):
    attr_name: str
    label: str
    default: Any
    # If True, the app rebuilds the module (keeping its spec values) when
    # this value changes.
    need_reset: bool = False

    @abstractmethod
    def draw(self, module) -> bool:
        """Draw the widget, write the value to the module, return True if it changed."""


@dataclass(kw_only=True)
class SliderFloat(Spec):
    min: float
    max: float

    def draw(self, module):
        val = getattr(module, self.attr_name)
        changed, val = imgui.slider_float(self.label, val, self.min, self.max)
        if changed:
            # Only on a change: the widget works in float32 and would otherwise
            # round the value every frame (0.6 becoming 0.6000000238).
            setattr(module, self.attr_name, val)
        return changed


@dataclass(kw_only=True)
class Choice(Spec):
    """Pick one string out of `options` with a combo box."""

    options: tuple[str, ...]

    def draw(self, module):
        options = list(self.options)
        idx = options.index(getattr(module, self.attr_name))
        changed, idx = imgui.combo(self.label, idx, options)
        if changed:
            setattr(module, self.attr_name, options[idx])
        return changed

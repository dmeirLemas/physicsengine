"""ImGui rendering and input glue for a moderngl-window window.

Why this file exists
--------------------
Dear ImGui 1.92 replaced the static font-atlas API with a dynamic texture
protocol (``ImTextureData`` objects that the renderer must create, update and
destroy on request). imgui-bundle 1.92 follows that protocol.

moderngl-window ships its own ImGui integration
(``moderngl_window.integrations.imgui_bundle.ModernglWindowRenderer``), but as
of moderngl-window 3.1.1 its *drawing* half still calls the removed
``get_tex_data_as_rgba32`` method and fails at construction time.  Its *input*
half, ``ModernglWindowMixin``, is independent of the texture API and still works.

imgui-bundle itself ships a drawing backend that implements the new protocol,
``ProgrammablePipelineRenderer``.  It uses PyOpenGL rather than ModernGL, and it
backs up and restores the GL state it touches, so it can share a context with
ModernGL without corrupting ModernGL's view of the state.

This module composes the two working halves into one class.  There is no
project logic here; it is purely an adapter between the libraries.  If
moderngl-window is fixed upstream, this file can be deleted and
``ModernglWindowRenderer`` used directly with the same call pattern.

Two platform pitfalls this module handles
-----------------------------------------
1. imgui-bundle's Linux wheel bundles its own ``libglfw.so.3``, built for X11
   only.  Whichever GLFW library is loaded first wins for the whole process, and
   imgui-bundle is normally imported before any window exists, so on a Wayland
   desktop the window silently runs under XWayland with a GLX context (instead
   of native Wayland with EGL).  Harmless, but it decides point 2.

2. PyOpenGL must ask the *right* windowing API for the current GL context, or
   every pointer-storing call (``glVertexAttribPointer`` etc.) raises
   "Attempt to retrieve context when no valid context".  PyOpenGL picks its
   platform once, at import, from environment variables, and on a Wayland
   session it guesses EGL, which cannot see a GLX context.  So PyOpenGL is
   imported *lazily* in ``ImGuiRenderer.__init__``, after the window exists,
   with ``PYOPENGL_PLATFORM`` set from the platform GLFW actually chose.
   This is also why the class *contains* the imgui-bundle backend instead of
   inheriting from it: the backend class cannot be named at import time.

How to use it
-------------
1. Create the window with moderngl-window and call
   ``moderngl_window.activate_context(window=window)``.
2. Call ``imgui.create_context()`` once.
3. Construct ``ImGuiRenderer(window)``.
4. Forward every window event to the method of the same name on this object
   (``resize``, ``key_event``, ``mouse_position_event``, ``mouse_drag_event``,
   ``mouse_scroll_event``, ``mouse_press_event``, ``mouse_release_event``,
   ``unicode_char_entered``).  They share the window callbacks' signatures.
5. Each frame: ``begin_frame(frame_time)``, emit widgets, ``end_frame()``.
   Draw the 3D scene *before* ``begin_frame``/``end_frame`` so the UI is on top.
6. Query ``imgui.get_io().want_capture_mouse`` / ``want_capture_keyboard``
   after forwarding an event to decide whether the scene should also see it.
7. Call ``shutdown()`` before destroying the window.
"""

import os

import glfw
from imgui_bundle import imgui
from moderngl_window.integrations.imgui_bundle import (
    ModernglWindowMixin,
    ModernglWindowRenderer,
)


def _select_pyopengl_platform() -> None:
    """Point PyOpenGL at the windowing API GLFW is really using.

    Must run before the first ``import OpenGL``.  An explicit user setting of
    ``PYOPENGL_PLATFORM`` is respected.
    """
    if "PYOPENGL_PLATFORM" in os.environ:
        return
    platform = glfw.get_platform()
    if platform == glfw.PLATFORM_WAYLAND:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif platform == glfw.PLATFORM_X11:
        os.environ["PYOPENGL_PLATFORM"] = "glx"
    # Other platforms (Win32, Cocoa): PyOpenGL's own detection is correct.


class ImGuiRenderer(ModernglWindowMixin):
    """ImGui draw + input bridge for a moderngl-window ``BaseWindow``.

    Input: inherited from ``ModernglWindowMixin`` (moderngl-window).  Its event
    handlers translate window events into ``imgui.get_io()`` state.  They expect
    ``self.wnd`` (the window), ``self.io`` and ``self.REVERSE_KEYMAP`` (window key
    constant -> ``imgui.Key``), all set up in ``__init__``.

    Drawing: delegated to an imgui-bundle ``ProgrammablePipelineRenderer`` held
    in ``self._backend`` (see pitfall 2 in the module docstring for why it is
    not a base class).
    """

    # Borrow the key map builder from moderngl-window's (otherwise broken)
    # renderer class rather than copying its table.  It only reads
    # ``self.wnd.keys`` and assigns ``self.REVERSE_KEYMAP``, so it is safe to
    # call on this class.
    _init_key_maps = ModernglWindowRenderer._init_key_maps

    def __init__(self, window) -> None:
        # The mixin's handlers read the window for size, pixel ratio and mouse
        # button state, so attach it before anything else can fire.
        self.wnd = window

        # Pitfall 2: choose PyOpenGL's platform now that GLFW has chosen its own,
        # then import the backend for the first time.
        _select_pyopengl_platform()
        from imgui_bundle.python_backends.opengl_backend_programmable import (
            ProgrammablePipelineRenderer,
        )

        # imgui-bundle's backend: checks that an ImGui context exists, compiles
        # its shader, creates its buffers, and sets the ``renderer_has_textures``
        # backend flag that enables the 1.92 texture protocol.
        self._backend = ProgrammablePipelineRenderer()
        self.io = self._backend.io

        self._init_key_maps()

        # Tell ImGui the logical window size and the logical->framebuffer scale
        # (differs from 1.0 on HiDPI displays).  ``resize`` keeps these current.
        self.io.display_size = self.wnd.size
        self.io.display_framebuffer_scale = (self.wnd.pixel_ratio, self.wnd.pixel_ratio)

    # ------------------------------------------------------------------ frame

    def begin_frame(self, frame_time: float) -> None:
        """Start an ImGui frame.

        ``frame_time`` is the wall-clock seconds since the previous frame.  ImGui
        uses it for double-click timing, cursor blink and animations; the backend
        initialises it to a constant 1/60, which is wrong at other refresh
        rates.  Must be positive, so clamp a zero first frame.
        """
        self.io.delta_time = max(frame_time, 1e-6)
        imgui.new_frame()

    def end_frame(self) -> None:
        """Finish the ImGui frame and draw it with the current GL context."""
        imgui.render()
        self._backend.render(imgui.get_draw_data())

    # --------------------------------------------------------------- teardown

    def shutdown(self) -> None:
        """Release the shader, buffers and ImGui-owned textures.

        Call before destroying the window, while the GL context is current.
        """
        self._backend.shutdown()

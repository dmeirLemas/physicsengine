# physengine

A physics simulator with one window and several simulations: massive-body systems, galaxy collisions, black holes, particle decays in a detector, and the atom. All modules share the same window, camera and controls.

## Requirements

- Python 3.12 or later. Developed on Python 3.14.
- A GPU or driver with OpenGL 4.6 (the shaders use GLSL 460). Developed on an Intel integrated GPU with Mesa.
- A desktop session. The window runs through GLFW; on Wayland it runs under XWayland.

first time you save. Saved scenarios also appear on the Home screen.

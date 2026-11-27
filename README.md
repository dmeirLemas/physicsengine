# physengine

A physics simulator with one window and several simulations: massive-body systems, galaxy collisions, black holes, particle decays in a detector, and the atom. All modules share the same window, camera and controls.

## Requirements

- Python 3.12 or later. Developed on Python 3.14.
- A GPU or driver with OpenGL 4.6 (the shaders use GLSL 460). Developed on an Intel integrated GPU with Mesa.
- A desktop session. The window runs through GLFW; on Wayland it runs under XWayland.

first time you save. Saved scenarios also appear on the Home screen.

## Recording

Click **Recording...** in the Simulation window to open the Recording window, below it on the left. Choose *Until stopped* or *For N seconds*, a frame rate (30 by default) and a resolution, then **Start**. *Half* resolution has 4x fewer pixels to copy and encode, so it slows the simulation less; use it in the heavier modules. While recording you can **Pause** and **Resume** (nothing is recorded while paused, so the video continues from where it was paused), **Stop and save**, or **Discard** the recording. **F9** starts and stops a recording from anywhere. Videos are saved as MP4 files in `recordings/`.

The video holds only the 3D view: the menu windows and other ImGui overlays (scale bar, labels, tour captions) are not in it. Resizing the window ends the recording. Encoding uses the ffmpeg binary from the `imageio-ffmpeg` package, or `ffmpeg` on the PATH.

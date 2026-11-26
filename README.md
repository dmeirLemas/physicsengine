# physengine

A physics simulator with one window and several simulations: massive-body systems, galaxy collisions, black holes, particle decays in a detector, and the atom. All modules share the same window, camera and controls.

## Requirements

- Python 3.12 or later. Developed on Python 3.14.
- A GPU or driver with OpenGL 4.6 (the shaders use GLSL 460). Developed on an Intel integrated GPU with Mesa.
- A desktop session. The window runs through GLFW; on Wayland it runs under XWayland.

No CUDA is needed.

## Setup

From the repository root:

```bash
python3 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install -e ".[dev]"
```

`.[dev]` adds pytest and ruff for the tests and the linter. Leave it out if you only want to run the app.

The install creates a `physengine` command inside `env/bin/`.

## Running the app

```bash
env/bin/physengine
```

The window opens on the **Home** screen, which describes each module and lists its saved scenarios. Choose a module with its Open button. To switch modules later, use the **Modules** box in the Simulation panel.

### Controls

- **Mouse:** left drag orbits; right or middle drag pans; the mouse wheel zooms.
- **Simulation panel:** Pause, Restart (keeps your edits), Full reset (restores every default), and Time Scale.
- **Guided tour:** F5, or the button on the Home screen, runs through every module on its own (about four minutes). Space holds it, Left and Right skip between stops, Esc ends it; dragging the view also holds it.
- **Scale bar:** the bar in the bottom-left corner gives the size of things at the distance of the point the camera looks at.

## Optional data

Two modules can use downloaded data. Both work without it, with less detail.

- **Atom:** a built-in excerpt of the AME2020 and NUBASE2020 nuclear tables is included. The **Download the tables** button in the Atom window fetches the full files from the IAEA into `data/atom/`.
- **Particles:** the CERN Open Data muon events (CC0 licence, record 545) are fetched with the download button in the Particles window into `data/opendata/`. The files are about 15 MB and are ignored by git.

## Saved scenarios

The massive-body, galaxy and black-hole modules can save their current setup as a JSON file in `saves/<module>/` at the repository root. The folder is created the first time you save. Saved scenarios also appear on the Home screen.

## Tests

```bash
env/bin/python -m pytest
```

The full suite takes several minutes. Run one file with, for example, `env/bin/python -m pytest tests/test_particles.py`. Tests that need the downloaded data are skipped if it is missing.

## Project layout

- `src/physengine/core/`: the window, camera, controls, parameter handling and the simulation loop.
- `src/physengine/modules/`: one module per simulation, with its physics and its window.
- `tests/`: the tests.
- `data/`: downloaded or bundled input data.
- `saves/`: scenarios you have saved.

## Troubleshooting

- **The window does not open, or ImGui reports "no valid context":** check the OpenGL version your driver reports (for example with `glxinfo | grep "OpenGL version"`).
- **The window looks stuck on a hidden or covered screen:** frame times of about one second mean the compositor is throttling the window. Bring it to the front.
- **The Atom or Particles dropdowns look wrong after an update:** restart the app. Scene lists are read when the app starts.

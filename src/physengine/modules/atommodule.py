"""The atom: electron clouds, the nucleus, radioactive decay, fission and fusion.

Written by Claude under a one-off override of CLAUDE.md (2026-11-23, atom
module). Review the physics.

One module, nine scenes (atom_scenes.SCENES), from the scale of the atom
(1e-10 m, lengths in Bohr radii) to that of the nucleus (1e-15 m, lengths in
fm). Each scene has its own length unit; the scene 'Atom: shells and
nucleus' holds both scales at once, with the nucleus at its true relative
size (zoom in by a factor 1e5) or magnified, which is labelled as display
only.

Computed from the physics, per scene (details in the physics files):
- orbitals: exact hydrogen-like wave functions, level energies and line
  wavelengths (atom_orbitals);
- nucleus: measured masses and binding energies (AME2020), half-lives and
  decay modes (NUBASE2020), the semi-empirical mass formula (atom_data);
- decays: Q-values from the masses, kinematics with recoil, the allowed beta
  spectrum, exact exponential decay statistics, the Bateman solution
  (atom_decay);
- fission and fusion: energy release from the masses, exact kinematics, the
  Coulomb barrier and the Gamow tunnelling factor (atom_reactions).
Approximate models, named where they are used: Slater's rules for
many-electron atoms; the fission yield, neutron number and neutron spectrum
parametrisations; the chain reaction (a schematic one-speed model).
Illustration only: where nucleons are drawn, how fast things move on screen,
the shapes of a nucleus that splits or fuses. The panel says so in every
scene.

Structure: the scenes (atom_scenes, NumPy only) hold the simulation state;
AtomRenderer draws what they describe; this class is the glue and the
interface (panels in atom_panels).
"""

import threading
import time

import numpy as np
from imgui_bundle import imgui

from physengine.core.module import BaseModule
from physengine.core.scalebar import draw_scale_bar, length_text
from physengine.core.spec import Choice, SliderFloat
from physengine.modules import atom_data, atom_panels
from physengine.modules import atom_scenes as scenes
from physengine.modules.atom_renderer import CUT_MODES, AtomRenderer

WINDOW_MARGIN = 10.0
CLICK_SLOP = 4.0  # a press and release closer than this is a click, not a drag
PICK_PIXELS = 9.0
LOG_SHOWN = 4
ZOOM_SECONDS = 1.6  # time constant of the animated zoom


class AtomModule(BaseModule):
    dt = 1.0 / 60.0  # seconds of animation per step

    def __init__(self) -> None:
        self.scene_name = scenes.DEFAULT_SCENE
        self.labels = "Shown"
        self.cloud_view = "Whole"
        self.cloud_brightness = 1.0
        self.specs = (
            Choice(
                attr_name="scene_name",
                label="Scene",
                default=scenes.DEFAULT_SCENE,
                options=tuple(scenes.SCENES),
                need_reset=True,
            ),
            Choice(
                attr_name="cloud_view",
                label="Cloud view",
                default="Whole",
                options=tuple(CUT_MODES),
            ),
            SliderFloat(
                attr_name="cloud_brightness",
                label="Cloud brightness",
                default=1.0,
                min=0.2,
                max=4.0,
            ),
            Choice(
                attr_name="labels", label="Labels", default="Shown", options=("Shown", "Hidden")
            ),
        )
        self.renderer = AtomRenderer()
        self.rng = np.random.default_rng()
        self.scene = None
        self.built_name = self.scene_name
        self.camera = None
        self.cloud_version = -1
        self.view_version = -1
        self.pending_view = True
        self.selected = None  # (title, [lines]) of what was clicked
        self.shown = None  # what the selection belongs to (see render)
        self.press = None
        self.state = {}  # the panels' own widget state (text boxes, zoom request)
        self.last_render = None
        self.downloader = None
        self.download_error = ""
        self.downloaded = False

    # ------------------------------------------------------------- lifecycle

    def setup(self, ctx):
        self.table = atom_data.table()
        self.renderer.setup(ctx)
        self.scene = scenes.SCENES[self.scene_name](self.table, self.rng)
        # The Scene spec changes scene_name a frame before the rebuild; the
        # panels must follow the scene that exists, not the one asked for.
        self.built_name = self.scene_name

    def step(self, dt):
        self.scene.step(dt)

    def render(self, ctx, camera):
        self.camera = camera
        scene = self.scene
        now = time.perf_counter()
        frame = 0.0 if self.last_render is None else min(now - self.last_render, 0.1)
        self.last_render = now
        if self.pending_view:
            camera.distance, camera.yaw, camera.pitch = scene.view
            self.pending_view = False
            self.view_version = scene.view_version
        elif self.view_version != scene.view_version:
            # The thing shown changed size: the camera is left as it is.
            self.view_version = scene.view_version
        target = self.state.get("zoom_to")
        if target is not None:
            # Equal steps in the logarithm: the same pace across five orders of magnitude.
            gap = np.log(target / camera.distance)
            if abs(gap) < 0.01:
                camera.distance = target
                self.state["zoom_to"] = None
            else:
                camera.distance *= float(np.exp(gap * min(frame * 4.0 / ZOOM_SECONDS, 1.0)))
        shown = (scene.view_version, getattr(scene, "events", 0), scene.cloud_version)
        if shown != self.shown:
            self.shown = shown
            self.selected = None  # what was clicked is no longer there
        if self.cloud_version != scene.cloud_version:
            self.renderer.set_cloud(scene.cloud())
            self.cloud_version = scene.cloud_version
        scene.work()
        balls, self.tags = scene.balls()
        self.balls = balls
        self.renderer.render(
            ctx,
            camera,
            balls,
            scene.streaks(),
            cut=self.cloud_view,
            slab=0.05 * getattr(scene, "extent", 1.0),
            cloud_brightness=0.16 * self.cloud_brightness,
            cloud_max_size=scene.cloud_size_limit,
        )

    def teardown(self):
        self.renderer.release()

    def discard_edits(self):
        scenes.reset_settings()

    # --------------------------------------------------------------- picking

    def _project(self, points, W, H):
        clip = np.hstack([points, np.ones((len(points), 1))]) @ self.camera.view_proj().T
        front = clip[:, 3] > 0.0
        w = np.where(front, clip[:, 3], 1.0)
        pixels = np.stack([(clip[:, 0] / w + 1.0) * 0.5 * W, (1.0 - clip[:, 1] / w) * 0.5 * H], 1)
        return pixels, front, w

    def _pick(self, mouse, W, H):
        """What is under the mouse: the nearest ball in front, or the cloud."""
        balls = self.balls
        if len(balls):
            pixels, front, w = self._project(balls[:, 0:3], W, H)
            # Radius on screen: world radius times pixels per unit at that depth.
            scale = self.camera.projection()[1, 1] * 0.5 * H
            reach = np.maximum(balls[:, 3] * scale / w, PICK_PIXELS)
            distance = np.linalg.norm(pixels - mouse, axis=1)
            hit = front & (distance <= reach) & (self.tags >= 0)
            if hit.any():
                candidates = np.flatnonzero(hit)
                best = candidates[np.argmin(w[candidates])]  # the one nearest the camera
                return self.scene.info_for(int(self.tags[best]))
        return None

    def _handle_click(self, W, H):
        io = imgui.get_io()
        mouse = np.array([io.mouse_pos.x, io.mouse_pos.y])
        over_ui = (
            imgui.is_window_hovered(imgui.HoveredFlags_.any_window) or imgui.is_any_item_active()
        )
        if imgui.is_mouse_clicked(0):
            self.press = None if over_ui else mouse
        if imgui.is_mouse_released(0) and self.press is not None:
            if np.linalg.norm(mouse - self.press) <= CLICK_SLOP:
                self.selected = self._pick(mouse, W, H)
            self.press = None

    # -------------------------------------------------------------------- UI

    def _draw_overlay(self, W, H):
        scene, camera = self.scene, self.camera
        draw_list = imgui.get_background_draw_list()
        if self.labels == "Shown":
            for where, text, rgb in scene.labels():
                at = camera.world_to_screen(where, W, H)
                if at is not None and text:
                    draw_list.add_text(
                        (at[0] + 8.0, at[1] - 8.0), imgui.get_color_u32((*rgb, 1.0)), text
                    )
        if scene.unit_metres:
            draw_scale_bar(camera, W, H, scene.unit_metres, scene.unit)

    def _draw_data_source(self):
        table = self.table
        if table.from_files:
            imgui.text_disabled("Data: AME2020 masses, NUBASE2020 half-lives (IAEA AMDC)")
            return
        atom_panels.note(
            f"Data: a built-in excerpt of {len(table.nuclides)} nuclides (from AME2020 and"
            " NUBASE2020); every other mass comes from the semi-empirical formula, good to"
            " a few MeV. The full tables (1.2 MB) can be fetched from the IAEA."
        )
        busy = self.downloader is not None and self.downloader.is_alive()
        if busy:
            imgui.text_disabled("Downloading...")
        elif imgui.button("Download the tables (www-nds.iaea.org)"):
            self.download_error = ""

            def work():
                try:
                    atom_data.download()
                    self.downloaded = True
                except (OSError, ValueError) as error:
                    self.download_error = f"Download failed: {error}"

            self.downloader = threading.Thread(target=work, daemon=True)
            self.downloader.start()
        if self.download_error:
            atom_panels.wrapped(self.download_error, atom_panels.WARN)
        if self.downloaded:
            self.downloaded = False
            self.reset_requested = True  # rebuild with the full tables

    def draw_ui(self):
        io = imgui.get_io()
        W, H = io.display_size.x, io.display_size.y
        scene = self.scene
        if self.camera is not None and W > 0 and H > 0:
            self._handle_click(W, H)
            self._draw_overlay(W, H)

        # Top right; in a window too narrow for both, below the Simulation
        # window, so that this one does not cover the Scene selector.
        top = WINDOW_MARGIN
        sim = imgui.internal.find_window_by_name("Simulation")
        if sim is not None and W - WINDOW_MARGIN - (atom_panels.WRAP + 40.0) < (
            sim.pos.x + sim.size.x + WINDOW_MARGIN
        ):
            top = sim.pos.y + sim.size.y + WINDOW_MARGIN
        imgui.set_next_window_pos((W - WINDOW_MARGIN, top), imgui.Cond_.always, (1.0, 0.0))
        imgui.set_next_window_size_constraints(
            (0.0, 0.0), (atom_panels.WRAP + 40.0, max(H - top - WINDOW_MARGIN, 1.0))
        )
        flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move
        if not imgui.begin("Atom", None, flags)[0]:
            imgui.end()
            return
        imgui.text_colored((0.75, 0.85, 1.0, 1.0), scene.title)
        # What this scene computes and what it only illustrates, always in view.
        atom_panels.note(scene.disclaimer)
        lines = scene.log[-LOG_SHOWN:]
        for k, text in enumerate(lines):
            if k == len(lines) - 1:
                atom_panels.wrapped(text, (1.0, 1.0, 1.0, 1.0))
            else:
                atom_panels.note(text)
        imgui.separator()
        atom_panels.PANELS[self.built_name](scene, self.state)
        imgui.separator()
        if self.selected is not None:
            title, lines = self.selected
            imgui.text_colored((1.0, 0.9, 0.5, 1.0), title)
            for text in lines:
                atom_panels.wrapped(text)
        else:
            imgui.text_disabled("Click a nucleon or particle for information")
        legend = atom_panels.LEGENDS.get(self.built_name, atom_panels.LEGENDS["nuclear"])
        if imgui.collapsing_header("Legend"):
            for color, text in legend:
                atom_panels.swatch(color, text)
        self._draw_data_source()
        imgui.end()

    def diagnostics(self):
        scene = self.scene
        lines = [f" Scene: {scene.title}; lengths in {scene.unit}"]
        if self.camera is not None and scene.unit_metres:
            height = 2.0 * self.camera.distance * np.tan(self.camera.fov / 2.0)
            lines.append(f" View height: {length_text(height * scene.unit_metres)}")
        lines.append(f" Drawn: {len(getattr(self, 'balls', []))} balls,"
                     f" {self.renderer.cloud_count} cloud points")  # fmt: skip
        lines.append(
            f" Data: {'AME2020 / NUBASE2020 files' if self.table.from_files else 'excerpt'}"
        )
        return "\n".join(lines)

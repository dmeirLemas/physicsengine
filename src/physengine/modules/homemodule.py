"""The home screen: what the application contains, and a way into each module.

UI only. Opening a module (or a saved scenario) asks the app to switch to it
through BaseModule.switch_requested.
"""

from imgui_bundle import imgui

from physengine.core.module import BaseModule
from physengine.modules import scenario_files
from physengine.modules.starfield import Starfield

# (registry name, kind of its saved scenarios, title, what it shows)
MODULES = (
    (
        "NbodyModule",
        "nbody",
        "Planets, stars and black holes",
        "A few bodies under gravity, with relativity: the solar system, binary and triple"
        " stars, orbits round a black hole, collisions that shatter planets, stars torn"
        " apart, and two black holes spiralling together with their gravitational waves."
        " Build your own system with the mouse.",
    ),
    (
        "Galaxies",
        "galaxy",
        "Colliding galaxies",
        "Two disc galaxies of stars in dark-matter halos pass through each other and merge,"
        " drawing out tidal tails. Choose the orbit and the tilt of each disc.",
    ),
    (
        "Black hole",
        "blackhole",
        "A black hole, as it would look",
        "Light bent round a black hole and its accretion disc, ray-traced, with or without"
        " spin. Bodies on exact relativistic orbits, a star pulled apart, and a ring of gas"
        " draining in.",
    ),
    (
        "Particles",
        None,
        "Particle decays in a detector",
        "Z and Higgs bosons decaying, with exact relativistic kinematics and measured"
        " branching ratios: tracks curling in the detector's magnetic field, and the"
        " invariant-mass peaks that reveal each particle. The proton collision itself is"
        " not simulated; real events recorded by CMS at CERN (open data) are included.",
    ),
    (
        "Atom",
        None,
        "The atom and its nucleus",
        "Electron clouds of hydrogen-like orbitals and the shells of heavier atoms; the"
        " nucleus with measured masses and binding energies; radioactive decay, one nucleus"
        " at a time, as a sample, and through whole decay chains; fission, a schematic chain"
        " reaction, and fusion with its tunnelling barrier. Energies, half-lives and"
        " statistics are computed from measured data; the pictures of nucleons are"
        " illustrations.",
    ),
)
WIDTH = 720.0


class HomeModule(BaseModule):
    def __init__(self) -> None:
        self.specs = ()
        self.background = Starfield()
        self.errors = {}  # (kind, name) -> why that saved scenario could not be opened

    def setup(self, ctx):
        self.background.setup(ctx)

    def step(self, dt):
        return

    def render(self, ctx, camera):
        camera.yaw += 0.0004  # a slow drift of the stars behind the menu
        self.background.render(ctx, camera)

    def teardown(self):
        self.background.release()

    def open_saved(self, registry_name, kind, name):
        """Open the saved scenario `name` in its module. Returns whether it could.

        The file is read here, before the switch: one that cannot be read, or
        is not a scenario of this kind, stays on the home screen with the
        reason in self.errors. What the scenario holds is checked by the
        module when it opens.
        """
        try:
            data = scenario_files.load(kind, name)
        except (OSError, ValueError) as error:
            self.errors[(kind, name)] = str(error) or type(error).__name__
            return False
        self.errors.pop((kind, name), None)
        scenario_files.request(kind, data)
        self.switch_requested = registry_name
        return True

    def draw_ui(self):
        io = imgui.get_io()
        W, H = io.display_size.x, io.display_size.y
        width = min(WIDTH, max(W - 40.0, 200.0))
        imgui.set_next_window_pos((W / 2.0, H / 2.0), imgui.Cond_.always, (0.5, 0.5))
        imgui.set_next_window_size_constraints((width, 0.0), (width, max(H - 20.0, 1.0)))
        flags = (
            imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_collapse
        )
        imgui.begin("Physics engine", None, flags)
        imgui.text_disabled("Five physics modules in one window. Pick one to open it.")
        imgui.text_disabled("Every module starts paused: untick Pause (or press Space) to run it.")
        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + width - 30.0)
        for registry_name, kind, title, text in MODULES:
            imgui.separator()
            imgui.text_colored((0.75, 0.88, 1.0, 1.0), title)
            imgui.text_wrapped(text)
            if imgui.button(f"Open##{registry_name}"):
                self.switch_requested = registry_name
            names = scenario_files.saved(kind) if kind else []
            if names and imgui.tree_node(f"Saved scenarios ({len(names)})##{registry_name}"):
                for i, name in enumerate(names):
                    if imgui.small_button(f"Open '{name}'##{registry_name}{i}"):
                        self.open_saved(registry_name, kind, name)
                    error = self.errors.get((kind, name))
                    if error:
                        imgui.text_colored((1.0, 0.55, 0.4, 1.0), f"Cannot open: {error}")
                imgui.tree_pop()
        imgui.pop_text_wrap_pos()
        imgui.end()

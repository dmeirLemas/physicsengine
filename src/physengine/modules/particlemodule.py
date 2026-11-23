"""Particle physics: collider events shown in a detector, with invariant-mass reconstruction.

Written by Claude under a one-off override of CLAUDE.md (2026-11-22, module
4). Review the physics.

Each event is generated (particle_events.py) or taken from CMS open data
(particle_opendata.py); every particle is followed through the magnet and
the detector's material until it decays, showers, stops or leaves
(particle_tracks.py, particle_material.py), and the event is then played
back in slow motion: light crosses the tracker (1.2 m) in 4 ns. The masses
measured in each event fill a histogram.

Computed from the physics: relativistic kinematics, decays (measured masses,
lifetimes and branching fractions, energy and momentum conservation), and
motion in the field between interactions with the material.
Approximate models: how momenta are shared in most many-body decays (phase
space); the material (mean energy loss by the Bethe formula without
fluctuations, Gaussian multiple scattering by the Highland formula,
bremsstrahlung, parametrised showers); the measurement (true momenta smeared
by the resolution the Gluckstern formula gives for the field and the
tracker's size; nothing is fitted to the paths drawn).
Not simulated: the collision itself; see the module docstring of
particle_events.py. The incoming protons and the close-up diagram are an
illustration of the usual production mechanism, shown for generated events
only. For recorded events only CMS's measurement of the two muons is real;
their paths are recomputed in this module's simplified detector.

The "Custom" event type shows particles the user places (particle_custom.py,
edited in the window of particle_editor.py): such an event is not a
collision, no protons are drawn for it, and its texts say so.

Threads: the next event to show is prepared in a background thread, and so
are the events added to the histogram in bulk; the main thread only collects
their results (step). A failure in either is caught and shown in the panel.
"""

import threading
import time
from collections import deque

import numpy as np
from imgui_bundle import imgui

from physengine.core.module import BaseModule
from physengine.core.spec import Choice, SliderFloat
from physengine.modules import particle_custom, particle_opendata, scenario_files
from physengine.modules.particle_data import ABOUT, PARTICLES
from physengine.modules.particle_editor import CustomWindow, forget_draft, set_draft
from physengine.modules.particle_events import (
    CUSTOM,
    MECHANISMS,
    SCENES,
    Event,
    Hist,
    MassHistogram,
    generate,
    histogram_masses,
    quick_masses,
)
from physengine.modules.particle_kinematics import mass
from physengine.modules.particle_material import region
from physengine.modules.particle_renderer import ParticleRenderer
from physengine.modules.particle_tracks import (
    ECAL_RADIUS,
    HCAL_RADIUS,
    MUON_LAYERS,
    SOLENOID_RADIUS,
    TRACKER_RADIUS,
    C,
    helix_radius,
)
from physengine.modules.starfield import Starfield

DEFAULT_SCENE = "Z -> mu+ mu-"
LEAD_IN = 2.5  # ns before the collision at which playback starts (protons flying in)
CUSTOM_LEAD_IN = 0.4  # ns before a custom event's particles start (there are no protons)
NO_HISTOGRAM = Hist("", 0.0, 1.0, 1)  # stands in where a custom setup has no histogram
MAX_SHOWN = 30.0  # ns of an event that are played (light reaches the outer muon station in 24)
HOLD_SECONDS = 1.0  # the finished event stays this long
FADE_SECONDS = 0.5  # and then fades over this long
BATCH = 500  # events added by the "fill" button
FILL_CHUNK = 250  # recorded events taken from the file at a time when filling
WINDOW_MARGIN = 10.0
MAX_LABELS = 12  # particle names drawn in the view
LOG_LINES = 6  # lines of commentary shown
PLAYED_ENERGY = 0.97
PICK_PIXELS = 10.0  # a click this close to a track selects it
CLICK_SLOP = 4.0  # a press and release closer than this is a click, not a drag
LIST_ROWS = 28  # particles listed in the panel
M_PROTON = 0.93827209  # GeV
WRAP = 360.0  # pixels: width of the panel's text

# Track widths in pixels by kind, and brightness of neutral particles
# (which a tracker does not see; they are drawn faint, or not at all).
WIDTH = {
    "muon": 2.2,
    "electron": 2.0,
    "hadron": 1.5,
    "photon": 1.0,
    "neutral": 1.0,
    "neutrino": 0.8,
}
NEUTRAL_DIM = 0.3
EM_COLOR, HADRONIC_COLOR = (0.3, 1.0, 0.45), (0.35, 0.6, 1.0)
PROTON_COLOR = (0.75, 0.88, 1.0)

PLACE = {
    "tracker": "tracker",
    "ecal": "electromagnetic calorimeter",
    "hcal": "hadron calorimeter",
    "coil": "magnet coil",
    "yoke": "iron yoke",
}
LEGEND = (
    ((1.0, 0.25, 0.2), "muon: goes through everything"),
    ((0.3, 1.0, 0.4), "electron: showers in the first calorimeter"),
    ((1.0, 0.9, 0.3), "photon (faint: the tracker does not see it)"),
    ((0.6, 0.85, 1.0), "charged hadron (pion, kaon, proton)"),
    ((0.35, 0.5, 1.0), "neutral hadron (faint)"),
    ((0.5, 0.5, 0.5), "neutrino (faint: never detected)"),
    (EM_COLOR, "bar: electromagnetic shower"),
    (HADRONIC_COLOR, "bar: hadronic shower"),
)
DETECTOR_LABELS = (
    (TRACKER_RADIUS, "tracker"),
    (ECAL_RADIUS + 0.12, "electromagnetic calorimeter"),
    (HCAL_RADIUS + 0.5, "hadron calorimeter"),
    (SOLENOID_RADIUS + 0.15, "magnet coil"),
    (MUON_LAYERS[-1], "muon stations in the iron yoke"),
)
ENDINGS = {
    "decay": "decayed",
    "shower": "ended in a shower",
    "stopped": "ran out of energy and stopped",
    "conversion": "turned into an electron-positron pair",
    "exit": "left the detector",
    "lost": "kept curling in the magnet (no longer followed)",
}


def play_end(event):
    """Time [ns] up to which an event is played: when the particles carrying
    PLAYED_ENERGY of the energy have decayed, showered, stopped or left. (The
    rest are slow stragglers curling in the field for tens of nanoseconds.)"""
    leaves = [
        (float(t.times[-1]), float(t.p4_end[0]))
        for t in event.tracks
        if t.end not in ("decay", "conversion")
    ]
    if not leaves:
        return min(event.duration, MAX_SHOWN)
    leaves.sort()
    total = sum(e for _, e in leaves)
    running = 0.0
    for when, energy in leaves:
        running += energy
        if running >= PLAYED_ENERGY * total:
            return min(when + 0.5, MAX_SHOWN)
    return min(event.duration, MAX_SHOWN)


def proton_speed_deficit(beam_tev):
    """1 - v/c of a proton of `beam_tev` TeV: 1 / (2 gamma^2) for gamma >> 1."""
    gamma = beam_tev * 1000.0 / M_PROTON
    return 0.5 / gamma**2


def narrate(event):
    """The event in words: (time [ns], text) in the order things happen."""
    if event.error or not event.tracks:
        return []  # nothing happened: the panel says why
    tracks = event.tracks
    if event.custom:
        # No protons and no collision: only what the user put there.
        count = sum(1 for t in tracks if t.parent < 0)
        what = (
            "particle starts where you placed it"
            if count == 1
            else "particles start where you placed them"
        )
        log = [(0.0, f"{count} {what} (by hand: this is not a collision)")]
    else:
        deficit = proton_speed_deficit(event.beam_energy)
        log = [
            (
                -LEAD_IN,
                f"Two protons fly in, {event.beam_energy:g} TeV each:"
                f" slower than light by {deficit * 299792458.0:.2g} m/s",
            )
        ]
    if event.custom:
        pass
    elif event.real:
        # Only what was measured: no production mechanism is asserted.
        log.append((0.0, "The protons collide. Of all that came out, the file holds the two muons"))
    elif event.mechanism in MECHANISMS:
        log.append((0.0, MECHANISMS[event.mechanism][3]))
    elif event.tracks:
        log.append((0.0, "The protons collide"))
    for i, t in enumerate(tracks):
        particle = PARTICLES[t.name]
        when = float(t.times[-1])
        children = [c.name for c in tracks if c.parent == i and c.origin == "decay"]
        energy = float(t.p4[0])
        if t.end == "decay" and children and event.custom:
            # Distances are from where the particle itself started.
            distance = float(np.linalg.norm(t.points[-1] - t.points[0]))
            far = f"{distance * 100:.0f} cm" if distance >= 0.01 else f"{distance * 1000:.2g} mm"
            if particle.width > 0.0:
                how = f"(mass {float(mass(t.p4)):.4g} GeV) decays at once"
            elif distance == 0.0:
                how = "decays where it is"
            else:
                how = f"decays {far} from where it started"
            log.append((when, f"{t.name} {how} -> {' '.join(children)}"))
        elif t.end == "decay" and children and particle.width == 0.0:
            distance = float(np.linalg.norm(t.points[-1]))
            far = f"{distance * 100:.0f} cm" if distance >= 0.01 else f"{distance * 1000:.2g} mm"
            log.append((when, f"{t.name} decays {far} from the collision -> {' '.join(children)}"))
        elif t.end == "shower" and (t.parent < 0 or (energy > 3.0 and t.origin == "decay")):
            where = region(t.shower.start)
            place = PLACE.get(where.name, "detector") if where is not None else "detector"
            log.append((when, f"{t.name} ({energy:.3g} GeV) showers in the {place}"))
        elif t.end == "stopped" and particle.kind == "muon":
            log.append((when, f"{t.name} ({energy:.3g} GeV) runs out of energy and stops"))
        elif t.end == "conversion" and energy > 5.0:
            log.append((when, f"photon ({energy:.3g} GeV) converts to e+ e- in the tracker"))
        if t.hits:
            log.append((float(t.hits[0][1]), f"{t.name} reaches the muon stations"))
    end = play_end(event)
    for label, m in event.masses[:2]:
        # A recorded pair's label says who measured it; a generated one is
        # "measured" by the resolution model of particle_events.
        prefix = "mass of" if event.real else "measured (model tracker): mass of"
        log.append((end, f"{prefix} {label} = {m:.4g} GeV"))
    if event.note:
        log.append((end, event.note))
    return sorted(log, key=lambda entry: entry[0])


def _seconds(value):
    """A time in seconds with a readable unit."""
    units = (("s", 1.0), ("ms", 1e-3), ("microseconds", 1e-6), ("ns", 1e-9), ("ps", 1e-12))
    for unit, size in units:
        if value >= size:
            return f"{value / size:.3g} {unit}"
    return f"{value:.2g} s"


def _mass_text(m):
    return f"{m * 1000:.4g} MeV" if m < 0.5 else f"{m:.5g} GeV"


class ParticleModule(BaseModule):
    dt = 1.0 / 60.0  # wall seconds per step; `speed` ns of the event each second

    def __init__(self) -> None:
        self.scene_name = DEFAULT_SCENE
        self.field = 3.8
        self.speed = 8.0
        self.playback = "Continuous"
        self.labels = "Shown"
        self.neutrals = "Faint"
        self.detector = "Shown"
        self.specs = (
            Choice(
                attr_name="scene_name",
                label="Event type",
                default=DEFAULT_SCENE,
                options=tuple(SCENES),
                need_reset=True,
            ),
            Choice(
                attr_name="playback",
                label="Playback",
                default="Continuous",
                options=("Continuous", "One at a time"),
            ),
            SliderFloat(attr_name="speed", label="Speed [ns/s]", default=8.0, min=0.5, max=40.0),
            SliderFloat(attr_name="field", label="Magnet [T]", default=3.8, min=0.5, max=6.0),
            Choice(
                attr_name="labels", label="Labels", default="Shown", options=("Shown", "Hidden")
            ),
            Choice(
                attr_name="neutrals",
                label="Neutral particles",
                default="Faint",
                options=("Faint", "Hidden"),
            ),
            Choice(
                attr_name="detector", label="Detector", default="Shown", options=("Shown", "Hidden")
            ),
        )
        self.renderer = ParticleRenderer()
        self.background = Starfield()
        self.rng = np.random.default_rng()
        self.camera = None
        self.event = None
        self.log = []
        self.end = 0.0
        self.now = -LEAD_IN  # ns relative to the collision of the event shown
        self.finished_at = None  # wall time when the event finished playing
        self.pending_upload = False
        self.pending_view = True
        self.events_seen = 0
        self.waiting_for_data = False  # the event shown stands in for a missing file
        # Bulk filling of the histogram, done by a background thread:
        self.to_fill = 0  # events asked for and not yet in the histogram (for the display)
        self.fill_wanted = 0  # events the thread has not started on yet
        self.fill_lock = threading.Lock()  # guards fill_wanted
        self.fill_results = deque()  # (events, masses) or (0, error text), from the thread
        self.fill_cancel = threading.Event()
        self.filler = None
        self.fill_error = ""
        self.next_event = None  # prepared in the background while this one plays
        self.worker = None
        self.downloader = None
        self.download_error = ""
        self.selected = None  # ("track", index) or ("boson", index)
        self.press = None  # where the mouse went down, for telling a click from a drag
        # The Custom event type: the setup that runs (never changed in place;
        # the threads keep the one they started with), a number that goes up
        # with every change of it (results made for an older one are dropped),
        # and whether an event of the newest setup is still to be shown.
        self.custom = particle_custom.current()
        self.version = 0
        self.custom_dirty = False  # the setup changed and no event of it is being made yet
        self.custom_waiting = False  # an event of the new setup is being made
        self.notice = ""  # why a saved scenario could not be opened
        self.editor = CustomWindow(self)
        self.lead_in = LEAD_IN
        # Remade in setup(): the app sets the event type after construction.
        self.histogram = MassHistogram(self._histogram() or NO_HISTOGRAM)

    # ---------------------------------------------------------------- events

    def _histogram(self):
        """The scene's Hist; None for a custom setup without a parent to measure."""
        if self.scene_name == CUSTOM:
            return particle_custom.histogram_for(self.custom)
        return SCENES[self.scene_name][1]

    def _missing_data(self):
        name = self.scene_name
        return name in particle_opendata.FILES and not particle_opendata.available(name)

    def _make(self, field, rng, custom=None):
        """An event to show. Never raises: an event that cannot be made (an
        unreadable data file, a failure in the generation) comes back empty
        with the reason in `error`, which the panel shows."""
        if self._missing_data():
            return Event(headline="These recorded events have not been downloaded yet")
        try:
            if self.scene_name == CUSTOM:
                return generate(self.scene_name, field, rng, custom=custom or self.custom)
            return generate(self.scene_name, field, rng)
        except Exception as error:  # noqa: BLE001 (shown to the user, see draw_ui)
            return Event(
                headline="This event could not be made",
                error=f"{type(error).__name__}: {error}",
                custom=self.scene_name == CUSTOM,
            )

    def _prepare_next(self):
        """Generate the next event in the background, so that the display does not
        stall while a busy event is being followed through the detector."""
        field, rng, custom = self.field, self.rng.spawn(1)[0], self.custom

        def work():
            self.next_event = (field, self._make(field, rng, custom))

        self.next_event = None
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def new_event(self):
        if self.worker is not None:
            self.worker.join()  # ends: _make cannot raise, and every path is bounded
        if self.custom_dirty:
            self.next_event = None  # prepared for a custom setup that has changed since
        self.custom_dirty = self.custom_waiting = False
        if self.next_event is None:
            self.next_event = (self.field, self._make(self.field, self.rng))
        self.shown_field, self.event = self.next_event
        self.waiting_for_data = self._missing_data() and not self.event.tracks
        self.log = narrate(self.event)
        self.end = play_end(self.event)
        self.lead_in = CUSTOM_LEAD_IN if self.event.custom else LEAD_IN
        self.now = -self.lead_in
        self.finished_at = None
        self.selected = None
        self.pending_upload = True
        if self.event.tracks:
            self.histogram.add(histogram_masses(self.event))
            self.events_seen += 1
        self._prepare_next()

    # ------------------------------------------------------- custom setups

    def apply_custom(self, setup):
        """Run the custom `setup` from now on (called by the editor window).

        Raises ValueError, with nothing changed, if it cannot be generated.
        The histogram starts again; the event of the new setup is made in the
        background and shown when it is ready (_follow_custom), so a large
        setup does not stall the window while it is being edited.
        """
        particle_custom.set_current(setup)
        self.custom = particle_custom.current()
        self.version += 1
        self.cancel_fill()
        self.fill_error = ""
        self.histogram = MassHistogram(self._histogram() or NO_HISTOGRAM)
        self.events_seen = 0
        self.custom_dirty = True

    def _follow_custom(self):
        """Main thread, every frame: once the setup has changed, make an event of the
        new one (when the thread is free) and show it as soon as it is there."""
        if self.custom_dirty and (self.worker is None or not self.worker.is_alive()):
            self.custom_dirty = False
            self.custom_waiting = True
            self._prepare_next()  # drops what was prepared for the old setup
        elif self.custom_waiting and self.next_event is not None:
            self.new_event()

    def scenario(self):
        """The custom setup that is running and the field, for scenario_files."""
        return {
            "scene_name": CUSTOM,
            "field": float(self.field),
            "setup": particle_custom.to_dict(self.custom),
        }

    def prepare_scenario(self, data):
        """Take a saved scenario (kind "particles"): the Custom event type with the
        saved setup and field. Raises ValueError, with nothing changed, if
        `data` is not one."""
        if not isinstance(data, dict):
            raise ValueError("a particle scenario must be a dictionary")
        if data.get("scene_name", CUSTOM) != CUSTOM:
            raise ValueError("only the Custom event type can be saved and opened")
        value = data.get("field", self.field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("the magnet's field must be a number")
        if not 0.5 <= value <= 6.0:  # the range of the Magnet slider
            raise ValueError("the magnet's field must be between 0.5 and 6 tesla")
        setup = particle_custom.from_dict(data.get("setup"))
        particle_custom.set_current(setup)
        set_draft(setup)
        self.scene_name = CUSTOM
        self.field = float(value)
        self.custom = particle_custom.current()

    def load_scenario(self, data):
        """Open a saved scenario while the module runs (the editor's Load button)."""
        was_custom = self.scene_name == CUSTOM
        self.prepare_scenario(data)
        if was_custom:
            self.apply_custom(self.custom)
        else:
            self.reset_requested = True

    def discard_edits(self):
        """Full reset: the custom setup goes back to its default."""
        particle_custom.reset()
        forget_draft()

    # ---------------------------------------------------- filling in bulk

    def request_fill(self, count):
        """Ask for `count` more events in the histogram (made in the background)."""
        if self._histogram() is None:
            return  # a custom setup that measures no mass
        self.fill_error = ""
        self.to_fill += count
        with self.fill_lock:
            self.fill_wanted += count

    def cancel_fill(self):
        with self.fill_lock:
            self.fill_wanted = 0
        self.to_fill = 0

    def _start_filler(self):
        scene, rng = self.scene_name, self.rng.spawn(1)[0]
        chunk = FILL_CHUNK if scene in particle_opendata.FILES else 1
        # A custom setup may change while the thread runs: it keeps the one it
        # started with, stops when a newer one is applied, and tags its
        # results so that those made for an older setup are dropped.
        custom, version = self.custom, self.version
        extra = {"custom": custom} if scene == CUSTOM else {}

        def work():
            try:
                while not self.fill_cancel.is_set() and self.version == version:
                    with self.fill_lock:
                        count = min(self.fill_wanted, chunk)
                        self.fill_wanted -= count
                    if count <= 0:
                        return
                    masses = quick_masses(scene, count, self.field, rng, **extra)
                    self.fill_results.append((count, masses, version))
            except Exception as error:  # noqa: BLE001 (shown to the user)
                self.fill_results.append((0, f"{type(error).__name__}: {error}", version))

        self.filler = threading.Thread(target=work, daemon=True)
        self.filler.start()

    def _collect_fill(self):
        """Main thread: take what the filling thread has made into the histogram,
        and (re)start the thread if there is work and it is not running."""
        while self.fill_results:
            count, result, version = self.fill_results.popleft()
            if version != self.version:
                continue  # made for a custom setup that has changed since
            if isinstance(result, str):
                self.fill_error = result
                self.cancel_fill()
                continue
            self.histogram.add(result)
            self.events_seen += count
            self.to_fill = max(self.to_fill - count, 0)
        if self.fill_wanted > 0 and (self.filler is None or not self.filler.is_alive()):
            self._start_filler()

    def _segments(self):
        """The event as ribbon segments: protons, tracks, showers, muon hits, decay points."""
        starts, t0s, ends, t1s, colors = [], [], [], [], []

        def add(a, ta, b, tb, rgb, width):
            starts.append(np.atleast_2d(a))
            t0s.append(np.atleast_1d(ta))
            ends.append(np.atleast_2d(b))
            t1s.append(np.atleast_1d(tb))
            colors.append(np.tile([*rgb, width], (len(np.atleast_2d(a)), 1)))

        # The two protons coming in along the beam line (a negative width:
        # shown only where they have just been), and the flash where they meet.
        custom = self.event.custom  # no protons and no collision to draw
        times = np.linspace(-LEAD_IN, 0.0, 40)
        for side in (-1.0, 1.0) if not custom else ():
            z = side * C * -times
            pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)
            add(pts[:-1], times[:-1], pts[1:], times[1:], PROTON_COLOR, -4.0)
        if self.event.tracks and not custom:
            for k in range(12):
                angle = 2.0 * np.pi * k / 12
                tip = 0.22 * np.array([np.cos(angle), np.sin(angle), 0.5 * np.sin(3 * angle)])
                add(np.zeros(3), 0.0, tip, 0.25, (1.0, 0.95, 0.8), -3.0)

        for track in self.event.tracks:
            particle = PARTICLES[track.name]
            neutral = particle.charge == 0
            if not (neutral and self.neutrals == "Hidden"):
                rgb = np.array(particle.color) * (NEUTRAL_DIM if neutral else 1.0)
                if len(track.points) > 1:
                    add(
                        track.points[:-1], track.times[:-1], track.points[1:], track.times[1:],
                        rgb, WIDTH[particle.kind],
                    )  # fmt: skip
            if track.shower is not None:
                # The shower's energy along its depth: slices whose width
                # grows with the energy left in each.
                sh = track.shower
                depths, weights = sh.profile(8)
                half = 0.5 * (depths[1] - depths[0])
                tint = EM_COLOR if sh.kind == "em" else HADRONIC_COLOR
                scale = 2.0 + 2.2 * np.log1p(sh.energy / 0.3)
                for depth, weight in zip(depths, weights, strict=True):
                    a = sh.start + sh.direction * (depth - half)
                    b = sh.start + sh.direction * (depth + half)
                    t = sh.time + (depth - half) / C
                    add(a, t, b, t + 2.0 * half / C, tint, max(scale * np.sqrt(weight * 8.0), 1.0))
            for where, t in track.hits or ():
                # A muon station fires: a short bar along the station.
                tangent = np.array([-where[1], where[0], 0.0])
                tangent /= np.linalg.norm(tangent)
                add(where - 0.12 * tangent, t, where + 0.12 * tangent, t, (1.0, 0.35, 0.25), 4.0)
            if track.end == "decay" and (particle.width == 0.0 or custom) and len(track.points) > 1:
                # Where it decayed: a bright dot.
                where, t = track.points[-1], track.times[-1]
                add(where, t, where + np.array([0.0, 0.0, 0.004]), t, (1.0, 1.0, 1.0), 3.5)
            elif custom and len(track.points) == 1 and track.parent < 0:
                # A particle placed at rest: a dot in its colour where it sits.
                where, t = track.points[0], track.times[0]
                add(where, t, where + np.array([0.0, 0.0, 0.004]), t, particle.color, 4.0)
        if not starts:
            # Nothing to draw at all (a custom event of neutral particles that are hidden).
            add(np.zeros(3), -1.0e6, np.zeros(3), -1.0e6, (0.0, 0.0, 0.0), 0.0)
        return (
            np.concatenate(starts),
            np.concatenate(t0s),
            np.concatenate(ends),
            np.concatenate(t1s),
            np.concatenate(colors),
        )

    # ------------------------------------------------------------- lifecycle

    def setup(self, ctx):
        pending = scenario_files.take_request("particles")  # opened from the home screen
        if pending is not None:
            try:
                self.prepare_scenario(pending)
            except Exception as error:  # nothing may escape setup(): it would end the app
                self.notice = f"Could not open the saved scenario: {error}"
        self.custom = particle_custom.current()
        self.histogram = MassHistogram(self._histogram() or NO_HISTOGRAM)
        self.renderer.setup(ctx)
        self.background.setup(ctx)
        self.new_event()

    def step(self, dt):
        self._collect_fill()
        if self.now < self.end:
            self.now = min(self.now + dt * self.speed, self.end)
            return
        if self.playback != "Continuous" or not self.event.tracks:
            return  # waits for "Next event" (or for the data)
        if self.finished_at is None:
            self.finished_at = time.perf_counter()
        elif time.perf_counter() - self.finished_at > HOLD_SECONDS + FADE_SECONDS:
            # (else: still being prepared; or about to be remade for a changed custom setup)
            if self.next_event is not None and not (self.custom_dirty or self.custom_waiting):
                self.new_event()

    def render(self, ctx, camera):
        self.camera = camera
        if self.pending_view:
            camera.distance, camera.yaw, camera.pitch = 15.0, 0.85, 0.3
            self.pending_view = False
        if self.pending_upload:
            self.renderer.set_event(*self._segments())
            self.pending_upload = False
        fade = 1.0
        if self.finished_at is not None:
            over = time.perf_counter() - self.finished_at - HOLD_SECONDS
            fade = float(np.clip(1.0 - over / FADE_SECONDS, 0.0, 1.0))
        self.background.render(ctx, camera)
        self.renderer.render(ctx, camera, self.now, fade, self.detector == "Shown")

    def teardown(self):
        self.fill_cancel.set()
        self.cancel_fill()
        for thread in (self.worker, self.filler):
            if thread is not None:
                thread.join()
        self.renderer.release()
        self.background.release()

    # -------------------------------------------------------------- picking

    def _project(self, points, W, H):
        """Pixel positions of world points (n, 3), and whether each is in front of the camera."""
        points = np.atleast_2d(points)
        clip = np.hstack([points, np.ones((len(points), 1))]) @ self.camera.view_proj().T
        front = clip[:, 3] > 0.0
        w = np.where(front, clip[:, 3], 1.0)
        pixels = np.stack([(clip[:, 0] / w + 1.0) * 0.5 * W, (1.0 - clip[:, 1] / w) * 0.5 * H], 1)
        return pixels, front

    def _travelled(self, track):
        """The part of a track's path already covered at the time shown."""
        k = int(np.searchsorted(track.times, self.now, side="right"))
        return track.points[: max(k, 1)]

    def _pick(self, mouse, W, H):
        """What is under the mouse: a track, or a boson at the collision point."""
        # The collision point first: every track starts there, and it is the
        # only place to click a boson, which has no track of its own.
        if self.event.bosons and self.now >= 0.0:
            origin, front = self._project(np.zeros(3), W, H)
            if front[0] and np.linalg.norm(origin[0] - mouse) < PICK_PIXELS:
                # Clicking it again goes to the next boson.
                k = self.selected[1] + 1 if self.selected and self.selected[0] == "boson" else 0
                return ("boson", k % len(self.event.bosons))
        best, best_distance = None, PICK_PIXELS
        for i, track in enumerate(self.event.tracks):
            if track.times[0] > self.now:
                continue
            if PARTICLES[track.name].charge == 0 and self.neutrals == "Hidden":
                continue
            pixels, front = self._project(self._travelled(track), W, H)
            if not front.any():
                continue
            pixels = pixels[front]
            # Distance to the path's segments, not only its points.
            if len(pixels) == 1:
                distance = float(np.linalg.norm(pixels[0] - mouse))
            else:
                a, b = pixels[:-1], pixels[1:]
                ab = b - a
                along = np.sum((mouse - a) * ab, 1) / np.maximum(np.sum(ab * ab, 1), 1e-9)
                nearest = a + np.clip(along, 0.0, 1.0)[:, None] * ab
                distance = float(np.min(np.linalg.norm(nearest - mouse, axis=1)))
            if distance < best_distance:
                best, best_distance = ("track", i), distance
        return best

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
        """Names next to the particles and the detector's layers, and the selection."""
        camera = self.camera
        draw_list = imgui.get_background_draw_list()
        if self.selected is not None:
            white = imgui.get_color_u32((1.0, 1.0, 1.0, 0.9))
            if self.selected[0] == "track":
                path = self._travelled(self.event.tracks[self.selected[1]])
                pixels, front = self._project(path[:: max(len(path) // 150, 1)], W, H)
                for a, b, fa, fb in zip(
                    pixels[:-1], pixels[1:], front[:-1], front[1:], strict=True
                ):
                    if fa and fb:
                        draw_list.add_line(tuple(a), tuple(b), white, 3.0)
                head, front = self._project(path[-1], W, H)
                if front[0]:
                    draw_list.add_circle(tuple(head[0]), 9.0, white, 0, 2.0)
            else:
                origin, front = self._project(np.zeros(3), W, H)
                if front[0]:
                    draw_list.add_circle(tuple(origin[0]), 12.0, white, 0, 2.0)
        if self.labels != "Shown":
            return
        if self.detector == "Shown":
            dim = imgui.get_color_u32((0.55, 0.6, 0.7, 0.75))
            for radius, text in DETECTOR_LABELS:
                at = camera.world_to_screen((0.0, radius, 0.0), W, H)
                if at is not None:
                    draw_list.add_text((at[0] + 4.0, at[1] - 14.0), dim, text)
        if self.now < 0.0 and not self.event.custom:
            # The incoming protons.
            for side in (-1.0, 1.0):
                at = camera.world_to_screen((0.0, 0.0, side * C * -self.now), W, H)
                if at is not None:
                    draw_list.add_text(
                        (at[0] + 8.0, at[1] - 16.0),
                        imgui.get_color_u32((*PROTON_COLOR, 1.0)),
                        f"proton  {self.event.beam_energy:g} TeV",
                    )
        # The most energetic particles that have appeared and are worth naming.
        shown = []
        for track in self.event.tracks:
            particle = PARTICLES[track.name]
            if track.times[0] > self.now or particle.width > 0.0:
                continue
            if particle.charge == 0 and self.neutrals == "Hidden":
                continue
            if track.origin == "radiated":
                continue
            if len(track.points) < 2 and not (self.event.custom and track.parent < 0):
                continue  # (a custom particle placed at rest is named where it sits)
            shown.append(track)
        shown.sort(key=lambda t: -t.p4[0])
        for track in shown[:MAX_LABELS]:
            at = camera.world_to_screen(self._travelled(track)[-1], W, H)
            if at is None:
                continue
            r, g, b = PARTICLES[track.name].color
            pt = float(np.hypot(track.p4[1], track.p4[2]))
            text = f"{track.name}  {pt:.3g} GeV" if pt >= 1.0 else track.name
            draw_list.add_text(
                (at[0] + 8.0, at[1] - 6.0), imgui.get_color_u32((r, g, b, 1.0)), text
            )

    def _draw_diagram(self):
        """The collision at the scale of quarks and gluons, as a diagram (an illustration)."""
        if self.event.real:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
            imgui.text_disabled(
                "A recorded event: which process made these two muons is not known event"
                " by event. Only the muons were measured."
            )
            imgui.pop_text_wrap_pos()
            return
        if self.event.mechanism not in MECHANISMS:
            imgui.text_disabled("No single process to draw for this event.")
            return
        a, b, made, text = MECHANISMS[self.event.mechanism]
        imgui.dummy((WRAP, 96.0))
        corner = imgui.get_item_rect_min()
        draw_list = imgui.get_window_draw_list()
        x0, y0 = corner.x, corner.y
        blue = imgui.get_color_u32((*PROTON_COLOR, 1.0))
        white = imgui.get_color_u32((1.0, 1.0, 1.0, 1.0))
        warm = imgui.get_color_u32((1.0, 0.8, 0.3, 1.0))
        grey = imgui.get_color_u32((0.6, 0.6, 0.65, 1.0))
        left, right = (x0 + 130.0, y0 + 48.0), (x0 + 215.0, y0 + 48.0)
        # Two protons (three quarks each); one constituent of each goes in.
        for sign, label in ((-1.0, a), (1.0, b)):
            centre = (x0 + 28.0, y0 + 48.0 + sign * 30.0)
            draw_list.add_circle(centre, 15.0, blue, 0, 1.5)
            for dx, dy in ((-5.0, -4.0), (6.0, -2.0), (0.0, 6.0)):
                draw_list.add_circle_filled((centre[0] + dx, centre[1] + dy), 2.5, blue)
            draw_list.add_line((centre[0] + 15.0, centre[1]), left, white, 1.5)
            draw_list.add_text((x0 + 60.0, centre[1] - (16.0 if sign < 0 else -3.0)), white, label)
        draw_list.add_text((x0 + 48.0, y0 + 41.0), grey, "protons")
        draw_list.add_line(left, right, warm, 2.5)
        draw_list.add_text((left[0] + 28.0, left[1] - 18.0), warm, made)
        outgoing = [t.name for t in self.event.tracks if t.parent < 0][:2]
        if self.event.bosons:
            outgoing = list(self.event.bosons[0]["daughters"])
        for k, name in enumerate(outgoing[:2]):
            tip = (x0 + 300.0, y0 + 48.0 + (-30.0 if k == 0 else 30.0))
            draw_list.add_line(right, tip, white, 1.5)
            draw_list.add_text((tip[0] + 6.0, tip[1] - 7.0), white, name)
        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
        imgui.text_wrapped(text)
        imgui.text_disabled(
            "Not to scale and not simulated: a proton is 1e-15 m across, and this is"
            " the usual mechanism as known from theory and experiment, not something"
            " the module computes."
        )
        imgui.pop_text_wrap_pos()

    def _draw_selected(self):
        """Everything about the selected particle."""
        kind, index = self.selected
        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
        if kind == "boson":
            boson = self.event.bosons[index]
            name, p4 = boson["name"], boson["p4"]
            particle = PARTICLES[name]
            title, about = ABOUT[name]
            imgui.text_colored((1.0, 0.9, 0.5, 1.0), f"{title}  ({name})")
            imgui.text_wrapped(about)
            m = float(mass(p4))
            imgui.text(f"Mass in this event: {m:.5g} GeV (usual value {particle.mass:.5g})")
            sideways = float(np.hypot(p4[1], p4[2]))
            imgui.text(f"Energy {p4[0]:.4g} GeV, sideways momentum {sideways:.3g} GeV")
            if boson["from"] == "the collision":
                made = (
                    "Put at the collision point by the generator (its production is not simulated)."
                )
            else:
                made = f"Made by the decay of {boson['from']}."
            imgui.text_wrapped(
                f"{made} It decays at once into {' and '.join(boson['daughters'])}"
                " and leaves no track: it is known only from the combined mass of what it decays to."
            )
            imgui.pop_text_wrap_pos()
            return
        tracks = self.event.tracks
        track = tracks[index]
        particle = PARTICLES[track.name]
        title, about = ABOUT.get(track.name, (track.name, ""))
        imgui.text_colored((*particle.color, 1.0), f"{title}  ({track.name})")
        imgui.text_wrapped(about)
        charge = "none" if particle.charge == 0 else f"{particle.charge:+d} e"
        if particle.ctau is None:
            life = "stable (here)"
        elif particle.ctau == 0.0:
            life = "decays at once"
            if particle.width > 0.0:
                life += f" (mass in this event {float(mass(track.p4)):.5g} GeV)"
        else:
            life = f"mean life {_seconds(particle.ctau / 299792458.0)} at rest"
        imgui.text(f"Mass {_mass_text(particle.mass)}   charge {charge}")
        imgui.text(life)
        energy, momentum = float(track.p4[0]), float(np.linalg.norm(track.p4[1:]))
        pt = float(np.hypot(track.p4[1], track.p4[2]))
        imgui.separator()
        imgui.text(f"Energy {energy:.4g} GeV, momentum {momentum:.4g} GeV (sideways {pt:.3g})")
        if particle.mass > 0.0:
            gamma = energy / particle.mass
            beta = momentum / energy
            if beta < 0.9999:
                speed = f"{beta:.4f} c"
            else:
                speed = f"c less {(1.0 - beta) * 299792458.0:.3g} m/s"
            imgui.text(f"Speed {speed};  its clock runs {gamma:.4g} times slow")
            if particle.ctau:
                flight = momentum / particle.mass * particle.ctau
                imgui.text(f"So on average it flies {flight:.3g} m before decaying")
        else:
            imgui.text("Speed: that of light")
        if particle.charge != 0 and self.shown_field > 0.0 and pt > 0.0:
            radius = helix_radius(pt, particle.charge, self.shown_field)
            imgui.text(f"Curls with radius {radius:.3g} m in the magnet")
        custom = self.event.custom
        if custom and track.parent < 0:
            imgui.text_wrapped(
                "Placed here by hand (custom setup): not something a collision is"
                " claimed to produce."
            )
        if track.parent < 0:
            source = "the collision" + (" (recorded by CMS)" if self.event.real else "")
            for boson in self.event.bosons:
                if track.name in boson["daughters"] and boson["name"] != "H":
                    source = f"the decay of the {ABOUT[boson['name']][0]}"
        else:
            how = {
                "decay": "the decay of",
                "radiated": "radiation from",
                "conversion": "the conversion of",
            }
            source = f"{how.get(track.origin, 'the')} {tracks[track.parent].name}"
        if not (custom and track.parent < 0):
            imgui.text_wrapped(f"Came from {source}.")
        ending = ENDINGS.get(track.end, track.end)
        where = float(np.linalg.norm(track.points[-1]))
        place = region(track.points[-1])
        inside = place is not None and track.end != "exit"
        place_text = f" in the {PLACE[place.name]}" if inside else ""
        children = [c.name for c in tracks if c.parent == index and c.origin == "decay"]
        to = f" into {' '.join(children)}" if children and track.end == "decay" else ""
        if custom:
            imgui.text_wrapped(
                f"It {ending}{to}{place_text}, {where:.3g} m from the centre of the detector,"
                f" {track.times[-1]:.3g} ns after the start."
            )
        else:
            imgui.text_wrapped(
                f"It {ending}{to}{place_text}, {where:.3g} m from the collision,"
                f" {track.times[-1]:.3g} ns after it."
            )
        if track.ionisation > 0.01:
            imgui.text(f"Lost {track.ionisation:.3g} GeV to the material on the way")
        imgui.pop_text_wrap_pos()

    def _draw_particle_list(self):
        """The event's particles as a list to click (also the way to the ones not drawn)."""
        tracks = self.event.tracks
        for k, boson in enumerate(self.event.bosons):
            chosen = self.selected == ("boson", k)
            label = f"{ABOUT[boson['name']][0]} (decays at once)##boson{k}"
            if imgui.selectable(label, chosen)[0]:
                self.selected = None if chosen else ("boson", k)
        depth = []
        for track in tracks:
            depth.append(0 if track.parent < 0 else depth[track.parent] + 1)
        for i, track in enumerate(tracks[:LIST_ROWS]):
            chosen = self.selected == ("track", i)
            text = f"{'  ' * depth[i]}{track.name}  {track.p4[0]:.3g} GeV##track{i}"
            if imgui.selectable(text, chosen)[0]:
                self.selected = None if chosen else ("track", i)
        if len(tracks) > LIST_ROWS:
            imgui.text_disabled(f"... and {len(tracks) - LIST_ROWS} more (click them in the view)")

    def _draw_histogram(self):
        hist = self._histogram()
        if hist is None:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
            imgui.text_disabled(
                "No histogram: this custom setup only fires particles, so there is no"
                " parent whose mass would be measured. Add a 'Decay a parent' entry for one."
            )
            imgui.pop_text_wrap_pos()
            return
        size = (WRAP, 90.0)
        counts = self.histogram.counts
        top = float(max(counts.max(), 1.0)) * 1.1
        shown = np.log10(counts + 1.0) if hist.log else counts
        scale_max = float(np.log10(top + 1.0)) if hist.log else top
        imgui.plot_histogram("##mass", shown.astype("f4"), 0, "", 0.0, scale_max, size)
        corner, far = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        draw_list = imgui.get_window_draw_list()
        gold = imgui.get_color_u32((1.0, 0.8, 0.3, 0.9))
        for value, name in hist.peaks:
            if hist.log:
                f = np.log(value / hist.low) / np.log(hist.high / hist.low)
            else:
                f = (value - hist.low) / (hist.high - hist.low)
            x = corner.x + f * (far.x - corner.x)
            draw_list.add_line((x, corner.y), (x, far.y), gold, 1.0)
            if len(hist.peaks) > 1:
                draw_list.add_text((x + 2.0, corner.y + 1.0), gold, name)
        imgui.text(f"{hist.low:g}")
        imgui.same_line(size[0] - 30.0)
        imgui.text(f"{hist.high:g}")
        imgui.text(hist.label + (" (counts too)" if hist.log else ""))
        # Median and width from the histogram's own counts (kept up to date as
        # masses are added; nothing is rescanned here).
        summary = self.histogram.summary()
        entries = self.histogram.entries
        if summary is None:
            imgui.text_disabled("no entries yet")
        elif hist.log:
            imgui.text(f"{entries} entries")
        else:
            # Width from the central 68% (robust against the long tails).
            _, median, half_width = summary
            imgui.text(f"{entries} entries: median {median:.4g}, +-{half_width:.3g} GeV")
        if self.scene_name == CUSTOM:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
            imgui.text_disabled(
                "Of the first 'Decay a parent' entry: everything visible that came from the"
                " parent. True momenta smeared by the model detector's resolution"
                " (tracker, about 1/B; calorimeter for electrons and photons)."
            )
            imgui.pop_text_wrap_pos()
        elif not self.event.real and self.scene_name != "Z -> e+ e-":
            imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
            imgui.text_disabled(
                "Momenta: true values smeared by a tracker's resolution, which follows the"
                " magnet's field (about 1/B); not fitted to the paths drawn."
            )
            imgui.pop_text_wrap_pos()
        if hist.peaks:
            known = ", ".join(f"{name} {value:.4g}" for value, name in hist.peaks)
            imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
            imgui.text_colored((1.0, 0.8, 0.3, 1.0), f"lines, known masses [GeV]: {known}")
            imgui.pop_text_wrap_pos()

    def _draw_download(self):
        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
        imgui.text_wrapped(
            "These are real collisions recorded by the CMS experiment in 2011, published on the"
            " CERN Open Data Portal (record 545). The files (15 MB) are not on this computer yet."
        )
        imgui.pop_text_wrap_pos()
        busy = self.downloader is not None and self.downloader.is_alive()
        if busy:
            imgui.text_disabled("Downloading...")
        elif imgui.button("Download from opendata.cern.ch"):
            self.download_error = ""

            def work():
                try:
                    particle_opendata.download()
                except OSError as error:
                    self.download_error = f"Download failed: {error}"

            self.downloader = threading.Thread(target=work, daemon=True)
            self.downloader.start()
        if self.download_error:
            imgui.text_colored((1.0, 0.5, 0.4, 1.0), self.download_error)

    def draw_ui(self):
        io = imgui.get_io()
        W, H = io.display_size.x, io.display_size.y
        if self.camera is not None and W > 0 and H > 0:
            self._handle_click(W, H)
            self._draw_overlay(W, H)
        if self.waiting_for_data and not self._missing_data():
            self.new_event()  # the data has arrived
        if self.scene_name == CUSTOM:
            self.editor.update()
            self._follow_custom()

        imgui.set_next_window_pos(
            (W - WINDOW_MARGIN, WINDOW_MARGIN), imgui.Cond_.always, (1.0, 0.0)
        )
        imgui.set_next_window_size_constraints((0.0, 0.0), (420.0, max(H - 2 * WINDOW_MARGIN, 1.0)))
        flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move
        if not imgui.begin("Event", None, flags)[0]:
            imgui.end()
            return

        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
        imgui.text_colored(
            (0.75, 0.85, 1.0, 1.0), self.event.headline or SCENES[self.scene_name][0]
        )
        imgui.pop_text_wrap_pos()
        if self._missing_data():
            self._draw_download()
            imgui.end()
            return
        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
        if self.notice:
            imgui.text_colored((1.0, 0.6, 0.3, 1.0), self.notice)
        if self.event.error:
            imgui.text_colored((1.0, 0.5, 0.4, 1.0), self.event.error)
            imgui.text_wrapped("'Next event' tries again; another event type can be chosen.")
        # What this is and is not, always in view.
        custom = self.scene_name == CUSTOM
        if custom:
            imgui.text_disabled(
                "A custom event: particles placed by hand, not a collision. It says nothing"
                " about what collisions produce. Computed: decays, kinematics, paths in the"
                " field. Approximate models: energy loss, scattering, showers, detector"
                " resolution."
            )
        elif self.event.real:
            imgui.text_disabled(
                "Measured by CMS: the two muons' momenta and charges. Computed here: their"
                " paths, in a simplified model of the detector."
            )
        else:
            imgui.text_disabled(
                "Not simulated: what the collision produces (drawn from simple distributions)."
                " Computed: decays, kinematics, paths in the field. Approximate models:"
                " energy loss, scattering, showers, detector resolution."
            )
        imgui.pop_text_wrap_pos()
        if custom:
            imgui.text_disabled(f"t = {max(self.now, 0.0):.1f} ns after the particles start")
        else:
            moment = "before" if self.now < 0.0 else "after"
            imgui.text_disabled(f"t = {abs(self.now):.1f} ns {moment} the collision")
        # What is happening, in words: the lines whose time has come.
        happened = [entry for entry in self.log if entry[0] <= self.now + 1e-9][-LOG_LINES:]
        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
        for k, (when, text) in enumerate(happened):
            line = f"{when:5.1f} ns  {text}"
            if k == len(happened) - 1:
                imgui.text_colored((1.0, 1.0, 1.0, 1.0), line)
            else:
                imgui.text_disabled(line)
        imgui.pop_text_wrap_pos()

        imgui.separator()
        if self.selected is not None:
            self._draw_selected()
        else:
            where = "" if custom else " (or the collision point)"
            imgui.text_disabled(f"Click a particle{where} for information")
        imgui.separator()
        self._draw_histogram()
        if self._histogram() is not None:
            if imgui.button(f"Add {BATCH} events"):
                self.request_fill(BATCH)
            imgui.same_line()
        if imgui.button("Next event"):
            self.new_event()
        if self.to_fill:
            # Made in the background (step collects them): the display keeps running.
            if imgui.button("Stop"):
                self.cancel_fill()
            imgui.same_line()
            imgui.text_disabled(f"adding events to the histogram: {self.to_fill} to go")
        if self.fill_error:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + WRAP)
            imgui.text_colored((1.0, 0.5, 0.4, 1.0), f"Adding events failed: {self.fill_error}")
            imgui.pop_text_wrap_pos()

        if not custom and imgui.collapsing_header("The collision, close up (diagram)"):
            self._draw_diagram()
        if imgui.collapsing_header("Particles in this event"):
            self._draw_particle_list()
        if imgui.collapsing_header("Legend"):
            for color, text in LEGEND:
                imgui.color_button(
                    f"##{text}", (*color, 1.0), imgui.ColorEditFlags_.no_tooltip, (12, 12)
                )
                imgui.same_line()
                imgui.text(text)
        imgui.text_disabled("More on what is computed and what is not (hover)")
        imgui.set_item_tooltip(
            "Computed from the physics: relativistic kinematics; decays with the\n"
            "measured masses, lifetimes and branching fractions; motion in the field.\n"
            "Approximate models: momentum sharing in most many-body decays (phase\n"
            "space); mean energy loss (Bethe), Gaussian scattering (Highland),\n"
            "bremsstrahlung, parametrised showers; the measurement (true momenta\n"
            "smeared by the resolution a tracker of this size has in this field,\n"
            "not fitted to the paths).\n"
            "Not simulated: what a collision produces (drawn from simple\n"
            "distributions matched to measured averages); how a Z, J/psi or Higgs\n"
            "is produced (the sentence and diagram are an illustration).\n"
            "'Real data' events: the muons' momenta and charges are CMS's\n"
            "measurements; their paths through the detector are recomputed here.\n"
            "Pairs of equal charge are shown but kept out of the histogram.\n"
            "'Custom' events: particles placed by hand with the momenta you set. They\n"
            "are not collisions; only their decays, paths and measurement are computed."
        )
        imgui.end()

    def diagnostics(self):
        counts = {}
        for track in self.event.tracks:
            counts[track.name] = counts.get(track.name, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        particles = ", ".join(f"{n} {name}" for name, n in ordered)
        charged = [t for t in self.event.tracks if PARTICLES[t.name].charge != 0 and t.parent < 0]
        lines = [
            f" Time: {self.now:.1f} ns relative to the {'start' if self.event.custom else 'collision'}"
            f" (light: {abs(self.now) * C:.1f} m)",
            f" Events: {self.events_seen}; this one: {len(self.event.tracks)} particles",
            f" {particles}",
        ]
        if charged:
            t = max(charged, key=lambda t: np.hypot(t.p4[1], t.p4[2]))
            pt = float(np.hypot(t.p4[1], t.p4[2]))
            radius = helix_radius(pt, 1, max(self.shown_field, 1e-9))
            lines.append(
                f" Stiffest track: {t.name}, p_T {pt:.3g} GeV,"
                f" radius {radius:.3g} m in {self.shown_field:.1f} T"
            )
        return "\n".join(lines)

"""Tests of saving and loading custom scenarios (plumbing: round trips only).

Nothing physical is checked here: a scenario that is saved and loaded must
give back the same bodies and settings, and a file that is not a scenario
must be refused with a message instead of an exception reaching the app.

Every test writes to a temporary folder (the save_dir fixture), never to the
repository's saves/. The fixture also drops, before and after each test, the
state the modules keep outside their instances (N-body and galaxy edits,
bodies added to a black-hole scene, a pending home-screen request).
"""

import json
import os

import numpy as np
import pytest

from physengine.modules import scenario_files
from physengine.modules.blackhole_editor import added_bodies, forget_added, remember
from physengine.modules.blackhole_scenes import orbiting
from physengine.modules.blackholemodule import BlackHoleModule
from physengine.modules.galaxy_editor import (
    clear_edits,
    has_edits,
    scene_for,
    scene_from_dict,
    scene_to_dict,
)
from physengine.modules.galaxy_encounters import SCENES as GALAXY_SCENES
from physengine.modules.galaxymodule import DEFAULT_SCENE, SINGLE, GalaxyModule
from physengine.modules.homemodule import MODULES, HomeModule
from physengine.modules.nbody_scenes import CUSTOM
from physengine.modules.nbodymodule import NEWTON, NBodyModule

KINDS = ("nbody", "blackhole", "galaxy")


def _forget_everything():
    for kind in KINDS:
        scenario_files.take_request(kind)
    NBodyModule().discard_edits()
    forget_added()
    clear_edits()


@pytest.fixture(autouse=True)
def save_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(scenario_files, "SAVE_DIR", tmp_path)
    _forget_everything()
    try:
        yield tmp_path
    finally:
        _forget_everything()


def _write(save_dir, kind, stem, text):
    folder = save_dir / kind
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.json").write_text(text)


# ------------------------------------------------------------ files and names


def test_files_round_trip_and_names_are_cleaned(save_dir):
    target = scenario_files.save("nbody", "my ../../scene?", {"a": [1, 2.5]})
    assert target.parent == save_dir / "nbody"  # the name cannot leave the folder
    assert target.name == "my scene.json"
    assert scenario_files.saved("nbody") == ["my scene"]
    assert scenario_files.load("nbody", "my scene") == {"a": [1, 2.5]}
    scenario_files.delete("nbody", "my scene")
    assert scenario_files.saved("nbody") == []
    with pytest.raises(ValueError):
        scenario_files.save("nbody", "???", {})


def test_file_records_kind_and_format_version(save_dir):
    target = scenario_files.save("galaxy", "one", {"x": 1})
    assert json.loads(target.read_text()) == {
        scenario_files.FORMAT_KEY: scenario_files.FORMAT_VERSION,
        "kind": "galaxy",
        "data": {"x": 1},
    }
    assert list(target.parent.iterdir()) == [target]  # no temporary file left behind


def test_older_file_without_kind_still_loads(save_dir):
    _write(save_dir, "nbody", "old", json.dumps({"bodies": [], "gravity": "Newton"}))
    assert scenario_files.load("nbody", "old") == {"bodies": [], "gravity": "Newton"}


def test_name_ending_in_a_space_after_truncation_can_be_loaded_and_deleted(save_dir):
    # 59 letters, a space, then more: cut at 60 characters the name ends in a space.
    typed = "a" * 59 + " tail"
    target = scenario_files.save("nbody", typed, {"v": 1})
    assert target.stem == "a" * 59 == scenario_files.clean_name(typed)
    (listed,) = scenario_files.saved("nbody")
    assert listed == target.stem
    assert scenario_files.load("nbody", listed) == {"v": 1}
    scenario_files.delete("nbody", listed)
    assert not target.exists()


def test_cleaning_is_deterministic_and_bounded():
    clean = scenario_files.clean_name
    assert clean("a/b") == clean("a.b") == clean("ab") == "ab"
    assert clean("  two  words  ") == "two  words"
    assert clean("../../etc/passwd") == "etcpasswd"
    assert clean("x" * 200) == "x" * scenario_files.MAX_NAME
    assert clean("???") == ""
    assert clean(clean("my ../ scene ")) == clean("my ../ scene ")  # cleaning twice changes nothing


def test_listed_file_is_the_one_loaded_and_deleted(save_dir):
    """A hand-placed 'a.b.json' is listed as 'a.b'; acting on it leaves 'ab.json' alone."""
    scenario_files.save("nbody", "ab", {"which": "ab"})
    _write(save_dir, "nbody", "a.b", json.dumps({"which": "a.b"}))
    assert sorted(scenario_files.saved("nbody")) == ["a.b", "ab"]
    assert scenario_files.load("nbody", "a.b") == {"which": "a.b"}
    assert scenario_files.load("nbody", "ab") == {"which": "ab"}
    assert scenario_files.delete_from_ui("nbody", "a.b") == (True, "Deleted 'a.b'")
    assert scenario_files.saved("nbody") == ["ab"]
    assert scenario_files.load("nbody", "ab") == {"which": "ab"}


def test_listing_is_newest_first(save_dir):
    for age, name in ((300, "oldest"), (200, "middle"), (100, "newest")):
        target = scenario_files.save("nbody", name, {})
        stamp = target.stat().st_mtime - age
        os.utime(target, (stamp, stamp))
    assert scenario_files.saved("nbody") == ["newest", "middle", "oldest"]


def test_names_that_collide_do_not_overwrite_without_confirmation(save_dir):
    saved, message = scenario_files.save_from_ui("nbody", "ab", lambda: {"n": 1})
    assert saved and message == "Saved 'ab'"
    for typed in ("a/b", "a.b", "ab"):
        assert scenario_files.exists("nbody", typed)
        saved, message = scenario_files.save_from_ui("nbody", typed, lambda: {"n": 2})
        assert not saved and "'ab' already exists" in message
        assert scenario_files.load("nbody", "ab") == {"n": 1}
    saved, message = scenario_files.save_from_ui("nbody", "a.b", lambda: {"n": 2}, confirmed=True)
    assert saved and message == "Saved 'ab'"
    assert scenario_files.load("nbody", "ab") == {"n": 2}
    with pytest.raises(FileExistsError):
        scenario_files.save("nbody", "ab", {"n": 3}, overwrite=False)
    assert scenario_files.load("nbody", "ab") == {"n": 2}


@pytest.mark.parametrize("name", ["../outside", "sub/inner", "..", ".", "", "a\\b", "x\0y"])
def test_load_and_delete_cannot_leave_the_folder(save_dir, name):
    outside = save_dir / "outside.json"
    outside.write_text("{}")
    _write(save_dir, "nbody/sub", "inner", "{}")
    with pytest.raises(ValueError):
        scenario_files.load("nbody", name)
    with pytest.raises(ValueError):
        scenario_files.delete("nbody", name)
    assert outside.exists() and (save_dir / "nbody" / "sub" / "inner.json").exists()


def test_save_cannot_leave_the_folder(save_dir):
    for typed in ("../../x", "/abs/olute", "..\\..\\x", "sub/x"):
        target = scenario_files.save("nbody", typed, {})
        assert target.parent == save_dir / "nbody"
    assert {f.parent for f in save_dir.rglob("*.json")} == {save_dir / "nbody"}


# ---------------------------------------------------- files that are not scenarios


@pytest.mark.parametrize(
    "text, expected",
    [
        ("{ not json", "not valid JSON"),
        ("", "not valid JSON"),
        ("[1, 2, 3]", "JSON object"),
        ("null", "JSON object"),
        ('"text"', "JSON object"),
        ("3.5", "JSON object"),
        ('{"physengine_scenario": 1, "kind": "galaxy", "data": {}}', "'galaxy' scenario"),
        ('{"physengine_scenario": 1, "data": {}}', "not a 'nbody' one"),
        ('{"physengine_scenario": 1, "kind": "nbody", "data": [1]}', "no scenario data"),
        ('{"physengine_scenario": 1, "kind": "nbody"}', "no scenario data"),
        ('{"physengine_scenario": 99, "kind": "nbody", "data": {}}', "newer version"),
        ('{"physengine_scenario": "one", "kind": "nbody", "data": {}}', "format version"),
    ],
)
def test_malformed_and_wrong_kind_files_are_refused(save_dir, text, expected):
    _write(save_dir, "nbody", "bad", text)
    with pytest.raises(ValueError, match=expected):
        scenario_files.load("nbody", "bad")
    # From the Load button: a message, on_load never called.
    called = []
    loaded, message = scenario_files.load_from_ui("nbody", "bad", called.append)
    assert not loaded and expected in message and called == []


def test_binary_and_missing_files_are_refused(save_dir):
    folder = save_dir / "nbody"
    folder.mkdir()
    (folder / "binary.json").write_bytes(b"\xff\xfe\x00\x80")
    with pytest.raises(ValueError, match="not a text file"):
        scenario_files.load("nbody", "binary")
    with pytest.raises(OSError):
        scenario_files.load("nbody", "missing")
    assert scenario_files.load_from_ui("nbody", "missing", lambda data: None)[0] is False


def test_wrong_kind_is_caught_when_a_file_is_moved_between_folders(save_dir):
    source = scenario_files.save("galaxy", "moved", {"scene_name": "Prograde"})
    _write(save_dir, "nbody", "moved", source.read_text())
    with pytest.raises(ValueError, match="'galaxy' scenario, not a 'nbody' one"):
        scenario_files.load("nbody", "moved")


@pytest.mark.parametrize(
    "error",
    [KeyError("bodies"), TypeError("no"), AttributeError("x"), IndexError(), ZeroDivisionError()],
)
def test_failing_on_load_becomes_a_message(save_dir, error):
    scenario_files.save("nbody", "fine", {"a": 1})

    def on_load(data):
        raise error

    loaded, message = scenario_files.load_from_ui("nbody", "fine", on_load)
    assert not loaded
    assert message.startswith("Could not load 'fine': ")
    assert message.endswith(str(error) or type(error).__name__)


def test_failing_make_data_or_unwritable_data_becomes_a_message(save_dir):
    def make_data():
        raise RuntimeError("nothing to save")

    assert scenario_files.save_from_ui("nbody", "x", make_data) == (
        False,
        "Could not save: nothing to save",
    )
    saved, message = scenario_files.save_from_ui("nbody", "x", lambda: {"a": np.zeros(2)})
    assert not saved and "JSON" in message
    saved, message = scenario_files.save_from_ui("nbody", "?!", lambda: {})
    assert not saved and "name" in message
    assert scenario_files.saved("nbody") == []  # nothing half-written


def test_successful_load_from_ui_hands_over_the_data(save_dir):
    scenario_files.save("nbody", "fine", {"a": 1})
    got = []
    assert scenario_files.load_from_ui("nbody", "fine", got.append) == (True, "Loaded 'fine'")
    assert got == [{"a": 1}]


# ------------------------------------------------------------- the modules


def test_nbody_scenario_round_trip():
    m = NBodyModule()
    m.scene_name = "Black hole flyby"
    m.gravity = NEWTON
    m.load_scene(m.scene_name)
    for _ in range(30):
        m.step(m.dt)
    scenario_files.save("nbody", "flyby", m.scenario())

    other = NBodyModule()
    other.prepare_scenario(scenario_files.load("nbody", "flyby"))
    assert other.scene_name == CUSTOM and other.gravity == NEWTON
    other.load_scene(other.scene_name)
    np.testing.assert_allclose(other.nb.pos, m.nb.pos)
    np.testing.assert_allclose(other.nb.vel, m.nb.vel)
    np.testing.assert_allclose(other.nb.mass, m.nb.mass)
    np.testing.assert_allclose(other.nb.radius, m.nb.radius)
    assert list(other.nb.black_hole) == list(m.nb.black_hole)
    assert other.scene.names == m.scene.names
    np.testing.assert_allclose(other.scene.radii, m.scene.radii)
    np.testing.assert_allclose(other.scene.colors, m.scene.colors)


def test_nbody_scenario_keeps_the_scene_pace():
    m = NBodyModule()
    m.scene_name = "Star torn by a black hole"
    m.load_scene(m.scene_name)
    other = NBodyModule()
    other.prepare_scenario(m.scenario())
    other.load_scene(other.scene_name)
    assert other.scene.years_per_second == m.scene.years_per_second


def test_black_hole_scenario_round_trip():
    m = BlackHoleModule()
    m.scene_name, m.spin, m.hole_mass_log = "Custom", 0.6, 7.0
    remember("Custom", orbiting("Probe", 9.0, 22.0, inclination_deg=15.0, temperature=8000.0))
    scenario_files.save("blackhole", "probe", m.scenario())
    forget_added()

    other = BlackHoleModule()
    other.prepare_scenario(scenario_files.load("blackhole", "probe"))
    assert (other.scene_name, other.spin, other.hole_mass_log) == ("Custom", 0.6, 7.0)
    specs = added_bodies("Custom")
    assert [s.name for s in specs] == ["Probe"]
    reference = orbiting("Probe", 9.0, 22.0, inclination_deg=15.0, temperature=8000.0)
    np.testing.assert_allclose(specs[0].x, reference.x)
    np.testing.assert_allclose(specs[0].w, reference.w)
    assert specs[0].temperature == 8000.0
    # The module builds the saved body: same bodies as the module that saved it.
    forget_added()
    remember("Custom", reference)
    m.load()
    other.load()
    assert other.names == m.names and other.names[-1] == "Probe"
    np.testing.assert_allclose(other.radius, m.radius)
    np.testing.assert_allclose(other.temperature, m.temperature)
    assert other.temperature[-1] == 8000.0


# ------------------------------------------------------------------- galaxy


def _edited_single():
    scene = scene_for(SINGLE)
    scene.galaxies[0].disc.mass *= 1.5
    scene.galaxies[0].halo.n = 1234
    return scene


def _edited_pair():
    name = next(
        n for n, s in GALAXY_SCENES.items() if s.encounter is not None and n != DEFAULT_SCENE
    )
    pair = scene_for(name)
    pair.encounter.pericentre = 7.5
    pair.encounter.discs[1].inclination = 71.0
    pair.galaxies[1].bulge.a = 0.9
    return name, pair


def _stub_building(module, monkeypatch):
    """Make setup() runnable without OpenGL or sampling galaxies: record load() calls."""
    built = []
    monkeypatch.setattr(module, "load", lambda: built.append(scene_for(module.scene_name)))
    monkeypatch.setattr(module.renderer, "setup", lambda *args: None)
    module.kind = module.galaxy = None
    return built


def test_galaxy_scenario_round_trip_through_a_file_and_the_module():
    for name, scene in ((SINGLE, _edited_single()), _edited_pair()):
        scenario_files.save("galaxy", "setup", scene_to_dict(name, scene))
        clear_edits()

        m = GalaxyModule()
        m.load_scenario(scenario_files.load("galaxy", "setup"))
        assert m.reset_requested and m.scene_name == name
        # What the rebuilt module's load() will build from:
        assert scene_for(name) == scene
        assert scene_for(name) != GALAXY_SCENES[name]


def test_galaxy_scene_dictionary_round_trip_keeps_types():
    name, pair = _edited_pair()
    back_name, back = scene_from_dict(json.loads(json.dumps(scene_to_dict(name, pair))))
    assert (back_name, back) == (name, pair)
    assert type(back.galaxies[0].halo.n) is int
    assert type(back.encounter.pericentre) is float
    # Whole numbers written for lengths (a hand-edited file) become floats.
    data = scene_to_dict(SINGLE, scene_for(SINGLE))
    data["scene"]["galaxies"][0]["disc"]["R_max"] = 20
    disc = scene_from_dict(data)[1].galaxies[0].disc
    assert disc.R_max == 20.0 and type(disc.R_max) is float


def test_galaxy_scenario_opened_from_the_home_screen(monkeypatch):
    name, pair = _edited_pair()
    scenario_files.save("galaxy", "pair", scene_to_dict(name, pair))
    scenario_files.request("galaxy", scenario_files.load("galaxy", "pair"))

    m = GalaxyModule()
    built = _stub_building(m, monkeypatch)
    m.setup(None)  # no GL context: the forces fall back to the CPU tree
    assert m.scene_name == name and built == [pair]
    assert m.scenario_message == ""
    assert scenario_files.take_request("galaxy") is None  # taken


def _break(data, path, value="<remove>"):
    """A deep copy of `data` with the entry at `path` replaced (or removed)."""
    data = json.loads(json.dumps(data))
    at = data
    for key in path[:-1]:
        at = at[key]
    if value == "<remove>":
        del at[path[-1]]
    else:
        at[path[-1]] = value
    return data


GALAXY_BREAKAGES = [
    (("scene_name",), "<remove>"),
    (("scene_name",), "No such scene"),
    (("scene_name",), None),
    (("scene_name",), ["Prograde"]),
    (("scene",), "<remove>"),
    (("scene",), None),
    (("scene",), [1, 2]),
    (("scene", "galaxies"), "<remove>"),
    (("scene", "galaxies"), None),
    (("scene", "galaxies"), []),
    (("scene", "galaxies"), {"0": {}}),
    (("scene", "galaxies", 0), None),
    (("scene", "galaxies", 0, "halo"), "<remove>"),
    (("scene", "galaxies", 0, "halo"), None),
    (("scene", "galaxies", 0, "halo", "a"), "<remove>"),
    (("scene", "galaxies", 0, "halo", "extra"), 1.0),
    (("scene", "galaxies", 0, "halo", "n"), 4000.5),
    (("scene", "galaxies", 0, "halo", "n"), "4000"),
    (("scene", "galaxies", 0, "halo", "n"), True),
    (("scene", "galaxies", 0, "halo", "n"), 0),
    (("scene", "galaxies", 0, "halo", "n"), -5),
    (("scene", "galaxies", 0, "halo", "n"), 10**9),
    (("scene", "galaxies", 0, "bulge", "mass"), None),
    (("scene", "galaxies", 0, "bulge", "mass"), "1.0"),
    (("scene", "galaxies", 0, "bulge", "mass"), 0.0),
    (("scene", "galaxies", 0, "bulge", "mass"), -1.0),
    (("scene", "galaxies", 0, "bulge", "mass"), float("nan")),
    (("scene", "galaxies", 0, "disc", "R_d"), float("inf")),
    (("scene", "galaxies", 0, "disc", "Q"), [1.5]),
    (("scene", "galaxies", 1, "disc", "z0"), "<remove>"),
    (("scene", "galaxies", 1), "<remove>"),  # a pair with one galaxy
    (("scene", "encounter"), "<remove>"),
    (("scene", "encounter"), None),  # two galaxies without an encounter
    (("scene", "encounter"), 3),
    (("scene", "encounter", "pericentre"), "<remove>"),
    (("scene", "encounter", "pericentre"), None),
    (("scene", "encounter", "pericentre"), -1.0),
    (("scene", "encounter", "eccentricity"), -0.5),
    (("scene", "encounter", "start_distance"), 1.0),  # inside the pericentre
    (("scene", "encounter", "speed"), 2.0),
    (("scene", "encounter", "discs"), None),
    (("scene", "encounter", "discs"), [{"inclination": 0.0, "argument": 0.0}]),
    (("scene", "encounter", "discs", 0), [0.0, 0.0]),
    (("scene", "encounter", "discs", 0, "argument"), "<remove>"),
    (("scene", "encounter", "discs", 1, "inclination"), "sixty"),
    (("scene", "encounter", "discs", 1, "tilt"), 1.0),
]


@pytest.mark.parametrize("path, value", GALAXY_BREAKAGES)
def test_broken_galaxy_scenario_is_refused_and_changes_nothing(path, value, monkeypatch):
    name, pair = _edited_pair()
    broken = _break(scene_to_dict(name, pair), path, value)
    with pytest.raises(ValueError):
        scene_from_dict(broken)

    # In the module: prepare_scenario and load_scenario raise ValueError and
    # leave the module and the edits as they were.
    m = GalaxyModule()
    for hook in (m.prepare_scenario, m.load_scenario):
        with pytest.raises(ValueError):
            hook(broken)
    assert m.scene_name == DEFAULT_SCENE and not m.reset_requested
    assert not has_edits(name) and not has_edits(DEFAULT_SCENE)

    # Opened from the home screen: the default scene, with a message.
    scenario_files.request("galaxy", broken)
    opened = GalaxyModule()
    built = _stub_building(opened, monkeypatch)
    opened.setup(None)
    assert opened.scene_name == DEFAULT_SCENE
    assert built == [GALAXY_SCENES[DEFAULT_SCENE]]
    assert "not opened" in opened.scenario_message


@pytest.mark.parametrize("data", [None, [], "scene", 5, {}])
def test_galaxy_scenario_of_the_wrong_shape_is_refused(data):
    with pytest.raises(ValueError):
        scene_from_dict(data)
    with pytest.raises(ValueError):
        GalaxyModule().prepare_scenario(data)


def test_single_galaxy_scenario_with_two_galaxies_is_refused():
    data = scene_to_dict(SINGLE, scene_for(SINGLE))
    data["scene"]["galaxies"] *= 2
    with pytest.raises(ValueError, match="galaxies"):
        scene_from_dict(data)


def test_galaxy_setup_that_cannot_be_built_falls_back_to_the_preset(monkeypatch):
    """A saved setup of the right form whose build fails: the preset, and a message."""
    name, pair = _edited_pair()
    scenario_files.request("galaxy", scene_to_dict(name, pair))
    m = GalaxyModule()
    built = []

    def load():
        built.append(scene_for(m.scene_name))
        if has_edits(m.scene_name):
            raise FloatingPointError("sampling failed")

    monkeypatch.setattr(m, "load", load)
    monkeypatch.setattr(m.renderer, "setup", lambda *args: None)
    m.kind = m.galaxy = None
    m.setup(None)
    assert built == [pair, GALAXY_SCENES[name]]
    assert "could not be built" in m.scenario_message and "sampling failed" in m.scenario_message
    assert not has_edits(name)


def test_galaxy_load_button_reports_a_broken_file(save_dir):
    name, pair = _edited_pair()
    broken = _break(scene_to_dict(name, pair), ("scene", "galaxies", 0, "halo", "a"))
    scenario_files.save("galaxy", "broken", broken)
    m = GalaxyModule()
    loaded, message = scenario_files.load_from_ui("galaxy", "broken", m.load_scenario)
    assert not loaded and "galaxy 1 halo: missing a" in message
    assert not m.reset_requested


# -------------------------------------------------------------- home screen


def test_home_screen_request_is_taken_once():
    scenario_files.request("nbody", {"bodies": []})
    assert scenario_files.take_request("galaxy") is None
    assert scenario_files.take_request("nbody") == {"bodies": []}
    assert scenario_files.take_request("nbody") is None


def test_home_screen_entries_match_the_registry():
    from physengine.modules.registry import REGISTRY

    assert [entry[0] for entry in MODULES] == [name for name in REGISTRY if name != "Home"]
    assert {entry[1] for entry in MODULES} == {*KINDS, None}


def test_home_screen_does_not_claim_the_collision_is_simulated():
    """CLAUDE.md, accuracy scope: decays and kinematics, not the collision physics."""
    title, text = next(entry[2:] for entry in MODULES if entry[0] == "Particles")
    assert "collision" not in title.lower()
    assert "collision itself is not simulated" in text


def test_home_screen_opens_a_saved_scenario():
    scenario_files.save("galaxy", "mine", scene_to_dict(SINGLE, _edited_single()))
    home = HomeModule()
    assert home.switch_requested is None
    assert home.open_saved("Galaxies", "galaxy", "mine") is True
    assert home.switch_requested == "Galaxies" and home.errors == {}
    assert scenario_files.take_request("nbody") is None  # left for the galaxy module only
    data = scenario_files.take_request("galaxy")
    assert scene_from_dict(data) == (SINGLE, _edited_single())


@pytest.mark.parametrize(
    "text, expected",
    [
        ("{ not json", "not valid JSON"),
        ("[]", "JSON object"),
        ('{"physengine_scenario": 1, "kind": "nbody", "data": {}}', "'nbody' scenario"),
    ],
)
def test_home_screen_keeps_an_unreadable_file_out(save_dir, text, expected):
    _write(save_dir, "galaxy", "bad", text)
    home = HomeModule()
    assert home.open_saved("Galaxies", "galaxy", "bad") is False
    assert home.switch_requested is None
    assert expected in home.errors[("galaxy", "bad")]
    assert scenario_files.take_request("galaxy") is None  # nothing left for a module
    # Once the file is repaired the entry opens and the error goes.
    _write(save_dir, "galaxy", "bad", json.dumps(scene_to_dict(SINGLE, scene_for(SINGLE))))
    assert home.open_saved("Galaxies", "galaxy", "bad") is True
    assert home.errors == {} and home.switch_requested == "Galaxies"


def test_home_screen_reports_a_file_that_disappeared(save_dir):
    home = HomeModule()
    assert home.open_saved("NbodyModule", "nbody", "gone") is False
    assert home.switch_requested is None and ("nbody", "gone") in home.errors

"""Saving and loading custom scenarios, shared by the modules. Plumbing and UI only.

A scenario is a JSON file, saves/<kind>/<name>.json at the top of the
repository, holding whatever the module of that kind needs to rebuild what
was on screen (each module has a `scenario()` method that makes the
dictionary and a `prepare_scenario(data)` that takes it back).

The file wraps the module's dictionary with the kind and a format version:
    {"physengine_scenario": 1, "kind": "nbody", "data": {...}}
so a file put in the wrong folder is refused. Older files hold the module's
dictionary alone and are still read.

Names. A name typed by the user is cleaned (`clean_name`) before it becomes
a file name; the controls show the cleaned name before saving. The files on
disk are listed, loaded and deleted by their real names (`saved`, `load`,
`delete` take the file's stem exactly as listed and do not clean it).

The home screen can also open a saved scenario: it leaves the data here
(`request`) and switches to the module, whose setup() picks it up
(`take_request`).
"""

import json
import os
import re
from pathlib import Path

from imgui_bundle import imgui

SAVE_DIR = Path(__file__).resolve().parents[3] / "saves"
FIELD_WIDTH = 170.0
MAX_NAME = 60
FORMAT_KEY = "physengine_scenario"  # its presence marks the wrapped format
FORMAT_VERSION = 1

# Per kind: the name typed in that module's save box, the last message shown,
# the listed file that is selected, and the name whose overwrite was asked.
_names = {}
_messages = {}
_selected = {}
_overwrite = {}
_request = None  # (kind, data) left by the home screen for a module about to open


# ------------------------------------------------------------------ names


def clean_name(name):
    """The file name (without .json) a typed name is saved under.

    Keeps letters, digits, spaces, - and _; drops everything else; at most
    MAX_NAME characters, without spaces at either end. Empty if nothing is left.
    """
    return re.sub(r"[^A-Za-z0-9 _-]", "", str(name))[:MAX_NAME].strip()


_clean = clean_name  # earlier name


def path(kind, name):
    """Where save() writes the scenario typed as `name` (the name is cleaned)."""
    return SAVE_DIR / kind / f"{clean_name(name)}.json"


def _file(kind, stem):
    """The existing-file path for a listed name: `stem` is used exactly as given.

    A stem that could name anything outside the kind's folder is refused.
    """
    if (
        not isinstance(stem, str)
        or stem in ("", ".", "..")
        or any(c in stem for c in ("/", "\\", "\0"))
    ):
        raise ValueError(f"'{stem}' is not the name of a saved scenario")
    return SAVE_DIR / kind / f"{stem}.json"


def exists(kind, name):
    """Whether saving under the typed `name` would replace a file."""
    return bool(clean_name(name)) and path(kind, name).is_file()


def saved(kind):
    """Names (file stems) of the saved scenarios of `kind`, newest first."""
    folder = SAVE_DIR / kind
    if not folder.is_dir():
        return []
    found = []
    for f in folder.glob("*.json"):
        try:
            if f.is_file():
                found.append((-f.stat().st_mtime, f.stem))
        except OSError:
            continue  # removed in the meantime
    return [stem for _, stem in sorted(found)]


# ------------------------------------------------------------------ files


def save(kind, name, data, overwrite=True):
    """Write `data` as scenario `name` (cleaned). Returns the file's path.

    With overwrite=False an existing file is left alone and FileExistsError
    is raised.
    """
    if not clean_name(name):
        raise ValueError("give the scenario a name (letters, digits, spaces, - or _)")
    if not isinstance(data, dict):
        raise ValueError("a scenario must be a dictionary")
    target = path(kind, name)
    if not overwrite and target.exists():
        raise FileExistsError(f"'{target.stem}' already exists")
    try:
        text = json.dumps({FORMAT_KEY: FORMAT_VERSION, "kind": kind, "data": data}, indent=1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the scenario cannot be written as JSON: {error}") from error
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and moved over it, so a failed write does not
    # leave half a file under the scenario's name.
    partial = target.with_name(target.name + ".part")
    try:
        partial.write_text(text, encoding="utf-8")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def load(kind, name):
    """The dictionary saved as the listed scenario `name` of `kind`.

    `name` is the file's stem exactly as saved() lists it. Raises ValueError
    with a readable message if the file is not a scenario of this kind (not
    JSON, not a JSON object, saved by another module, or from a newer
    format), OSError if it cannot be read. What the dictionary holds is for
    the module to check.
    """
    target = _file(kind, name)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"'{name}' is not a text file") from error
    except RecursionError as error:
        raise ValueError(f"'{name}' is nested too deeply to read") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"'{name}' is not valid JSON (line {error.lineno}: {error.msg})"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError(f"'{name}' is not a scenario (a JSON object was expected)")
    if FORMAT_KEY not in raw:
        return raw  # an older file: the module's dictionary alone
    version = raw[FORMAT_KEY]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"'{name}' has an unreadable format version")
    if version > FORMAT_VERSION:
        raise ValueError(f"'{name}' was saved by a newer version of the program")
    if raw.get("kind") != kind:
        raise ValueError(f"'{name}' is a '{raw.get('kind')}' scenario, not a '{kind}' one")
    data = raw.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"'{name}' holds no scenario data")
    return data


def delete(kind, name):
    """Remove the listed scenario `name` (the file's stem, exactly)."""
    _file(kind, name).unlink(missing_ok=True)


# ------------------------------------------------- home screen hand-off


def request(kind, data):
    """Leave `data` for the module of `kind` that is about to be opened."""
    global _request
    _request = (kind, data)


def take_request(kind):
    """The data left by request() for this kind, once; None if there is none."""
    global _request
    if _request is not None and _request[0] == kind:
        data = _request[1]
        _request = None
        return data
    return None


# ------------------------------------------------------------- controls


def _describe(error):
    return str(error) or type(error).__name__


def save_from_ui(kind, name, make_data, confirmed=False):
    """What the Save button does. Returns (saved, message).

    An existing file is replaced only when `confirmed`; otherwise nothing is
    written and the message asks for the confirmation. No exception leaves
    this function: a failing make_data() or write becomes the message.
    """
    stem = clean_name(name)
    try:
        if stem and not confirmed and exists(kind, name):
            return False, f"'{stem}' already exists: press Overwrite to replace it"
        save(kind, name, make_data(), overwrite=confirmed)
    except Exception as error:  # noqa: BLE001 (must not take the app down)
        return False, f"Could not save: {_describe(error)}"
    return True, f"Saved '{stem}'"


def load_from_ui(kind, name, on_load):
    """What the Load button does. Returns (loaded, message).

    No exception leaves this function: an unreadable file, or an on_load
    that fails in any way, becomes the message.
    """
    try:
        on_load(load(kind, name))
    except Exception as error:  # noqa: BLE001 (must not take the app down)
        return False, f"Could not load '{name}': {_describe(error)}"
    return True, f"Loaded '{name}'"


def delete_from_ui(kind, name):
    """What the Delete button does. Returns (deleted, message)."""
    try:
        delete(kind, name)
    except (OSError, ValueError) as error:
        return False, f"Could not delete '{name}': {_describe(error)}"
    return True, f"Deleted '{name}'"


def draw(kind, make_data, on_load, note=None):
    """The save/load controls, inside a module's own window.

    `make_data()` returns the dictionary to save; `on_load(data)` is called
    with a loaded one. `note`, if given, is shown under the Save button (for
    a module to say what exactly is saved).
    """
    if not imgui.collapsing_header("Save / load"):
        return
    imgui.set_next_item_width(FIELD_WIDTH)
    _, _names[kind] = imgui.input_text(f"Name##{kind}", _names.get(kind, ""))
    typed = _names[kind]
    stem = clean_name(typed)
    if _overwrite.get(kind) != stem:
        _overwrite[kind] = None  # the name changed: ask again
    asking = _overwrite.get(kind) is not None
    imgui.same_line()
    if asking:
        if imgui.button(f"Overwrite##{kind}"):
            _, _messages[kind] = save_from_ui(kind, typed, make_data, confirmed=True)
            _overwrite[kind] = None
        imgui.same_line()
        if imgui.button(f"Cancel##{kind}"):
            _overwrite[kind] = None
            _messages[kind] = ""
    elif imgui.button(f"Save##{kind}"):
        done, _messages[kind] = save_from_ui(kind, typed, make_data)
        if not done and exists(kind, typed):
            _overwrite[kind] = stem
    # The name the file gets, whenever it is not what was typed.
    if typed and not stem:
        imgui.text_disabled("Use letters, digits, spaces, - or _ in the name")
    elif stem != typed:
        imgui.text_disabled(f"Will be saved as '{stem}'")
    if note:
        imgui.text_disabled(note)

    names = saved(kind)
    if not names:
        imgui.text_disabled("Nothing saved yet")
    for i, name in enumerate(names):
        if imgui.selectable(f"{name}##{kind}{i}", _selected.get(kind) == name)[0]:
            _selected[kind] = name
            _names[kind] = name
    chosen = _selected.get(kind)
    if chosen in names:
        if imgui.button(f"Load##{kind}"):
            _, _messages[kind] = load_from_ui(kind, chosen, on_load)
        imgui.same_line()
        if imgui.button(f"Delete##{kind}"):
            done, _messages[kind] = delete_from_ui(kind, chosen)
            if done:
                _selected[kind] = None
    if _messages.get(kind):
        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + 2.0 * FIELD_WIDTH)
        imgui.text_disabled(_messages[kind])
        imgui.pop_text_wrap_pos()

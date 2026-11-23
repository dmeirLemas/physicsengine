"""Real collisions for the particle-physics module: CMS open data.

Written by Claude (2026-11-22, module 4). No physics here: this downloads
and reads files of events recorded by the CMS experiment at the LHC in 2011
(proton-proton collisions at 7 TeV) and released for education:

    T. McCauley (2017), "Datasets derived from the Run2011A SingleElectron,
    SingleMu, DoubleElectron, and DoubleMu primary datasets", CERN Open Data
    Portal, https://opendata.cern.ch/record/545  (licence CC0).

Each event in these files has two muons: their momenta as CMS measured
them, and their charges. In Zmumu.csv 3.4% of the pairs have two muons of
the same charge (they cannot come from one Z); the module shows such events
labelled as what they are and keeps them out of the mass histograms, which
are of opposite-charge pairs. Nothing else of the collision is in the files (the
hundreds of other particles were not kept), so that is all that can be
shown. The portal notes that the files are a selection for teaching and
"not suitable for a full physics analysis".

To fetch the files (15 MB in all) once:
    python -m physengine.modules.particle_opendata
or use the Download button in the module.
"""

import csv
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://opendata.cern.ch/record/545/files/"
# scene name -> file
FILES = {
    "Real data: Z -> mu+ mu-": "Zmumu.csv",
    "Real data: J/psi -> mu+ mu-": "Jpsimumu.csv",
    "Real data: all muon pairs": "Dimuon_DoubleMu.csv",
}
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "opendata"
M_MU = 0.1056584  # GeV
BEAM_ENERGY = 3.5  # TeV per proton in 2011

_cache = {}


def path(scene):
    return DATA_DIR / FILES[scene]


def available(scene):
    return path(scene).exists()


def download(scene=None):
    """Fetch the file of `scene` (or all files) into DATA_DIR. Returns the paths written."""
    written = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in [scene] if scene else list(FILES):
        target = path(name)
        if not target.exists():
            partial = target.with_suffix(".part")
            urllib.request.urlretrieve(BASE_URL + FILES[name], partial)
            partial.rename(target)
            written.append(target)
    return written


def read_csv(file):
    """The events in `file`: (four-momenta (n, 2, 4) [GeV], charges (n, 2)).

    Files with px, py, pz use those; files with only pt, eta, phi are
    converted (px = pt cos phi, py = pt sin phi, pz = pt sinh eta). The
    energy is recomputed with the muon mass either way.

    A line is skipped if it is short, if a value is missing, not a number
    or not finite, if a muon has no momentum, or if a charge is not +1 or
    -1. A file with no usable line gives empty arrays (0, 2, 4) and (0, 2).
    """
    momenta, charges = [], []
    with open(file, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                pair = []
                for k in ("1", "2"):
                    if "px" + k in row:
                        p = [float(row["px" + k]), float(row["py" + k]), float(row["pz" + k])]
                    else:
                        pt, eta, phi = (
                            float(row["pt" + k]),
                            float(row["eta" + k]),
                            float(row["phi" + k]),
                        )
                        with np.errstate(over="ignore", invalid="ignore"):
                            # (an absurd eta overflows to inf: rejected below)
                            p = [pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)]
                    pair.append([np.sqrt(M_MU**2 + p[0] ** 2 + p[1] ** 2 + p[2] ** 2), *p])
                q = [float(row["Q1"]), float(row["Q2"])]
            except (KeyError, ValueError, TypeError, OverflowError):
                continue  # an incomplete or unreadable line (a short one gives None)
            if not np.all(np.isfinite(pair)) or any(abs(v) != 1.0 for v in q):
                continue
            if any(v[0] <= M_MU for v in pair):
                continue  # no momentum: nothing to follow
            momenta.append(pair)
            charges.append([int(v) for v in q])
    if not momenta:
        return np.zeros((0, 2, 4)), np.zeros((0, 2), dtype=int)
    return np.array(momenta), np.array(charges)


def load(scene):
    """The events of `scene` (see read_csv), read once and kept."""
    if scene not in _cache:
        _cache[scene] = read_csv(path(scene))
    return _cache[scene]


def opposite_sign(charges):
    """Which pairs have muons of opposite charge (boolean per event)."""
    return charges[:, 0] * charges[:, 1] < 0


if __name__ == "__main__":
    for written in download():
        print("downloaded", written)
    for name in FILES:
        print(name, len(load(name)[0]), "events")

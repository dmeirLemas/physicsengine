"""Constants, elements and nuclear data (masses, half-lives, decay modes) for the atom module.

Written by Claude under a one-off override of CLAUDE.md (2026-11-23, atom
module). Review the physics.

Units used throughout the atom module:
    energy   MeV for nuclei, eV for electrons (stated per function)
    mass     MeV / c^2
    length   fm (1e-15 m) for nuclei, Bohr radii a0 for electron clouds
    time     seconds

Constants: CODATA 2018 recommended values (Tiesinga et al., Rev. Mod. Phys.
93, 025010, 2021), written out below.

Nuclear data, in order of preference:
1. The Atomic Mass Evaluation 2020 (Wang, Huang, Kondev, Audi, Naimi, Chinese
   Physics C 45, 030003, 2021) and NUBASE2020 (Kondev, Wang, Huang, Naimi,
   Audi, Chinese Physics C 45, 030001, 2021), as published by the IAEA Atomic
   Mass Data Center. The two text files are downloaded once into data/atom/
   (python -m physengine.modules.atom_data, or the button in the module).
2. Without the files: EXCERPT below (about 150 nuclides copied by a script
   from those same two files on 2026-11-23: the light nuclei, the natural
   decay series and the nuclides the scenes use), and for every other
   nuclide the semi-empirical mass formula (see semf_binding), which is
   good to a few MeV only (3 MeV rms). source() says which of the three a mass is.

Values marked '#' in the files (estimated from systematics, not measured)
are used and flagged as estimated.
"""

# ruff: noqa: E501  (the excerpt's rows are long)
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ constants
# CODATA 2018.
C_LIGHT = 299792458.0  # m/s (exact)
U_MEV = 931.49410242  # atomic mass unit [MeV]
M_ELECTRON = 0.51099895000  # MeV
M_PROTON = 938.27208816  # MeV
M_NEUTRON = 939.56542052  # MeV
HBARC = 197.3269804  # MeV fm
ALPHA = 7.2973525693e-3  # fine-structure constant (1 / 137.035999084)
E2 = ALPHA * HBARC  # e^2 / (4 pi eps0) = 1.43996 MeV fm
RYDBERG_EV = 13.605693122994  # Rydberg energy for an infinitely heavy nucleus [eV]
BOHR_RADIUS_M = 5.29177210903e-11  # a0 [m]
BOHR_RADIUS_FM = BOHR_RADIUS_M * 1e15
HC_EV_NM = 1239.841984  # h c [eV nm]
K_BOLTZMANN_EV = 8.617333262e-5  # eV / K
AVOGADRO = 6.02214076e23

R0_FM = 1.2  # nuclear radius constant: R = R0 A^(1/3) (textbook value; 1.2 to 1.25 are in use)

MINUTE, HOUR, DAY = 60.0, 3600.0, 86400.0
YEAR = 31556926.0  # s: the tropical year NUBASE2020 uses for its half-lives

# NUBASE half-life units [s].
_TIME_UNITS = {
    "ys": 1e-24, "zs": 1e-21, "as": 1e-18, "fs": 1e-15, "ps": 1e-12, "ns": 1e-9,
    "us": 1e-6, "ms": 1e-3, "s": 1.0, "m": MINUTE, "h": HOUR, "d": DAY, "y": YEAR,
    "ky": 1e3 * YEAR, "My": 1e6 * YEAR, "Gy": 1e9 * YEAR, "Ty": 1e12 * YEAR,
    "Py": 1e15 * YEAR, "Ey": 1e18 * YEAR, "Zy": 1e21 * YEAR, "Yy": 1e24 * YEAR,
}  # fmt: skip

# ------------------------------------------------------------------- elements
SYMBOLS = (
    "n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge"
    " As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm"
    " Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U"
    " Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
NAMES = (
    "neutron hydrogen helium lithium beryllium boron carbon nitrogen oxygen fluorine neon"
    " sodium magnesium aluminium silicon phosphorus sulfur chlorine argon potassium calcium"
    " scandium titanium vanadium chromium manganese iron cobalt nickel copper zinc gallium"
    " germanium arsenic selenium bromine krypton rubidium strontium yttrium zirconium niobium"
    " molybdenum technetium ruthenium rhodium palladium silver cadmium indium tin antimony"
    " tellurium iodine xenon caesium barium lanthanum cerium praseodymium neodymium promethium"
    " samarium europium gadolinium terbium dysprosium holmium erbium thulium ytterbium lutetium"
    " hafnium tantalum tungsten rhenium osmium iridium platinum gold mercury thallium lead"
    " bismuth polonium astatine radon francium radium actinium thorium protactinium uranium"
    " neptunium plutonium americium curium berkelium californium einsteinium fermium"
    " mendelevium nobelium lawrencium rutherfordium dubnium seaborgium bohrium hassium"
    " meitnerium darmstadtium roentgenium copernicium nihonium flerovium moscovium livermorium"
    " tennessine oganesson"
).split()
Z_MAX = len(SYMBOLS) - 1
_Z_OF = {s.lower(): z for z, s in enumerate(SYMBOLS)}


def label(Z, A, isomer=""):
    """'U-238', 'Pa-234m', 'n' (free neutron), 'p' is written H-1."""
    if Z == 0 and A == 1:
        return "n"
    return f"{SYMBOLS[Z]}-{A}{isomer}"


def parse_label(text):
    """(Z, A, isomer) from 'U-238', 'u238', '238U', 'Pa-234m', 'n'. None if not understood."""
    text = text.strip().replace(" ", "")
    if text.lower() in ("n", "neutron"):
        return 0, 1, ""
    if text.lower() in ("p", "proton"):
        return 1, 1, ""
    if text.lower() in ("d", "deuteron"):
        return 1, 2, ""
    if text.lower() in ("t", "triton"):
        return 1, 3, ""
    match = re.fullmatch(r"([A-Za-z]{1,2})-?(\d{1,3})(m?)", text) or None
    if match:
        symbol, mass, iso = match.groups()
    else:
        # Mass number first: '152Sm' is samarium, '234Pam' the Pa isomer. The
        # letters are read as a whole symbol first, so that a symbol ending
        # in 'm' (Sm, Am, Cm, Pm, Tm, Fm) is not taken for an isomer mark.
        match = re.fullmatch(r"(\d{1,3})-?([A-Za-z]{1,3})", text)
        if not match:
            return None
        mass, letters = match.groups()
        if letters.lower() in _Z_OF:
            symbol, iso = letters, ""
        elif letters.endswith("m") and letters[:-1].lower() in _Z_OF:
            symbol, iso = letters[:-1], "m"
        else:
            return None
    Z = _Z_OF.get(symbol.lower())
    if Z is None or Z == 0:
        return None
    return Z, int(mass), iso


# ----------------------------------------------------------------- nuclide data


@dataclass
class Nuclide:
    """One nuclear state as NUBASE lists it.

    half_life: seconds; inf for stable; None when not known.
    modes: [(mode, fraction)] with fractions as listed (they need not sum
    to 1). Mode names are NUBASE's: 'A' alpha, 'B-', 'B+' (positron emission
    and electron capture together), 'EC', 'e+', 'IT' (isomer to ground state),
    'SF', 'n', 'p', and rarer ones which the module does not follow.
    """

    Z: int
    A: int
    isomer: str = ""  # "" ground state, "m" first isomer
    mass_excess: float = 0.0  # MeV, of the neutral atom in this state
    estimated: bool = False  # mass from systematics ('#' in the file)
    excitation: float = 0.0  # MeV above the ground state
    half_life: float | None = None
    half_life_text: str = ""
    modes: list = field(default_factory=list)
    abundance: float = 0.0  # natural isotopic abundance, fraction

    @property
    def N(self):
        return self.A - self.Z

    @property
    def name(self):
        return label(self.Z, self.A, self.isomer)

    @property
    def stable(self):
        return self.half_life == np.inf


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "atom"
BASE_URL = "https://www-nds.iaea.org/amdc/ame2020/"
MASS_FILE = "mass_1.mas20.txt"
NUBASE_FILE = "nubase_4.mas20.txt"
FILES = (MASS_FILE, NUBASE_FILE)


def available():
    """Whether both data files are on this computer."""
    return all((DATA_DIR / name).exists() for name in FILES)


def download():
    """Fetch the two files into DATA_DIR (1.2 MB). Returns the paths written.

    A plain anonymous GET: nothing about the user or the project is sent.
    """
    written = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DATA_DIR / name
        if target.exists():
            continue
        request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "python-urllib"})
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(request, timeout=60) as response:
            partial.write_bytes(response.read())
        partial.rename(target)
        written.append(target)
    reload()
    return written


def _number(text):
    """A number as the AME / NUBASE files write it: '#' in place of the decimal
    point marks an estimate. Returns (value, estimated) or (None, False)."""
    text = text.strip()
    estimated = "#" in text
    text = text.replace("#", ".").replace(" ", "")
    if not text or text == "*":
        return None, False
    try:
        return float(text), estimated
    except ValueError:
        return None, False


def read_ame(path):
    """{(Z, A): (mass excess [MeV], estimated)} from the AME2020 mass table.

    Fixed columns (format line of the file): a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,...
    so N is columns 5-9, Z 10-14, A 15-19 and the mass excess [keV] 29-42
    (1-based). Lines are data once the header line starting '1N-Z' has passed.
    """
    masses = {}
    started = False
    with open(path, encoding="ascii", errors="replace") as handle:
        for line in handle:
            if not started:
                started = line.startswith("1N-Z")
                continue
            if len(line) < 42:
                continue
            try:
                Z, A = int(line[9:14]), int(line[14:19])
            except ValueError:
                continue
            value, estimated = _number(line[28:42])
            if value is not None:
                masses[(Z, A)] = (value * 1e-3, estimated)
    return masses


_MODE = re.compile(r"^([A-Za-z0-9+\-]+?)\s*([=~<>])\s*([0-9.]+(?:[eE][+\-]?\d+)?)")


def parse_modes(text):
    """([(mode, fraction)], abundance) from NUBASE's decay-mode field.

    'B-=64.06 6;A=35.94 6;B-A~0.014' gives B- 0.6406, A 0.3594, B-A 0.00014.
    Modes given only as an upper or lower limit, or with '?', are left out
    (not observed, or not measured). 'IS' is the isotopic abundance.
    """
    modes, abundance = [], 0.0
    for item in text.split(";"):
        match = _MODE.match(item.strip())
        if not match:
            continue
        name, operator, value = match.groups()
        fraction = float(value) / 100.0
        if name == "IS":
            abundance = fraction
        elif operator in "=~" and fraction > 0.0:
            modes.append((name, min(fraction, 1.0)))
    return modes, abundance


def parse_half_life(value, unit):
    """Seconds from NUBASE's half-life field ('4.463', 'Gy'); inf for 'stbl';
    None for unknown or particle-unstable states without a number."""
    value = value.strip()
    if value.startswith("stbl"):
        return np.inf
    number, _ = _number(value.lstrip("<>~"))
    if number is None or unit.strip() not in _TIME_UNITS:
        return None
    return number * _TIME_UNITS[unit.strip()]


def read_nubase(path):
    """{(Z, A, isomer): Nuclide} from NUBASE2020: ground states and first isomers ('m').

    Fixed columns as documented at the top of the file (1-based): A 1-3,
    ZZZi 5-8, state letter 17, mass excess [keV] 19-31, excitation energy
    [keV] 43-54, half-life 70-78, its unit 79-80, decay modes 120-209.
    """
    nuclides = {}
    with open(path, encoding="ascii", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or len(line) < 80:
                continue
            try:
                A, Z, state = int(line[0:3]), int(line[4:7]), line[7]
            except ValueError:
                continue
            if state == "0":
                isomer = ""
            elif state == "1" and line[16] == "m":
                isomer = "m"
            else:
                continue
            excess, estimated = _number(line[18:31])
            if excess is None:
                continue
            excitation, _ = _number(line[42:54])
            modes, abundance = parse_modes(line[119:209])
            nuclides[(Z, A, isomer)] = Nuclide(
                Z=Z,
                A=A,
                isomer=isomer,
                mass_excess=excess * 1e-3,
                estimated=estimated,
                excitation=(excitation or 0.0) * 1e-3,
                half_life=parse_half_life(line[69:78], line[78:80]),
                half_life_text=" ".join(line[69:80].split()),
                modes=modes,
                abundance=abundance,
            )
    return nuclides


# Mass excesses [MeV] of the neutron and H-1 (AME2020): the building blocks
# of every binding energy. Copied from the file.
EXCESS_N = 8.07131806
EXCESS_H1 = 7.288971064

# -------------------------------------------------- semi-empirical mass formula
# Coefficients [MeV]: a linear least-squares fit of the formula below, made
# for this module on 2026-11-23, to the 2455 binding energies with A >= 20
# that AME2020 lists as measured (not estimated). The fit leaves 3.2 MeV rms
# (largest deviation 27 MeV, at doubly magic nuclei: the formula has no
# shells). test_atom.py repeats the fit when the data files are present.
# Textbook sets differ from these by a few percent, depending on the masses
# fitted and on whether the Coulomb term is written Z^2 or Z (Z - 1).
SEMF_VOLUME = 15.46
SEMF_SURFACE = 17.02
SEMF_COULOMB = 0.6985
SEMF_ASYMMETRY = 22.62
SEMF_PAIRING = 12.17


def semf_terms(Z, A):
    """The five terms of the Bethe-Weizsaecker formula [MeV], as a dict.

    B = a_V A - a_S A^(2/3) - a_C Z (Z - 1) / A^(1/3) - a_A (A - 2 Z)^2 / A + delta,
    delta = +a_P / sqrt(A) for even Z and even N, -a_P / sqrt(A) for odd-odd,
    0 for odd A.
    """
    Z, A = np.asarray(Z, dtype=float), np.asarray(A, dtype=float)
    N = A - Z
    even_even = (Z % 2 == 0) & (N % 2 == 0)
    odd_odd = (Z % 2 == 1) & (N % 2 == 1)
    pairing = SEMF_PAIRING / np.sqrt(A) * (even_even.astype(float) - odd_odd.astype(float))
    return {
        "volume": SEMF_VOLUME * A,
        "surface": -SEMF_SURFACE * A ** (2.0 / 3.0),
        "Coulomb": -SEMF_COULOMB * Z * (Z - 1.0) / A ** (1.0 / 3.0),
        "asymmetry": -SEMF_ASYMMETRY * (A - 2.0 * Z) ** 2 / A,
        "pairing": pairing,
    }


def semf_binding(Z, A):
    """Binding energy [MeV] from the semi-empirical mass formula (see semf_terms)."""
    terms = semf_terms(Z, A)
    return sum(terms.values())


def semf_mass_excess(Z, A):
    """Mass excess [MeV] of the neutral atom from the formula's binding energy."""
    return Z * EXCESS_H1 + (A - Z) * EXCESS_N - float(semf_binding(Z, A))


# --------------------------------------------------------------------- excerpt
# Copied by a script from the AME2020 / NUBASE2020 files on 2026-11-23 (not
# from memory). name: (mass excess [MeV], estimated, excitation [MeV],
# half-life [s] or inf or None, NUBASE half-life text, modes, abundance).
inf = np.inf
# fmt: off
EXCERPT = {
    # EXCERPT-BEGIN
    "n": (8.071318100000001, False, 0.0, 609.8, "609.8 s", (("B-", 1.0), ), 0.0),
    "H-1": (7.288971064, False, 0.0, inf, "stbl", (), 0.999855),
    "H-2": (13.135722895, False, 0.0, inf, "stbl", (), 0.000145),
    "H-3": (14.949810900000001, False, 0.0, 388781328.32, "12.32 y", (("B-", 1.0), ), 0.0),
    "He-3": (14.931218880000001, False, 0.0, inf, "stbl", (), 2e-06),
    "He-4": (2.42491587, False, 0.0, inf, "stbl", (), 0.9999979999999999),
    "Li-6": (14.0868804, False, 0.0, inf, "stbl", (), 0.048499999999999995),
    "Li-7": (14.907105, False, 0.0, inf, "stbl", (), 0.9515),
    "Be-7": (15.769, False, 0.0, 4598208.0, "53.22 d", (("EC", 1.0), ), 0.0),
    "Be-8": (4.94167, False, 0.0, 8.190000000000001e-17, "81.9 as", (("A", 1.0), ), 0.0),
    "Be-9": (11.348450000000001, False, 0.0, inf, "stbl", (), 1.0),
    "B-8": (22.921599999999998, False, 0.0, 0.7719, "771.9 ms", (("B+", 1.0), ("B+A", 1.0), ), 0.0),
    "B-10": (12.050611000000002, False, 0.0, inf, "stbl", (), 0.19649999999999998),
    "B-11": (8.667708000000001, False, 0.0, inf, "stbl", (), 0.8035),
    "C-11": (10.6494, False, 0.0, 1220.412, "20.3402 m", (("B+", 1.0), ), 0.0),
    "C-12": (0.0, False, 0.0, inf, "stbl", (), 0.9894),
    "C-13": (3.1250093000000003, False, 0.0, inf, "stbl", (), 0.0106),
    "C-14": (3.019893, False, 0.0, 179874478200.0, "5.70 ky", (("B-", 1.0), ), 0.0),
    "N-13": (5.345479999999999, False, 0.0, 597.9, "9.965 m", (("B+", 1.0), ), 0.0),
    "N-14": (2.8634168, False, 0.0, inf, "stbl", (), 0.9962050000000001),
    "N-15": (0.1014381, False, 0.0, inf, "stbl", (), 0.003795),
    "O-15": (2.8556, False, 0.0, 122.266, "122.266 s", (("B+", 1.0), ), 0.0),
    "O-16": (-4.7370021, False, 0.0, inf, "stbl", (), 0.9975700000000001),
    "O-17": (-0.8087641999999999, False, 0.0, inf, "stbl", (), 0.0003835),
    "O-18": (-0.7828163, False, 0.0, inf, "stbl", (), 0.002045),
    "F-18": (0.8731, False, 0.0, 6584.04, "109.734 m", (("B+", 1.0), ), 0.0),
    "Ne-20": (-7.041932200000001, False, 0.0, inf, "stbl", (), 0.9048),
    "Ne-22": (-8.024716, False, 0.0, inf, "stbl", (), 0.0925),
    "Na-22": (-5.18139, False, 0.0, 82107965.75940001, "2.6019 y", (("B+", 1.0), ("e+", 0.9057), ("EC", 0.0943), ), 0.0),
    "Mg-26": (-16.214544, False, 0.0, inf, "stbl", (), 0.11025),
    "P-32": (-24.30488, False, 0.0, 1232841.6, "14.269 d", (("B-", 1.0), ), 0.0),
    "S-32": (-26.015537100000003, False, 0.0, inf, "stbl", (), 0.9484999999999999),
    "Ar-38": (-34.71483, False, 0.0, inf, "stbl", (), 0.000629),
    "Ar-40": (-35.0399, False, 0.0, inf, "stbl", (), 0.996035),
    "K-40": (-33.5355, False, 0.0, 3.9383043648e+16, "1.248 Gy", (("B-", 0.8928), ("B+", 0.1072), ), 0.000117),
    "Ca-40": (-34.846402000000005, False, 0.0, inf, "stbl", (), 0.96941),
    "Ca-44": (-41.4687, False, 0.0, inf, "stbl", (), 0.02086),
    "Ti-50": (-51.43187, False, 0.0, inf, "stbl", (), 0.0518),
    "Fe-56": (-60.60716000000001, False, 0.0, inf, "stbl", (), 0.91754),
    "Co-60": (-61.650400000000005, False, 0.0, 166349179.7164, "5.2714 y", (("B-", 1.0), ), 0.0),
    "Ni-60": (-64.47319999999999, False, 0.0, inf, "stbl", (), 0.262231),
    "Ni-62": (-66.7464, False, 0.0, inf, "stbl", (), 0.036345),
    "Zn-68": (-70.0072, False, 0.0, inf, "stbl", (), 0.1845),
    "Ge-74": (-73.42245100000001, False, 0.0, inf, "stbl", (), 0.3652),
    "Kr-80": (-77.8935, False, 0.0, inf, "stbl", (), 0.022860000000000002),
    "Kr-89": (-76.53580000000001, False, 0.0, 189.0, "3.15 m", (("B-", 1.0), ), 0.0),
    "Kr-90": (-74.9593, False, 0.0, 32.32, "32.32 s", (("B-", 1.0), ), 0.0),
    "Kr-92": (-68.7693, False, 0.0, 1.84, "1.840 s", (("B-", 1.0), ("B-n", 0.000332), ), 0.0),
    "Rb-93": (-72.62, False, 0.0, 5.84, "5.84 s", (("B-", 1.0), ("B-n", 0.0139), ), 0.0),
    "Sr-86": (-84.52310000000001, False, 0.0, inf, "stbl", (), 0.0986),
    "Sr-90": (-85.95089999999999, False, 0.0, 912310730.66, "28.91 y", (("B-", 1.0), ), 0.0),
    "Sr-94": (-78.8457, False, 0.0, 75.3, "75.3 s", (("B-", 1.0), ), 0.0),
    "Sr-95": (-75.117, False, 0.0, 23.9, "23.90 s", (("B-", 1.0), ), 0.0),
    "Y-90": (-86.4969, False, 0.0, 230580.0, "64.05 h", (("B-", 1.0), ), 0.0),
    "Y-97": (-76.115, False, 0.0, 3.75, "3.75 s", (("B-", 1.0), ("B-n", 0.00055), ), 0.0),
    "Zr-90": (-88.77255000000001, False, 0.0, inf, "stbl", (), 0.5145000000000001),
    "Zr-92": (-88.45902000000001, False, 0.0, inf, "stbl", (), 0.17149999999999999),
    "Zr-98": (-81.282, False, 0.0, 30.7, "30.7 s", (("B-", 1.0), ), 0.0),
    "Zr-100": (-76.373, False, 0.0, 7.1, "7.1 s", (("B-", 1.0), ), 0.0),
    "Mo-99": (-85.97011, False, 0.0, 237355.2, "65.932 h", (("B-", 1.0), ), 0.0),
    "Mo-103": (-80.95400000000001, False, 0.0, 67.5, "67.5 s", (("B-", 1.0), ), 0.0),
    "Tc-99": (-87.3279, False, 0.0, 6661667078600.0, "211.1 ky", (("B-", 1.0), ), 0.0),
    "Tc-99m": (-87.1852, False, 0.14268360000000002, 21623.76, "6.0066 h", (("IT", 1.0), ), 0.0),
    "Ru-98": (-88.22500000000001, False, 0.0, inf, "stbl", (), 0.0187),
    "Ru-99": (-87.6254, False, 0.0, inf, "stbl", (), 0.1276),
    "Pd-104": (-89.39510000000001, False, 0.0, inf, "stbl", (), 0.1114),
    "Cd-110": (-90.348, False, 0.0, inf, "stbl", (), 0.1247),
    "Sn-116": (-91.52598, False, 0.0, inf, "stbl", (), 0.1454),
    "Sn-131": (-77.265, False, 0.0, 56.0, "56.0 s", (("B-", 1.0), ), 0.0),
    "Te-122": (-90.3133, False, 0.0, inf, "stbl", (), 0.0255),
    "Te-134": (-82.5338, False, 0.0, 2508.0, "41.8 m", (("B-", 1.0), ), 0.0),
    "Te-135": (-77.7288, False, 0.0, 19.0, "19.0 s", (("B-", 1.0), ), 0.0),
    "I-131": (-87.4427, False, 0.0, 693351.3600000001, "8.0249 d", (("B-", 1.0), ), 0.0),
    "I-136": (-79.545, False, 0.0, 83.4, "83.4 s", (("B-", 1.0), ), 0.0),
    "Xe-128": (-89.860534, False, 0.0, inf, "stbl", (), 0.0191),
    "Xe-131": (-88.413575, False, 0.0, inf, "stbl", (), 0.21231999999999998),
    "Xe-139": (-75.64460000000001, False, 0.0, 39.68, "39.68 s", (("B-", 1.0), ), 0.0),
    "Xe-140": (-72.9865, False, 0.0, 13.6, "13.60 s", (("B-", 1.0), ), 0.0),
    "Cs-137": (-86.5458, False, 0.0, 947970057.04, "30.04 y", (("B-", 1.0), ), 0.0),
    "Cs-141": (-74.477, False, 0.0, 24.84, "24.84 s", (("B-", 1.0), ("B-n", 0.000342), ), 0.0),
    "Ba-134": (-88.95, False, 0.0, inf, "stbl", (), 0.0242),
    "Ba-137": (-87.7214, False, 0.0, inf, "stbl", (), 0.11230000000000001),
    "Ba-137m": (-87.05974, False, 0.661659, 153.12, "2.552 m", (("IT", 1.0), ), 0.0),
    "Ba-141": (-79.732, False, 0.0, 1096.2, "18.27 m", (("B-", 1.0), ), 0.0),
    "Ba-143": (-73.937, False, 0.0, 14.5, "14.5 s", (("B-", 1.0), ), 0.0),
    "Ba-144": (-71.767, False, 0.0, 11.73, "11.73 s", (("B-", 1.0), ), 0.0),
    "Ce-140": (-88.0742, False, 0.0, inf, "stbl", (), 0.88449),
    "Sm-146": (-80.996, False, 0.0, 2145870968000000.0, "68 My", (("A", 1.0), ), 0.0),
    "Sm-152": (-74.763, False, 0.0, inf, "stbl", (), 0.26739999999999997),
    "Gd-158": (-70.69, False, 0.0, inf, "stbl", (), 0.2484),
    "Dy-164": (-65.9676, False, 0.0, inf, "stbl", (), 0.2826),
    "Yb-170": (-60.763929, False, 0.0, inf, "stbl", (), 0.029820000000000003),
    "Hf-176": (-54.5764, False, 0.0, inf, "stbl", (), 0.0526),
    "W-182": (-48.2461, False, 0.0, inf, "stbl", (), 0.265),
    "Os-188": (-41.1373, False, 0.0, inf, "stbl", (), 0.1324),
    "Pt-194": (-34.7601, False, 0.0, inf, "stbl", (), 0.32864),
    "Hg-200": (-29.5033, False, 0.0, inf, "stbl", (), 0.2314),
    "Tl-205": (-23.8208, False, 0.0, inf, "stbl", (), 0.70485),
    "Tl-207": (-21.034, False, 0.0, 286.2, "4.77 m", (("B-", 1.0), ), 0.0),
    "Tl-208": (-16.7501, False, 0.0, 183.18, "3.053 m", (("B-", 1.0), ), 0.0),
    "Tl-209": (-13.645, False, 0.0, 129.72, "2.162 m", (("B-", 1.0), ), 0.0),
    "Tl-210": (-9.247, False, 0.0, 78.0, "1.30 m", (("B-", 1.0), ), 0.0),
    "Pb-206": (-23.7855, False, 0.0, inf, "stbl", (), 0.24100000000000002),
    "Pb-207": (-22.452, False, 0.0, inf, "stbl", (), 0.221),
    "Pb-208": (-21.7485, False, 0.0, inf, "stbl", (), 0.524),
    "Pb-209": (-17.6146, False, 0.0, 11646.0, "3.235 h", (("B-", 1.0), ), 0.0),
    "Pb-210": (-14.7284, False, 0.0, 700563757.1999999, "22.20 y", (("B-", 1.0), ), 0.0),
    "Pb-211": (-10.493, False, 0.0, 2169.768, "36.1628 m", (("B-", 1.0), ), 0.0),
    "Pb-212": (-7.5489, False, 0.0, 38257.200000000004, "10.627 h", (("B-", 1.0), ), 0.0),
    "Pb-214": (-0.183, False, 0.0, 1623.6, "27.06 m", (("B-", 1.0), ), 0.0),
    "Bi-209": (-18.258599999999998, False, 0.0, 6.342942126e+26, "20.1 Ey", (("A", 1.0), ), 1.0),
    "Bi-210": (-14.7919, False, 0.0, 433036.8, "5.012 d", (("B-", 1.0), ), 0.0),
    "Bi-211": (-11.859, False, 0.0, 128.4, "2.14 m", (("A", 1.0), ("B-", 0.0027600000000000003), ), 0.0),
    "Bi-212": (-8.1179, False, 0.0, 3633.0, "60.55 m", (("B-", 0.6406000000000001), ("A", 0.3594), ("B-A", 0.00014000000000000001), ), 0.0),
    "Bi-213": (-5.232, False, 0.0, 2736.0, "45.60 m", (("B-", 0.9791), ("A", 0.0209), ), 0.0),
    "Bi-214": (-1.201, False, 0.0, 1194.0, "19.9 m", (("B-", 0.99979), ("A", 0.00021), ), 0.0),
    "Po-210": (-15.953100000000001, False, 0.0, 11955686.4, "138.376 d", (("A", 1.0), ), 0.0),
    "Po-211": (-12.432500000000001, False, 0.0, 0.516, "516 ms", (("A", 1.0), ), 0.0),
    "Po-212": (-10.3694, False, 0.0, 2.944e-07, "294.4 ns", (("A", 1.0), ), 0.0),
    "Po-213": (-6.654, False, 0.0, 3.705e-06, "3.705 us", (("A", 1.0), ), 0.0),
    "Po-214": (-4.47, False, 0.0, 0.00016346999999999998, "163.47 us", (("A", 1.0), ), 0.0),
    "Po-215": (-0.5418, False, 0.0, 0.001781, "1.781 ms", (("A", 1.0), ), 0.0),
    "Po-216": (1.7823, False, 0.0, 0.14400000000000002, "144.0 ms", (("A", 1.0), ), 0.0),
    "Po-218": (8.3567, False, 0.0, 185.82, "3.097 m", (("A", 0.9998), ("B-", 0.0002), ), 0.0),
    "At-217": (4.3950000000000005, False, 0.0, 0.032600000000000004, "32.6 ms", (("A", 0.99992), ), 0.0),
    "At-218": (8.1, False, 0.0, 1.28, "1.28 s", (("A", 1.0), ), 0.0),
    "Rn-218": (5.2174, False, 0.0, 0.03375, "33.75 ms", (("A", 1.0), ), 0.0),
    "Rn-219": (8.8293, False, 0.0, 3.96, "3.96 s", (("A", 1.0), ), 0.0),
    "Rn-220": (10.612, False, 0.0, 55.6, "55.6 s", (("A", 1.0), ), 0.0),
    "Rn-222": (16.372, False, 0.0, 330177.6, "3.8215 d", (("A", 1.0), ), 0.0),
    "Fr-221": (13.277000000000001, False, 0.0, 288.06, "4.801 m", (("A", 1.0), ), 0.0),
    "Fr-223": (18.3823, False, 0.0, 1320.0, "22.00 m", (("B-", 1.0), ), 0.0),
    "Ra-223": (17.2332, False, 0.0, 988001.28, "11.4352 d", (("A", 1.0), ), 0.0),
    "Ra-224": (18.8258, False, 0.0, 313770.24, "3.6316 d", (("A", 1.0), ), 0.0),
    "Ra-225": (21.993000000000002, False, 0.0, 1280448.0, "14.82 d", (("B-", 1.0), ), 0.0),
    "Ra-226": (23.6676, False, 0.0, 50491081600.0, "1.600 ky", (("A", 1.0), ), 0.0),
    "Ra-228": (28.9402, False, 0.0, 181452324.5, "5.75 y", (("B-", 1.0), ), 0.0),
    "Ac-225": (21.637, False, 0.0, 857001.6000000001, "9.9190 d", (("A", 1.0), ), 0.0),
    "Ac-227": (25.8495, False, 0.0, 687057392.872, "21.772 y", (("B-", 0.9862000000000001), ("A", 0.0138), ), 0.0),
    "Ac-228": (28.8947, False, 0.0, 22140.0, "6.15 h", (("B-", 1.0), ), 0.0),
    "Th-227": (25.8048, False, 0.0, 1615075.2000000002, "18.693 d", (("A", 1.0), ), 0.0),
    "Th-228": (26.7709, False, 0.0, 60352620.975, "1.9125 y", (("A", 1.0), ), 0.0),
    "Th-229": (29.5855, False, 0.0, 249804626216.0, "7.916 ky", (("A", 1.0), ), 0.0),
    "Th-230": (30.8625, False, 0.0, 2379392220400.0, "75.4 ky", (("A", 1.0), ), 0.0002),
    "Th-231": (33.8158, False, 0.0, 91872.0, "25.52 h", (("B-", 1.0), ), 0.0),
    "Th-232": (35.4467, False, 0.0, 4.41796964e+17, "14.0 Gy", (("A", 1.0), ), 0.9998),
    "Th-234": (40.613, False, 0.0, 2082844.8, "24.107 d", (("B-", 1.0), ), 0.0),
    "Pa-231": (33.4243, False, 0.0, 1030333633900.0, "32.65 ky", (("A", 1.0), ), 1.0),
    "Pa-233": (37.4894, False, 0.0, 2330640.0, "26.975 d", (("B-", 1.0), ), 0.0),
    "Pa-234": (40.339, False, 0.0, 24120.0, "6.70 h", (("B-", 1.0), ), 0.0),
    "Pa-234m": (40.4179, False, 0.079, 69.54, "1.159 m", (("B-", 1.0), ("IT", 0.0016), ), 0.0),
    "U-233": (36.9191, False, 0.0, 5023547049940.0, "159.19 ky", (("A", 1.0), ), 0.0),
    "U-234": (38.145, False, 0.0, 7747225333000.0, "245.5 ky", (("A", 1.0), ), 5.4000000000000005e-05),
    "U-235": (40.918800000000005, False, 0.0, 2.2216075904e+16, "704 My", (("A", 1.0), ), 0.007204),
    "U-236": (42.4446, False, 0.0, 739063206920000.0, "23.42 My", (("A", 1.0), ), 0.0),
    "U-238": (47.3077, False, 0.0, 1.40838560738e+17, "4.463 Gy", (("A", 1.0), ), 0.9927419999999999),
    "U-240": (52.7155, False, 0.0, 50760.0, "14.1 h", (("B-", 1.0), ), 0.0),
    "Np-237": (44.8716, False, 0.0, 67658049344000.01, "2.144 My", (("A", 1.0), ), 0.0),
    "Np-240": (52.316, False, 0.0, 3714.0, "61.9 m", (("B-", 1.0), ), 0.0),
    "Pu-239": (48.5882, False, 0.0, 760837485860.0, "24.11 ky", (("A", 1.0), ), 0.0),
    "Pu-240": (50.1253, False, 0.0, 207044991486.0, "6.561 ky", (("A", 1.0), ), 0.0),
    "Pu-242": (54.7169, False, 0.0, 11833847250000.0, "375 ky", (("A", 1.0), ), 0.0),
    "Pu-244": (59.806000000000004, False, 0.0, 2565578083800000.0, "81.3 My", (("A", 0.9987699999999999), ("SF", 0.00123), ), 0.0),
    "Am-241": (52.93430000000001, False, 0.0, 13651526187.6, "432.6 y", (("A", 1.0), ), 0.0),
    "Cm-248": (67.3927, False, 0.0, 10981810248000.0, "348 ky", (("A", 0.9161), ("SF", 0.0839), ), 0.0),
    "Cf-248": (67.238, False, 0.0, 28814400.0, "333.5 d", (("A", 1.0), ), 0.0),
    "Cf-252": (76.03460000000001, False, 0.0, 83468069.27, "2.645 y", (("A", 0.9689719999999999), ("SF", 0.031027999999999997), ), 0.0),
    "Fm-254": (80.9025, False, 0.0, 11664.0, "3.240 h", (("A", 0.999408), ("SF", 0.0005920000000000001), ), 0.0),
    "No-260": (95.61, True, 0.0, 0.106, "106 ms", (("SF", 1.0), ), 0.0),
    # EXCERPT-END
}
# fmt: on

# Which state a decay feeds when it is not the ground state. NUBASE does not
# hold this; entered from memory (standard decay schemes), to be checked:
# Th-234 beta decays to the 1.16 min isomer Pa-234m (taken as all of it);
# Cs-137 beta decays to Ba-137m in 94.7% of decays (the source of the
# 662 keV gamma line); Sr-90's daughter Y-90 and the others go to ground
# states. (parent Z, A, mode) -> (isomer, fraction).
FEEDS_ISOMER = {
    (90, 234, "B-"): ("m", 1.0),
    (55, 137, "B-"): ("m", 0.947),
    (42, 99, "B-"): ("m", 0.876),
}


# Nuclides whose 'B+' in NUBASE is in fact almost all electron capture
# although positron emission is energetically allowed. From memory, to be
# checked: K-40 (positrons in about 0.001% of decays).
CAPTURE_ONLY = {(19, 40)}


class Table:
    """Masses and decay data, from the files if present, else the excerpt and the formula."""

    def __init__(self, use_files=True):
        self.masses = {}  # (Z, A) -> (mass excess [MeV], estimated)
        self.nuclides = {}  # (Z, A, isomer) -> Nuclide
        self.from_files = False
        self.error = ""
        if use_files and available():
            try:
                self.masses = read_ame(DATA_DIR / MASS_FILE)
                self.nuclides = read_nubase(DATA_DIR / NUBASE_FILE)
                self.from_files = len(self.masses) > 3000 and len(self.nuclides) > 3000
                if not self.from_files:
                    self.error = "the data files are incomplete"
            except OSError as error:
                self.error = str(error)
        if not self.from_files:
            self.masses, self.nuclides = {}, {}
            for name, row in EXCERPT.items():
                parsed = parse_label(name)
                if parsed is None:
                    raise ValueError(f"excerpt row {name!r} is not a nuclide label")
                Z, A, isomer = parsed
                excess, estimated, excitation, half_life, text, modes, abundance = row
                self.nuclides[(Z, A, isomer)] = Nuclide(
                    Z, A, isomer, excess, estimated, excitation, half_life, text,
                    list(modes), abundance,
                )  # fmt: skip
                if not isomer:
                    self.masses[(Z, A)] = (excess, estimated)

    # -- masses

    def measured(self, Z, A):
        """Whether the table holds a mass for this nuclide (measured or estimated by AME)."""
        return (Z, A) in self.masses

    def source(self, Z, A):
        """Where mass_excess(Z, A) comes from, in words."""
        if (Z, A) in self.masses:
            where = "AME2020" if self.from_files else "AME2020 (built-in excerpt)"
            return where + (", estimated from systematics" if self.masses[(Z, A)][1] else "")
        return "semi-empirical mass formula (no table value)"

    def mass_excess(self, Z, A, isomer=""):
        """Mass excess [MeV] of the neutral atom: M c^2 - A u."""
        if isomer:
            state = self.nuclides.get((Z, A, isomer))
            if state is not None:
                return state.mass_excess
        if (Z, A) in self.masses:
            return self.masses[(Z, A)][0]
        if Z == 0 and A == 1:
            return EXCESS_N
        return semf_mass_excess(Z, A)

    def mass(self, Z, A, isomer=""):
        """Mass [MeV] of the neutral atom (of the free neutron for Z = 0)."""
        return A * U_MEV + self.mass_excess(Z, A, isomer)

    def nuclear_mass(self, Z, A):
        """Mass [MeV] of the bare nucleus: the atom less its electrons. The
        electrons' binding energy (13.6 eV for hydrogen, 0.7 MeV in all for
        uranium) is neglected."""
        return self.mass(Z, A) - Z * M_ELECTRON

    def binding(self, Z, A):
        """Binding energy [MeV]: Z M(H-1) + N M(n) - M(Z, A), atomic masses."""
        return Z * EXCESS_H1 + (A - Z) * EXCESS_N - self.mass_excess(Z, A)

    def exists(self, Z, A):
        """Whether Z, A is a nuclide this table can say anything about."""
        return 0 <= Z <= Z_MAX and A >= max(Z, 1) and (Z > 0 or A == 1)

    # -- decay data

    def get(self, Z, A, isomer=""):
        """The Nuclide, or None if NUBASE (or the excerpt) does not list it."""
        return self.nuclides.get((Z, A, isomer))

    def isotopes(self, Z):
        """Mass numbers listed for element Z, ascending."""
        return sorted(A for (z, A) in self.masses if z == Z)

    def default_isotope(self, Z):
        """A representative mass number for element Z: the most abundant
        natural isotope; failing that the longest-lived one listed; failing
        that (nothing listed) roughly the valley of stability, A = 2 Z + 0.006 Z^2."""
        listed = [n for (z, _, iso), n in self.nuclides.items() if z == Z and not iso]
        natural = [n for n in listed if n.abundance > 0.0]
        if natural:
            return max(natural, key=lambda n: n.abundance).A
        lived = [n for n in listed if n.half_life is not None]
        if lived:
            return max(lived, key=lambda n: n.half_life).A
        return max(int(round(2.0 * Z + 0.006 * Z * Z)), 1)


_table = None


def table():
    """The shared Table, built on first use."""
    global _table
    if _table is None:
        _table = Table()
    return _table


def reload():
    """Read the files again (after a download)."""
    global _table
    _table = Table()
    return _table


def format_time(seconds):
    """A time with a readable unit: '4.46 billion years', '164 microseconds'."""
    if seconds is None:
        return "unknown"
    if seconds == np.inf:
        return "stable"
    units = (
        (1e9 * YEAR, "billion years"), (1e6 * YEAR, "million years"), (YEAR, "years"),
        (DAY, "days"), (HOUR, "hours"), (MINUTE, "minutes"), (1.0, "s"), (1e-3, "ms"),
        (1e-6, "microseconds"), (1e-9, "ns"), (1e-12, "ps"), (1e-15, "fs"),
    )  # fmt: skip
    if seconds >= 1e12 * YEAR:
        return f"{seconds / YEAR:.3g} years"
    for size, unit in units:
        if seconds >= size:
            return f"{seconds / size:.3g} {unit}"
    return f"{seconds:.3g} s"


def format_energy(mev):
    """An energy given in MeV with a readable unit."""
    a = abs(mev)
    if a >= 1.0:
        return f"{mev:.4g} MeV"
    if a >= 1e-3:
        return f"{mev * 1e3:.4g} keV"
    return f"{mev * 1e6:.4g} eV"


if __name__ == "__main__":
    for written in download():
        print("downloaded", written)
    t = reload()
    print(len(t.masses), "masses,", len(t.nuclides), "states; from files:", t.from_files)

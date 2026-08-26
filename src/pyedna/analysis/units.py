"""Unit defaults and conversion helpers for analysis outputs."""

DEFAULT_ANALYSIS_UNITS = {
    "energy": "eV",
    "coupling": "cm-1",
    "distance": "angstrom",
}

SUPPORTED_ENERGY_UNITS = {"hartree", "au", "e_h", "ev", "cm-1"}
SUPPORTED_DISTANCE_UNITS = {"angstrom", "a", "bohr", "nm"}

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_CM = 219474.63136320
ANGSTROM_TO_BOHR = 1.8897261246257702


def merged_units(units=None):
    merged = dict(DEFAULT_ANALYSIS_UNITS)
    if units:
        merged.update(units)
    return merged


def energy_factor(unit):
    normalized = unit.lower()
    if normalized in ("hartree", "au", "e_h"):
        return 1.0
    if normalized == "ev":
        return HARTREE_TO_EV
    if normalized == "cm-1":
        return HARTREE_TO_CM
    raise ValueError(f"Unsupported energy unit '{unit}'")


def distance_factor(unit):
    normalized = unit.lower()
    if normalized in ("angstrom", "a"):
        return 1.0
    if normalized == "bohr":
        return ANGSTROM_TO_BOHR
    if normalized == "nm":
        return 0.1
    raise ValueError(f"Unsupported distance unit '{unit}'")

import regex as re
import numpy as np

ENERGIES_RE = re.compile(
    r"""
    ^\s*ENERGY\|\s+Total\s+FORCE_EVAL\s+\(\s*QS\s*\)\s+energy
    (?:\s+[\[\(][^\]\)]+[\]\)])?
    \s*:?\s+(?P<energy>[-+]?\d+\.\d+(?:[Ee][+-]?\d+)?)
    """,
    re.VERBOSE | re.MULTILINE
)


def parse_energies_list(output_file):

    energies_list = []
    for match in ENERGIES_RE.finditer(output_file):
        energies_list.append(match["energy"])
    if energies_list:
        return np.array(energies_list, dtype=float)
    else:
        return None

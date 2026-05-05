import regex as re
import numpy as np


# ENERGIES_RE = re.compile(
#     r"ENERGY\|\s*Total\sFORCE_EVAL\s\( QS \)\senergy\s\[hartree\]\s+(-?\d+\.\d+)"
# )

ENERGIES_RE = re.compile(
    r"ENERGY\|\s*Total\s+FORCE_EVAL\s+\(\s*QS\s*\)\s+energy\s+\[hartree\]\s+"
    r"(?P<energy>[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)"
    r"\s+FORCES\|\s*Atomic\s+forces\s+\[hartree/bohr\]"
)


def parse_energies_list(output_file):

    energies_list = []
    for match in ENERGIES_RE.finditer(output_file):
        energies_list.append(match.group(1))
    if energies_list:
        return np.array(energies_list, dtype=float)
    else:
        return None

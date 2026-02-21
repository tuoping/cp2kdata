import regex as re
import numpy as np

# ATOMIC_FORCES_RE = re.compile(
#     r"""
#     \sATOMIC\sFORCES\sin\s\[a\.u\.\]\s*\n
#     \n
#     \s\#.+\n
#     (
#         \s+(?P<atom>\d+)
#         \s+(?P<kind>\d+)
#         \s+(?P<element>\w+)
#         \s+(?P<x>[\s-]\d+\.\d+)
#         \s+(?P<y>[\s-]\d+\.\d+)
#         \s+(?P<z>[\s-]\d+\.\d+)
#         \n
#     )+
#     """,
#     re.VERBOSE
# )


FLOAT_RE = r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"

ATOMIC_FORCES_RE = re.compile(
    rf"""
    ^\s*FORCES\|\s+Atomic\s+forces\s+\[hartree/bohr\]\s*\n
    ^\s*FORCES\|\s+Atom\s+x\s+y\s+z\s+\|f\|\s*\n
    (?:
        ^\s*FORCES\|\s+
        (?P<atom>\d+)\s+                       # atom index
        (?P<x>{FLOAT_RE})\s+                   # Fx
        (?P<y>{FLOAT_RE})\s+                   # Fy
        (?P<z>{FLOAT_RE})\s+                   # Fz
        {FLOAT_RE}\s*                          # |f|, ignored
        \n
    )+
    """,
    re.VERBOSE | re.MULTILINE,
)


def parse_atomic_forces_list(output_file):
    atomic_forces_list = []
    for match in ATOMIC_FORCES_RE.finditer(output_file):
        atomic_forces = []
        for x, y, z in zip(*match.captures("x", "y", "z")):
            atomic_forces.append([x, y, z])
        atomic_forces_list.append(atomic_forces)
    if atomic_forces_list:
        return np.array(atomic_forces_list, dtype=float)
    else:
        return None

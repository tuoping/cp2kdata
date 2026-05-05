import regex as re
import numpy as np
'''
ATOMIC_FORCES_RE = re.compile(
    r"""
    ^\s*FORCES\|\s+Atomic\s+forces\s+\[hartree/bohr\]\s*\n     # block header
    ^\s*FORCES\|\s+Atom\s+x\s+y\s+z\s+\|f\|\s*\n               # column header
    (
        ^\s*FORCES\|\s+\d+\s+                                   # atom index
        (?P<x>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+
        (?P<y>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+
        (?P<z>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+
        [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s*                  # |f| (ignored)
        \n
    )+                                                          # one or more atoms
    """,
    re.VERBOSE | re.MULTILINE
)
'''

ATOMIC_FORCES_RE = re.compile(
    r"""
    ^\s*FORCES\|\s+Atomic\s+forces\s+\[hartree/bohr\]\s*\r?\n
    ^\s*FORCES\|\s+Atom\s+x\s+y\s+z\s+\|f\|\s*\r?\n

    (?P<forces_block>
        (?:
            ^\s*FORCES\|\s+\d+\s+
            [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s+
            [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s+
            [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s+
            [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s*
            \r?\n
        )+
    )

    ^\s*FORCES\|\s+Sum\s+
    [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s+
    [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s+
    [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s*\r?\n

    ^\s*FORCES\|\s+Total\s+atomic\s+force\s+
    [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s*\r?\n

    \r?\n
    ^\s*STRESS\|\s+Analytical\s+stress\s+tensor\s+\[bar\]\s*\r?\n
    ^\s*STRESS\|\s+x\s+y\s+z\s*\r?\n
    ^\s*STRESS\|\s+x\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s*\r?\n
    ^\s*STRESS\|\s+y\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s*\r?\n
    ^\s*STRESS\|\s+z\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s*\r?\n

    [\s\S]*?

    ^\s*\*{10,}\s*\r?\n
    ^\s*\*{3}\s*BRENT\s*-\s*NUMBER\s+OF\s+ENERGY\s+EVALUATIONS\s*:\s*
        (?P<n_eval>\d+)\s*\*{2,}\s*\r?\n
    ^\s*\*{10,}\s*\r?\n
    \r?\n

    ^\s*OPT\|\s+\*+
    """,
    re.VERBOSE | re.MULTILINE,
)

FORCE_LINE_RE = re.compile(
    r"""
    ^\s*FORCES\|\s+\d+\s+
    (?P<x>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+
    (?P<y>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+
    (?P<z>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+
    [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s*$
    """,
    re.VERBOSE | re.MULTILINE,
)


def parse_atomic_forces_list(output_file: str):
    """
    Parse one or more 'FORCES| Atomic forces [hartree/bohr]' blocks from a text blob.

    Returns:
        np.ndarray of shape (n_blocks, n_atoms, 3) with floats (x,y,z), or None if no blocks found.
    """
    atomic_forces_list = []
    for block_match  in ATOMIC_FORCES_RE.finditer(output_file):
        forces_block = block_match.group("forces_block")

        forces = [
            [float(line_match.group("x")),
             float(line_match.group("y")),
             float(line_match.group("z"))]
            for line_match in FORCE_LINE_RE.finditer(forces_block)
        ]
        atomic_forces_list.append(forces)
    if atomic_forces_list:
        return np.array(atomic_forces_list, dtype=float)
    else:
        return None


MD_ATOMIC_FORCES_RE = re.compile(
    r"""
    ^\s*FORCES\|\s+Atomic\s+forces\s+\[hartree/bohr\]\s*\r?\n
    ^\s*FORCES\|\s+Atom\s+x\s+y\s+z\s+\|f\|\s*\r?\n

    (?P<forces_block>
        (?:
            ^\s*FORCES\|\s+\d+\s+
            [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s+
            [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s+
            [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s+
            [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s*
            \r?\n
        )+
    )

    ^\s*FORCES\|\s+Sum\s+
    [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s+
    [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s+
    [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s*\r?\n

    ^\s*FORCES\|\s+Total\s+atomic\s+force\s+
    [+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?\s*\r?\n

    \r?\n
    ^\s*STRESS\|\s+Analytical\s+stress\s+tensor\s+\[bar\]\s*\r?\n
    ^\s*STRESS\|\s+x\s+y\s+z\s*\r?\n
    ^\s*STRESS\|\s+x\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s*\r?\n
    ^\s*STRESS\|\s+y\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s*\r?\n
    ^\s*STRESS\|\s+z\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s+
    [-+]?\d+\.\d+E[-+]\d+\s*\r?\n

    [\s\S]*?

    ^\s*MD\|\s+\*+
    """,
    re.VERBOSE | re.MULTILINE,
)

def parse_atomic_forces_list_md(output_file: str):
    """
    Parse one or more 'FORCES| Atomic forces [hartree/bohr]' blocks from a text blob.

    Returns:
        np.ndarray of shape (n_blocks, n_atoms, 3) with floats (x,y,z), or None if no blocks found.
    """
    atomic_forces_list = []
    for block_match  in MD_ATOMIC_FORCES_RE.finditer(output_file):
        forces_block = block_match.group("forces_block")

        forces = [
            [float(line_match.group("x")),
             float(line_match.group("y")),
             float(line_match.group("z"))]
            for line_match in FORCE_LINE_RE.finditer(forces_block)
        ]
        atomic_forces_list.append(forces)
    if atomic_forces_list:
        return np.array(atomic_forces_list, dtype=float)
    else:
        return None

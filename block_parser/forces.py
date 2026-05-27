import regex as re
import numpy as np

FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"

STATIC_ATOMIC_FORCES_RE = re.compile(
    rf"""
    ^\s*FORCES\|\s+Atomic\s+forces\s+\[hartree/bohr\]\s*\r?\n
    ^\s*FORCES\|\s+Atom\s+x\s+y\s+z\s+\|f\|\s*\r?\n

    (?P<forces_block>
        (?:
            ^\s*FORCES\|\s+\d+\s+
            {FLOAT}\s+
            {FLOAT}\s+
            {FLOAT}\s+
            {FLOAT}\s*
            \r?\n
        )+
    )

    ^\s*FORCES\|\s+Sum\s+
    {FLOAT}\s+
    {FLOAT}\s+
    {FLOAT}\s*\r?\n

    ^\s*FORCES\|\s+Total\s+atomic\s+force\s+
    {FLOAT}\s*(?:\r?\n)?
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


def parse_atomic_forces_list_static(output_file: str):
    """
    Parse one or more 'FORCES| Atomic forces [hartree/bohr]' blocks from a text blob.

    Returns:
        np.ndarray of shape (n_blocks, n_atoms, 3) with floats (x,y,z), or None if no blocks found.
    """
    atomic_forces_list = []
    for block_match  in STATIC_ATOMIC_FORCES_RE.finditer(output_file):
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

    # Anything between stress tensor and optional BRENT / OPT block
    (?:(?!^[^\S\r\n]*OPT\|)[\s\S])*?

    # Optional BRENT block
    (?:
        ^[^\S\r\n]*\*{10,}[^\S\r\n]*\r?\n
        ^[^\S\r\n]*\*{3}[^\S\r\n]*BRENT[^\S\r\n]*-[^\S\r\n]*NUMBER[^\S\r\n]+OF[^\S\r\n]+ENERGY[^\S\r\n]+EVALUATIONS[^\S\r\n]*:[^\S\r\n]*
            (?P<n_eval>\d+)[^\S\r\n]*\*{2,}[^\S\r\n]*\r?\n
        ^[^\S\r\n]*\*{10,}[^\S\r\n]*\r?\n
    )?

    [^\S\r\n]*

    ^[^\S\r\n]*OPT\|[^\S\r\n]*\*+[^\S\r\n]*\r?\n

    ^[^\S\r\n]*OPT\|[^\S\r\n]*Step[^\S\r\n]+number[^\S\r\n]+
        (?P<step>\d+)[^\S\r\n]*\r?\n

    ^[^\S\r\n]*OPT\|[^\S\r\n]*Optimization[^\S\r\n]+method[^\S\r\n]+
        (?P<method>\S+)[^\S\r\n]*\r?\n

    ^[^\S\r\n]*OPT\|[^\S\r\n]*Total[^\S\r\n]+energy[^\S\r\n]+\[hartree\][^\S\r\n]+
        (?P<energy>[-+]?\d+\.\d+)[^\S\r\n]*\r?\n

    # Optional internal pressure line
    (?:
        ^[^\S\r\n]*OPT\|[^\S\r\n]*Internal[^\S\r\n]+pressure[^\S\r\n]+\[bar\][^\S\r\n]+
            (?P<pressure>[-+]?\d+\.\d+)[^\S\r\n]*\r?\n
    )?
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
        step = int(block_match.group("step"))
        if step == 0:
            continue
        forces_block = block_match.group("forces_block")

        forces = [
            [float(line_match.group("x")),
             float(line_match.group("y")),
             float(line_match.group("z"))]
            for line_match in FORCE_LINE_RE.finditer(forces_block)
        ]
        atomic_forces_list.append(forces)
        print("        matched forces")
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

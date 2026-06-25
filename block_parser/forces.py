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

import regex as re

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
HSPACE = r"[^\S\r\n]"

ATOMIC_FORCES_RE_V23 = re.compile(
    rf"""
    ^{HSPACE}*ATOMIC{HSPACE}+FORCES{HSPACE}+in{HSPACE}+\[a\.u\.\]{HSPACE}*\r?\n
    (?:^{HSPACE}*\r?\n)*
    ^{HSPACE}*\#{HSPACE}+Atom{HSPACE}+Kind{HSPACE}+Element{HSPACE}+X{HSPACE}+Y{HSPACE}+Z{HSPACE}*\r?\n

    (?P<forces_block>
        (?:
            ^{HSPACE}*\d+{HSPACE}+\d+{HSPACE}+\S+{HSPACE}+
            {FLOAT}{HSPACE}+
            {FLOAT}{HSPACE}+
            {FLOAT}{HSPACE}*
            \r?\n
        )+
    )

    ^{HSPACE}*SUM{HSPACE}+OF{HSPACE}+ATOMIC{HSPACE}+FORCES{HSPACE}+
    {FLOAT}{HSPACE}+
    {FLOAT}{HSPACE}+
    {FLOAT}{HSPACE}+
    {FLOAT}{HSPACE}*\r?\n

    ^{HSPACE}*\r?\n

    ^{HSPACE}*STRESS\|{HSPACE}+Analytical{HSPACE}+stress{HSPACE}+tensor{HSPACE}+\[GPa\]{HSPACE}*\r?\n
    ^{HSPACE}*STRESS\|{HSPACE}+x{HSPACE}+y{HSPACE}+z{HSPACE}*\r?\n

    ^{HSPACE}*STRESS\|{HSPACE}+x{HSPACE}+
        (?P<xx>{FLOAT}){HSPACE}+
        (?P<xy>{FLOAT}){HSPACE}+
        (?P<xz>{FLOAT}){HSPACE}*\r?\n

    ^{HSPACE}*STRESS\|{HSPACE}+y{HSPACE}+
        (?P<yx>{FLOAT}){HSPACE}+
        (?P<yy>{FLOAT}){HSPACE}+
        (?P<yz>{FLOAT}){HSPACE}*\r?\n

    ^{HSPACE}*STRESS\|{HSPACE}+z{HSPACE}+
        (?P<zx>{FLOAT}){HSPACE}+
        (?P<zy>{FLOAT}){HSPACE}+
        (?P<zz>{FLOAT}){HSPACE}*\r?\n

    ^{HSPACE}*STRESS\|{HSPACE}*1/3{HSPACE}+Trace{HSPACE}+{FLOAT}{HSPACE}*\r?\n
    ^{HSPACE}*STRESS\|{HSPACE}*Determinant{HSPACE}+{FLOAT}{HSPACE}*\r?\n

    ^{HSPACE}*\r?\n

    ^{HSPACE}*STRESS\|{HSPACE}+Eigenvectors{HSPACE}+and{HSPACE}+eigenvalues{HSPACE}+of{HSPACE}+the{HSPACE}+analytical{HSPACE}+stress{HSPACE}+tensor{HSPACE}+\[GPa\]{HSPACE}*\r?\n
    ^{HSPACE}*STRESS\|{HSPACE}+1{HSPACE}+2{HSPACE}+3{HSPACE}*\r?\n
    ^{HSPACE}*STRESS\|{HSPACE}+Eigenvalues{HSPACE}+{FLOAT}{HSPACE}+{FLOAT}{HSPACE}+{FLOAT}{HSPACE}*\r?\n
    ^{HSPACE}*STRESS\|{HSPACE}+x{HSPACE}+{FLOAT}{HSPACE}+{FLOAT}{HSPACE}+{FLOAT}{HSPACE}*\r?\n
    ^{HSPACE}*STRESS\|{HSPACE}+y{HSPACE}+{FLOAT}{HSPACE}+{FLOAT}{HSPACE}+{FLOAT}{HSPACE}*\r?\n
    ^{HSPACE}*STRESS\|{HSPACE}+z{HSPACE}+{FLOAT}{HSPACE}+{FLOAT}{HSPACE}+{FLOAT}{HSPACE}*\r?\n

    ^{HSPACE}*\r?\n

    # Optional "Informations at step" block
    (?:
        ^{HSPACE}*-{{5,}}{HSPACE}+Informations{HSPACE}+at{HSPACE}+step{HSPACE}*={HSPACE}*
            (?P<info_step>\d+){HSPACE}*-{{5,}}{HSPACE}*\r?\n

        ^{HSPACE}*Optimization{HSPACE}+Method{HSPACE}*={HSPACE}*
            (?P<method>\S+){HSPACE}*\r?\n

        ^{HSPACE}*Total{HSPACE}+Energy{HSPACE}*={HSPACE}*
            (?P<energy>{FLOAT}){HSPACE}*\r?\n

        (?:
            ^{HSPACE}*Internal{HSPACE}+Pressure{HSPACE}+\[bar\]{HSPACE}*={HSPACE}*
                (?P<pressure>{FLOAT}){HSPACE}*\r?\n
        )?

        # Remaining lines until the dashed end of the information block
        (?:
            ^(?!{HSPACE}*-{{5,}}{HSPACE}*$)[^\r\n]*\r?\n
        )*

        ^{HSPACE}*-{{5,}}{HSPACE}*\r?\n

        (?:
            ^{HSPACE}*Estimated{HSPACE}+peak{HSPACE}+process{HSPACE}+memory{HSPACE}+after{HSPACE}+this{HSPACE}+step{HSPACE}+\[MiB\]{HSPACE}+
                (?P<memory_mib>{FLOAT}){HSPACE}*\r?\n
        )?

        ^{HSPACE}*\r?\n
    )?

    ^{HSPACE}*-{{5,}}{HSPACE}*\r?\n
    ^{HSPACE}*OPTIMIZATION{HSPACE}+STEP:{HSPACE}+
        (?P<step>\d+){HSPACE}*\r?\n
    ^{HSPACE}*-{{5,}}{HSPACE}*\r?\n
    """,
    re.VERBOSE | re.MULTILINE,
)

FORCE_LINE_RE_V23 = re.compile(
    rf"""
    ^{HSPACE}*\d+{HSPACE}+\d+{HSPACE}+\S+{HSPACE}+
    (?P<x>{FLOAT}){HSPACE}+
    (?P<y>{FLOAT}){HSPACE}+
    (?P<z>{FLOAT}){HSPACE}*$
    """,
    re.VERBOSE | re.MULTILINE,
)


ATOMIC_FORCES_RE_other = re.compile(
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


from .header_info import Cp2kInfo
def parse_atomic_forces_list(output_file: str, cp2k_info: Cp2kInfo):
    """
    Parse one or more 'FORCES| Atomic forces [hartree/bohr]' blocks from a text blob.

    Returns:
        np.ndarray of shape (n_blocks, n_atoms, 3) with floats (x,y,z), or None if no blocks found.
    """
    atomic_forces_list = []
    if cp2k_info.version in ['2023.2']:
        ATOMIC_FORCES_RE = ATOMIC_FORCES_RE_V23
        FORCE_LINE_RE_select = FORCE_LINE_RE_V23
    else:
        ATOMIC_FORCES_RE = ATOMIC_FORCES_RE_other
        FORCE_LINE_RE_select = FORCE_LINE_RE
    for block_match  in ATOMIC_FORCES_RE.finditer(output_file):
        step = int(block_match.group("step"))
        if step == 0:
            continue
        forces_block = block_match.group("forces_block")
        forces = [
            [float(line_match.group("x")),
             float(line_match.group("y")),
             float(line_match.group("z"))]
            for line_match in FORCE_LINE_RE_select.finditer(forces_block)
        ]
        atomic_forces_list.append(forces)
    if atomic_forces_list:
        return np.array(atomic_forces_list, dtype=float)
    else:
        raise RuntimeError
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

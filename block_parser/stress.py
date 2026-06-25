import regex as re
import numpy as np
from .header_info import Cp2kInfo

FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"

STATIC_STRESS_RE = re.compile(
    rf"""
    ^\s*STRESS\|\s+Analytical\s+stress\s+tensor\s+\[bar\]\s*\r?\n
    ^\s*STRESS\|\s+x\s+y\s+z\s*\r?\n
    ^\s*STRESS\|\s+x\s+
        (?P<xx>{FLOAT})\s+
        (?P<xy>{FLOAT})\s+
        (?P<xz>{FLOAT})\s*\r?\n
    ^\s*STRESS\|\s+y\s+
        (?P<yx>{FLOAT})\s+
        (?P<yy>{FLOAT})\s+
        (?P<yz>{FLOAT})\s*\r?\n
    ^\s*STRESS\|\s+z\s+
        (?P<zx>{FLOAT})\s+
        (?P<zy>{FLOAT})\s+
        (?P<zz>{FLOAT})\s*(?:\r?\n)?
    """,
    re.VERBOSE | re.MULTILINE,
)


def parse_stress_tensor_list_static(output_file):
    stress_tensor_list = []
    for match in STATIC_STRESS_RE.finditer(output_file):
        stress_tensor = [
            [match["xx"], match["xy"], match["xz"]],
            [match["yx"], match["yy"], match["yz"]],
            [match["zx"], match["zy"], match["zz"]]
        ]
        stress_tensor_list.append(stress_tensor)
    if stress_tensor_list:
        return np.array(stress_tensor_list, dtype=float)
    else:
        # raise Exception("stress not parsed")
        return None


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
HSPACE = r"[^\S\r\n]"

STRESS_RE_V23 = re.compile(
    rf"""
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


STRESS_RE_other = re.compile(
    r"""
    ^[^\S\r\n]*STRESS\|[^\S\r\n]+Analytical[^\S\r\n]+stress[^\S\r\n]+tensor[^\S\r\n]+\[bar\][^\S\r\n]*\r?\n
    ^[^\S\r\n]*STRESS\|[^\S\r\n]+x[^\S\r\n]+y[^\S\r\n]+z[^\S\r\n]*\r?\n

    ^[^\S\r\n]*STRESS\|[^\S\r\n]+x[^\S\r\n]+
        (?P<xx>[-+]?\d+\.\d+E[-+]\d+)[^\S\r\n]+
        (?P<xy>[-+]?\d+\.\d+E[-+]\d+)[^\S\r\n]+
        (?P<xz>[-+]?\d+\.\d+E[-+]\d+)[^\S\r\n]*\r?\n

    ^[^\S\r\n]*STRESS\|[^\S\r\n]+y[^\S\r\n]+
        (?P<yx>[-+]?\d+\.\d+E[-+]\d+)[^\S\r\n]+
        (?P<yy>[-+]?\d+\.\d+E[-+]\d+)[^\S\r\n]+
        (?P<yz>[-+]?\d+\.\d+E[-+]\d+)[^\S\r\n]*\r?\n

    ^[^\S\r\n]*STRESS\|[^\S\r\n]+z[^\S\r\n]+
        (?P<zx>[-+]?\d+\.\d+E[-+]\d+)[^\S\r\n]+
        (?P<zy>[-+]?\d+\.\d+E[-+]\d+)[^\S\r\n]+
        (?P<zz>[-+]?\d+\.\d+E[-+]\d+)[^\S\r\n]*\r?\n

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



def parse_stress_tensor_list(output_file,
                       cp2k_info: Cp2kInfo,
    ):
    stress_tensor_list = []
    steps_list = []
    if cp2k_info.version in ['2023.2']:
        STRESS_RE = STRESS_RE_V23
    else:
        STRESS_RE = STRESS_RE_other
    for match in STRESS_RE.finditer(output_file):
        step = int(match.group("step"))
        if step == 0:
            continue
        stress_tensor = [
            [match["xx"], match["xy"], match["xz"]],
            [match["yx"], match["yy"], match["yz"]],
            [match["zx"], match["zy"], match["zz"]]
        ]
        stress_tensor_list.append(stress_tensor)

        pressure = match.group("pressure") if match.groupdict().get("pressure") is not None else None

        steps_list.append({
            "step": step,
            "has_pressure": pressure is not None,
        })
    if stress_tensor_list:
        return np.array(stress_tensor_list, dtype=float), steps_list
    else:
        # raise Exception("stress not parsed")
        return None, None


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
MD_STRESS_RE = re.compile(
    rf"""
    ^\s*STRESS\|\s+Analytical\s+stress\s+tensor\s+\[bar\]\s*\r?\n
    ^\s*STRESS\|\s+x\s+y\s+z\s*\r?\n

    ^\s*STRESS\|\s+x\s+
        (?P<xx>{FLOAT})\s+
        (?P<xy>{FLOAT})\s+
        (?P<xz>{FLOAT})\s*\r?\n

    ^\s*STRESS\|\s+y\s+
        (?P<yx>{FLOAT})\s+
        (?P<yy>{FLOAT})\s+
        (?P<yz>{FLOAT})\s*\r?\n

    ^\s*STRESS\|\s+z\s+
        (?P<zx>{FLOAT})\s+
        (?P<zy>{FLOAT})\s+
        (?P<zz>{FLOAT})\s*\r?\n

    [\s\S]*?

    ^\s*MD\|\s+\*+
    """,
    re.VERBOSE | re.MULTILINE,
)


def parse_stress_tensor_list_md(output_file):
    stress_tensor_list = []
    for match in MD_STRESS_RE.finditer(output_file):
        stress_tensor = [
            [match["xx"], match["xy"], match["xz"]],
            [match["yx"], match["yy"], match["yz"]],
            [match["zx"], match["zy"], match["zz"]]
        ]
        stress_tensor_list.append(stress_tensor)
    if stress_tensor_list:
        return np.array(stress_tensor_list, dtype=float)
    else:
        # raise Exception("stress not parsed")
        return None

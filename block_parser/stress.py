import regex as re
import numpy as np

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


import regex as re

STRESS_RE = re.compile(
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



def parse_stress_tensor_list(output_file):
    stress_tensor_list = []
    steps_list = []
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

        print("        matched stress")
        pressure = match.group("pressure")

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

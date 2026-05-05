import regex as re
import numpy as np

STRESS_RE = re.compile(
    r"""
    ^\s*STRESS\|\s+Analytical\s+stress\s+tensor\s+\[bar\]\s*\r?\n
    ^\s*STRESS\|\s+x\s+y\s+z\s*\r?\n

    ^\s*STRESS\|\s+x\s+
        (?P<xx>[-+]?\d+\.\d+E[-+]\d+)\s+
        (?P<xy>[-+]?\d+\.\d+E[-+]\d+)\s+
        (?P<xz>[-+]?\d+\.\d+E[-+]\d+)\s*\r?\n

    ^\s*STRESS\|\s+y\s+
        (?P<yx>[-+]?\d+\.\d+E[-+]\d+)\s+
        (?P<yy>[-+]?\d+\.\d+E[-+]\d+)\s+
        (?P<yz>[-+]?\d+\.\d+E[-+]\d+)\s*\r?\n

    ^\s*STRESS\|\s+z\s+
        (?P<zx>[-+]?\d+\.\d+E[-+]\d+)\s+
        (?P<zy>[-+]?\d+\.\d+E[-+]\d+)\s+
        (?P<zz>[-+]?\d+\.\d+E[-+]\d+)\s*\r?\n

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



def parse_stress_tensor_list(output_file):
    stress_tensor_list = []
    for match in STRESS_RE.finditer(output_file):
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

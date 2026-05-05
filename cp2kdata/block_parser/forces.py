import regex as re
import numpy as np

ATOMIC_FORCES_RE = re.compile(
    r"""
    \sATOMIC\sFORCES\sin\s\[a\.u\.\]\s*\n
    \n
    \s\#.+\n
    (
        \s+(?P<atom>\d+)
        \s+(?P<kind>\d+)
        \s+(?P<element>\w+)
        \s+(?P<x>[\s-]\d+\.\d+)
        \s+(?P<y>[\s-]\d+\.\d+)
        \s+(?P<z>[\s-]\d+\.\d+)
        \n
    )+
    """,
    re.VERBOSE
)

FORCES_HEADER_RE = re.compile(
    r"^\s*FORCES\|\s+Atomic\s+forces\s+\[.*\]\s*$",
    re.MULTILINE
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

    atomic_forces_list = []
    lines = output_file.splitlines()
    in_block = False
    current = []
    for line in lines:
        if FORCES_HEADER_RE.match(line):
            in_block = True
            current = []
            continue
        if in_block:
            if line.lstrip().startswith("FORCES|"):
                parts = line.split()
                if len(parts) >= 5 and parts[1].isdigit():
                    current.append([parts[2], parts[3], parts[4]])
                    continue
                if parts[1] in ("Sum", "Total"):
                    if current:
                        atomic_forces_list.append(current)
                    in_block = False
            else:
                if current:
                    atomic_forces_list.append(current)
                in_block = False

    if in_block and current:
        atomic_forces_list.append(current)
    if atomic_forces_list:
        return np.array(atomic_forces_list, dtype=float)
    return None

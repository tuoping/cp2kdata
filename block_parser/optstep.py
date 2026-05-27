import regex as re

OPT_STEP_RE = re.compile(
    r"""
    ^[^\S\r\n]*OPT\|[^\S\r\n]*\*+[^\S\r\n]*\r?\n
    ^[^\S\r\n]*OPT\|[^\S\r\n]*Step[^\S\r\n]+number[^\S\r\n]+
        (?P<step>\d+)[^\S\r\n]*\r?\n

    ^[^\S\r\n]*OPT\|[^\S\r\n]*Optimization[^\S\r\n]+method[^\S\r\n]+
        (?P<method>\S+)[^\S\r\n]*\r?\n

    ^[^\S\r\n]*OPT\|[^\S\r\n]*Total[^\S\r\n]+energy[^\S\r\n]+\[hartree\][^\S\r\n]+
        (?P<energy>[-+]?\d+\.\d+)[^\S\r\n]*\r?\n

    (?:
        ^[^\S\r\n]*OPT\|[^\S\r\n]*Internal[^\S\r\n]+pressure[^\S\r\n]+\[bar\][^\S\r\n]+
            (?P<pressure>[-+]?\d+\.\d+)[^\S\r\n]*\r?\n
    )?
    """,
    re.VERBOSE | re.MULTILINE,
)


def parse_opt_step(output_text):
    steps_list = []

    for match in OPT_STEP_RE.finditer(output_text):
        step = int(match.group("step"))

        if step == 0:
            continue

        pressure = match.group("pressure")

        steps_list.append({
            "step": step,
            "has_pressure": pressure is not None,
        })

    return steps_list

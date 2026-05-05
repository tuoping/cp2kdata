import regex as re

OPT_STEP_RE = re.compile(
    r"""
    ^\s*OPT\|\s+\*+\s*\r?\n
    ^\s*OPT\|\s*Step\s+number\s*
    (?P<step>\d+)
    (?P<block>[\s\S]*?)
    (?=^\s*OPT\|\s+\*+|\Z)
    """,
    re.VERBOSE | re.MULTILINE,
)

def parse_opt_step(output_file) -> float:
    steps_list = []
    
    for match in OPT_STEP_RE.finditer(output_file):
        step = int(match.group("step"))
        if step == 0:
            continue
        block = match.group("block")
    
        has_pressure_deviation = bool(
            re.search(r"^\s*OPT\|\s*Pressure\s+deviation\s+\[bar\]", block, re.MULTILINE)
        )
    
        steps_list.append({
            "step": step,
            "has_pressure_deviation": has_pressure_deviation,
        })
    return steps_list

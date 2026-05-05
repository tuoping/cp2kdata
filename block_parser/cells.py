import regex as re
import numpy as np
from ase.geometry.cell import cellpar_to_cell
from ase.geometry.cell import cell_to_cellpar
from typing import List
from .header_info import Cp2kInfo
from ..units import au2A

ALL_CELL_RE = re.compile(
    r"""
    ^\s*OPT\|\s*Pressure\s+deviation\s+\[bar\]\s*
        (?P<pressure_deviation>[-+]?\d+\.\d+)\s*\r?\n
    ^\s*OPT\|\s*Pressure\s+tolerance\s+\[bar\]\s*
        (?P<pressure_tolerance>[-+]?\d+\.\d+)\s*\r?\n
    ^\s*OPT\|\s*Pressure\s+is\s+converged\s*
        (?P<pressure_converged>YES|NO)\s*\r?\n
    ^\s*OPT\|\s*\*+\s*\r?\n
    ^\s*OPT\|\s*Estimated\s+peak\s+process\s+memory\s+after\s+this\s+step\s+\[MiB\]\s*
        (?P<memory_mib>\d+)\s*\r?\n

    \s*\r?\n

    ^\s*CELL\|\s*Volume\s+\[angstrom\^3\]:\s*
        (?P<volume>[-+]?\d+\.\d+)\s*\r?\n

    ^\s*CELL\|\s*Vector\s+a\s+\[angstrom\]:\s*
        (?P<xx>[-+]?\d+\.\d+)\s+
        (?P<xy>[-+]?\d+\.\d+)\s+
        (?P<xz>[-+]?\d+\.\d+)\s+
        \|a\|\s*=\s*\S+\s*\r?\n

    ^\s*CELL\|\s*Vector\s+b\s+\[angstrom\]:\s*
        (?P<yx>[-+]?\d+\.\d+)\s+
        (?P<yy>[-+]?\d+\.\d+)\s+
        (?P<yz>[-+]?\d+\.\d+)\s+
        \|b\|\s*=\s*\S+\s*\r?\n

    ^\s*CELL\|\s*Vector\s+c\s+\[angstrom\]:\s*
        (?P<zx>[-+]?\d+\.\d+)\s+
        (?P<zy>[-+]?\d+\.\d+)\s+
        (?P<zz>[-+]?\d+\.\d+)\s+
        \|c\|\s*=\s*\S+
    """,
    re.VERBOSE | re.MULTILINE,
)


INIT_CELL_RE = re.compile(
    r"""
    ^\s*CELL\|\s*Volume\s+\[angstrom\^3\]:\s*
        (?P<volume>[-+]?\d+\.\d+)\s*\r?\n

    ^\s*CELL\|\s*Vector\s+a\s+\[angstrom\]:\s*
        (?P<xx>[-+]?\d+\.\d+)\s+
        (?P<xy>[-+]?\d+\.\d+)\s+
        (?P<xz>[-+]?\d+\.\d+)\s+
        \|a\|\s*=\s*\S+\s*\r?\n

    ^\s*CELL\|\s*Vector\s+b\s+\[angstrom\]:\s*
        (?P<yx>[-+]?\d+\.\d+)\s+
        (?P<yy>[-+]?\d+\.\d+)\s+
        (?P<yz>[-+]?\d+\.\d+)\s+
        \|b\|\s*=\s*\S+\s*\r?\n

    ^\s*CELL\|\s*Vector\s+c\s+\[angstrom\]:\s*
        (?P<zx>[-+]?\d+\.\d+)\s+
        (?P<zy>[-+]?\d+\.\d+)\s+
        (?P<zz>[-+]?\d+\.\d+)\s+
        \|c\|\s*=\s*\S+
    """,
    re.VERBOSE | re.MULTILINE,
)


def parse_all_cells(output_file):
    for match in INIT_CELL_RE.finditer(output_file):
        cell = [
            [match["xx"], match["xy"], match["xz"]],
            [match["yx"], match["yy"], match["yz"]],
            [match["zx"], match["zy"], match["zz"]]
        ]
        init_cell = cell
        break
    all_cells = [init_cell]
    for match in ALL_CELL_RE.finditer(output_file):
        # print(match)
        cell = [
            [match["xx"], match["xy"], match["xz"]],
            [match["yx"], match["yy"], match["yz"]],
            [match["zx"], match["zy"], match["zz"]]
        ]
        all_cells.append(cell)

    if all_cells:
        return np.array(all_cells, dtype=float)
    else:
        return None


ALL_MD_CELL_RE_V7 = re.compile(
    r"""
    \sCELL\sLNTHS\[bohr\]\s{13}=\s
    \s{3}(?P<a>\d\.\d{7}E(\+|\-)\d{2})
    \s{3}(?P<b>\d\.\d{7}E(\+|\-)\d{2})
    \s{3}(?P<c>\d\.\d{7}E(\+|\-)\d{2})
    \n
    #skip one line
    (.{80}\n){1}
    #parse angles
    \sCELL\sANGLS\[deg\]\s{14}=\s
    \s{3}(?P<alpha>\d\.\d{7}E(\+|\-)\d{2})
    \s{3}(?P<beta>\d\.\d{7}E(\+|\-)\d{2})
    \s{3}(?P<gamma>\d\.\d{7}E(\+|\-)\d{2})
    """,
    re.VERBOSE
)

ALL_MD_CELL_RE_V2023 = re.compile(
    r"""
    #parse cell lengths
    \sMD\|\sCell\slengths\s\[bohr\]\s{8}
    \s{2}(?P<a>\d\.\d{8}E(\+|\-)\d{2})
    \s{2}(?P<b>\d\.\d{8}E(\+|\-)\d{2})
    \s{2}(?P<c>\d\.\d{8}E(\+|\-)\d{2})
    \n
    #skip three lines
    (.{80}\n){3}
    #parse angles
    (\sMD\|\sCell\sangles\s\[deg\]\s{10}
    \s{2}(?P<alpha>\d\.\d{8}E(\+|\-)\d{2})
    \s{2}(?P<beta>\d\.\d{8}E(\+|\-)\d{2})
    \s{2}(?P<gamma>\d\.\d{8}E(\+|\-)\d{2}))?
    """,
    re.VERBOSE
)


def parse_all_md_cells(output_file: List[str],
                       cp2k_info: Cp2kInfo,
                       init_cell_info=None):
    # init_cell_info are used for npt_I parse.
    # because npt_I doesn't include angle info in MD| block

    # notice that the cell of step 0 is excluded from MD| block

    # choose parser according to cp2k_info.version
    if cp2k_info.version in ['9.1', '2022.2', '2023.1', '2023.2', '2024.1']:
        ALL_MD_CELL_RE = ALL_MD_CELL_RE_V2023
    elif cp2k_info.version in ['7.1']:
        ALL_MD_CELL_RE = ALL_MD_CELL_RE_V7
    else:
        WARNING = f"cp2k version={cp2k_info.version} is not supported yet \
                    for parsing MD cell from cp2k log files."
        raise NotImplementedError(WARNING)

    all_md_cells = []
    if init_cell_info is None:
        # for NPT_F parser, cell info is complete in MD| block
        for match in ALL_MD_CELL_RE.finditer(output_file):
            # print(match)
            cell = [match["a"], match["b"], match["c"],
                    match["alpha"], match["beta"], match["gamma"]]
            cell = np.array(cell, dtype=float)
            # convert bohr to angstrom
            cell[:3] = cell[:3] * au2A
            # make sure cell length are in angstrom and cell angles are in degree before sent to cellpar_to_cell
            #TODO: replace this cellpar_to_cell with more accurate functions in the future
            cell = cellpar_to_cell(cell)
            all_md_cells.append(cell)
    else:
        # for NPT_I parser, cell angle info is lost in MD| block
        init_cell_param = cell_to_cellpar(init_cell_info)
        init_cell_angles = init_cell_param[3:]
        for match in ALL_MD_CELL_RE.finditer(output_file):
            # print(match)
            cell = [match["a"], match["b"], match["c"],
                    match["alpha"], match["beta"], match["gamma"]]
            cell = np.array(cell, dtype=float)
            cell[3:] = init_cell_angles
            # convert bohr to angstrom
            cell[:3] = cell[:3] * au2A
            # make sure cell length are in angstrom and cell angles are in degree before sent to cellpar_to_cell
            #TODO: replace this cellpar_to_cell with more accurate functions in the future
            cell = cellpar_to_cell(cell)
            all_md_cells.append(cell)

    if all_md_cells:
        return np.array(all_md_cells, dtype=float)
    else:
        return None

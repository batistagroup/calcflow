# Placeholder for TDDFT Block Parsers

import re
from collections.abc import Sequence

from calcflow.parsers.qchem.typing import (
    Atom,
    DipoleMomentData,
    ExcitedStateAtomPopulation,
    ExcitedStateDetailedAnalysis,
    ExcitedStateExcitonDifferenceDM,
    ExcitedStateMulliken,
    ExcitedStateMultipole,
    ExcitedStateNOData,
    LineIterator,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- Excited State Analysis Block (Unrelaxed Density Matrices) ---
EXCITED_STATE_ANALYSIS_HEADER_PAT = re.compile(r"^\s*Excited State Analysis\s*$")
UNRELAXED_DM_HEADER_PAT = re.compile(r"^\s*Analysis of Unrelaxed Density Matrices\s*$")
STATE_HEADER_PAT = re.compile(r"^\s*(Singlet|Triplet)\s+(\d+)\s*:\s*$")

# --- Ground State Reference Patterns (within Excited State Analysis block) ---
GROUND_STATE_REF_HEADER_PAT = re.compile(r"^\s*Ground State \(Reference\)\s*:\s*$")
# Note: NOS_HEADER_PAT, MULTIPOLE_DM_HEADER_PAT are reused for Ground State Ref.

# --- Sub-section patterns for Excited State Analysis ---
# NOs
NOS_HEADER_PAT = re.compile(r"^\s*NOs\s*$")
NOS_FRONTIER_PAT = re.compile(r"^\s+Occupation of frontier NOs:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)")
NOS_NUM_ELECTRONS_PAT = re.compile(r"^\s+Number of electrons:\s*(-?\d+\.\d+)")
NOS_UNPAIRED_PAT = re.compile(r"^\s+Number of unpaired electrons: n_u =\s*(-?\d+\.\d+), n_u,nl =\s*(-?\d+\.\d+)")
NOS_PR_NO_PAT = re.compile(r"^\s+NO participation ratio \(PR_NO\):\s*(-?\d+\.\d+)")

# Mulliken Population (State/Difference DM)
MULLIKEN_POP_HEADER_PAT = re.compile(r"^\s*Mulliken Population Analysis \(State/Difference DM\)\s*$")
# Atom      Charge (e)              h+              e-           Del q
# --------------------------------------------------------------------
#  1 H       -0.283169        0.016896       -0.530620       -0.513723
MULLIKEN_ATOM_LINE_PAT = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)
MULLIKEN_SUM_LINE_PAT = re.compile(r"^\s*Sum:\s+(-?\d+\.\d+)")  # Marks end of atom list for Mulliken

# Mulliken Population (Ground State DM within Excited State Analysis)
MULLIKEN_GS_POP_HEADER_PAT = re.compile(r"^\s*Mulliken Population Analysis \(State DM\)\s*$")
# Atom      Charge (e)
# --------------------
#  1 H        0.230554
MULLIKEN_GS_ATOM_LINE_PAT = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s*$")

# Multipole Moment Analysis (Density Matrix)
MULTIPOLE_DM_HEADER_PAT = re.compile(r"^\s*Multipole moment analysis of the density matrix\s*$")
MOLECULAR_CHARGE_PAT = re.compile(r"^\s+Molecular charge:\s+(-?\d+\.\d+)")
NUM_ELECTRONS_MULTIPOLE_PAT = re.compile(
    r"^\s+Number of electrons:\s+(-?\d+\.\d+)"
)  # Can be distinct from NOs num_electrons
CENTER_ELECTRONIC_CHARGE_PAT = re.compile(
    r"^\s+Center of electronic charge \[Ang\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"
)
CENTER_NUCLEAR_CHARGE_PAT = re.compile(
    r"^\s+Center of nuclear charge \[Ang\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"
)
# Dipole Moment [D]:                     1.409939  -> This is the total dipole for this section
DIPOLE_TOTAL_PAT = re.compile(r"^\s+Dipole moment \[D\]:\s*(-?\d+\.\d+)")
DIPOLE_MOMENT_COMPONENTS_PAT = re.compile(
    r"^\s+Cartesian components \[D\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"
)
# RMS size of the density [Ang]:         0.950243
RMS_TOTAL_SIZE_PAT = re.compile(r"^\s+RMS size of the density \[Ang\]:\s*(-?\d+\.\d+)")
RMS_DENSITY_SIZE_COMPONENTS_PAT = re.compile(
    r"^\s+Cartesian components \[Ang\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"
)

# Exciton Analysis (Difference Density Matrix)
EXCITON_DIFF_DM_HEADER_PAT = re.compile(r"^\s*Exciton analysis of the difference density matrix\s*$")
EXCITON_RH_PAT = re.compile(r"^\s*<r_h> \[Ang\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]")
EXCITON_RE_PAT = re.compile(r"^\s*<r_e> \[Ang\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]")
EXCITON_RE_RH_SEP_PAT = re.compile(r"^\s*\|<r_e - r_h>\| \[Ang\]:\s*(-?\d+\.\d+)")
EXCITON_SIZE_HEADER_PAT = re.compile(
    r"^\s+(Hole|Electron) size \[Ang\]:"
)  # Total is on this line if present after colon
EXCITON_SIZE_VALUE_PAT = re.compile(r"^\s+(Hole|Electron) size \[Ang\]:\s*(-?\d+\.\d+)")
EXCITON_SIZE_COMPONENTS_PAT = re.compile(
    r"^\s+Cartesian components \[Ang\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"
)

SECTION_SEPARATOR_PAT = re.compile(r"^\s*-{20,}\s*$")  # A line of dashes, often ends state blocks
BLANK_LINE_PAT = re.compile(r"^\s*$")


class UnrelaxedExcitedStatePropertiesParser(SectionParser):  # Renamed from ExcitedStateAnalysisParser
    """Parses the 'Analysis of Unrelaxed Density Matrices' for each excited state."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        # This parser is expected to be called after the Ground State (Reference) block.
        return UNRELAXED_DM_HEADER_PAT.search(line) is not None

    def parse(self, iterator: LineIterator, first_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting parsing of Unrelaxed Density Matrices for excited states.")
        # first_line is "Analysis of Unrelaxed Density Matrices"
        # Consume this line, the actual state parsing starts from the next line.

        inter_state_line_buffer: list[str] = []

        while True:
            try:
                if inter_state_line_buffer:
                    line = inter_state_line_buffer.pop(0)
                else:
                    line = next(iterator)
            except StopIteration:
                logger.debug("Reached end of file while looking for next unrelaxed excited state.")
                break

            # Check for end of this specific analysis type or start of another major block
            # e.g., "Transition Density Matrix Analysis" or "SA-NTO Decomposition"
            if (
                re.search(r"^\s*Transition Density Matrix Analysis\s*$", line)
                or re.search(r"^\s*SA-NTO Decomposition\s*$", line)
                or
                # Add other known top-level block headers that might follow ESA
                EXCITED_STATE_ANALYSIS_HEADER_PAT.search(line)  # Should not happen if structured well
            ):
                logger.debug(f"Found start of next major block ('{line.strip()}'). Ending Unrelaxed DM Analysis.")
                inter_state_line_buffer.insert(0, line)  # Push back this line for the main orchestrator
                break

            match_state = STATE_HEADER_PAT.search(line)
            if not match_state:
                if SECTION_SEPARATOR_PAT.search(line) and not line.strip("-\\s"):
                    logger.debug("Separator found between unrelaxed excited states or before next major block.")
                elif (
                    not BLANK_LINE_PAT.search(line) and line.strip()
                ):  # Log if not blank and not a known separator/header
                    logger.debug(f"Skipping non-state-header line in Unrelaxed DM: {line.strip()}")
                # If not a state header, and not a major block ender, it's either consumed (blank/sep) or logged.
                # The loop continues to get the next line.
                continue

            multiplicity = match_state.group(1)
            state_number = int(match_state.group(2))
            logger.debug(f"Parsing unrelaxed DM analysis for {multiplicity} {state_number}")

            no_data_es: ExcitedStateNOData | None = None
            mulliken_data_es: ExcitedStateMulliken | None = None
            multipole_data_es: ExcitedStateMultipole | None = None
            exciton_data_es: ExcitedStateExcitonDifferenceDM | None = None

            parsed_nos_for_es = False
            parsed_mulliken_for_es = False
            parsed_multipole_for_es = False
            parsed_exciton_for_es = False

            intra_state_line_buffer: list[str] = []  # Buffer for lines within the current state

            MAX_SUBSECTION_LINES = 50
            for _i_subsec in range(MAX_SUBSECTION_LINES):
                sub_section_line: str
                try:
                    if intra_state_line_buffer:
                        sub_section_line = intra_state_line_buffer.pop(0)
                    else:
                        sub_section_line = next(iterator)
                except StopIteration:
                    logger.warning(f"File ended while parsing subsections for unrelaxed {multiplicity} {state_number}.")
                    break

                # End of current state's subsections?
                if STATE_HEADER_PAT.search(sub_section_line) or (
                    SECTION_SEPARATOR_PAT.search(sub_section_line) and not sub_section_line.strip("-\\s")
                ):
                    logger.debug(
                        f"End of subsections for unrelaxed {multiplicity} {state_number} due to: {sub_section_line.strip()}"
                    )
                    inter_state_line_buffer.insert(0, sub_section_line)
                    break

                # End of overall unrelaxed DM analysis block?
                if re.search(r"^\s*Transition Density Matrix Analysis\s*$", sub_section_line) or re.search(
                    r"^\s*SA-NTO Decomposition\s*$", sub_section_line
                ):
                    logger.debug(
                        f"Next major block '{sub_section_line.strip()}' found. Ending current unrelaxed state parsing."
                    )
                    inter_state_line_buffer.insert(0, sub_section_line)
                    break

                if NOS_HEADER_PAT.search(sub_section_line) and not parsed_nos_for_es:
                    no_data_es = self._parse_no_data(iterator, sub_section_line)  # type: ignore
                    if no_data_es and no_data_es.participation_ratio_pr_no is not None:
                        parsed_nos_for_es = True
                    intra_state_line_buffer.clear()  # Assume sub-parser handles iterator
                    continue

                # Note: For excited states, it's MULLIKEN_POP_HEADER_PAT (State/Difference DM)
                if MULLIKEN_POP_HEADER_PAT.search(sub_section_line) and not parsed_mulliken_for_es:
                    mulliken_data_es = self._parse_mulliken_state_dm(iterator, sub_section_line, results.input_geometry)  # type: ignore
                    if mulliken_data_es:
                        parsed_mulliken_for_es = True
                    intra_state_line_buffer.clear()
                    continue

                if MULTIPOLE_DM_HEADER_PAT.search(sub_section_line) and not parsed_multipole_for_es:
                    multipole_data_es = self._parse_multipole_state_dm(iterator, sub_section_line)  # type: ignore
                    if multipole_data_es:
                        parsed_multipole_for_es = True
                    intra_state_line_buffer.clear()
                    continue

                if EXCITON_DIFF_DM_HEADER_PAT.search(sub_section_line) and not parsed_exciton_for_es:
                    exciton_data_es = self._parse_exciton_state_diff_dm(iterator, sub_section_line)  # type: ignore
                    if exciton_data_es:
                        parsed_exciton_for_es = True
                    intra_state_line_buffer.clear()
                    continue

                if not sub_section_line.strip():
                    continue

                logger.debug(
                    f"Buffering unexpected line in unrelaxed {multiplicity} {state_number}: {sub_section_line.strip()}"
                )
                intra_state_line_buffer.append(sub_section_line)
                if len(intra_state_line_buffer) > 10:
                    logger.warning(f"Too many unexpected lines for unrelaxed {multiplicity} {state_number}.")
                    inter_state_line_buffer.extend(intra_state_line_buffer)
                    intra_state_line_buffer.clear()
                    break

            # Store parsed data for the current excited state
            if parsed_nos_for_es or parsed_mulliken_for_es or parsed_multipole_for_es or parsed_exciton_for_es:
                results.excited_state_detailed_analyses_list.append(
                    ExcitedStateDetailedAnalysis(
                        state_number=state_number,
                        multiplicity=multiplicity,
                        no_data=no_data_es,
                        mulliken_analysis=mulliken_data_es,
                        multipole_analysis=multipole_data_es,
                        exciton_difference_dm_analysis=exciton_data_es,
                    )
                )
            else:
                logger.warning(f"No sub-section data for unrelaxed {multiplicity} {state_number}.")

            if intra_state_line_buffer:  # Push back any unused lines from this state
                inter_state_line_buffer.extend(intra_state_line_buffer)

        # Push back any remaining unconsumed lines from inter_state_line_buffer (e.g. the major block terminator)
        if inter_state_line_buffer:
            iterator = iter(inter_state_line_buffer + list(iterator))

        logger.info(
            f"Parsed unrelaxed DM analysis for {len(results.excited_state_detailed_analyses_list)} excited states."
        )

    def _parse_no_data(self, iterator: LineIterator, first_no_line: str) -> ExcitedStateNOData:
        frontier_occ: list[float] | None = None
        num_e: float | None = None
        nu: float | None = None
        nunl: float | None = None
        pr_no: float | None = None
        line = first_no_line
        for _i in range(5):
            if not NOS_HEADER_PAT.search(line) and _i == 0:
                if line is not first_no_line:
                    iterator = iter([line] + list(iterator))
                return ExcitedStateNOData()
            if m := NOS_FRONTIER_PAT.search(line):
                frontier_occ = [float(m.group(1)), float(m.group(2))]
            if m := NOS_NUM_ELECTRONS_PAT.search(line):
                num_e = float(m.group(1))
            if m := NOS_UNPAIRED_PAT.search(line):
                nu, nunl = float(m.group(1)), float(m.group(2))
            if m := NOS_PR_NO_PAT.search(line):
                pr_no = float(m.group(1))
                break
            try:
                prev_line = line
                line = next(iterator)
                if (
                    SECTION_SEPARATOR_PAT.search(line)
                    or MULLIKEN_POP_HEADER_PAT.search(line)
                    or MULTIPOLE_DM_HEADER_PAT.search(line)
                    or STATE_HEADER_PAT.search(line)
                    or EXCITON_DIFF_DM_HEADER_PAT.search(line)
                ):  # Added EXCITON_DIFF_DM_HEADER_PAT
                    iterator = iter([line] + list(iterator))
                    line = prev_line
                    break
            except StopIteration:
                break
        return ExcitedStateNOData(frontier_occ, num_e, nu, nunl, pr_no)

    def _parse_mulliken_state_dm(
        self, iterator: LineIterator, first_mulliken_line: str, input_geometry: Sequence[Atom] | None
    ) -> ExcitedStateMulliken | None:
        if not MULLIKEN_POP_HEADER_PAT.search(first_mulliken_line):
            return None
        populations: list[ExcitedStateAtomPopulation] = []
        line_buffer: list[str] = []
        try:
            _ = next(iterator)
            _ = next(iterator)  # Skip headers
            while True:
                line = next(iterator)
                line_buffer.append(line)
                if MULLIKEN_SUM_LINE_PAT.search(line) or not line.strip():
                    line_buffer.pop()
                    iterator = iter(line_buffer + list(iterator))
                    break
                if m := MULLIKEN_ATOM_LINE_PAT.search(line):
                    atom_idx, charge_e, h_plus, e_minus, del_q = (
                        int(m.group(1)),
                        float(m.group(3)),
                        float(m.group(4)),
                        float(m.group(5)),
                        float(m.group(6)),
                    )
                    atom_sym = (
                        input_geometry[atom_idx - 1].symbol
                        if input_geometry and 0 < atom_idx <= len(input_geometry)
                        else m.group(2)
                    )
                    populations.append(
                        ExcitedStateAtomPopulation(atom_idx - 1, atom_sym, charge_e, h_plus, e_minus, del_q)
                    )
                elif (
                    SECTION_SEPARATOR_PAT.search(line)
                    or MULTIPOLE_DM_HEADER_PAT.search(line)
                    or STATE_HEADER_PAT.search(line)
                    or EXCITON_DIFF_DM_HEADER_PAT.search(line)
                ):
                    line_buffer.pop()
                    iterator = iter(line_buffer + [line] + list(iterator))
                    break
                elif not line.strip():
                    continue
                else:
                    logger.warning(f"Unexpected line in Mulliken (State/Diff DM): {line.strip()}")
                    line_buffer.pop()
                    iterator = iter(line_buffer + [line] + list(iterator))
                    break
        except StopIteration:
            iterator = iter(line_buffer + list(iterator))
        return ExcitedStateMulliken(populations=populations) if populations else None

    def _parse_multipole_state_dm(
        self, iterator: LineIterator, first_multipole_line: str
    ) -> ExcitedStateMultipole | None:
        if not MULTIPOLE_DM_HEADER_PAT.search(first_multipole_line):
            if first_multipole_line.strip():
                iterator = iter([first_multipole_line] + list(iterator))
            return None
        mol_chg, num_e, cec_xyz, cnc_xyz, dip_tot, dip_xyz, rms_tot, rms_xyz = [None] * 8
        line_buffer: list[str] = []
        line = first_multipole_line
        try:
            for _ in range(10):
                if not MULTIPOLE_DM_HEADER_PAT.search(line) and _ == 0 and line is not first_multipole_line:
                    break
                if m := MOLECULAR_CHARGE_PAT.search(line):
                    mol_chg = float(m.group(1))
                if m := NUM_ELECTRONS_MULTIPOLE_PAT.search(line):
                    num_e = float(m.group(1))
                if m := CENTER_ELECTRONIC_CHARGE_PAT.search(line):
                    cec_xyz = tuple(map(float, m.groups()))
                if m := CENTER_NUCLEAR_CHARGE_PAT.search(line):
                    cnc_xyz = tuple(map(float, m.groups()))
                if m := DIPOLE_TOTAL_PAT.search(line):
                    dip_tot = float(m.group(1))
                if m := DIPOLE_MOMENT_COMPONENTS_PAT.search(line):
                    dip_xyz = tuple(map(float, m.groups()))
                if m := RMS_TOTAL_SIZE_PAT.search(line):
                    rms_tot = float(m.group(1))
                if (
                    RMS_DENSITY_SIZE_COMPONENTS_PAT.search(line)
                    and rms_tot is not None
                    and dip_xyz is not None
                    and (m_rms := RMS_DENSITY_SIZE_COMPONENTS_PAT.search(line))
                ):
                    rms_xyz = tuple(map(float, m_rms.groups()))  # type: ignore
                    break
                try:
                    prev_line = line
                    line = next(iterator)
                    line_buffer.append(line)
                    if (
                        EXCITON_DIFF_DM_HEADER_PAT.search(line)
                        or SECTION_SEPARATOR_PAT.search(line)
                        or STATE_HEADER_PAT.search(line)
                        or MULLIKEN_POP_HEADER_PAT.search(line)
                        or UNRELAXED_DM_HEADER_PAT.search(line)
                    ):  # UNRELAXED_DM_HEADER_PAT is not for ES context, but harmless here
                        iterator = iter([line] + list(iterator))
                        line_buffer.pop()
                        line = prev_line
                        break
                except StopIteration:
                    break
        except StopIteration:
            pass

        final_dipole: DipoleMomentData | None = None
        if dip_xyz and dip_tot is not None:
            final_dipole = DipoleMomentData(dip_xyz[0], dip_xyz[1], dip_xyz[2], dip_tot)
        elif dip_xyz:
            final_dipole = DipoleMomentData(
                dip_xyz[0], dip_xyz[1], dip_xyz[2], (dip_xyz[0] ** 2 + dip_xyz[1] ** 2 + dip_xyz[2] ** 2) ** 0.5
            )

        if not (
            mol_chg is not None
            or num_e is not None
            or final_dipole is not None
            or cec_xyz is not None
            or cnc_xyz is not None
            or rms_xyz is not None
        ):
            if line_buffer:
                iterator = iter(line_buffer + list(iterator))  # Push back if nothing found
            return None
        return ExcitedStateMultipole(mol_chg, num_e, cec_xyz, cnc_xyz, final_dipole, rms_xyz)  # type: ignore

    def _parse_exciton_state_diff_dm(
        self, iterator: LineIterator, first_exciton_line: str
    ) -> ExcitedStateExcitonDifferenceDM | None:
        if not EXCITON_DIFF_DM_HEADER_PAT.search(first_exciton_line):
            if first_exciton_line.strip():
                iterator = iter([first_exciton_line] + list(iterator))
            return None
        rh_cen, re_cen, re_rh_sep, hole_xyz, elec_xyz = [None] * 5
        line_buffer: list[str] = []
        line = first_exciton_line
        try:
            for _ in range(10):
                if not EXCITON_DIFF_DM_HEADER_PAT.search(line) and _ == 0 and line is not first_exciton_line:
                    break
                if m := EXCITON_RH_PAT.search(line):
                    rh_cen = tuple(map(float, m.groups()))
                if m := EXCITON_RE_PAT.search(line):
                    re_cen = tuple(map(float, m.groups()))
                if m := EXCITON_RE_RH_SEP_PAT.search(line):
                    re_rh_sep = float(m.group(1))
                if EXCITON_SIZE_HEADER_PAT.search(line) and "Hole" in line:
                    try:
                        next_line = next(iterator)
                        line_buffer.append(next_line)
                        if m_h := EXCITON_SIZE_COMPONENTS_PAT.search(next_line):
                            hole_xyz = tuple(map(float, m_h.groups()))
                        line = next_line
                    except StopIteration:
                        break
                if EXCITON_SIZE_HEADER_PAT.search(line) and "Electron" in line:
                    try:
                        next_line = next(iterator)
                        line_buffer.append(next_line)
                        if m_e := EXCITON_SIZE_COMPONENTS_PAT.search(next_line):
                            elec_xyz = tuple(map(float, m_e.groups()))
                        line = next_line
                        if elec_xyz:
                            break
                    except StopIteration:
                        break
                if rh_cen and re_cen and re_rh_sep and hole_xyz and elec_xyz:
                    break
                try:
                    prev_line = line
                    line = next(iterator)
                    line_buffer.append(line)
                    if (
                        STATE_HEADER_PAT.search(line)
                        or SECTION_SEPARATOR_PAT.search(line)
                        or MULLIKEN_POP_HEADER_PAT.search(line)
                        or MULTIPOLE_DM_HEADER_PAT.search(line)
                    ):
                        iterator = iter([line] + list(iterator))
                        line_buffer.pop()
                        line = prev_line
                        break
                except StopIteration:
                    break
        except StopIteration:
            pass

        if not any([rh_cen, re_cen, re_rh_sep, hole_xyz, elec_xyz]):
            if line_buffer:
                iterator = iter(line_buffer + list(iterator))
            return None
        return ExcitedStateExcitonDifferenceDM(rh_cen, re_cen, re_rh_sep, hole_xyz, elec_xyz)

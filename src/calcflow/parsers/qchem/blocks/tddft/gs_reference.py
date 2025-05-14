# Placeholder for TDDFT Block Parsers

import re
from collections.abc import Sequence

from calcflow.parsers.qchem.typing import (
    Atom,
    DipoleMomentData,
    ExcitedStateAtomPopulation,
    ExcitedStateMulliken,
    ExcitedStateMultipole,
    ExcitedStateNOData,
    GroundStateReferenceAnalysis,
    LineIterator,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- Excited State Analysis Block (Unrelaxed Density Matrices) ---
UNRELAXED_DM_HEADER_PAT = re.compile(r"^\s*Analysis of Unrelaxed Density Matrices\s*$")
STATE_HEADER_PAT = re.compile(r"^\s*(Singlet|Triplet)\s+(\d+)\s*:\s*$")

# # --- Ground State Reference Patterns (within Excited State Analysis block) ---
GROUND_STATE_REF_HEADER_PAT = re.compile(r"^\s*Ground State \(Reference\)\s*:\s*$")

# NOs
NOS_HEADER_PAT = re.compile(r"^\s*NOs\s*$")
NOS_FRONTIER_PAT = re.compile(r"^\s+Occupation of frontier NOs:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)")
NOS_NUM_ELECTRONS_PAT = re.compile(r"^\s+Number of electrons:\s*(-?\d+\.\d+)")
NOS_UNPAIRED_PAT = re.compile(r"^\s+Number of unpaired electrons: n_u =\s*(-?\d+\.\d+), n_u,nl =\s*(-?\d+\.\d+)")
NOS_PR_NO_PAT = re.compile(r"^\s+NO participation ratio \(PR_NO\):\s*(-?\d+\.\d+)")

# # Mulliken Population (State/Difference DM)
MULLIKEN_POP_HEADER_PAT = re.compile(r"^\s*Mulliken Population Analysis \(State/Difference DM\)\s*$")
MULLIKEN_SUM_LINE_PAT = re.compile(r"^\s*Sum:\s+(-?\d+\.\d+)")  # Marks end of atom list for Mulliken

# # Mulliken Population (Ground State DM within Excited State Analysis)
MULLIKEN_GS_POP_HEADER_PAT = re.compile(r"^\s*Mulliken Population Analysis \(State DM\)\s*$")
MULLIKEN_GS_ATOM_LINE_PAT = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s*$")

# # Multipole Moment Analysis (Density Matrix)
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

SECTION_SEPARATOR_PAT = re.compile(r"^\s*-{20,}\s*$")  # A line of dashes, often ends state blocks
BLANK_LINE_PAT = re.compile(r"^\s*$")


class GroundStateReferenceParser(SectionParser):
    """Parses the 'Ground State (Reference) :' block within Excited State Analysis."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        # This parser is expected to be called when the Ground State block is confirmed.
        # The main orchestrator would find "Excited State Analysis", then look for this.
        return GROUND_STATE_REF_HEADER_PAT.search(line) is not None

    def parse(self, iterator: LineIterator, first_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting parsing of Ground State (Reference) block.")
        # first_line is "Ground State (Reference) :" - this line is already consumed by the main parser in core.py
        # The iterator is positioned at the line AFTER "Ground State (Reference) :"

        gs_no_data: ExcitedStateNOData | None = None
        gs_mulliken_data: ExcitedStateMulliken | None = None
        gs_multipole_data: ExcitedStateMultipole | None = None

        max_lines_to_scan = 150  # Safeguard for the entire GS Ref block
        lines_scanned = 0

        while lines_scanned < max_lines_to_scan:
            try:
                line = next(iterator)
                lines_scanned += 1
            except StopIteration:
                logger.debug("Reached end of iterator during Ground State Reference parsing.")
                break

            # Priority 1: Check for the hard stop signal for this entire block
            if UNRELAXED_DM_HEADER_PAT.search(line):
                logger.debug(
                    f"Found '{UNRELAXED_DM_HEADER_PAT.pattern}' header; current GS Ref section ends. Buffering line."
                )
                results.buffered_line = line  # Buffer the line
                break

            # Priority 2: Check if all expected components have been parsed
            # If so, the current line 'line' is the first line *after* the GS ref block content.
            if gs_no_data and gs_mulliken_data and gs_multipole_data:
                logger.debug("All expected components of Ground State Reference parsed. Buffering current line.")
                results.buffered_line = line  # Buffer the line that came after the last component
                break

            # Priority 3: Try to parse specific sub-sections if not already done
            if gs_no_data is None and NOS_HEADER_PAT.search(line):
                gs_no_data = self._parse_no_data(iterator, line, results)
                if results.buffered_line:
                    break  # Sub-parser decided the section is over
                continue

            if gs_mulliken_data is None and MULLIKEN_GS_POP_HEADER_PAT.search(line):
                gs_mulliken_data = self._parse_mulliken_ground_state_dm(iterator, line, results.input_geometry, results)
                if results.buffered_line:
                    break
                continue

            if gs_multipole_data is None and MULTIPOLE_DM_HEADER_PAT.search(line):
                gs_multipole_data = self._parse_multipole_state_dm(iterator, line, results)
                if results.buffered_line:
                    break
                continue

            # Priority 4: Handle lines that are not headers for this parser, not UNRELAXED_DM,
            # and not yet at the point where all components are parsed.
            # These are typically blank lines or separators *within* the GS Reference block.
            if line.strip() == "" or SECTION_SEPARATOR_PAT.search(line):
                continue  # Consume and move to the next line

            # Fallthrough: Line is unrecognized within the GS Ref block content area.
            # This means the GS Reference section is over, or we've encountered unexpected data.
            # Buffer the line and terminate parsing for this section.
            logger.warning(
                f"Unrecognized line encountered in Ground State Reference block: '{line.strip()}'. "
                "Assuming end of this block and buffering line."
            )
            results.buffered_line = line  # Buffer the unrecognized line
            break  # Exit the while loop, GS Reference parsing is done.

        if lines_scanned >= max_lines_to_scan and not results.buffered_line:
            logger.warning("GroundStateReferenceParser exceeded max lines scan.")

        # Store results if any component was parsed
        if gs_no_data or gs_mulliken_data or gs_multipole_data:
            results.ground_state_reference_analysis = GroundStateReferenceAnalysis(
                no_data=gs_no_data,
                mulliken_analysis=gs_mulliken_data,
                multipole_analysis=gs_multipole_data,
            )
            logger.info("Stored Ground State Reference analysis data.")
        else:
            # This will be hit if the loop exited (e.g., UNRELAXED_DM_HEADER found) before any sub-section was parsed.
            logger.warning("No data components (NOs, Mulliken, Multipole) parsed for Ground State Reference block.")
            results.parsing_warnings.append(
                "No data components (NOs, Mulliken, Multipole) parsed for Ground State Reference block."
            )

        logger.debug("Finished parsing Ground State (Reference) block.")

    def _parse_no_data(
        self, iterator: LineIterator, first_no_line: str, results: _MutableCalculationData
    ) -> ExcitedStateNOData:
        frontier_occ: list[float] | None = None
        num_e: float | None = None
        nu: float | None = None
        nunl: float | None = None
        pr_no: float | None = None
        line = first_no_line
        for _i in range(5):
            if not NOS_HEADER_PAT.search(line) and _i == 0:
                if line is not first_no_line and line.strip():  # If it was a meaningful non-header line
                    results.buffered_line = line
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
                    or MULLIKEN_GS_POP_HEADER_PAT.search(line)
                    or MULTIPOLE_DM_HEADER_PAT.search(line)
                    or STATE_HEADER_PAT.search(line)
                    or UNRELAXED_DM_HEADER_PAT.search(line)
                ):
                    results.buffered_line = line  # Buffer the terminating line
                    line = prev_line  # effectively unread 'line' for current processing
                    break
            except StopIteration:
                break
        return ExcitedStateNOData(frontier_occ, num_e, nu, nunl, pr_no)

    def _parse_mulliken_ground_state_dm(
        self,
        iterator: LineIterator,
        first_mulliken_gs_line: str,
        input_geometry: Sequence[Atom] | None,
        results: _MutableCalculationData,
    ) -> ExcitedStateMulliken | None:
        if not MULLIKEN_GS_POP_HEADER_PAT.search(first_mulliken_gs_line):
            if first_mulliken_gs_line.strip():  # If it was a meaningful line, push it back
                results.buffered_line = first_mulliken_gs_line
            return None

        populations: list[ExcitedStateAtomPopulation] = []
        internal_separator_pat = re.compile(r"^\s*-{10,}\s*$")  # For separators like "---------"

        try:
            # first_mulliken_gs_line is the header. Consume known sub-headers.
            _ = next(iterator)  # Consumes "Atom      Charge (e)" line
            _ = next(iterator)  # Consumes "--------------------" (the first one, column names)

            # Iterator is now at the first potential atom line.
            while True:
                line = next(iterator)

                # Check for known next section headers FIRST to ensure we push them back
                if (
                    MULTIPOLE_DM_HEADER_PAT.search(line)
                    or UNRELAXED_DM_HEADER_PAT.search(line)
                    or STATE_HEADER_PAT.search(line)
                ):
                    results.buffered_line = line  # Buffer the terminating line
                    break

                if m_atom := MULLIKEN_GS_ATOM_LINE_PAT.search(line):
                    idx, sym_from_output, chg = int(m_atom.group(1)), m_atom.group(2), float(m_atom.group(3))
                    atom_symbol = sym_from_output  # Default to symbol from output
                    if input_geometry and 0 < idx <= len(input_geometry):
                        atom_symbol = input_geometry[idx - 1].symbol
                    elif input_geometry:  # Mismatch, log warning
                        logger.warning(
                            f"Mulliken atom index {idx} out of range for input geometry size {len(input_geometry)}."
                        )
                    populations.append(ExcitedStateAtomPopulation(idx - 1, atom_symbol, chg, None, None, None))
                    continue  # Continue to next line for more atoms

                # Check for the internal separator line (e.g., "--------------------") AFTER atoms
                if internal_separator_pat.search(line):
                    # This is the separator before the "Sum:" line. Consume it and continue.
                    continue

                if MULLIKEN_SUM_LINE_PAT.search(line):
                    # Consumed the "Sum:" line. This is the end of Mulliken atom data.
                    break

                if not line.strip():
                    logger.debug("Blank line encountered in Mulliken GS, assuming end of atom data.")
                    break  # Exit parsing Mulliken atoms

                # If the line is not any of the above:
                logger.warning(f"Unexpected line in Mulliken (GS State DM) table: {line.strip()}")
                results.buffered_line = line  # Buffer this unexpected line
                break

        except StopIteration:
            logger.debug("Iterator ended during Mulliken (GS State DM) atom/sum parsing.")

        if not populations:
            logger.debug("No Mulliken populations found or parsed in GS State DM block.")
            return None
        return ExcitedStateMulliken(populations=populations)

    def _parse_multipole_state_dm(
        self, iterator: LineIterator, first_multipole_line: str, results: _MutableCalculationData
    ) -> ExcitedStateMultipole | None:
        if not MULTIPOLE_DM_HEADER_PAT.search(first_multipole_line):
            if first_multipole_line.strip():
                results.buffered_line = first_multipole_line
            return None
        mol_chg: float | None = None
        num_e: float | None = None
        cec_xyz: tuple[float, float, float] | None = None
        cnc_xyz: tuple[float, float, float] | None = None
        dip_tot: float | None = None
        dip_xyz: tuple[float, float, float] | None = None
        rms_xyz: tuple[float, float, float] | None = None

        line_buffer: list[str] = []
        line = first_multipole_line
        try:
            for _ in range(10):  # Max 10 lines for the section
                if not MULTIPOLE_DM_HEADER_PAT.search(line) and _ == 0 and line is not first_multipole_line:
                    # This should not be hit if called correctly with the header line
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
                    # rms_tot = float(m.group(1))
                    pass

                # Refined logic for RMS density components
                if m_rms_match := RMS_DENSITY_SIZE_COMPONENTS_PAT.search(line):
                    # This specific line is for RMS density components.
                    rms_xyz = tuple(map(float, m_rms_match.groups()))
                    # This is typically the last relevant line in the GS multipole block.
                    break

                try:
                    line = next(iterator)
                    line_buffer.append(line)
                    if (
                        EXCITON_DIFF_DM_HEADER_PAT.search(line)  # Should not appear in GS multipole
                        or SECTION_SEPARATOR_PAT.search(line)  # Could be a separator before UNRELAXED_DM
                        or STATE_HEADER_PAT.search(line)  # Should not appear here
                        or MULLIKEN_POP_HEADER_PAT.search(line)  # Distinguish from MULLIKEN_GS_POP_HEADER_PAT
                        or UNRELAXED_DM_HEADER_PAT.search(line)  # Definite end of GS reference context
                    ):
                        results.buffered_line = line  # Buffer the terminating line
                        line_buffer.pop()  # Remove from local buffer as it's now globally buffered
                        # line = prev_line # No need to re-process prev_line, loop will break
                        break
                except StopIteration:
                    break
        except StopIteration:  # Outer loop's StopIteration
            pass

        final_dipole: DipoleMomentData | None = None
        if dip_xyz and dip_tot is not None:
            final_dipole = DipoleMomentData(dip_xyz[0], dip_xyz[1], dip_xyz[2], dip_tot)
        elif dip_xyz:  # Fallback if total dipole line was missing but components were found
            calculated_total = (dip_xyz[0] ** 2 + dip_xyz[1] ** 2 + dip_xyz[2] ** 2) ** 0.5
            final_dipole = DipoleMomentData(dip_xyz[0], dip_xyz[1], dip_xyz[2], calculated_total)
            logger.debug(f"Calculated total dipole {calculated_total:.4f} from components for multipole section.")

        if not (
            mol_chg is not None
            or num_e is not None
            or final_dipole is not None
            or cec_xyz is not None
            or cnc_xyz is not None
            or rms_xyz is not None  # Crucial check: if rms_xyz parsed, this should lead to object creation
        ):
            logger.warning(
                f"Failed to parse significant multipole data. Fields: mol_chg={mol_chg}, num_e={num_e}, "
                f"final_dipole_present={final_dipole is not None}, cec_xyz_present={cec_xyz is not None}, "
                f"cnc_xyz_present={cnc_xyz is not None}, rms_xyz_present={rms_xyz is not None}"
            )
            if line_buffer:  # Push back any lines consumed by this sub-parser if it effectively failed
                # This part is tricky. If sub-parser fails AND buffered lines exist, what to do?
                # For now, if significant data was parsed, we assume the buffer is from *after* this section.
                # If no significant data, then the first line of line_buffer might be the one to re-process.
                # This suggests maybe sub-parsers shouldn't use their own line_buffer if they can use results.buffered_line.
                # For simplicity now, if a sub-parser fails badly, it might not buffer correctly.
                # The main parser loop buffering should be the primary guard.
                # Let's remove this iterator recreation to avoid prior issues.
                # iterator = iter(line_buffer + list(iterator))
                logger.warning(
                    "Sub-parser _parse_multipole_state_dm had leftover lines in its internal buffer, not re-buffering globally."
                )
            return None

        logger.debug(
            f"Successfully parsed multipole data: Dipole total = {final_dipole.total_debye if final_dipole else None}, RMS_XYZ = {rms_xyz}"
        )
        return ExcitedStateMultipole(mol_chg, num_e, cec_xyz, cnc_xyz, final_dipole, rms_xyz)

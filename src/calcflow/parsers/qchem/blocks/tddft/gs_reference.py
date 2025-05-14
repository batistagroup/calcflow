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
        # first_line is "Ground State (Reference) :"

        gs_no_data: ExcitedStateNOData | None = None
        gs_mulliken_data: ExcitedStateMulliken | None = None
        gs_multipole_data: ExcitedStateMultipole | None = None

        # Consume the header line passed as first_line, iterator starts from the next line.
        # However, our sub-parsers expect the header of their specific section.

        line_buffer: list[str] = []  # To push back the UNRELAXED_DM_HEADER_PAT if found by this parser

        while True:
            try:
                line = next(iterator)
                line_buffer.append(line)  # Buffer lines in case we need to push them back
            except StopIteration:
                logger.debug("Reached end of iterator during Ground State Reference parsing.")
                break

            if UNRELAXED_DM_HEADER_PAT.search(line):
                logger.debug("Found 'Analysis of Unrelaxed Density Matrices' header; ending GS parse.")
                # Push back this line for the next parser
                iterator = iter([line] + list(iterator))
                line_buffer.pop()  # Remove it from our buffer as it's handled
                break

            # Skip blank lines or section separators that are not the main unrelaxed DM header
            if not line.strip() or (SECTION_SEPARATOR_PAT.search(line) and not UNRELAXED_DM_HEADER_PAT.search(line)):
                line_buffer.pop()  # Consume and discard this line from buffer
                continue

            # Check for NOs section
            if NOS_HEADER_PAT.search(line) and gs_no_data is None:
                logger.debug("Parsing NOs for Ground State Reference.")
                # _parse_no_data expects 'line' to be the NOS_HEADER
                gs_no_data = self._parse_no_data(iterator, line)
                line_buffer.clear()  # Sub-parser manages iterator, clear buffer
                continue  # Get fresh line with next(iterator)

            # Check for Mulliken Population (State DM for GS)
            elif MULLIKEN_GS_POP_HEADER_PAT.search(line) and gs_mulliken_data is None:
                logger.debug("Parsing Mulliken (State DM) for Ground State Reference.")
                gs_mulliken_data = self._parse_mulliken_ground_state_dm(iterator, line, results.input_geometry)
                line_buffer.clear()
                continue

            # Check for Multipole Moment Analysis
            elif MULTIPOLE_DM_HEADER_PAT.search(line) and gs_multipole_data is None:
                logger.debug("Parsing Multipole DM for Ground State Reference.")
                gs_multipole_data = self._parse_multipole_state_dm(iterator, line)
                line_buffer.clear()
                continue

            # If line is not a known sub-header, and not yet the UNRELAXED_DM_HEADER
            # it might be an issue or just an unexpected line within GS block.
            # For now, we consume it from buffer and log if it's not blank/sep.
            # The main loop continues with next(iterator).
            if line.strip() and not SECTION_SEPARATOR_PAT.search(line):  # Avoid logging for separators already handled
                logger.debug(f"Skipping unrecognized line in Ground State Reference block: {line.strip()}")
            if line_buffer:
                line_buffer.pop()

        if gs_no_data or gs_mulliken_data or gs_multipole_data:
            results.ground_state_reference_analysis = GroundStateReferenceAnalysis(
                no_data=gs_no_data,
                mulliken_analysis=gs_mulliken_data,
                multipole_analysis=gs_multipole_data,
            )
            logger.info("Stored Ground State Reference analysis data.")
        else:
            logger.warning("No data parsed for Ground State Reference block.")

        # Push back any remaining lines in line_buffer if they were not consumed
        # This happens if loop broke due to StopIteration before UNRELAXED_DM_HEADER was found
        if line_buffer:
            iterator = iter(line_buffer + list(iterator))

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
                    or MULLIKEN_GS_POP_HEADER_PAT.search(line)
                    or MULTIPOLE_DM_HEADER_PAT.search(line)
                    or STATE_HEADER_PAT.search(line)
                    or UNRELAXED_DM_HEADER_PAT.search(line)
                ):
                    iterator = iter([line] + list(iterator))
                    line = prev_line
                    break
            except StopIteration:
                break
        return ExcitedStateNOData(frontier_occ, num_e, nu, nunl, pr_no)

    def _parse_mulliken_ground_state_dm(
        self, iterator: LineIterator, first_mulliken_gs_line: str, input_geometry: Sequence[Atom] | None
    ) -> ExcitedStateMulliken | None:
        if not MULLIKEN_GS_POP_HEADER_PAT.search(first_mulliken_gs_line):
            if first_mulliken_gs_line.strip():  # If it was a meaningful line, push it back
                iterator = iter([first_mulliken_gs_line] + list(iterator))
            return None

        populations: list[ExcitedStateAtomPopulation] = []
        internal_separator_pat = re.compile(r"^\s*-{10,}\s*$")  # For separators like "---------"

        try:
            # first_mulliken_gs_line is the header. Consume known sub-headers.
            _ = next(iterator)  # Consumes "Atom      Charge (e)" line
            _ = next(iterator)  # Consumes "--------------------" (the first one, column names)

            # Iterator is now at the first potential atom line.
            while True:
                line = next(iterator)  # Read the next line

                # Check for known next section headers FIRST to ensure we push them back
                if (
                    MULTIPOLE_DM_HEADER_PAT.search(line)
                    or UNRELAXED_DM_HEADER_PAT.search(line)
                    or STATE_HEADER_PAT.search(line)
                ):
                    iterator = iter([line] + list(iterator))
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
                    logger.debug(f"Consumed internal Mulliken separator: {line.strip()}")
                    continue

                if MULLIKEN_SUM_LINE_PAT.search(line):
                    # Consumed the "Sum:" line. This is the end of Mulliken atom data.
                    logger.debug(f"Consumed Mulliken Sum line: {line.strip()}")
                    break

                if not line.strip():
                    logger.debug("Blank line encountered in Mulliken GS, assuming end of atom data.")
                    break  # Exit parsing Mulliken atoms

                # If the line is not any of the above:
                logger.warning(f"Unexpected line in Mulliken (GS State DM) table: {line.strip()}")
                iterator = iter([line] + list(iterator))  # Push this unexpected line back
                break

        except StopIteration:
            logger.debug("Iterator ended during Mulliken (GS State DM) atom/sum parsing.")

        if not populations:
            logger.debug("No Mulliken populations found or parsed in GS State DM block.")
            return None
        return ExcitedStateMulliken(populations=populations)

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
                    logger.debug(f"Parsed RMS density components: {rms_xyz}")
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
                        iterator = iter([line] + list(iterator))  # Push back the terminating line
                        line_buffer.pop()  # Remove from local buffer
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
                iterator = iter(line_buffer + list(iterator))
            return None

        logger.debug(
            f"Successfully parsed multipole data: Dipole total = {final_dipole.total_debye if final_dipole else None}, RMS_XYZ = {rms_xyz}"
        )
        return ExcitedStateMultipole(mol_chg, num_e, cec_xyz, cnc_xyz, final_dipole, rms_xyz)

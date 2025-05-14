# Placeholder for TDDFT Block Parsers

import re
from collections.abc import Sequence

from calcflow.exceptions import ParsingError
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


class ExcitedStateAnalysisParser(SectionParser):
    """Parses the 'Excited State Analysis' block for unrelaxed density matrices."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return EXCITED_STATE_ANALYSIS_HEADER_PAT.search(line) is not None

    def _parse_no_data(self, iterator: LineIterator, first_no_line: str) -> ExcitedStateNOData:
        """Helper to parse the NOs subsection."""
        frontier_occ: list[float] | None = None
        num_e: float | None = None
        nu: float | None = None
        nunl: float | None = None
        pr_no: float | None = None

        line = first_no_line
        consumed_lines_buffer: list[str] = [line]

        for _i in range(5):  # Max lines for NOs section
            if not NOS_HEADER_PAT.search(line) and _i == 0:  # first_no_line should be the header
                logger.warning(f"Expected NOs header, got: {line.strip()}")
                # Return early or raise, depending on strictness.
                # For now, assume data is missing and return partially filled/empty object.
                return ExcitedStateNOData()

            match_frontier = NOS_FRONTIER_PAT.search(line)
            if match_frontier:
                frontier_occ = [float(match_frontier.group(1)), float(match_frontier.group(2))]

            match_num_e = NOS_NUM_ELECTRONS_PAT.search(line)
            if match_num_e:
                num_e = float(match_num_e.group(1))

            match_unpaired = NOS_UNPAIRED_PAT.search(line)
            if match_unpaired:
                nu = float(match_unpaired.group(1))
                nunl = float(match_unpaired.group(2))

            match_pr_no = NOS_PR_NO_PAT.search(line)
            if match_pr_no:
                pr_no = float(match_pr_no.group(1))
                # This is usually the last line of NOs data before Mulliken or separator
                break

            try:
                line = next(iterator)
                consumed_lines_buffer.append(line)
                # If we see a separator or a new known header, stop for NOs
                if (
                    SECTION_SEPARATOR_PAT.search(line)
                    or MULLIKEN_POP_HEADER_PAT.search(line)
                    or STATE_HEADER_PAT.search(line)
                ):
                    iterator = iter([line] + list(iterator))  # Push back current line
                    consumed_lines_buffer.pop()
                    break
            except StopIteration:
                break

        return ExcitedStateNOData(
            frontier_occupations=frontier_occ,
            num_electrons=num_e,
            num_unpaired_electrons_nu=nu,
            num_unpaired_electrons_nunl=nunl,
            participation_ratio_pr_no=pr_no,
        )

    def _parse_mulliken_state_dm(
        self, iterator: LineIterator, first_mulliken_line: str, input_geometry: Sequence[Atom] | None
    ) -> ExcitedStateMulliken | None:
        """Helper to parse Mulliken Population Analysis (State/Difference DM)."""
        if not MULLIKEN_POP_HEADER_PAT.search(first_mulliken_line):
            logger.warning(f"Expected Mulliken Pop header, got: {first_mulliken_line.strip()}")
            return None

        populations: list[ExcitedStateAtomPopulation] = []
        line_buffer: list[str] = []

        try:
            # Consume the column header line (Atom Charge (e) ...)
            # and the dashed line (----)
            _ = next(iterator)  # Column header
            _ = next(iterator)  # Dashed line

            while True:
                line = next(iterator)
                line_buffer.append(line)

                if MULLIKEN_SUM_LINE_PAT.search(line) or not line.strip():  # End of Mulliken data or blank line after
                    line_buffer.pop()  # Remove Sum line or blank line from buffer
                    iterator = iter(line_buffer + list(iterator))  # Push back unused lines
                    break

                match_atom = MULLIKEN_ATOM_LINE_PAT.search(line)
                if match_atom:
                    atom_idx_one_based = int(match_atom.group(1))
                    # symbol_from_output = match_atom.group(2) # Can be used for validation
                    charge_e = float(match_atom.group(3))
                    hole_charge = float(match_atom.group(4))
                    electron_charge = float(match_atom.group(5))
                    delta_charge = float(match_atom.group(6))

                    atom_symbol = "?"
                    if input_geometry and 0 < atom_idx_one_based <= len(input_geometry):
                        atom_symbol = input_geometry[atom_idx_one_based - 1].symbol
                    else:
                        logger.warning(f"Could not map Mulliken atom index {atom_idx_one_based} to input geometry.")

                    populations.append(
                        ExcitedStateAtomPopulation(
                            atom_index=atom_idx_one_based - 1,  # Convert to 0-based
                            symbol=atom_symbol,
                            charge_e=charge_e,
                            hole_charge=hole_charge,
                            electron_charge=electron_charge,
                            delta_charge=delta_charge,
                        )
                    )
                elif (
                    SECTION_SEPARATOR_PAT.search(line)
                    or MULTIPOLE_DM_HEADER_PAT.search(line)
                    or STATE_HEADER_PAT.search(line)
                ):
                    # Reached end of Mulliken block due to next section
                    line_buffer.pop()  # Remove matched line from buffer
                    iterator = iter(line_buffer + [line] + list(iterator))  # Push back all, including current
                    break
                elif not line.strip():  # Allow blank lines within table if any, but Sum or separator should stop it
                    continue
                else:
                    logger.warning(f"Unexpected line in Mulliken (State/Diff DM) table: {line.strip()}")
                    # Heuristic: if unexpected line, assume Mulliken part is done to avoid infinite loop
                    line_buffer.pop()  # Remove this unexpected line
                    iterator = iter(line_buffer + [line] + list(iterator))  # Push back with this line
                    break

        except StopIteration:
            logger.warning("File ended unexpectedly during Mulliken (State/Diff DM) parsing.")
            iterator = iter(line_buffer + list(iterator))  # Push back what we have if file ends

        if not populations:
            logger.debug("No Mulliken populations found in State/Difference DM block.")
            return None
        return ExcitedStateMulliken(populations=populations)

    def _parse_multipole_state_dm(
        self, iterator: LineIterator, first_multipole_line: str
    ) -> ExcitedStateMultipole | None:
        """Helper to parse Multipole moment analysis of the density matrix (for an excited state)."""
        if not MULTIPOLE_DM_HEADER_PAT.search(first_multipole_line):
            logger.warning(f"Expected Multipole DM header, got: {first_multipole_line.strip()}")
            return None

        mol_charge: float | None = None
        num_e: float | None = None
        center_elec_charge: tuple[float, float, float] | None = None
        center_nuc_charge: tuple[float, float, float] | None = None
        dipole_total: float | None = None
        dipole_xyz: tuple[float, float, float] | None = None
        rms_density_total: float | None = None  # QChem output has "RMS size of the density [Ang]:         0.747014"
        rms_density_xyz: tuple[float, float, float] | None = None

        line_buffer: list[str] = []

        try:
            line = first_multipole_line  # Already consumed by caller, passed as arg
            # Loop for properties within this subsection
            for _ in range(10):  # Max lines for this subsection
                if not MULTIPOLE_DM_HEADER_PAT.search(line) and _ == 0 and line is not first_multipole_line:
                    # This case should ideally not be hit if called correctly with first_multipole_line being the header
                    logger.warning(f"Multipole DM parsing started with non-header: {line.strip()}")
                    break

                match_mol_chg = MOLECULAR_CHARGE_PAT.search(line)
                if match_mol_chg:
                    mol_charge = float(match_mol_chg.group(1))

                match_num_e = NUM_ELECTRONS_MULTIPOLE_PAT.search(line)
                if match_num_e:
                    num_e = float(match_num_e.group(1))

                match_cec = CENTER_ELECTRONIC_CHARGE_PAT.search(line)
                if match_cec:
                    center_elec_charge = (
                        float(match_cec.group(1)),
                        float(match_cec.group(2)),
                        float(match_cec.group(3)),
                    )

                match_cnc = CENTER_NUCLEAR_CHARGE_PAT.search(line)
                if match_cnc:
                    center_nuc_charge = (
                        float(match_cnc.group(1)),
                        float(match_cnc.group(2)),
                        float(match_cnc.group(3)),
                    )

                match_dip_tot = DIPOLE_TOTAL_PAT.search(line)
                if match_dip_tot:
                    dipole_total = float(match_dip_tot.group(1))

                match_dip_xyz = DIPOLE_MOMENT_COMPONENTS_PAT.search(line)
                if match_dip_xyz:
                    dipole_xyz = (
                        float(match_dip_xyz.group(1)),
                        float(match_dip_xyz.group(2)),
                        float(match_dip_xyz.group(3)),
                    )

                match_rms_tot = RMS_TOTAL_SIZE_PAT.search(line)  # For "RMS size of the density"
                if match_rms_tot:
                    rms_density_total = float(match_rms_tot.group(1))

                # This pattern (RMS_DENSITY_SIZE_COMPONENTS_PAT) is used for density AND later for hole/electron sizes
                # Context is important. Here, we are in multipole section, so it is for density.
                if (
                    RMS_DENSITY_SIZE_COMPONENTS_PAT.search(line)
                    and rms_density_total is not None
                    and dipole_xyz is not None
                ):  # Heuristic: RMS components usually last or after dipole
                    m = RMS_DENSITY_SIZE_COMPONENTS_PAT.search(line)
                    if m:  # mypy complains without this check
                        rms_density_xyz = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                        # This is usually the last piece of info before exciton or separator
                        break

                try:
                    line = next(iterator)
                    line_buffer.append(line)
                    # Check for end of this sub-block
                    if (
                        EXCITON_DIFF_DM_HEADER_PAT.search(line)
                        or SECTION_SEPARATOR_PAT.search(line)
                        or STATE_HEADER_PAT.search(line)
                        or MULLIKEN_POP_HEADER_PAT.search(line)
                    ):
                        iterator = iter([line] + list(iterator))  # Push back current line
                        line_buffer.pop()
                        break
                except StopIteration:
                    break

        except StopIteration:  # Should be caught by inner loop, but as a safeguard
            logger.warning("File ended unexpectedly during Multipole (State DM) parsing.")

        final_dipole_data: DipoleMomentData | None = None
        if dipole_xyz is not None and dipole_total is not None:
            final_dipole_data = DipoleMomentData(
                x_debye=dipole_xyz[0], y_debye=dipole_xyz[1], z_debye=dipole_xyz[2], total_debye=dipole_total
            )
        elif dipole_xyz is not None and dipole_total is None:
            # If only components found, calculate total (common if total is not on a separate line)
            total_calc = (dipole_xyz[0] ** 2 + dipole_xyz[1] ** 2 + dipole_xyz[2] ** 2) ** 0.5
            final_dipole_data = DipoleMomentData(
                x_debye=dipole_xyz[0], y_debye=dipole_xyz[1], z_debye=dipole_xyz[2], total_debye=total_calc
            )
            logger.debug(f"Calculated total dipole {total_calc:.4f} from components for state DM.")

        if mol_charge is None and num_e is None and final_dipole_data is None:
            logger.debug("No significant multipole data found in State DM block.")
            # Push back all consumed lines if nothing was found
            if line_buffer:
                iterator = iter(line_buffer + list(iterator))
            return None

        # Push back any unconsumed lines from buffer if parsing ended prematurely or some lines were not used.
        # This is tricky, as some lines might be intentionally consumed and processed.
        # The break conditions should handle pushing back the *next* section's header.
        if line_buffer and (
            EXCITON_DIFF_DM_HEADER_PAT.search(line_buffer[-1])
            or SECTION_SEPARATOR_PAT.search(line_buffer[-1])
            or STATE_HEADER_PAT.search(line_buffer[-1])
        ):
            pass  # Already handled by break conditions
        elif line_buffer:  # If buffer has content not pushed back by break conditions
            iterator = iter(line_buffer + list(iterator))

        return ExcitedStateMultipole(
            molecular_charge=mol_charge,
            num_electrons=num_e,
            center_electronic_charge_ang=center_elec_charge,
            center_nuclear_charge_ang=center_nuc_charge,
            dipole_moment_debye=final_dipole_data,
            rms_density_size_ang=rms_density_xyz,  # Only storing components for now, total was for parsing aid
        )

    def _parse_exciton_state_diff_dm(
        self, iterator: LineIterator, first_exciton_line: str
    ) -> ExcitedStateExcitonDifferenceDM | None:
        """Helper to parse Exciton analysis of the difference density matrix."""
        if not EXCITON_DIFF_DM_HEADER_PAT.search(first_exciton_line):
            logger.warning(f"Expected Exciton Diff DM header, got: {first_exciton_line.strip()}")
            return None

        rh_center: tuple[float, float, float] | None = None
        re_center: tuple[float, float, float] | None = None
        re_rh_sep: float | None = None
        hole_size_xyz: tuple[float, float, float] | None = None
        electron_size_xyz: tuple[float, float, float] | None = None
        # QChem sometimes prints total hole/electron size on the header line, sometimes not.
        # We'll capture components and can calculate total if needed, or store if directly available.

        line_buffer: list[str] = []
        line = first_exciton_line  # Already consumed by caller

        try:
            for _ in range(10):  # Max lines for this subsection
                if not EXCITON_DIFF_DM_HEADER_PAT.search(line) and _ == 0 and line is not first_exciton_line:
                    logger.warning(f"Exciton Diff DM parsing started with non-header: {line.strip()}")
                    break

                match_rh = EXCITON_RH_PAT.search(line)
                if match_rh:
                    rh_center = (float(match_rh.group(1)), float(match_rh.group(2)), float(match_rh.group(3)))

                match_re = EXCITON_RE_PAT.search(line)
                if match_re:
                    re_center = (float(match_re.group(1)), float(match_re.group(2)), float(match_re.group(3)))

                match_sep = EXCITON_RE_RH_SEP_PAT.search(line)
                if match_sep:
                    re_rh_sep = float(match_sep.group(1))

                # Hole Size Parsing
                if EXCITON_SIZE_HEADER_PAT.search(line) and "Hole" in line:
                    # Total might be on this line or next. Components are on next.
                    # Try to get components on the next line
                    try:
                        next_line_for_hole = next(iterator)
                        line_buffer.append(next_line_for_hole)
                        match_hole_xyz = EXCITON_SIZE_COMPONENTS_PAT.search(next_line_for_hole)
                        if match_hole_xyz:
                            hole_size_xyz = (
                                float(match_hole_xyz.group(1)),
                                float(match_hole_xyz.group(2)),
                                float(match_hole_xyz.group(3)),
                            )
                        line = next_line_for_hole  # Consume the components line to continue outer loop
                    except StopIteration:
                        logger.warning("File ended while looking for hole size components.")
                        break

                # Electron Size Parsing (similar logic to Hole Size)
                if EXCITON_SIZE_HEADER_PAT.search(line) and "Electron" in line:
                    try:
                        next_line_for_electron = next(iterator)
                        line_buffer.append(next_line_for_electron)
                        match_electron_xyz = EXCITON_SIZE_COMPONENTS_PAT.search(next_line_for_electron)
                        if match_electron_xyz:
                            electron_size_xyz = (
                                float(match_electron_xyz.group(1)),
                                float(match_electron_xyz.group(2)),
                                float(match_electron_xyz.group(3)),
                            )
                        line = next_line_for_electron  # Consume the components line
                        # This is often the last part of the exciton analysis for a state
                        # Check if we should break based on electron_size_xyz being found
                        if electron_size_xyz:
                            break  # Assume end of exciton block for this state
                    except StopIteration:
                        logger.warning("File ended while looking for electron size components.")
                        break

                # If all expected data for exciton is found, we can break early
                if rh_center and re_center and re_rh_sep and hole_size_xyz and electron_size_xyz:
                    break

                try:
                    line = next(iterator)
                    line_buffer.append(line)
                    # Check for end of this sub-block (e.g., start of a new state or section separator)
                    if (
                        STATE_HEADER_PAT.search(line)
                        or SECTION_SEPARATOR_PAT.search(line)
                        or MULLIKEN_POP_HEADER_PAT.search(line)
                        or MULTIPOLE_DM_HEADER_PAT.search(line)
                    ):
                        iterator = iter([line] + list(iterator))  # Push back current line
                        line_buffer.pop()
                        break
                except StopIteration:
                    break

        except StopIteration:
            logger.warning("File ended unexpectedly during Exciton (Difference DM) parsing.")

        if not (rh_center or re_center or re_rh_sep or hole_size_xyz or electron_size_xyz):
            logger.debug("No significant exciton data found in Difference DM block.")
            if line_buffer:
                iterator = iter(line_buffer + list(iterator))
            return None

        # Push back any unconsumed lines
        if line_buffer and (STATE_HEADER_PAT.search(line_buffer[-1]) or SECTION_SEPARATOR_PAT.search(line_buffer[-1])):
            pass
        elif line_buffer:
            iterator = iter(line_buffer + list(iterator))

        return ExcitedStateExcitonDifferenceDM(
            hole_center_ang=rh_center,
            electron_center_ang=re_center,
            electron_hole_separation_ang=re_rh_sep,
            hole_size_ang=hole_size_xyz,
            electron_size_ang=electron_size_xyz,
        )

    def parse(self, iterator: LineIterator, first_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting parsing of Excited State Analysis block.")

        try:
            line = next(iterator)  # Consume "Analysis of Unrelaxed Density Matrices"
            if not UNRELAXED_DM_HEADER_PAT.search(line):
                logger.warning(f"Expected 'Analysis of Unrelaxed Density Matrices' header, got: {line.strip()}")
                # Potentially push back 'line' and return if this is a critical error
                # For now, proceed, but parsing might fail for the first state.
                # iterator = iter([line] + list(iterator))
                # return
        except StopIteration as e:
            raise ParsingError("File ended unexpectedly after Excited State Analysis header.") from e

        current_state_num = 0
        lines_buffer: list[str] = []  # To push back lines if a state isn't fully parsed or at the end

        while True:  # Loop over each state (Singlet 1, Singlet 2, ...)
            try:
                if lines_buffer:  # Process buffered lines first
                    line = lines_buffer.pop(0)
                else:
                    line = next(iterator)
            except StopIteration:
                logger.debug("Reached end of file while looking for next state in Excited State Analysis.")
                break

            if SECTION_SEPARATOR_PAT.search(line) and not line.strip("-\s"):  # Ensure it's not part of some other text
                # This often separates states or ends the whole analysis block for a state.
                # Check if the *next* line is a new state header or end of major block
                try:
                    next_peek_line = next(iterator)
                    lines_buffer.append(next_peek_line)  # Buffer it immediately
                    if not STATE_HEADER_PAT.search(next_peek_line) and not (
                        EXCITED_STATE_ANALYSIS_HEADER_PAT.search(next_peek_line)
                        or re.search(r"^\s*Transition Density Matrix Analysis\s*$", next_peek_line)
                        or re.search(r"^\s*SA-NTO Decomposition\s*$", next_peek_line)
                    ):
                        # If not a new state or major block, this separator was internal to a sub-parser
                        # or the end of the entire ESA block. For now, assume end of ESA block.
                        logger.debug(
                            "Separator found, and next line doesn't start a new state or known major block. Ending Excited State Analysis parsing."
                        )
                        # iterator = iter(lines_buffer + list(iterator)) # Push back all buffered
                        # lines_buffer.clear()
                        break  # Exit main state loop
                except StopIteration:
                    break  # End of file

            match_state = STATE_HEADER_PAT.search(line)
            if not match_state:
                if not BLANK_LINE_PAT.search(line) and line.strip():  # Avoid logging for blank lines
                    logger.debug(f"Skipping non-state-header line in ESA: {line.strip()}")
                # Check for end of major block if no state header is found after a while
                if (
                    EXCITED_STATE_ANALYSIS_HEADER_PAT.search(line)  # Should not happen
                    or re.search(r"^\s*Transition Density Matrix Analysis\s*$", line)
                    or re.search(r"^\s*SA-NTO Decomposition\s*$", line)
                ):
                    logger.debug("Found start of next major block. Ending Excited State Analysis parsing.")
                    lines_buffer.insert(0, line)  # Push back this line
                    break
                continue

            multiplicity = match_state.group(1)
            state_number = int(match_state.group(2))
            current_state_num = state_number  # For logging
            logger.debug(f"Parsing analysis for {multiplicity} {state_number}")

            # Initialize data holders for the current state
            no_data: ExcitedStateNOData | None = None
            # ... initialize other sub-data holders (Mulliken, Multipole, Exciton) ...

            # Loop for sub-sections of this state (NOs, Mulliken, Multipole, Exciton)
            # This needs careful management of the iterator and line peeking/pushing back
            # For now, let's try to parse NOs as a first step

            temp_state_lines_buffer: list[str] = []
            parsed_nos_for_state = False
            # Add similar flags for other subsections
            parsed_mulliken_state_dm = False
            parsed_multipole_state_dm = False
            current_mulliken_data: ExcitedStateMulliken | None = None
            current_multipole_data: ExcitedStateMultipole | None = None
            current_exciton_diff_dm_data: ExcitedStateExcitonDifferenceDM | None = None
            parsed_exciton_diff_dm = False

            for _ in range(50):  # Max lines to check for subsections of a single state
                try:
                    if temp_state_lines_buffer:  # process buffered line for this state first
                        line = temp_state_lines_buffer.pop(0)
                    else:
                        line = next(iterator)
                except StopIteration:
                    logger.warning(f"File ended while parsing subsections for {multiplicity} {state_number}.")
                    # iterator = iter(temp_state_lines_buffer + list(iterator)) # Push back any remaining
                    # temp_state_lines_buffer.clear()
                    break  # Break from subsection loop

                # If we encounter a new state header or the main section separator, this state is done.
                if STATE_HEADER_PAT.search(line) or SECTION_SEPARATOR_PAT.search(line):
                    # iterator = iter([line] + temp_state_lines_buffer + list(iterator)) # Push back current and buffered
                    # temp_state_lines_buffer.clear()
                    lines_buffer.insert(0, line)  # Push back this line for the outer loop
                    break  # End of this state's subsections

                if NOS_HEADER_PAT.search(line) and not parsed_nos_for_state:
                    no_data_parsed = self._parse_no_data(iterator, line)  # iterator is advanced by helper
                    if no_data_parsed.participation_ratio_pr_no is not None:  # Check if it found something meaningful
                        no_data = no_data_parsed
                        parsed_nos_for_state = True
                    # The _parse_no_data should have consumed its lines or pushed back unconsumed ones.
                    # The iterator should be at the line *after* the NOs block.
                    continue  # Important: restart loop to get next line for next subsection.

                if MULLIKEN_POP_HEADER_PAT.search(line) and not parsed_mulliken_state_dm:
                    mulliken_data_parsed = self._parse_mulliken_state_dm(iterator, line, results.input_geometry)
                    if mulliken_data_parsed:
                        logger.debug(f"Parsed Mulliken (State/Diff DM) for state {state_number}")
                        # Need to store mulliken_data_parsed for the current state analysis
                        parsed_mulliken_state_dm = True
                        current_mulliken_data = mulliken_data_parsed
                    continue

                if MULTIPOLE_DM_HEADER_PAT.search(line) and not parsed_multipole_state_dm:
                    multipole_data_parsed = self._parse_multipole_state_dm(iterator, line)
                    if multipole_data_parsed:
                        logger.debug(f"Parsed Multipole (State DM) for state {state_number}")
                        # Need to store multipole_data_parsed for the current state analysis
                        parsed_multipole_state_dm = True
                        current_multipole_data = multipole_data_parsed
                    continue

                if EXCITON_DIFF_DM_HEADER_PAT.search(line) and not parsed_exciton_diff_dm:
                    exciton_data_parsed = self._parse_exciton_state_diff_dm(iterator, line)
                    if exciton_data_parsed:
                        logger.debug(f"Parsed Exciton (Diff DM) for state {state_number}")
                        parsed_exciton_diff_dm = True
                        current_exciton_diff_dm_data = exciton_data_parsed
                    continue

                if not line.strip():  # Skip blank lines between subsections
                    continue

                # If line is not a recognized subsection header and not blank, buffer it for now
                # Or log as unparsed within the state. For simplicity, buffer first.
                temp_state_lines_buffer.append(line)

            # After attempting to parse all subsections for the state:
            if (
                no_data or parsed_mulliken_state_dm or parsed_multipole_state_dm or parsed_exciton_diff_dm
            ):  # Or other checks to see if any meaningful data was parsed for the state
                # Retrieve the actual parsed data objects (no_data, mulliken_data, etc.)
                # For now, placeholders are used in ExcitedStateDetailedAnalysis constructor
                # Need to ensure that mulliken_data_parsed (and others) are captured and used here.
                # This part needs refinement to collect all parsed subsections.
                current_mulliken_data_to_store: ExcitedStateMulliken | None = None
                if parsed_mulliken_state_dm and "mulliken_data_parsed" in locals() and current_mulliken_data:
                    current_mulliken_data_to_store = current_mulliken_data

                current_multipole_data_to_store: ExcitedStateMultipole | None = None
                if parsed_multipole_state_dm and "multipole_data_parsed" in locals() and current_multipole_data:
                    current_multipole_data_to_store = current_multipole_data

                current_exciton_data_to_store: ExcitedStateExcitonDifferenceDM | None = None
                if parsed_exciton_diff_dm and "exciton_data_parsed" in locals() and current_exciton_diff_dm_data:
                    current_exciton_data_to_store = current_exciton_diff_dm_data

                results.excited_state_detailed_analyses_list.append(
                    ExcitedStateDetailedAnalysis(
                        state_number=state_number,
                        multiplicity=multiplicity,
                        no_data=no_data,  # This should be the actual no_data object
                        mulliken_analysis=current_mulliken_data_to_store,
                        multipole_analysis=current_multipole_data_to_store,
                        exciton_difference_dm_analysis=current_exciton_data_to_store,  # Placeholder
                    )
                )
            else:
                logger.warning(
                    f"No substantial data parsed for {multiplicity} {state_number} in Excited State Analysis."
                )

            # Push back any unconsumed lines from this state's parsing attempt
            if temp_state_lines_buffer:
                lines_buffer.extend(temp_state_lines_buffer)
                temp_state_lines_buffer.clear()

            # Check if the line that broke the subsection loop (now in lines_buffer[0] if pushed back)
            # indicates the end of all states.
            if lines_buffer and (
                re.search(r"^\s*Transition Density Matrix Analysis\s*$", lines_buffer[0])
                or re.search(r"^\s*SA-NTO Decomposition\s*$", lines_buffer[0])
            ):
                logger.debug("Next major block detected. Finalizing Excited State Analysis.")
                break  # Exit main state loop

        # After the loop, push back any remaining unconsumed lines from the lines_buffer
        if lines_buffer:
            iterator = iter(lines_buffer + list(iterator))

        logger.info(f"Parsed detailed analysis for {len(results.excited_state_detailed_analyses_list)} states.")

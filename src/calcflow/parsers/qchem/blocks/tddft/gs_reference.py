# Placeholder for TDDFT Block Parsers

import re
from collections.abc import Sequence
from typing import Literal

from calcflow.parsers.qchem.typing import (
    Atom,
    DipoleMoment,
    GroundStateAtomPopulation,
    GroundStateMulliken,
    GroundStateMultipole,
    GroundStateNOData,
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
NOS_RKS_HEADER_PAT = re.compile(r"^\s*NOs\s*(?!(\s*(alpha|beta|spin-traced)\s*))\s*$")
NOS_ALPHA_HEADER_PAT = re.compile(r"^\s*NOs \((alpha)\)\s*$")
NOS_BETA_HEADER_PAT = re.compile(r"^\s*NOs \((beta)\)\s*$")
NOS_SPIN_TRACED_HEADER_PAT = re.compile(r"^\s*NOs \((spin-traced)\)\s*$")

NOS_FRONTIER_LABEL_PAT = re.compile(r"^\s*Occupation of frontier NOs:\s*$")
NOS_FRONTIER_VALUES_PAT = re.compile(r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")

NOS_NUM_ELECTRONS_PAT = re.compile(r"^\s+Number of electrons:\s*(-?\d+\.\d+)")
NOS_UNPAIRED_PAT = re.compile(r"^\s+Number of unpaired electrons: n_u =\s*(-?\d+\.\d+), n_u,nl =\s*(-?\d+\.\d+)")
NOS_PR_NO_PAT = re.compile(r"^\s+NO participation ratio \(PR_NO\):\s*(-?\d+\.\d+)")

# # Mulliken Population (State/Difference DM)
MULLIKEN_POP_HEADER_PAT = re.compile(r"^\s*Mulliken Population Analysis \(State/Difference DM\)\s*$")
MULLIKEN_SUM_LINE_PAT = re.compile(r"^\s*Sum:\s+(-?\d+\.\d+)")  # Marks end of atom list for Mulliken

# # Mulliken Population (Ground State DM within Excited State Analysis)
MULLIKEN_GS_POP_HEADER_PAT = re.compile(r"^\s*Mulliken Population Analysis \(State DM\)\s*$")

# Atom line (RKS): index, symbol, charge
MULLIKEN_GS_ATOM_RKS_PAT = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s*$")
# Atom line (UKS): index, symbol, charge, spin
MULLIKEN_GS_ATOM_UKS_PAT = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")

# Separator line within Mulliken tables (e.g., before Sum)
MULLIKEN_INTERNAL_SEPARATOR_PAT = re.compile(r"^\s*-{10,}\s*$")

# Sum line: charge sum, optional spin sum
MULLIKEN_SUM_LINE_PAT = re.compile(r"^\s*Sum:\s+(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?\s*$")

# # Multipole Moment Analysis (Density Matrix)
MULTIPOLE_DM_HEADER_PAT = re.compile(r"^\s*Multipole moment analysis of the density matrix\s*$")
MOLECULAR_CHARGE_PAT = re.compile(r"^\s+Molecular charge:\s+(-?\d+\.\d+)")
NUM_ELECTRONS_MULTIPOLE_PAT = re.compile(
    r"^\s+Number of electrons:\s+(-?\d+\.\d+)"
)  # Can be distinct from NOs n_electrons
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

        gs_no_data_rks_or_spin_traced: GroundStateNOData | None = None
        gs_no_data_alpha: GroundStateNOData | None = None
        gs_no_data_beta: GroundStateNOData | None = None
        gs_mulliken_data: GroundStateMulliken | None = None
        gs_multipole_data: GroundStateMultipole | None = None

        max_lines_to_scan = 200  # Increased safeguard for potentially more NO blocks
        lines_scanned = 0

        while lines_scanned < max_lines_to_scan:
            try:
                line = results.buffered_line if results.buffered_line else next(iterator)
                results.buffered_line = None  # Consume buffer
                lines_scanned += 1
            except StopIteration:  # pragma: no cover
                logger.debug("Reached end of iterator during Ground State Reference parsing.")  # pragma: no cover
                break  # pragma: no cover

            if UNRELAXED_DM_HEADER_PAT.search(line):
                logger.debug(
                    f"Found '{UNRELAXED_DM_HEADER_PAT.pattern}' header; current GS Ref section ends. Buffering line."
                )
                results.buffered_line = line
                break

            # Try to parse specific sub-sections if not already done
            # Order matters: check for more specific UKS NOs before generic RKS NOs.
            if gs_no_data_alpha is None and NOS_ALPHA_HEADER_PAT.search(line):
                gs_no_data_alpha = self._parse_no_data(iterator, line, results, "alpha")
                if results.buffered_line:  # Sub-parser buffered a line, re-evaluate it
                    continue
                # If no buffered line, it means sub-parser consumed up to its end or iterator end
                # So, we should try to get a new line in the next iteration.
                # No 'continue' here if sub-parser successfully finished without buffering means iterator is at next line.
                # Correction: The sub-parser *always* either buffers or implies iterator is exhausted for its section.
                # The main loop should always re-evaluate from results.buffered_line or next(iterator).
                # The 'continue' ensures that if a sub-parser does its job and potentially buffers,
                # the main loop re-evaluates the state, including the new buffered line.
                continue

            if gs_no_data_beta is None and NOS_BETA_HEADER_PAT.search(line):
                gs_no_data_beta = self._parse_no_data(iterator, line, results, "beta")
                if results.buffered_line:
                    continue
                continue

            if gs_no_data_rks_or_spin_traced is None and NOS_SPIN_TRACED_HEADER_PAT.search(line):
                gs_no_data_rks_or_spin_traced = self._parse_no_data(iterator, line, results, "spin-traced")
                if results.buffered_line:
                    continue
                continue

            # This must come AFTER specific (alpha, beta, spin-traced) checks
            if gs_no_data_rks_or_spin_traced is None and NOS_RKS_HEADER_PAT.search(line):
                gs_no_data_rks_or_spin_traced = self._parse_no_data(iterator, line, results, "rks")
                if results.buffered_line:
                    continue
                continue

            if gs_mulliken_data is None and MULLIKEN_GS_POP_HEADER_PAT.search(line):
                gs_mulliken_data = self._parse_mulliken_ground_state_dm(iterator, line, results.input_geometry, results)
                if results.buffered_line:
                    continue
                continue

            if gs_multipole_data is None and MULTIPOLE_DM_HEADER_PAT.search(line):
                gs_multipole_data = self._parse_multipole_state_dm(iterator, line, results)
                if results.buffered_line:
                    continue
                continue

            if line.strip() == "" or SECTION_SEPARATOR_PAT.search(line):
                continue

            logger.warning(
                f"Unrecognized line encountered in Ground State Reference block: '{line.strip()}'. "
                "Assuming end of this block and buffering line."
            )  # pragma: no cover
            results.buffered_line = line  # pragma: no cover
            break  # pragma: no cover

        if lines_scanned >= max_lines_to_scan and not results.buffered_line:  # pragma: no cover
            logger.warning("GroundStateReferenceParser exceeded max lines scan.")  # pragma: no cover

        if (
            gs_no_data_alpha
            or gs_no_data_beta
            or gs_no_data_rks_or_spin_traced
            or gs_mulliken_data
            or gs_multipole_data
        ):
            results.gs_reference_analysis = GroundStateReferenceAnalysis(
                no_data_rks_or_spin_traced=gs_no_data_rks_or_spin_traced,
                no_data_alpha=gs_no_data_alpha,
                no_data_beta=gs_no_data_beta,
                mulliken=gs_mulliken_data,
                multipole=gs_multipole_data,
            )
            logger.info("Stored Ground State Reference analysis data.")
        else:  # pragma: no cover
            logger.warning("No data components parsed for Ground State Reference block.")  # pragma: no cover
            # results.parsing_warnings.append("No data components parsed for Ground State Reference block.") # Too noisy

        logger.debug("Finished parsing Ground State (Reference) block.")

    def _parse_no_data(
        self,
        iterator: LineIterator,
        header_line: str,  # This is the NOs header line, e.g. "NOs (alpha)"
        results: _MutableCalculationData,
        no_type: Literal["alpha", "beta", "spin-traced", "rks"],
    ) -> GroundStateNOData:
        logger.debug(f"Parsing NO data for type: {no_type}, starting with header: '{header_line.strip()}'")
        # The header_line itself has been "consumed" by the check in the main parse method.
        # We now parse the lines immediately following it.

        frontier_occ: list[float] | None = None
        num_e: float | None = None
        nu: float | None = None
        nunl: float | None = None
        pr_no: float | None = None

        expecting_frontier_values = False  # State to track if next line should be frontier values

        max_data_lines_for_no_block = 7  # Header + Label + Values + N_electrons + N_unpaired + PR_NO + Separator
        lines_read_in_block = 0

        while lines_read_in_block < max_data_lines_for_no_block:
            try:
                line = next(iterator)
                lines_read_in_block += 1
            except StopIteration:  # pragma: no cover
                logger.debug(f"NOs ({no_type}): Iterator ended while parsing data lines.")  # pragma: no cover
                break  # pragma: no cover

            if expecting_frontier_values:
                if m_vals := NOS_FRONTIER_VALUES_PAT.search(line):
                    frontier_occ = [float(m_vals.group(1)), float(m_vals.group(2))]
                    logger.debug(f"NOs ({no_type}): Found frontier_occ values: {frontier_occ}")
                    expecting_frontier_values = False  # Reset flag
                    continue  # Values processed, move to next data line or terminator
                else:  # pragma: no cover
                    logger.warning(
                        f"NOs ({no_type}): Expected frontier occupation values but found: '{line.strip()}'. "
                        "Stopping parse for this NO block and buffering line."
                    )  # pragma: no cover
                    results.buffered_line = line  # pragma: no cover
                    break  # pragma: no cover

            # Check for terminators indicating end of THIS specific NO block
            # or start of a new major section.
            is_terminator = False
            if SECTION_SEPARATOR_PAT.search(line) or BLANK_LINE_PAT.search(line):
                is_terminator = True
                logger.debug(f"NOs ({no_type}): Found separator/blank line: '{line.strip()}'. End of NO block.")
            elif (
                MULLIKEN_GS_POP_HEADER_PAT.search(line)
                or MULTIPOLE_DM_HEADER_PAT.search(line)
                or STATE_HEADER_PAT.search(line)
                or UNRELAXED_DM_HEADER_PAT.search(line)
                or NOS_RKS_HEADER_PAT.search(line)
                or NOS_ALPHA_HEADER_PAT.search(line)
                or NOS_BETA_HEADER_PAT.search(line)
                or NOS_SPIN_TRACED_HEADER_PAT.search(line)
            ):
                is_terminator = True
                logger.debug(f"NOs ({no_type}): Found next section header: '{line.strip()}'. End of NO block.")

            if is_terminator:
                results.buffered_line = line
                break

            # Parse field labels / data for the current line
            if NOS_FRONTIER_LABEL_PAT.search(line):
                logger.debug(f"NOs ({no_type}): Found frontier occupation label: '{line.strip()}'")
                expecting_frontier_values = True  # Set flag to parse next line for values
                continue

            if m := NOS_NUM_ELECTRONS_PAT.search(line):
                num_e = float(m.group(1))
                logger.debug(f"NOs ({no_type}): Found num_e: {num_e}")
                continue

            if no_type in ["spin-traced", "rks"]:  # These fields only appear for spin-traced or RKS
                if m := NOS_UNPAIRED_PAT.search(line):
                    nu, nunl = float(m.group(1)), float(m.group(2))
                    logger.debug(f"NOs ({no_type}): Found unpaired e: nu={nu}, nunl={nunl}")
                    continue
                if m := NOS_PR_NO_PAT.search(line):
                    pr_no = float(m.group(1))
                    logger.debug(f"NOs ({no_type}): Found pr_no: {pr_no}")
                    continue

            logger.warning(
                f"NOs ({no_type}): Unrecognized data line: '{line.strip()}'. Assuming end of NO block."
            )  # pragma: no cover
            results.buffered_line = line  # pragma: no cover
            break  # pragma: no cover

        return GroundStateNOData(
            frontier_occupations=frontier_occ,
            n_electrons=num_e,
            n_unpaired=nu,
            n_unpaired_nl=nunl,
            pr_no=pr_no,
        )

    def _parse_mulliken_ground_state_dm(
        self,
        iterator: LineIterator,
        first_mulliken_gs_line: str,  # This is the MULLIKEN_GS_POP_HEADER_PAT line
        input_geometry: Sequence[Atom] | None,
        results: _MutableCalculationData,
    ) -> GroundStateMulliken | None:
        logger.debug(f"Parsing Mulliken GS State DM, starting with header: '{first_mulliken_gs_line.strip()}'")

        populations: list[GroundStateAtomPopulation] = []
        has_spin_column = False

        try:
            column_header_line = next(iterator)
            logger.debug(f"Mulliken GS: Consumed column header: '{column_header_line.strip()}'")
            if "Spin (e)" in column_header_line:
                has_spin_column = True
            logger.debug(f"Mulliken GS: Detected spin column: {has_spin_column}")

            separator_line = next(iterator)
            logger.debug(f"Mulliken GS: Consumed separator line: '{separator_line.strip()}'")

            max_atom_lines = 200
            for i in range(max_atom_lines):
                try:
                    line = next(iterator)
                    logger.debug(f"Mulliken GS: Processing atom line candidate ({i + 1}): '{line.strip()}'")
                except StopIteration:  # pragma: no cover
                    logger.debug("Mulliken GS: Iterator ended while expecting atom or sum line.")  # pragma: no cover
                    break  # pragma: no cover

                # Check for terminators first (next major section headers)
                if (
                    MULTIPOLE_DM_HEADER_PAT.search(line)
                    or UNRELAXED_DM_HEADER_PAT.search(line)
                    or STATE_HEADER_PAT.search(line)
                    or NOS_RKS_HEADER_PAT.search(line)
                    or NOS_ALPHA_HEADER_PAT.search(line)
                    or NOS_BETA_HEADER_PAT.search(line)
                    or NOS_SPIN_TRACED_HEADER_PAT.search(line)
                ):
                    results.buffered_line = line
                    logger.debug(f"Mulliken GS: Found next section header: '{line.strip()}'. Buffering.")
                    break

                m_atom_match = None
                if has_spin_column:
                    m_atom_match = MULLIKEN_GS_ATOM_UKS_PAT.search(line)
                else:
                    m_atom_match = MULLIKEN_GS_ATOM_RKS_PAT.search(line)

                if m_atom_match:
                    idx = int(m_atom_match.group(1))
                    sym_from_output = m_atom_match.group(2)
                    chg = float(m_atom_match.group(3))
                    spin_val: float | None = None
                    if has_spin_column:
                        spin_val = float(m_atom_match.group(4))

                    atom_symbol = sym_from_output
                    if input_geometry and 0 < idx <= len(input_geometry):
                        atom_symbol = input_geometry[idx - 1].symbol
                    elif input_geometry:  # pragma: no cover
                        logger.warning(
                            f"Mulliken atom index {idx} out of range for input geometry size {len(input_geometry)}."
                        )  # pragma: no cover

                    populations.append(
                        GroundStateAtomPopulation(
                            atom_index=idx - 1,
                            symbol=atom_symbol,
                            charge_e=chg,
                            spin_e=spin_val,
                        )
                    )
                    logger.debug(f"Mulliken GS: Parsed atom {idx} {atom_symbol} Charge={chg} Spin={spin_val}")
                    continue

                if MULLIKEN_SUM_LINE_PAT.search(line):
                    logger.debug(f"Mulliken GS: Found sum line: '{line.strip()}'. End of atom data.")
                    break

                # Check for blank lines or internal separators specifically for Mulliken block
                if BLANK_LINE_PAT.search(line) or MULLIKEN_INTERNAL_SEPARATOR_PAT.search(line):
                    logger.debug(f"Mulliken GS: Encountered blank/separator line: '{line.strip()}'. Continuing.")
                    continue

                logger.warning(
                    f"Mulliken GS: Unexpected line in atom table: '{line.strip()}'. Buffering and exiting Mulliken parse."
                )  # pragma: no cover
                results.buffered_line = line  # pragma: no cover
                break  # pragma: no cover

        except StopIteration:  # pragma: no cover
            logger.debug(
                "Iterator ended prematurely during Mulliken GS State DM parsing (e.g., after header/separator)."
            )  # pragma: no cover

        if not populations and first_mulliken_gs_line:  # pragma: no cover
            logger.debug(
                "No Mulliken populations parsed in GS State DM block, though header was found."
            )  # pragma: no cover

        return GroundStateMulliken(populations=populations)

    def _parse_multipole_state_dm(
        self, iterator: LineIterator, first_multipole_line: str, results: _MutableCalculationData
    ) -> GroundStateMultipole | None:
        if not MULTIPOLE_DM_HEADER_PAT.search(first_multipole_line):  # pragma: no cover
            if first_multipole_line.strip():  # pragma: no cover
                results.buffered_line = first_multipole_line  # pragma: no cover
            return None  # pragma: no cover
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
                    g1, g2, g3 = m.groups()
                    cec_xyz = (float(g1), float(g2), float(g3))
                if m := CENTER_NUCLEAR_CHARGE_PAT.search(line):
                    g1, g2, g3 = m.groups()
                    cnc_xyz = (float(g1), float(g2), float(g3))
                if m := DIPOLE_TOTAL_PAT.search(line):
                    dip_tot = float(m.group(1))
                if m := DIPOLE_MOMENT_COMPONENTS_PAT.search(line):
                    g1, g2, g3 = m.groups()
                    dip_xyz = (float(g1), float(g2), float(g3))
                if m := RMS_TOTAL_SIZE_PAT.search(line):
                    # rms_tot = float(m.group(1))
                    pass

                # Refined logic for RMS density components
                if m_rms_match := RMS_DENSITY_SIZE_COMPONENTS_PAT.search(line):
                    # This specific line is for RMS density components.
                    g1, g2, g3 = m_rms_match.groups()
                    rms_xyz = (float(g1), float(g2), float(g3))
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
                except StopIteration:  # pragma: no cover
                    break  # pragma: no cover
        except StopIteration:  # pragma: no cover
            pass  # pragma: no cover

        final_dipole: DipoleMoment | None = None
        if dip_xyz and dip_tot is not None:
            final_dipole = DipoleMoment(dip_xyz[0], dip_xyz[1], dip_xyz[2], dip_tot)
        elif dip_xyz:  # Fallback if total dipole line was missing but components were found
            calculated_total = (dip_xyz[0] ** 2 + dip_xyz[1] ** 2 + dip_xyz[2] ** 2) ** 0.5
            final_dipole = DipoleMoment(dip_xyz[0], dip_xyz[1], dip_xyz[2], calculated_total)
            logger.debug(
                f"Calculated total dipole {calculated_total:.4f} from components for multipole section."
            )  # pragma: no cover

        if not (
            mol_chg is not None
            or num_e is not None
            or final_dipole is not None
            or cec_xyz is not None
            or cnc_xyz is not None
            or rms_xyz is not None  # Crucial check: if rms_xyz parsed, this should lead to object creation
        ):  # pragma: no cover
            logger.warning(
                f"Failed to parse significant multipole data. Fields: mol_chg={mol_chg}, num_e={num_e}, "
                f"final_dipole_present={final_dipole is not None}, cec_xyz_present={cec_xyz is not None}, "
                f"cnc_xyz_present={cnc_xyz is not None}, rms_xyz_present={rms_xyz is not None}"
            )  # pragma: no cover
            if line_buffer:  # pragma: no cover
                logger.warning(
                    "Sub-parser _parse_multipole_state_dm had leftover lines in its internal buffer, not re-buffering globally."
                )  # pragma: no cover
            return None  # pragma: no cover

        logger.debug(
            f"Successfully parsed multipole data: Dipole total = {final_dipole.magnitude if final_dipole else None}, RMS_XYZ = {rms_xyz}"
        )
        return GroundStateMultipole(mol_chg, num_e, cec_xyz, cnc_xyz, final_dipole, rms_xyz)

# Placeholder for TDDFT Block Parsers

import re
from collections.abc import Sequence

from calcflow.parsers.qchem.typing import (
    Atom,
    ExcitonAnalysisTransitionDM,
    LineIterator,
    SectionParser,
    TransitionDensityMatrixDetailedAnalysis,
    TransitionDMAtomPopulation,
    TransitionDMCTNumbers,
    TransitionDMMulliken,
    _MutableCalculationData,
)
from calcflow.utils import logger

STATE_HEADER_PAT = re.compile(r"^\s*(Singlet|Triplet)\s+(\d+)\s*:\s*$")

# --- Transition Density Matrix Analysis Block ---
TRANS_DM_ANALYSIS_HEADER_PAT = re.compile(r"^\s*Transition Density Matrix Analysis\s*$")
# STATE_HEADER_PAT is reused: r"^\s*(Singlet|Triplet)\s+(\d+)\s*:\s*$"

# Sub-section patterns for Transition DM Analysis
# Mulliken Population (Transition DM)
MULLIKEN_TRANS_DM_HEADER_PAT = re.compile(r"^\s*Mulliken Population Analysis \(Transition DM\)\s*$")
# Atom      Trans. (e)              h+              e-           Del q
# --------------------------------------------------------------------
#  1 H       -0.000020        0.016997       -0.530519       -0.513522
MULLIKEN_TRANS_DM_ATOM_LINE_PAT = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)
MULLIKEN_TRANS_DM_SUM_LINE_PAT = re.compile(r"^\s*Sum:\s+(-?\d+\.\d+)")
SUM_ABS_TRANS_CHARGES_PAT = re.compile(r"^\s*Sum of absolute trans\. charges, QTa =\s*(-?\d+\.\d+)")
SUM_SQ_TRANS_CHARGES_PAT = re.compile(r"^\s*Sum of squared  trans\. charges, QT2 =\s*(-?\d+\.\d+)")

# CT Numbers (Mulliken)
CT_NUMBERS_HEADER_PAT = re.compile(r"^\s*CT numbers \(Mulliken\)\s*$")
CT_OMEGA_PAT = re.compile(r"^\s*omega\s*=\s*(-?\d+\.\d+)")
CT_TWO_ALPHA_BETA_PAT = re.compile(r"^\s*2<alpha\|beta>\s*=\s*(-?\d+\.\d+)")
CT_LOC_PAT = re.compile(r"^\s*LOC\s*=\s*(-?\d+\.\d+)")
CT_LOCA_PAT = re.compile(r"^\s*LOCa\s*=\s*(-?\d+\.\d+)")
CT_PHE_PAT = re.compile(r"^\s*<Phe>\s*=\s*(-?\d+\.\d+)")

# Exciton Analysis (Transition Density Matrix)
EXCITON_TRANS_DM_HEADER_PAT = re.compile(r"^\s*Exciton analysis of the transition density matrix\s*$")
TRANS_DIPOLE_MOMENT_TOTAL_PAT = re.compile(r"^\s*Trans\. dipole moment\s*\[D\]:\s*(-?\d+\.\d+)")
CARTESIAN_COMPONENTS_D_PAT = re.compile(
    r"^\s*Cartesian components\s*\[D\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\]"
)
TRANS_R_SQUARED_AU_PAT = re.compile(r"^\s*Transition <r\^2>\s*\[a\.u\.\]:\s*(-?\d+\.\d+)")
CARTESIAN_COMPONENTS_AU_PAT = re.compile(
    r"^\s*Cartesian components\s*\[a\.u\.\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\]"
)
EXCITON_RH_PAT = re.compile(r"^\s*<r_h>\s*\[Ang\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\]")
EXCITON_RE_PAT = re.compile(r"^\s*<r_e>\s*\[Ang\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\]")
EXCITON_RE_RH_SEP_PAT = re.compile(r"^\s*\|<r_e - r_h>\|\s*\[Ang\]:\s*(-?\d+\.\d+)")
EXCITON_SIZE_LINE_PAT = re.compile(
    r"^\s*(Hole size|Electron size)\s*\[Ang\]:\s*(-?\d+\.\d+)"
)  # For scalar hole/electron size
CARTESIAN_COMPONENTS_ANG_PAT = re.compile(
    r"^\s*Cartesian components\s*\[Ang\]:\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\]"
)  # For Angstrom components
RMS_ELECTRON_HOLE_SEP_PAT = re.compile(r"^\s*RMS electron-hole separation\s*\[Ang\]:\s*(-?\d+\.\d+)")
COVARIANCE_RH_RE_PAT = re.compile(r"^\s*Covariance\(r_h, r_e\)\s*\[Ang\^2\]:\s*(-?\d+\.\d+)")
CORRELATION_COEFF_PAT = re.compile(r"^\s*Correlation coefficient:\s*(-?\d+\.\d+)")
CENTER_OF_MASS_SIZE_PAT = re.compile(r"^\s*Center-of-mass size\s*\[Ang\]:\s*(-?\d+\.\d+)")

SECTION_SEPARATOR_PAT = re.compile(r"^\s*-{20,}\s*$")  # A line of dashes, often ends state blocks
BLANK_LINE_PAT = re.compile(r"^\s*$")


class TransitionDensityMatrixParser(SectionParser):
    """Parses the 'Transition Density Matrix Analysis' block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return TRANS_DM_ANALYSIS_HEADER_PAT.search(line) is not None

    def _parse_mulliken_transition_dm(
        self, iterator: LineIterator, first_mulliken_line: str, input_geometry: Sequence[Atom] | None
    ) -> tuple[TransitionDMMulliken | None, float | None, float | None]:
        """Helper for Mulliken Population Analysis (Transition DM). Returns data, QTa, QT2."""
        if not MULLIKEN_TRANS_DM_HEADER_PAT.search(first_mulliken_line):
            logger.warning(f"Expected Mulliken Trans DM header, got: {first_mulliken_line.strip()}")
            return None, None, None

        populations: list[TransitionDMAtomPopulation] = []
        qta: float | None = None
        qt2: float | None = None
        line_buffer: list[str] = []
        line = first_mulliken_line

        try:
            # Consume table headers (Atom Trans.(e)..., and ---- lines)
            _ = next(iterator)  # Column header
            _ = next(iterator)  # Dashed line

            # Parse atom lines
            while True:
                line = next(iterator)
                line_buffer.append(line)
                match_atom = MULLIKEN_TRANS_DM_ATOM_LINE_PAT.search(line)
                if match_atom:
                    atom_idx_one_based = int(match_atom.group(1))
                    charge_trans_e = float(match_atom.group(3))
                    h_plus = float(match_atom.group(4))
                    e_minus = float(match_atom.group(5))
                    del_q = float(match_atom.group(6))
                    atom_symbol = "?"
                    if input_geometry and 0 < atom_idx_one_based <= len(input_geometry):
                        atom_symbol = input_geometry[atom_idx_one_based - 1].symbol
                    populations.append(
                        TransitionDMAtomPopulation(
                            atom_index=atom_idx_one_based - 1,
                            symbol=atom_symbol,
                            transition_charge_e=charge_trans_e,
                            hole_charge=h_plus,
                            electron_charge=e_minus,
                            delta_charge=del_q,
                        )
                    )
                elif MULLIKEN_TRANS_DM_SUM_LINE_PAT.search(line):
                    line_buffer.pop()  # Remove sum line
                    break  # End of atom table
                elif not line.strip():
                    continue
                else:  # Unexpected line in table
                    logger.warning(f"Unexpected line in Mulliken Trans DM table: {line.strip()}")
                    line_buffer.pop()  # Remove it
                    iterator = iter(line_buffer + [line] + list(iterator))  # Push back with current
                    break

            # After atom table, look for QTa and QT2
            for _ in range(4):  # Look a few lines ahead
                line = next(iterator)
                line_buffer.append(line)
                if SUM_ABS_TRANS_CHARGES_PAT.search(line):
                    qta = float(SUM_ABS_TRANS_CHARGES_PAT.search(line).group(1))
                elif SUM_SQ_TRANS_CHARGES_PAT.search(line):
                    qt2 = float(SUM_SQ_TRANS_CHARGES_PAT.search(line).group(1))
                elif (
                    CT_NUMBERS_HEADER_PAT.search(line)
                    or EXCITON_TRANS_DM_HEADER_PAT.search(line)
                    or STATE_HEADER_PAT.search(line)
                    or SECTION_SEPARATOR_PAT.search(line)
                ):
                    line_buffer.pop()
                    iterator = iter(line_buffer + [line] + list(iterator))
                    break  # Start of next subsection
                if qta is not None and qt2 is not None:
                    break

        except StopIteration:
            logger.warning("File ended unexpectedly during Mulliken Trans DM parsing.")

        iterator = iter(line_buffer + list(iterator))  # Push back unconsumed from buffer
        mulliken_data = TransitionDMMulliken(populations=populations) if populations else None
        return mulliken_data, qta, qt2

    def _parse_ct_numbers_trans_dm(self, iterator: LineIterator, first_ct_line: str) -> TransitionDMCTNumbers | None:
        """Helper to parse CT numbers (Mulliken) for Transition DM."""
        if not CT_NUMBERS_HEADER_PAT.search(first_ct_line):
            logger.warning(f"Expected CT Numbers header, got: {first_ct_line.strip()}")
            return None

        omega, two_alpha_beta, loc, loca, phe = None, None, None, None, None
        line_buffer: list[str] = []
        line = first_ct_line  # Already consumed by caller

        try:
            for _ in range(6):  # Max ~5 data lines + header
                if not CT_NUMBERS_HEADER_PAT.search(line) and _ == 0 and line is not first_ct_line:
                    logger.warning(f"CT Numbers parsing started with non-header: {line.strip()}")
                    break

                if CT_OMEGA_PAT.search(line):
                    omega = float(CT_OMEGA_PAT.search(line).group(1))
                elif CT_TWO_ALPHA_BETA_PAT.search(line):
                    two_alpha_beta = float(CT_TWO_ALPHA_BETA_PAT.search(line).group(1))
                elif CT_LOC_PAT.search(line):
                    loc = float(CT_LOC_PAT.search(line).group(1))
                elif CT_LOCA_PAT.search(line):
                    loca = float(CT_LOCA_PAT.search(line).group(1))
                elif CT_PHE_PAT.search(line):
                    phe = float(CT_PHE_PAT.search(line).group(1))
                    # This is typically the last line of CT numbers
                    break

                try:
                    line = next(iterator)
                    line_buffer.append(line)
                    # Check for end of this sub-block
                    if (
                        EXCITON_TRANS_DM_HEADER_PAT.search(line)
                        or STATE_HEADER_PAT.search(line)
                        or SECTION_SEPARATOR_PAT.search(line)
                        or MULLIKEN_TRANS_DM_HEADER_PAT.search(line)
                    ):
                        iterator = iter([line] + list(iterator))  # Push back current line
                        line_buffer.pop()
                        break
                except StopIteration:
                    break
        except StopIteration:
            logger.warning("File ended unexpectedly during CT Numbers (Trans DM) parsing.")

        if not any([omega, two_alpha_beta, loc, loca, phe]):
            logger.debug("No significant CT Numbers found in Transition DM block.")
            if line_buffer:  # Push back if we consumed lines but found nothing
                iterator = iter(line_buffer + list(iterator))
            return None

        # Push back any unconsumed lines only if the last one was a clear break header
        if line_buffer and (
            EXCITON_TRANS_DM_HEADER_PAT.search(line_buffer[-1])
            or STATE_HEADER_PAT.search(line_buffer[-1])
            or SECTION_SEPARATOR_PAT.search(line_buffer[-1])
        ):
            pass  # Already handled by break conditions (iterator already has the line)
        elif (
            line_buffer
        ):  # If buffer has content not pushed back by break conditions (e.g. StopIteration or all items parsed)
            iterator = iter(line_buffer + list(iterator))

        return TransitionDMCTNumbers(
            omega=omega,
            two_alpha_beta_overlap=two_alpha_beta,
            loc=loc,
            loc_a=loca,
            phe_overlap=phe,
        )

    def _parse_exciton_trans_dm(
        self, iterator: LineIterator, first_exciton_line: str
    ) -> ExcitonAnalysisTransitionDM | None:
        """Helper to parse Exciton analysis of the transition density matrix."""
        if not EXCITON_TRANS_DM_HEADER_PAT.search(first_exciton_line):
            logger.warning(f"Expected Exciton Trans DM header, got: {first_exciton_line.strip()}")
            return None

        data = {
            "total_transition_dipole_moment": None,
            "transition_dipole_moment_components": None,
            "transition_r_squared_au": None,
            "transition_r_squared_au_components": None,
            "hole_position_ang": None,
            "electron_position_ang": None,
            "hole_electron_distance_ang": None,
            "hole_size_ang": None,
            "hole_size_ang_components": None,
            "electron_size_ang": None,
            "electron_size_ang_components": None,
            "rms_electron_hole_separation_ang": None,
            "rms_electron_hole_separation_ang_components": None,
            "covariance_rh_re_ang_sq": None,
            "correlation_coefficient": None,
            "center_of_mass_size_ang": None,
            "center_of_mass_size_ang_components": None,
        }
        line_buffer: list[str] = []
        # first_exciton_line is already consumed by the caller check
        # and corresponds to EXCITON_TRANS_DM_HEADER_PAT

        try:
            # Context variable to know what Cartesian components refer to
            # e.g. "dipole", "r_squared", "hole_size", "electron_size", "rms_sep", "cm_size"
            last_scalar_metric_type: str | None = None

            for _i in range(25):  # Read ahead a fixed number of lines for this block
                line = next(iterator)
                line_buffer.append(line)

                if (
                    STATE_HEADER_PAT.search(line)
                    or SECTION_SEPARATOR_PAT.search(line)
                    or MULLIKEN_TRANS_DM_HEADER_PAT.search(line)
                    or CT_NUMBERS_HEADER_PAT.search(line)
                ):
                    line_buffer.pop()  # Remove the terminating line
                    iterator = iter([line] + list(iterator))  # Push back this line
                    break

                m = TRANS_DIPOLE_MOMENT_TOTAL_PAT.search(line)
                if m:
                    data["total_transition_dipole_moment"] = float(m.group(1))
                    last_scalar_metric_type = "dipole"
                    continue

                m = CARTESIAN_COMPONENTS_D_PAT.search(line)
                if m and last_scalar_metric_type == "dipole":
                    data["transition_dipole_moment_components"] = (
                        float(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                    )
                    last_scalar_metric_type = None  # Reset context
                    continue

                m = TRANS_R_SQUARED_AU_PAT.search(line)
                if m:
                    data["transition_r_squared_au"] = float(m.group(1))
                    last_scalar_metric_type = "r_squared"
                    continue

                m = CARTESIAN_COMPONENTS_AU_PAT.search(line)
                if m and last_scalar_metric_type == "r_squared":
                    data["transition_r_squared_au_components"] = (
                        float(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                    )
                    last_scalar_metric_type = None  # Reset context
                    continue

                m = EXCITON_RH_PAT.search(line)
                if m:
                    data["hole_position_ang"] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                    continue

                m = EXCITON_RE_PAT.search(line)
                if m:
                    data["electron_position_ang"] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                    continue

                m = EXCITON_RE_RH_SEP_PAT.search(line)
                if m:
                    data["hole_electron_distance_ang"] = float(m.group(1))
                    continue

                m = EXCITON_SIZE_LINE_PAT.search(line)
                if m:
                    size_type, value = m.group(1), float(m.group(2))
                    if size_type == "Hole size":
                        data["hole_size_ang"] = value
                        last_scalar_metric_type = "hole_size"
                    elif size_type == "Electron size":
                        data["electron_size_ang"] = value
                        last_scalar_metric_type = "electron_size"
                    continue

                # Generic Angstrom components, relies on last_scalar_metric_type
                m = CARTESIAN_COMPONENTS_ANG_PAT.search(line)
                if m:
                    components = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                    if last_scalar_metric_type == "hole_size":
                        data["hole_size_ang_components"] = components
                        last_scalar_metric_type = None
                    elif last_scalar_metric_type == "electron_size":
                        data["electron_size_ang_components"] = components
                        last_scalar_metric_type = None
                    elif last_scalar_metric_type == "rms_sep":  # Context set by RMS_ELECTRON_HOLE_SEP_PAT
                        data["rms_electron_hole_separation_ang_components"] = components
                        last_scalar_metric_type = None
                    elif last_scalar_metric_type == "cm_size":  # Context set by CENTER_OF_MASS_SIZE_PAT
                        data["center_of_mass_size_ang_components"] = components
                        last_scalar_metric_type = None
                    else:  # Could be <r_h> or <r_e> if they weren't direct matches, but they are.
                        # This path is more for RMS sep and CM size components
                        logger.debug(
                            f"CARTESIAN_COMPONENTS_ANG_PAT matched but context '{last_scalar_metric_type}' unclear: {line.strip()}"
                        )
                    continue

                m = RMS_ELECTRON_HOLE_SEP_PAT.search(line)
                if m:
                    data["rms_electron_hole_separation_ang"] = float(m.group(1))
                    last_scalar_metric_type = "rms_sep"
                    continue

                m = COVARIANCE_RH_RE_PAT.search(line)
                if m:
                    data["covariance_rh_re_ang_sq"] = float(m.group(1))
                    last_scalar_metric_type = None  # No components
                    continue

                m = CORRELATION_COEFF_PAT.search(line)
                if m:
                    data["correlation_coefficient"] = float(m.group(1))
                    last_scalar_metric_type = None  # No components
                    continue

                m = CENTER_OF_MASS_SIZE_PAT.search(line)
                if m:
                    data["center_of_mass_size_ang"] = float(m.group(1))
                    last_scalar_metric_type = "cm_size"
                    continue

                if not line.strip():  # Skip blank lines
                    continue

                # If no pattern matched, it might be the end or an unknown line
                logger.debug(f"Unparsed line in Exciton Trans DM: {line.strip()}")
                # Do not break here, allow loop to continue for other lines or stop via iteration limit/headers

        except StopIteration:
            logger.debug("File ended during Exciton Trans DM parsing for a state.")
        finally:  # Push back any unconsumed lines from local buffer if loop broke early
            # This is tricky if the loop finishes by iterating 25 times vs. header break.
            # The header break condition already handles pushback of the header itself.
            # If it finishes due to _i, unconsumed lines in line_buffer are lost unless handled.
            # However, standard practice is to let outer loop manage iterator.
            # The main concern is that if we _don't_ find a terminating header,
            # the line_buffer might contain lines that the _next_ parser needs.
            # For simplicity here, if loop terminates by exhaustion, this buffer is not pushed back.
            # This assumes subsections are contiguous and well-behaved.
            pass

        if not any(v is not None for v in data.values()):
            logger.debug("No significant Exciton Trans DM data found.")
            # Push back all consumed lines if nothing was found
            iterator = iter(line_buffer + list(iterator))
            return None

        return ExcitonAnalysisTransitionDM(**data)

    def parse(self, iterator: LineIterator, first_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting parsing of Transition Density Matrix Analysis block.")
        # Similar structure to ExcitedStateAnalysisParser: loop through states (Singlet X, ...)
        # Within each state, parse Mulliken (Trans DM), CT numbers, Exciton (Trans DM)

        current_qta: float | None = None  # For current state
        current_qt2: float | None = None  # For current state
        lines_buffer: list[str] = []

        while True:
            try:
                if lines_buffer:
                    line = lines_buffer.pop(0)
                else:
                    line = next(iterator)
            except StopIteration:
                break

            if SECTION_SEPARATOR_PAT.search(line) and not line.strip("-\s"):
                try:
                    next_peek_line = next(iterator)
                    lines_buffer.append(next_peek_line)
                    if not STATE_HEADER_PAT.search(next_peek_line) and not (
                        TRANS_DM_ANALYSIS_HEADER_PAT.search(next_peek_line)
                        or re.search(r"^\s*SA-NTO Decomposition\s*$", next_peek_line)
                    ):
                        break
                except StopIteration:
                    break

            match_state = STATE_HEADER_PAT.search(line)
            if not match_state:
                if not BLANK_LINE_PAT.search(line) and line.strip():
                    logger.debug(f"Skipping non-state-header line in TransDM: {line.strip()}")
                if TRANS_DM_ANALYSIS_HEADER_PAT.search(line) or re.search(r"^\s*SA-NTO Decomposition\s*$", line):
                    lines_buffer.insert(0, line)
                    break
                continue

            multiplicity = match_state.group(1)
            state_number = int(match_state.group(2))
            logger.debug(f"Parsing Transition DM analysis for {multiplicity} {state_number}")

            current_mulliken_trans_dm: TransitionDMMulliken | None = None
            current_ct_numbers: TransitionDMCTNumbers | None = None
            current_exciton_trans_dm: ExcitonAnalysisTransitionDM | None = None
            # Add flags for parsed subsections for this state
            parsed_mulliken_tdm = False
            parsed_ct_numbers_tdm = False
            parsed_exciton_tdm = False

            temp_state_lines_buffer: list[str] = []
            # Reset qta/qt2 for the new state
            current_qta = None
            current_qt2 = None

            for _ in range(50):  # Max lines for subsections of one state
                try:
                    if temp_state_lines_buffer:
                        line = temp_state_lines_buffer.pop(0)
                    else:
                        line = next(iterator)
                except StopIteration:
                    break

                if STATE_HEADER_PAT.search(line) or SECTION_SEPARATOR_PAT.search(line):
                    lines_buffer.insert(0, line)
                    break

                if MULLIKEN_TRANS_DM_HEADER_PAT.search(line) and not parsed_mulliken_tdm:
                    mdata, qta_val, qt2_val = self._parse_mulliken_transition_dm(iterator, line, results.input_geometry)
                    if mdata:
                        current_mulliken_trans_dm = mdata
                        current_qta = qta_val  # Store qta
                        current_qt2 = qt2_val  # Store qt2
                        # Removed direct assignment to mdata fields
                        parsed_mulliken_tdm = True
                    continue

                if CT_NUMBERS_HEADER_PAT.search(line) and not parsed_ct_numbers_tdm:
                    ct_data = self._parse_ct_numbers_trans_dm(iterator, line)
                    if ct_data:
                        logger.debug(f"Parsed CT Numbers for state {state_number}")
                        current_ct_numbers = ct_data
                        parsed_ct_numbers_tdm = True
                    continue

                if EXCITON_TRANS_DM_HEADER_PAT.search(line) and not parsed_exciton_tdm:
                    exciton_data = self._parse_exciton_trans_dm(iterator, line)
                    if exciton_data:
                        logger.debug(f"Parsed Exciton (Trans DM) for state {state_number}")
                        current_exciton_trans_dm = exciton_data
                        parsed_exciton_tdm = True
                    continue

                if not line.strip():
                    continue
                temp_state_lines_buffer.append(line)

            if current_mulliken_trans_dm or current_ct_numbers or current_exciton_trans_dm:  # or other data found
                results.transition_density_matrix_detailed_analyses_list.append(
                    TransitionDensityMatrixDetailedAnalysis(
                        state_number=state_number,
                        multiplicity=multiplicity,
                        mulliken_analysis=current_mulliken_trans_dm,
                        ct_numbers=current_ct_numbers,
                        exciton_analysis=current_exciton_trans_dm,
                        sum_abs_trans_charges_qta=current_qta,
                        sum_sq_trans_charges_qt2=current_qt2,
                    )
                )

            if temp_state_lines_buffer:
                lines_buffer.extend(temp_state_lines_buffer)

            if lines_buffer and (re.search(r"^\s*SA-NTO Decomposition\s*$", lines_buffer[0])):
                break

        if lines_buffer:
            iterator = iter(lines_buffer + list(iterator))

        logger.info(
            f"Parsed Transition DM analysis for {len(results.transition_density_matrix_detailed_analyses_list)} states."
        )

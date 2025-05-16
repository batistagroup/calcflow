import re
from typing import Any

from calcflow.parsers.qchem.typing import (
    ExcitonAnalysisTMData,
    ExcitonPropertiesSet,
    LineIterator,
    SectionParser,
    TransitionDensityMatrixDetailedAnalysis,
    TransitionDMAtomPopulation,
    TransitionDMCTNumbers,
    TransitionDMMulliken,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- Helper functions --- #


def safe_float(value: str | None) -> float | None:
    """Attempt to convert a string to a float, returning None on ValueError."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):  # pragma: no cover
        return None  # pragma: no cover


FLOAT_PATTERN = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
TRIPLET_PATTERN = re.compile(
    r"[\(\[]?\s*"
    rf"({FLOAT_PATTERN})[\s,]+"
    rf"({FLOAT_PATTERN})[\s,]+"
    rf"({FLOAT_PATTERN})\s*[\)\]]?"
)


def extract_float_triplet(text: str) -> tuple[float, float, float] | None:
    """Extracts a tuple of three floats from a string. Handles [x,y,z] or (x,y,z) or x y z."""
    if not isinstance(text, str):
        return None
    match = TRIPLET_PATTERN.search(text)
    if match:
        try:
            return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        except ValueError:  # pragma: no cover
            return None  # pragma: no cover
    return None


# --- End Helper functions --- #

# Known next section headers that terminate this parser
KNOWN_NEXT_SECTION_STARTS = [
    "SA-NTO Decomposition",  # Spin-Unrestricted or Spin-Restricted
    "Natural Transition Orbital Analysis",  # Alternative NTO block name
    "Spin-Flip NTO analysis",  # Another NTO variant
    "SCF Analysis",  # For cases where TDM is last block before final SCF summary
]


class TransitionDensityMatrixParser(SectionParser):
    """
    Parses the "Transition Density Matrix Analysis" section from Q-Chem output.
    This section typically follows excited state calculations (TDDFT/TDA) and provides
    detailed analysis for each state's transition density matrix, supporting RKS and UKS.
    """

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """
        Checks if the line marks the beginning of the Transition Density Matrix Analysis section.
        """
        return "Transition Density Matrix Analysis" in line and not current_data.parsed_tddft_transition_dm_analysis

    def parse(self, iterator: LineIterator, current_line_header: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting parsing of Transition Density Matrix Analysis section.")

        active_line: str | None
        try:
            # Consume the line after the matched header (current_line_header) and any decorative lines
            active_line = next(iterator)
            while active_line is not None and ("----" in active_line or not active_line.strip()):
                active_line = next(iterator)
        except StopIteration:  # pragma: no cover
            logger.warning(
                "EOF reached unexpectedly after Transition Density Matrix Analysis header."
            )  # pragma: no cover
            results.parsed_tddft_transition_dm_analysis = True  # pragma: no cover
            return  # pragma: no cover

        request_break_main_loop = False
        if active_line is None:  # Check if EOF was hit during initial fluff consumption
            request_break_main_loop = True

        while not request_break_main_loop and active_line is not None:
            line_to_process_this_iteration: str
            if results.buffered_line:
                line_to_process_this_iteration = results.buffered_line
                results.buffered_line = None
            else:
                line_to_process_this_iteration = active_line

            if any(known_start in line_to_process_this_iteration for known_start in KNOWN_NEXT_SECTION_STARTS):
                logger.debug(f"TDM parsing stopped by next section: {line_to_process_this_iteration.strip()}")
                results.buffered_line = line_to_process_this_iteration
                request_break_main_loop = True
                continue

            # Updated regex for state header to include "Excited State"
            state_match = re.match(
                r"^\s*(Singlet|Triplet|Excited State)\s+(\d+)\s*:\s*$", line_to_process_this_iteration.strip()
            )
            if state_match:
                multiplicity_label, state_number_str = state_match.groups()
                state_number = int(state_number_str)
                logger.debug(f"Parsing TDM for {multiplicity_label} state {state_number}")

                mulliken_data: TransitionDMMulliken | None = None
                ct_numbers_data: TransitionDMCTNumbers | None = None
                exciton_analysis_data: ExcitonAnalysisTMData | None = None  # Updated type

                line_within_state: str | None
                try:
                    line_within_state = next(iterator)
                except StopIteration:  # pragma: no cover
                    logger.debug(
                        f"EOF after {multiplicity_label} state {state_number} header during TDM parsing."
                    )  # pragma: no cover
                    request_break_main_loop = True  # pragma: no cover
                    active_line = None  # pragma: no cover
                    continue  # pragma: no cover

                while line_within_state is not None:
                    current_line_in_state_stripped = line_within_state.strip()

                    if not current_line_in_state_stripped or current_line_in_state_stripped.startswith("----"):
                        try:
                            line_within_state = next(iterator)
                        except StopIteration:
                            line_within_state = None
                        continue

                    is_new_state_header = re.match(
                        r"^\s*(Singlet|Triplet|Excited State)\s+(\d+)\s*:\s*$", current_line_in_state_stripped
                    )
                    is_next_major_tdm_section = any(
                        known_start in line_within_state for known_start in KNOWN_NEXT_SECTION_STARTS
                    )

                    if is_new_state_header or is_next_major_tdm_section:
                        results.buffered_line = line_within_state
                        line_within_state = None
                    else:
                        if "Mulliken Population Analysis (Transition DM)" in line_within_state:
                            mulliken_data, line_within_state = self._parse_mulliken_tdm(
                                iterator, line_within_state, results
                            )
                        elif "CT numbers (Mulliken)" in line_within_state:
                            ct_numbers_data, line_within_state = self._parse_ct_numbers(
                                iterator, line_within_state, results
                            )
                        elif "Exciton analysis of the transition density matrix" in line_within_state:
                            exciton_analysis_data, line_within_state = self._parse_exciton_analysis_tdm(
                                iterator, line_within_state, results
                            )
                        else:
                            logger.debug(
                                f"Unparsed line '{current_line_in_state_stripped}' in TDM state {state_number}, assuming end of state."
                            )  # pragma: no cover
                            results.buffered_line = line_within_state  # pragma: no cover
                            line_within_state = None  # pragma: no cover

                if mulliken_data or ct_numbers_data or exciton_analysis_data:
                    analysis = TransitionDensityMatrixDetailedAnalysis(
                        state_number=state_number,
                        multiplicity=multiplicity_label,  # Use the label from regex
                        mulliken=mulliken_data,
                        ct_numbers=ct_numbers_data,
                        exciton_analysis=exciton_analysis_data,
                    )
                    results.transition_density_matrix_detailed_analyses_list.append(analysis)
                    logger.debug(f"Stored TDM Analysis for {multiplicity_label} state {state_number}")

                if line_within_state is None and not results.buffered_line:
                    request_break_main_loop = True
                    active_line = None
            else:
                logger.debug(
                    f"TDM parsing stopped by unrecognized line: {line_to_process_this_iteration.strip()} or end of section."
                )  # pragma: no cover
                results.buffered_line = line_to_process_this_iteration  # pragma: no cover
                request_break_main_loop = True  # pragma: no cover
                continue

            if not request_break_main_loop and not results.buffered_line:
                try:
                    active_line = next(iterator)
                except StopIteration:  # pragma: no cover
                    logger.debug("EOF reached at the end of TDM main parsing loop.")  # pragma: no cover
                    active_line = None  # pragma: no cover
                    request_break_main_loop = True  # pragma: no cover

        results.parsed_tddft_transition_dm_analysis = True
        count = len(results.transition_density_matrix_detailed_analyses_list)
        if count > 0:
            logger.info(f"Finished parsing TDM Analysis. Found {count} states.")
        elif not results.buffered_line or any(
            known_start in results.buffered_line for known_start in KNOWN_NEXT_SECTION_STARTS
        ):
            # Only warn if we didn't stop due to finding next section or a legitimate unrecognized line
            pass  # Warning was too noisy, allow empty sections.
        # else:
        #     logger.warning("Parsed TDM Analysis section but found no detailed state analyses and not cleanly terminated.")

    def _parse_mulliken_tdm(
        self, iterator: LineIterator, current_block_line: str, results: _MutableCalculationData
    ) -> tuple[TransitionDMMulliken | None, str | None]:
        populations: list[TransitionDMAtomPopulation] = []
        sum_qta: float | None = None
        sum_qt2: float | None = None
        is_uks_format = False  # Flag to determine table format

        current_parse_line: str | None
        try:
            current_parse_line = next(iterator)  # Line after "Mulliken..." header
        except StopIteration:  # pragma: no cover
            logger.warning("EOF immediately after Mulliken TDM header.")  # pragma: no cover
            return None, None  # pragma: no cover

        try:
            # Determine table format (RKS vs UKS) by inspecting the header
            header_line_found = False
            while current_parse_line is not None:
                line_strip = current_parse_line.strip()
                if not line_strip or line_strip.startswith("----"):  # Skip empty/decorator
                    current_parse_line = next(iterator)
                    continue
                # This is the actual header line
                if "h+ (alpha)" in line_strip and "e- (alpha)" in line_strip:
                    is_uks_format = True
                elif "h+" in line_strip and "e-" in line_strip:  # RKS check
                    is_uks_format = False
                else:  # Should not happen if header is standard
                    logger.warning(f"Unrecognized Mulliken TDM table header: {line_strip}")  # pragma: no cover
                    # Default to RKS or try to proceed, but this is risky
                header_line_found = True
                current_parse_line = next(iterator)  # Consume header line
                break  # Header processed

            if not header_line_found and current_parse_line is None:  # EOF before header
                logger.warning("EOF before Mulliken TDM table header could be identified.")  # pragma: no cover
                return None, None  # pragma: no cover

            # Skip table decorator "----" if present after header
            while current_parse_line is not None and (
                current_parse_line.strip().startswith("----") or not current_parse_line.strip()
            ):
                current_parse_line = next(iterator)

            # Parse atom lines
            while current_parse_line is not None and not current_parse_line.strip().startswith("----"):
                if current_parse_line.strip().startswith("Sum:"):  # Stop if Sum: is encountered
                    break
                parts = current_parse_line.split()
                try:
                    atom_index = int(parts[0]) - 1
                    symbol = parts[1]
                    transition_charge_e_val = safe_float(parts[2])
                    if transition_charge_e_val is None:
                        logger.warning(
                            f"Skipping Mulliken TDM atom line due to unparsable transition_charge_e: '{current_parse_line.strip()}'"
                        )
                        # The main loop will advance current_parse_line via next(iterator)
                        continue  # Skip this atom entry

                    if is_uks_format:
                        if len(parts) >= 7:
                            populations.append(
                                TransitionDMAtomPopulation(
                                    atom_index=atom_index,
                                    symbol=symbol,
                                    transition_charge_e=transition_charge_e_val,  # Now a confirmed float
                                    hole_charge_alpha_uks=safe_float(parts[3]),
                                    hole_charge_beta_uks=safe_float(parts[4]),
                                    electron_charge_alpha_uks=safe_float(parts[5]),
                                    electron_charge_beta_uks=safe_float(parts[6]),
                                )
                            )
                        else:
                            logger.warning(
                                f"UKS Mulliken TDM line has too few parts: '{current_parse_line.strip()}'"
                            )  # pragma: no cover
                    else:  # RKS format
                        if len(parts) >= 5:  # atom_idx, symbol, trans_charge, h+, e- are mandatory
                            populations.append(
                                TransitionDMAtomPopulation(
                                    atom_index=atom_index,
                                    symbol=symbol,
                                    transition_charge_e=transition_charge_e_val,  # Now a confirmed float
                                    hole_charge_rks=safe_float(parts[3]),
                                    electron_charge_rks=safe_float(parts[4]),
                                    delta_charge_rks=safe_float(parts[5]) if len(parts) > 5 else None,
                                )
                            )
                        else:
                            logger.warning(
                                f"RKS Mulliken TDM line has too few parts: '{current_parse_line.strip()}'"
                            )  # pragma: no cover
                except (ValueError, IndexError) as e:  # pragma: no cover
                    logger.warning(
                        f"Could not parse Mulliken TDM atom line: '{current_parse_line.strip()}'. Error: {e}"
                    )  # pragma: no cover
                current_parse_line = next(iterator)

            # ... (rest of QTa, QT2 parsing remains similar, ensure current_parse_line is advanced correctly)
            # At this point, current_parse_line is expected to be the "----" separator after the atom table,
            # or the "Sum:" line if the separator was missing, or None if EOF.

            # Consume the "----" separator line if it's the current line
            if current_parse_line is not None and current_parse_line.strip().startswith("----"):
                current_parse_line = next(iterator)  # Advance to the next line (should be "Sum: ...")

            # Consume the "Sum: ..." line if it's the current line
            if current_parse_line is not None and current_parse_line.strip().startswith("Sum:"):
                current_parse_line = next(iterator)  # Advance to the next line (e.g., blank line or QTa line)

            while current_parse_line is not None and (
                not current_parse_line.strip() or current_parse_line.strip().startswith("----")
            ):
                current_parse_line = next(iterator)

            if current_parse_line and "Sum of absolute trans. charges, QTa" in current_parse_line:
                match_qta = re.search(r"QTa\s*=\s*(-?\d+\.\d+)", current_parse_line)
                if match_qta:
                    sum_qta = safe_float(match_qta.group(1))
                current_parse_line = next(iterator)

            while current_parse_line is not None and (
                not current_parse_line.strip() or current_parse_line.strip().startswith("----")
            ):
                current_parse_line = next(iterator)

            if current_parse_line and "Sum of squared  trans. charges, QT2" in current_parse_line:
                match_qt2 = re.search(r"QT2\s*=\s*(-?\d+\.\d+)", current_parse_line)
                if match_qt2:
                    sum_qt2 = safe_float(match_qt2.group(1))
                current_parse_line = next(iterator)

            mulliken_obj = TransitionDMMulliken(populations, sum_qta, sum_qt2)
            return mulliken_obj, current_parse_line

        except StopIteration:  # pragma: no cover
            mulliken_obj = TransitionDMMulliken(populations, sum_qta, sum_qt2)  # pragma: no cover
            return (
                mulliken_obj if populations or sum_qta is not None or sum_qt2 is not None else None
            ), None  # pragma: no cover
        except Exception as e:  # pragma: no cover
            logger.error(
                f"Error in _parse_mulliken_tdm. Last line: '{current_parse_line}'. Error: {e}", exc_info=True
            )  # pragma: no cover
            return (
                TransitionDMMulliken(populations, sum_qta, sum_qt2) if populations else None
            ), current_parse_line  # pragma: no cover

    def _parse_ct_numbers(
        self, iterator: LineIterator, current_block_line: str, results: _MutableCalculationData
    ) -> tuple[TransitionDMCTNumbers | None, str | None]:
        ct_data: dict[str, Any] = {}
        found_any_data = False

        # Regex to capture main value and optional (alpha: val, beta: val)
        # Key, MainVal, _, AlphaVal, _, BetaVal (simplified for one order of alpha/beta)
        # or Key, MainVal for simple cases
        ct_pattern_spin = re.compile(
            r"^\s*(omega|LOC|LOCa|<Phe>)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"  # Key and Main Value
            r"(?:\s*\(alpha:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"  # Optional Alpha Value
            r",\s*beta:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\))?"  # Optional Beta Value
        )
        ct_pattern_simple = re.compile(r"^\s*(2<alpha\|beta>)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

        current_parse_line: str | None
        try:
            current_parse_line = next(iterator)  # Line after "CT numbers (Mulliken)" header
        except StopIteration:  # pragma: no cover
            logger.warning("EOF immediately after CT numbers header.")  # pragma: no cover
            return None, None  # pragma: no cover

        try:
            while current_parse_line is not None:
                line_strip = current_parse_line.strip()
                if not line_strip:  # Empty line signifies end of CT numbers block
                    try:
                        current_parse_line = next(iterator)
                    except StopIteration:
                        current_parse_line = None
                    break  # End of this block

                match_spin = ct_pattern_spin.match(line_strip)
                match_simple = ct_pattern_simple.match(line_strip)

                if match_spin:
                    key, val_str = match_spin.group(1), match_spin.group(2)
                    alpha_val_str, beta_val_str = match_spin.group(3), match_spin.group(4)

                    ct_data[key.lower()] = safe_float(val_str)
                    found_any_data = True
                    if alpha_val_str and beta_val_str:
                        ct_data[f"{key.lower()}_alpha"] = safe_float(alpha_val_str)
                        ct_data[f"{key.lower()}_beta"] = safe_float(beta_val_str)
                    current_parse_line = next(iterator)
                elif match_simple:
                    key, val_str = match_simple.group(1), match_simple.group(2)
                    # Clean key for dict: 2<alpha|beta> -> two_alpha_beta_overlap
                    clean_key = "two_alpha_beta_overlap" if key == "2<alpha|beta>" else key.lower()
                    ct_data[clean_key] = safe_float(val_str)
                    found_any_data = True
                    current_parse_line = next(iterator)
                else:  # Line doesn't match CT number format, end of this block
                    break

            if not found_any_data:
                return None, current_parse_line

            # Map dict to dataclass
            # Ensure all keys are present even if None, for dataclass constructor
            return TransitionDMCTNumbers(
                omega=ct_data.get("omega"),
                omega_alpha=ct_data.get("omega_alpha"),
                omega_beta=ct_data.get("omega_beta"),
                two_alpha_beta_overlap=ct_data.get("two_alpha_beta_overlap"),
                loc=ct_data.get("loc"),
                loc_alpha=ct_data.get("loc_alpha"),
                loc_beta=ct_data.get("loc_beta"),
                loc_a=ct_data.get("loca"),  # Note: key from regex is "LOCa", dict key "loca"
                loc_a_alpha=ct_data.get("loca_alpha"),
                loc_a_beta=ct_data.get("loca_beta"),
                phe_overlap=ct_data.get("<phe>"),  # Note: key from regex is "<Phe>", dict key "<phe>"
                phe_overlap_alpha=ct_data.get("<phe>_alpha"),
                phe_overlap_beta=ct_data.get("<phe>_beta"),
            ), current_parse_line

        except StopIteration:  # pragma: no cover
            if not found_any_data:  # pragma: no cover
                return None, current_parse_line  # pragma: no cover
            # If found_any_data is true, it implies we successfully parsed some data before hitting EOF.
            # In this case, returning the partially parsed data is valid behavior, not an error to be fully ignored.
            return TransitionDMCTNumbers(
                omega=ct_data.get("omega"),
                omega_alpha=ct_data.get("omega_alpha"),
                omega_beta=ct_data.get("omega_beta"),
                two_alpha_beta_overlap=ct_data.get("two_alpha_beta_overlap"),
                loc=ct_data.get("loc"),
                loc_alpha=ct_data.get("loc_alpha"),
                loc_beta=ct_data.get("loc_beta"),
                loc_a=ct_data.get("loca"),
                loc_a_alpha=ct_data.get("loca_alpha"),
                loc_a_beta=ct_data.get("loca_beta"),
                phe_overlap=ct_data.get("<phe>"),
                phe_overlap_alpha=ct_data.get("<phe>_alpha"),
                phe_overlap_beta=ct_data.get("<phe>_beta"),
            ), None  # Return None as the next line because iterator is exhausted
        except Exception as e:  # pragma: no cover
            logger.error(
                f"Error in _parse_ct_numbers. Line: '{current_parse_line}'. Error: {e}", exc_info=True
            )  # pragma: no cover
            return None, current_parse_line  # pragma: no cover

    def _parse_single_exciton_block(
        self, iterator: LineIterator, initial_block_line: str | None, results: _MutableCalculationData
    ) -> tuple[ExcitonPropertiesSet | None, str | None]:
        """Parses one block of exciton properties (Total, Alpha, or Beta, or a full RKS block)."""
        exciton_data_dict: dict[str, Any] = {}

        current_parse_line = initial_block_line

        if current_parse_line is None:
            return None, None

        # If initial_block_line is a sub-header (Total, Alpha, Beta), consume it
        # and advance current_parse_line to the first actual data line of that sub-block.
        # For RKS, initial_block_line is already the first data line, so this check won't trigger.
        stripped_initial = current_parse_line.strip()
        if (
            stripped_initial.startswith("Total:")
            or stripped_initial.startswith("Alpha spin:")
            or stripped_initial.startswith("Beta spin:")
        ):
            try:
                current_parse_line = next(iterator)  # This is now the first data line of the sub-block
            except StopIteration:  # pragma: no cover
                logger.debug(f"EOF immediately after exciton sub-header: {stripped_initial}")  # pragma: no cover
                return None, None  # pragma: no cover

        # Now current_parse_line is the first actual data line to process, or None if EOF after header.

        try:
            while current_parse_line is not None:
                line_strip = current_parse_line.strip()

                # Determine if the current line is a terminator for the current exciton sub-block.
                is_new_exciton_sub_block = (
                    line_strip.startswith("Total:")
                    or line_strip.startswith("Alpha spin:")
                    or line_strip.startswith("Beta spin:")
                )
                is_other_tdm_analysis = any(
                    sh_key in line_strip for sh_key in ["Mulliken Population Analysis", "CT numbers (Mulliken)"]
                )
                is_new_state_header = re.match(r"^\s*(Singlet|Triplet|Excited State)\s+(\d+)\s*:\s*$", line_strip)
                is_major_section_header = any(known_start in line_strip for known_start in KNOWN_NEXT_SECTION_STARTS)

                if is_new_exciton_sub_block or is_other_tdm_analysis or is_new_state_header or is_major_section_header:
                    break

                if not line_strip:  # Skip blank lines within the current sub-block
                    try:
                        current_parse_line = next(iterator)
                        continue
                    except StopIteration:  # pragma: no cover
                        current_parse_line = None  # pragma: no cover
                        break  # pragma: no cover

                processed_by_helper, line_after_helper = self._process_exciton_line_and_potential_vector(
                    current_parse_line, iterator, exciton_data_dict, results
                )

                if processed_by_helper:
                    current_parse_line = line_after_helper
                else:
                    logger.warning(
                        f"Unrecognized line in exciton data block, stopping parsing for this block: '{current_parse_line.strip()}'"
                    )  # pragma: no cover
                    break  # pragma: no cover

            exciton_data_dict.pop("_expecting_vector_for_key", None)
            props_set = ExcitonPropertiesSet(**exciton_data_dict) if exciton_data_dict else None

            return props_set, current_parse_line

        except StopIteration:  # pragma: no cover
            exciton_data_dict.pop("_expecting_vector_for_key", None)  # pragma: no cover
            return (ExcitonPropertiesSet(**exciton_data_dict) if exciton_data_dict else None), None  # pragma: no cover
        except Exception as e:  # pragma: no cover
            logger.error(
                f"Error creating ExcitonPropertiesSet or during parsing. Data: {exciton_data_dict}. Last line processed: '{current_parse_line}'. Error: {e}",
                exc_info=True,
            )  # pragma: no cover
            return None, current_parse_line  # pragma: no cover

    def _parse_exciton_analysis_tdm(
        self, iterator: LineIterator, current_block_line: str, results: _MutableCalculationData
    ) -> tuple[ExcitonAnalysisTMData | None, str | None]:
        total_props: ExcitonPropertiesSet | None = None
        alpha_props: ExcitonPropertiesSet | None = None
        beta_props: ExcitonPropertiesSet | None = None

        # current_block_line is "Exciton analysis of the transition density matrix"
        # The actual content (or first sub-header like "Total:") starts on the next line.

        line_after_main_header: str | None
        try:
            line_after_main_header = next(iterator)
            # Skip any blank lines immediately after the main "Exciton analysis..." header
            while line_after_main_header is not None and not line_after_main_header.strip():
                line_after_main_header = next(iterator)
        except StopIteration:  # pragma: no cover
            logger.debug(
                "EOF reached after 'Exciton analysis...' header and potential blank lines."
            )  # pragma: no cover
            return None, None  # pragma: no cover

        if line_after_main_header is None:  # EOF after main header and blanks
            logger.debug("Exciton section seems empty after header.")  # pragma: no cover
            return None, None  # pragma: no cover

        # This is the first significant line for exciton analysis.
        # It could be "Total:" (for UKS), or a data line like "Trans. dipole moment..." (for RKS).
        # Explicitly type as str | None, even though line_after_main_header is narrowed to str here.
        current_line_for_block_parsing: str | None = line_after_main_header

        # Try to parse "Total:" block (UKS) or the entire block (RKS)
        if current_line_for_block_parsing is not None and current_line_for_block_parsing.strip().startswith("Total:"):
            logger.debug("Parsing 'Total:' exciton block (UKS assumption).")
            total_props, current_line_for_block_parsing = self._parse_single_exciton_block(
                iterator, current_line_for_block_parsing, results
            )
        elif current_line_for_block_parsing is not None and not (
            current_line_for_block_parsing.strip().startswith("Alpha spin:")
            or current_line_for_block_parsing.strip().startswith("Beta spin:")
        ):
            # If it's not starting with "Alpha" or "Beta" either, assume it's an RKS block
            # (or UKS "Total" without the header, though less likely)
            logger.debug("Parsing as RKS exciton block or UKS 'Total' block (no explicit 'Total:' header).")
            total_props, current_line_for_block_parsing = self._parse_single_exciton_block(
                iterator, current_line_for_block_parsing, results
            )
        # If it was an RKS block, current_line_for_block_parsing will be the terminator of that block.
        # If it was a UKS "Total:" block, current_line_for_block_parsing is now the line after "Total:" block (e.g., "Alpha spin:" or other terminator).

        # Check for "Alpha spin:" block
        if current_line_for_block_parsing is not None and current_line_for_block_parsing.strip().startswith(
            "Alpha spin:"
        ):
            logger.debug("Parsing 'Alpha spin:' exciton block.")
            alpha_props, current_line_for_block_parsing = self._parse_single_exciton_block(
                iterator, current_line_for_block_parsing, results
            )

        # Check for "Beta spin:" block
        if current_line_for_block_parsing is not None and current_line_for_block_parsing.strip().startswith(
            "Beta spin:"
        ):
            logger.debug("Parsing 'Beta spin:' exciton block.")
            beta_props, current_line_for_block_parsing = self._parse_single_exciton_block(
                iterator, current_line_for_block_parsing, results
            )

        # After attempting to parse all potential blocks, current_line_for_block_parsing
        # holds the line that terminated the last successfully parsed block, or the line
        # that was not a recognized exciton block header. This line should be returned.

        if total_props or alpha_props or beta_props:
            exciton_data = ExcitonAnalysisTMData(
                total_properties=total_props, alpha_spin_properties=alpha_props, beta_spin_properties=beta_props
            )
            return exciton_data, current_line_for_block_parsing

        # No properties parsed, but return the line that was being considered.
        return None, current_line_for_block_parsing

    # _process_exciton_line_and_potential_vector remains largely the same,
    # as it populates a given dictionary. Ensure it handles StopIteration from `next(iterator)` gracefully.
    def _process_exciton_line_and_potential_vector(
        self,
        current_line: str,
        iterator: LineIterator,
        data: dict[str, Any],  # This dict is for one ExcitonPropertiesSet
        results: _MutableCalculationData,
    ) -> tuple[bool, str | None]:  # Returns (was_processed, next_line_to_process_by_caller)
        l_s = current_line.strip()

        # Check if this line is a known terminator for the *current exciton sub-block*
        # or a header for a *new exciton sub-block*. If so, do not process.
        if (
            l_s.startswith("Total:")
            or l_s.startswith("Alpha spin:")
            or l_s.startswith("Beta spin:")
            or any(sh_key in l_s for sh_key in ["Mulliken Population Analysis", "CT numbers (Mulliken)"])
            or any(known_start in l_s for known_start in KNOWN_NEXT_SECTION_STARTS)
            or re.match(r"^\s*(Singlet|Triplet|Excited State)\s+(\d+)\s*:\s*$", l_s)
        ):
            return False, current_line  # Not processed, caller should handle this line

        scalar_patterns = {
            r"Trans\. dipole moment \[D\]:": "total_transition_dipole_moment",
            r"Transition <r\^2> \[a\.u\.\]:": "transition_r_squared_au",
            r"\|<r_e - r_h>\| \[Ang\]:": "hole_electron_distance_ang",
            r"Covariance\(r_h, r_e\) \[Ang\^2\]:": "covariance_rh_re_ang_sq",
            r"Correlation coefficient:": "correlation_coefficient",
        }
        vector_same_line_patterns = {
            r"Cartesian components \[D\]:": "transition_dipole_moment_components",
            r"Cartesian components \[a\.u\.\]:": "transition_r_squared_au_components",
            r"<r_h> \[Ang\]:": "hole_position_ang",
            r"<r_e> \[Ang\]:": "electron_position_ang",
        }
        scalar_then_vector_patterns = {
            r"Hole size \[Ang\]:": ("hole_size_ang", "hole_size_ang_components"),
            r"Electron size \[Ang\]:": ("electron_size_ang", "electron_size_ang_components"),
            r"RMS electron-hole separation \[Ang\]:": (
                "rms_electron_hole_separation_ang",
                "rms_electron_hole_separation_ang_components",
            ),
            r"Center-of-mass size \[Ang\]:": ("center_of_mass_size_ang", "center_of_mass_size_ang_components"),
        }

        for pattern_str, (s_key, v_key) in scalar_then_vector_patterns.items():
            match = re.search(pattern_str, l_s)
            if match:
                val_str = l_s[match.end() :].strip()
                s_val = safe_float(val_str)
                if s_val is not None:
                    data[s_key] = s_val

                try:
                    # next() on Iterator[str] returns str or raises StopIteration.
                    line_for_vector_components: str = next(iterator)

                    # line_for_vector_components is now guaranteed to be str here.
                    nl_s: str = line_for_vector_components.strip()
                    prefix_match = re.match(r"^Cartesian components\s*\[[^\]]+\]:\s*", nl_s)
                    vector_parse_target: str = nl_s[prefix_match.end() :].strip() if prefix_match else nl_s

                    triplet = extract_float_triplet(vector_parse_target)
                    if triplet:
                        data[v_key] = triplet
                    else:
                        logger.warning(
                            f"Failed to parse vector '{v_key}' from line '{line_for_vector_components}' (target: '{vector_parse_target}')"
                        )  # pragma: no cover

                    # Try to get the line *after* the vector line for the caller.
                    try:
                        # This next(iterator) either returns str or raises StopIteration
                        return True, next(iterator)
                    except StopIteration:  # pragma: no cover
                        return True, None  # pragma: no cover

                except (
                    StopIteration  # pragma: no cover
                ):  # This catches StopIteration for `next(iterator)` fetching `line_for_vector_components`
                    logger.warning(
                        f"EOF after scalar '{s_key}', expected vector components for '{v_key}'."
                    )  # pragma: no cover
                    return True, None  # pragma: no cover

        for pattern_str, key in scalar_patterns.items():
            match = re.search(pattern_str, l_s)
            if match:
                val_str = l_s[match.end() :].strip()
                val = safe_float(val_str)
                if val is not None:
                    data[key] = val
                try:
                    # next(iterator) returns str here, or StopIteration is caught
                    return True, next(iterator)
                except StopIteration:  # pragma: no cover
                    return True, None  # pragma: no cover

        for pattern_str, key in vector_same_line_patterns.items():
            match = re.search(pattern_str, l_s)
            if match:
                val_part = l_s[match.end() :].strip()
                triplet = extract_float_triplet(val_part)
                if triplet:
                    data[key] = triplet
                else:
                    logger.warning(f"Could not parse same-line vector '{key}' from '{val_part}'")  # pragma: no cover
                try:
                    # next(iterator) returns str here, or StopIteration is caught
                    return True, next(iterator)
                except StopIteration:  # pragma: no cover
                    return True, None  # pragma: no cover

        return False, current_line

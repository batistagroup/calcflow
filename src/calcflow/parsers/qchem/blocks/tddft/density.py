import re
from typing import Any

from calcflow.parsers.qchem.typing import (
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

# --- Helper functions --- #


def safe_float(value: str | None) -> float | None:
    """Attempt to convert a string to a float, returning None on ValueError."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


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
        except ValueError:
            return None
    return None


# --- End Helper functions --- #

# Known next section headers that terminate this parser
KNOWN_NEXT_SECTION_STARTS = [
    "SA-NTO Decomposition",  # Spin-Unrestricted or Spin-Restricted
    "Natural Transition Orbital Analysis",  # Alternative NTO block name
    "Spin-Flip NTO analysis",  # Another NTO variant
]


class TransitionDensityMatrixParser(SectionParser):
    """
    Parses the "Transition Density Matrix Analysis" section from Q-Chem output.
    This section typically follows excited state calculations (TDDFT/TDA) and provides
    detailed analysis for each state's transition density matrix.
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
        except StopIteration:
            logger.warning("EOF reached unexpectedly after Transition Density Matrix Analysis header.")
            results.parsed_tddft_transition_dm_analysis = True
            return

        request_break_main_loop = False
        if active_line is None:  # Check if EOF was hit during initial fluff consumption
            request_break_main_loop = True

        while not request_break_main_loop and active_line is not None:
            # active_line is guaranteed to be str here by the loop condition
            line_to_process_this_iteration: str
            if results.buffered_line:
                line_to_process_this_iteration = results.buffered_line
                results.buffered_line = None
            else:
                line_to_process_this_iteration = active_line  # type: ignore[assignment] # Mypy struggles here, but logic is sound

            if any(known_start in line_to_process_this_iteration for known_start in KNOWN_NEXT_SECTION_STARTS):
                logger.debug(f"TDM parsing stopped by next section: {line_to_process_this_iteration.strip()}")
                results.buffered_line = line_to_process_this_iteration
                request_break_main_loop = True
                continue  # Check loop condition again

            state_match = re.match(r"^\s*(Singlet|Triplet)\s+(\d+)\s*:\s*$", line_to_process_this_iteration.strip())
            if state_match:
                multiplicity, state_number_str = state_match.groups()
                state_number = int(state_number_str)
                logger.debug(f"Parsing TDM for {multiplicity} state {state_number}")

                mulliken_data: TransitionDMMulliken | None = None
                ct_numbers_data: TransitionDMCTNumbers | None = None
                exciton_analysis_data: ExcitonAnalysisTransitionDM | None = None

                line_within_state: str | None
                try:
                    # First line of state's content or fluff after "Singlet X:" header
                    line_within_state = next(iterator)
                except StopIteration:
                    logger.debug(f"EOF after {multiplicity} state {state_number} header during TDM parsing.")
                    request_break_main_loop = True
                    active_line = None  # Ensure outer loop terminates
                    continue  # Check loop condition

                # Inner loop for parsing sub-sections of the current state
                while line_within_state is not None:
                    current_line_in_state_stripped = line_within_state.strip()

                    if not current_line_in_state_stripped or current_line_in_state_stripped.startswith("----"):
                        try:
                            line_within_state = next(iterator)
                        except StopIteration:
                            line_within_state = None  # EOF
                        continue  # Next iteration of inner loop

                    is_new_state_header = re.match(
                        r"^\s*(Singlet|Triplet)\s+(\d+)\s*:\s*$", current_line_in_state_stripped
                    )
                    is_next_major_tdm_section = any(
                        known_start in line_within_state for known_start in KNOWN_NEXT_SECTION_STARTS
                    )

                    if is_new_state_header or is_next_major_tdm_section:
                        results.buffered_line = line_within_state  # Buffer for main loop processing
                        line_within_state = None  # Signal to end this state's inner loop
                        # Do not break here, let sub-parser calls be skipped by line_within_state being None
                    else:
                        # Call sub-parsers
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
                            )
                            results.buffered_line = line_within_state  # Buffer this unrecognized line
                            line_within_state = None  # Signal to end this state's inner loop
                            # Do not break here, let loop condition handle it

                # After processing a state's inner loop (exited because line_within_state is None)
                if mulliken_data or ct_numbers_data or exciton_analysis_data:
                    analysis = TransitionDensityMatrixDetailedAnalysis(
                        state_number=state_number,
                        multiplicity=multiplicity,
                        mulliken_analysis=mulliken_data,
                        ct_numbers=ct_numbers_data,
                        exciton_analysis=exciton_analysis_data,
                    )
                    results.transition_density_matrix_detailed_analyses_list.append(analysis)
                    logger.debug(f"Stored TDM Analysis for {multiplicity} state {state_number}")

                # If a sub-parser hit EOF, line_within_state is None.
                # If a new state/section was buffered, results.buffered_line is set.
                # The main loop's active_line update at the end will handle fetching the next line if needed.
                if line_within_state is None and not results.buffered_line:  # EOF from sub-parser, nothing buffered
                    request_break_main_loop = True
                    active_line = None  # Ensure outer loop terminates

                # No specific update to active_line here; main loop end or buffered_line handles it.

            else:  # Not a state_match, not a known_next_section
                logger.debug(f"TDM parsing stopped by unrecognized line: {line_to_process_this_iteration.strip()}")
                results.buffered_line = line_to_process_this_iteration
                request_break_main_loop = True
                # continue not strictly needed due to flag, but harmless
                continue

            # Fetch next line for the main loop if not breaking and no new line was buffered
            if not request_break_main_loop and not results.buffered_line:
                try:
                    active_line = next(iterator)
                except StopIteration:
                    logger.debug("EOF reached at the end of TDM main parsing loop.")
                    active_line = None  # Ensures loop condition `active_line is not None` fails
                    request_break_main_loop = True  # Explicitly signal break

        # End of main while loop
        results.parsed_tddft_transition_dm_analysis = True
        count = len(results.transition_density_matrix_detailed_analyses_list)
        if count > 0:
            logger.info(f"Finished parsing TDM Analysis. Found {count} states.")
        elif not results.buffered_line:  # Only warn if we didn't stop due to finding next section
            logger.warning("Parsed TDM Analysis section but found no detailed state analyses.")

    def _parse_mulliken_tdm(
        self, iterator: LineIterator, current_block_line: str, results: _MutableCalculationData
    ) -> tuple[TransitionDMMulliken | None, str | None]:
        populations: list[TransitionDMAtomPopulation] = []
        sum_qta: float | None = None
        sum_qt2: float | None = None

        current_parse_line: str | None
        try:
            current_parse_line = next(iterator)  # Line after the "Mulliken Population Analysis (Transition DM)" header
        except StopIteration:
            logger.warning("EOF immediately after Mulliken TDM header.")
            return None, None

        try:
            # Skip table headers and any initial empty lines
            while current_parse_line is not None and (
                current_parse_line.strip().startswith("Atom      Trans. (e)")
                or current_parse_line.strip().startswith("----")
                or not current_parse_line.strip()
            ):
                current_parse_line = next(iterator)

            # Parse atom lines: loop until a "----" separator line
            while current_parse_line is not None and not current_parse_line.strip().startswith("----"):
                # Also stop if "Sum:" is encountered prematurely (e.g., if "----" is missing)
                if current_parse_line.strip().startswith("Sum:"):
                    break
                parts = current_parse_line.split()
                if len(parts) >= 5:
                    try:
                        populations.append(
                            TransitionDMAtomPopulation(
                                atom_index=int(parts[0]) - 1,
                                symbol=parts[1],
                                transition_charge_e=float(parts[2]),
                                hole_charge=float(parts[3]),
                                electron_charge=float(parts[4]),
                                delta_charge=float(parts[5]) if len(parts) > 5 else None,
                            )
                        )
                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Could not parse Mulliken TDM atom line: '{current_parse_line.strip()}'. Error: {e}"
                        )
                current_parse_line = next(iterator)

            # At this point, current_parse_line is expected to be the "----" separator after the atom table,
            # or the "Sum:" line if the separator was missing, or None if EOF.

            # Consume the "----" separator line if it's the current line
            if current_parse_line is not None and current_parse_line.strip().startswith("----"):
                current_parse_line = next(iterator)  # Advance to the next line (should be "Sum: ...")

            # Consume the "Sum: ..." line if it's the current line
            if current_parse_line is not None and current_parse_line.strip().startswith("Sum:"):
                current_parse_line = next(iterator)  # Advance to the next line (e.g., blank line or QTa line)

            # There might be another "----" after sum in some formats, or just blank lines.
            # Let's be flexible and skip any blank lines or separators before QTa.
            while current_parse_line is not None and (
                not current_parse_line.strip() or current_parse_line.strip().startswith("----")
            ):
                current_parse_line = next(iterator)

            # Parse QTa
            if current_parse_line and "Sum of absolute trans. charges, QTa" in current_parse_line:
                match_qta = re.search(r"QTa\s*=\s*(-?\d+\.\d+)", current_parse_line)
                if match_qta:
                    sum_qta = safe_float(match_qta.group(1))
                else:
                    logger.warning(f"Mulliken QTa regex failed for line: '{current_parse_line.strip()}'")
                current_parse_line = next(iterator)

            # Skip any blank lines or separators before QT2
            while current_parse_line is not None and (
                not current_parse_line.strip() or current_parse_line.strip().startswith("----")
            ):
                current_parse_line = next(iterator)

            # Parse QT2
            if current_parse_line and "Sum of squared  trans. charges, QT2" in current_parse_line:
                match_qt2 = re.search(r"QT2\s*=\s*(-?\d+\.\d+)", current_parse_line)
                if match_qt2:
                    sum_qt2 = safe_float(match_qt2.group(1))
                else:
                    logger.warning(f"Mulliken QT2 regex failed for line: '{current_parse_line.strip()}'")
                current_parse_line = next(iterator)

            mulliken_obj = TransitionDMMulliken(populations, sum_qta, sum_qt2)
            return mulliken_obj, current_parse_line

        except StopIteration:
            # If EOF is hit at any point during parsing of table or QTa/QT2
            mulliken_obj = TransitionDMMulliken(populations, sum_qta, sum_qt2)
            # Return data if any was collected, otherwise None for the object part
            return mulliken_obj if populations or sum_qta is not None or sum_qt2 is not None else None, None
        except Exception as e:
            logger.error(f"Error in _parse_mulliken_tdm. Line: '{current_parse_line}'. Error: {e}", exc_info=True)
            # Attempt to return any partially collected data along with the problematic line
            return TransitionDMMulliken(populations, sum_qta, sum_qt2) if populations else None, current_parse_line

    def _parse_ct_numbers(
        self, iterator: LineIterator, current_block_line: str, results: _MutableCalculationData
    ) -> tuple[TransitionDMCTNumbers | None, str | None]:
        omega_val: float | None = None
        two_alpha_beta_overlap_val: float | None = None
        loc_val: float | None = None
        loc_a_val: float | None = None
        phe_overlap_val: float | None = None
        found_any_data = False

        current_parse_line: str | None
        try:
            current_parse_line = next(iterator)  # Line after "CT numbers (Mulliken)" header
        except StopIteration:
            logger.warning("EOF immediately after CT numbers header.")
            return None, None

        try:
            while current_parse_line is not None:
                line_strip = current_parse_line.strip()
                if not line_strip:  # Empty line signifies end of CT numbers block
                    try:
                        current_parse_line = next(iterator)
                    except StopIteration:
                        current_parse_line = None  # EOF after empty line
                    break

                match = re.match(r"^\s*(omega|2<alpha\|beta>|LOC|LOCa|<Phe>)\s*=\s*(-?\d+\.\d+)", line_strip)
                if match:
                    key, val_str = match.group(1), match.group(2)
                    val = safe_float(val_str)
                    if val is None:
                        logger.warning(f"Could not convert CT number value '{val_str}' to float for key '{key}'")
                    else:
                        found_any_data = True
                        if key == "omega":
                            omega_val = val
                        elif key == "2<alpha|beta>":
                            two_alpha_beta_overlap_val = val
                        elif key == "LOC":
                            loc_val = val
                        elif key == "LOCa":
                            loc_a_val = val
                        elif key == "<Phe>":
                            phe_overlap_val = val
                        else:
                            # Should not happen with the regex pattern, but good for safety
                            logger.warning(f"Unmapped CT key: {key}")

                    current_parse_line = next(iterator)
                else:  # Line doesn't match CT number format, end of this block
                    break

            if not found_any_data:
                return None, current_parse_line

            ct_numbers_obj = TransitionDMCTNumbers(
                omega=omega_val,
                two_alpha_beta_overlap=two_alpha_beta_overlap_val,
                loc=loc_val,
                loc_a=loc_a_val,
                phe_overlap=phe_overlap_val,
            )
            return ct_numbers_obj, current_parse_line
        except StopIteration:
            if not found_any_data:
                return None, None
            ct_numbers_obj_eof = TransitionDMCTNumbers(
                omega=omega_val,
                two_alpha_beta_overlap=two_alpha_beta_overlap_val,
                loc=loc_val,
                loc_a=loc_a_val,
                phe_overlap=phe_overlap_val,
            )
            return ct_numbers_obj_eof, None
        except Exception as e:  # Catches the __init__ error if keys are still wrong, or other issues
            logger.error(f"Error in _parse_ct_numbers. Line: '{current_parse_line}'. Error: {e}", exc_info=True)
            return None, current_parse_line  # Return problematic line

    def _parse_exciton_analysis_tdm(
        self, iterator: LineIterator, current_block_line: str, results: _MutableCalculationData
    ) -> tuple[ExcitonAnalysisTransitionDM | None, str | None]:
        exciton_data_dict: dict[str, Any] = {}
        current_parse_line: str | None
        try:
            current_parse_line = next(iterator)  # Line after "Exciton analysis..." header
        except StopIteration:
            logger.warning("EOF immediately after Exciton analysis header.")
            return None, None

        try:
            while current_parse_line is not None:
                line_strip = current_parse_line.strip()

                if not line_strip:  # Empty line might end this block or be skippable fluff
                    try:
                        next_line_peek = next(iterator)
                        # If the line after empty is relevant to other TDM sub-parsers, or new state/section, then empty line was a separator
                        if (
                            any(sh_key in next_line_peek for sh_key in ["Mulliken Population", "CT numbers"])
                            or any(known_start in next_line_peek for known_start in KNOWN_NEXT_SECTION_STARTS)
                            or re.match(r"^\s*(Singlet|Triplet)\s+(\d+)\s*:\s*$", next_line_peek.strip())
                        ):
                            results.buffered_line = next_line_peek
                            current_parse_line = None  # End this sub-parser, main loop will use buffered line
                            break
                        else:  # Empty line is part of exciton or just fluff, continue with next_line_peek
                            current_parse_line = next_line_peek
                            continue  # Loop will re-evaluate current_parse_line (which is now next_line_peek)
                    except StopIteration:  # Empty line was the last thing in the file
                        current_parse_line = None  # Signal EOF
                        break  # Break from exciton parsing loop

                # Pass the iterator so helper can advance if it consumes multiple lines internally
                # THIS IS THE KEY CHANGE AREA
                processed_by_helper, line_after_helper = self._process_exciton_line_and_potential_vector(
                    current_parse_line, iterator, exciton_data_dict, results
                )

                if processed_by_helper:
                    current_parse_line = line_after_helper  # Helper determined the next line to process
                else:
                    # Line not processed by helper: it's start of new TDM sub-section, new state, or major section.
                    # This sub-parser (exciton) is done with it. current_parse_line is the offending line.
                    break

            # Clean up any lingering expectation key, though ideally it should be cleared by the helper.
            exciton_data_dict.pop("_expecting_vector_for_key", None)
            exciton_obj = ExcitonAnalysisTransitionDM(**exciton_data_dict) if exciton_data_dict else None
            return exciton_obj, current_parse_line

        except StopIteration:  # Should be handled by helper if it consumes last line
            exciton_data_dict.pop("_expecting_vector_for_key", None)
            return ExcitonAnalysisTransitionDM(**exciton_data_dict) if exciton_data_dict else None, None
        except Exception as e:
            logger.error(
                f"Error in _parse_exciton_analysis_tdm. Line: '{current_parse_line}'. Error: {e}", exc_info=True
            )
            return None, current_parse_line

    # Renamed and refactored helper
    def _process_exciton_line_and_potential_vector(
        self,
        current_line: str,
        iterator: LineIterator,
        data: dict[str, Any],
        results: _MutableCalculationData,  # For logging potential parse errors directly
    ) -> tuple[bool, str | None]:  # Returns (was_processed, next_line_to_process_by_caller)
        l_s = current_line.strip()

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

        # Priority 1: Scalar that expects a vector on the NEXT line
        for pattern_str, (s_key, v_key) in scalar_then_vector_patterns.items():
            match = re.search(pattern_str, l_s)
            if match:
                val_str = l_s[match.end() :].strip()
                s_val = safe_float(val_str)
                if s_val is not None:
                    data[s_key] = s_val
                else:
                    logger.warning(f"Could not parse scalar '{s_key}' from '{val_str}' (expecting vector next)")

                # Now, immediately try to parse the next line for the vector components
                next_line_for_vector: str | None = None
                try:
                    next_line_for_vector = next(iterator)
                    if next_line_for_vector is None:  # Should be StopIteration, but defensive
                        logger.warning(f"EOF after scalar '{s_key}', expected vector '{v_key}'.")
                        return True, None  # Processed scalar, but EOF before vector

                    nl_s = next_line_for_vector.strip()
                    prefix_match = re.match(r"^Cartesian components \[.+?\]:\s*", nl_s)
                    vector_parse_target = nl_s
                    if prefix_match:
                        vector_parse_target = nl_s[prefix_match.end() :].strip()

                    triplet = extract_float_triplet(vector_parse_target)
                    if triplet:
                        data[v_key] = triplet
                        logger.debug(f"Parsed scalar '{s_key}' and its vector '{v_key}': {triplet}")
                    else:
                        logger.warning(
                            f"Failed to parse vector '{v_key}' from line '{nl_s}' (target: '{vector_parse_target}')"
                        )
                    # Return True (processed current line + next line), and the line *after* the vector line
                    try:
                        return True, next(iterator)
                    except StopIteration:
                        return True, None  # EOF after vector line

                except StopIteration:  # For next(iterator) to get next_line_for_vector
                    logger.warning(f"EOF after scalar '{s_key}', expected vector '{v_key}'.")
                    return True, None  # Processed the scalar line, hit EOF
                except Exception as e:
                    logger.error(
                        f"Error processing vector for {s_key} on line '{next_line_for_vector}': {e}", exc_info=True
                    )
                    # Scalar was processed, but vector failed. What line to return?
                    # Safest to return the problematic vector line for the caller to decide/break.
                    return True, next_line_for_vector

        # Priority 2: Scalar-only patterns
        for pattern_str, key in scalar_patterns.items():
            match = re.search(pattern_str, l_s)
            if match:
                val_str = l_s[match.end() :].strip()
                val = safe_float(val_str)
                if val is not None:
                    data[key] = val
                else:
                    logger.warning(f"Could not parse scalar '{key}' from '{val_str}'")
                try:  # Processed this line, return next line from iterator
                    return True, next(iterator)
                except StopIteration:
                    return True, None  # EOF after this line

        # Priority 3: Vector-on-same-line patterns
        for pattern_str, key in vector_same_line_patterns.items():
            match = re.search(pattern_str, l_s)
            if match:
                val_part = l_s[match.end() :].strip()
                triplet = extract_float_triplet(val_part)
                if triplet:
                    data[key] = triplet
                else:
                    logger.warning(f"Could not parse same-line vector '{key}' from '{val_part}'")
                try:  # Processed this line, return next line from iterator
                    return True, next(iterator)
                except StopIteration:
                    return True, None  # EOF after this line

        return False, current_line  # Line not processed by any pattern

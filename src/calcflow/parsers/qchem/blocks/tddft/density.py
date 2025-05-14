import re
from typing import Any, cast

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


def safe_float(value: str) -> float | None:
    """Attempt to convert a string to a float, returning None on ValueError."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


FLOAT_PATTERN = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
TRIPLET_PATTERN = re.compile(
    r"[\[\(]?\s*"
    rf"({FLOAT_PATTERN})[\s,]+"
    rf"({FLOAT_PATTERN})[\s,]+"
    rf"({FLOAT_PATTERN})\s*[\]\)]?"
)


def extract_float_triplet(text: str) -> tuple[float, float, float] | None:
    """Extracts a tuple of three floats from a string. Handles [x,y,z] or x y z."""
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

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """
        Parses the entire Transition Density Matrix Analysis block.
        `current_line` is the line containing "Transition Density Matrix Analysis".
        """
        logger.debug("Starting parsing of Transition Density Matrix Analysis section.")

        try:
            line = next(iterator)  # Consume the line after the matched header (current_line)
            while "----" in line or not line.strip():  # Consume decorative lines
                line = next(iterator)
        except StopIteration:
            logger.warning("EOF reached unexpectedly after Transition Density Matrix Analysis header.")
            results.parsed_tddft_transition_dm_analysis = True
            return

        # Main loop to parse each state's analysis
        while True:
            if results.buffered_line:
                line = results.buffered_line
                results.buffered_line = None
            # else: line is already set from previous iteration or initial consumption

            if any(known_start in line for known_start in KNOWN_NEXT_SECTION_STARTS):
                logger.debug(f"Transition Density Matrix Analysis parsing stopped by next section: {line.strip()}")
                results.buffered_line = line
                break

            state_match = re.match(r"^\s*(Singlet|Triplet)\s+(\d+)\s*:\s*$", line.strip())
            if state_match:
                multiplicity = state_match.group(1)
                state_number = int(state_match.group(2))
                logger.debug(f"Parsing Transition DM for {multiplicity} state {state_number}")

                mulliken_data: TransitionDMMulliken | None = None
                ct_numbers_data: TransitionDMCTNumbers | None = None
                exciton_analysis_data: ExcitonAnalysisTransitionDM | None = None

                # This variable will hold the current line being processed *within* this state's block
                line_for_current_state: str | None = None

                try:
                    # Fetch the first line of content/fluff *after* the "Singlet X :" header
                    line_for_current_state = next(iterator)

                    while line_for_current_state is not None:
                        line_strip = line_for_current_state.strip()

                        # Explicitly skip empty lines or '----' decorative lines first
                        if not line_strip or line_strip.startswith("----"):
                            try:
                                line_for_current_state = next(iterator)
                            except StopIteration:
                                line_for_current_state = None  # EOF
                            continue  # Continue to the next line of the state

                        # Check if this line signals end of current state (i.e., start of new state or major section)
                        if any(
                            known_start in line_for_current_state for known_start in KNOWN_NEXT_SECTION_STARTS
                        ) or re.match(r"^\\s*(Singlet|Triplet)\\s+(\\d+)\\s*:\\s*$", line_strip):
                            results.buffered_line = line_for_current_state  # Push back for outer loop
                            line_for_current_state = None  # Signal to break from this state's parsing
                            break

                        # Now, check for actual content sections
                        if "Mulliken Population Analysis (Transition DM)" in line_for_current_state:
                            mulliken_data, line_for_current_state = self._parse_mulliken_tdm(
                                iterator, line_for_current_state, results
                            )
                        elif "CT numbers (Mulliken)" in line_for_current_state:
                            ct_numbers_data, line_for_current_state = self._parse_ct_numbers(
                                iterator, line_for_current_state, results
                            )
                        elif "Exciton analysis of the transition density matrix" in line_for_current_state:
                            exciton_analysis_data, line_for_current_state = self._parse_exciton_analysis_tdm(
                                iterator, line_for_current_state, results
                            )
                        else:
                            # Unrecognized line within a state's block means this state's data is done
                            logger.debug(
                                f"Unparsed line '{line_strip}' within state {state_number}, assuming end of state's data."
                            )
                            results.buffered_line = line_for_current_state
                            line_for_current_state = None  # Signal to break from this state's parsing
                            break

                        # If a sub-parser hit EOF, line_for_current_state will be None
                        if line_for_current_state is None:
                            break

                except StopIteration:
                    logger.debug(
                        f"EOF reached while parsing {multiplicity} state {state_number} in Transition DM Analysis."
                    )
                    line_for_current_state = None  # Signal EOF

                if mulliken_data or ct_numbers_data or exciton_analysis_data:
                    analysis = TransitionDensityMatrixDetailedAnalysis(
                        state_number=state_number,
                        multiplicity=multiplicity,
                        mulliken_analysis=mulliken_data,
                        ct_numbers=ct_numbers_data,
                        exciton_analysis=exciton_analysis_data,
                    )
                    results.transition_density_matrix_detailed_analyses_list.append(analysis)
                    logger.debug(f"Stored Transition DM Analysis for {multiplicity} state {state_number}")

                # Determine how the outer loop should get its next `line`
                if results.buffered_line:
                    # A line (e.g. next state header) was buffered. Outer loop will use this.
                    # No action needed here for `line` variable of outer loop.
                    pass
                elif line_for_current_state is None:
                    # EOF was hit within this state's parsing. Outer loop should break.
                    line = None  # Signal outer loop to break
                else:
                    # This state finished, and the last `line_for_current_state`
                    # (returned by a sub-parser) was not a new state/section or fluff.
                    # This implies it was an unrecognized line that ended the state.
                    # The spec says such a line should be pushed back.
                    # However, the logic above already buffers it if it was unrecognized.
                    # This path should ideally not be hit if the above `else` for unrecognized lines
                    # correctly buffers and sets line_for_current_state to None.
                    # For safety, if line_for_current_state is NOT None and nothing is buffered,
                    # it means the inner while loop exited because line_for_current_state was not None,
                    # but it wasn't a new state/section (that would have set results.buffered_line).
                    # This indicates an unprocessed line that should be the start for the next outer loop iteration.
                    line = line_for_current_state

            else:  # Line is not a state start, and not a known next section (checked at top of outer loop)
                logger.debug(f"Transition Density Matrix Analysis parsing stopped by unrecognized line: {line.strip()}")
                results.buffered_line = line
                break  # Break main loop

            # Prepare `line` for the next iteration of the main outer loop.
            if results.buffered_line:
                # This will be picked up at the top of the while True loop.
                pass
            elif line is None:  # EOF was signaled from state parsing
                break
            else:  # Current state processed, fetch next line for potential new state or end.
                try:
                    line = next(iterator)
                except StopIteration:
                    logger.debug("EOF reached at the end of Transition Density Matrix Analysis main loop.")
                    break

        results.parsed_tddft_transition_dm_analysis = True
        logger.info(
            f"Finished parsing Transition Density Matrix Analysis. Found {len(results.transition_density_matrix_detailed_analyses_list)} states."
        )

    def _parse_mulliken_tdm(
        self, iterator: LineIterator, current_block_line: str, results: _MutableCalculationData
    ) -> tuple[TransitionDMMulliken | None, str | None]:
        populations: list[TransitionDMAtomPopulation] = []
        sum_qta: float | None = None
        sum_qt2: float | None = None
        line: str | None = current_block_line  # This is "Mulliken Population Analysis..."

        try:
            line = next(iterator)  # Headers: "Atom Trans. (e) ..." & "----"
            while "----" in line or "Atom      Trans. (e)" in line or not line.strip():
                line = next(iterator)

            while "----" not in line and "Sum:" not in line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        atom_index = int(parts[0]) - 1
                        symbol = parts[1]
                        trans_e = float(parts[2])
                        h_plus = float(parts[3])
                        e_minus = float(parts[4])
                        del_q = float(parts[5]) if len(parts) > 5 else None
                        populations.append(
                            TransitionDMAtomPopulation(atom_index, symbol, trans_e, h_plus, e_minus, del_q)
                        )
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse Mulliken TDM atom line: '{line.strip()}'. Error: {e}")
                line = next(iterator)

            if "Sum:" in line.strip():
                line = next(iterator)
            if "----" in line.strip():
                line = next(iterator)  # Consume final separator

            while line is not None and not line.strip():
                line = next(iterator)
            if line and "Sum of absolute trans. charges, QTa" in line:
                match = re.search(r"QTa\s*=\s*(-?\d+\.\d+)", line)
                if match:
                    sum_qta = float(match.group(1))
                line = next(iterator)

            while line is not None and not line.strip():
                line = next(iterator)
            if line and "Sum of squared  trans. charges, QT2" in line:
                match = re.search(r"QT2\s*=\s*(-?\d+\.\d+)", line)
                if match:
                    sum_qt2 = float(match.group(1))
                line = next(iterator)

            return TransitionDMMulliken(populations, sum_qta, sum_qt2), line
        except StopIteration:
            return TransitionDMMulliken(populations, sum_qta, sum_qt2) if populations else None, None
        except Exception as e:
            results.parsing_errors.append(f"Error in _parse_mulliken_tdm: {e} on line '{line}'")
            logger.error(f"Error in _parse_mulliken_tdm: {e} on line '{line}'")
            return None, line

    def _parse_ct_numbers(
        self, iterator: LineIterator, current_block_line: str, results: _MutableCalculationData
    ) -> tuple[TransitionDMCTNumbers | None, str | None]:
        ct_data_dict: dict[str, float] = {}
        line: str | None = current_block_line  # This is "CT numbers..."
        try:
            line = next(iterator)
            while line is not None and "=" in line:
                line_strip = line.strip()
                if not line_strip:
                    line = next(iterator)
                    continue
                match = re.match(r"^\s*(omega|2<alpha\|beta>|LOC|LOCa|<Phe>)\s*=\s*(-?\d+\.\d+)", line_strip)
                if match:
                    key, val_str = match.group(1), match.group(2)
                    norm_key = key.lower()
                    if key == "2<alpha|beta>":
                        norm_key = "two_alpha_beta_overlap"
                    elif key == "<Phe>":
                        norm_key = "phe_overlap"
                    ct_data_dict[norm_key] = float(val_str)
                else:
                    break
                line = next(iterator)

            return TransitionDMCTNumbers(**cast(dict, ct_data_dict)), line
        except StopIteration:
            return TransitionDMCTNumbers(**cast(dict, ct_data_dict)) if ct_data_dict else None, None
        except Exception as e:
            results.parsing_errors.append(f"Error in _parse_ct_numbers: {e} on line '{line}'")
            logger.error(f"Error in _parse_ct_numbers: {e} on line '{line}'")
            return None, line

    def _parse_exciton_analysis_tdm(
        self, iterator: LineIterator, current_block_line: str, results: _MutableCalculationData
    ) -> tuple[ExcitonAnalysisTransitionDM | None, str | None]:
        exciton_data_dict: dict[str, Any] = {}
        line: str | None = current_block_line  # This is "Exciton analysis..."

        # Helper to parse scalar values and component lines for exciton analysis
        def _process_exciton_line(l_str: str, data: dict[str, Any]):
            l_s = l_str.strip()
            # Scalar patterns (value on same line)
            scalar_patterns = {
                "Trans. dipole moment [D]:": "total_transition_dipole_moment",
                "Transition <r^2> [a.u.]:": "transition_r_squared_au",
                "|<r_e - r_h>| [Ang]:": "hole_electron_distance_ang",
                "Covariance(r_h, r_e) [Ang^2]:": "covariance_rh_re_ang_sq",
                "Correlation coefficient:": "correlation_coefficient",
            }
            # Vector patterns (value on same line, starts with key)
            vector_same_line_patterns = {
                "Cartesian components [D]:": "transition_dipole_moment_components",
                "Cartesian components [a.u.]:": "transition_r_squared_au_components",
                "<r_h> [Ang]:": "hole_position_ang",
                "<r_e> [Ang]:": "electron_position_ang",
            }
            # Patterns where scalar is on current line, vector on next
            scalar_then_vector_patterns = {
                "Hole size [Ang]:": ("hole_size_ang", "hole_size_ang_components"),
                "Electron size [Ang]:": ("electron_size_ang", "electron_size_ang_components"),
                "RMS electron-hole separation [Ang]:": (
                    "rms_electron_hole_separation_ang",
                    "rms_electron_hole_separation_ang_components",
                ),
                "Center-of-mass size [Ang]:": ("center_of_mass_size_ang", "center_of_mass_size_ang_components"),
            }

            # Check for pending vector components from previous line
            pending_vector_key = data.get("_pending_vector_key")
            if pending_vector_key:
                triplet = extract_float_triplet(l_s)
                if triplet:
                    data[pending_vector_key] = triplet
                data.pop("_pending_vector_key")
                return True  # Line processed

            for pattern, key in scalar_patterns.items():
                if l_s.startswith(pattern):
                    val = safe_float(l_s.replace(pattern, "").strip())
                    if val is not None:
                        data[key] = val
                    return True

            for pattern, key in vector_same_line_patterns.items():
                if l_s.startswith(pattern):
                    triplet = extract_float_triplet(l_s.replace(pattern, ""))
                    if triplet:
                        data[key] = triplet
                    return True

            for pattern, (s_key, v_key) in scalar_then_vector_patterns.items():
                if l_s.startswith(pattern):
                    val = safe_float(l_s.replace(pattern, "").strip())
                    if val is not None:
                        data[s_key] = val
                    data["_pending_vector_key"] = v_key  # Expect vector on next line
                    return True
            return False  # Line not processed by this helper

        try:
            line = next(iterator)  # Consume "Exciton analysis..." line passed as current_block_line
            while line is not None:
                line_strip = line.strip()
                if not line_strip:  # Empty line can mark end of sub-block
                    # Peek ahead to see if it's truly the end or just a blank line within
                    try:
                        peek_line = next(iterator)
                        results.buffered_line = peek_line  # Must buffer peeked line
                        if (
                            not peek_line.strip()
                            or any(known_start in peek_line for known_start in KNOWN_NEXT_SECTION_STARTS)
                            or re.match(r"^\s*(Singlet|Triplet)\s+(\d+)\s*:\s*$", peek_line.strip())
                            or "Mulliken Population Analysis" in peek_line
                            or "CT numbers (Mulliken)" in peek_line
                        ):
                            break  # End of exciton block confirmed by empty line then new section/state
                        else:  # Not end, continue with peek_line as current line
                            line = peek_line
                            results.buffered_line = None  # consumed peek_line
                            if not _process_exciton_line(line, exciton_data_dict):
                                break  # If peeked line is not part of exciton, break
                    except StopIteration:
                        break  # EOF means end of block
                else:
                    if not _process_exciton_line(line, exciton_data_dict):
                        break  # Line not processed, assume end of exciton data for this state

                if not exciton_data_dict.get("_pending_vector_key") and line is not None:
                    # Only advance if not waiting for vector comp and not EOF
                    line = next(iterator)
                elif line is None:  # Should not happen if _process_exciton_line is last call before loop check
                    break

            exciton_data_dict.pop("_pending_vector_key", None)  # Clean up
            return ExcitonAnalysisTransitionDM(**exciton_data_dict), line

        except StopIteration:
            exciton_data_dict.pop("_pending_vector_key", None)
            return ExcitonAnalysisTransitionDM(**exciton_data_dict) if exciton_data_dict else None, None
        except Exception as e:
            results.parsing_errors.append(f"Error in _parse_exciton_analysis_tdm: {e} on line '{line}'")
            logger.error(f"Error in _parse_exciton_analysis_tdm: {e} on line '{line}'")
            return None, line

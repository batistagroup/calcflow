import re
from collections.abc import Iterator

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import (
    LineIterator,
    ScfData,
    ScfEnergyComponents,
    ScfIteration,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

FLOAT_PAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"  # Common float pattern


# --- SCF Parser --- #
SCF_CONVERGED_LINE_PAT = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
SCF_DIIS_ITER_PAT = re.compile(
    rf"^\s*(\d+)\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}"
)
SCF_SOSCF_ITER_PAT = re.compile(
    rf"^\s*(\d+)\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}"
)
SCF_ENERGY_COMPONENTS_START_PAT = re.compile(r"TOTAL SCF ENERGY")
SCF_NUCLEAR_REP_PAT = re.compile(r"^\s*Nuclear Repulsion\s*:\s*(-?\d+\.\d+)")
SCF_ELECTRONIC_PAT = re.compile(r"^\s*Electronic Energy\s*:\s*(-?\d+\.\d+)")
SCF_ONE_ELECTRON_PAT = re.compile(r"^\s*One Electron Energy\s*:\s*(-?\d+\.\d+)")
SCF_TWO_ELECTRON_PAT = re.compile(r"^\s*Two Electron Energy\s*:\s*(-?\d+\.\d+)")
SCF_XC_PAT = re.compile(r"^\s*E\(XC\)\s*:\s*(-?\d+\.\d+)")


# Moved _parse_scf_block logic here
class ScfParser(SectionParser):
    """Parses the SCF calculation block."""

    def __init__(self) -> None:
        self.converged: bool = False
        self.n_iterations: int = 0
        self.last_scf_energy_eh: float | None = None
        self.nuclear_rep_eh: float | None = None
        self.electronic_eh: float | None = None
        self.one_electron_eh: float | None = None
        self.two_electron_eh: float | None = None
        self.xc_eh: float | None = None
        self.iteration_history: list[ScfIteration] = []
        self.latest_iter_num: int = 0
        self._current_table_type: str | None = None

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        # Trigger on the iteration table headers, only if SCF not already parsed
        return not current_data.parsed_scf and ("D-I-I-S" in line or "S-O-S-C-F" in line)

    def _reset_state(self) -> None:
        """Resets parser state for a new parsing run."""
        self.converged = False
        self.n_iterations = 0
        self.last_scf_energy_eh = None
        self.nuclear_rep_eh = None
        self.electronic_eh = None
        self.one_electron_eh = None
        self.two_electron_eh = None
        self.xc_eh = None
        self.iteration_history = []
        self.latest_iter_num = 0
        self._current_table_type = None

    def _parse_iteration_line(self, line_stripped: str) -> bool:
        """Parses a single line potentially belonging to an iteration table."""
        iter_data: ScfIteration | None = None
        parsed_iter_line = False

        if self._current_table_type == "DIIS":
            diis_match = SCF_DIIS_ITER_PAT.match(line_stripped)
            if diis_match:
                vals = diis_match.groups()
                try:
                    iter_data = ScfIteration(
                        iteration=int(vals[0]),
                        energy_eh=float(vals[1]),
                        delta_e_eh=float(vals[2]),
                        rmsdp=float(vals[3]),
                        maxdp=float(vals[4]),
                        diis_error=float(vals[5]),
                        damping=float(vals[6]),
                        time_sec=float(vals[7]),
                    )
                    parsed_iter_line = True
                except (ValueError, IndexError) as e:
                    raise ParsingError(f"Could not parse DIIS iteration: {line_stripped}") from e
        elif self._current_table_type == "SOSCF":
            soscf_match = SCF_SOSCF_ITER_PAT.match(line_stripped)
            if soscf_match:
                vals = soscf_match.groups()
                try:
                    iter_data = ScfIteration(
                        iteration=int(vals[0]),
                        energy_eh=float(vals[1]),
                        delta_e_eh=float(vals[2]),
                        rmsdp=float(vals[3]),
                        maxdp=float(vals[4]),
                        max_gradient=float(vals[5]),
                        time_sec=float(vals[6]),
                    )
                    parsed_iter_line = True
                except (ValueError, IndexError) as e:
                    raise ParsingError(f"Could not parse SOSCF iteration: {line_stripped}") from e

        if iter_data:
            self.iteration_history.append(iter_data)
            self.latest_iter_num = max(self.latest_iter_num, iter_data.iteration)
            self.last_scf_energy_eh = iter_data.energy_eh

        return parsed_iter_line

    def _parse_iteration_tables(self, iterator: LineIterator, initial_line: str) -> tuple[Iterator[str], str | None]:
        """Parses the DIIS and SOSCF iteration tables."""
        logger.debug("Parsing SCF iteration tables...")

        # Determine initial table type and consume header/dashes
        if "D-I-I-S" in initial_line:
            self._current_table_type = "DIIS"
            next(iterator, None)  # Consume dashed line
        elif "S-O-S-C-F" in initial_line:
            self._current_table_type = "SOSCF"
            next(iterator, None)  # Consume dashed line
        else:
            # Should not happen based on `matches`
            raise ParsingError("Unexpected initial line for iteration table parsing.")

        last_line = None
        for line in iterator:
            last_line = line
            line_stripped = line.strip()

            # Check for table type change
            if "D-I-I-S" in line:
                self._current_table_type = "DIIS"
                next(iterator, None)  # Header
                next(iterator, None)  # Dashes
                continue
            if "S-O-S-C-F" in line:
                self._current_table_type = "SOSCF"
                next(iterator, None)  # Header
                next(iterator, None)  # Dashes
                continue

            # Try parsing as an iteration line
            if self._parse_iteration_line(line_stripped):
                continue

            # Check for convergence message
            if line_stripped.startswith(("*", " ")) and "SCF CONVERGED AFTER" in line:
                conv_match = SCF_CONVERGED_LINE_PAT.search(line)
                if conv_match:
                    self.converged = True
                    self.n_iterations = int(conv_match.group(1))
                    logger.debug("Found convergence line within iteration table.")
                    # Convergence found, exit table parsing, let outer loop handle next step
                    return iterator, last_line
                continue  # Just a comment line

            # Check for end of table conditions (blank line or start of components)
            if not line_stripped or SCF_ENERGY_COMPONENTS_START_PAT.search(line):
                logger.debug("Iteration table parsing finished.")
                return iterator, last_line  # Return the line that terminated the table

        # Reached end of iterator
        logger.debug("Iteration table parsing finished (end of file).")
        return iterator, last_line  # Return the last line processed

    def _parse_energy_components(self, iterator: LineIterator, initial_line: str) -> tuple[Iterator[str], str | None]:
        """Parses the 'TOTAL SCF ENERGY' component block."""
        logger.debug("Parsing SCF energy components...")
        found_all_mandatory = False
        last_line: str | None = initial_line  # Start processing with the initial line

        while last_line is not None:
            line_stripped = last_line.strip()

            # Check for start again (might be the initial_line)
            if SCF_ENERGY_COMPONENTS_START_PAT.search(last_line):
                pass

            nr_match = SCF_NUCLEAR_REP_PAT.search(last_line)
            if nr_match:
                self.nuclear_rep_eh = float(nr_match.group(1))

            el_match = SCF_ELECTRONIC_PAT.search(last_line)
            if el_match:
                self.electronic_eh = float(el_match.group(1))

            one_el_match = SCF_ONE_ELECTRON_PAT.search(last_line)
            if one_el_match:
                self.one_electron_eh = float(one_el_match.group(1))

            two_el_match = SCF_TWO_ELECTRON_PAT.search(last_line)
            if two_el_match:
                self.two_electron_eh = float(two_el_match.group(1))

            xc_match = SCF_XC_PAT.search(last_line)
            if xc_match:
                self.xc_eh = float(xc_match.group(1))

            # Check if we found all mandatory components
            if not found_all_mandatory and None not in [
                self.nuclear_rep_eh,
                self.electronic_eh,
                self.one_electron_eh,
                self.two_electron_eh,
            ]:
                found_all_mandatory = True
                logger.debug("Found all mandatory SCF components.")

            # --- Check for termination conditions ---
            # 1. Found all mandatory components AND the current line wasn't parsed
            # 2. Encountered a known strict terminator pattern
            should_terminate = False
            # if found_all_mandatory and not parsed_this_line: # Removed this condition
            #      should_terminate = True
            #      logger.debug(f"SCF component parsing finished: Found mandatory and non-component line: {line_stripped}")

            strict_terminators = (
                "SCF CONVERGENCE",  # Section start
                "ORBITAL ENERGIES",
                "MULLIKEN POPULATION ANALYSIS",
                "LOEWDIN POPULATION ANALYSIS",
                "MAYER POPULATION ANALYSIS",
                "DFT DISPERSION CORRECTION",
                "FINAL SINGLE POINT ENERGY",
                "TIMINGS",
                "------",
            )
            is_terminator_line = any(term in last_line for term in strict_terminators)

            # Special case: dashes before mandatory components are found should be ignored
            if is_terminator_line and line_stripped.startswith("------") and not found_all_mandatory:
                is_terminator_line = False  # Ignore this terminator for now

            if is_terminator_line:
                should_terminate = True
                logger.debug(f"SCF component parsing finished due to terminator: {line_stripped}")

            if should_terminate:
                return iterator, last_line  # Return the line that terminated parsing

            # If not terminating, get the next line
            last_line = next(iterator, None)

        # Reached end of iterator
        logger.debug("SCF component parsing finished (end of file).")
        return iterator, last_line  # Return None as last line

    def _finalize_scf_data(self, results: _MutableCalculationData) -> None:
        """Validates parsed data and creates the ScfData object."""
        logger.debug("Finalizing SCF data...")

        # --- Final Checks ---
        warning_convergence_missing = "SCF convergence line 'SCF CONVERGED AFTER ...' not found."
        if not self.converged:
            logger.warning(f"{warning_convergence_missing} Assuming not converged.")
            # Don't add to results.warnings yet, might be inferred later

        # Correct n_iterations based on history if necessary
        warning_iteration_mismatch = None
        if self.n_iterations == 0 and self.latest_iter_num > 0:
            self.n_iterations = self.latest_iter_num
        elif self.n_iterations > 0 and self.latest_iter_num > self.n_iterations:
            warning_iteration_mismatch = (
                f"Convergence line reported {self.n_iterations} cycles, but found {self.latest_iter_num} in history."
            )
            logger.warning(f"{warning_iteration_mismatch} Using history count.")
            self.n_iterations = self.latest_iter_num
        elif self.n_iterations == 0 and not self.iteration_history:
            # This is logged implicitly if convergence is false and components missing later
            pass

        warning_history_missing = None
        if self.converged and not self.iteration_history and self.n_iterations > 0:
            warning_history_missing = (
                f"SCF convergence line reported {self.n_iterations} cycles, but no iteration history parsed."
            )
            logger.warning(warning_history_missing)

        # --- Check Mandatory Components ---
        found_all_mandatory_components = None not in [
            self.nuclear_rep_eh,
            self.electronic_eh,
            self.one_electron_eh,
            self.two_electron_eh,
        ]

        # Only raise error if SCF seemed to run but components are missing
        if (self.converged or self.n_iterations > 0 or self.iteration_history) and not found_all_mandatory_components:
            missing_comps = [
                comp
                for comp, val in zip(
                    ["nuclear_rep_eh", "electronic_eh", "one_electron_eh", "two_electron_eh"],
                    [self.nuclear_rep_eh, self.electronic_eh, self.one_electron_eh, self.two_electron_eh],
                    strict=False,
                )
                if val is None
            ]
            logger.error(f"Missing mandatory SCF energy components after SCF ran/converged: {missing_comps}")
            raise ParsingError("Could not parse all required SCF energy components after SCF execution.")
        elif not found_all_mandatory_components:
            # SCF didn't run or failed early, components might be missing. This is not an error state.
            warning_components_missing_no_run = "SCF did not run or failed early; energy components not found."
            logger.warning(warning_components_missing_no_run)
            results.parsing_warnings.append(warning_components_missing_no_run)
            results.parsed_scf = True  # Mark as parsed/attempted
            logger.debug("SCF block parsing finished without finding components (likely SCF did not run).")
            return  # Cannot create ScfData without components if SCF didn't run

        # Assertions are valid now because we expect components if we reached here
        assert self.nuclear_rep_eh is not None
        assert self.electronic_eh is not None
        assert self.one_electron_eh is not None
        assert self.two_electron_eh is not None

        components = ScfEnergyComponents(
            nuclear_repulsion_eh=self.nuclear_rep_eh,
            electronic_eh=self.electronic_eh,
            one_electron_eh=self.one_electron_eh,
            two_electron_eh=self.two_electron_eh,
            xc_eh=self.xc_eh,
        )

        # --- Determine Final Energy ---
        final_scf_energy: float
        if self.iteration_history:
            final_scf_energy = self.iteration_history[-1].energy_eh
        elif self.last_scf_energy_eh is not None:
            final_scf_energy = self.last_scf_energy_eh
            logger.warning(
                f"Using last parsed SCF energy {final_scf_energy} as iteration history is missing/incomplete."
            )
        else:
            # This path implies components were found, but no energy value was stored from iterations.
            # This should only happen if the SCF converged in 0 iterations (unlikely) or parsing failed severely.
            # Let's rely on the electronic energy + nuclear repulsion as a fallback ONLY if converged.
            # This is a heuristic and might be inaccurate if xc_eh is significant or components aren't exact.
            if self.converged:
                final_scf_energy = self.electronic_eh + self.nuclear_rep_eh
                warning_fallback_energy = f"No iteration energy found despite convergence. Using Electronic + Nuclear Repulsion: {final_scf_energy:.8f} Eh."
                logger.warning(warning_fallback_energy)
                results.parsing_warnings.append(warning_fallback_energy)
                # Also update last_scf_energy_eh if it was None
                if self.last_scf_energy_eh is None:
                    self.last_scf_energy_eh = final_scf_energy
            else:
                # Not converged, no history, components found (implies failure after component printout?)
                # This state is ambiguous. Raising an error is safer.
                raise ParsingError("Failed to determine final SCF energy (calculation state unclear).")

        # --- Create Result Object ---
        scf_result = ScfData(
            converged=self.converged,
            energy_eh=final_scf_energy,
            components=components,
            n_iterations=self.n_iterations,
            iteration_history=tuple(self.iteration_history),
        )
        results.scf = scf_result
        results.parsed_scf = True
        logger.debug(f"Successfully parsed SCF data: Converged={self.converged}, Energy={final_scf_energy:.8f} Eh")

        # Append warnings collected during finalization
        if not self.converged:
            # Append the missing convergence warning only if it wasn't implicitly handled
            # by the 'SCF did not run' warning earlier.
            if not found_all_mandatory_components:  # If components also missing, that warning was already added
                pass
            else:  # Components found, but convergence line missing
                results.parsing_warnings.append(warning_convergence_missing)

        if warning_iteration_mismatch:
            results.parsing_warnings.append(warning_iteration_mismatch)
        if warning_history_missing:
            results.parsing_warnings.append(warning_history_missing)

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parses the entire SCF block by coordinating helper methods."""
        self._reset_state()
        logger.debug(f"Starting SCF block parsing triggered by: {current_line.strip()}")

        try:
            # --- Stage 1: Parse Iteration Tables ---
            iterator, last_line_from_iter = self._parse_iteration_tables(iterator, current_line)
            current_processing_line = last_line_from_iter  # Line that ended table parsing

            # --- Stage 2: Search for and Parse Energy Components ---
            found_components_start = False
            while current_processing_line is not None:
                if SCF_ENERGY_COMPONENTS_START_PAT.search(current_processing_line):
                    found_components_start = True
                    iterator, last_line_from_comp = self._parse_energy_components(iterator, current_processing_line)
                    current_processing_line = last_line_from_comp  # Line that ended component parsing
                    break  # Exit component search loop

                # Check for late convergence message if not found during iteration parsing
                if not self.converged:
                    conv_match_late = SCF_CONVERGED_LINE_PAT.search(current_processing_line)
                    if conv_match_late:
                        self.converged = True
                        self.n_iterations = int(conv_match_late.group(1))
                        logger.debug("Found convergence line after iteration tables.")

                # Check for terminators that signal the end of anything related to SCF *before* components start
                terminators_post_scf = (
                    "ORBITAL ENERGIES",
                    "MULLIKEN POPULATION ANALYSIS",
                    "LOEWDIN POPULATION ANALYSIS",
                    "MAYER POPULATION ANALYSIS",
                    "DFT DISPERSION CORRECTION",
                    "FINAL SINGLE POINT ENERGY",
                    "TIMINGS",
                )
                if any(term in current_processing_line for term in terminators_post_scf):
                    logger.debug(
                        f"SCF parsing loop finished before components due to terminator: {current_processing_line.strip()}"
                    )
                    break  # Exit search loop

                # Get next line if components not started and no terminator hit
                current_processing_line = next(iterator, None)

            # If components never started, log it.
            if not found_components_start:
                warning_components_section_missing = "SCF Energy Components section ('TOTAL SCF ENERGY') not found."
                logger.warning(warning_components_section_missing)
                # Append this warning only if SCF seemed to run (converged or had iterations)
                # Otherwise, the lack of components is expected.
                if self.converged or self.iteration_history:
                    results.parsing_warnings.append(warning_components_section_missing)

            # --- Stage 3: Finalize and Create Data Object ---
            self._finalize_scf_data(results)

        except ParsingError as e:
            logger.error(f"ParsingError during SCF block processing: {e}", exc_info=False)  # Log only message
            results.parsing_warnings.append(f"SCF Block: {e}")
            results.parsed_scf = True  # Mark as attempted even on error
        except Exception as e:
            logger.error(f"Unexpected error during SCF block parsing: {e}", exc_info=True)
            results.parsing_errors.append("SCF Block: Unexpected error during parsing.")
            results.parsed_scf = True  # Mark as attempted even on error

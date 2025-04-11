import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import (
    LineIterator,
    ScfData,
    ScfEnergyComponents,
    ScfIteration,
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
class ScfParser:
    """Parses the SCF calculation block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        # Trigger on the iteration table headers, only if SCF not already parsed
        return not current_data.parsed_scf and ("D-I-I-S" in line or "S-O-S-C-F" in line)

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug(f"Starting SCF block parsing triggered by: {current_line.strip()}")
        converged = False
        n_iterations = 0
        last_scf_energy_eh: float | None = None
        nuclear_rep_eh: float | None = None
        electronic_eh: float | None = None
        one_electron_eh: float | None = None
        two_electron_eh: float | None = None
        xc_eh: float | None = None
        latest_iter_num = 0
        iteration_history: list[ScfIteration] = []

        in_iteration_table = True
        current_table_type = None
        in_scf_component_section = False
        found_all_mandatory_components = False

        # Determine table type from the initial trigger line
        if "D-I-I-S" in current_line:
            current_table_type = "DIIS"
            next(iterator, None)  # Consume dashed line
        elif "S-O-S-C-F" in current_line:
            current_table_type = "SOSCF"
            next(iterator, None)  # Consume dashed line
        else:
            # This case should not happen due to `matches` logic
            logger.error("SCF Parser called with unexpected initial line.")
            return

        try:
            # Use loop_iterator = iter([current_line] + list(iterator)) ? No, iterator is stateful
            # Need to process lines *from* the iterator
            for line in iterator:
                line_stripped = line.strip()

                # --- State 1: Parsing Iteration Tables --- #
                if in_iteration_table:
                    if "D-I-I-S" in line:
                        current_table_type = "DIIS"
                        next(iterator, None)  # Header
                        next(iterator, None)  # Dashes
                        continue
                    if "S-O-S-C-F" in line:
                        current_table_type = "SOSCF"
                        next(iterator, None)  # Header
                        next(iterator, None)  # Dashes
                        continue

                    iter_data: ScfIteration | None = None
                    if current_table_type == "DIIS":
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
                            except (ValueError, IndexError) as e:
                                raise ParsingError(f"Could not parse DIIS iteration: {line_stripped}") from e
                    elif current_table_type == "SOSCF":
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
                            except (ValueError, IndexError) as e:
                                raise ParsingError(f"Could not parse SOSCF iteration: {line_stripped}") from e

                    if iter_data:
                        iteration_history.append(iter_data)
                        latest_iter_num = max(latest_iter_num, iter_data.iteration)
                        last_scf_energy_eh = iter_data.energy_eh
                        continue
                    elif line_stripped.startswith(("*", " ")) and "SCF CONVERGED AFTER" in line:
                        conv_match = SCF_CONVERGED_LINE_PAT.search(line)
                        if conv_match:
                            converged = True
                            n_iterations = int(conv_match.group(1))
                            in_iteration_table = False
                            # Don't continue here, might be the start of components immediately
                        # else: just a comment line, continue
                        continue
                    elif not line_stripped:  # Blank line can end the table
                        in_iteration_table = False
                        continue  # Process next line normally
                    elif SCF_ENERGY_COMPONENTS_START_PAT.search(line):
                        in_iteration_table = False
                        in_scf_component_section = True
                        # Fall through to component parsing for this line

                # --- State 2: Parsing SCF Energy Components --- #
                if in_scf_component_section:  # Use 'if' not 'elif' to catch fall-through
                    parsed_this_line = False
                    nr_match = SCF_NUCLEAR_REP_PAT.search(line)
                    if nr_match:
                        nuclear_rep_eh = float(nr_match.group(1))
                        parsed_this_line = True

                    el_match = SCF_ELECTRONIC_PAT.search(line)
                    if el_match:
                        electronic_eh = float(el_match.group(1))
                        parsed_this_line = True

                    one_el_match = SCF_ONE_ELECTRON_PAT.search(line)
                    if one_el_match:
                        try:
                            one_electron_eh = float(one_el_match.group(1))
                            parsed_this_line = True
                        except ValueError:
                            logger.error(
                                f"Could not convert One Electron Energy: {one_el_match.group(1)}", exc_info=True
                            )

                    two_el_match = SCF_TWO_ELECTRON_PAT.search(line)
                    if two_el_match:
                        try:
                            two_electron_eh = float(two_el_match.group(1))
                            parsed_this_line = True
                        except ValueError:
                            logger.error(
                                f"Could not convert Two Electron Energy: {two_el_match.group(1)}", exc_info=True
                            )

                    xc_match = SCF_XC_PAT.search(line)
                    if xc_match:
                        try:
                            xc_eh = float(xc_match.group(1))
                            parsed_this_line = True
                        except ValueError:
                            logger.error(f"Could not convert XC Energy: {xc_match.group(1)}", exc_info=True)

                    if not found_all_mandatory_components and None not in [
                        nuclear_rep_eh,
                        electronic_eh,
                        one_electron_eh,
                        two_electron_eh,
                    ]:
                        found_all_mandatory_components = True

                    # Check for termination conditions
                    if not parsed_this_line or found_all_mandatory_components:
                        strict_terminators = (
                            "SCF CONVERGENCE",
                            "ORBITAL ENERGIES",
                            "MULLIKEN POPULATION ANALYSIS",
                            "LOEWDIN POPULATION ANALYSIS",
                            "MAYER POPULATION ANALYSIS",
                            "DFT DISPERSION CORRECTION",
                            "FINAL SINGLE POINT ENERGY",
                            "TIMINGS",
                            "------",
                        )
                        is_terminator = False
                        if any(term in line for term in strict_terminators):
                            if line.strip().startswith("------") and not found_all_mandatory_components:
                                pass  # Ignore dashes before components found
                            else:
                                is_terminator = True

                        if is_terminator:
                            logger.debug(f"SCF component parsing finished due to terminator: {line.strip()}")
                            in_scf_component_section = False
                            break  # Exit SCF block parsing loop

                    continue  # Continue processing lines in component section

                # --- State 3: After iterations, before/after components --- #
                else:
                    # Check again if components start here
                    if SCF_ENERGY_COMPONENTS_START_PAT.search(line):
                        in_scf_component_section = True
                        continue  # Go back to component parsing state for this line

                    # Check for late convergence message if not found earlier
                    if not converged:
                        conv_match_late = SCF_CONVERGED_LINE_PAT.search(line)
                        if conv_match_late:
                            converged = True
                            n_iterations = int(conv_match_late.group(1))
                            continue  # Found convergence, process next line

                    # Check for terminators that signal the end of anything related to SCF
                    terminators_post_scf = (
                        "ORBITAL ENERGIES",
                        "MULLIKEN POPULATION ANALYSIS",
                        "LOEWDIN POPULATION ANALYSIS",
                        "MAYER POPULATION ANALYSIS",
                        "DFT DISPERSION CORRECTION",
                        "FINAL SINGLE POINT ENERGY",
                        "TIMINGS",
                    )
                    if any(term in line for term in terminators_post_scf):
                        logger.debug(f"SCF parsing loop finished due to post-SCF terminator: {line.strip()}")
                        break  # Exit SCF block parsing loop

        except Exception as e:
            logger.error(f"Error during SCF block parsing: {e}", exc_info=True)
            raise ParsingError("Failed during SCF block parsing.") from e

        # --- Final Checks and Object Creation --- #
        if not converged:
            logger.warning("SCF convergence line 'SCF CONVERGED AFTER ...' not found. Assuming not converged.")

        if n_iterations == 0 and latest_iter_num > 0:
            n_iterations = latest_iter_num
        elif n_iterations > 0 and latest_iter_num > n_iterations:
            logger.warning(
                f"Convergence line reported {n_iterations} cycles, but found {latest_iter_num} in history. Using history count."
            )
            n_iterations = latest_iter_num
        elif n_iterations == 0 and not iteration_history:
            logger.warning("No SCF iterations found or parsed, and convergence line not found.")

        if converged and not iteration_history and n_iterations > 0:
            logger.warning(f"SCF convergence line reported {n_iterations} cycles, but no iteration history parsed.")
            # Allow continuing if energy components are found, but log heavily.

        # Check mandatory components only if SCF seems to have run
        if converged or n_iterations > 0 or iteration_history:
            if not found_all_mandatory_components:
                missing_comps = [
                    comp
                    for comp, val in zip(
                        ["nuclear_rep_eh", "electronic_eh", "one_electron_eh", "two_electron_eh"],
                        [nuclear_rep_eh, electronic_eh, one_electron_eh, two_electron_eh],
                        strict=False,
                    )
                    if val is None
                ]
                logger.error(f"Missing mandatory SCF energy components after SCF ran/converged: {missing_comps}")
                raise ParsingError("Could not parse all required SCF energy components after SCF execution.")
        else:
            # If SCF didn't run, components might be missing, which is okay, but log it.
            if not found_all_mandatory_components:
                logger.warning("SCF did not run or failed immediately; energy components not found.")
            # We cannot create ScfData without components if it didn't run. Return without setting results.scf
            results.parsed_scf = True  # Mark as parsed/attempted
            logger.warning("SCF block parsing finished without finding components (likely SCF did not run).")
            return

        # Assertions only valid if components were expected and found
        assert nuclear_rep_eh is not None
        assert electronic_eh is not None
        assert one_electron_eh is not None
        assert two_electron_eh is not None

        components = ScfEnergyComponents(
            nuclear_repulsion_eh=nuclear_rep_eh,
            electronic_eh=electronic_eh,
            one_electron_eh=one_electron_eh,
            two_electron_eh=two_electron_eh,
            xc_eh=xc_eh,
        )

        final_scf_energy: float
        if iteration_history:
            final_scf_energy = iteration_history[-1].energy_eh
        elif last_scf_energy_eh is not None:
            # This case should ideally be covered by components being present if SCF converged
            final_scf_energy = last_scf_energy_eh
            logger.warning(
                f"Using last parsed SCF energy {final_scf_energy} as iteration history is missing/incomplete."
            )
        else:
            # If converged but somehow no energy found (unlikely given checks)
            if converged:
                raise ParsingError("SCF converged but failed to extract final energy.")
            else:  # Not converged, no history, no components
                raise ParsingError("Failed to determine final SCF energy (calculation may have failed early).")

        scf_result = ScfData(
            converged=converged,
            energy_eh=final_scf_energy,
            components=components,
            n_iterations=n_iterations,
            iteration_history=tuple(iteration_history),
        )
        results.scf = scf_result
        results.parsed_scf = True
        logger.debug(f"Successfully parsed SCF data: {str(scf_result)}")

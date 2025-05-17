import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.typing import (
    LineIterator,
    ScfIteration,
    ScfResults,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- Regex Patterns --- #

# Start marker for the SCF section (looks for the specific header)
SCF_START_PAT = re.compile(r"^\s*General SCF calculation program by")

# SCF Iteration line
# Example:   7     -75.3118844639      5.35e-08  Convergence criterion met
SCF_ITER_PAT = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+([\d\.eE+-]+)")
ROOTHAAN_STEP_PAT = re.compile(r"Roothaan Step")  # For marking Roothaan steps

# SCF Convergence marker (on the same line as the last iteration)
SCF_CONVERGENCE_PAT = re.compile(r"Convergence criterion met")

# Final SCF energy line (distinct from the "Total energy" line)
SCF_FINAL_ENERGY_PAT = re.compile(r"^\s*SCF\s+energy\s*=\s*(-?\d+\.\d+)")

# SMD Summary Patterns (appear before SCF final energy in output)
SMD_SUMMARY_START_PAT = re.compile(r"^\s*Summary of SMD free energies:")
G_PCM_PAT = re.compile(r"^\s*G_PCM\s*=\s*(-?\d+\.\d+)\s*kcal/mol")
G_CDS_PAT = re.compile(r"^\s*G_CDS\s*=\s*(-?\d+\.\d+)\s*kcal/mol")
# G_ENP is E_SCF including G_PCM, should match the SCF_FINAL_ENERGY_PAT value
G_ENP_PAT = re.compile(r"^\s*G_ENP\s*=\s*(-?\d+\.\d+)\s*a\.u\.")
# G_TOT is the final total energy with SMD, should match FINAL_ENERGY_PAT from sp.py
G_TOT_PAT = re.compile(r"^\s*G\(tot\)\s*=\s*(-?\d+\.\d+)\s*a\.u\.")

# End marker (heuristic based on common following sections or dashes)
SCF_ITER_TABLE_END_PAT = re.compile(r"^\s*-{20,}")  # Dashed line after table
SCF_BLOCK_END_HEURISTIC_PAT = re.compile(r"Orbital Energies|Mulliken Net Atomic Charges|Multipole Moments|^\s*-{60}")
# Add the normal termination pattern here as well for the heuristic search
NORMAL_TERM_PAT = re.compile(r"Thank you very much for using Q-Chem")

# --- MOM Specific Patterns ---
MOM_ACTIVE_PAT = re.compile(r"^\s*Maximum Overlap Method Active")
IMOM_METHOD_PAT = re.compile(r"^\s*IMOM method")  # Assuming IMOM is the primary method for now
MOM_OVERLAP_PAT = re.compile(r"^\s*MOM overlap:\s+(-?\d+\.\d+)\s+/\s+(-?\d+\.?\d*)")


class ScfParser(SectionParser):
    """Parses the SCF calculation block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line indicates the start of the SCF calculation header."""
        return (SCF_START_PAT.search(line) is not None) and not current_data.parsed_scf

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parse SCF iterations, final results, and SMD summary, handling multiple SCF tables."""
        logger.debug("Starting parsing of SCF block.")

        # Initialize SMD fields
        results.smd_g_pcm_kcal_mol = None
        results.smd_g_cds_kcal_mol = None
        results.smd_g_enp_au = None
        results.smd_g_tot_au = None

        all_iterations_collected: list[ScfIteration] = []
        overall_converged_status: bool = False  # Tracks convergence of the *last* processed segment
        final_scf_energy_from_explicit_line: float | None = None
        line_after_all_scf_tables: str | None = None

        # Initial search for the very first "Cycle       Energy         DIIS error" header
        line = current_line  # current_line is the one that matched SCF_START_PAT
        initial_header_found = False
        try:
            while "Cycle       Energy         DIIS error" not in line:
                line = next(iterator)
            initial_header_found = True
        except StopIteration:
            logger.warning("No 'Cycle Energy DIIS error' header found after SCF block start.")
            results.parsed_scf = True  # Mark as attempted
            return

        # Outer loop: Process one SCF iteration table per iteration
        # `line` currently holds the first "Cycle..." header or was set by a previous loop break.
        while initial_header_found or ("Cycle       Energy         DIIS error" in line):
            if not initial_header_found:  # For subsequent tables
                logger.debug(f"Found subsequent 'Cycle Energy DIIS error' header: {line.strip()}")
            initial_header_found = False  # Reset flag

            logger.debug(f"Processing an SCF iteration table starting with: {line.strip()}")
            try:
                # Consume "---------------------------------------" separator after header
                # The header itself is in `line`, next(iterator) gets the separator
                separator_line = next(iterator)
                if not separator_line.strip().startswith("---"):
                    logger.warning(f"Expected separator '---' after SCF header, got: {separator_line.strip()}")
                    # Attempt to continue, but this might indicate a parsing issue
            except StopIteration:
                logger.error("File ended unexpectedly after an SCF table header.")
                line_after_all_scf_tables = line  # Preserve current line
                break  # Abrupt end, break outer loop

            current_segment_converged = False
            pending_mom_active_signal = False
            pending_mom_method_type: str | None = None
            pending_mom_overlap_current: float | None = None
            pending_mom_overlap_target: float | None = None

            # Inner loop: Process iterations within the current table
            while True:
                try:
                    iter_line = next(iterator)
                except StopIteration:
                    logger.error("File ended unexpectedly while parsing SCF iterations in a segment.")
                    overall_converged_status = current_segment_converged
                    line_after_all_scf_tables = ""  # File ended
                    line = ""  # To ensure outer loop condition fails
                    break  # Break inner loop, which will lead to outer loop break

                # 1. Check for MOM context lines
                if MOM_ACTIVE_PAT.search(iter_line):
                    pending_mom_active_signal = True
                    logger.debug("MOM active signal received.")
                    continue
                if IMOM_METHOD_PAT.search(iter_line):
                    pending_mom_method_type = "IMOM"
                    logger.debug("MOM method type 'IMOM' received.")
                    continue
                match_mom_overlap = MOM_OVERLAP_PAT.search(iter_line)
                if match_mom_overlap:
                    try:
                        pending_mom_overlap_current = float(match_mom_overlap.group(1))
                        pending_mom_overlap_target = float(match_mom_overlap.group(2))
                        logger.debug(
                            f"MOM overlap {pending_mom_overlap_current}/{pending_mom_overlap_target} received."
                        )
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse MOM overlap from line: {iter_line.strip()}")
                        pending_mom_overlap_current = None
                        pending_mom_overlap_target = None
                    continue

                # 2. Parse SCF iteration data
                match_iter = SCF_ITER_PAT.search(iter_line)
                if match_iter:
                    try:
                        iteration_num = int(match_iter.group(1))
                        energy = float(match_iter.group(2))
                        diis_error = float(match_iter.group(3))
                        step_type = "Roothaan" if ROOTHAAN_STEP_PAT.search(iter_line) else None

                        actual_mom_active = pending_mom_active_signal
                        actual_mom_method = pending_mom_method_type if actual_mom_active else None
                        actual_overlap_curr = pending_mom_overlap_current if actual_mom_active else None
                        actual_overlap_targ = pending_mom_overlap_target if actual_mom_active else None

                        all_iterations_collected.append(
                            ScfIteration(
                                iteration=iteration_num,
                                energy=energy,
                                diis_error=diis_error,
                                step_type=step_type,
                                mom_active=actual_mom_active if actual_mom_active else None,
                                mom_method_type=actual_mom_method,
                                mom_overlap_current=actual_overlap_curr,
                                mom_overlap_target=actual_overlap_targ,
                            )
                        )
                        pending_mom_active_signal = False
                        pending_mom_method_type = None
                        pending_mom_overlap_current = None
                        pending_mom_overlap_target = None

                        if SCF_CONVERGENCE_PAT.search(iter_line):
                            current_segment_converged = True
                            logger.debug(f"SCF segment converged at iteration {iteration_num}.")
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse SCF iteration data from line: {iter_line.strip()}")
                    continue  # Next line for inner loop

                # 3. Check for end of current iteration table / start of new / end of block
                if SCF_ITER_TABLE_END_PAT.search(iter_line):  # "-------------------"
                    logger.debug("End of current SCF iteration table (dashed line).")
                    overall_converged_status = current_segment_converged
                    try:
                        line = next(iterator)  # Prepare `line` for the outer loop's condition
                    except StopIteration:
                        logger.warning("File ended immediately after SCF table separator.")
                        line = ""  # Ensure outer loop terminates
                    break  # Break inner loop

                if "Cycle       Energy         DIIS error" in iter_line:
                    logger.debug("New SCF table header encountered, ending current segment.")
                    overall_converged_status = current_segment_converged
                    line = iter_line  # This is the header for the next segment
                    break  # Break inner loop

                if SCF_BLOCK_END_HEURISTIC_PAT.search(iter_line) or NORMAL_TERM_PAT.search(iter_line):
                    logger.debug(f"SCF block end heuristic encountered: {iter_line.strip()}")
                    overall_converged_status = current_segment_converged
                    line_after_all_scf_tables = iter_line
                    line = ""  # To terminate outer loop
                    break  # Break inner loop

                if iter_line.strip():  # Some other non-blank, unrecognized line
                    logger.warning(f"Unrecognized non-blank line, assuming end of SCF tables: {iter_line.strip()}")
                    overall_converged_status = current_segment_converged
                    line_after_all_scf_tables = iter_line
                    line = ""  # To terminate outer loop
                    break  # Break inner loop
                # If blank line, inner loop continues to get next line

            # End of inner loop (iterations for current table)
            if not ("Cycle       Energy         DIIS error" in line or line == ""):
                # If inner loop broke NOT because a new header was found or file ended
                # but due to heuristics or unrecognized line, ensure outer loop also terminates.
                if line_after_all_scf_tables is None:
                    line_after_all_scf_tables = line
                line = ""  # force outer loop termination

        # End of outer loop (SCF tables)

        if not all_iterations_collected:
            logger.warning("No SCF iterations were collected.")
            results.parsed_scf = True  # Mark as attempted
            return

        # --- Post-iteration-tables parsing (SMD, Final SCF Energy) ---
        smd_summary_found = False
        scf_energy_line_found = False  # For "SCF energy = ..." line

        # `line_after_all_scf_tables` is the line that ended iteration parsing (e.g., "Orbital Energies").
        # This line has ALREADY BEEN READ from the iterator.
        # The search for SMD/Final Energy needs to start by EXAMINING this line,
        # and THEN subsequent lines from `next(iterator)`.

        current_search_line = line_after_all_scf_tables

        try:
            while current_search_line is not None:  # Process current_search_line, then get next
                # Check for SMD Summary Block Start
                if not smd_summary_found and SMD_SUMMARY_START_PAT.search(current_search_line):
                    logger.debug("Found 'Summary of SMD free energies:' block.")
                    smd_summary_found = True
                    # current_search_line is the "Summary..." header. SMD content starts on next line.
                    while True:  # Inner loop for SMD block lines
                        try:
                            smd_content_line = next(iterator)
                        except StopIteration:
                            logger.warning("File ended during SMD summary parsing.")
                            current_search_line = None  # Signal outer search loop to terminate
                            break  # Break SMD inner loop

                        match_g_pcm = G_PCM_PAT.search(smd_content_line)
                        if match_g_pcm:
                            results.smd_g_pcm_kcal_mol = float(match_g_pcm.group(1))
                            continue
                        match_g_cds = G_CDS_PAT.search(smd_content_line)
                        if match_g_cds:
                            results.smd_g_cds_kcal_mol = float(match_g_cds.group(1))
                            continue
                        match_g_enp = G_ENP_PAT.search(smd_content_line)
                        if match_g_enp:
                            results.smd_g_enp_au = float(match_g_enp.group(1))
                            continue

                        match_g_tot = G_TOT_PAT.search(smd_content_line)
                        if match_g_tot:
                            results.smd_g_tot_au = float(match_g_tot.group(1))
                            # current_search_line will be updated after this inner loop
                            break  # Exit SMD inner loop.

                        if SCF_FINAL_ENERGY_PAT.search(smd_content_line):  # SCF energy found within SMD block
                            current_search_line = smd_content_line  # This line is the SCF energy line
                            # Break inner SMD loop; outer search loop will process current_search_line.
                            break

                        if not smd_content_line.strip():  # Blank line
                            logger.debug("Blank line in SMD summary, assuming end of its content.")
                            break  # Exit SMD inner loop

                    if current_search_line is None:
                        break  # File ended within SMD parse
                    if SCF_FINAL_ENERGY_PAT.search(current_search_line):  # If inner SMD loop broke on SCF energy line
                        pass  # Fall through to the SCF_FINAL_ENERGY_PAT check below for current_search_line
                    else:  # SMD loop finished (e.g. on G_TOT or blank line), get next line for outer search
                        try:
                            current_search_line = next(iterator)
                        except StopIteration:
                            current_search_line = None
                        continue  # With the new current_search_line

                # Check for Final SCF Energy
                match = SCF_FINAL_ENERGY_PAT.search(current_search_line)
                if match:
                    final_scf_energy_from_explicit_line = float(match.group(1))
                    scf_energy_line_found = True
                    logger.debug(f"Found final SCF energy: {final_scf_energy_from_explicit_line}")
                    break  # Found SCF energy, exit this search loop

                # Check safety break conditions for SMD/Final Energy search
                if SCF_BLOCK_END_HEURISTIC_PAT.search(current_search_line) or NORMAL_TERM_PAT.search(
                    current_search_line
                ):
                    if not scf_energy_line_found:
                        logger.warning(f"Final SCF energy line not found. Stopped at: {current_search_line.strip()}")
                    break  # Stop searching

                # Get next line for the post-processing search loop
                try:
                    current_search_line = next(iterator)
                except StopIteration:
                    if not scf_energy_line_found:
                        logger.warning("File ended before finding final SCF energy line (or after SMD summary).")
                    current_search_line = None  # Signal loop termination

        except StopIteration:  # Should be handled by current_search_line = None
            logger.warning("File ended unexpectedly during post-SCF iteration parsing.")
        except (ValueError, IndexError):
            logger.error(
                f"Could not parse SMD or final SCF energy value from line: {current_search_line.strip() if current_search_line else 'N/A'}",
                exc_info=True,
            )

        # --- Finalize ScfResults ---
        energy_to_store: float
        if final_scf_energy_from_explicit_line is not None:
            energy_to_store = final_scf_energy_from_explicit_line
        elif all_iterations_collected:
            energy_to_store = all_iterations_collected[-1].energy
            if overall_converged_status and not final_scf_energy_from_explicit_line:
                logger.warning(
                    "Using energy from last SCF iteration as final SCF energy; explicit 'SCF energy =' line not found or parsed."
                )
        else:  # Should be caught by the `if not all_iterations_collected:` check earlier
            raise ParsingError("Cannot determine energy for ScfResults: no iterations and no explicit final energy.")

        # Validate G_ENP vs SCF energy if both found
        if smd_summary_found and results.smd_g_enp_au is not None and final_scf_energy_from_explicit_line is not None:
            if abs(results.smd_g_enp_au - final_scf_energy_from_explicit_line) > 1e-6:
                logger.warning(
                    f"Mismatch between G_ENP from SMD summary ({results.smd_g_enp_au:.8f}) "
                    f"and explicit SCF energy ({final_scf_energy_from_explicit_line:.8f}). Using explicit SCF energy."
                )
        elif smd_summary_found and (results.smd_g_enp_au is None or results.smd_g_tot_au is None):
            logger.warning("SMD summary block identified, but some energy components (G_ENP, G_TOT) were not parsed.")

        results.scf = ScfResults(
            converged=overall_converged_status,
            energy=energy_to_store,
            n_iterations=len(all_iterations_collected),
            iterations=all_iterations_collected,
        )
        results.parsed_scf = True
        logger.info(
            f"Parsed SCF data. Converged: {overall_converged_status}, Energy: {energy_to_store:.8f}, Iterations: {len(all_iterations_collected)}."
        )

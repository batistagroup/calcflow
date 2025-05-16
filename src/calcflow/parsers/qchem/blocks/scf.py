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
        """Parse the SCF iterations and final results, including SMD summary if present."""
        logger.debug("Starting parsing of SCF block.")
        iterations: list[ScfIteration] = []
        converged = False
        final_scf_energy: float | None = None

        # Initialize SMD fields (assuming they exist on results object)
        results.smd_g_pcm_kcal_mol = None
        results.smd_g_cds_kcal_mol = None
        results.smd_g_enp_au = None
        results.smd_g_tot_au = None

        # Variables to hold MOM details 'pending' for the next SCF iteration line
        pending_mom_active_signal = False
        pending_mom_method_type: str | None = None
        pending_mom_overlap_current: float | None = None
        pending_mom_overlap_target: float | None = None

        # current_line already matched SCF_START_PAT
        # Find the start of the iteration table
        try:
            line = current_line
            while "Cycle       Energy         DIIS error" not in line:
                # TODO: Could parse SCF settings (Exchange, Correlation, etc.) here
                line = next(iterator)
            # Found the header line, consume the separator line below it
            _ = next(iterator)  # Consume "---------------------------------------"

        except StopIteration as e:
            raise ParsingError("Unexpected end of file while searching for SCF iteration table.") from e

        # --- Loop through SCF iteration lines --- #
        line = ""  # Reset line for the loop logic
        while True:
            try:
                line = next(iterator)
            except StopIteration as e:
                logger.error("File ended unexpectedly while parsing SCF iterations.")
                # If iterations were found, store partial results before raising
                if iterations:
                    results.scf = ScfResults(
                        converged=False,  # Assume not converged if file ends abruptly
                        energy=iterations[-1].energy,
                        n_iterations=len(iterations),
                        iterations=iterations,
                    )
                    results.parsed_scf = True  # Mark as attempted
                raise ParsingError("Unexpected end of file in SCF iteration block.") from e

            # 1. Check for MOM context lines first
            if MOM_ACTIVE_PAT.search(line):
                pending_mom_active_signal = True
                logger.debug("MOM active signal received for the next cycle.")
                continue

            if IMOM_METHOD_PAT.search(line):  # Assumes IMOM for now
                pending_mom_method_type = "IMOM"
                logger.debug("MOM method type 'IMOM' received for the next cycle.")
                continue

            match_mom_overlap = MOM_OVERLAP_PAT.search(line)
            if match_mom_overlap:
                try:
                    pending_mom_overlap_current = float(match_mom_overlap.group(1))
                    pending_mom_overlap_target = float(match_mom_overlap.group(2))
                    logger.debug(
                        f"MOM overlap {pending_mom_overlap_current}/{pending_mom_overlap_target} received for the next cycle."
                    )
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse MOM overlap from line: {line.strip()}")
                    pending_mom_overlap_current = None  # Ensure reset on error
                    pending_mom_overlap_target = None
                continue

            # 2. Check for end of iteration table (heuristic: dashed line)
            if SCF_ITER_TABLE_END_PAT.search(line):
                logger.debug("Found likely end marker for SCF iteration table.")
                break

            # Parse iteration data
            match_iter = SCF_ITER_PAT.search(line)
            if match_iter:
                try:
                    iteration = int(match_iter.group(1))
                    energy = float(match_iter.group(2))
                    diis_error = float(match_iter.group(3))

                    # Use pending MOM data if MOM was signaled as active
                    actual_mom_active = pending_mom_active_signal
                    actual_mom_method = pending_mom_method_type if actual_mom_active else None
                    actual_overlap_curr = pending_mom_overlap_current if actual_mom_active else None
                    actual_overlap_targ = pending_mom_overlap_target if actual_mom_active else None

                    iterations.append(
                        ScfIteration(
                            iteration=iteration,
                            energy=energy,
                            diis_error=diis_error,
                            mom_active=actual_mom_active if actual_mom_active else None,  # Store True or None
                            mom_method_type=actual_mom_method,
                            mom_overlap_current=actual_overlap_curr,
                            mom_overlap_target=actual_overlap_targ,
                        )
                    )

                    # Reset pending MOM details after consumption for this iteration
                    pending_mom_active_signal = False
                    pending_mom_method_type = None
                    pending_mom_overlap_current = None
                    pending_mom_overlap_target = None

                    # Check for convergence on the same line
                    if SCF_CONVERGENCE_PAT.search(line):
                        converged = True
                        logger.debug(f"SCF converged at iteration {iteration}.")
                        # Don't break yet, need the final SCF energy line which is usually after the table

                except (ValueError, IndexError):
                    logger.warning(f"Could not parse SCF iteration data from line: {line.strip()}")
                continue  # Process next iteration line
            else:
                # If the line wasn't an iteration and not the end marker, assume table ended
                if line.strip():  # Ignore blank lines
                    logger.debug(
                        f"Non-iteration/MOM line encountered after SCF table started, assuming end of table: {line.strip()}"
                    )
                break  # Assume table ended

        # --- Look for SMD Summary and final SCF energy line --- #
        # The loop above broke, 'line' holds the line that ended the SCF iteration table loop.
        # We continue searching from the *next* line for SMD summary and then SCF energy.

        smd_summary_found = False
        scf_energy_line_found = False

        try:
            # 'line' is the line that ended the previous loop (e.g. dashed line or unexpected)
            # We need to start reading new lines
            while True:  # Loop to find SMD summary and SCF energy
                line = next(iterator)

                # Check for SMD Summary Block Start
                if not smd_summary_found and SMD_SUMMARY_START_PAT.search(line):
                    logger.debug("Found 'Summary of SMD free energies:' block.")
                    smd_summary_found = True
                    # Now parse the contents of the SMD summary block
                    while True:  # Inner loop for SMD block lines
                        smd_line = next(iterator)  # Read next line for SMD content

                        match_g_pcm = G_PCM_PAT.search(smd_line)
                        if match_g_pcm:
                            results.smd_g_pcm_kcal_mol = float(match_g_pcm.group(1))
                            logger.debug(f"Parsed smd_g_pcm_kcal_mol: {results.smd_g_pcm_kcal_mol}")
                            continue

                        match_g_cds = G_CDS_PAT.search(smd_line)
                        if match_g_cds:
                            results.smd_g_cds_kcal_mol = float(match_g_cds.group(1))
                            logger.debug(f"Parsed smd_g_cds_kcal_mol: {results.smd_g_cds_kcal_mol}")
                            continue

                        match_g_enp = G_ENP_PAT.search(smd_line)
                        if match_g_enp:
                            results.smd_g_enp_au = float(match_g_enp.group(1))
                            logger.debug(f"Parsed smd_g_enp_au: {results.smd_g_enp_au}")
                            continue

                        match_g_tot = G_TOT_PAT.search(smd_line)
                        if match_g_tot:
                            results.smd_g_tot_au = float(match_g_tot.group(1))
                            logger.debug(f"Parsed smd_g_tot_au: {results.smd_g_tot_au}")
                            # This is the last expected line in the SMD summary block
                            # before the "SCF energy =" line typically.
                            break  # Exit SMD inner loop

                        # If it's not an SMD line and not the start of SCF energy, it might be an issue
                        # or the end of the SMD block content if format varies.
                        # For now, we break on G_TOT_PAT or if SCF_FINAL_ENERGY_PAT is found next.
                        if SCF_FINAL_ENERGY_PAT.search(smd_line):  # Check if next line is already SCF energy
                            line = smd_line  # Pass this line to the outer loop
                            break  # Exit SMD inner loop, outer loop will process it
                        if not smd_line.strip():  # if blank line, could be end of SMD block content
                            logger.debug("Blank line encountered in SMD summary, assuming end of its content.")
                            break  # Exit SMD inner loop
                    # After SMD block parsing, continue to look for SCF energy or end heuristics
                    if SCF_FINAL_ENERGY_PAT.search(line):  # If current `line` became SCF energy line
                        pass  # let it be processed below
                    else:  # Read next line after SMD block finished or G_TOT was the last parsed
                        line = next(iterator)

                # Check for Final SCF Energy
                match_final_scf = SCF_FINAL_ENERGY_PAT.search(line)
                if match_final_scf:
                    final_scf_energy = float(match_final_scf.group(1))
                    scf_energy_line_found = True
                    logger.debug(f"Found final SCF energy: {final_scf_energy}")
                    break  # Found SCF energy, exit this search loop

                # Check safety break conditions
                if SCF_BLOCK_END_HEURISTIC_PAT.search(line) or NORMAL_TERM_PAT.search(line):
                    logger.warning(
                        f"Could not find 'SCF energy =' line after iteration table (or SMD summary). Stopped search at: {line.strip()}"
                    )
                    break  # Stop searching

        except StopIteration:
            logger.warning("File ended before finding 'SCF energy =' line (or after SMD summary).")
        except (ValueError, IndexError):
            logger.error(f"Could not parse SMD or final SCF energy value from line: {line.strip()}", exc_info=True)
            # Allow None, warning will be issued below

        if smd_summary_found and results.smd_g_enp_au is not None and final_scf_energy is not None:
            if abs(results.smd_g_enp_au - final_scf_energy) > 1e-6:  # Tolerance for float comparison
                logger.warning(
                    f"Mismatch between G_ENP from SMD summary ({results.smd_g_enp_au:.8f}) "
                    f"and SCF energy ({final_scf_energy:.8f}). Using SCF energy for ScfResults."
                )
        elif smd_summary_found and (results.smd_g_enp_au is None or results.smd_g_tot_au is None):
            logger.warning(
                "SMD summary block was identified, but some energy components (G_ENP, G_TOT) were not parsed."
            )

        # --- Store results --- #
        if not iterations:
            logger.warning("No SCF iterations found in SCF block.")
            results.parsed_scf = True  # Mark as attempted, but no data
            return

        # Use the parsed final SCF energy if found, otherwise use the last iteration energy
        energy_to_store = final_scf_energy if final_scf_energy is not None else iterations[-1].energy
        if final_scf_energy is None and converged:  # Only warn if converged but couldn't find explicit SCF energy
            logger.warning(
                "Using energy from last SCF iteration as final SCF energy because 'SCF energy =' line was not found."
            )
        elif not scf_energy_line_found and converged:
            logger.warning("'SCF energy =' line was not found after SCF iterations.")

        results.scf = ScfResults(
            converged=converged,
            energy=energy_to_store,
            n_iterations=len(iterations),
            iterations=iterations,
        )
        results.parsed_scf = True
        logger.info(
            f"Parsed SCF data. Converged: {converged}, Energy: {energy_to_store:.8f}, Iterations: {len(iterations)}."
        )

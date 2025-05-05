import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.typing import (
    LineIterator,
    ScfData,
    ScfIteration,
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

# End marker (heuristic based on common following sections or dashes)
SCF_ITER_TABLE_END_PAT = re.compile(r"^\s*-{20,}")  # Dashed line after table
SCF_BLOCK_END_HEURISTIC_PAT = re.compile(r"Orbital Energies|Mulliken Net Atomic Charges|Multipole Moments|^\s*-{60}")
# Add the normal termination pattern here as well for the heuristic search
NORMAL_TERM_PAT = re.compile(r"Thank you very much for using Q-Chem")


class ScfParser(SectionParser):
    """Parses the SCF calculation block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line indicates the start of the SCF calculation header."""
        return (SCF_START_PAT.search(line) is not None) and not current_data.parsed_scf

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parse the SCF iterations and final results."""
        logger.debug("Starting parsing of SCF block.")
        iterations: list[ScfIteration] = []
        converged = False
        final_scf_energy: float | None = None

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
                    results.scf = ScfData(
                        converged=False,  # Assume not converged if file ends abruptly
                        energy_eh=iterations[-1].energy_eh,
                        n_iterations=len(iterations),
                        iteration_history=iterations,
                    )
                    results.parsed_scf = True  # Mark as attempted
                raise ParsingError("Unexpected end of file in SCF iteration block.") from e

            # Check for end of iteration table (heuristic: dashed line)
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
                    iterations.append(ScfIteration(iteration=iteration, energy_eh=energy, diis_error=diis_error))

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
                        f"Non-iteration line encountered after SCF table started, assuming end of table: {line.strip()}"
                    )
                break  # Assume table ended

        # --- Look for final SCF energy line --- #
        # The loop above broke, 'line' holds the line that ended the loop (dashed line or unexpected content)
        # Continue searching from the *next* line
        try:
            while final_scf_energy is None:
                # Get the next line first inside the loop
                line = next(iterator)
                match_final = SCF_FINAL_ENERGY_PAT.search(line)
                if match_final:
                    final_scf_energy = float(match_final.group(1))
                    logger.debug(f"Found final SCF energy: {final_scf_energy}")
                    # Don't break here, let the loop condition handle it or StopIteration

                # Check safety break conditions AFTER checking for the energy
                # Use the NORMAL_TERM_PAT defined globally or passed in context
                elif SCF_BLOCK_END_HEURISTIC_PAT.search(line) or NORMAL_TERM_PAT.search(line):
                    logger.warning(
                        f"Could not find 'SCF energy =' line after iteration table. Stopped search at section starting with: {line.strip()}"
                    )
                    break  # Stop searching

        except StopIteration:
            logger.warning("File ended before finding 'SCF energy =' line after iteration table.")
        except (ValueError, IndexError):
            logger.error(f"Could not parse final SCF energy value from line: {line.strip()}", exc_info=True)
            # Allow None, warning will be issued below

        # --- Store results --- #
        if not iterations:
            logger.warning("No SCF iterations found in SCF block.")
            results.parsed_scf = True  # Mark as attempted, but no data
            return

        # Use the parsed final SCF energy if found, otherwise use the last iteration energy
        energy_to_store = final_scf_energy if final_scf_energy is not None else iterations[-1].energy_eh
        if final_scf_energy is None and converged:  # Only warn if converged but couldn't find explicit SCF energy
            logger.warning("Using energy from last SCF iteration as final SCF energy.")

        results.scf = ScfData(
            converged=converged,
            energy_eh=energy_to_store,
            n_iterations=len(iterations),
            iteration_history=iterations,
        )
        results.parsed_scf = True
        logger.info(
            f"Parsed SCF data. Converged: {converged}, Energy: {energy_to_store:.8f}, Iterations: {len(iterations)}."
        )

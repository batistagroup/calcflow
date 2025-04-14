import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import (
    LineIterator,  # Added import for type hint
    OptimizationCycleData,
    RelaxationStepData,
    _MutableCalculationData,  # Reuse or adapt parts of this
    _MutableOptData,
)
from calcflow.utils import logger

# Regex patterns for RelaxationStepParser
RELAX_START_PAT = re.compile(r"^\s*ORCA GEOMETRY RELAXATION STEP")
CONV_TABLE_HEADER_PAT = re.compile(r"^\s*-+\|Geometry convergence\|-+")
CONV_TABLE_LINE_PAT = re.compile(r"^\s*([\w\s]+?)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(YES|NO)")
TRUST_RADIUS_PAT = re.compile(r"^\s*New trust radius\s+\.{3}\s+(\d+\.\d+)")
OPT_CONVERGED_PAT = re.compile(r"\*\*\*\s+THE OPTIMIZATION HAS CONVERGED\s+\*\*\*")


class RelaxationStepParser:
    """Parses the geometry relaxation step details and convergence table."""

    def matches(self, line: str, current_data: _MutableCalculationData | _MutableOptData) -> bool:
        """Checks if the line marks the start of the relaxation step block."""
        return bool(RELAX_START_PAT.search(line))

    def parse(
        self,
        iterator: LineIterator,
        current_line: str,
        results: _MutableOptData,
        current_cycle_data: OptimizationCycleData | None,
    ) -> None:
        """Parses the convergence table and other relaxation step info."""
        if not current_cycle_data:
            logger.warning("RelaxationStepParser matched outside of an optimization cycle. Skipping.")
            return

        logger.debug(f"Parsing Geometry Relaxation Step block starting near line: {current_line}")
        energy_change: float | None = None
        rms_gradient: float | None = None
        max_gradient: float | None = None
        rms_step: float | None = None
        max_step: float | None = None
        converged_items: dict[str, bool] = {}
        trust_radius: float | None = None
        in_convergence_table = False

        try:
            while True:
                line = next(iterator)

                if CONV_TABLE_HEADER_PAT.search(line):
                    in_convergence_table = True
                    # Skip the header line itself and the separator line below it
                    try:
                        next(iterator)  # Consume the 'Item Value Tolerance Converged' line
                        next(iterator)  # Consume the '---------------------' line
                    except StopIteration as e:
                        raise ParsingError("File ended unexpectedly within convergence table header.") from e
                    continue

                if in_convergence_table:
                    if line.strip().startswith("........") or line.strip().startswith("---"):
                        # End of the main convergence table section
                        in_convergence_table = False
                        continue

                    match_conv = CONV_TABLE_LINE_PAT.search(line)
                    if match_conv:
                        item = match_conv.group(1).strip()
                        value = float(match_conv.group(2))
                        converged = match_conv.group(4) == "YES"
                        converged_items[item] = converged

                        # Assign specific values based on item name
                        if item == "Energy change":
                            energy_change = value
                        elif item == "RMS gradient":
                            rms_gradient = value
                        elif item == "MAX gradient":
                            max_gradient = value
                        elif item == "RMS step":
                            rms_step = value
                        elif item == "MAX step":
                            max_step = value
                    else:
                        # Line within table block but doesn't match pattern - might be end or unexpected format
                        logger.debug(f"Exiting convergence table parsing due to non-matching line: {line.strip()}")
                        in_convergence_table = False  # Assume table ended

                # Look for trust radius *after* the table
                match_trust = TRUST_RADIUS_PAT.search(line)
                if match_trust:
                    try:
                        trust_radius = float(match_trust.group(1))
                        logger.debug(f"Parsed trust radius: {trust_radius}")
                        # Trust radius often appears near the end of the relaxation block info
                        # We might want to break here or look for a more definitive end pattern
                        # For now, continue, assuming it might appear before other tables
                        continue
                    except (ValueError, IndexError):
                        # Log warning but don't fail the whole parse for trust radius
                        logger.warning(f"Could not parse trust radius value from line: {line.strip()}", exc_info=True)

                # Define break conditions more carefully
                # Break if we hit the start of the next cycle or final eval (handled by main loop)
                # Remove the double-blank check
                # Use a more specific marker like 'Geometry step timings:' to end the block
                if line.strip().startswith("Geometry step timings:"):
                    logger.debug("Exiting RelaxationStepParser at 'Geometry step timings:'.")
                    break

                # Check for the convergence message within the relaxation block
                match_opt_converged = OPT_CONVERGED_PAT.search(line)
                if match_opt_converged:
                    logger.info("Optimization Converged message found (within RelaxationStepParser).")
                    # Set final status only if no error occurred
                    if results.termination_status != "ERROR":
                        results.termination_status = "CONVERGED"
                    # Continue parsing the rest of the relaxation block
                    continue

        except StopIteration:
            logger.warning("Reached end of file while parsing relaxation step block.")
        except ParsingError:  # Re-raise critical parsing errors
            raise
        except Exception as e:
            raise ParsingError("An unexpected error occurred during relaxation step parsing.") from e

        # Validation - check if essential parts were parsed
        if not converged_items:
            logger.warning(
                f"Relaxation step block parsed for cycle {current_cycle_data.cycle_number}, but no convergence items found."
            )
            # Decide if this is critical? Probably not, allow continuation.

        relax_data = RelaxationStepData(
            energy_change=energy_change,
            rms_gradient=rms_gradient,
            max_gradient=max_gradient,
            rms_step=rms_step,
            max_step=max_step,
            converged_items=converged_items,
            trust_radius=trust_radius,
        )
        current_cycle_data.relaxation_step = relax_data
        logger.debug(f"Stored RelaxationStepData for cycle {current_cycle_data.cycle_number}")

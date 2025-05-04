import re
from typing import final

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.typing import AtomicCharges, LineIterator, SectionParser, _MutableCalculationData
from calcflow.utils import logger

# --- Regex Patterns ---
# Start of the Mulliken charges block
MULLIKEN_START_PAT = re.compile(r"^\s*Ground-State Mulliken Net Atomic Charges")
# Line containing atom index and charge
CHARGE_LINE_PAT = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+(-?\d+\.\d+)")
# End of the charges table (typically a line of dashes or blank line)
TABLE_END_PAT = re.compile(r"^\s*-+\s*$|^\s*$")
# Line indicating sum of charges (can also signal end)
SUM_CHARGE_PAT = re.compile(r"^\s*Sum of atomic charges\s*=")


@final
class MullikenChargesParser(SectionParser):
    """Parses the Mulliken Net Atomic Charges section from Q-Chem output."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line indicates the start of the Mulliken charges block."""
        # Only parse if not already parsed to avoid duplicates
        return MULLIKEN_START_PAT.search(line) is not None and not current_data.parsed_mulliken_charges

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Extracts Mulliken charges and updates the results."""
        logger.debug("Entering Mulliken Charges block parsing.")
        charges: dict[int, float] = {}
        # Store the line number from the main iterator for better error reporting
        # This requires the main loop to pass it, or we estimate based on current_line.
        # For now, use relative line numbers within the block.
        block_line_num = 0
        parsed_charge_count = 0

        try:
            # Skip the blank line after the header
            _ = next(iterator)
            # Skip the column titles line
            _ = next(iterator)
            # Skip the separator line (---) before the actual charges
            _ = next(iterator)
            block_line_num += 3  # Account for the three skipped lines

            current_block_line = ""
            while True:
                current_block_line = next(iterator)
                block_line_num += 1

                # Check for end conditions *first*
                if TABLE_END_PAT.search(current_block_line) or SUM_CHARGE_PAT.search(current_block_line):
                    logger.debug(f"Found end of Mulliken charges table at block line {block_line_num}.")
                    break

                match = CHARGE_LINE_PAT.search(current_block_line)
                if match:
                    try:
                        atom_index_one_based = int(match.group(1))
                        charge = float(match.group(3))
                        # Store using 0-based index
                        charges[atom_index_one_based - 1] = charge
                        parsed_charge_count += 1
                        logger.debug(f"Parsed charge: Atom {atom_index_one_based - 1}, Charge {charge}")
                    except (ValueError, IndexError) as parse_err:
                        logger.warning(
                            f"Could not parse charge line content: '{current_block_line.strip()}': {parse_err}"
                        )
                        results.parsing_warnings.append(
                            f"Could not parse Mulliken charge line: {current_block_line.strip()}"
                        )
                else:
                    # Log unexpected lines more clearly
                    logger.warning(
                        f"Unexpected non-charge line format in Mulliken charges block (line {block_line_num}): "
                        f"'{current_block_line.strip()}'"
                    )
                    results.parsing_warnings.append(
                        f"Unexpected line in Mulliken charges block: {current_block_line.strip()}"
                    )
                    # Continue parsing, assuming it might be a minor formatting glitch or extra info

        except StopIteration as e:
            logger.error("File ended unexpectedly while parsing Mulliken charges table.")
            # Append any charges found before the unexpected end
            if charges:
                logger.warning("Appending partially parsed Mulliken charges due to unexpected file end.")
                mulliken_data = AtomicCharges(method="Mulliken", charges=charges)
                results.atomic_charges.append(mulliken_data)
                results.parsed_mulliken_charges = True  # Mark as parsed even if incomplete
                results.parsing_errors.append("Incomplete Mulliken charges section found.")
            raise ParsingError("Incomplete Mulliken charges section found.") from e

        except Exception as e:
            # Log the specific line where the error occurred if possible
            err_line_info = (
                f"near block line {block_line_num}: '{current_block_line.strip()}'" if block_line_num > 0 else ""
            )
            logger.error(f"Error parsing Mulliken charges block {err_line_info}: {e}", exc_info=True)
            # Still try to append partial data if any charges were found
            if charges:
                logger.warning("Appending partially parsed Mulliken charges due to error.")
                mulliken_data = AtomicCharges(method="Mulliken", charges=charges)
                results.atomic_charges.append(mulliken_data)
                results.parsed_mulliken_charges = True
                results.parsing_errors.append(f"Error during Mulliken parsing: {e}")
            raise ParsingError(f"Failed to parse Mulliken charges section: {e}") from e

        # --- Appending Results (moved outside the main try block) ---
        if not charges:
            logger.warning("Mulliken charges block processed, but no charges were extracted.")
            results.parsing_warnings.append("Mulliken charges block parsed, but no charges found.")
            # Set parsed flag to true even if empty, to prevent re-parsing attempts
            results.parsed_mulliken_charges = True
        else:
            logger.debug(f"Successfully extracted {parsed_charge_count} charge entries.")
            mulliken_data = AtomicCharges(method="Mulliken", charges=charges)
            results.atomic_charges.append(mulliken_data)
            results.parsed_mulliken_charges = True
            logger.info(f"Successfully parsed and stored {len(charges)} Mulliken charges.")

        # The line causing the loop break has already been consumed by next(iterator)
        # No extra consumption needed here.

        logger.debug("Exiting Mulliken Charges block parsing.")

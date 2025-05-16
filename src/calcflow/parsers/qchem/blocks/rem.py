import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.typing import LineIterator, SectionParser, _MutableCalculationData
from calcflow.utils import logger

# --- Regex Patterns --- #
REM_START_PAT = re.compile(r"^\s*\$rem", re.IGNORECASE)
REM_END_PAT = re.compile(r"^\s*\$end", re.IGNORECASE)

# Generic key-value pattern for most rem variables
# It captures a key (e.g., METHOD, BASIS, UNRESTRICTED)
# and its value. Handles cases with or without explicit equals sign.
# Value can be a single word, a number, or quoted strings (not handled yet, assumes simple values).
REM_KV_PAT = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*(?:=\s*|\s+)(\S+)")


class RemBlockParser(SectionParser):
    """Parses the $rem block to extract key calculation parameters.
    It stores all found key-value pairs in results.rem dictionary.
    Also, it specifically extracts method, basis, and solvent_method to dedicated fields for convenience.
    """

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line starts the $rem block."""
        # Allow parsing $rem if it hasn't been marked as fully parsed yet.
        # This also helps prevent re-entry if $rem somehow appears multiple times.
        return REM_START_PAT.search(line) is not None and not current_data.parsed_rem_block

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parse lines within the $rem block until $end is found."""
        logger.debug("Starting parsing of $rem block.")

        # Ensure results.rem dictionary exists
        if results.rem is None:
            results.rem = {}

        while True:
            try:
                line = next(iterator)
            except StopIteration as e:
                logger.error("File ended unexpectedly while parsing $rem block.")
                raise ParsingError(f"Unexpected end of file in $rem block started near '{current_line.strip()}'") from e

            if REM_END_PAT.search(line):
                logger.debug("Found $end, finishing $rem block parsing.")
                break

            kv_match = REM_KV_PAT.match(line)
            if kv_match:
                key = kv_match.group(1).lower()  # Store keys in lowercase for consistency
                value = kv_match.group(2)
                results.rem[key] = value
                logger.debug(f"Parsed from $rem: {key} = {value}")

                # For convenience, also populate specific common fields
                if key == "method" and not results.parsed_meta_method:
                    results.calculation_method = value
                    results.parsed_meta_method = True
                    logger.debug(f"Set specific field calculation_method: {results.calculation_method}")
                elif key == "basis" and not results.parsed_meta_basis:
                    results.basis_set = value
                    results.parsed_meta_basis = True
                    logger.debug(f"Set specific field basis_set: {results.basis_set}")
                elif key == "solvent_method" and getattr(results, "solvent_method", None) is None:
                    # Assuming solvent_method should also be stored in lowercase in its specific field
                    results.solvent_method = value.lower()
                    logger.debug(f"Set specific field solvent_method: {results.solvent_method}")

        # Mark that the rem block has been processed.
        results.parsed_rem_block = True

        # Final checks for essential dedicated fields (optional, based on requirements)
        if not results.parsed_meta_method:
            logger.warning("METHOD not found within the parsed $rem block.")
            results.parsing_warnings.append("METHOD not found in $rem block.")
        if not results.parsed_meta_basis:
            logger.warning("BASIS not found within the parsed $rem block.")
            results.parsing_warnings.append("BASIS not found in $rem block.")
        # solvent_method check can remain as is or be adapted based on how often it's truly essential.

        return

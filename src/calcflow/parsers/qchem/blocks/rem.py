import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.typing import LineIterator, SectionParser, _MutableCalculationData
from calcflow.utils import logger

# --- Regex Patterns --- #
REM_START_PAT = re.compile(r"^\s*\$rem", re.IGNORECASE)
REM_END_PAT = re.compile(r"^\s*\$end", re.IGNORECASE)

# Patterns for specific variables within $rem
METHOD_PAT = re.compile(r"^\s*METHOD\s+(\S+)", re.IGNORECASE)
BASIS_PAT = re.compile(r"^\s*BASIS\s+(\S+)", re.IGNORECASE)
# Add patterns for other rem variables if needed later


class RemBlockParser(SectionParser):
    """Parses the $rem block to extract key calculation parameters like method and basis."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line starts the $rem block."""
        # Match only if method or basis hasn't been parsed yet, to avoid re-entry if format is odd
        # Although, typically $rem appears only once near the beginning.
        if REM_START_PAT.search(line):
            # Check if we still need to parse method or basis
            return not current_data.parsed_meta_method or not current_data.parsed_meta_basis
        return False

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parse lines within the $rem block until $end is found."""
        logger.debug("Starting parsing of $rem block.")

        # Consume the start line ($rem)
        # The main loop already consumed it via next(), so we start iterating from the next line.

        while True:
            try:
                line = next(iterator)
            except StopIteration as e:
                logger.error("File ended unexpectedly while parsing $rem block.")
                # Raise an error as an unterminated $rem block is problematic.
                raise ParsingError(f"Unexpected end of file in $rem block started near '{current_line.strip()}'") from e

            # Check for the end of the block
            if REM_END_PAT.search(line):
                logger.debug("Found $end, finishing $rem block parsing.")
                break

            # Check for METHOD if not already found
            if not results.parsed_meta_method:
                match_method = METHOD_PAT.search(line)
                if match_method:
                    results.calculation_method = match_method.group(1)
                    results.parsed_meta_method = True
                    logger.debug(f"Parsed calculation_method: {results.calculation_method}")
                    # Continue to check for basis on the same line (unlikely but possible)

            # Check for BASIS if not already found
            if not results.parsed_meta_basis:
                match_basis = BASIS_PAT.search(line)
                if match_basis:
                    results.basis_set = match_basis.group(1)
                    results.parsed_meta_basis = True
                    logger.debug(f"Parsed basis_set: {results.basis_set}")
                    # Continue to potentially find other rem variables on the same line

            # Optimization: If both method and basis are found, we could potentially break early,
            # but it's safer to parse until $end to handle complex $rem blocks or future needs.

        # Check if essential parameters were found (optional, based on requirements)
        if not results.parsed_meta_method:
            logger.warning("METHOD not found within the parsed $rem block.")
            results.parsing_warnings.append("METHOD not found in $rem block.")
        if not results.parsed_meta_basis:
            logger.warning("BASIS not found within the parsed $rem block.")
            results.parsing_warnings.append("BASIS not found in $rem block.")

        return

import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.typing import LineIterator, SectionParser, _MutableCalculationData
from calcflow.utils import logger

# --- Regex Patterns --- #
SMX_START_PAT = re.compile(r"^\s*\$smx", re.IGNORECASE)
SMX_END_PAT = re.compile(r"^\s*\$end", re.IGNORECASE)
SOLVENT_NAME_PAT = re.compile(r"^\s*solvent\s+(\S+)", re.IGNORECASE)


class SmxBlockParser(SectionParser):
    """Parses the $smx block to extract the solvent name."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line starts the $smx block."""
        # Match only if solvent_name hasn't been parsed yet.
        if SMX_START_PAT.search(line):
            # Assuming `solvent_name` attribute exists on `current_data`
            # and is initialized to None.
            return getattr(current_data, "solvent_name", None) is None
        return False

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parse lines within the $smx block until $end is found."""
        logger.debug("Starting parsing of $smx block.")

        # The main loop already consumed the start line ($smx)

        line_number = 0  # For error reporting if file ends unexpectedly
        try:
            while True:
                line = next(iterator)
                line_number += 1

                if SMX_END_PAT.search(line):
                    logger.debug("Found $end, finishing $smx block parsing.")
                    break

                match_solvent = SOLVENT_NAME_PAT.search(line)
                if match_solvent:
                    # Assuming `solvent_name` attribute exists on `results`
                    results.solvent_name = match_solvent.group(1).lower()  # Store solvent name in lowercase
                    logger.debug(f"Parsed solvent_name: {results.solvent_name}")
                    # Typically, only 'solvent' is in $smx, but loop continues to $end

            if getattr(results, "solvent_name", None) is None:
                logger.warning("Solvent name not found within the parsed $smx block.")
                results.parsing_warnings.append("Solvent name not found in $smx block.")

        except StopIteration as e:
            logger.error(f"File ended unexpectedly while parsing $smx block after {line_number} lines.")
            raise ParsingError(f"Unexpected end of file in $smx block started near '{current_line.strip()}'") from e

        return

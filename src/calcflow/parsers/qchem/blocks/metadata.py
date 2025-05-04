import re

from calcflow.parsers.qchem.typing import LineIterator, SectionParser, _MutableCalculationData
from calcflow.utils import logger

# --- Regex Patterns --- #
QCHEM_VERSION_PAT = re.compile(r"^ Q-Chem (\d+\.\d+(\.\d+)?), Q-Chem, Inc")
HOST_PAT = re.compile(r"^ Host: (\S+)")
RUN_DATE_PAT = re.compile(r"^ Q-Chem begins on (.*)")  # Captures the whole date string


class MetadataParser(SectionParser):
    """Parses general metadata lines like Q-Chem version, host, and run date."""

    # Store patterns as class attributes for easy access
    _patterns = {
        "qchem_version": QCHEM_VERSION_PAT,
        "host": HOST_PAT,
        "run_date": RUN_DATE_PAT,
    }
    _parsed_flags = {
        "qchem_version": "parsed_meta_version",
        "host": "parsed_meta_host",
        "run_date": "parsed_meta_run_date",
    }

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Checks if the line matches any metadata pattern that hasn't been parsed yet."""
        for key, pattern in self._patterns.items():
            if pattern.search(line):
                # Check if the corresponding flag exists and is False
                flag_name = self._parsed_flags[key]
                if not getattr(current_data, flag_name, False):
                    return True
        return False

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parses the matched metadata line and updates results. Consumes only the current line."""
        parsed_any = False
        for key, pattern in self._patterns.items():
            match = pattern.search(current_line)
            if match:
                flag_name = self._parsed_flags[key]
                # Double-check flag ensure we don't overwrite
                if not getattr(results, flag_name, False):
                    value = match.group(1).strip()
                    setattr(results, key, value)
                    setattr(results, flag_name, True)  # Mark as parsed
                    logger.debug(f"Parsed metadata - {key}: {value}")
                    parsed_any = True
                    # Since this parser only handles single lines, break after first match on a line
                    break

        if not parsed_any:
            # This should ideally not happen if matches() returned True
            logger.warning(
                f"MetadataParser called on line, but no pattern matched or data already parsed: {current_line.strip()}"
            )

        # No need to advance the iterator, parse handles only the current_line
        return

import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import DispersionCorrectionData, LineIterator, SectionParser, _MutableCalculationData
from calcflow.utils import logger

# --- Dispersion Correction Parser --- #
DISPERSION_START_PAT = re.compile(r"DFT DISPERSION CORRECTION")
DISPERSION_METHOD_PAT = re.compile(r"(DFTD\d V\d.*?)(?:\n|$)")  # Basic capture
DISPERSION_ENERGY_PAT = re.compile(r"Dispersion correction\s+(\S+)")


class DispersionParser(SectionParser):
    """Parses the dispersion correction block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return not current_data.parsed_dispersion and bool(DISPERSION_START_PAT.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting dispersion correction block parsing.")
        results.parsed_dispersion = True  # Mark as attempted immediately
        # Consume header lines (----)
        next(iterator, None)
        method: str | None = None
        energy: float | None = None

        try:
            for line in iterator:
                method_match = DISPERSION_METHOD_PAT.search(line)
                if method_match and method is None:  # Capture first method line encountered
                    method = method_match.group(1).strip()
                    # TODO: Refine method extraction (e.g., 'D3BJ', 'D4')
                    continue

                energy_match = DISPERSION_ENERGY_PAT.search(line)
                if energy_match:
                    try:
                        energy = float(energy_match.group(1))
                    except (ValueError, IndexError) as e:
                        raise ParsingError(f"Could not parse dispersion energy: {line.strip()}") from e
                    break  # Found energy, assume done

                # Define terminators
                if "FINAL SINGLE POINT ENERGY" in line or "TIMINGS" in line or line.strip() == "-" * 60:
                    # If we haven't found energy yet, log warning before breaking
                    if energy is None:
                        logger.warning(f"Exiting dispersion block prematurely due to terminator: '{line.strip()}'")
                    break

            if method is None or energy is None:
                missing = [n for n, v in [("Method", method), ("Energy", energy)] if v is None]
                raise ParsingError(f"Dispersion Correction block found but could not parse: {', '.join(missing)}")

            dispersion_data = DispersionCorrectionData(method=method, energy=energy)
            results.dispersion_correction = dispersion_data
            logger.debug(f"Successfully parsed dispersion correction: {repr(dispersion_data)}")

        except Exception as e:
            # If it's the specific error we raise for missing data, let it propagate
            if isinstance(e, ParsingError):
                raise
            # Otherwise, log unexpected errors and mark as attempted
            logger.error(f"Error parsing dispersion correction block: {e}", exc_info=True)
            # results.parsed_dispersion = True  # Already set at the start

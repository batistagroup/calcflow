import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import DispersionCorrectionData, LineIterator, _MutableCalculationData
from calcflow.utils import logger

# --- Dispersion Correction Parser --- #
DISPERSION_START_PAT = re.compile(r"DFT DISPERSION CORRECTION")
DISPERSION_METHOD_PAT = re.compile(r"(DFTD\d V\d.*?)(?:\n|$)")  # Basic capture
DISPERSION_ENERGY_PAT = re.compile(r"Dispersion correction\s+(-?\d+\.\d+)")


class DispersionParser:
    """Parses the dispersion correction block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return not current_data.parsed_dispersion and bool(DISPERSION_START_PAT.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting dispersion correction block parsing.")
        # Consume header lines (----)
        next(iterator, None)
        method: str | None = None
        energy_eh: float | None = None

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
                        energy_eh = float(energy_match.group(1))
                    except (ValueError, IndexError) as e:
                        raise ParsingError(f"Could not parse dispersion energy: {line.strip()}") from e
                    break  # Found energy, assume done

                # Define terminators
                if "FINAL SINGLE POINT ENERGY" in line or "TIMINGS" in line or line.strip() == "-" * 60:
                    # If we haven't found energy yet, log warning before breaking
                    if energy_eh is None:
                        logger.warning(f"Exiting dispersion block prematurely due to terminator: '{line.strip()}'")
                    break

            if method is None or energy_eh is None:
                missing = [n for n, v in [("Method", method), ("Energy", energy_eh)] if v is None]
                raise ParsingError(f"Dispersion Correction block found but could not parse: {', '.join(missing)}")

            dispersion_data = DispersionCorrectionData(method=method, energy_eh=energy_eh)
            results.dispersion_correction = dispersion_data
            results.parsed_dispersion = True
            logger.debug(f"Successfully parsed dispersion correction: {repr(dispersion_data)}")

        except Exception as e:
            logger.error(f"Error parsing dispersion correction block: {e}", exc_info=True)
            results.parsed_dispersion = True  # Mark as attempted

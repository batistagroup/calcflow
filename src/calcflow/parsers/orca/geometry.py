import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import Atom, LineIterator, _MutableCalculationData
from calcflow.utils import logger

# --- Geometry Parser --- #
GEOMETRY_START_PAT = re.compile(r"CARTESIAN COORDINATES \(ANGSTROEM\)")
GEOMETRY_LINE_PAT = re.compile(r"^\s*([A-Za-z]{1,3})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


class GeometryParser:
    """Parses the input geometry block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return not current_data.parsed_geometry and bool(GEOMETRY_START_PAT.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting geometry block parsing.")
        # Consume header lines (assumes at least one separator/blank line)
        next(iterator, None)
        geometry: list[Atom] = []
        try:
            for line in iterator:
                line_stripped = line.strip()
                if not line_stripped:
                    break  # Blank line marks the end

                match = GEOMETRY_LINE_PAT.match(line_stripped)
                if match:
                    symbol, x, y, z = match.groups()
                    geometry.append(Atom(symbol=symbol, x=float(x), y=float(y), z=float(z)))
                else:
                    logger.warning(f"Unexpected line format in geometry block: {line_stripped}")
                    break
            if not geometry:
                raise ParsingError("Geometry block found but no atoms could be parsed.")
            results.input_geometry = tuple(geometry)
            results.parsed_geometry = True
            logger.debug(f"Successfully parsed {len(geometry)} atoms.")
        except Exception as e:
            logger.error(f"Error parsing geometry block: {e}", exc_info=True)
            raise ParsingError("Failed to parse geometry block.") from e

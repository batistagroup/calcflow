import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import DipoleMoment, LineIterator, SectionParser, _MutableCalculationData
from calcflow.utils import logger

# --- Dipole Moment Parser --- #
DIPOLE_START_PAT = re.compile(r"DIPOLE MOMENT")
DIPOLE_TOTAL_LINE_PAT = re.compile(r"Total Dipole Moment\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
DIPOLE_MAG_AU_PAT = re.compile(r"Magnitude \(a\.u\.\)\s+:\s+(\d+\.\d+)")
DIPOLE_MAG_DEBYE_PAT = re.compile(r"Magnitude \(Debye\)\s+:\s+(\d+\.\d+)")


class DipoleParser(SectionParser):
    """Parses the dipole moment block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return not current_data.parsed_dipole and bool(DIPOLE_START_PAT.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting dipole moment block parsing.")
        # Consume header lines (----)
        next(iterator, None)
        x_au: float | None = None
        y_au: float | None = None
        z_au: float | None = None
        total_au: float | None = None
        magnitude: float | None = None

        try:
            for line in iterator:
                total_match = DIPOLE_TOTAL_LINE_PAT.search(line)
                if total_match:
                    try:
                        x_au, y_au, z_au = map(float, total_match.groups())
                    except (ValueError, IndexError) as e:
                        raise ParsingError(f"Could not parse dipole components: {line.strip()}") from e
                    continue

                mag_au_match = DIPOLE_MAG_AU_PAT.search(line)
                if mag_au_match:
                    try:
                        total_au = float(mag_au_match.group(1))
                    except (ValueError, IndexError) as e:
                        raise ParsingError(f"Could not parse dipole magnitude (a.u.): {line.strip()}") from e
                    continue

                mag_debye_match = DIPOLE_MAG_DEBYE_PAT.search(line)
                if mag_debye_match:
                    try:
                        magnitude = float(mag_debye_match.group(1))
                    except (ValueError, IndexError) as e:
                        raise ParsingError(f"Could not parse dipole magnitude (Debye): {line.strip()}") from e
                    continue

                if "Rotational spectrum" in line:
                    if None in [x_au, y_au, z_au, total_au, magnitude]:
                        logger.warning(f"Exiting dipole block prematurely due to terminator: '{line.strip()}'")
                    break

                if None not in [x_au, y_au, z_au, total_au, magnitude]:
                    logger.debug("All dipole components found, breaking loop.")
                    break

            if None in [x_au, y_au, z_au, total_au, magnitude]:
                missing = [
                    name
                    for name, val in zip(
                        ["X", "Y", "Z", "Mag(au)", "Mag(Debye)"],
                        [x_au, y_au, z_au, total_au, magnitude],
                        strict=False,
                    )
                    if val is None
                ]
                raise ParsingError(f"Dipole Moment block found but could not parse components: {', '.join(missing)}")

            assert x_au is not None and y_au is not None and z_au is not None
            assert total_au is not None and magnitude is not None

            dipole_data = DipoleMoment(x_au=x_au, y_au=y_au, z_au=z_au, total_au=total_au, magnitude=magnitude)
            results.dipole_moment = dipole_data
            results.parsed_dipole = True
            logger.debug(f"Successfully parsed dipole moment: {repr(dipole_data)}")

        except ParsingError:
            logger.error("ParsingError encountered during dipole moment parsing.", exc_info=True)
            results.parsed_dipole = True
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing dipole moment block: {e}", exc_info=True)
            results.parsed_dipole = True

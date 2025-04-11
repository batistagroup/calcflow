import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import DipoleMomentData, LineIterator, _MutableCalculationData
from calcflow.utils import logger

# --- Dipole Moment Parser --- #
DIPOLE_START_PAT = re.compile(r"DIPOLE MOMENT")
DIPOLE_TOTAL_LINE_PAT = re.compile(r"Total Dipole Moment\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
DIPOLE_MAG_AU_PAT = re.compile(r"Magnitude \(a\.u\.\)\s+:\s+(\d+\.\d+)")
DIPOLE_MAG_DEBYE_PAT = re.compile(r"Magnitude \(Debye\)\s+:\s+(\d+\.\d+)")


class DipoleParser:
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
        total_debye: float | None = None

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
                        total_debye = float(mag_debye_match.group(1))
                    except (ValueError, IndexError) as e:
                        raise ParsingError(f"Could not parse dipole magnitude (Debye): {line.strip()}") from e
                    continue  # Continue to check for terminators

                # Refined termination condition: only break on clear end markers
                # Removed 'or "--------" in line' check which caused premature exit
                if "Rotational spectrum" in line:
                    # If we haven't found all components yet, log a warning but break
                    if None in [x_au, y_au, z_au, total_au, total_debye]:
                        logger.warning(f"Exiting dipole block prematurely due to terminator: '{line.strip()}'")
                    break  # End of dipole section

                # Optional: Add a check to break if all values are found, just in case
                # the 'Rotational spectrum' line is missing in some outputs.
                if None not in [x_au, y_au, z_au, total_au, total_debye]:
                    logger.debug("All dipole components found, breaking loop.")
                    break

            if None in [x_au, y_au, z_au, total_au, total_debye]:
                missing = [
                    name
                    for name, val in zip(
                        ["X", "Y", "Z", "Mag(au)", "Mag(Debye)"],
                        [x_au, y_au, z_au, total_au, total_debye],
                        strict=False,
                    )
                    if val is None
                ]
                raise ParsingError(f"Dipole Moment block found but could not parse components: {', '.join(missing)}")

            assert x_au is not None and y_au is not None and z_au is not None
            assert total_au is not None and total_debye is not None

            dipole_data = DipoleMomentData(x_au=x_au, y_au=y_au, z_au=z_au, total_au=total_au, total_debye=total_debye)
            results.dipole_moment = dipole_data
            results.parsed_dipole = True
            logger.debug(f"Successfully parsed dipole moment: {repr(dipole_data)}")

        except Exception as e:
            logger.error(f"Error parsing dipole moment block: {e}", exc_info=True)
            results.parsed_dipole = True  # Mark as attempted

import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import LineIterator, Orbital, OrbitalData, SectionParser, _MutableCalculationData
from calcflow.utils import logger

# --- Orbitals Parser --- #
ORBITALS_START_PAT = re.compile(r"ORBITAL ENERGIES")
ORBITAL_LINE_PAT = re.compile(r"^\s*(\d+)\s+([\d\.]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
OCC_THRESHOLD = 0.1  # Threshold to consider an orbital occupied


class OrbitalsParser(SectionParser):
    """Parses the orbital energies block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return not current_data.parsed_orbitals and bool(ORBITALS_START_PAT.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting orbital energies block parsing.")
        # Consume header lines (---, blank)
        next(iterator, None)
        next(iterator, None)
        orbitals: list[Orbital] = []
        homo_index: int | None = None
        lumo_index: int | None = None
        last_occupied_index = -1

        try:
            for line in iterator:
                line_stripped = line.strip()
                if not line_stripped or "*Virtual orbitals printed" in line_stripped or "--------" in line_stripped:
                    break

                match = ORBITAL_LINE_PAT.match(line_stripped)
                if match:
                    try:
                        idx_str, occ_str, eh_str, ev_str = match.groups()
                        idx = int(idx_str)
                        occ = float(occ_str)
                        eh = float(eh_str)
                        ev = float(ev_str)
                        orbitals.append(Orbital(index=idx, occupation=occ, energy_eh=eh, energy_ev=ev))
                        if occ > OCC_THRESHOLD:
                            last_occupied_index = idx
                    except ValueError as e:
                        raise ParsingError(f"Could not parse orbital line values: {line_stripped}") from e
                elif orbitals:  # Stop if format breaks after finding some orbitals
                    break

            if not orbitals:
                raise ParsingError("Orbital energies block found but no orbitals could be parsed.")

            if last_occupied_index != -1:
                homo_index = last_occupied_index
                if homo_index + 1 < len(orbitals):
                    # Check index consistency
                    if orbitals[homo_index + 1].index == homo_index + 1:
                        lumo_index = homo_index + 1
                    else:
                        logger.warning(
                            f"LUMO index mismatch. Expected {homo_index + 1}, found {orbitals[homo_index + 1].index}"
                        )

            orbital_data = OrbitalData(orbitals=tuple(orbitals), homo_index=homo_index, lumo_index=lumo_index)
            results.orbitals = orbital_data
            results.parsed_orbitals = True
            logger.debug(f"Successfully parsed orbital data: {repr(orbital_data)}")

        except Exception as e:
            logger.error(f"Error parsing orbital energies block: {e}", exc_info=True)
            # Allow parsing to continue for other sections
            results.parsed_orbitals = True  # Mark as attempted

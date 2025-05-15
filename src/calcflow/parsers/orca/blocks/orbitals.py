import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import LineIterator, Orbital, OrbitalsSet, SectionParser, _MutableCalculationData
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

        # --- Loop for parsing orbital lines ---
        try:
            for line in iterator:
                line_stripped = line.strip()
                # Stop conditions for the loop
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
                        # Catch specific error during value conversion
                        raise ParsingError(f"Could not parse orbital line values: {line_stripped}") from e
                elif orbitals:  # Stop if format breaks *after* finding some orbitals
                    logger.warning(f"Non-orbital line encountered after parsing orbitals, stopping: {line_stripped}")
                    break
                # If line doesn't match and no orbitals parsed yet, continue searching (ignore unexpected lines)

        except ParsingError:
            # Re-raise specific parsing errors caught above
            raise
        except Exception as e:
            # Catch unexpected errors *during the loop*
            logger.error(f"Unexpected error parsing orbital lines: {e}", exc_info=True)
            results.parsed_orbitals = True  # Mark as attempted even if loop failed unexpectedly
            return  # Exit parsing for this section

        # --- Post-loop validation and processing ---
        if not orbitals:
            # Raise error if block was entered but no orbitals were successfully parsed
            raise ParsingError("Orbital energies block found but no orbitals could be parsed.")

        # Attempt to determine HOMO/LUMO and finalize data
        try:
            if last_occupied_index != -1:
                homo_index = last_occupied_index
                lumo_candidate_index = homo_index + 1  # Expected index
                if lumo_candidate_index < len(orbitals):
                    actual_lumo_index = orbitals[lumo_candidate_index].index
                    if actual_lumo_index == lumo_candidate_index:
                        lumo_index = lumo_candidate_index
                    else:
                        # Log if the index directly after HOMO isn't sequential
                        logger.warning(
                            f"LUMO index mismatch. Expected {lumo_candidate_index}, found {actual_lumo_index}"
                        )
                # If lumo_candidate_index >= len(orbitals), means HOMO was the last orbital, lumo_index remains None

            orbital_data = OrbitalsSet(orbitals=tuple(orbitals), homo_index=homo_index, lumo_index=lumo_index)
            results.orbitals = orbital_data
            results.parsed_orbitals = True
            logger.debug(f"Successfully parsed orbital data: {repr(orbital_data)}")

        except Exception as e:
            # Catch unexpected errors during HOMO/LUMO logic or OrbitalsSet creation
            logger.error(f"Error processing found orbitals or determining HOMO/LUMO: {e}", exc_info=True)
            results.parsed_orbitals = True  # Mark as attempted, even if post-processing failed

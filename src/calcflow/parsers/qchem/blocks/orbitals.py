import re
from typing import Final

from calcflow.parsers.qchem.typing import (
    LineIterator,
    Orbital,
    OrbitalData,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- Regex Patterns ---
ORBITAL_HEADER_PAT: Final[re.Pattern[str]] = re.compile(r"^\s*Orbital Energies \(a\.u\.\)")
SPIN_SECTION_PAT: Final[re.Pattern[str]] = re.compile(r"^\s*(Alpha|Beta) MOs", re.IGNORECASE)
OCCUPIED_MARKER: Final[str] = "-- Occupied --"
VIRTUAL_MARKER: Final[str] = "-- Virtual --"
ENERGY_LINE_PAT: Final[re.Pattern[str]] = re.compile(r"^\s*(-?\d+\.\d+(\s+|$))+")
SECTION_SEPARATOR: Final[str] = "---"


class OrbitalParser(SectionParser):
    """Parses the molecular orbital energy section into OrbitalData."""

    def matches(self, line: str, data: _MutableCalculationData | None = None) -> bool:
        """Check if the line matches the orbital energy section header."""
        # Need data to check if orbitals already parsed
        if data and data.parsed_orbitals:
            return False
        return bool(ORBITAL_HEADER_PAT.search(line))

    def parse(self, iterator: LineIterator, current_line: str, data: _MutableCalculationData) -> None:
        """Parses the orbital energy block, populating data.orbitals."""
        if data.parsed_orbitals:
            logger.warning("Attempting to parse Orbital Energies section again. Skipping.")
            # Consume the block if necessary, or rely on outer loop structure
            return

        logger.debug("Parsing Orbital Energies section.")

        # Initialize OrbitalData structure
        is_unrestricted = str(data.rem.get("unrestricted", "false")).lower() == "true"
        alpha_orbitals_list: list[Orbital] = []
        beta_orbitals_list: list[Orbital] | None = [] if is_unrestricted else None

        current_spin_type: str | None = None
        parsing_occupied: bool = False
        parsing_virtual: bool = False

        # Orbital counters
        alpha_idx_counter = 0
        beta_idx_counter = 0

        # First line after header is often separator or blank
        # We start iterating *within* the block content

        line_iterator_with_start = iter([current_line] + list(iterator))
        # Skip the header line itself which triggered the match
        try:
            next(line_iterator_with_start)
        except StopIteration:
            logger.warning("Orbital section header found at end of file.")
            return

        for line_num, line in enumerate(line_iterator_with_start, start=1):  # Relative line number within block
            line_stripped = line.strip()

            # End conditions for the orbital block
            if not line_stripped or line_stripped.startswith(SECTION_SEPARATOR):
                # Allow one separator after header, break on subsequent ones/empty lines
                if line_num > 2:  # Simple heuristic
                    logger.debug(f"End of Orbital Energies section detected near relative line {line_num}.")
                    break
                else:
                    continue  # Skip initial separators/blank lines

            spin_match = SPIN_SECTION_PAT.search(line_stripped)
            if spin_match:
                current_spin_type = spin_match.group(1).lower()
                parsing_occupied = False  # Reset state for new spin type
                parsing_virtual = False
                logger.debug(f"Detected spin type: {current_spin_type}")
                continue

            # Infer spin type for restricted calculations if header not found yet
            if current_spin_type is None:
                if not is_unrestricted:
                    current_spin_type = "alpha"
                    logger.debug("Assuming 'alpha' spin type for restricted calculation.")
                else:
                    # In unrestricted, wait for explicit Alpha/Beta header
                    logger.warning(
                        f"Line '{line_stripped}' encountered before spin type identified in unrestricted calc. Skipping."
                    )
                    continue

            if OCCUPIED_MARKER in line_stripped:
                parsing_occupied = True
                parsing_virtual = False
                logger.debug(f"Parsing occupied {current_spin_type} orbitals.")
                continue
            elif VIRTUAL_MARKER in line_stripped:
                parsing_occupied = False
                parsing_virtual = True
                logger.debug(f"Parsing virtual {current_spin_type} orbitals.")
                continue

            # Only parse energies if we are clearly inside occupied or virtual sections
            if parsing_occupied or parsing_virtual:
                energy_match = ENERGY_LINE_PAT.search(line_stripped)
                if energy_match:
                    try:
                        energies = [float(e) for e in energy_match.group(0).split()]
                        target_list: list[Orbital] | None
                        idx_counter: int

                        if current_spin_type == "alpha":
                            target_list = alpha_orbitals_list
                            idx_counter = alpha_idx_counter
                        elif current_spin_type == "beta" and beta_orbitals_list is not None:
                            target_list = beta_orbitals_list
                            idx_counter = beta_idx_counter
                        else:
                            logger.warning(
                                f"Parsed beta energies but calculation seems restricted or list not initialized: {line_stripped}"
                            )
                            continue  # Skip these energies

                        for energy in energies:
                            orbital = Orbital(index=idx_counter, energy_eh=energy)
                            target_list.append(orbital)
                            idx_counter += 1

                        # Update the main counters
                        if current_spin_type == "alpha":
                            alpha_idx_counter = idx_counter
                        elif current_spin_type == "beta":
                            beta_idx_counter = idx_counter

                        logger.debug(f"Parsed {len(energies)} {current_spin_type} energies: {energies}")

                    except ValueError:
                        logger.warning(f"Could not parse float from energy line: '{line_stripped}'")
                        # Decide whether to continue or raise based on robustness needs
                # else: Line doesn't contain energies, could be other info within the block
                # logger.debug(f"Non-energy line encountered in {current_spin_type} { 'occupied' if parsing_occupied else 'virtual'}: '{line_stripped}'")
            # else: # We are between sections (e.g., after spin header but before occupied/virtual marker)
            # logger.debug(f"Line outside occupied/virtual section: '{line_stripped}'")

        # --- Final Processing --- #
        if not alpha_orbitals_list and (not beta_orbitals_list or not is_unrestricted):
            data.parsing_warnings.append("No orbital energies were parsed.")
            logger.warning("No orbital energies were parsed from the Orbital Energies section.")
        else:
            # Assign the parsed lists to the main data structure
            data.orbitals = OrbitalData(
                alpha_orbitals=alpha_orbitals_list,
                beta_orbitals=beta_orbitals_list,
                # HOMO/LUMO indices can be calculated later if needed
            )
            data.parsed_orbitals = True  # Mark as parsed
            logger.debug(
                f"Finished parsing Orbital Energies section. Found {len(alpha_orbitals_list)} alpha MOs"
                + (f" and {len(beta_orbitals_list)} beta MOs." if beta_orbitals_list else ".")
            )

import re
from typing import Final

from calcflow.exceptions import ParsingError
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
# Use the long dashed line as the definitive end marker
BLOCK_END_SEPARATOR_PAT = re.compile(r"^\s*-{60,}")
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

        # The current_line is the header, we start processing lines *after* it.
        # Consume the separator line immediately following the header.
        try:
            line_after_header = next(iterator)
            if not BLOCK_END_SEPARATOR_PAT.search(line_after_header.strip()):
                logger.warning(
                    f"Expected separator line '------' after Orbital Energies header, got: '{line_after_header.strip()}'. Continuing parsing."
                )
                # Decide how to handle this - maybe try parsing line_after_header? For now, log and continue.
        except StopIteration:
            logger.warning("File ended immediately after Orbital Energies header.")
            data.parsing_warnings.append("Incomplete Orbital Energies section (ended after header).")
            data.parsed_orbitals = False  # Mark as not parsed
            return  # Nothing more to parse

        line_num = 0  # Relative line number within the block, starting after the separator
        try:
            while True:  # Loop until break or StopIteration
                line = next(iterator)
                line_num += 1
                line_stripped = line.strip()

                # End condition check: Only the long dashed line terminates the block
                if BLOCK_END_SEPARATOR_PAT.search(line_stripped):
                    logger.debug(
                        f"End of Orbital Energies section detected at block line {line_num} ('{line_stripped}')."
                    )
                    break  # Exit the while loop

                # Skip blank lines explicitly
                if not line_stripped:
                    logger.debug(f"Skipping blank line {line_num}")
                    continue

                # Check for spin section header (Alpha/Beta MOs)
                spin_match = SPIN_SECTION_PAT.search(line_stripped)
                if spin_match:
                    current_spin_type = spin_match.group(1).lower()
                    parsing_occupied = False
                    parsing_virtual = False
                    logger.debug(f"Detected spin type: {current_spin_type}")
                    continue

                # Infer spin type for restricted calculations if header not found yet
                if current_spin_type is None:
                    if not is_unrestricted:
                        current_spin_type = "alpha"
                        logger.debug("Assuming 'alpha' spin type for restricted calculation.")
                    else:
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
                        # Validate all parts of the line *before* processing
                        parsed_energies_on_line: list[float] = []
                        try:
                            # Attempt to convert all non-empty parts
                            parts = line_stripped.split()
                            parsed_energies_on_line = [float(p) for p in parts]
                        except ValueError as e:
                            # Treat malformed line as a fatal error
                            logger.error(
                                f"Malformed energy line encountered: '{line_stripped}'. Aborting orbital parsing."
                            )
                            data.parsing_errors.append(f"Malformed energy line: '{line_stripped}'")
                            raise ParsingError(f"Malformed orbital energy line: '{line_stripped}'") from e
                            # No need for all_parts_valid anymore or continue

                        # If we reach here, all parts were valid floats
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
                            continue

                        for energy in parsed_energies_on_line:
                            orbital = Orbital(index=idx_counter, energy_eh=energy)
                            target_list.append(orbital)
                            idx_counter += 1

                        if current_spin_type == "alpha":
                            alpha_idx_counter = idx_counter
                        elif current_spin_type == "beta":
                            beta_idx_counter = idx_counter

                        logger.debug(
                            f"Parsed {len(parsed_energies_on_line)} {current_spin_type} energies: {parsed_energies_on_line}"
                        )

        except StopIteration:
            # File ended while parsing orbitals. This might be okay if it's the last section.
            logger.warning("File ended during Orbital Energies parsing.")
            # Also record this warning in the results data
            data.parsing_warnings.append("File ended during Orbital Energies parsing.")
            # Allow processing of orbitals found so far.

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

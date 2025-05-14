# Placeholder for TDDFT Block Parsers

import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.typing import (
    ExcitedStateProperties,
    LineIterator,
    OrbitalTransition,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- TDDFT/TDA Excitation Energies Block ---
TDA_HEADER_PAT = re.compile(r"^\s+TDDFT/TDA Excitation Energies\s*$")
TDDFT_HEADER_PAT = re.compile(r"^\s+TDDFT Excitation Energies\s*$")
EXCITED_STATE_PAT = re.compile(r"^ Excited state\s+(\d+): excitation energy \(eV\) =\s+(-?\d+\.\d+)")
TOTAL_ENERGY_PAT = re.compile(r"^ Total energy for state\s+\d+:\s+(-?\d+\.\d+) au")
MULTIPLICITY_PAT = re.compile(r"^\s+Multiplicity: (Singlet|Triplet)")
TRANS_MOM_PAT = re.compile(r"^\s+Trans\. Mom\.:\s+(-?\d+\.\d+) X\s+(-?\d+\.\d+) Y\s+(-?\d+\.\d+) Z")
STRENGTH_PAT = re.compile(r"^\s+Strength\s+:\s+(-?\d+\.\d+)")
AMPLITUDE_PAT = re.compile(r"^\s+(X: )?D\(\s*(\d+)\) --> V\(\s*(\d+)\) amplitude =\s+(-?\d+\.\d+)")
END_OF_EXCITATION_BLOCK_PAT = re.compile(r"^ -{3,}$")  # End of individual excitation block


class TDDFTExcitationEnergiesParser(SectionParser):
    """Parses TDDFT/TDA and TDDFT excitation energy blocks."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line matches the start of a TDDFT excitation block."""
        # Avoid re-parsing if already done, though one file can have TDA and TDDFT sections
        is_tda = TDA_HEADER_PAT.search(line) is not None
        is_tddft = TDDFT_HEADER_PAT.search(line) is not None

        if is_tda and not current_data.parsed_tda_excitations:
            return True
        return is_tddft and not current_data.parsed_tddft_excitations

    def parse(self, iterator: LineIterator, first_line: str, results: _MutableCalculationData) -> None:
        """Extract all excited states from the TDDFT or TDA block."""
        is_tda_block = TDA_HEADER_PAT.search(first_line) is not None
        # is_tddft_block = TDDFT_HEADER_PAT.search(first_line) is not None # Redundant with else

        target_list = results.tda_excited_states_list if is_tda_block else results.tddft_excited_states_list
        block_type = "TDA" if is_tda_block else "TDDFT"
        logger.debug(f"Starting parsing of {block_type} excitation energies block.")

        # Consume the header line already processed by matches/first_line
        # The iterator is now at the line AFTER the header.

        line_num = 0  # Relative to start of block for logging
        try:
            while True:  # Loop over excited states within this block
                line_num += 1
                try:
                    line = next(iterator)
                except StopIteration:
                    logger.warning(f"{block_type} block ended before '---' terminator.")
                    break

                if END_OF_EXCITATION_BLOCK_PAT.search(line):
                    logger.debug(f"Found end of {block_type} excitation block.")
                    break  # End of the entire "TDDFT/TDA Excitation Energies" or "TDDFT Excitation Energies" section

                match_es = EXCITED_STATE_PAT.search(line)
                if not match_es:
                    # This might be an empty line or some other unexpected format
                    if line.strip():  # Log if not just an empty line
                        logger.warning(f"Unexpected line in {block_type} block: {line.strip()}")
                    continue

                state_number = int(match_es.group(1))
                exc_energy_ev = float(match_es.group(2))

                # Read subsequent lines for this specific state
                total_energy_au: float | None = None
                multiplicity: str | None = None
                trans_mom_x, trans_mom_y, trans_mom_z = None, None, None
                osc_strength: float | None = None
                transitions: list[OrbitalTransition] = []

                # Loop for details of a single excited state
                for _i_prop in range(10):  # Max lines to look for properties for one state
                    line_num += 1
                    try:
                        line = next(iterator)
                    except StopIteration as e:
                        raise ParsingError(
                            f"File ended unexpectedly while parsing state {state_number} in {block_type} block."
                        ) from e

                    if EXCITED_STATE_PAT.search(line) or END_OF_EXCITATION_BLOCK_PAT.search(line):
                        # Found start of next state or end of block, push back and break
                        iterator = iter([line, *iterator])  # Prepend line back
                        line_num -= 1  # Adjust because we didn't consume this line
                        break

                    match_tot_e = TOTAL_ENERGY_PAT.search(line)
                    if match_tot_e:
                        total_energy_au = float(match_tot_e.group(1))
                        continue

                    match_mult = MULTIPLICITY_PAT.search(line)
                    if match_mult:
                        multiplicity = match_mult.group(1)
                        continue

                    match_tm = TRANS_MOM_PAT.search(line)
                    if match_tm:
                        trans_mom_x = float(match_tm.group(1))
                        trans_mom_y = float(match_tm.group(2))
                        trans_mom_z = float(match_tm.group(3))
                        continue

                    match_str = STRENGTH_PAT.search(line)
                    if match_str:
                        osc_strength = float(match_str.group(1))
                        continue

                    match_amp = AMPLITUDE_PAT.search(line)
                    if match_amp:
                        # Group 1 is "X: " or None
                        # Group 2 is from_orb_idx
                        # Group 3 is to_orb_idx
                        # Group 4 is amplitude
                        from_idx = int(match_amp.group(2))
                        to_idx = int(match_amp.group(3))
                        amp_val = float(match_amp.group(4))
                        transitions.append(
                            OrbitalTransition(
                                from_orbital_type="D",  # Assuming Donor from "D(...)"
                                from_orbital_index=from_idx,
                                to_orbital_type="V",  # Assuming Virtual from "V(...)"
                                to_orbital_index=to_idx,
                                amplitude=amp_val,
                            )
                        )
                        continue

                    if not line.strip():  # Skip empty lines silently
                        continue

                    # If we reach here, the line is not recognized within an excited state property block
                    logger.warning(f"Unparsed line in state {state_number} of {block_type} block: {line.strip()}")

                if total_energy_au is None or multiplicity is None:
                    raise ParsingError(
                        f"Missing essential data (total energy or multiplicity) for state {state_number} in {block_type}."
                    )

                target_list.append(
                    ExcitedStateProperties(
                        state_number=state_number,
                        excitation_energy_ev=exc_energy_ev,
                        total_energy_au=total_energy_au,
                        multiplicity=multiplicity,
                        trans_moment_x=trans_mom_x,
                        trans_moment_y=trans_mom_y,
                        trans_moment_z=trans_mom_z,
                        oscillator_strength=osc_strength,
                        transitions=transitions,
                    )
                )

            if is_tda_block:
                results.parsed_tda_excitations = True
            else:
                results.parsed_tddft_excitations = True

        except ParsingError as e:  # Catch specific parsing errors from this block
            logger.error(f"Error parsing {block_type} excitations near line ~{line_num}: {e}")
            raise  # Re-raise to be caught by the main loop
        except Exception as e:  # Catch unexpected errors
            logger.error(f"Unexpected error in {block_type}ExcitationParser near line ~{line_num}: {e}", exc_info=True)
            raise ParsingError(f"Critical error in {block_type}ExcitationParser.") from e

        logger.info(f"Successfully parsed {len(target_list)} states from {block_type} block.")

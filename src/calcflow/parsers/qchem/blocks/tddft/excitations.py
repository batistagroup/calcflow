import re
from typing import Literal, cast

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.typing import (
    ExcitedStateProperties,
    LineIterator,
    OrbitalTransition,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- TDDFT/TDA Excitation Energies Block Patterns ---
TDA_HEADER_PAT = re.compile(r"^\s*TDDFT/TDA\s+Excitation\s+Energies\s*$")
TDDFT_HEADER_PAT = re.compile(r"^\s*TDDFT\s+Excitation\s+Energies\s*$")
EXCITED_STATE_PAT = re.compile(r"^ Excited state\s+(\d+): excitation energy \(eV\) =\s+(-?\d+\.\d+)")
TOTAL_ENERGY_PAT = re.compile(r"^ Total energy for state\s+\d+:\s+(-?\d+\.\d+) au")
MULTIPLICITY_PAT = re.compile(r"^\s+Multiplicity: (Singlet|Triplet)")
TRANS_MOM_PAT = re.compile(r"^\s+Trans\. Mom\.:\s+(-?\d+\.\d+) X\s+(-?\d+\.\d+) Y\s+(-?\d+\.\d+) Z")
STRENGTH_PAT = re.compile(r"^\s+Strength\s+:\s+(-?\d+\.\d+)")
AMPLITUDE_PAT = re.compile(r"^\s+(X: )?D\(\s*(\d+)\) --> V\(\s*(\d+)\) amplitude =\s+(-?\d+\.\d+)")
# Corrected pattern to handle leading spaces and at least 3 hyphens
END_OF_EXCITATION_BLOCK_PAT = re.compile(r"^\s*-{3,}$")


def _parse_excited_states_data(
    iterator: LineIterator,
    block_type: str,
    target_list: list[ExcitedStateProperties],
) -> None:
    """
    Helper function to parse the actual excited states data once a block is identified.
    Consumes lines from the iterator and populates the target_list.
    Raises ParsingError if critical issues are found.
    """
    line_num_in_block = 0  # Relative line number for logging within this helper
    pushed_back_line: str | None = None  # Variable to hold a pushed-back line

    try:
        while True:  # Loop over excited states within this block
            current_line_for_state_header: str
            if pushed_back_line is not None:
                current_line_for_state_header = pushed_back_line
                pushed_back_line = None
                # This line was already accounted for in line_num_in_block when first read
            else:
                line_num_in_block += 1
                try:
                    current_line_for_state_header = next(iterator)
                except StopIteration:
                    # This means the file/iterator ended when we expected start of a new state or block terminator
                    if (
                        line_num_in_block == 1 and not target_list
                    ):  # Check if this is the very first attempt to read data
                        logger.debug(f"{block_type} data block appears empty after header/separator.")
                    else:
                        logger.warning(
                            f"{block_type} data ended before '---' terminator pattern (line ~{line_num_in_block})."
                        )
                    break  # End of file reached before block terminator

            if END_OF_EXCITATION_BLOCK_PAT.search(current_line_for_state_header):
                logger.debug(f"Found end of {block_type} excitation data block at relative line ~{line_num_in_block}.")
                break  # Expected end of this specific excitation block

            match_es = EXCITED_STATE_PAT.search(current_line_for_state_header)
            if not match_es:
                if current_line_for_state_header.strip():  # Log if not just an empty line
                    logger.warning(
                        f"Unexpected line in {block_type} data section (line ~{line_num_in_block}), expected state data or '---': {current_line_for_state_header.strip()}"
                    )
                # If it wasn't a state header or the end pattern, and not empty,
                # it might be an empty line between states or an issue. We consume and try next.
                continue

            state_number = int(match_es.group(1))
            exc_energy_ev = float(match_es.group(2))
            logger.debug(f"Parsing properties for {block_type} State {state_number} (line ~{line_num_in_block})")

            total_energy_au: float | None = None
            multiplicity: str | None = None
            trans_mom_x, trans_mom_y, trans_mom_z = None, None, None
            osc_strength: float | None = None
            transitions: list[OrbitalTransition] = []

            # Loop for details of a single excited state
            for _i_prop in range(10):  # Max lines to look for properties for one state
                # We are looking for properties of the current state_number
                # line_num_in_block was for the state header, now for property line
                line_num_in_block += 1
                try:
                    current_prop_line = next(iterator)
                except StopIteration as e:
                    raise ParsingError(
                        f"File ended unexpectedly while parsing properties for state {state_number} in {block_type} block (line ~{line_num_in_block})."
                    ) from e

                # Check if this line is the start of the next state or end of the entire block
                if EXCITED_STATE_PAT.search(current_prop_line) or END_OF_EXCITATION_BLOCK_PAT.search(current_prop_line):
                    pushed_back_line = current_prop_line  # Store for the outer while loop
                    line_num_in_block -= (
                        1  # This line is not consumed for current state's props, nor by outer loop yet.
                    )
                    logger.debug(
                        f"Property scan for state {state_number} ended; found next state/block end. Pushing back: '{pushed_back_line.strip()}'"
                    )
                    break  # Exit property loop, outer loop will process pushed_back_line

                match_tot_e = TOTAL_ENERGY_PAT.search(current_prop_line)
                if match_tot_e:
                    total_energy_au = float(match_tot_e.group(1))
                    continue

                match_mult = MULTIPLICITY_PAT.search(current_prop_line)
                if match_mult:
                    multiplicity = match_mult.group(1)
                    continue

                match_tm = TRANS_MOM_PAT.search(current_prop_line)
                if match_tm:
                    trans_mom_x = float(match_tm.group(1))
                    trans_mom_y = float(match_tm.group(2))
                    trans_mom_z = float(match_tm.group(3))
                    continue

                match_str = STRENGTH_PAT.search(current_prop_line)
                if match_str:
                    osc_strength = float(match_str.group(1))
                    continue

                match_amp = AMPLITUDE_PAT.search(current_prop_line)
                if match_amp:
                    from_idx = int(match_amp.group(2))
                    to_idx = int(match_amp.group(3))
                    amp_val = float(match_amp.group(4))
                    transitions.append(
                        OrbitalTransition(
                            from_label="D",
                            from_idx=from_idx,
                            to_label="V",
                            to_idx=to_idx,
                            amplitude=amp_val,
                        )
                    )
                    continue

                if not current_prop_line.strip():  # Skip empty lines silently
                    continue

                logger.warning(
                    f"Unparsed line in state {state_number} of {block_type} data (line ~{line_num_in_block}): {current_prop_line.strip()}"
                )

            if total_energy_au is None or multiplicity is None:
                raise ParsingError(
                    f"Missing essential data (total energy or multiplicity) for state {state_number} in {block_type} (line ~{line_num_in_block})."
                )

            target_list.append(
                ExcitedStateProperties(
                    state_number=state_number,
                    excitation_energy_ev=exc_energy_ev,
                    total_energy_au=total_energy_au,
                    multiplicity=cast(Literal["Singlet", "Triplet"], multiplicity),
                    trans_moment_x=trans_mom_x,
                    trans_moment_y=trans_mom_y,
                    trans_moment_z=trans_mom_z,
                    oscillator_strength=osc_strength,
                    transitions=transitions,
                )
            )
    except ParsingError as e:
        logger.error(f"Error parsing {block_type} states data near relative line ~{line_num_in_block}: {e}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in _parse_excited_states_data for {block_type} near relative line ~{line_num_in_block}: {e}",
            exc_info=True,
        )
        raise ParsingError(f"Critical error during {block_type} state parsing.") from e


class TDAExcitationEnergiesParser(SectionParser):
    """Parses TDDFT/TDA Excitation Energies block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return TDA_HEADER_PAT.search(line) is not None and not current_data.parsed_tda_excitations

    def parse(self, iterator: LineIterator, first_line: str, results: _MutableCalculationData) -> None:
        block_type = "TDA"
        target_list = results.tda_excited_states_list
        logger.debug(f"Starting parsing of {block_type} excitation energies block header: {first_line.strip()}")

        line_after_header: str | None = None
        try:
            line_after_header = next(iterator)
        except StopIteration:
            logger.warning(f"{block_type} block ended prematurely after header (missing separator/data).")
            results.parsed_tda_excitations = True  # Mark as handled
            return

        if END_OF_EXCITATION_BLOCK_PAT.search(line_after_header):
            logger.debug(f"Consumed separator line for {block_type}: {line_after_header.strip()}")
            # Separator found, _parse_excited_states_data will read from the line *after* this separator.
            _parse_excited_states_data(iterator, block_type, target_list)
        else:
            # Line after header was NOT "---". It could be data or something else.
            logger.warning(
                f"Expected '---' separator after {block_type} header, but found: '{line_after_header.strip()}'. "
                f"Attempting to parse data starting with this line."
            )

            def _generate_prefixed_lines(first_data_line: str, remaining_iterator: LineIterator) -> LineIterator:
                yield first_data_line
                yield from remaining_iterator

            prefixed_iterator = _generate_prefixed_lines(line_after_header, iterator)
            _parse_excited_states_data(prefixed_iterator, block_type, target_list)

        results.parsed_tda_excitations = True
        logger.info(f"Successfully parsed {len(target_list)} states from {block_type} block.")


class TDDFTExcitationEnergiesParser(SectionParser):
    """Parses TDDFT Excitation Energies block (distinct from TDA/TDDFT)."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return TDDFT_HEADER_PAT.search(line) is not None and not current_data.parsed_tddft_excitations

    def parse(self, iterator: LineIterator, first_line: str, results: _MutableCalculationData) -> None:
        block_type = "TDDFT"
        target_list = results.tddft_excited_states_list
        logger.debug(f"Starting parsing of {block_type} excitation energies block header: {first_line.strip()}")

        line_after_header: str | None = None
        try:
            line_after_header = next(iterator)
        except StopIteration:
            logger.warning(f"{block_type} block ended prematurely after header (missing separator/data).")
            results.parsed_tddft_excitations = True  # Mark as handled
            return

        if END_OF_EXCITATION_BLOCK_PAT.search(line_after_header):
            logger.debug(f"Consumed separator line for {block_type}: {line_after_header.strip()}")
            # Separator found, _parse_excited_states_data will read from the line *after* this separator.
            _parse_excited_states_data(iterator, block_type, target_list)
        else:
            # Line after header was NOT "---". It could be data or something else.
            logger.warning(
                f"Expected '---' separator after {block_type} header, but found: '{line_after_header.strip()}'. "
                f"Attempting to parse data starting with this line."
            )

            def _generate_prefixed_lines(first_data_line: str, remaining_iterator: LineIterator) -> LineIterator:
                yield first_data_line
                yield from remaining_iterator

            prefixed_iterator = _generate_prefixed_lines(line_after_header, iterator)
            _parse_excited_states_data(prefixed_iterator, block_type, target_list)

        results.parsed_tddft_excitations = True
        logger.info(f"Successfully parsed {len(target_list)} states from {block_type} block.")

import re
from typing import Literal, cast

from calcflow.parsers.qchem.typing import (
    LineIterator,
    NTOContribution,
    NTOStateAnalysis,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger


class NTODecompositionParser(SectionParser):
    """Parses the SA-NTO Decomposition section from Q-Chem output."""

    SECTION_START_TOKEN = "SA-NTO Decomposition"
    SECTION_END_TOKEN = "================================================================================"
    STATE_HEADER_PATTERN = re.compile(r"^\s*(Singlet|Triplet|Multiplet)\s*(\d+)\s*:")
    NTO_CONTRIBUTION_PATTERN = re.compile(
        r"""
        ^\s*
        ([HV])           # (1) hole reference letter
        ([+-])\s*        # (2) hole sign
        (\d+)            # (3) hole digits
        \s*->\s*
        ([LV])           # (4) electron reference letter
        ([+-])\s*        # (5) electron sign
        (\d+)            # (6) electron digits
        \s*:\s*
        (-?\d+\.\d+)     # (7) coefficient
        \s*\(\s*
        (\d+\.\d+)       # (8) percent
        \s*%\s*\)\s*$
    """,
        re.VERBOSE,
    )

    OMEGA_PATTERN = re.compile(
        r"^\s*omega\s*=\s*(\d+\.\d+)\s*%\s*$",
        re.IGNORECASE,
    )

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line marks the beginning of the SA-NTO decomposition section."""
        return self.SECTION_START_TOKEN in line and not current_data.parsed_sa_nto_decomposition

    def parse(self, iterator: LineIterator, first_line: str, results: _MutableCalculationData) -> None:
        """Parse the SA-NTO Decomposition section."""
        logger.debug("Starting SA-NTO Decomposition parsing.")

        # Consume the section header line if it's the first_line, or advance iterator
        if self.SECTION_START_TOKEN not in first_line:
            # This case should ideally not be hit if matches() is called correctly
            # but as a safeguard:
            line = first_line
            if self.SECTION_START_TOKEN not in line:
                logger.warning(f"NTO Parser expected start token '{self.SECTION_START_TOKEN}' but got: {line.strip()}")
                results.parsing_errors.append(f"NTO Parser: Expected start token, got: {line.strip()}")
                # Cannot proceed if the first line isn't the header.
                # According to spec, don't push back if it's the passed 'first_line'.
                return
        # Consume the decorative line after the header, e.g., "----------------------------------------------------------------------------"
        try:
            line = next(iterator)
            if "---" not in line:  # Simple check for the separator
                logger.warning(f"NTO Parser: Expected separator line after header, got: {line.strip()}")
                results.parsing_warnings.append(f"NTO Parser: Missing separator after header: {line.strip()}")
                # Attempt to continue, but this line might be important for the next parser
                # if it's not part of this section.
                results.buffered_line = line
                results.parsed_sa_nto_decomposition = True  # Mark as attempted
                return

        except StopIteration:
            logger.warning("File ended unexpectedly after NTO section header.")
            results.parsing_warnings.append("File ended unexpectedly after NTO section header.")
            results.parsed_sa_nto_decomposition = True
            return

        current_state_analysis: NTOStateAnalysis | None = None
        contributions_list: list[NTOContribution] = []

        while True:
            try:
                line = next(iterator)
            except StopIteration:
                logger.warning("File ended unexpectedly while parsing SA-NTO Decomposition.")
                results.parsing_warnings.append("File ended unexpectedly during SA-NTO Decomposition parsing.")
                if current_state_analysis:
                    # Finalize the last state if file ends
                    updated_contributions = tuple(contributions_list)
                    final_state = NTOStateAnalysis(
                        state_number=current_state_analysis.state_number,
                        multiplicity=current_state_analysis.multiplicity,
                        contributions=updated_contributions,
                        omega_percent=current_state_analysis.omega_percent,  # May be None if file ended early
                    )
                    results.nto_state_analyses_list.append(final_state)
                results.parsed_sa_nto_decomposition = True
                return

            if self.SECTION_END_TOKEN in line:
                logger.debug("Found SA-NTO Decomposition section end token.")
                if current_state_analysis:
                    # Finalize the last parsed state before exiting
                    updated_contributions = tuple(contributions_list)
                    final_state = NTOStateAnalysis(
                        state_number=current_state_analysis.state_number,
                        multiplicity=current_state_analysis.multiplicity,
                        contributions=updated_contributions,
                        omega_percent=current_state_analysis.omega_percent,
                    )
                    results.nto_state_analyses_list.append(final_state)
                results.parsed_sa_nto_decomposition = True
                return

            state_match = self.STATE_HEADER_PATTERN.match(line)
            if state_match:
                if current_state_analysis:
                    # Finalize the previous state
                    updated_contributions = tuple(contributions_list)
                    prev_state = NTOStateAnalysis(
                        state_number=current_state_analysis.state_number,
                        multiplicity=current_state_analysis.multiplicity,
                        contributions=updated_contributions,
                        omega_percent=current_state_analysis.omega_percent,
                    )
                    results.nto_state_analyses_list.append(prev_state)
                    contributions_list = []  # Reset for new state

                multiplicity_str = state_match.group(1)
                state_number = int(state_match.group(2))
                logger.debug(f"Parsing NTOs for {multiplicity_str} state {state_number}")
                current_state_analysis = NTOStateAnalysis(
                    state_number=state_number,
                    multiplicity=multiplicity_str,
                    contributions=[],  # placeholder
                    omega_percent=None,
                )
                # Consume the "Decomposition into state-averaged NTOs" line and the "----" separator
                try:
                    next(iterator)  # "Decomposition..."
                    next(iterator)  # "----"
                except StopIteration:
                    logger.warning(f"File ended unexpectedly after state header for {multiplicity_str} {state_number}.")
                    results.parsing_warnings.append(
                        f"File ended after state header for {multiplicity_str} {state_number}."
                    )
                    if current_state_analysis:  # Save the started state
                        results.nto_state_analyses_list.append(current_state_analysis)  # contributions will be empty
                    results.parsed_sa_nto_decomposition = True
                    return
                continue

            contribution_match = self.NTO_CONTRIBUTION_PATTERN.match(line)
            if contribution_match:
                if not current_state_analysis:
                    logger.warning(f"Found NTO contribution without a state header: {line.strip()}")
                    results.parsing_warnings.append(f"Orphaned NTO contribution: {line.strip()}")
                    continue  # Skip this line

                hole_ref = contribution_match.group(1)  # "H" or "V"
                assert hole_ref in ["H", "V"]
                hole_sign = contribution_match.group(2)  # "+" or "-"
                hole_num = contribution_match.group(3)  # e.g. "2"
                elec_ref = contribution_match.group(4)  # "L" or "V"
                assert elec_ref in ["L", "V"]
                elec_sign = contribution_match.group(5)  # "+" or "-"
                elec_num = contribution_match.group(6)  # e.g. "3"
                coeff = float(contribution_match.group(7))
                weight = float(contribution_match.group(8))

                hole_offset = int(hole_sign + hole_num)  # e.g. "-2"
                elec_offset = int(elec_sign + elec_num)  # e.g. "+3"

                contribution = NTOContribution(
                    hole_reference=cast(Literal["H", "V"], hole_ref),
                    hole_offset=hole_offset,
                    electron_reference=cast(Literal["L", "V"], elec_ref),
                    electron_offset=elec_offset,
                    coefficient=coeff,
                    weight_percent=weight,
                )
                contributions_list.append(contribution)
                continue

            omega_match = self.OMEGA_PATTERN.match(line)
            if omega_match:
                if not current_state_analysis:
                    logger.warning(f"Found omega value without a state header: {line.strip()}")
                    results.parsing_warnings.append(f"Orphaned omega value: {line.strip()}")
                    continue

                omega_val = float(omega_match.group(1))
                # Update the current_state_analysis directly; it's mutable here until list append
                current_state_analysis = NTOStateAnalysis(
                    state_number=current_state_analysis.state_number,
                    multiplicity=current_state_analysis.multiplicity,
                    contributions=tuple(current_state_analysis.contributions),  # Keep existing if any
                    omega_percent=omega_val,
                )
                # Note: The contributions for this state are finalized when the *next* state header
                # appears, or section ends, or omega appears IF omega always comes after all contributions.
                # The current logic adds the completed state (with omega) when the next state_match occurs or section ends.
                # If omega finalizes the contribution list for the current state:
                current_state_analysis_with_omega = NTOStateAnalysis(
                    state_number=current_state_analysis.state_number,
                    multiplicity=current_state_analysis.multiplicity,
                    contributions=tuple(contributions_list),  # Use the accumulated contributions
                    omega_percent=omega_val,
                )
                # Replace the placeholder current_state_analysis with this one.
                # The final append to results.nto_state_analyses_list happens at state change or section end.
                current_state_analysis = current_state_analysis_with_omega
                # contributions_list is implicitly reset when a new state is encountered
                # or when this one is finalized.
                continue

            # If the line is not any of the above, it's an unrecognized line or start of a new section
            if line.strip():  # Avoid pushing back empty lines if they are not significant
                logger.warning(
                    f"Unrecognized line in SA-NTO Decomposition section or start of a new section: '{line.strip()}'"
                )
                results.parsing_warnings.append(f"Unrecognized line in NTO (or new section): {line.strip()}")
                if current_state_analysis:
                    # Finalize the last state being processed
                    updated_contributions = tuple(contributions_list)
                    final_state = NTOStateAnalysis(
                        state_number=current_state_analysis.state_number,
                        multiplicity=current_state_analysis.multiplicity,
                        contributions=updated_contributions,
                        omega_percent=current_state_analysis.omega_percent,
                    )
                    results.nto_state_analyses_list.append(final_state)

                results.buffered_line = line
                results.parsed_sa_nto_decomposition = True  # Mark as attempted/partially parsed
                return

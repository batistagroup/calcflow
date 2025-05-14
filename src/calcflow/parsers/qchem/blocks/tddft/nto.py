import re

from calcflow.parsers.qchem.core import NORMAL_TERM_PAT
from calcflow.parsers.qchem.typing import (
    LineIterator,
    NTOContribution,
    NTOStateAnalysis,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- SA-NTO Decomposition Block ---
NTO_SECTION_HEADER_PAT = re.compile(r"^\s*SA-NTO Decomposition\s*$")
STATE_HEADER_PAT = re.compile(r"^\s*(Singlet|Triplet)\s+(\d+)\s*:\s*$")  # Common state header
NTO_DECOMP_INTRO_PAT = re.compile(r"^\s*Decomposition into state-averaged NTOs\s*$")
NTO_CONTRIBUTION_PAT = re.compile(r"^\s*(H-|H)\s*(\d+)\s*->\s*(L\+|L)\s*(\d+):\s*(-?\d+\.\d+)\s*\(\s*(\d+\.\d+)%\)")
NTO_OMEGA_PAT = re.compile(r"^\s*omega\s*=\s*(\d+\.\d+)%\s*$")
SECTION_SEPARATOR_PAT = re.compile(r"^\s*-{20,}\s*$")  # Common separator
BLANK_LINE_PAT = re.compile(r"^\s*$")


class NTOParser(SectionParser):
    """Parses the 'SA-NTO Decomposition' block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return NTO_SECTION_HEADER_PAT.search(line) is not None

    def parse(self, iterator: LineIterator, first_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting parsing of SA-NTO Decomposition block.")

        if results.tddft_data is None:
            logger.error("NTOParser called but results.tddft_data is None. This should not happen.")
            # Optionally, initialize it here if that's a desired fallback, though it indicates a logic error elsewhere.
            # from calcflow.parsers.qchem.typing import TddftData
            # results.tddft_data = TddftData()
            return  # Or raise an error

        if results.tddft_data.nto_state_analyses is None:
            results.tddft_data.nto_state_analyses = []

        lines_buffer: list[str] = []

        # The first_line is the NTO_SECTION_HEADER_PAT, already consumed by matches()
        # We now expect state headers or blank lines.

        while True:
            try:
                if lines_buffer:
                    line = lines_buffer.pop(0)
                else:
                    line = next(iterator)
            except StopIteration:
                logger.debug("End of file reached during NTO parsing.")
                break

            # Check for termination conditions (e.g., start of another major section or normal termination)
            if NORMAL_TERM_PAT.search(line):
                logger.debug(f"NTO parsing stopped by normal termination pattern: {line.strip()}")
                lines_buffer.insert(0, line)  # Push back for other parsers
                break

            # Add other potential section headers that might terminate NTO block if necessary
            # For now, relying on NORMAL_TERM_PAT and specific parsing logic.

            match_state = STATE_HEADER_PAT.search(line)
            if not match_state:
                if not BLANK_LINE_PAT.search(line) and line.strip() and not SECTION_SEPARATOR_PAT.search(line):
                    logger.debug(f"Skipping non-state-header line in NTO: {line.strip()}")
                # If we see another NTO_SECTION_HEADER_PAT, it's an error or unexpected format
                if NTO_SECTION_HEADER_PAT.search(line) and line is not first_line:
                    logger.warning(f"Unexpected second NTO section header: {line.strip()}")
                    lines_buffer.insert(0, line)
                    break
                continue  # Look for a state header

            multiplicity = match_state.group(1)
            state_number = int(match_state.group(2))
            logger.debug(f"Parsing NTOs for {multiplicity} {state_number}")

            current_contributions: list[NTOContribution] = []
            current_omega: float | None = None

            # Expect "Decomposition into state-averaged NTOs" line next, or separator line
            try:
                line = next(iterator)  # Consume the line after state header
                if SECTION_SEPARATOR_PAT.search(line):  # Consume separator (e.g. Singlet 1 : \n -------)
                    line = next(iterator)

                if not NTO_DECOMP_INTRO_PAT.search(line):
                    if not BLANK_LINE_PAT.search(line):  # Allow for blank lines before the intro
                        logger.warning(
                            f"Expected NTO decomposition intro for state {state_number}, got: {line.strip()}"
                        )
                    # Try to find intro line if skipped by blank line
                    line = next(iterator)
                    if not NTO_DECOMP_INTRO_PAT.search(line):
                        logger.warning(
                            f"Still no NTO decomposition intro for state {state_number} after skipping one line, found: {line.strip()}"
                        )
                        # Decide how to proceed: skip state or try to parse contributions anyway?
                        # For now, let's assume the intro is critical and if not found, we might have issues.
                        # However, the contribution pattern is quite specific.
                        # Let's buffer this line and try to parse contributions.
                        lines_buffer.insert(0, line)

            except StopIteration:
                logger.warning(f"File ended unexpectedly after NTO state header for state {state_number}.")
                break

            # Parse contributions and omega for the current state
            # Max ~10 contributions + omega + few blank lines expected per state
            for _ in range(15):
                try:
                    if lines_buffer and (
                        NTO_CONTRIBUTION_PAT.search(lines_buffer[0]) or NTO_OMEGA_PAT.search(lines_buffer[0])
                    ):
                        line = lines_buffer.pop(0)  # Use buffered line if it's a contribution/omega
                    else:
                        line = next(iterator)
                except StopIteration:
                    break

                match_contrib = NTO_CONTRIBUTION_PAT.search(line)
                if match_contrib:
                    hole_type = match_contrib.group(1)
                    hole_idx = int(match_contrib.group(2))
                    elec_type = match_contrib.group(3)
                    elec_idx = int(match_contrib.group(4))
                    coeff = float(match_contrib.group(5))
                    weight = float(match_contrib.group(6))
                    current_contributions.append(
                        NTOContribution(
                            hole_nto_type=hole_type,
                            hole_nto_index=hole_idx,
                            electron_nto_type=elec_type,
                            electron_nto_index=elec_idx,
                            coefficient=coeff,
                            weight_percent=weight,
                        )
                    )
                    continue

                match_omega = NTO_OMEGA_PAT.search(line)
                if match_omega:
                    current_omega = float(match_omega.group(1))
                    # Omega usually signifies the end of contributions for this state in simple cases
                    # but let's not break here to catch any other potential lines or if format varies.
                    # Typically, after omega, it's a blank line then next state or end of section.
                    continue

                if BLANK_LINE_PAT.search(line):  # Skip blank lines
                    continue

                # If line is a new state header or section separator, or normal term, then this state is done
                if (
                    STATE_HEADER_PAT.search(line)
                    or SECTION_SEPARATOR_PAT.search(line)
                    or NORMAL_TERM_PAT.search(line)
                    or NTO_SECTION_HEADER_PAT.search(line)
                ):
                    lines_buffer.insert(0, line)  # Push back for outer loop or next parser
                    break

                # If it's not a contribution, omega, blank, or a known separator/header, log it.
                if line.strip():  # Avoid logging for lines that are just whitespace
                    logger.debug(f"Unexpected line in NTO state {state_number} analysis: {line.strip()}")
                    lines_buffer.insert(0, line)  # Buffer it in case it's a misplaced header for next state
                    break

            if current_contributions or current_omega is not None:
                results.tddft_data.nto_state_analyses.append(
                    NTOStateAnalysis(
                        state_number=state_number,
                        multiplicity=multiplicity,
                        contributions=current_contributions,
                        omega_percent=current_omega,
                    )
                )
            elif match_state:  # If we matched a state but found nothing, log it
                logger.debug(f"No NTO contributions or omega found for state {multiplicity} {state_number}.")

        if lines_buffer:  # Push back any remaining unconsumed lines
            iterator = iter(lines_buffer + list(iterator))

        logger.info(f"Parsed SA-NTO Decomposition for {len(results.tddft_data.nto_state_analyses)} states.")

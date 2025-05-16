import re
from typing import Literal, NamedTuple, cast

from calcflow.parsers.qchem.typing import (
    LineIterator,
    NTOContribution,
    NTOStateAnalysis,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger


class _CurrentNTOState(NamedTuple):
    state_number: int
    multiplicity: str
    contributions: list[NTOContribution]
    omega_rks: float | None
    omega_alpha: float | None
    omega_beta: float | None
    active_spin_channel: Literal["alpha", "beta", "none"]


class NTODecompositionParser(SectionParser):
    """Parses the SA-NTO Decomposition section from Q-Chem output, supporting RKS and UKS."""

    SECTION_START_TOKEN = "SA-NTO Decomposition"
    SECTION_END_TOKEN = "================================================================================"
    # Adjusted to include "Excited State" for UKS
    STATE_HEADER_PATTERN = re.compile(r"^\s*(Singlet|Triplet|Excited State)\s*(\d+)\s*:")
    DECOMPOSITION_LINE_PATTERN = re.compile(r"^\s*Decomposition into state-averaged NTOs\s*$")
    ALPHA_SPIN_HEADER_PATTERN = re.compile(r"^\s*Alpha spin:\s*$")
    BETA_SPIN_HEADER_PATTERN = re.compile(r"^\s*Beta spin:\s*$")

    NTO_CONTRIBUTION_PATTERN = re.compile(
        r"^\s* ([HV]) ([+-])\s* (\d+) \s*->\s* ([LV]) ([+-])\s* (\d+) \s*:\s* (-?\d+\.\d+) \s*\(\s* (\d+\.\d+) \s*%\s*\)\s*$",
        re.VERBOSE,
    )
    OMEGA_PATTERN = re.compile(
        r"^\s*omega\s*=\s*(\d+\.\d+)\s*%$",
        re.IGNORECASE,
    )

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line marks the beginning of the SA-NTO decomposition section."""
        return self.SECTION_START_TOKEN in line and not current_data.parsed_sa_nto_decomposition

    def _finalize_current_state(
        self, current_state_data: _CurrentNTOState | None, results: _MutableCalculationData
    ) -> None:
        if not current_state_data:
            return

        total_omega: float | None = None
        final_omega_alpha = current_state_data.omega_alpha
        final_omega_beta = current_state_data.omega_beta

        if current_state_data.omega_rks is not None:  # RKS case
            total_omega = current_state_data.omega_rks
        elif final_omega_alpha is not None or final_omega_beta is not None:  # UKS case
            total_omega = (final_omega_alpha or 0.0) + (final_omega_beta or 0.0)
            # Avoid 0.0 if no omegas were found and no contributions, unless explicitly 0.0
            if (
                total_omega == 0.0
                and not current_state_data.contributions
                and (final_omega_alpha is None and final_omega_beta is None)
            ):
                total_omega = None

        if total_omega is not None and abs(total_omega) < 1e-9 and not current_state_data.contributions:  # noqa: SIM102
            # if total_omega is effectively zero and there are no contributions, set to None
            # unless it was explicitly parsed as 0.0 from an RKS omega. noqa for readability
            if not (current_state_data.omega_rks is not None and abs(current_state_data.omega_rks) < 1e-9):
                total_omega = None

        state_to_append = NTOStateAnalysis(
            state_number=current_state_data.state_number,
            multiplicity=current_state_data.multiplicity,
            contributions=tuple(current_state_data.contributions),
            omega_percent=total_omega,
            omega_alpha_percent=final_omega_alpha,
            omega_beta_percent=final_omega_beta,
        )
        results.nto_state_analyses_list.append(state_to_append)
        logger.debug(f"Finalized NTOStateAnalysis: {state_to_append}")

    def parse(self, iterator: LineIterator, first_line: str, results: _MutableCalculationData) -> None:
        """Parse the SA-NTO Decomposition section."""
        logger.debug("Starting SA-NTO Decomposition parsing.")

        # Consume the section header line and the following decorative line
        # if self.SECTION_START_TOKEN not in first_line: ... (already handled by matches logic usually)
        try:
            line_after_header = next(iterator)
            if "---" not in line_after_header:
                logger.warning(
                    f"NTO Parser: Expected separator line after header, got: {line_after_header.strip()}"
                )  # pragma: no cover
                results.parsing_warnings.append(
                    f"NTO Parser: Missing separator after header: {line_after_header.strip()}"
                )  # pragma: no cover
                results.buffered_line = line_after_header  # pragma: no cover
                results.parsed_sa_nto_decomposition = True  # pragma: no cover
                return  # pragma: no cover
        except StopIteration:  # pragma: no cover
            logger.warning("File ended unexpectedly after NTO section header.")  # pragma: no cover
            results.parsing_warnings.append("File ended unexpectedly after NTO section header.")  # pragma: no cover
            results.parsed_sa_nto_decomposition = True  # pragma: no cover
            return  # pragma: no cover

        current_state_data: _CurrentNTOState | None = None

        while True:
            try:
                line = next(iterator)
            except StopIteration:  # pragma: no cover
                logger.warning("File ended unexpectedly while parsing SA-NTO Decomposition.")  # pragma: no cover
                results.parsing_warnings.append(
                    "File ended unexpectedly during SA-NTO Decomposition parsing."
                )  # pragma: no cover
                self._finalize_current_state(current_state_data, results)  # pragma: no cover
                results.parsed_sa_nto_decomposition = True  # pragma: no cover
                return  # pragma: no cover

            if self.SECTION_END_TOKEN in line:
                logger.debug("Found SA-NTO Decomposition section end token.")
                self._finalize_current_state(current_state_data, results)
                results.parsed_sa_nto_decomposition = True
                return

            # Add logging for repr(line) before attempting to match the state header
            logger.debug(f"NTO Parser: Attempting STATE_HEADER_PATTERN match on line: {repr(line)}")
            state_match = self.STATE_HEADER_PATTERN.match(line)
            if state_match:
                self._finalize_current_state(current_state_data, results)

                multiplicity_str = state_match.group(1)
                state_number = int(state_match.group(2))
                logger.debug(f"Parsing NTOs for {multiplicity_str} state {state_number}")
                current_state_data = _CurrentNTOState(
                    state_number=state_number,
                    multiplicity=multiplicity_str,
                    contributions=[],
                    omega_rks=None,
                    omega_alpha=None,
                    omega_beta=None,
                    active_spin_channel="none",
                )
                # Consume the separator line like "-------------------------"
                try:
                    sep_line = next(iterator)
                    if "---" not in sep_line:  # pragma: no cover
                        logger.warning(
                            f"Expected separator '---' after state header for {multiplicity_str} {state_number}, got: {sep_line.strip()}"
                        )  # pragma: no cover
                        # This might be an issue, but we try to continue
                except StopIteration:  # pragma: no cover
                    logger.warning(
                        f"File ended unexpectedly after state header for {multiplicity_str} {state_number}."
                    )  # pragma: no cover
                    results.parsing_warnings.append(
                        f"File ended after state header for {multiplicity_str} {state_number}."
                    )  # pragma: no cover
                    results.parsed_sa_nto_decomposition = True  # pragma: no cover
                    return  # pragma: no cover
                continue

            if not current_state_data:  # Should have state data if not a state header or section end
                if line.strip():  # pragma: no cover
                    logger.warning(f"Skipping line in NTO section, no active state: {line.strip()}")  # pragma: no cover
                continue

            # Update current_state_data fields directly, it's a NamedTuple, so we replace it.
            temp_contributions = current_state_data.contributions
            temp_omega_rks = current_state_data.omega_rks
            temp_omega_alpha = current_state_data.omega_alpha
            temp_omega_beta = current_state_data.omega_beta
            temp_active_spin = current_state_data.active_spin_channel
            # Always derive the spin flag; default to alpha for RKS/`none`
            is_alpha: bool = temp_active_spin != "beta"

            if self.DECOMPOSITION_LINE_PATTERN.match(line):
                # This line is informational for RKS, just consume.
                # Ensure we are in "none" spin channel context if this line appears.
                if temp_active_spin != "none":  # pragma: no cover
                    logger.warning(
                        f"'Decomposition...' line found unexpectedly in {temp_active_spin} context."
                    )  # pragma: no cover
                continue

            elif self.ALPHA_SPIN_HEADER_PATTERN.match(line):
                temp_active_spin = "alpha"
                current_state_data = current_state_data._replace(active_spin_channel=temp_active_spin)
                continue

            elif self.BETA_SPIN_HEADER_PATTERN.match(line):
                temp_active_spin = "beta"
                current_state_data = current_state_data._replace(active_spin_channel=temp_active_spin)
                continue

            contribution_match = self.NTO_CONTRIBUTION_PATTERN.match(line)
            if contribution_match:
                hole_ref = contribution_match.group(1)
                hole_sign = contribution_match.group(2)
                hole_num = contribution_match.group(3)
                elec_ref = contribution_match.group(4)
                elec_sign = contribution_match.group(5)
                elec_num = contribution_match.group(6)
                coeff = float(contribution_match.group(7))
                weight = float(contribution_match.group(8))

                hole_offset = int(hole_sign + hole_num)
                elec_offset = int(elec_sign + elec_num)

                contribution = NTOContribution(
                    hole_reference=cast(Literal["H", "V"], hole_ref),
                    hole_offset=hole_offset,
                    electron_reference=cast(Literal["L", "V"], elec_ref),
                    electron_offset=elec_offset,
                    coefficient=coeff,
                    weight_percent=weight,
                    is_alpha_spin=is_alpha,
                )
                temp_contributions.append(contribution)
                current_state_data = current_state_data._replace(contributions=temp_contributions)
                continue

            omega_match = self.OMEGA_PATTERN.match(line)
            if omega_match:
                omega_val = float(omega_match.group(1))
                if temp_active_spin == "alpha":
                    temp_omega_alpha = omega_val
                    current_state_data = current_state_data._replace(omega_alpha=temp_omega_alpha)
                elif temp_active_spin == "beta":
                    temp_omega_beta = omega_val
                    current_state_data = current_state_data._replace(omega_beta=temp_omega_beta)
                elif temp_active_spin == "none":  # RKS
                    temp_omega_rks = omega_val
                    current_state_data = current_state_data._replace(omega_rks=temp_omega_rks)
                else:  # Should not happen # pragma: no cover
                    logger.error(f"Omega parsed in unknown spin state: {temp_active_spin}")  # pragma: no cover
                continue

            # If the line is not any of the above, it's an unrecognized line or start of a new section
            if line.strip():  # pragma: no cover
                logger.warning(
                    f"Unrecognized line in SA-NTO Decomposition section or start of a new section: '{line.strip()}'"
                )  # pragma: no cover
                results.parsing_warnings.append(
                    f"Unrecognized line in NTO (or new section): {line.strip()}"
                )  # pragma: no cover
                self._finalize_current_state(current_state_data, results)  # pragma: no cover
                current_state_data = None  # pragma: no cover

                results.buffered_line = line  # pragma: no cover
                results.parsed_sa_nto_decomposition = True  # pragma: no cover
                return  # pragma: no cover

import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import (
    GradientData,
    LineIterator,
    OptimizationCycleData,
    _MutableCalculationData,
    _MutableOptData,
)
from calcflow.utils import logger

GRADIENT_START_PAT = re.compile(r"^\s*CARTESIAN GRADIENT")
GRADIENT_LINE_PAT = re.compile(r"^\s*\d+\s+[A-Za-z]+\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
GRADIENT_NORM_PAT = re.compile(r"^\s*Norm of the Cartesian gradient\s+\.{3}\s+(\d+\.\d+)")
GRADIENT_RMS_PAT = re.compile(r"^\s*RMS gradient\s+\.{3}\s+(\d+\.\d+)")
GRADIENT_MAX_PAT = re.compile(r"^\s*MAX gradient\s+\.{3}\s+(\d+\.\d+)")


class GradientParser:
    """Parses the CARTESIAN GRADIENT block within an optimization cycle."""

    def matches(self, line: str, current_data: _MutableCalculationData | _MutableOptData) -> bool:
        """Checks if the line marks the start of the Cartesian Gradient block."""
        # This parser specifically targets the Cartesian Gradient block.
        # We assume Dispersion Gradient is handled elsewhere or ignored if not needed.
        return bool(GRADIENT_START_PAT.search(line))

    def parse(
        self,
        iterator: LineIterator,
        current_line: str,
        results: _MutableOptData,
        current_cycle_data: OptimizationCycleData | None,
    ) -> None:
        """Parses the gradient values and summary statistics."""
        if not current_cycle_data:
            logger.warning("GradientParser matched outside of an optimization cycle. Skipping.")
            # Consume until end of block? For now, just return.
            return

        logger.debug(f"Parsing Cartesian Gradient block starting near line: {current_line}")
        gradients: dict[int, tuple[float, float, float]] = {}
        norm: float | None = None
        rms: float | None = None
        max_grad: float | None = None
        atom_index = 0

        try:
            # Consume the separator line and the following blank line
            try:
                next(iterator)
                next(iterator)
            except StopIteration as e:
                raise ParsingError("File ended unexpectedly immediately after gradient header.") from e

            while True:
                line = next(iterator)

                match_grad = GRADIENT_LINE_PAT.match(line)
                if match_grad:
                    gx = float(match_grad.group(1))
                    gy = float(match_grad.group(2))
                    gz = float(match_grad.group(3))
                    gradients[atom_index] = (gx, gy, gz)
                    atom_index += 1
                    continue

                match_norm = GRADIENT_NORM_PAT.search(line)
                if match_norm:
                    norm = float(match_norm.group(1))

                match_rms = GRADIENT_RMS_PAT.search(line)
                if match_rms:
                    rms = float(match_rms.group(1))

                match_max = GRADIENT_MAX_PAT.search(line)
                if match_max:
                    max_grad = float(match_max.group(1))
                    break

        except StopIteration:
            logger.warning("Reached end of file while parsing gradient block.")
        except ParsingError:  # Re-raise critical parsing errors
            raise
        except Exception as e:
            raise ParsingError("An unexpected error occurred during gradient parsing.") from e

        # Validation
        if not gradients:
            raise ParsingError("Gradient block found but no gradient lines parsed.")
        if norm is None or rms is None or max_grad is None:
            logger.warning(f"Parsed gradients but missing summary stats (Norm: {norm}, RMS: {rms}, Max: {max_grad})")
            # Allow continuation, but data will be incomplete
            # Set default values or handle missing data downstream
            norm = norm or 0.0
            rms = rms or 0.0
            max_grad = max_grad or 0.0

        gradient_data = GradientData(gradients=gradients, norm=norm, rms=rms, max=max_grad)
        current_cycle_data.gradient = gradient_data
        logger.debug(f"Stored GradientData for cycle {current_cycle_data.cycle_number}")

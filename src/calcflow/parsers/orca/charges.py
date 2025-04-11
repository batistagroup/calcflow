import re
from re import Pattern
from typing import Literal

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.typing import AtomicCharges, LineIterator, _MutableCalculationData
from calcflow.utils import logger

# --- Charges Parsers --- #
MULLIKEN_CHARGES_START_PAT = re.compile(r"MULLIKEN ATOMIC CHARGES")
LOEWDIN_CHARGES_START_PAT = re.compile(r"LOEWDIN ATOMIC CHARGES")
CHARGE_LINE_PAT = re.compile(r"^\s*(\d+)\s+[A-Za-z]{1,3}\s+:\s+(-?\d+\.\d+)")


class ChargesParser:
    """Parses atomic charges blocks (Mulliken, Loewdin)."""

    def __init__(self, method: Literal["Mulliken", "Loewdin"], start_pattern: Pattern[str]):
        self.method = method
        self.start_pattern = start_pattern

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        # Allow multiple charge blocks, but only parse each *type* once if needed
        # has_parsed_this_method = any(c.method == self.method for c in current_data.atomic_charges)
        # For now, let's allow reparsing if the block appears again (might overwrite)
        # If strict single parse is needed, use: `and not has_parsed_this_method`
        return bool(self.start_pattern.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug(f"Starting {self.method} charges block parsing.")
        # Consume header lines (----)
        next(iterator, None)
        charges: dict[int, float] = {}
        try:
            for line in iterator:
                line_stripped = line.strip()
                if not line_stripped:
                    break

                match = CHARGE_LINE_PAT.match(line_stripped)
                if match:
                    try:
                        idx_str, charge_str = match.groups()
                        atom_index = int(idx_str)
                        charge = float(charge_str)
                        charges[atom_index] = charge
                    except ValueError as e:
                        raise ParsingError(f"Could not parse {self.method} charge line values: {line_stripped}") from e
                elif charges:  # Stop if format breaks
                    break

            if not charges:
                raise ParsingError(f"{self.method} charges block found but no charges could be parsed.")

            charge_data = AtomicCharges(method=self.method, charges=charges)
            # Append or replace? Let's append for now.
            results.atomic_charges.append(charge_data)
            logger.debug(f"Successfully parsed {self.method} charges: {repr(charge_data)}")

        except Exception as e:
            logger.error(f"Error parsing {self.method} charges block: {e}", exc_info=True)
            # Allow parsing to continue

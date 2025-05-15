import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.blocks.charges import LOEWDIN_CHARGES_START_PAT, MULLIKEN_CHARGES_START_PAT, ChargesParser
from calcflow.parsers.orca.blocks.dipole import DipoleParser
from calcflow.parsers.orca.blocks.dispersion import DispersionParser
from calcflow.parsers.orca.blocks.geometry import GeometryParser
from calcflow.parsers.orca.blocks.orbitals import OrbitalsParser
from calcflow.parsers.orca.blocks.scf import ScfParser
from calcflow.parsers.orca.typing import (
    Atom,
    AtomicCharges,
    DipoleMomentData,
    DispersionCorrectionData,
    OrbitalsSet,
    ScfData,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- Regex Patterns (Grouped by section) --- #

# Patterns not tied to a specific block parser
FINAL_ENERGY_PAT = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
NORMAL_TERM_PAT = re.compile(r"\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*")
ERROR_TERM_PAT = re.compile(r"(TERMINATING THE PROGRAM|ORCA finished with error)")


@dataclass(frozen=True)
class CalculationData:
    """Top-level container for parsed ORCA calculation results."""

    raw_output: str = field(repr=False)
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"]
    input_geometry: Sequence[Atom] | None = None
    final_energy_eh: float | None = None
    scf: ScfData | None = None
    orbitals: OrbitalsSet | None = None
    atomic_charges: list[AtomicCharges] = field(default_factory=list)
    dipole_moment: DipoleMomentData | None = None
    dispersion_correction: DispersionCorrectionData | None = None

    @classmethod
    def from_mutable(cls, mutable_data: _MutableCalculationData) -> "CalculationData":
        """Creates an immutable CalculationData from the mutable version."""
        return cls(
            raw_output=mutable_data.raw_output,
            termination_status=mutable_data.termination_status,
            input_geometry=mutable_data.input_geometry,
            final_energy_eh=mutable_data.final_energy_eh,
            scf=mutable_data.scf,
            orbitals=mutable_data.orbitals,
            atomic_charges=list(mutable_data.atomic_charges),  # Ensure list copy
            dipole_moment=mutable_data.dipole_moment,
            dispersion_correction=mutable_data.dispersion_correction,
        )

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        lines.append(f"  termination_status='{self.termination_status}',")
        lines.append(
            f"  final_energy_eh={self.final_energy_eh:.8f},"
            if self.final_energy_eh is not None
            else "  final_energy_eh=None,"
        )
        n_atoms = len(self.input_geometry) if self.input_geometry else 0
        lines.append(f"  input_geometry=({n_atoms} Atoms),")

        # Add summaries for optional components if they exist
        if self.scf:
            lines.append(f"  scf={str(self.scf)},")
        if self.orbitals:
            lines.append(f"  orbitals={repr(self.orbitals)},")
        if self.atomic_charges:
            lines.append(f"  atomic_charges={repr(self.atomic_charges)},")
        if self.dipole_moment:
            lines.append(f"  dipole_moment={repr(self.dipole_moment)},")
        if self.dispersion_correction:
            lines.append(f"  dispersion_correction={repr(self.dispersion_correction)},")

        # Clean up trailing comma on last line if any optional fields were added
        if len(lines) > 4 and lines[-1].endswith(","):
            lines[-1] = lines[-1].rstrip(",")

        lines.append(")")
        return "\n".join(lines)


# --- Parser Registry --- #
# Order might matter if sections can be nested or ambiguous, but generally shouldn't
# Place more specific/complex parsers (like SCF) earlier if needed.
PARSER_REGISTRY: Sequence[SectionParser] = [
    GeometryParser(),
    ScfParser(),  # SCF must come before orbitals/charges which might appear inside guess prints
    OrbitalsParser(),
    ChargesParser("Mulliken", MULLIKEN_CHARGES_START_PAT),
    ChargesParser("Loewdin", LOEWDIN_CHARGES_START_PAT),
    # Add MayerParser here if implemented
    DipoleParser(),
    DispersionParser(),
]


# --- Main Parsing Function --- #
def parse_orca_sp_output(output: str) -> CalculationData:
    """
    Parses the text output of an ORCA calculation using a registry of section parsers.

    Args:
        output: The string content of the ORCA output file.

    Returns:
        A CalculationData object containing the parsed results.

    Raises:
        ParsingError: If essential components (like geometry) cannot be parsed,
                      or if critical errors occur during parsing.
    """
    lines = output.splitlines()
    # Use PeekingIterator if lookahead is needed, but avoid for now
    line_iterator = iter(lines)
    results = _MutableCalculationData(raw_output=output)  # Start with mutable data

    current_line_num = 0
    try:
        while True:  # Use explicit break or StopIteration
            try:
                line = next(line_iterator)
                current_line_num += 1
            except StopIteration:
                logger.debug("Reached end of input.")
                break

            # Check registered block parsers first
            parser_found = False
            for parser in PARSER_REGISTRY:
                try:
                    if parser.matches(line, results):
                        parser.parse(line_iterator, line, results)
                        parser_found = True
                        break  # Only one parser should handle the start of a block
                except ParsingError as e:
                    # Propagate critical errors, log others based on parser logic
                    logger.error(f"Parser {type(parser).__name__} failed critically: {e}", exc_info=True)
                    raise  # Re-raise critical errors
                except Exception as e:
                    # Catch unexpected errors within a parser's logic
                    logger.error(f"Unexpected error in {type(parser).__name__}: {e}", exc_info=True)
                    # Decide whether to raise or try to continue
                    # For robustness, let's try to continue, but log it as an error.
                    # Mark the relevant section as attempted if possible (depends on parser impl)

            if parser_found:
                continue  # Move to the next line after block parsing consumed lines

            # --- Handle Line-Specific Information (if not handled by a block parser) ---
            match_final_energy = FINAL_ENERGY_PAT.search(line)
            if match_final_energy:
                try:
                    # Overwrite if found multiple times, last one is usually relevant
                    results.final_energy_eh = float(match_final_energy.group(1))
                    logger.debug(f"Found Final Single Point Energy: {results.final_energy_eh}")
                except (ValueError, IndexError) as e:
                    logger.error(f"Could not parse final energy value from line: {line.strip()}", exc_info=True)
                    # Decide if this is critical - likely yes?
                    raise ParsingError("Failed to parse final energy value.") from e
                continue

            # --- Termination Status ---
            if NORMAL_TERM_PAT.search(line):
                results.termination_status = "NORMAL"
                logger.debug("Found Normal Termination.")
                # Don't break, continue parsing until the end
            elif ERROR_TERM_PAT.search(line) and results.termination_status == "UNKNOWN":
                results.termination_status = "ERROR"
                logger.debug("Found Error Termination.")

    except ParsingError:
        # Re-raise critical parsing errors immediately
        raise
    except Exception as e:
        # Catch unexpected errors during the main loop iteration
        logger.critical(f"Unexpected error in main parsing loop at line ~{current_line_num}: {e}", exc_info=True)
        # Preserve original traceback if possible?
        # traceback.print_exc() # Avoid direct print
        results.termination_status = "ERROR"  # Mark as error if unexpected exception happens
        # Optionally re-raise as a ParsingError
        raise ParsingError(f"An unexpected error occurred during parsing: {e}") from e

    # --- Final Checks ---
    if results.input_geometry is None:
        logger.error("Input geometry block was not found or parsed.")
        raise ParsingError("Input geometry block was not found in the output file.")

    # Refine termination status if still UNKNOWN
    if results.termination_status == "UNKNOWN":
        logger.warning("Termination status unknown after parsing, assuming ERROR.")
        results.termination_status = "ERROR"

    # Final check for SCF data if expected (depends on calculation type?)
    # For now, just a warning if it wasn't parsed successfully.
    if results.scf is None and results.parsed_scf:  # Check if parsing was attempted but failed
        logger.warning("SCF data block parsing was attempted but failed or did not find required components.")
    elif not results.parsed_scf:  # Check if the block wasn't even found
        logger.warning("SCF data block was not found.")

    logger.info(f"ORCA parsing finished. Status: {results.termination_status}, Final Energy: {results.final_energy_eh}")
    return CalculationData.from_mutable(results)  # Convert back to immutable dataclass

import re
from collections.abc import Sequence

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.typing import (
    CalculationData,
    CalculationMetadata,
    LineIterator,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- Regex Patterns (Compile patterns for efficiency) ---

# Metadata Patterns
QCHEM_VERSION_PAT = re.compile(r"^ Q-Chem (\d+\.\d+(\.\d+)?), Q-Chem, Inc")
HOST_PAT = re.compile(r"^ Host: (\S+)")
RUN_DATE_PAT = re.compile(r"^ Q-Chem begins on (.*)")  # Captures the whole date string

# Input Section Patterns (from $rem block)
METHOD_PAT = re.compile(r"^METHOD\s+(\S+)", re.IGNORECASE)
BASIS_PAT = re.compile(r"^BASIS\s+(\S+)", re.IGNORECASE)

# Energy/Structure Patterns
NUCLEAR_REPULSION_PAT = re.compile(r"^ Nuclear Repulsion Energy =\s+(-?\d+\.\d+)")
# Use 'Total energy =' after SCF as the final SP energy
FINAL_ENERGY_PAT = re.compile(r"^ Total energy =\s+(-?\d+\.\d+)")

# Termination Patterns
NORMAL_TERM_PAT = re.compile(r"Thank you very much for using Q-Chem\. Have a nice day\.")
# TODO: Identify robust patterns for various Q-Chem error terminations
ERROR_TERM_PAT = re.compile(r"(ERROR:|error:|aborting|failed)", re.IGNORECASE)

# --- Parser Registry --- #
# This list will hold instances of SectionParser implementations (e.g., GeometryParser, ScfParser)
# Order might matter if sections can be nested or ambiguous.
PARSER_REGISTRY: Sequence[SectionParser] = [
    # Add specific block parsers here later
    # e.g., GeometryParser(), ScfParser(), OrbitalsParser(), ChargesParser(), DipoleParser(), DispersionParser()
]


# --- Main Parsing Function --- #
def parse_qchem_sp_output(output: str) -> CalculationData:
    """
    Parses the text output of a Q-Chem single-point calculation.

    Args:
        output: The string content of the Q-Chem output file.

    Returns:
        A CalculationData object containing the parsed results.

    Raises:
        ParsingError: If essential components cannot be parsed or critical errors occur.
    """
    lines = output.splitlines()
    line_iterator: LineIterator = iter(lines)
    # Initialize mutable data structure
    results = _MutableCalculationData(raw_output=output)
    # Keep track of parsed metadata components separately
    _meta: dict[str, str | None] = {
        "qchem_version": None,
        "host": None,
        "run_date": None,
        "calculation_method": None,
        "basis_set": None,
    }

    # --- State flags for context-dependent parsing --- #
    in_rem_block = False

    current_line_num = 0
    try:
        while True:  # Use explicit break or StopIteration
            try:
                line = next(line_iterator)
                current_line_num += 1
            except StopIteration:
                logger.debug("Reached end of input.")
                break

            # --- Handle Block Parsing --- #
            parser_found = False
            # TODO: Implement block parser logic
            # for parser in PARSER_REGISTRY:
            #     try:
            #         if parser.matches(line, results):
            #             parser.parse(line_iterator, line, results)
            #             parser_found = True
            #             # Reset context flags if necessary after block parsing
            #             in_rem_block = False
            #             break # Only one parser should handle the start of a block
            #     except ParsingError as e:
            #         logger.error(f"Parser {type(parser).__name__} failed critically: {e}", exc_info=True)
            #         raise # Re-raise critical errors
            #     except Exception as e:
            #         logger.error(f"Unexpected error in {type(parser).__name__}: {e}", exc_info=True)
            #         results.parsing_errors.append(f"Error in {type(parser).__name__} near line {current_line_num}: {e}")
            #         # Decide whether to continue or raise

            if parser_found:
                continue  # Move to the next line after block parsing consumed lines

            # --- Handle Context Flags --- #
            if "$rem" in line:
                in_rem_block = True
                continue
            elif "$end" in line and in_rem_block:
                in_rem_block = False
                continue

            # --- Handle Line-Specific Information --- #

            # Metadata
            match_version = QCHEM_VERSION_PAT.search(line)
            if match_version and _meta["qchem_version"] is None:
                _meta["qchem_version"] = match_version.group(1)
                logger.debug(f"Found Q-Chem Version: {_meta['qchem_version']}")
                continue

            match_host = HOST_PAT.search(line)
            if match_host and _meta["host"] is None:
                _meta["host"] = match_host.group(1)
                logger.debug(f"Found Host: {_meta['host']}")
                continue

            match_date = RUN_DATE_PAT.search(line)
            if match_date and _meta["run_date"] is None:
                _meta["run_date"] = match_date.group(1).strip()
                logger.debug(f"Found Run Date: {_meta['run_date']}")
                continue

            # $rem block info (only parse if in_rem_block)
            if in_rem_block:
                match_method = METHOD_PAT.search(line)
                if match_method and _meta["calculation_method"] is None:
                    _meta["calculation_method"] = match_method.group(1)
                    logger.debug(f"Found Method: {_meta['calculation_method']}")
                    continue

                match_basis = BASIS_PAT.search(line)
                if match_basis and _meta["basis_set"] is None:
                    _meta["basis_set"] = match_basis.group(1)
                    logger.debug(f"Found Basis Set: {_meta['basis_set']}")
                    continue

            # Energy Components
            match_nuc_rep = NUCLEAR_REPULSION_PAT.search(line)
            if match_nuc_rep and results.nuclear_repulsion_eh is None:
                try:
                    results.nuclear_repulsion_eh = float(match_nuc_rep.group(1))
                    logger.debug(f"Found Nuclear Repulsion Energy: {results.nuclear_repulsion_eh}")
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse Nuclear Repulsion from line: {line.strip()}")
                continue

            match_final_energy = FINAL_ENERGY_PAT.search(line)
            if match_final_energy:
                try:
                    # Q-Chem might print this multiple times (e.g., SCF, Total). Last one is likely best.
                    results.final_energy_eh = float(match_final_energy.group(1))
                    logger.debug(f"Found Potential Final Energy: {results.final_energy_eh}")
                except (ValueError, IndexError) as e:
                    logger.error(f"Could not parse final energy value from line: {line.strip()}", exc_info=True)
                    raise ParsingError("Failed to parse final energy value.") from e
                continue

            # --- Termination Status --- #
            if NORMAL_TERM_PAT.search(line):
                results.termination_status = "NORMAL"
                logger.debug("Found Normal Termination signature.")
                # Don't break yet, allow parsing final sections if any

            elif ERROR_TERM_PAT.search(line) and results.termination_status == "UNKNOWN":
                results.termination_status = "ERROR"
                logger.debug(f"Found potential Error Termination signature in line: {line.strip()}")
                # Continue parsing to gather as much data as possible

    except ParsingError:
        results.termination_status = "ERROR"  # Mark as error if parsing fails critically
        # Populate metadata before raising
        results.metadata = CalculationMetadata(**_meta)  # type: ignore[arg-type]
        raise
    except Exception as e:
        logger.critical(f"Unexpected error in main parsing loop near line ~{current_line_num}: {e}", exc_info=True)
        results.termination_status = "ERROR"
        results.parsing_errors.append(f"Critical error near line {current_line_num}: {e}")
        # Populate metadata before raising
        results.metadata = CalculationMetadata(**_meta)  # type: ignore[arg-type]
        raise ParsingError(f"An unexpected error occurred during parsing: {e}") from e

    # --- Final Checks and Refinements --- #

    # Create the final immutable metadata object before final checks
    # Filter out None values before passing to the dataclass constructor
    final_meta_dict = {k: v for k, v in _meta.items() if v is not None}
    results.metadata = CalculationMetadata(**final_meta_dict)  # type: ignore[arg-type]

    # Example Check: Ensure geometry was parsed (will be handled by GeometryParser later)
    # if results.standard_orientation_geometry is None:
    #     logger.error("Standard orientation geometry block was not found or parsed.")
    #     # Decide if this is critical for SP - maybe allow missing for now?
    #     # raise ParsingError("Standard orientation geometry block was not found.")
    #     results.parsing_warnings.append("Standard orientation geometry not parsed.")

    # Refine termination status if still UNKNOWN
    if results.termination_status == "UNKNOWN":
        logger.warning("Termination status unknown after parsing, assuming ERROR.")
        results.termination_status = "ERROR"

    # Add more final checks as needed (e.g., SCF convergence status if parsed)

    logger.info(
        f"Q-Chem SP parsing finished. Status: {results.termination_status}, Final Energy: {results.final_energy_eh}"
    )

    # Convert mutable data to the final immutable structure
    return CalculationData.from_mutable(results)

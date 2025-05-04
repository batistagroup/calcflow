import re
from collections.abc import Sequence

from calcflow.exceptions import ParsingError

# Import block parsers
from calcflow.parsers.qchem.blocks import GeometryParser, MetadataParser, RemBlockParser, ScfParser
from calcflow.parsers.qchem.typing import (
    CalculationData,
    LineIterator,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- Regex Patterns (Compile patterns for efficiency) ---

# Metadata Patterns
# QCHEM_VERSION_PAT = re.compile(r"^ Q-Chem (\d+\.\d+(\.\d+)?), Q-Chem, Inc")
# HOST_PAT = re.compile(r"^ Host: (\S+)")
# RUN_DATE_PAT = re.compile(r"^ Q-Chem begins on (.*)")

# Input Section Patterns (from $rem block)
# METHOD_PAT = re.compile(r"^METHOD\s+(\S+)", re.IGNORECASE)
# BASIS_PAT = re.compile(r"^BASIS\s+(\S+)", re.IGNORECASE)

# Energy/Structure Patterns
NUCLEAR_REPULSION_PAT = re.compile(r"^ Nuclear Repulsion Energy =\s+(-?\d+\.\d+)")
# Use 'Total energy =' after SCF as the final SP energy
FINAL_ENERGY_PAT = re.compile(r"^ Total energy =\s+(-?\d+\.\d+)")

# Termination Patterns
NORMAL_TERM_PAT = re.compile(r"Thank you very much for using Q-Chem\. Have a nice day\.")
# TODO: Identify robust patterns for various Q-Chem error terminations
ERROR_TERM_PAT = re.compile(r"(ERROR:|error:|aborting|failed)", re.IGNORECASE)

# --- Parser Registry --- #
# Order matters: Parse $rem before geometry/scf which might appear later.
# Metadata can appear anywhere.
PARSER_REGISTRY: Sequence[SectionParser] = [
    MetadataParser(),
    RemBlockParser(),
    GeometryParser(),
    ScfParser(),
    # Add other specific block parsers here later
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
    results = _MutableCalculationData(raw_output=output)

    current_line_num = 0
    try:
        while True:
            try:
                line = next(line_iterator)
                current_line_num += 1
            except StopIteration:
                logger.debug("Reached end of input.")
                break

            # --- Handle Block Parsing --- #
            parser_found = False
            for parser in PARSER_REGISTRY:
                try:
                    if parser.matches(line, results):
                        block_start_line = current_line_num
                        logger.debug(f"Line {block_start_line}: Matched {type(parser).__name__}")
                        # MetadataParser is special, doesn't advance iterator beyond current line
                        # Other parsers consume lines within their parse method.
                        parser.parse(line_iterator, line, results)
                        parser_found = True
                        break  # Only one parser should handle the start of a block
                except ParsingError as e:
                    logger.error(
                        f"Parser {type(parser).__name__} failed critically near line {block_start_line}: {e}",
                        exc_info=True,
                    )
                    raise
                except StopIteration as e:
                    logger.error(
                        f"Parser {type(parser).__name__} unexpectedly consumed end of iterator near line {block_start_line}.",
                        exc_info=True,
                    )
                    raise ParsingError(f"File ended unexpectedly during {type(parser).__name__} parsing.") from e
                except Exception as e:
                    logger.error(
                        f"Unexpected error in {type(parser).__name__} near line {block_start_line}: {e}", exc_info=True
                    )
                    results.parsing_errors.append(f"Error in {type(parser).__name__} near line {block_start_line}: {e}")
                    raise ParsingError(
                        f"Unexpected error in {type(parser).__name__} near line {block_start_line}: {e}"
                    ) from e

            if parser_found:
                # If a parser matched and executed, it potentially consumed lines.
                # The main loop's next() call will get the line *after* the block.
                # MetadataParser is an exception as it only parses the current line,
                # but the main loop still calls next() after it runs.
                continue

            # --- Handle Line-Specific Information (if not handled by a block parser) ---

            # Metadata checks removed (handled by MetadataParser)
            # match_version = QCHEM_VERSION_PAT.search(line)
            # if match_version and _meta["qchem_version"] is None:
            #     _meta["qchem_version"] = match_version.group(1)
            #     logger.debug(f"Found Q-Chem Version: {_meta['qchem_version']}")
            #     continue
            # ... (host, date checks removed) ...

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

            elif ERROR_TERM_PAT.search(line) and results.termination_status == "UNKNOWN":
                results.termination_status = "ERROR"
                logger.debug(f"Found potential Error Termination signature in line: {line.strip()}")

    except ParsingError:
        results.termination_status = "ERROR"
        # Removed metadata population from _meta
        # final_meta_dict = {k: v for k, v in _meta.items() if v is not None}
        # results.metadata = CalculationMetadata(**final_meta_dict) # type: ignore[arg-type]
        raise
    except Exception as e:
        logger.critical(f"Unexpected error in main parsing loop near line ~{current_line_num}: {e}", exc_info=True)
        results.termination_status = "ERROR"
        results.parsing_errors.append(f"Critical error near line {current_line_num}: {e}")
        # Removed metadata population from _meta
        # final_meta_dict = {k: v for k, v in _meta.items() if v is not None}
        # results.metadata = CalculationMetadata(**final_meta_dict) # type: ignore[arg-type]
        raise ParsingError(f"An unexpected error occurred during parsing: {e}") from e

    # --- Final Checks and Refinements --- #

    # Removed metadata population from _meta
    # final_meta_dict = {k: v for k, v in _meta.items() if v is not None}
    # results.metadata = CalculationMetadata(**final_meta_dict) # type: ignore[arg-type]

    # Example Check: Ensure standard orientation geometry was parsed
    if results.standard_orientation_geometry is None:
        logger.warning("Standard orientation geometry block was not found or parsed.")
        results.parsing_warnings.append("Standard orientation geometry not parsed.")
    if results.input_geometry is None:
        logger.warning("Input geometry block ($molecule) was not found or parsed.")
        results.parsing_warnings.append("Input geometry ($molecule) not parsed.")

    # Refine termination status if still UNKNOWN
    if results.termination_status == "UNKNOWN":
        logger.warning("Termination status unknown after parsing, assuming ERROR.")
        results.termination_status = "ERROR"

    logger.info(
        f"Q-Chem SP parsing finished. Status: {results.termination_status}, Final Energy: {results.final_energy_eh}"
    )

    # Convert mutable data to the final immutable structure
    # CalculationData.from_mutable handles metadata construction now
    return CalculationData.from_mutable(results)

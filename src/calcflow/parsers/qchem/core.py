import re
from collections.abc import Sequence

from calcflow.exceptions import ParsingError

# Import block parsers
from calcflow.parsers.qchem.blocks import (
    GeometryParser,
    MetadataParser,
    MullikenChargesParser,
    MultipoleParser,
    RemBlockParser,
    ScfParser,
)
from calcflow.parsers.qchem.blocks.orbitals import OrbitalParser
from calcflow.parsers.qchem.blocks.smx import SmxBlockParser

# Import TDDFT block parsers
from calcflow.parsers.qchem.blocks.tddft import (
    GroundStateReferenceParser,
    NTODecompositionParser,
    TDAExcitationEnergiesParser,
    TDDFTExcitationEnergiesParser,
    TransitionDensityMatrixParser,
    UnrelaxedExcitedStatePropertiesParser,
)
from calcflow.parsers.qchem.typing import (
    CalculationData,
    LineIterator,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- Regex Patterns (Compile patterns for efficiency) ---
# Energy/Structure Patterns
NUCLEAR_REPULSION_PAT = re.compile(r"^ Nuclear Repulsion Energy =\s+(-?\d+\.\d+)")
# Use 'Total energy =' after SCF as the final SP energy
FINAL_ENERGY_PAT = re.compile(r"^ Total energy =\s+(-?\d+\.\d+)")

# Termination Patterns
# Make the pattern search for the message anywhere on the line, ignoring surrounding characters
NORMAL_TERM_PAT = re.compile(r"^ {8}\* {2}Thank you very much for using Q-Chem\. {2}Have a nice day\. {2}\*$")

ERROR_TERM_PAT = re.compile(r"(ERROR:|error:|aborting|failed)", re.IGNORECASE)

# --- Parser Registry --- #
# Order matters: Parse $rem before geometry/scf which might appear later.
# Metadata can appear anywhere.
PARSER_REGISTRY_SP: Sequence[SectionParser] = [
    MetadataParser(),
    RemBlockParser(),
    SmxBlockParser(),
    GeometryParser(),
    ScfParser(),
    OrbitalParser(),
    MullikenChargesParser(),
    MultipoleParser(),
    # Add other specific block parsers here later for SP if any
]


# TransitionDensityMatrixParser()
# NTOParser()


PARSER_REGISTRY_TDDFT: Sequence[SectionParser] = [
    MetadataParser(),
    RemBlockParser(),
    SmxBlockParser(),  # Often used with TDDFT for solvation
    GeometryParser(),
    ScfParser(),  # SCF is a prerequisite for TDDFT
    # TDDFT specific parsers
    TDAExcitationEnergiesParser(),
    TDDFTExcitationEnergiesParser(),
    GroundStateReferenceParser(),  # Parses specific GS ref data within ESA block
    UnrelaxedExcitedStatePropertiesParser(),
    TransitionDensityMatrixParser(),
    NTODecompositionParser(),
    OrbitalParser(),  # Orbitals are relevant
    # Ground state properties usually appear before TDDFT specific blocks
    MultipoleParser(),  # Main ground state multipoles
    MullikenChargesParser(),  # Main ground state charges
    # TransitionDensityMatrixParser(), # Add when ready
    # NTOParser(), # Add when ready
]


# --- Main Parsing Function (Internal Generic) --- #
def _parse_qchem_generic_output(output: str, parser_registry: Sequence[SectionParser]) -> CalculationData:
    """
    Parses the text output of a Q-Chem calculation using a given parser registry.

    Args:
        output: The string content of the Q-Chem output file.
        parser_registry: A sequence of SectionParser instances to use.

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
                if results.buffered_line is not None:
                    line = results.buffered_line
                    results.buffered_line = None  # Consume the buffered line
                else:
                    line = next(line_iterator)
                    current_line_num += 1
            except StopIteration:
                if results.buffered_line is not None:  # Should not happen if logic is correct
                    logger.error("StopIteration reached with a buffered line pending. This is a bug.")
                    line = results.buffered_line
                    results.buffered_line = None
                    # Continue to process this last buffered line
                else:
                    logger.debug("Core main loop: Reached end of input.")
                    break

            # --- Handle Block Parsing --- #
            parser_found = False
            # Keep track of which parser matched for logging after the loop
            # and for error messages if parse() fails.
            successful_parser_name_for_block = "None"
            match_line_num = -1  # Line number where a parser matched

            for parser in parser_registry:  # Use the passed parser_registry
                current_parser_being_tried = type(parser).__name__
                try:
                    if parser.matches(line, results):
                        match_line_num = current_line_num  # Capture line number of the match
                        successful_parser_name_for_block = current_parser_being_tried

                        logger.info(
                            f"Core dispatch: Line {match_line_num} ('{line.strip()}')"
                            f" MATCHED by {successful_parser_name_for_block}. Calling .parse()."
                        )

                        parser.parse(line_iterator, line, results)

                        logger.info(f"Core dispatch: {successful_parser_name_for_block} .parse() completed.")
                        parser_found = True
                        break  # Only one parser should handle the start of a block
                except ParsingError as e:
                    err_line_ref = match_line_num if match_line_num != -1 else current_line_num
                    logger.error(
                        f"Parser {current_parser_being_tried} failed critically near line {err_line_ref}: {e}",
                        exc_info=True,
                    )
                    raise
                except StopIteration as e:
                    err_line_ref = match_line_num if match_line_num != -1 else current_line_num
                    logger.error(
                        f"Parser {current_parser_being_tried} unexpectedly consumed end of iterator near line {err_line_ref}.",
                        exc_info=True,
                    )
                    raise ParsingError(f"File ended unexpectedly during {current_parser_being_tried} parsing.") from e
                except Exception as e:
                    err_line_ref = match_line_num if match_line_num != -1 else current_line_num
                    logger.error(
                        f"Unexpected error in {current_parser_being_tried} near line {err_line_ref}: {e}", exc_info=True
                    )
                    results.parsing_errors.append(
                        f"Error in {current_parser_being_tried} near line {err_line_ref}: {e}"
                    )
                    raise ParsingError(
                        f"Unexpected error in {current_parser_being_tried} near line {err_line_ref}: {e}"
                    ) from e

            # If a block parser handled the line, move to the next line
            if parser_found:
                continue

            # --- Handle Standalone Information (if not handled by a block parser) --- #

            # Termination Status (checked only if not part of a block)
            if NORMAL_TERM_PAT.search(line):
                results.termination_status = "NORMAL"
                logger.debug("Found Normal Termination signature.")
                # Continue parsing, other info might follow
                continue
            elif ERROR_TERM_PAT.search(line) and results.termination_status == "UNKNOWN":
                results.termination_status = "ERROR"
                logger.debug(f"Found potential Error Termination signature in line: {line.strip()}")
                # Continue parsing, might find more specific errors
                continue

            # Energy Components (checked only if not part of a block)
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
                    # Overwrite if found multiple times, the last one after SCF is desired
                    results.final_energy_eh = float(match_final_energy.group(1))
                    logger.debug(f"Found Potential Final Energy: {results.final_energy_eh}")
                except (ValueError, IndexError) as e:
                    logger.error(f"Could not parse final energy value from line: {line.strip()}", exc_info=True)
                    raise ParsingError("Failed to parse final energy value.") from e
                continue

            # --- Add other standalone pattern checks here if needed ---

    except ParsingError:
        # Ensure status is ERROR if a specific parsing exception occurred
        results.termination_status = "ERROR"
        # Re-raise the original error for clarity
        raise
    except Exception as e:
        logger.critical(
            f"Unexpected critical error in main parsing loop near line ~{current_line_num}: {e}", exc_info=True
        )
        results.termination_status = "ERROR"
        results.parsing_errors.append(f"Critical error near line {current_line_num}: {e}")
        raise ParsingError(f"An unexpected critical error occurred during parsing: {e}") from e

    # --- Final Checks and Refinements --- #

    # Example Check: Ensure standard orientation geometry was parsed
    if results.standard_orientation_geometry is None:
        logger.warning("Standard orientation geometry block was not found or parsed.")
        results.parsing_warnings.append("Standard orientation geometry not parsed.")
    if results.input_geometry is None:
        # This is more critical than standard orientation
        logger.error("Input geometry block ($molecule) was not found or parsed.")
        results.parsing_warnings.append("Input geometry ($molecule) not parsed.")
        # Depending on requirements, this could be a ParsingError
        # raise ParsingError("Input geometry ($molecule) not parsed.")

    # Refine termination status if still UNKNOWN
    if results.termination_status == "UNKNOWN":
        logger.warning("Termination status unknown after parsing, assuming ERROR.")
        results.termination_status = "ERROR"

    logger.info(
        f"Q-Chem SP parsing finished. Status: {results.termination_status}, Final Energy: {results.final_energy_eh}"
    )

    # Convert mutable data to the final immutable structure
    return CalculationData.from_mutable(results)


# --- Public Entry Point for SP Calculations --- #
def parse_qchem_sp_output(output: str) -> CalculationData:
    """
    Parses the text output of a Q-Chem single-point calculation.

    Args:
        output: The string content of the Q-Chem output file.

    Returns:
        A CalculationData object containing the parsed results.
    """
    logger.info("Starting Q-Chem Single Point (SP) output parsing.")
    return _parse_qchem_generic_output(output, PARSER_REGISTRY_SP)


# --- Public Entry Point for TDDFT Calculations --- #
def parse_qchem_tddft_output(output: str) -> CalculationData:
    """
    Parses the text output of a Q-Chem TDDFT calculation.

    Args:
        output: The string content of the Q-Chem output file.

    Returns:
        A CalculationData object containing the parsed results.
    """
    logger.info("Starting Q-Chem TDDFT output parsing.")
    return _parse_qchem_generic_output(output, PARSER_REGISTRY_TDDFT)

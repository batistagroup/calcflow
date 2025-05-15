import re

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.typing import Atom, LineIterator, SectionParser, _MutableCalculationData
from calcflow.utils import logger

# --- Regex Patterns --- #

# Start markers for geometry blocks
INPUT_GEOM_START_PAT = re.compile(r"^\s*\$molecule")
STANDARD_GEOM_START_PAT = re.compile(r"^\s*Standard Nuclear Orientation \(Angstroms\)")

# End markers for geometry blocks
INPUT_GEOM_END_PAT = re.compile(r"^\s*\$end")
STANDARD_GEOM_END_PAT = re.compile(r"^\s*-{60}")  # The line of dashes after the table

# Atom line patterns (adapt based on exact format)
# Input format: Symbol X Y Z
INPUT_ATOM_PAT = re.compile(r"^\s*([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
# Standard Orientation format: Index Symbol X Y Z
STANDARD_ATOM_PAT = re.compile(r"^\s*\d+\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")

# Header/separator lines to skip
INPUT_GEOM_SKIP_PAT = re.compile(r"^\s*\d+\s+\d+\s*$")  # Skips the charge/multiplicity line
STANDARD_GEOM_HEADER_PAT = re.compile(r"^\s*I\s+Atom\s+X\s+Y\s+Z")
SEPARATOR_LINE_PAT = re.compile(r"^\s*-{20,}")  # Matches dashed lines


class GeometryParser(SectionParser):
    """Parses both input ($molecule) and standard orientation geometries."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line starts either the input or standard geometry block."""
        # Only parse if not already parsed to avoid issues with multiple blocks (if any)
        if INPUT_GEOM_START_PAT.search(line) and not current_data.parsed_input_geometry:
            return True
        return (STANDARD_GEOM_START_PAT.search(line) is not None) and not current_data.parsed_standard_geometry

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parse the identified geometry block."""
        atoms: list[Atom] = []
        parsing_input_geom = INPUT_GEOM_START_PAT.search(current_line) is not None
        parsing_standard_geom = STANDARD_GEOM_START_PAT.search(current_line) is not None

        if parsing_input_geom:
            logger.debug("Starting parsing of input geometry ($molecule).")
            # Consume the start line and potentially the charge/multiplicity line
            try:
                line = next(iterator)
                if not INPUT_GEOM_SKIP_PAT.search(line):
                    # If it wasn't the charge/mult line, parse it as an atom
                    match = INPUT_ATOM_PAT.search(line)
                    if match:
                        atoms.append(
                            Atom(
                                symbol=match.group(1),
                                x=float(match.group(2)),
                                y=float(match.group(3)),
                                z=float(match.group(4)),
                            )
                        )
                    else:
                        logger.warning(f"Unexpected line format after $molecule: {line.strip()}")
                # Else: skipped charge/mult line
            except StopIteration as e:
                raise ParsingError("Unexpected end of file after $molecule start.") from e

        elif parsing_standard_geom:
            logger.debug("Starting parsing of standard orientation geometry.")
            # Consume the start line, header, and separator
            try:
                _ = next(iterator)  # Header line (I Atom X Y Z)
                _ = next(iterator)  # Separator line (---)
            except StopIteration as e:
                raise ParsingError("Unexpected end of file after standard orientation header.") from e
        else:
            # This should not happen if matches() worked correctly
            raise ParsingError("GeometryParser.parse called on non-matching line.")

        # --- Loop through atom lines --- #
        while True:
            try:
                line = next(iterator)
            except StopIteration as e:
                logger.error("File ended unexpectedly while parsing geometry block.")
                raise ParsingError(
                    f"Unexpected end of file in geometry block starting near '{current_line.strip()}'"
                ) from e

            # Check for end markers
            if parsing_input_geom and INPUT_GEOM_END_PAT.search(line):
                logger.debug("Found $end, finishing input geometry parsing.")
                break
            if parsing_standard_geom and STANDARD_GEOM_END_PAT.search(line):
                logger.debug("Found standard orientation end marker, finishing parsing.")
                break

            # Parse atom lines
            if parsing_input_geom:
                match = INPUT_ATOM_PAT.search(line)
                if match:
                    atoms.append(
                        Atom(
                            symbol=match.group(1),
                            x=float(match.group(2)),
                            y=float(match.group(3)),
                            z=float(match.group(4)),
                        )
                    )
                else:
                    # Ignore blank lines, log others
                    if line.strip():
                        logger.warning(f"Skipping unexpected line in $molecule block: {line.strip()}")
            elif parsing_standard_geom:
                match = STANDARD_ATOM_PAT.search(line)
                if match:
                    atoms.append(
                        Atom(
                            symbol=match.group(1),
                            x=float(match.group(2)),
                            y=float(match.group(3)),
                            z=float(match.group(4)),
                        )
                    )
                else:
                    # Ignore blank lines, log others
                    if line.strip():
                        logger.warning(f"Skipping unexpected line in standard orientation block: {line.strip()}")

        # --- Store results --- #
        if not atoms:
            logger.warning(f"No atoms found in geometry block starting near '{current_line.strip()}'")
            # Raise error or just warn?
            # raise ParsingError(f"No atoms found in geometry block starting near '{current_line.strip()}'")

        if parsing_input_geom:
            results.input_geometry = atoms
            results.parsed_input_geometry = True
            logger.info(f"Parsed input geometry with {len(atoms)} atoms.")
        elif parsing_standard_geom:
            results.final_geometry = atoms
            results.parsed_standard_geometry = True
            logger.info(f"Parsed standard orientation geometry with {len(atoms)} atoms.")

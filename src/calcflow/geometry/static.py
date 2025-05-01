from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from calcflow.typing import AtomCoords


def _parse_atom_line(line: str, line_num: int, file_path: Path) -> AtomCoords:
    """Parses a single atom line from an XYZ file."""
    parts = line.split()
    if len(parts) != 4:
        raise ValueError(
            f"Invalid XYZ file '{file_path}' at line {line_num}: "
            f"Expected 4 columns (Symbol X Y Z), found {len(parts)}. Line: '{line}'"
        )
    symbol = parts[0]
    try:
        coords = (float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError as e:
        raise ValueError(
            f"Invalid XYZ file '{file_path}' at line {line_num}: Could not parse coordinates. {e}. Line: '{line}'"
        ) from e
    return symbol, coords


def parse_xyz(file: Path | str) -> tuple[int, str, list[AtomCoords]]:
    """Parses an XYZ file, validating its structure and content.

    Reads an XYZ file, extracts the number of atoms, the comment line,
    and parses the atom coordinate lines into structured data.
    Validates atom counts and coordinate formats.

    Args:
        file: Path to the XYZ file.

    Returns:
        A tuple containing:
            - num_atoms (int): The number of atoms declared in the file.
            - comment (str): The comment line from the file.
            - atoms (list[AtomCoords]): A list of tuples, each containing
              the atom symbol (str) and its coordinates (tuple[float, float, float]).

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is invalid.
    """
    file_path = Path(file)
    if not file_path.is_file():
        raise FileNotFoundError(f"XYZ file not found: {file_path}")

    with file_path.open("r") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    if len(raw_lines) < 2:
        raise ValueError(f"Invalid XYZ file '{file_path}': Needs at least 2 lines.")

    try:
        num_atoms = int(raw_lines[0])
    except ValueError as e:
        raise ValueError(
            f"Invalid XYZ file '{file_path}': First line must be an integer. Found: '{raw_lines[0]}'"
        ) from e

    comment = raw_lines[1]
    atom_lines_raw = raw_lines[2:]

    if len(atom_lines_raw) != num_atoms:
        raise ValueError(
            f"Invalid XYZ file '{file_path}': Declared atom count ({num_atoms}) "
            f"does not match actual atom lines found ({len(atom_lines_raw)})."
        )

    atoms: list[AtomCoords] = []
    for i, line in enumerate(atom_lines_raw):
        # Line number in the original file context (add 3: count, comment, 1-based index)
        line_number_in_file = i + 3
        try:
            atom_data = _parse_atom_line(line, line_number_in_file, file_path)
            atoms.append(atom_data)
        except ValueError as e:
            # Re-raise with context if needed, or handle differently
            raise e

    return num_atoms, comment, atoms


@dataclass(frozen=True)
class Geometry:
    """Represents a molecular geometry structure.

    Attributes:
        num_atoms: The number of atoms in the geometry.
        comment: The comment line associated with the geometry.
        atoms: A sequence of tuples, each containing the atom symbol (str)
               and its coordinates (tuple[float, float, float]).
               The order matches the input file.
    """

    num_atoms: int
    comment: str
    atoms: Sequence[AtomCoords]

    @classmethod
    def from_xyz_file(cls, file: Path | str) -> "Geometry":
        """Create Geometry instance from an xyz file.

        Args:
            file: Path to xyz file.

        Returns:
            Geometry instance with coordinates from the file.
        """
        num_atoms, comment, atoms = parse_xyz(file)
        return cls(num_atoms=num_atoms, comment=comment, atoms=atoms)

    @property
    def unique_elements(self) -> set[str]:
        """Returns a set of unique element symbols present in the geometry."""
        return {symbol.upper() for symbol, _ in self.atoms}

    def __repr__(self) -> str:
        """Returns a concise representation of the Geometry object."""
        # Avoid printing potentially very long atoms list
        return f"{self.__class__.__name__}(num_atoms={self.num_atoms}, comment='{self.comment}')"

    def __str__(self) -> str:
        """Returns the geometry in XYZ file format."""
        return f"{self.num_atoms}\n{self.comment}\n{self.get_coordinate_block()}"

    def get_coordinate_block(self) -> str:
        """Returns a string block of the coordinate lines formatted for XYZ."""
        atom_lines = [
            # Format numbers with sufficient precision and alignment
            f"{symbol:<3} {coords[0]:>15.8f} {coords[1]:>15.8f} {coords[2]:>15.8f}"
            for symbol, coords in self.atoms
        ]
        return "\n".join(atom_lines)

    def to_xyz_file(self, file_path: Path | str) -> None:
        """Writes the geometry to an XYZ format file.

        Args:
            file_path: The path where the XYZ file will be saved.
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(str(self))

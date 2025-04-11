from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from calcflow.geometry.static import Geometry, _parse_atom_line
from calcflow.typing import AtomCoords


def _parse_single_frame(
    num_atoms: int, comment_line: str, atom_lines: list[str], file_path: Path, frame_start_line: int
) -> Geometry:
    """Parses a single geometry frame from pre-split lines."""
    if len(atom_lines) != num_atoms:
        raise ValueError(
            f"Invalid frame in '{file_path}' starting near line {frame_start_line}: "
            f"Expected {num_atoms} atom lines based on header, found {len(atom_lines)}."
        )

    atoms: list[AtomCoords] = []
    for i, line in enumerate(atom_lines):
        # Line number within the context of the *entire* file
        line_number_in_file = frame_start_line + 2 + i
        try:
            atom_data = _parse_atom_line(line, line_number_in_file, file_path)
            atoms.append(atom_data)
        except ValueError as e:
            # Re-raise with context
            raise ValueError(f"Error parsing frame in '{file_path}' starting near line {frame_start_line}: {e}") from e

    return Geometry(num_atoms=num_atoms, comment=comment_line, atoms=atoms)


def _iter_xyz_trajectory_frames(f: TextIO, file_path: Path) -> Iterator[Geometry]:
    """Yields Geometry objects frame by frame from an extended XYZ trajectory file stream."""
    line_num = 0
    while True:
        frame_start_line = line_num + 1

        # 1. Read num_atoms line
        num_atoms_line = f.readline()
        if not num_atoms_line:
            break  # End of file
        line_num += 1
        num_atoms_line = num_atoms_line.strip()
        if not num_atoms_line:  # Handle potential empty lines between frames
            continue

        try:
            num_atoms = int(num_atoms_line)
            if num_atoms <= 0:
                raise ValueError(f"Atom count must be positive, found {num_atoms}")
        except ValueError as e:
            raise ValueError(
                f"Invalid XYZ trajectory '{file_path}' at line {line_num}: "
                f"Could not parse atom count. Error: {e}. Line: '{num_atoms_line}'"
            ) from e

        # 2. Read comment line
        comment_line = f.readline()
        if not comment_line:
            raise ValueError(
                f"Invalid XYZ trajectory '{file_path}': Unexpected EOF after atom count at line {line_num}."
            )
        line_num += 1
        comment = comment_line.strip()

        # 3. Read atom coordinate lines
        atom_lines: list[str] = []
        for _ in range(num_atoms):
            atom_line = f.readline()
            if not atom_line:
                raise ValueError(
                    f"Invalid XYZ trajectory '{file_path}': Unexpected EOF while reading atom coordinates "
                    f"for frame starting at line {frame_start_line}. Expected {num_atoms} atom lines."
                )
            line_num += 1
            atom_lines.append(atom_line.strip())

        # 4. Parse and yield the frame
        yield _parse_single_frame(num_atoms, comment, atom_lines, file_path, frame_start_line)


@dataclass(frozen=True)
class Trajectory:
    """Represents a time-ordered sequence of molecular geometries.

    Attributes:
        frames: A sequence of Geometry objects representing the trajectory snapshots.
    """

    frames: Sequence[Geometry]

    @classmethod
    def from_xyz_trj_file(cls, file: Path | str) -> "Trajectory":
        """Create a Trajectory instance from an extended XYZ trajectory file (_trj.xyz).

        Args:
            file: Path to the trajectory file.

        Returns:
            Trajectory instance containing all frames from the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file format is invalid or inconsistent.
        """
        file_path = Path(file)
        if not file_path.is_file():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")

        try:
            with file_path.open("r") as f:
                all_frames = list(_iter_xyz_trajectory_frames(f, file_path))
        except ValueError as e:
            # Add file context to parsing errors
            raise ValueError(f"Error parsing trajectory file '{file_path}': {e}") from e

        if not all_frames:
            raise ValueError(f"Trajectory file '{file_path}' contains no valid frames.")

        # Optional: Add consistency check (e.g., all frames have same num_atoms)
        first_frame_atoms = all_frames[0].num_atoms
        if not all(frame.num_atoms == first_frame_atoms for frame in all_frames):
            raise ValueError(f"Inconsistent number of atoms across frames in trajectory file '{file_path}'.")

        return cls(frames=all_frames)

    def __len__(self) -> int:
        """Returns the number of frames in the trajectory."""
        return len(self.frames)

    def __getitem__(self, index: int) -> Geometry:
        """Returns the Geometry frame at the specified index."""
        return self.frames[index]

    def __iter__(self) -> Iterator[Geometry]:
        """Returns an iterator over the Geometry frames."""
        return iter(self.frames)

    def __repr__(self) -> str:
        """Returns a multi-line representation showing each frame."""
        if not self.frames:
            return f"{self.__class__.__name__}(frames=[])"

        frame_reprs = [f"    {repr(frame)}," for frame in self.frames]
        return f"{self.__class__.__name__}(frames=[\n" + "\n".join(frame_reprs) + "\n])"

from pathlib import Path

import pytest

from calcflow.geometry.static import Geometry
from calcflow.geometry.trajectory import Trajectory, _parse_single_frame

# --- Test Data and Fixtures ---


@pytest.fixture
def geom1() -> Geometry:
    """Fixture for a sample Geometry object (Frame 1)."""
    return Geometry(num_atoms=2, comment="Frame 1", atoms=[("H", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.0))])


@pytest.fixture
def geom2() -> Geometry:
    """Fixture for a sample Geometry object (Frame 2)."""
    return Geometry(num_atoms=2, comment="Frame 2", atoms=[("H", (0.1, 0.0, 0.0)), ("O", (0.1, 0.0, 1.1))])


@pytest.fixture
def geom3() -> Geometry:
    """Fixture for a sample Geometry object (Frame 3)."""
    return Geometry(num_atoms=2, comment="Frame 3", atoms=[("H", (0.2, 0.0, 0.0)), ("O", (0.2, 0.0, 1.2))])


@pytest.fixture
def geom_with_orca_energy() -> Geometry:
    """Fixture for a Geometry with ORCA energy format."""
    return Geometry(
        num_atoms=2,
        comment="Coordinates from ORCA-job opt E -981.614502119079",
        atoms=[("H", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.0))],
        energy=-981.614502119079,
    )


@pytest.fixture
def geom_with_generic_energy() -> Geometry:
    """Fixture for a Geometry with generic energy format."""
    return Geometry(
        num_atoms=2,
        comment="Energy = -100.5 Hartree",
        atoms=[("H", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.0))],
        energy=-100.5,
    )


@pytest.fixture
def sample_trajectory(geom1: Geometry, geom2: Geometry, geom3: Geometry) -> Trajectory:
    """Fixture for a sample Trajectory object with three frames."""
    return Trajectory(frames=[geom1, geom2, geom3])


# --- Test Cases ---

# Test Trajectory.from_xyz_trj_file


def test_from_xyz_trj_file_valid_multi_frame(tmp_path: Path, geom1: Geometry, geom2: Geometry) -> None:
    """Test loading a valid multi-frame XYZ trajectory file."""
    # Arrange
    file_content = """\
2
Frame 1
H 0.0 0.0 0.0
O 0.0 0.0 1.0
2
Frame 2
H 0.1 0.0 0.0
O 0.1 0.0 1.1
"""
    trj_file = tmp_path / "valid_multi.xyz"
    trj_file.write_text(file_content)

    # Act
    trajectory = Trajectory.from_xyz_trj_file(trj_file)

    # Assert
    assert len(trajectory) == 2
    assert trajectory[0] == geom1
    assert trajectory[1] == geom2
    assert trajectory.frames[0].atoms == geom1.atoms
    assert trajectory.frames[1].atoms == geom2.atoms


def test_from_xyz_trj_file_valid_single_frame(tmp_path: Path, geom1: Geometry) -> None:
    """Test loading a valid single-frame XYZ trajectory file."""
    # Arrange
    file_content = """\
2
Frame 1
H 0.0 0.0 0.0
O 0.0 0.0 1.0
"""
    trj_file = tmp_path / "valid_single.xyz"
    trj_file.write_text(file_content)

    # Act
    trajectory = Trajectory.from_xyz_trj_file(trj_file)

    # Assert
    assert len(trajectory) == 1
    assert trajectory[0] == geom1


def test_from_xyz_trj_file_valid_with_empty_lines(tmp_path: Path, geom1: Geometry, geom2: Geometry) -> None:
    """Test loading a valid trajectory file with empty lines between frames."""
    # Arrange
    file_content = """\
2
Frame 1
H 0.0 0.0 0.0
O 0.0 0.0 1.0

2
Frame 2
H 0.1 0.0 0.0
O 0.1 0.0 1.1

"""  # Note trailing newline
    trj_file = tmp_path / "valid_empty_lines.xyz"
    trj_file.write_text(file_content)

    # Act
    trajectory = Trajectory.from_xyz_trj_file(trj_file)

    # Assert
    assert len(trajectory) == 2
    assert trajectory[0] == geom1
    assert trajectory[1] == geom2


def test_from_xyz_trj_file_not_found() -> None:
    """Test FileNotFoundError when the trajectory file does not exist."""
    # Arrange
    non_existent_file = Path("non_existent_trajectory.xyz")

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Trajectory file not found"):
        Trajectory.from_xyz_trj_file(non_existent_file)


def test_from_xyz_trj_file_empty_file(tmp_path: Path) -> None:
    """Test ValueError when the trajectory file is empty."""
    # Arrange
    trj_file = tmp_path / "empty.xyz"
    trj_file.touch()  # Create empty file

    # Act & Assert
    with pytest.raises(ValueError, match="contains no valid frames"):
        Trajectory.from_xyz_trj_file(trj_file)


@pytest.mark.parametrize(
    "content, error_match",
    [
        ("invalid_count\nComment\nH 0 0 0", "Could not parse atom count"),  # Non-integer count
        ("0\nComment\n", "Atom count must be positive"),  # Zero count
        ("-1\nComment\n", "Atom count must be positive"),  # Negative count
        ("1", "Unexpected EOF after atom count"),  # EOF after count
        ("1\nComment", "Unexpected EOF while reading atom coordinates"),  # EOF after comment
        ("2\nComment\nH 0 0 0", "Unexpected EOF while reading atom coordinates"),  # Mismatched count (less)
        ("1\nComment\nH 0 0 0\nO 1 1 1", "Could not parse atom count"),  # Mismatched count (more)
        ("1\nComment\nH 0 0", "Expected 4 columns"),
    ],
    ids=[
        "invalid_count_format",
        "zero_count",
        "negative_count",
        "eof_after_count",
        "eof_after_comment",
        "mismatched_count_less",
        "mismatched_count_more",
        "invalid_coord_format",
    ],
)
def test_from_xyz_trj_file_invalid_format(tmp_path: Path, content: str, error_match: str) -> None:
    """Test various invalid file format scenarios."""
    # Arrange
    trj_file = tmp_path / "invalid.xyz"
    trj_file.write_text(content)

    # Act & Assert
    with pytest.raises(ValueError, match=error_match):
        Trajectory.from_xyz_trj_file(trj_file)


def test_from_xyz_trj_file_mismatched_count_explicit_check(tmp_path: Path) -> None:
    """Test explicit check for mismatched atom count within _parse_single_frame.

    Note: With the current reading logic, an EOF error occurs before this check
    is typically reachable when fewer lines than expected are present.
    This test case reflects the actual error encountered.
    """
    # Arrange
    file_content = """\
2
Comment
H 0 0 0
"""  # Only one atom line provided, header says 2
    trj_file = tmp_path / "invalid_mismatch.xyz"
    trj_file.write_text(file_content)

    # Act & Assert
    # Corrected: The actual error raised is EOF during coordinate reading.
    with pytest.raises(ValueError, match="Unexpected EOF while reading atom coordinates"):
        Trajectory.from_xyz_trj_file(trj_file)


def test_from_xyz_trj_file_inconsistent_atoms_across_frames(tmp_path: Path) -> None:
    """Test ValueError when frames have inconsistent numbers of atoms."""
    # Arrange
    file_content = """\
2
Frame 1
H 0.0 0.0 0.0
O 0.0 0.0 1.0
3
Frame 2 - 3 atoms!
H 0.1 0.0 0.0
O 0.1 0.0 1.1
C 0.0 1.0 0.0
"""
    trj_file = tmp_path / "inconsistent.xyz"
    trj_file.write_text(file_content)

    # Act & Assert
    with pytest.raises(ValueError, match="Inconsistent number of atoms across frames"):
        Trajectory.from_xyz_trj_file(trj_file)


# Test Trajectory Instance Methods


def test_trajectory_len(sample_trajectory: Trajectory, geom1: Geometry) -> None:
    """Test the __len__ method."""
    # Arrange
    empty_traj = Trajectory(frames=[])
    single_traj = Trajectory(frames=[geom1])

    # Act & Assert
    assert len(empty_traj) == 0
    assert len(single_traj) == 1
    assert len(sample_trajectory) == 3


def test_trajectory_getitem(sample_trajectory: Trajectory, geom1: Geometry, geom2: Geometry, geom3: Geometry) -> None:
    """Test the __getitem__ method for valid indices and IndexError."""
    # Arrange (handled by fixtures)

    # Act & Assert
    assert sample_trajectory[0] == geom1
    assert sample_trajectory[1] == geom2
    assert sample_trajectory[2] == geom3
    assert sample_trajectory[-1] == geom3  # Test negative indexing
    assert sample_trajectory[-3] == geom1

    with pytest.raises(IndexError):
        _ = sample_trajectory[3]
    with pytest.raises(IndexError):
        _ = sample_trajectory[-4]


def test_trajectory_iter(sample_trajectory: Trajectory, geom1: Geometry, geom2: Geometry, geom3: Geometry) -> None:
    """Test the __iter__ method."""
    # Arrange (handled by fixtures)
    expected_frames = [geom1, geom2, geom3]

    # Act
    iterated_frames = list(sample_trajectory)  # Consume the iterator

    # Assert
    assert iterated_frames == expected_frames

    # Test iteration multiple times
    count = 0
    for frame in sample_trajectory:
        assert frame == expected_frames[count]
        count += 1
    assert count == 3


# Test Trajectory Representation


def test_trajectory_repr_empty() -> None:
    """Test the __repr__ method for an empty trajectory."""
    # Arrange
    empty_traj = Trajectory(frames=[])

    # Act
    repr_str = repr(empty_traj)

    # Assert
    assert repr_str == "Trajectory(frames=[])"


def test_trajectory_repr_with_frames(geom1: Geometry, geom2: Geometry) -> None:
    """Test the __repr__ method for a trajectory with frames."""
    # Arrange
    traj = Trajectory(frames=[geom1, geom2])
    expected_repr = f"Trajectory(frames=[\n    {repr(geom1)},\n    {repr(geom2)},\n])"

    # Act
    repr_str = repr(traj)

    # Assert
    assert repr_str == expected_repr


# Test Energy Parsing in Trajectory Files


def test_from_xyz_trj_file_with_orca_energy_format(tmp_path: Path) -> None:
    """Test parsing trajectory file with ORCA energy format in comment lines."""
    # Arrange
    file_content = """\
2
Coordinates from ORCA-job opt E -981.614502119079
H 0.0 0.0 0.0
O 0.0 0.0 1.0
2
Coordinates from ORCA-job opt E -981.614197297592
H 0.1 0.0 0.0
O 0.1 0.0 1.1
"""
    trj_file = tmp_path / "orca_energy.xyz"
    trj_file.write_text(file_content)

    # Act
    trajectory = Trajectory.from_xyz_trj_file(trj_file)

    # Assert
    assert len(trajectory) == 2
    assert trajectory[0].energy == -981.614502119079
    assert trajectory[1].energy == -981.614197297592
    assert trajectory[0].comment == "Coordinates from ORCA-job opt E -981.614502119079"
    assert trajectory[1].comment == "Coordinates from ORCA-job opt E -981.614197297592"


def test_from_xyz_trj_file_with_orca_different_job_types(tmp_path: Path) -> None:
    """Test parsing ORCA trajectory files with different job types (opt, freq, etc.)."""
    # Arrange
    file_content = """\
2
Coordinates from ORCA-job opt E -981.614502119079
H 0.0 0.0 0.0
O 0.0 0.0 1.0
2
Coordinates from ORCA-job freq E -863.785941254139
H 0.1 0.0 0.0
O 0.1 0.0 1.1
2
Coordinates from ORCA-job sp E -500.123456789
H 0.2 0.0 0.0
O 0.2 0.0 1.2
"""
    trj_file = tmp_path / "orca_job_types.xyz"
    trj_file.write_text(file_content)

    # Act
    trajectory = Trajectory.from_xyz_trj_file(trj_file)

    # Assert
    assert len(trajectory) == 3
    assert trajectory[0].energy == -981.614502119079  # opt job
    assert trajectory[1].energy == -863.785941254139  # freq job
    assert trajectory[2].energy == -500.123456789  # sp job


def test_from_xyz_trj_file_with_generic_energy_formats(tmp_path: Path) -> None:
    """Test parsing trajectory file with various generic energy formats."""
    # Arrange
    file_content = """\
2
Energy = -100.5 Hartree
H 0.0 0.0 0.0
O 0.0 0.0 1.0
2
E: -200.123456
H 0.1 0.0 0.0
O 0.1 0.0 1.1
2
energy=-300.789e-2
H 0.2 0.0 0.0
O 0.2 0.0 1.2
"""
    trj_file = tmp_path / "generic_energy.xyz"
    trj_file.write_text(file_content)

    # Act
    trajectory = Trajectory.from_xyz_trj_file(trj_file)

    # Assert
    assert len(trajectory) == 3
    assert trajectory[0].energy == -100.5
    assert trajectory[1].energy == -200.123456
    assert trajectory[2].energy == -3.00789  # -300.789e-2


def test_from_xyz_trj_file_without_energy_backward_compatibility(tmp_path: Path) -> None:
    """Test that trajectory files without energy still work (backward compatibility)."""
    # Arrange
    file_content = """\
2
Frame 1 - no energy here
H 0.0 0.0 0.0
O 0.0 0.0 1.0
2
Just a regular comment
H 0.1 0.0 0.0
O 0.1 0.0 1.1
"""
    trj_file = tmp_path / "no_energy.xyz"
    trj_file.write_text(file_content)

    # Act
    trajectory = Trajectory.from_xyz_trj_file(trj_file)

    # Assert
    assert len(trajectory) == 2
    assert trajectory[0].energy is None
    assert trajectory[1].energy is None
    assert trajectory[0].comment == "Frame 1 - no energy here"
    assert trajectory[1].comment == "Just a regular comment"


def test_from_xyz_trj_file_mixed_energy_and_no_energy(tmp_path: Path) -> None:
    """Test trajectory file with some frames having energy and others not."""
    # Arrange
    file_content = """\
2
Coordinates from ORCA-job opt E -981.614502119079
H 0.0 0.0 0.0
O 0.0 0.0 1.0
2
Frame without energy
H 0.1 0.0 0.0
O 0.1 0.0 1.1
2
Energy = -500.25
H 0.2 0.0 0.0
O 0.2 0.0 1.2
"""
    trj_file = tmp_path / "mixed_energy.xyz"
    trj_file.write_text(file_content)

    # Act
    trajectory = Trajectory.from_xyz_trj_file(trj_file)

    # Assert
    assert len(trajectory) == 3
    assert trajectory[0].energy == -981.614502119079
    assert trajectory[1].energy is None
    assert trajectory[2].energy == -500.25


def test_from_xyz_trj_file_scientific_notation_energy(tmp_path: Path) -> None:
    """Test parsing energy values in scientific notation."""
    # Arrange
    file_content = """\
2
Coordinates from ORCA-job opt E -9.81614502119079e+2
H 0.0 0.0 0.0
O 0.0 0.0 1.0
2
Energy = 1.5E-4
H 0.1 0.0 0.0
O 0.1 0.0 1.1
"""
    trj_file = tmp_path / "scientific_energy.xyz"
    trj_file.write_text(file_content)

    # Act
    trajectory = Trajectory.from_xyz_trj_file(trj_file)

    # Assert
    assert len(trajectory) == 2
    assert trajectory[0].energy == -981.614502119079  # -9.81614502119079e+2
    assert trajectory[1].energy == 0.00015  # 1.5E-4


def test_from_xyz_trj_file_invalid_energy_format_ignored(tmp_path: Path) -> None:
    """Test that invalid energy formats are gracefully ignored (energy set to None)."""
    # Arrange
    file_content = """\
2
Energy = not_a_number
H 0.0 0.0 0.0
O 0.0 0.0 1.0
2
E = 
H 0.1 0.0 0.0
O 0.1 0.0 1.1
2
Multiple E = 100 E = 200 values
H 0.2 0.0 0.0
O 0.2 0.0 1.2
"""
    trj_file = tmp_path / "invalid_energy.xyz"
    trj_file.write_text(file_content)

    # Act
    trajectory = Trajectory.from_xyz_trj_file(trj_file)

    # Assert
    assert len(trajectory) == 3
    assert trajectory[0].energy is None  # Invalid format ignored
    assert trajectory[1].energy is None  # Empty value ignored
    assert trajectory[2].energy == 100.0  # First valid match found


# Test _parse_single_frame Function Directly


def test_parse_single_frame_atom_count_mismatch() -> None:
    """Test _parse_single_frame raises ValueError when atom line count doesn't match expected."""
    # Arrange
    num_atoms = 3
    comment_line = "Test frame"
    atom_lines = ["H 0.0 0.0 0.0", "O 0.0 0.0 1.0"]  # Only 2 lines, but num_atoms says 3

    # Act & Assert
    with pytest.raises(
        ValueError,
        match=r"Expected 3 atom lines based on header, found 2",
    ):
        _parse_single_frame(num_atoms, comment_line, atom_lines)


def test_parse_single_frame_too_many_atom_lines() -> None:
    """Test _parse_single_frame raises ValueError when more atom lines than expected."""
    # Arrange
    num_atoms = 2
    comment_line = "Test frame"
    atom_lines = ["H 0.0 0.0 0.0", "O 0.0 0.0 1.0", "C 1.0 1.0 1.0"]  # 3 lines, but num_atoms says 2

    # Act & Assert
    with pytest.raises(
        ValueError,
        match=r"Expected 2 atom lines based on header, found 3",
    ):
        _parse_single_frame(num_atoms, comment_line, atom_lines)

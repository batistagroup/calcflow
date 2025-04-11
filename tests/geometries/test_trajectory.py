from pathlib import Path

import pytest

from calcflow.geometry.static import Geometry
from calcflow.geometry.trajectory import Trajectory

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

from pathlib import Path

import pytest

from calcflow.geometry.static import Geometry, parse_xyz
from calcflow.typing import AtomCoords

# --- Fixtures ---


@pytest.fixture
def valid_xyz_content() -> str:
    """Provides content of a valid XYZ file, formatted as expected from Geometry output."""
    # This now matches the output format of Geometry.__str__ / get_coordinate_block
    # Whitespace must precisely match the f-string output:
    # f"{symbol:<3} {coords[0]:>15.8f} {coords[1]:>15.8f} {coords[2]:>15.8f}"
    return """3
Water molecule frame 0
O        0.00000000      0.00000000      0.00000000
H        0.75716000      0.58624000      0.00000000
H       -0.75716000      0.58624000      0.00000000
"""


@pytest.fixture
def valid_xyz_data() -> tuple[int, str, list[AtomCoords]]:
    """Provides the expected parsed data from valid_xyz_content."""
    num_atoms = 3
    comment = "Water molecule frame 0"
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.75716, 0.58624, 0.0)),
        ("H", (-0.75716, 0.58624, 0.0)),
    ]
    return num_atoms, comment, atoms


@pytest.fixture
def xyz_file(tmp_path: Path, valid_xyz_content: str) -> Path:
    """Creates a temporary valid XYZ file and returns its path."""
    file_path = tmp_path / "test_water.xyz"
    file_path.write_text(valid_xyz_content)
    return file_path


# We need a fixture for the *original* input format file for testing from_xyz_file
@pytest.fixture
def input_xyz_content() -> str:
    """Provides content of a typical input XYZ file (might differ slightly from output format)."""
    return """3
Water molecule frame 0
O  0.00000  0.00000  0.00000
H  0.75716  0.58624  0.00000
H -0.75716  0.58624  0.00000
"""


@pytest.fixture
def input_xyz_file(tmp_path: Path, input_xyz_content: str) -> Path:
    """Creates a temporary XYZ file with typical input formatting."""
    file_path = tmp_path / "input_water.xyz"
    file_path.write_text(input_xyz_content)
    return file_path


# --- Tests for parse_xyz ---


def test_parse_xyz_valid_file(input_xyz_file: Path, valid_xyz_data: tuple[int, str, list[AtomCoords]]) -> None:
    """Tests parsing a valid XYZ file (using typical input format)."""
    # Arrange
    expected_num_atoms, expected_comment, expected_atoms = valid_xyz_data

    # Act
    num_atoms, comment, atoms = parse_xyz(input_xyz_file)

    # Assert
    assert num_atoms == expected_num_atoms
    assert comment == expected_comment
    assert len(atoms) == expected_num_atoms
    for actual_atom, expected_atom in zip(atoms, expected_atoms, strict=False):
        assert actual_atom[0] == expected_atom[0]  # Symbol
        assert actual_atom[1] == pytest.approx(expected_atom[1])  # Coords


def test_parse_xyz_file_not_found() -> None:
    """Tests parsing a non-existent file."""
    # Arrange
    non_existent_file = Path("non_existent_file.xyz")

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="XYZ file not found"):
        parse_xyz(non_existent_file)


def test_parse_xyz_invalid_atom_count_format(tmp_path: Path) -> None:
    """Tests parsing an XYZ file with a non-integer atom count."""
    # Arrange
    content = "three\nComment\nO 0 0 0"
    file_path = tmp_path / "invalid_count.xyz"
    file_path.write_text(content)

    # Act / Assert
    with pytest.raises(ValueError, match="First line must be an integer"):
        parse_xyz(file_path)


def test_parse_xyz_insufficient_lines(tmp_path: Path) -> None:
    """Tests parsing an XYZ file with fewer than 2 lines."""
    # Arrange
    content = "1"  # Only one line
    file_path = tmp_path / "short.xyz"
    file_path.write_text(content)

    # Act / Assert
    with pytest.raises(ValueError, match="Needs at least 2 lines"):
        parse_xyz(file_path)


def test_parse_xyz_mismatched_atom_count(tmp_path: Path) -> None:
    """Tests parsing an XYZ file where declared count != actual lines."""
    # Arrange
    content = """2
Comment line - should be 3 atoms
O 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.0 1.0 0.0
"""
    file_path = tmp_path / "mismatch.xyz"
    file_path.write_text(content)

    # Act / Assert
    # Use raw string for regex pattern
    with pytest.raises(ValueError, match=r"Declared atom count \(2\) does not match actual atom lines found \(3\)"):
        parse_xyz(file_path)


def test_parse_xyz_invalid_coordinate_format(tmp_path: Path) -> None:
    """Tests parsing an XYZ file with non-float coordinates."""
    # Arrange
    content = """1
Comment line
O 0.0 zero 0.0
"""
    file_path = tmp_path / "bad_coords.xyz"
    file_path.write_text(content)

    # Act / Assert
    with pytest.raises(ValueError, match="Could not parse coordinates"):
        parse_xyz(file_path)


def test_parse_xyz_invalid_line_format(tmp_path: Path) -> None:
    """Tests parsing an XYZ file with incorrect number of columns."""
    # Arrange
    content = """1
Comment line
O 0.0 0.0
"""  # Missing Z coordinate
    file_path = tmp_path / "bad_line.xyz"
    file_path.write_text(content)

    # Act / Assert
    with pytest.raises(ValueError, match="Expected 4 columns"):
        parse_xyz(file_path)


def test_parse_xyz_handles_extra_empty_lines(tmp_path: Path, valid_xyz_data: tuple[int, str, list[AtomCoords]]) -> None:
    """Tests parsing a valid XYZ file with extra empty lines at the end."""
    # Arrange
    content = """3

Water molecule frame 0

O  0.00000  0.00000  0.00000

H  0.75716  0.58624  0.00000
H -0.75716  0.58624  0.00000


"""  # Extra empty lines
    file_path = tmp_path / "empty_lines.xyz"
    file_path.write_text(content)
    expected_num_atoms, expected_comment, expected_atoms = valid_xyz_data

    # Act
    num_atoms, comment, atoms = parse_xyz(file_path)

    # Assert
    assert num_atoms == expected_num_atoms
    assert comment == expected_comment
    assert len(atoms) == expected_num_atoms
    for actual_atom, expected_atom in zip(atoms, expected_atoms, strict=False):
        assert actual_atom[0] == expected_atom[0]
        assert actual_atom[1] == pytest.approx(expected_atom[1])


# --- Tests for Geometry class ---


@pytest.fixture
def geometry_instance(valid_xyz_data: tuple[int, str, list[AtomCoords]]) -> Geometry:
    """Creates a Geometry instance from valid parsed data."""
    num_atoms, comment, atoms = valid_xyz_data
    return Geometry(num_atoms=num_atoms, comment=comment, atoms=atoms)


def test_geometry_from_xyz_file(input_xyz_file: Path, valid_xyz_data: tuple[int, str, list[AtomCoords]]) -> None:
    """Tests creating a Geometry instance using from_xyz_file (with typical input format)."""
    # Arrange
    expected_num_atoms, expected_comment, expected_atoms = valid_xyz_data

    # Act
    # Use the fixture with the original input format
    geometry = Geometry.from_xyz_file(input_xyz_file)

    # Assert
    assert isinstance(geometry, Geometry)
    assert geometry.num_atoms == expected_num_atoms
    assert geometry.comment == expected_comment
    assert len(geometry.atoms) == expected_num_atoms
    for actual_atom, expected_atom in zip(geometry.atoms, expected_atoms, strict=False):
        assert actual_atom[0] == expected_atom[0]
        assert actual_atom[1] == pytest.approx(expected_atom[1])


def test_geometry_str_representation(geometry_instance: Geometry, valid_xyz_content: str) -> None:
    """Tests the __str__ method output matches the *expected output* format."""
    # Arrange
    # Expected content now comes from the updated valid_xyz_content fixture
    expected_output = valid_xyz_content.strip()

    # Act
    actual_output = str(geometry_instance)

    # Assert
    assert actual_output.splitlines() == expected_output.splitlines()


def test_geometry_get_coordinate_block(geometry_instance: Geometry) -> None:
    """Tests the get_coordinate_block method output."""
    # Arrange
    # Generate the expected block dynamically using the same logic as implementation
    expected_lines = [
        f"{symbol:<3} {coords[0]:>15.8f} {coords[1]:>15.8f} {coords[2]:>15.8f}"
        for symbol, coords in geometry_instance.atoms
    ]
    expected_block = "\n".join(expected_lines)

    # Act
    actual_block = geometry_instance.get_coordinate_block()

    # Assert
    assert actual_block.splitlines() == expected_block.splitlines()


def test_geometry_to_xyz_file(geometry_instance: Geometry, tmp_path: Path, valid_xyz_content: str) -> None:
    """Tests writing the geometry to an XYZ file, matching expected output format."""
    # Arrange
    output_file = tmp_path / "output.xyz"
    # Expected content comes from the updated valid_xyz_content fixture
    expected_content = valid_xyz_content.strip()

    # Act
    geometry_instance.to_xyz_file(output_file)

    # Assert
    assert output_file.is_file()
    actual_content = output_file.read_text().strip()
    assert actual_content.splitlines() == expected_content.splitlines()

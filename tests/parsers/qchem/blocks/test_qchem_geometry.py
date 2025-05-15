import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.blocks.geometry import GeometryParser
from calcflow.parsers.qchem.typing import Atom, _MutableCalculationData


@pytest.fixture
def parser() -> GeometryParser:
    """Fixture for the GeometryParser."""
    return GeometryParser()


@pytest.fixture
def initial_data() -> _MutableCalculationData:
    """Provides a fresh _MutableCalculationData instance for each test."""
    return _MutableCalculationData(raw_output="")


# === Test Data ===

INPUT_GEOM_BLOCK_VALID = [
    "$molecule",
    "0 1",
    "H        1.36499000      1.69385000     -0.19748000",
    "O        2.32877000      1.56294000     -0.04168000",
    "H        2.70244000      1.31157000     -0.91665000",
    "$end",
    "Some other line",
]

STANDARD_GEOM_BLOCK_VALID = [
    "             Standard Nuclear Orientation (Angstroms)",
    "    I     Atom           X                Y                Z",
    " ----------------------------------------------------------------",
    "    1      H       1.3649900000     1.6938500000    -0.1974800000",
    "    2      O       2.3287700000     1.5629400000    -0.0416800000",
    "    3      H       2.7024400000     1.3115700000    -0.9166500000",
    " ----------------------------------------------------------------",
    " Nuclear Repulsion Energy =",
]

INPUT_GEOM_BLOCK_UNTERMINATED = [
    "$molecule",
    "0 1",
    "C  0.0 0.0 0.0",
    # Missing $end
]

STANDARD_GEOM_BLOCK_UNTERMINATED = [
    "             Standard Nuclear Orientation (Angstroms)",
    "    I     Atom           X                Y                Z",
    " ----------------------------------------------------------------",
    "    1      C       0.0 0.0 0.0",
    # Missing closing dashes
]

INPUT_GEOM_BLOCK_NO_ATOMS = [
    "$molecule",
    "0 1",
    # No atom lines
    "$end",
]

STANDARD_GEOM_BLOCK_NO_ATOMS = [
    "             Standard Nuclear Orientation (Angstroms)",
    "    I     Atom           X                Y                Z",
    " ----------------------------------------------------------------",
    # No atom lines
    " ----------------------------------------------------------------",
]

INPUT_GEOM_UNEXPECTED_LINE = [
    "$molecule",
    "0 1",
    "H        1.36499000      1.69385000     -0.19748000",
    "This is an unexpected line",
    "O        2.32877000      1.56294000     -0.04168000",
    "$end",
]

STANDARD_GEOM_UNEXPECTED_LINE = [
    "             Standard Nuclear Orientation (Angstroms)",
    "    I     Atom           X                Y                Z",
    " ----------------------------------------------------------------",
    "    1      H       1.3649900000     1.6938500000    -0.1974800000",
    "This is an unexpected line",
    "    2      O       2.3287700000     1.5629400000    -0.0416800000",
    " ----------------------------------------------------------------",
]


# === Tests for matches() ===


def test_geom_matches_input_start(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() identifies the start of the $molecule block."""
    assert parser.matches("$molecule", initial_data) is True
    assert parser.matches(" $molecule ", initial_data) is True


def test_geom_matches_standard_start(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() identifies the start of the Standard Orientation block."""
    assert parser.matches("             Standard Nuclear Orientation (Angstroms)", initial_data) is True


def test_geom_does_not_match_other_lines(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() returns False for non-matching lines."""
    assert parser.matches("$rem", initial_data) is False
    assert parser.matches("H        1.36499000      1.69385000     -0.19748000", initial_data) is False
    assert parser.matches("", initial_data) is False


def test_geom_does_not_match_if_input_parsed(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() returns False for $molecule if input geometry is already parsed."""
    initial_data.parsed_input_geometry = True
    assert parser.matches("$molecule", initial_data) is False


def test_geom_does_not_match_if_standard_parsed(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() returns False for Standard Orientation if it's already parsed."""
    initial_data.parsed_standard_geometry = True
    assert parser.matches("             Standard Nuclear Orientation (Angstroms)", initial_data) is False


def test_geom_matches_standard_if_input_parsed(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() still matches Standard Orientation even if input geom is parsed."""
    initial_data.parsed_input_geometry = True
    assert parser.matches("             Standard Nuclear Orientation (Angstroms)", initial_data) is True


def test_geom_matches_input_if_standard_parsed(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() still matches $molecule even if Standard Orientation is parsed."""
    initial_data.parsed_standard_geometry = True
    assert parser.matches("$molecule", initial_data) is True


# === Tests for parse() ===


def test_parse_input_geometry_valid(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing a valid $molecule block."""
    line_iter = iter(INPUT_GEOM_BLOCK_VALID)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_input_geometry is True
    assert results.final_geometry is None
    assert results.input_geometry is not None
    assert len(results.input_geometry) == 3
    assert results.input_geometry[0] == Atom(symbol="H", x=1.36499, y=1.69385, z=-0.19748)
    assert results.input_geometry[1] == Atom(symbol="O", x=2.32877, y=1.56294, z=-0.04168)
    assert results.input_geometry[2] == Atom(symbol="H", x=2.70244, y=1.31157, z=-0.91665)
    # Ensure iterator stopped correctly
    assert next(line_iter) == "Some other line"


def test_parse_standard_geometry_valid(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing a valid Standard Nuclear Orientation block."""
    line_iter = iter(STANDARD_GEOM_BLOCK_VALID)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_standard_geometry is True
    assert results.input_geometry is None
    assert results.final_geometry is not None
    assert len(results.final_geometry) == 3
    assert results.final_geometry[0] == Atom(symbol="H", x=1.36499, y=1.69385, z=-0.19748)
    assert results.final_geometry[1] == Atom(symbol="O", x=2.32877, y=1.56294, z=-0.04168)
    assert results.final_geometry[2] == Atom(symbol="H", x=2.70244, y=1.31157, z=-0.91665)
    # Ensure iterator stopped correctly
    assert next(line_iter) == " Nuclear Repulsion Energy ="


def test_parse_input_geom_unterminated(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Test ParsingError for unterminated $molecule block."""
    line_iter = iter(INPUT_GEOM_BLOCK_UNTERMINATED)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    with pytest.raises(ParsingError, match="Unexpected end of file in geometry block"):
        parser.parse(line_iter, start_line, results)
    assert results.parsed_input_geometry is False  # Should not be marked parsed


def test_parse_standard_geom_unterminated(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Test ParsingError for unterminated Standard Orientation block."""
    line_iter = iter(STANDARD_GEOM_BLOCK_UNTERMINATED)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    with pytest.raises(ParsingError, match="Unexpected end of file in geometry block"):
        parser.parse(line_iter, start_line, results)
    assert results.parsed_standard_geometry is False  # Should not be marked parsed


def test_parse_input_geom_no_atoms(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing $molecule block with no atom lines (should warn)."""
    line_iter = iter(INPUT_GEOM_BLOCK_NO_ATOMS)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # Should parse successfully but log a warning (verify log capture if needed)
    parser.parse(line_iter, start_line, results)

    assert results.parsed_input_geometry is True
    assert results.input_geometry == []


def test_parse_standard_geom_no_atoms(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing Standard Orientation block with no atom lines (should warn)."""
    line_iter = iter(STANDARD_GEOM_BLOCK_NO_ATOMS)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # Should parse successfully but log a warning
    parser.parse(line_iter, start_line, results)

    assert results.parsed_standard_geometry is True
    assert results.final_geometry == []


def test_parse_input_geom_unexpected_line(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing $molecule block with unexpected lines (should skip/warn)."""
    line_iter = iter(INPUT_GEOM_UNEXPECTED_LINE)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # Should parse successfully, skipping the bad line (verify log capture if needed)
    parser.parse(line_iter, start_line, results)

    assert results.parsed_input_geometry is True
    assert results.input_geometry is not None
    assert len(results.input_geometry) == 2  # Only two valid atoms parsed
    assert results.input_geometry[0].symbol == "H"
    assert results.input_geometry[1].symbol == "O"


def test_parse_standard_geom_unexpected_line(parser: GeometryParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing Standard Orientation block with unexpected lines (should skip/warn)."""
    line_iter = iter(STANDARD_GEOM_UNEXPECTED_LINE)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # Should parse successfully, skipping the bad line
    parser.parse(line_iter, start_line, results)

    assert results.parsed_standard_geometry is True
    assert results.final_geometry is not None
    assert len(results.final_geometry) == 2  # Only two valid atoms parsed
    assert results.final_geometry[0].symbol == "H"
    assert results.final_geometry[1].symbol == "O"

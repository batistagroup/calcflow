import math
from collections.abc import Iterator

import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers import orca
from calcflow.parsers.orca.geometry import GeometryParser
from calcflow.parsers.orca.typing import Atom, _MutableCalculationData


def test_input_geometry(parsed_sp_data: orca.CalculationData) -> None:
    """Test that the input geometry is parsed correctly."""
    assert parsed_sp_data.input_geometry is not None
    geometry = parsed_sp_data.input_geometry
    assert len(geometry) == 3
    # Check the first atom (H)
    assert geometry[0].symbol == "H"
    assert math.isclose(geometry[0].x, 1.364990, rel_tol=1e-6)
    assert math.isclose(geometry[0].y, 1.693850, rel_tol=1e-6)
    assert math.isclose(geometry[0].z, -0.197480, rel_tol=1e-6)
    # Check the second atom (O)
    assert geometry[1].symbol == "O"
    assert math.isclose(geometry[1].x, 2.328770, rel_tol=1e-6)
    assert math.isclose(geometry[1].y, 1.562940, rel_tol=1e-6)
    assert math.isclose(geometry[1].z, -0.041680, rel_tol=1e-6)
    # Check the third atom (H)
    assert geometry[2].symbol == "H"
    assert math.isclose(geometry[2].x, 2.702440, rel_tol=1e-6)
    assert math.isclose(geometry[2].y, 1.311570, rel_tol=1e-6)
    assert math.isclose(geometry[2].z, -0.916650, rel_tol=1e-6)


# --- Unit Tests for GeometryParser ---


@pytest.fixture
def geometry_parser() -> GeometryParser:
    """Fixture for GeometryParser instance."""
    return GeometryParser()


@pytest.fixture
def mutable_data() -> _MutableCalculationData:
    """Fixture for a clean _MutableCalculationData instance."""
    return _MutableCalculationData(raw_output="")


def test_geometry_parser_matches_start_line(
    geometry_parser: GeometryParser, mutable_data: _MutableCalculationData
) -> None:
    """Test that GeometryParser.matches identifies the start line."""
    line = "CARTESIAN COORDINATES (ANGSTROEM)"
    assert geometry_parser.matches(line, mutable_data) is True


def test_geometry_parser_matches_returns_false_if_already_parsed(
    geometry_parser: GeometryParser, mutable_data: _MutableCalculationData
) -> None:
    """Test that GeometryParser.matches returns False if geometry is already parsed."""
    line = "CARTESIAN COORDINATES (ANGSTROEM)"
    mutable_data.parsed_geometry = True
    assert geometry_parser.matches(line, mutable_data) is False


def test_geometry_parser_matches_returns_false_for_other_lines(
    geometry_parser: GeometryParser, mutable_data: _MutableCalculationData
) -> None:
    """Test that GeometryParser.matches returns False for non-matching lines."""
    lines = ["Some other line", "ATOM X Y Z", ""]
    for line in lines:
        assert geometry_parser.matches(line, mutable_data) is False


def test_geometry_parser_parses_correctly(
    geometry_parser: GeometryParser, mutable_data: _MutableCalculationData
) -> None:
    """Test successful parsing of a standard geometry block."""
    start_line = "CARTESIAN COORDINATES (ANGSTROEM)"
    lines = [
        "---------------------------------",
        " H          1.364990    1.693850   -0.197480",
        " O          2.328770    1.562940   -0.041680",
        " H          2.702440    1.311570   -0.916650",
        "",
        "Some other line after",
    ]
    iterator = iter(lines)
    geometry_parser.parse(iterator, start_line, mutable_data)

    assert mutable_data.parsed_geometry is True
    assert mutable_data.input_geometry is not None
    geometry = mutable_data.input_geometry
    assert len(geometry) == 3
    assert geometry[0] == Atom(symbol="H", x=1.364990, y=1.693850, z=-0.197480)
    assert geometry[1] == Atom(symbol="O", x=2.328770, y=1.562940, z=-0.041680)
    assert geometry[2] == Atom(symbol="H", x=2.702440, y=1.311570, z=-0.916650)
    # Ensure the iterator stopped at the right place (after the blank line)
    assert next(iterator) == "Some other line after"


def test_geometry_parser_handles_empty_block(
    geometry_parser: GeometryParser, mutable_data: _MutableCalculationData
) -> None:
    """Test parsing when the block is present but contains no atoms."""
    start_line = "CARTESIAN COORDINATES (ANGSTROEM)"
    lines: list[str] = ["", "Next section"]
    iterator = iter(lines)
    with pytest.raises(ParsingError, match="Geometry block found but no atoms could be parsed."):
        geometry_parser.parse(iterator, start_line, mutable_data)
    assert mutable_data.parsed_geometry is False
    assert mutable_data.input_geometry is None


def test_geometry_parser_handles_malformed_line(
    geometry_parser: GeometryParser, mutable_data: _MutableCalculationData, caplog: pytest.LogCaptureFixture
) -> None:
    """Test parsing stops and logs warning on a malformed geometry line."""
    start_line = "CARTESIAN COORDINATES (ANGSTROEM)"
    lines = [
        "---------------------------------",
        " H          1.364990    1.693850   -0.197480",
        " O          INVALID     LINE",
        " H          2.702440    1.311570   -0.916650",
        "",
    ]
    iterator = iter(lines)
    geometry_parser.parse(iterator, start_line, mutable_data)

    assert mutable_data.parsed_geometry is True  # Parser stops *after* finding at least one atom
    assert mutable_data.input_geometry is not None
    geometry = mutable_data.input_geometry
    assert len(geometry) == 1  # Only the first atom is parsed
    assert geometry[0] == Atom(symbol="H", x=1.364990, y=1.693850, z=-0.197480)
    assert "Unexpected line format in geometry block: O          INVALID     LINE" in caplog.text
    # Iterator should have stopped after the invalid line was encountered and processing stopped
    assert next(iterator) == " H          2.702440    1.311570   -0.916650"


def test_geometry_parser_handles_abrupt_end(
    geometry_parser: GeometryParser, mutable_data: _MutableCalculationData
) -> None:
    """Test parsing when the input ends unexpectedly within the geometry block."""
    start_line = "CARTESIAN COORDINATES (ANGSTROEM)"
    lines = [
        "---------------------------------",
        " H          1.364990    1.693850   -0.197480",
        " O          2.328770    1.562940   -0.041680",
        # End of input
    ]
    iterator: Iterator[str] = iter(lines)

    # The error occurs when the loop finishes and checks if geometry has items
    geometry_parser.parse(iterator, start_line, mutable_data)

    assert mutable_data.parsed_geometry is True
    assert mutable_data.input_geometry is not None
    geometry = mutable_data.input_geometry
    assert len(geometry) == 2  # Parses atoms until end of iterator
    assert geometry[0] == Atom(symbol="H", x=1.364990, y=1.693850, z=-0.197480)
    assert geometry[1] == Atom(symbol="O", x=2.328770, y=1.562940, z=-0.041680)


def test_geometry_parser_no_atoms_found_raises_error(
    geometry_parser: GeometryParser, mutable_data: _MutableCalculationData
) -> None:
    """Test that ParsingError is raised if the block is found but no atoms are parsed."""
    start_line = "CARTESIAN COORDINATES (ANGSTROEM)"
    lines = [
        "---------------------------------",
        # No atom lines
        "",
        "Next Section",
    ]
    iterator: Iterator[str] = iter(lines)

    with pytest.raises(ParsingError, match="Geometry block found but no atoms could be parsed."):
        geometry_parser.parse(iterator, start_line, mutable_data)

    assert mutable_data.parsed_geometry is False
    assert mutable_data.input_geometry is None

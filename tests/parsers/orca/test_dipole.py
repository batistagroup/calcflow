import builtins  # For mocking float
import math
from unittest.mock import patch

import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers import orca
from calcflow.parsers.orca.dipole import DipoleParser
from calcflow.parsers.orca.typing import DipoleMomentData, _MutableCalculationData


@pytest.fixture
def dipole_parser() -> DipoleParser:
    """Provides an instance of the DipoleParser."""
    return DipoleParser()


def test_dipole_moment(parsed_sp_data: orca.CalculationData) -> None:
    """Test the parsed dipole moment data."""
    dipole = parsed_sp_data.dipole_moment
    assert dipole is not None
    assert isinstance(dipole, orca.DipoleMomentData)

    assert math.isclose(dipole.x_au, -0.319770911, rel_tol=1e-7)
    assert math.isclose(dipole.y_au, -0.065576153, rel_tol=1e-7)
    assert math.isclose(dipole.z_au, -0.559981644, rel_tol=1e-7)
    assert math.isclose(dipole.total_au, 0.648176757, rel_tol=1e-7)
    assert math.isclose(dipole.total_debye, 1.647534386, rel_tol=1e-7)


# --- Tests for DipoleParser.matches ---


def test_dipole_parser_matches_correct_line(dipole_parser: DipoleParser, mutable_data: _MutableCalculationData) -> None:
    """Verify the parser matches the starting line when not already parsed."""
    line = "                             DIPOLE MOMENT"
    mutable_data.parsed_dipole = False
    assert dipole_parser.matches(line, mutable_data) is True


def test_dipole_parser_does_not_match_if_already_parsed(
    dipole_parser: DipoleParser, mutable_data: _MutableCalculationData
) -> None:
    """Verify the parser does not match if the dipole has already been parsed."""
    line = "                             DIPOLE MOMENT"
    mutable_data.parsed_dipole = True
    assert dipole_parser.matches(line, mutable_data) is False


def test_dipole_parser_does_not_match_other_lines(
    dipole_parser: DipoleParser, mutable_data: _MutableCalculationData
) -> None:
    """Verify the parser does not match unrelated lines."""
    line = "Some other random line in the output file"
    mutable_data.parsed_dipole = False
    assert dipole_parser.matches(line, mutable_data) is False


# --- Tests for DipoleParser.parse ---

MINIMAL_DIPOLE_BLOCK = """
                             ----------------------------
                             DIPOLE MOMENT
                             ----------------------------
    Total Dipole Moment    :  -0.3197709   -0.0655762   -0.5599816
    Magnitude (a.u.)       :   0.6481768
    Magnitude (Debye)      :   1.6475344
-----------------------------------------------------------------------------
Rotational spectrum
""".strip().splitlines()


def test_dipole_parser_successful_parse(dipole_parser: DipoleParser, mutable_data: _MutableCalculationData) -> None:
    """Test successful parsing of a minimal, valid dipole block."""
    iterator = iter(MINIMAL_DIPOLE_BLOCK)
    # Find the start line
    start_line = ""
    for line in iterator:
        if dipole_parser.matches(line, mutable_data):
            start_line = line
            break
    else:
        pytest.fail("Dipole start pattern not found in minimal block")

    dipole_parser.parse(iterator, start_line, mutable_data)

    assert mutable_data.parsed_dipole is True
    dipole = mutable_data.dipole_moment
    assert dipole is not None
    assert isinstance(dipole, DipoleMomentData)
    assert math.isclose(dipole.x_au, -0.3197709, rel_tol=1e-6)
    assert math.isclose(dipole.y_au, -0.0655762, rel_tol=1e-6)
    assert math.isclose(dipole.z_au, -0.5599816, rel_tol=1e-6)
    assert math.isclose(dipole.total_au, 0.6481768, rel_tol=1e-6)
    assert math.isclose(dipole.total_debye, 1.6475344, rel_tol=1e-6)


INVALID_COMPONENTS_BLOCK = """
                             ----------------------------
                             DIPOLE MOMENT
                             ----------------------------
    Total Dipole Moment    :     INVALID    -0.0655762   -0.5599816
    Magnitude (a.u.)       :   0.6481768
    Magnitude (Debye)      :   1.6475344
-----------------------------------------------------------------------------
Rotational spectrum
""".strip().splitlines()


def test_dipole_parser_malformed_components(dipole_parser: DipoleParser, mutable_data: _MutableCalculationData) -> None:
    """Test parsing failure when dipole components are not valid floats."""
    iterator = iter(INVALID_COMPONENTS_BLOCK)
    start_line = next(iterator)  # Consume header
    start_line = next(iterator)  # Consume title

    # Expect the specific error from the line causing the failure
    with pytest.raises(ParsingError, match=r"Dipole Moment block found but could not parse components: X, Y, Z"):
        dipole_parser.parse(iterator, start_line, mutable_data)
    assert mutable_data.parsed_dipole is True  # Marked as attempted even on error
    assert mutable_data.dipole_moment is None


INVALID_MAG_AU_BLOCK = """
                             ----------------------------
                             DIPOLE MOMENT
                             ----------------------------
    Total Dipole Moment    :  -0.3197709   -0.0655762   -0.5599816
    Magnitude (a.u.)       :   INVALID
    Magnitude (Debye)      :   1.6475344
-----------------------------------------------------------------------------
Rotational spectrum
""".strip().splitlines()


def test_dipole_parser_malformed_magnitude_au(
    dipole_parser: DipoleParser, mutable_data: _MutableCalculationData
) -> None:
    """Test parsing failure when magnitude (a.u.) is not a valid float."""
    iterator = iter(INVALID_MAG_AU_BLOCK)
    start_line = next(iterator)
    start_line = next(iterator)

    # Expect the specific error from the line causing the failure
    with pytest.raises(ParsingError, match=r"Dipole Moment block found but could not parse components: Mag\(au\)"):
        dipole_parser.parse(iterator, start_line, mutable_data)
    assert mutable_data.parsed_dipole is True
    assert mutable_data.dipole_moment is None


INVALID_MAG_DEBYE_BLOCK = """
                             ----------------------------
                             DIPOLE MOMENT
                             ----------------------------
    Total Dipole Moment    :  -0.3197709   -0.0655762   -0.5599816
    Magnitude (a.u.)       :   0.6481768
    Magnitude (Debye)      :   INVALID
-----------------------------------------------------------------------------
Rotational spectrum
""".strip().splitlines()


def test_dipole_parser_malformed_magnitude_debye(
    dipole_parser: DipoleParser, mutable_data: _MutableCalculationData
) -> None:
    """Test parsing failure when magnitude (Debye) is not a valid float."""
    iterator = iter(INVALID_MAG_DEBYE_BLOCK)
    start_line = next(iterator)
    start_line = next(iterator)

    # Expect the specific error from the line causing the failure
    with pytest.raises(ParsingError, match=r"Dipole Moment block found but could not parse components: Mag\(Debye\)"):
        dipole_parser.parse(iterator, start_line, mutable_data)
    assert mutable_data.parsed_dipole is True
    assert mutable_data.dipole_moment is None


INCOMPLETE_BLOCK_TERMINATOR = """
                             ----------------------------
                             DIPOLE MOMENT
                             ----------------------------
    Total Dipole Moment    :  -0.3197709   -0.0655762   -0.5599816
    Magnitude (a.u.)       :   0.6481768
Rotational spectrum
""".strip().splitlines()  # Missing Debye magnitude


def test_dipole_parser_premature_termination_with_terminator(
    dipole_parser: DipoleParser, mutable_data: _MutableCalculationData, caplog: pytest.LogCaptureFixture
) -> None:
    """Test parsing failure when a terminator appears before all components are found."""
    iterator = iter(INCOMPLETE_BLOCK_TERMINATOR)
    start_line = next(iterator)
    start_line = next(iterator)

    with pytest.raises(ParsingError, match="Dipole Moment block found but could not parse components: Mag\\(Debye\\)"):
        dipole_parser.parse(iterator, start_line, mutable_data)

    assert mutable_data.parsed_dipole is True
    assert mutable_data.dipole_moment is None
    # Check if the warning about premature exit was logged (due to the refined logic)
    assert any("Exiting dipole block prematurely" in record.message for record in caplog.records)


INCOMPLETE_BLOCK_EOF = """
                             ----------------------------
                             DIPOLE MOMENT
                             ----------------------------
    Total Dipole Moment    :  -0.3197709   -0.0655762   -0.5599816
    Magnitude (a.u.)       :   0.6481768
""".strip().splitlines()  # Missing Debye and terminator


def test_dipole_parser_incomplete_block_eof(dipole_parser: DipoleParser, mutable_data: _MutableCalculationData) -> None:
    """Test parsing failure when the input ends before all components are found."""
    iterator = iter(INCOMPLETE_BLOCK_EOF)
    start_line = next(iterator)
    start_line = next(iterator)

    with pytest.raises(ParsingError, match="Dipole Moment block found but could not parse components: Mag\\(Debye\\)"):
        dipole_parser.parse(iterator, start_line, mutable_data)
    assert mutable_data.parsed_dipole is True
    assert mutable_data.dipole_moment is None


# --- Test for Unexpected Exceptions ---

# Use a block that would normally parse correctly
UNEXPECTED_ERROR_BLOCK = """
                             ----------------------------
                             DIPOLE MOMENT
                             ----------------------------
    Total Dipole Moment    :  -0.3197709   -0.0655762   -0.5599816
    Magnitude (a.u.)       :   0.6481768
    Magnitude (Debye)      :   1.6475344
-----------------------------------------------------------------------------
Rotational spectrum
""".strip().splitlines()


def test_dipole_parser_unexpected_exception(
    dipole_parser: DipoleParser, mutable_data: _MutableCalculationData, caplog: pytest.LogCaptureFixture
) -> None:
    """Test handling of an unexpected generic Exception during parsing."""
    iterator = iter(UNEXPECTED_ERROR_BLOCK)
    start_line = ""
    for line in iterator:
        if dipole_parser.matches(line, mutable_data):
            start_line = line
            break
    else:
        pytest.fail("Dipole start pattern not found in test block")

    original_float = builtins.float  # Store original float implementation

    def mock_float(value: str) -> float:
        """Mock float to raise Exception on specific input."""
        if value == "1.6475344":  # Target the Debye value parsing
            raise Exception("Simulated unexpected error!")
        # We must handle potential ValueError here if the input isn't a float
        try:
            return original_float(value)
        except ValueError as e:
            raise ValueError(f"mock_float received non-float value: {value}") from e

    # Patch builtins.float within the scope of this 'with' block
    with patch("builtins.float", side_effect=mock_float):
        # Parsing should complete without raising the Exception outwards because it's caught
        dipole_parser.parse(iterator, start_line, mutable_data)

    # Verify the outcome
    assert mutable_data.parsed_dipole is True  # Marked as attempted
    assert mutable_data.dipole_moment is None  # Parsing failed internally
    assert "Unexpected error parsing dipole moment block" in caplog.text
    assert "Simulated unexpected error!" in caplog.text

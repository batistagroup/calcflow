import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.blocks.orbitals import OrbitalParser
from calcflow.parsers.qchem.typing import _MutableCalculationData


@pytest.fixture
def parser() -> OrbitalParser:
    """Fixture for the OrbitalParser."""
    return OrbitalParser()


@pytest.fixture
def initial_data() -> _MutableCalculationData:
    """Provides a fresh _MutableCalculationData instance for each test."""
    return _MutableCalculationData(raw_output="")


@pytest.fixture
def initial_unrestricted_data() -> _MutableCalculationData:
    """Provides a fresh _MutableCalculationData for unrestricted tests."""
    data = _MutableCalculationData(raw_output="")
    data.rem["unrestricted"] = "true"  # Simulate unrestricted setting
    return data


# === Test Data ===

VALID_RESTRICTED_ORBITALS = [
    "                    Orbital Energies (a.u.)",
    " --------------------------------------------------------------",
    "",
    " Alpha MOs",  # Sometimes explicit for restricted too
    " -- Occupied --",
    "-18.9359  -1.0353  -0.5339  -0.3304  -0.2436",
    " -- Virtual --",
    "  0.4341   0.5498",
    " --------------------------------------------------------------",
    "",
    "          Ground-State Mulliken Net Atomic Charges",  # Next block
]

VALID_UNRESTRICTED_ORBITALS = [
    "                    Orbital Energies (a.u.)",
    " --------------------------------------------------------------",
    "",
    " Alpha MOs",
    " -- Occupied --",
    " -10.0   -1.0",
    " -- Virtual --",
    "   1.0    2.0",
    "",
    " Beta MOs",
    " -- Occupied --",
    " -11.0   -2.0",
    " -- Virtual --",
    "   0.5    1.5",
    " --------------------------------------------------------------",
    "",
    "          Ground-State Mulliken Net Atomic Charges",
]

UNTERMINATED_ORBITALS = [
    "                    Orbital Energies (a.u.)",
    " --------------------------------------------------------------",
    " Alpha MOs",
    " -- Occupied --",
    " -10.0",
    # File ends
]

NO_ORBITALS_FOUND = [
    "                    Orbital Energies (a.u.)",
    " --------------------------------------------------------------",
    # No Alpha/Beta/Occupied/Virtual markers or energies
    " --------------------------------------------------------------",
    "          Ground-State Mulliken Net Atomic Charges",
]

MALFORMED_ENERGY_LINE = [
    "                    Orbital Energies (a.u.)",
    " --------------------------------------------------------------",
    " Alpha MOs",
    " -- Occupied --",
    " -10.0   MALFORMED   -1.0",  # Malformed part in the middle
    " -- Virtual --",
    "  1.0    2.0",
    " --------------------------------------------------------------",
    "          Ground-State Mulliken Net Atomic Charges",
]

START_LINE = "                    Orbital Energies (a.u.)"
NON_START_LINE = " Alpha MOs"

# === Tests for matches() ===


def test_orbitals_matches_start_line(parser: OrbitalParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() identifies the orbitals start line."""
    assert parser.matches(START_LINE, initial_data) is True


def test_orbitals_does_not_match_other_lines(parser: OrbitalParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() returns False for non-matching lines."""
    assert parser.matches(NON_START_LINE, initial_data) is False
    assert parser.matches("", initial_data) is False
    assert parser.matches("$rem", initial_data) is False


def test_orbitals_does_not_match_if_already_parsed(
    parser: OrbitalParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() returns False if orbitals are already parsed."""
    initial_data.parsed_orbitals = True
    assert parser.matches(START_LINE, initial_data) is False


# === Tests for parse() ===


def test_parse_valid_restricted_orbitals(parser: OrbitalParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing a valid restricted orbital block."""
    line_iter = iter(VALID_RESTRICTED_ORBITALS)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_orbitals is True
    assert results.orbitals is not None
    assert results.orbitals.alpha_orbitals is not None
    assert results.orbitals.beta_orbitals is None  # Restricted
    assert len(results.orbitals.alpha_orbitals) == 7
    expected_energies = [-18.9359, -1.0353, -0.5339, -0.3304, -0.2436, 0.4341, 0.5498]
    for i, orb in enumerate(results.orbitals.alpha_orbitals):
        assert orb.index == i
        assert orb.energy_eh == pytest.approx(expected_energies[i])
    assert not results.parsing_errors
    assert not results.parsing_warnings
    # Ensure iterator stopped correctly after the closing dashes.
    # The next line in the test data is an empty string.
    assert next(line_iter) == ""


def test_parse_valid_unrestricted_orbitals(
    parser: OrbitalParser, initial_unrestricted_data: _MutableCalculationData
) -> None:
    """Test parsing a valid unrestricted orbital block."""
    line_iter = iter(VALID_UNRESTRICTED_ORBITALS)
    # Use the unrestricted data fixture for matching and parsing
    start_line = next(line for line in line_iter if parser.matches(line, initial_unrestricted_data))
    results = initial_unrestricted_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_orbitals is True
    assert results.orbitals is not None
    assert results.orbitals.alpha_orbitals is not None
    assert results.orbitals.beta_orbitals is not None  # Unrestricted

    assert len(results.orbitals.alpha_orbitals) == 4
    expected_alpha_energies = [-10.0, -1.0, 1.0, 2.0]
    for i, orb in enumerate(results.orbitals.alpha_orbitals):
        assert orb.index == i
        assert orb.energy_eh == pytest.approx(expected_alpha_energies[i])

    assert len(results.orbitals.beta_orbitals) == 4
    expected_beta_energies = [-11.0, -2.0, 0.5, 1.5]
    for i, orb in enumerate(results.orbitals.beta_orbitals):
        assert orb.index == i
        assert orb.energy_eh == pytest.approx(expected_beta_energies[i])

    assert not results.parsing_errors
    assert not results.parsing_warnings
    # Ensure iterator stopped correctly after the closing dashes.
    # The next line in the test data is an empty string.
    assert next(line_iter) == ""


def test_parse_unterminated_orbitals(parser: OrbitalParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing an unterminated orbital block (should warn, not error)."""
    line_iter = iter(UNTERMINATED_ORBITALS)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # Should parse what it found and issue a warning
    parser.parse(line_iter, start_line, results)

    assert results.parsed_orbitals is True  # Still marked as parsed
    assert results.orbitals is not None
    assert results.orbitals.alpha_orbitals is not None
    assert len(results.orbitals.alpha_orbitals) == 1  # Only one energy parsed
    assert results.orbitals.alpha_orbitals[0].energy_eh == pytest.approx(-10.0)
    assert results.orbitals.beta_orbitals is None
    assert not results.parsing_errors
    assert len(results.parsing_warnings) == 1
    assert "File ended during Orbital Energies parsing" in results.parsing_warnings[0]


def test_parse_no_orbitals_found(parser: OrbitalParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing a block with header but no actual orbital data (should warn)."""
    line_iter = iter(NO_ORBITALS_FOUND)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # Should parse successfully but add a warning
    parser.parse(line_iter, start_line, results)

    assert results.parsed_orbitals is False  # Not marked parsed as no orbitals were found
    assert results.orbitals is None
    assert not results.parsing_errors
    assert len(results.parsing_warnings) == 1
    assert "No orbital energies were parsed" in results.parsing_warnings[0]


def test_parse_malformed_energy_line(parser: OrbitalParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing with a malformed energy line (should raise ParsingError)."""
    line_iter = iter(MALFORMED_ENERGY_LINE)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # Expect a ParsingError due to the malformed line
    with pytest.raises(ParsingError, match="Malformed orbital energy line: '-10.0   MALFORMED   -1.0'"):
        parser.parse(line_iter, start_line, results)

    # Ensure parsing stopped and didn't partially succeed incorrectly
    assert results.parsed_orbitals is False  # Should not be marked as parsed
    assert results.orbitals is None
    assert len(results.parsing_errors) == 1  # Should have recorded the error
    assert "Malformed energy line: '-10.0   MALFORMED   -1.0'" in results.parsing_errors[0]
    # Warnings might or might not exist depending on exact path, so don't assert count

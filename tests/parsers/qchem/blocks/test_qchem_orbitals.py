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
        assert orb.energy == pytest.approx(expected_energies[i])
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
        assert orb.energy == pytest.approx(expected_alpha_energies[i])

    assert len(results.orbitals.beta_orbitals) == 4
    expected_beta_energies = [-11.0, -2.0, 0.5, 1.5]
    for i, orb in enumerate(results.orbitals.beta_orbitals):
        assert orb.index == i
        assert orb.energy == pytest.approx(expected_beta_energies[i])

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
    assert results.orbitals.alpha_orbitals[0].energy == pytest.approx(-10.0)
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


# Assuming parsed_tddft_uks_pc2_data is a fixture defined in conftest.py
# or can be imported if it's from another test module context.
# For this example, we'll write the test as if the fixture is available.
# If it needs to be defined here or imported, that would be an additional step.


def test_parse_beta_orbitals_from_uks_tddft_output(parsed_tddft_uks_pc2_data: _MutableCalculationData) -> None:
    """Test parsing of beta orbitals from a real UKS TDDFT output.

    Uses the parsed_tddft_uks_pc2_data fixture which should correspond to
    data/calculations/examples/qchem/h2o/tddft-uks-pc2.out
    """
    results = parsed_tddft_uks_pc2_data  # Fixture directly provides _MutableCalculationData or CalculationData

    assert results.orbitals is not None, "OrbitalsSet should be parsed"
    assert results.orbitals.alpha_orbitals is not None, "Alpha orbitals should be present"
    assert results.orbitals.beta_orbitals is not None, "Beta orbitals should be present for UKS"

    # Based on tddft-uks-pc2.out for H2O (5 occupied, 53 virtual for each spin)
    # Alpha orbitals check (briefly, focus is beta here)
    # Count based on common H2O calculation: 5 occupied (1s O, 2 sp-like O, 2 O-H sigma) + virtuals
    # From the file: 5 occupied alpha, 53 virtual alpha.
    assert len(results.orbitals.alpha_orbitals) == 58, "Expected 58 alpha orbitals for H2O UKS example"

    # Beta Orbitals Checks
    beta_orbs = results.orbitals.beta_orbitals
    assert len(beta_orbs) == 58, "Expected 58 beta orbitals for H2O UKS example (5 occ + 53 virt)"

    # Occupied Beta MOs energies from the provided snippet:
    # -19.2346  -1.1182  -0.6261  -0.4888  -0.4147
    expected_occ_beta_energies = [-19.2346, -1.1182, -0.6261, -0.4888, -0.4147]
    num_occ_beta = len(expected_occ_beta_energies)

    for i in range(num_occ_beta):
        assert beta_orbs[i].index == i, f"Index mismatch for occupied beta orbital {i}"
        assert beta_orbs[i].energy == pytest.approx(expected_occ_beta_energies[i]), (
            f"Energy mismatch for occupied beta orbital {i}"
        )

    # Virtual Beta MOs energies from the provided snippet:
    # Line 1: 0.0878   0.1389   0.3614   0.3715   0.4265   0.4394   0.5388   0.7688
    # Line 2: 0.8532   0.8877   0.9614   1.1406   1.2650   1.3179   1.5082   1.5393
    # ... (many lines) ...
    # Last line: 6.4981   6.8786   6.8988   7.4257  14.8185
    expected_virt_beta_energies_sample = {
        0: 0.0878,  # First virtual (index 5 overall, index 0 of virtuals)
        1: 0.1389,
        7: 0.7688,  # Last on first line of virtuals
        8: 0.8532,  # First on second line of virtuals
        15: 1.5393,  # Last on second line of virtuals
        # Let's pick some from the middle based on manual counting if possible, or stick to ends
        # Total virtual = 53. Last index of virtuals is 52.
        # (53 - 5) = 48. The 5th from last virtual overall index will be 53-5=48. virt_idx = 48-5 = 43
        # (53 - 1) = 52. The last virtual overall index will be 53-1=52. virt_idx = 52-5 = 47
        (53 - 5): 6.4981,  # 5th from last virtual orbital
        (53 - 4): 6.8786,
        (53 - 3): 6.8988,
        (53 - 2): 7.4257,
        (53 - 1): 14.8185,  # Last virtual orbital
    }

    for virt_idx_offset, expected_energy in expected_virt_beta_energies_sample.items():
        overall_idx = num_occ_beta + virt_idx_offset
        assert beta_orbs[overall_idx].index == overall_idx, (
            f"Index mismatch for virtual beta orbital at overall index {overall_idx}"
        )
        assert beta_orbs[overall_idx].energy == pytest.approx(expected_energy), (
            f"Energy mismatch for virtual beta orbital at overall_idx {overall_idx} (virtual offset {virt_idx_offset})"
        )

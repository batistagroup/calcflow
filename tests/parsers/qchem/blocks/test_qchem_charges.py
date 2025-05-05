import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.blocks.charges import MullikenChargesParser
from calcflow.parsers.qchem.typing import _MutableCalculationData


@pytest.fixture
def parser() -> MullikenChargesParser:
    """Fixture for the MullikenChargesParser."""
    return MullikenChargesParser()


@pytest.fixture
def initial_data() -> _MutableCalculationData:
    """Provides a fresh _MutableCalculationData instance for each test."""
    return _MutableCalculationData(raw_output="")


# === Test Data ===

VALID_MULLIKEN_BLOCK = [
    "          Ground-State Mulliken Net Atomic Charges",
    "",
    "     Atom                 Charge (a.u.)",
    "  ----------------------------------------",
    "      1 H                     0.173083",
    "      2 O                    -0.346478",
    "      3 H                     0.173395",
    "  ----------------------------------------",
    "  Sum of atomic charges =     0.000000",
    "",
    " -----------------------------------------------------------------",
]

UNTERMINATED_MULLIKEN_BLOCK = [
    "          Ground-State Mulliken Net Atomic Charges",
    "",
    "     Atom                 Charge (a.u.)",
    "  ----------------------------------------",
    "      1 H                     0.173083",
    # File ends abruptly
]

EMPTY_MULLIKEN_BLOCK = [
    "          Ground-State Mulliken Net Atomic Charges",
    "",
    "     Atom                 Charge (a.u.)",
    "  ----------------------------------------",
    # No charge lines
    "  ----------------------------------------",
    "  Sum of atomic charges =     0.000000",
]

MALFORMED_LINE_MULLIKEN_BLOCK = [
    "          Ground-State Mulliken Net Atomic Charges",
    "",
    "     Atom                 Charge (a.u.)",
    "  ----------------------------------------",
    "      1 H                     0.173083",
    "      MALFORMED LINE",
    "      3 H                     0.173395",
    "  ----------------------------------------",
    "  Sum of atomic charges =     0.000000",
]

START_LINE = "          Ground-State Mulliken Net Atomic Charges"
NON_START_LINE = "      1 H                     0.173083"

# === Tests for matches() ===


def test_charges_matches_start_line(parser: MullikenChargesParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() identifies the Mulliken charges start line."""
    assert parser.matches(START_LINE, initial_data) is True


def test_charges_does_not_match_other_lines(
    parser: MullikenChargesParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() returns False for non-matching lines."""
    assert parser.matches(NON_START_LINE, initial_data) is False
    assert parser.matches("", initial_data) is False
    assert parser.matches("$rem", initial_data) is False


def test_charges_does_not_match_if_already_parsed(
    parser: MullikenChargesParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() returns False if Mulliken charges are already parsed."""
    initial_data.parsed_mulliken_charges = True
    assert parser.matches(START_LINE, initial_data) is False


# === Tests for parse() ===


def test_parse_valid_mulliken_block(parser: MullikenChargesParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing a valid Mulliken charges block."""
    line_iter = iter(VALID_MULLIKEN_BLOCK)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_mulliken_charges is True
    assert len(results.atomic_charges) == 1
    mulliken_data = results.atomic_charges[0]
    assert mulliken_data.method == "Mulliken"
    assert len(mulliken_data.charges) == 3
    assert mulliken_data.charges[0] == pytest.approx(0.173083)  # 0-based index
    assert mulliken_data.charges[1] == pytest.approx(-0.346478)
    assert mulliken_data.charges[2] == pytest.approx(0.173395)
    assert not results.parsing_errors
    assert not results.parsing_warnings
    # Ensure iterator stopped correctly after consuming the end marker (dashed line)
    # The next line should be the 'Sum of atomic charges' line.
    assert next(line_iter) == "  Sum of atomic charges =     0.000000"


def test_parse_unterminated_mulliken_block(
    parser: MullikenChargesParser, initial_data: _MutableCalculationData
) -> None:
    """Test ParsingError for an unterminated Mulliken block."""
    line_iter = iter(UNTERMINATED_MULLIKEN_BLOCK)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    with pytest.raises(ParsingError, match="Incomplete Mulliken charges section found."):
        parser.parse(line_iter, start_line, results)

    # Check that partial results were stored before raising
    assert results.parsed_mulliken_charges is True  # Marked as parsed even if incomplete
    assert len(results.atomic_charges) == 1
    mulliken_data = results.atomic_charges[0]
    assert mulliken_data.method == "Mulliken"
    assert len(mulliken_data.charges) == 1  # Only the first charge was parsed
    assert mulliken_data.charges[0] == pytest.approx(0.173083)
    assert len(results.parsing_errors) == 1
    assert "Incomplete Mulliken charges section found." in results.parsing_errors[0]


def test_parse_empty_mulliken_block(parser: MullikenChargesParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing a Mulliken block with no charge lines (should warn)."""
    line_iter = iter(EMPTY_MULLIKEN_BLOCK)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # Should parse successfully but add a warning
    parser.parse(line_iter, start_line, results)

    assert results.parsed_mulliken_charges is True
    assert len(results.atomic_charges) == 0  # No charges stored
    assert not results.parsing_errors
    assert len(results.parsing_warnings) == 1
    assert "Mulliken charges block parsed, but no charges found" in results.parsing_warnings[0]


def test_parse_malformed_line_mulliken_block(
    parser: MullikenChargesParser, initial_data: _MutableCalculationData
) -> None:
    """Test parsing a Mulliken block with a malformed line (should skip/warn)."""
    line_iter = iter(MALFORMED_LINE_MULLIKEN_BLOCK)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # Should parse successfully, skipping the bad line and adding a warning
    parser.parse(line_iter, start_line, results)

    assert results.parsed_mulliken_charges is True
    assert len(results.atomic_charges) == 1
    mulliken_data = results.atomic_charges[0]
    assert mulliken_data.method == "Mulliken"
    assert len(mulliken_data.charges) == 2  # Only atoms 1 (index 0) and 3 (index 2)
    assert 0 in mulliken_data.charges
    assert 1 not in mulliken_data.charges  # Malformed line skipped
    assert 2 in mulliken_data.charges
    assert mulliken_data.charges[0] == pytest.approx(0.173083)
    assert mulliken_data.charges[2] == pytest.approx(0.173395)
    assert not results.parsing_errors
    assert len(results.parsing_warnings) == 1
    assert "Unexpected line in Mulliken charges block" in results.parsing_warnings[0]

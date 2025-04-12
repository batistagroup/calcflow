import logging
import math
import re
import unittest.mock
from pathlib import Path
from typing import Literal

import pytest
from pytest import LogCaptureFixture

from calcflow.exceptions import ParsingError
from calcflow.parsers import orca
from calcflow.parsers.orca.charges import (
    CHARGE_LINE_PAT,
    LOEWDIN_CHARGES_START_PAT,
    MULLIKEN_CHARGES_START_PAT,
    ChargesParser,
)
from calcflow.parsers.orca.typing import LineIterator, _MutableCalculationData

# Load the example output file content once
EXAMPLE_SP_OUT_PATH = Path(__file__).resolve().parents[3] / "data" / "calculations" / "examples" / "sp.out"
EXAMPLE_SP_OUT = EXAMPLE_SP_OUT_PATH.read_text()


@pytest.fixture(scope="module")
def parsed_sp_data() -> orca.CalculationData:
    """Fixture to parse the standard single point output file."""
    return orca.parse_orca_output(EXAMPLE_SP_OUT)


def test_atomic_charges(parsed_sp_data: orca.CalculationData) -> None:
    """Test the parsed atomic charges (Mulliken and Loewdin)."""
    charges_list = parsed_sp_data.atomic_charges
    assert charges_list is not None
    assert len(charges_list) == 2  # Expecting Mulliken and Loewdin

    methods_found = {charge.method for charge in charges_list}
    assert methods_found == {"Mulliken", "Loewdin"}

    for charges in charges_list:
        assert len(charges.charges) == 3  # 3 atoms
        if charges.method == "Mulliken":
            assert math.isclose(charges.charges[0], 0.172827, rel_tol=1e-6)
            assert math.isclose(charges.charges[1], -0.346096, rel_tol=1e-6)
            assert math.isclose(charges.charges[2], 0.173269, rel_tol=1e-6)
        elif charges.method == "Loewdin":
            assert math.isclose(charges.charges[0], 0.120674, rel_tol=1e-6)
            assert math.isclose(charges.charges[1], -0.241589, rel_tol=1e-6)
            assert math.isclose(charges.charges[2], 0.120916, rel_tol=1e-6)


def test_atomic_charges_integration(parsed_sp_data: orca.CalculationData) -> None:
    """Test the parsed atomic charges (Mulliken and Loewdin) via full file parsing."""
    charges_list = parsed_sp_data.atomic_charges
    assert charges_list is not None
    assert len(charges_list) == 2  # Expecting Mulliken and Loewdin

    methods_found = {charge.method for charge in charges_list}
    assert methods_found == {"Mulliken", "Loewdin"}

    for charges in charges_list:
        assert len(charges.charges) == 3  # 3 atoms
        if charges.method == "Mulliken":
            assert math.isclose(charges.charges[0], 0.172827, rel_tol=1e-6)
            assert math.isclose(charges.charges[1], -0.346096, rel_tol=1e-6)
            assert math.isclose(charges.charges[2], 0.173269, rel_tol=1e-6)
        elif charges.method == "Loewdin":
            assert math.isclose(charges.charges[0], 0.120674, rel_tol=1e-6)
            assert math.isclose(charges.charges[1], -0.241589, rel_tol=1e-6)
            assert math.isclose(charges.charges[2], 0.120916, rel_tol=1e-6)


# --- Unit Tests for ChargesParser ---


@pytest.mark.parametrize(
    "method, pattern, line, expected",
    [
        ("Mulliken", MULLIKEN_CHARGES_START_PAT, "MULLIKEN ATOMIC CHARGES", True),
        ("Loewdin", LOEWDIN_CHARGES_START_PAT, "LOEWDIN ATOMIC CHARGES", True),
        ("Mulliken", MULLIKEN_CHARGES_START_PAT, "Some other line", False),
        ("Loewdin", LOEWDIN_CHARGES_START_PAT, "  MULLIKEN ATOMIC CHARGES (something else)", False),
    ],
)
def test_charges_parser_matches(
    method: Literal["Mulliken", "Loewdin"], pattern: re.Pattern[str], line: str, expected: bool
) -> None:
    """Verify the matches method correctly identifies start lines."""
    parser = ChargesParser(method=method, start_pattern=pattern)
    results = _MutableCalculationData(raw_output="")  # Minimal data needed
    assert parser.matches(line, results) is expected


@pytest.mark.parametrize(
    "method, pattern",
    [
        ("Mulliken", MULLIKEN_CHARGES_START_PAT),
        ("Loewdin", LOEWDIN_CHARGES_START_PAT),
    ],
)
def test_charges_parser_parse_success(method: Literal["Mulliken", "Loewdin"], pattern: re.Pattern[str]) -> None:
    """Test successful parsing of a standard charge block."""
    lines = [
        "---------------------",
        "   0 O  :    -0.516815",
        "   1 H  :     0.258685",
        "   2 H  :     0.258130",
        "",
        "SUM OF CHARGES",
    ]
    iterator: LineIterator = iter(lines)
    results = _MutableCalculationData(raw_output="")
    parser = ChargesParser(method=method, start_pattern=pattern)
    start_line = f"{method.upper()} ATOMIC CHARGES"  # Dummy start line, not used by parse

    parser.parse(iterator, start_line, results)

    assert len(results.atomic_charges) == 1
    charge_data = results.atomic_charges[0]
    assert charge_data.method == method
    assert len(charge_data.charges) == 3
    assert math.isclose(charge_data.charges[0], -0.516815)
    assert math.isclose(charge_data.charges[1], 0.258685)
    assert math.isclose(charge_data.charges[2], 0.258130)


def test_charges_parser_parse_empty_block() -> None:
    """Test parsing when the block contains no valid charge lines."""
    lines = [
        "---------------------",
        # No valid lines
        "",
        "SUM OF CHARGES",
    ]
    iterator: LineIterator = iter(lines)
    results = _MutableCalculationData(raw_output="")
    parser = ChargesParser(method="Mulliken", start_pattern=MULLIKEN_CHARGES_START_PAT)
    start_line = "MULLIKEN ATOMIC CHARGES"

    with pytest.raises(ParsingError, match="Mulliken charges block found but no charges could be parsed."):
        parser.parse(iterator, start_line, results)
    assert not results.atomic_charges


def test_charges_parser_parse_malformed_line() -> None:
    """Test parsing stops gracefully if a malformed line is encountered after valid ones."""
    lines = [
        "---------------------",
        "   0 O  :    -0.516815",
        "   1 H  :     invalid_charge",  # Malformed line
        "   2 H  :     0.258130",  # Should not be parsed
        "",
        "SUM OF CHARGES",
    ]
    iterator: LineIterator = iter(lines)
    results = _MutableCalculationData(raw_output="")
    parser = ChargesParser(method="Loewdin", start_pattern=LOEWDIN_CHARGES_START_PAT)
    start_line = "LOEWDIN ATOMIC CHARGES"

    # The current implementation raises an error on malformed lines
    with pytest.raises(ParsingError, match="Could not parse Loewdin charge line values"):
        parser.parse(iterator, start_line, results)


def test_charges_parser_parse_unexpected_end() -> None:
    """Test parsing when the input ends abruptly within the block."""
    lines = [
        "---------------------",
        "   0 O  :    -0.516815",
        # Input ends here
    ]
    iterator: LineIterator = iter(lines)
    results = _MutableCalculationData(raw_output="")
    parser = ChargesParser(method="Mulliken", start_pattern=MULLIKEN_CHARGES_START_PAT)
    start_line = "MULLIKEN ATOMIC CHARGES"

    # The loop finishes, and the parsed data should be stored
    parser.parse(iterator, start_line, results)

    assert len(results.atomic_charges) == 1
    charge_data = results.atomic_charges[0]
    assert charge_data.method == "Mulliken"
    assert len(charge_data.charges) == 1
    assert math.isclose(charge_data.charges[0], -0.516815)


def test_charges_parser_handles_unexpected_exception(caplog: LogCaptureFixture) -> None:
    """Test that a generic Exception during line processing is caught and logged."""
    lines = [
        "---------------------",
        "   0 O  :    -0.516815",
        "   1 H  :     0.258685",  # This line will cause the mocked error
        "   2 H  :     0.258130",  # Should not be reached
        "",
    ]
    iterator: LineIterator = iter(lines)
    results = _MutableCalculationData(raw_output="")
    parser = ChargesParser(method="Mulliken", start_pattern=MULLIKEN_CHARGES_START_PAT)
    start_line = "MULLIKEN ATOMIC CHARGES"

    # Simulate an unexpected error when trying to process the second line's match
    mock_match_that_raises = unittest.mock.Mock()
    mock_match_that_raises.groups.side_effect = Exception("Simulated unexpected error")

    # We need a real match object for the first successful call
    real_match_line_1 = re.match(CHARGE_LINE_PAT, lines[1])

    # Patch the entire pattern object within the charges module
    with unittest.mock.patch("calcflow.parsers.orca.charges.CHARGE_LINE_PAT") as mock_pattern:
        # Configure the .match() method of our mocked pattern object
        mock_pattern.match.side_effect = [
            real_match_line_1,  # First call works normally
            mock_match_that_raises,  # Second call returns the mock that will raise
            None,  # Subsequent calls return None (or could be real matches if needed)
        ]

        with caplog.at_level(logging.ERROR):
            # The parse method should complete without raising the exception
            parser.parse(iterator, start_line, results)

    # Verify the exception was logged
    assert "Unexpected error parsing Mulliken charges block" in caplog.text
    assert "Simulated unexpected error" in caplog.text

    # Verify that NO charges were added because the block parsing failed mid-way
    # The parser logs the unexpected error and discards partial data from the failed block.
    assert not results.atomic_charges  # Should be empty

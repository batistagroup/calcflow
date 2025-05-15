import math
from collections.abc import Iterator

import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers import orca
from calcflow.parsers.orca.blocks.dispersion import DispersionParser
from calcflow.parsers.orca.typing import _MutableCalculationData

# --- Test Data --- #

# Minimal valid block
VALID_BLOCK_D3 = [
    "------------------------------------",
    "   DFT DISPERSION CORRECTION V3.0   ",
    "------------------------------------",
    "DFTD3 V3.0 Correction",
    "Parameter scaling factor =  1.00000000",
    "Dispersion correction      -0.01234567 Eh",
    "---------------------",
]

# Block with D4 method
VALID_BLOCK_D4 = [
    "------------------------------------",
    "   DFT DISPERSION CORRECTION        ",
    "------------------------------------",
    "DFTD4 V1.0 Rev 2 Correction",
    "Parameters used: DFTD4 default",
    "Dispersion correction       0.00987654 Eh",
    "---------------------",
]

# Block missing method
MISSING_METHOD_BLOCK = [
    "------------------------------------",
    "   DFT DISPERSION CORRECTION V3.0   ",
    "------------------------------------",
    "Parameter scaling factor =  1.00000000",
    "Dispersion correction      -0.01234567 Eh",
    "---------------------",
]

# Block missing energy
MISSING_ENERGY_BLOCK = [
    "------------------------------------",
    "   DFT DISPERSION CORRECTION V3.0   ",
    "------------------------------------",
    "DFTD3 V3.0 Correction",
    "Parameter scaling factor =  1.00000000",
    "---------------------",
]

# Block with invalid energy format
INVALID_ENERGY_BLOCK = [
    "------------------------------------",
    "   DFT DISPERSION CORRECTION V3.0   ",
    "------------------------------------",
    "DFTD3 V3.0 Correction",
    "Parameter scaling factor =  1.00000000",
    "Dispersion correction       invalid Eh",
    "---------------------",
]

# Block terminated prematurely
TERMINATED_BLOCK_TIMINGS = [
    "------------------------------------",
    "   DFT DISPERSION CORRECTION V3.0   ",
    "------------------------------------",
    "DFTD3 V3.0 Correction",
    "TIMINGS",
]

TERMINATED_BLOCK_FINAL_ENERGY = [
    "------------------------------------",
    "   DFT DISPERSION CORRECTION V3.0   ",
    "------------------------------------",
    "DFTD3 V3.0 Correction",
    "FINAL SINGLE POINT ENERGY",
]

TERMINATED_BLOCK_HYPHENS = [
    "------------------------------------",
    "   DFT DISPERSION CORRECTION V3.0   ",
    "------------------------------------",
    "DFTD3 V3.0 Correction",
    "-" * 60,
]


# --- Fixtures ---


@pytest.fixture
def parser() -> DispersionParser:
    """Returns an instance of the DispersionParser."""
    return DispersionParser()


@pytest.fixture
def initial_data() -> _MutableCalculationData:
    """Returns a default mutable calculation data object."""
    return _MutableCalculationData(raw_output="")


def _list_to_iterator(lines: list[str]) -> Iterator[str]:
    """Helper to convert list of lines to an iterator."""
    return iter(lines)


def test_integration_dispersion_correction(parsed_sp_data: orca.CalculationData) -> None:
    """Test the parsed dispersion correction data from a real file."""
    disp = parsed_sp_data.dispersion_correction
    assert disp is not None
    assert isinstance(disp, orca.DispersionCorrectionData)

    # Example file uses DFTD3 with very small energy
    assert "DFTD3" in disp.method
    assert math.isclose(disp.energy, -0.000001638, abs_tol=1e-9)


# --- Unit Tests for DispersionParser ---


# Test `matches` method
def test_matches_positive(parser: DispersionParser, initial_data: _MutableCalculationData) -> None:
    """Test `matches` returns True for the correct starting line when not already parsed."""
    line = "Begin DFT DISPERSION CORRECTION calculation"
    initial_data.parsed_dispersion = False
    assert parser.matches(line, initial_data) is True


def test_matches_negative_already_parsed(parser: DispersionParser, initial_data: _MutableCalculationData) -> None:
    """Test `matches` returns False if dispersion has already been parsed."""
    line = "Begin DFT DISPERSION CORRECTION calculation"
    initial_data.parsed_dispersion = True
    assert parser.matches(line, initial_data) is False


def test_matches_negative_incorrect_line(parser: DispersionParser, initial_data: _MutableCalculationData) -> None:
    """Test `matches` returns False for lines not matching the start pattern."""
    line = "Some other calculation line"
    initial_data.parsed_dispersion = False
    assert parser.matches(line, initial_data) is False


# Test `parse` method - Success cases
@pytest.mark.parametrize(
    "block_lines, expected_method_part, expected_energy",
    [
        (VALID_BLOCK_D3, "DFTD3 V3.0 Correction", -0.01234567),
        (VALID_BLOCK_D4, "DFTD4 V1.0 Rev 2 Correction", 0.00987654),
    ],
)
def test_parse_success(
    parser: DispersionParser,
    initial_data: _MutableCalculationData,
    block_lines: list[str],
    expected_method_part: str,
    expected_energy: float,
) -> None:
    """Test successful parsing of valid dispersion blocks."""
    iterator = _list_to_iterator(block_lines)
    start_line = next(iterator)  # Consume the start line ("---") as parse expects it to be handled already by main loop
    parser.parse(iterator, start_line, initial_data)

    assert initial_data.parsed_dispersion is True
    assert initial_data.dispersion_correction is not None
    assert initial_data.dispersion_correction.method == expected_method_part
    assert math.isclose(initial_data.dispersion_correction.energy, expected_energy, abs_tol=1e-9)


# Test `parse` method - Failure cases
@pytest.mark.parametrize(
    "block_lines, error_message_part",
    [
        (MISSING_METHOD_BLOCK, "Method"),
        (MISSING_ENERGY_BLOCK, "Energy"),
        (INVALID_ENERGY_BLOCK, "Could not parse dispersion energy"),
        (TERMINATED_BLOCK_TIMINGS, "Energy"),  # Expects error because energy wasn't found before terminator
        (TERMINATED_BLOCK_FINAL_ENERGY, "Energy"),
        (TERMINATED_BLOCK_HYPHENS, "Energy"),
        ([], "Method, Energy"),  # Empty iterator after header
    ],
)
def test_parse_failure(
    parser: DispersionParser,
    initial_data: _MutableCalculationData,
    block_lines: list[str],
    error_message_part: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test parsing failures due to missing data, invalid formats, or premature termination."""
    iterator = _list_to_iterator(block_lines)
    start_line = "DFT DISPERSION CORRECTION"  # Dummy start line for context

    with pytest.raises(ParsingError) as excinfo:
        parser.parse(iterator, start_line, initial_data)

    assert error_message_part in str(excinfo.value)
    assert initial_data.parsed_dispersion is True  # Should be marked as attempted even on error
    assert initial_data.dispersion_correction is None

    # Check for specific warning on premature termination before energy found
    if "Energy" in error_message_part and block_lines:  # Check if block wasn't empty
        terminator_line = block_lines[-1].strip()
        if terminator_line in ("TIMINGS", "FINAL SINGLE POINT ENERGY", "-" * 60):
            assert f"Exiting dispersion block prematurely due to terminator: '{terminator_line}'" in caplog.text

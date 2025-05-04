from typing import Any

import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.blocks.gradient import GradientParser
from calcflow.parsers.orca.typing import OptimizationCycleData, _MutableOptData


# Helper function to create a mock iterator
def _create_iterator(lines: list[str]) -> Any:
    # In Python 3.9+, TextIO can be used, but for broader compatibility:
    # We return 'Any' as the exact iterator type isn't critical for these tests.
    return iter(lines)


# Mock data structures needed for the parser
@pytest.fixture
def mock_results() -> _MutableOptData:
    """Provides a mock _MutableOptData object."""
    # Explicitly type the list and provide required raw_output
    return _MutableOptData(raw_output="", cycles=list[OptimizationCycleData]())


@pytest.fixture
def mock_cycle_data() -> OptimizationCycleData:
    """Provides a mock OptimizationCycleData object."""
    return OptimizationCycleData(cycle_number=1)


# Instantiate the parser
@pytest.fixture
def parser() -> GradientParser:
    """Provides an instance of GradientParser."""
    return GradientParser()


# Test Cases for Exceptions
def test_parse_raises_parsing_error_on_eof_after_header(
    parser: GradientParser, mock_results: _MutableOptData, mock_cycle_data: OptimizationCycleData
) -> None:
    """Test ParsingError when EOF occurs immediately after the gradient header."""
    lines = [
        "------------------",
        "CARTESIAN GRADIENT",
        "------------------",
        # Missing blank line and subsequent content
    ]
    iterator = _create_iterator(lines)
    current_line = next(iterator)  # Consume the first separator line
    current_line = next(iterator)  # Consume the header line

    with pytest.raises(ParsingError, match="File ended unexpectedly immediately after gradient header."):
        parser.parse(iterator, current_line, mock_results, mock_cycle_data)


def test_parse_raises_parsing_error_when_no_gradient_lines_present(
    parser: GradientParser, mock_results: _MutableOptData, mock_cycle_data: OptimizationCycleData
) -> None:
    """Test ParsingError when the block contains no gradient lines."""
    lines = [
        "------------------",
        "CARTESIAN GRADIENT",
        "------------------",
        "",
        # Missing gradient lines
        "Norm of the Cartesian gradient     ...    0.0827686008",
        "RMS gradient                       ...    0.0275895336",
        "MAX gradient                       ...    0.0570268254",
    ]
    iterator = _create_iterator(lines)
    current_line = next(iterator)
    current_line = next(iterator)

    with pytest.raises(ParsingError, match="Gradient block found but no gradient lines parsed."):
        parser.parse(iterator, current_line, mock_results, mock_cycle_data)

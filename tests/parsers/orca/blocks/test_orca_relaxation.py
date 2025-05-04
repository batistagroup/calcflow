import copy
from typing import Any

import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.blocks.relaxation import RelaxationStepParser
from calcflow.parsers.orca.typing import (
    OptimizationCycleData,
    RelaxationStepData,
    _MutableOptData,
)


# Helper function to create a mock iterator
def _create_iterator(lines: list[str]) -> Any:
    return iter(lines)


# Mock data structures needed for the parser
@pytest.fixture
def mock_results() -> _MutableOptData:
    """Provides a mock _MutableOptData object."""
    return _MutableOptData(raw_output="", cycles=list[OptimizationCycleData]())


@pytest.fixture
def mock_cycle_data() -> OptimizationCycleData:
    """Provides a mock OptimizationCycleData object."""
    return OptimizationCycleData(cycle_number=5)  # Example cycle number


# Instantiate the parser
@pytest.fixture
def parser() -> RelaxationStepParser:
    """Provides an instance of RelaxationStepParser."""
    return RelaxationStepParser()


# --- Test `matches` Method ---


def test_matches_returns_true_for_start_line(parser: RelaxationStepParser, mock_results: _MutableOptData) -> None:
    line = "       ORCA GEOMETRY RELAXATION STEP   5"
    assert parser.matches(line, mock_results) is True


def test_matches_returns_false_for_other_lines(parser: RelaxationStepParser, mock_results: _MutableOptData) -> None:
    lines = [
        "Some other line",
        "------------------------------------",
        "",
        "Item                Value      Threshold  Converged",
    ]
    for line in lines:
        assert parser.matches(line, mock_results) is False


# --- Test `parse` Method - Normal Operation ---

VALID_RELAXATION_BLOCK = [
    "       ORCA GEOMETRY RELAXATION STEP   5",
    "       *********************************",
    "",
    "-------------------------|Geometry convergence|-------------------------",
    " Item                Value      Threshold  Converged",
    "------------------------------------------------------------------------",
    " Energy change       -0.00012345  0.00000500    NO",
    " RMS gradient         0.00067890  0.00010000    NO",
    " MAX gradient         0.00123456  0.00030000    NO",
    " RMS step             0.00234567  0.00200000    NO",
    " MAX step             0.00345678  0.00400000    YES",
    "........................................................................",
    "",
    " Step calculation ... OK",
    " (Trust radius    =  0.300000)",
    "",
    " New trust radius ...    0.450000",
    "",
    " *** Geometry optimization cycle   5 ***",
    " Geometry step timings:",
    "  Fock matrix formation step  :    0.11 sec",
]


def test_parse_valid_block(
    parser: RelaxationStepParser, mock_results: _MutableOptData, mock_cycle_data: OptimizationCycleData
) -> None:
    """Test parsing a standard, complete relaxation block."""
    iterator = _create_iterator(VALID_RELAXATION_BLOCK)
    current_line = next(iterator)  # Consume the start line

    parser.parse(iterator, current_line, mock_results, mock_cycle_data)

    assert mock_cycle_data.relaxation_step is not None
    relax_data = mock_cycle_data.relaxation_step
    assert isinstance(relax_data, RelaxationStepData)

    assert relax_data.energy_change == -0.00012345
    assert relax_data.rms_gradient == 0.00067890
    assert relax_data.max_gradient == 0.00123456
    assert relax_data.rms_step == 0.00234567
    assert relax_data.max_step == 0.00345678

    assert relax_data.converged_items == {
        "Energy change": False,
        "RMS gradient": False,
        "MAX gradient": False,
        "RMS step": False,
        "MAX step": True,
    }
    assert relax_data.trust_radius == 0.450000


def test_parse_block_with_convergence_message(
    parser: RelaxationStepParser, mock_results: _MutableOptData, mock_cycle_data: OptimizationCycleData
) -> None:
    """Test parsing a block that includes the final convergence message."""
    block_with_convergence = [
        "       ORCA GEOMETRY RELAXATION STEP   8",
        # ... convergence table ...
        "-------------------------|Geometry convergence|-------------------------",
        " Item                Value      Threshold  Converged",
        "------------------------------------------------------------------------",
        " Energy change       -0.00000100  0.00000500    YES",
        " RMS gradient         0.00008000  0.00010000    YES",
        " MAX gradient         0.00025000  0.00030000    YES",
        " RMS step             0.00150000  0.00200000    YES",
        " MAX step             0.00300000  0.00400000    YES",
        "........................................................................",
        "",
        " STEP CONVERGED - OPTIMIZATION RUN IS PERFORMING FINAL CALCULATIONS",
        " *** FINAL SINGLE POINT CALCULATION ***",  # Often follows convergence
        "",
        " *** THE OPTIMIZATION HAS CONVERGED ***",
        "",
        " New trust radius ...    0.300000",  # Trust radius can appear even after convergence message
        "",
        " Geometry step timings:",  # End marker
    ]
    iterator = _create_iterator(block_with_convergence)
    current_line = next(iterator)

    parser.parse(iterator, current_line, mock_results, mock_cycle_data)

    assert mock_cycle_data.relaxation_step is not None
    assert mock_results.termination_status == "CONVERGED"
    assert mock_cycle_data.relaxation_step.trust_radius == 0.300000  # Check trust radius parsing still works


def test_parse_block_without_trust_radius(
    parser: RelaxationStepParser, mock_results: _MutableOptData, mock_cycle_data: OptimizationCycleData
) -> None:
    """Test parsing when the 'New trust radius' line is missing."""
    block_no_trust = [
        "       ORCA GEOMETRY RELAXATION STEP   2",
        "-------------------------|Geometry convergence|-------------------------",
        " Item                Value      Threshold  Converged",
        "------------------------------------------------------------------------",
        " Energy change       -0.00100000  0.00000500    NO",
        " RMS gradient         0.00500000  0.00010000    NO",
        " MAX gradient         0.00800000  0.00030000    NO",
        " RMS step             0.01000000  0.00200000    NO",
        " MAX step             0.01500000  0.00400000    NO",
        "........................................................................",
        "",
        " Geometry step timings:",  # End marker immediately after table
    ]
    iterator = _create_iterator(block_no_trust)
    current_line = next(iterator)

    parser.parse(iterator, current_line, mock_results, mock_cycle_data)

    assert mock_cycle_data.relaxation_step is not None
    relax_data = mock_cycle_data.relaxation_step
    assert relax_data.trust_radius is None
    assert relax_data.energy_change == -0.00100000  # Check other values parsed


# --- Test `parse` Method - Error Conditions ---


def test_parse_eof_in_header(
    parser: RelaxationStepParser, mock_results: _MutableOptData, mock_cycle_data: OptimizationCycleData
) -> None:
    """Test ParsingError if EOF occurs within the convergence table header lines."""
    lines = [
        "       ORCA GEOMETRY RELAXATION STEP   1",
        "-------------------------|Geometry convergence|-------------------------",
        " Item                Value      Threshold  Converged",
        # Missing separator line and rest of file
    ]
    iterator = _create_iterator(lines)
    current_line = next(iterator)

    with pytest.raises(ParsingError, match="File ended unexpectedly within convergence table header."):
        parser.parse(iterator, current_line, mock_results, mock_cycle_data)


def test_parse_eof_during_table_parsing(
    parser: RelaxationStepParser, mock_results: _MutableOptData, mock_cycle_data: OptimizationCycleData
) -> None:
    """Test behavior (warning, no error) if EOF occurs during table parsing."""
    lines = [
        "       ORCA GEOMETRY RELAXATION STEP   1",
        "-------------------------|Geometry convergence|-------------------------",
        " Item                Value      Threshold  Converged",
        "------------------------------------------------------------------------",
        " Energy change       -0.00012345  0.00000500    NO",
        " RMS gradient         0.00067890  0.00010000    NO",
        # EOF here
    ]
    iterator = _create_iterator(lines)
    current_line = next(iterator)

    # Expecting graceful exit with a warning (logged), not an exception
    parser.parse(iterator, current_line, mock_results, mock_cycle_data)

    assert mock_cycle_data.relaxation_step is not None
    relax_data = mock_cycle_data.relaxation_step
    # Check that the items parsed before EOF are present
    assert "Energy change" in relax_data.converged_items
    assert "RMS gradient" in relax_data.converged_items
    assert len(relax_data.converged_items) == 2


def test_parse_called_without_cycle_data(
    parser: RelaxationStepParser, mock_results: _MutableOptData, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that parsing is skipped if current_cycle_data is None."""
    iterator = _create_iterator(VALID_RELAXATION_BLOCK)  # Use valid block for input
    current_line = next(iterator)
    initial_results_state = copy.copy(mock_results)  # Use copy.copy

    with caplog.at_level("WARNING"):
        parser.parse(iterator, current_line, mock_results, None)  # Pass None for cycle data

    assert "RelaxationStepParser matched outside of an optimization cycle. Skipping." in caplog.text
    # Ensure results were not modified
    assert mock_results == initial_results_state

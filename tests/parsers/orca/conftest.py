from pathlib import Path

import pytest

from calcflow.parsers import orca
from calcflow.parsers.orca.typing import _MutableCalculationData

# Load the example output file content once
EXAMPLE_SP_OUT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "calculations" / "examples" / "h2o" / "sp" / "sp.out"
)
EXAMPLE_SP_OUT = EXAMPLE_SP_OUT_PATH.read_text()


@pytest.fixture(scope="module")
def parsed_sp_data() -> orca.CalculationData:
    """Fixture to parse the standard single point output file."""
    return orca.parse_orca_sp_output(EXAMPLE_SP_OUT)


@pytest.fixture
def mutable_data() -> _MutableCalculationData:
    """Provides a fresh mutable data object for testing."""
    return _MutableCalculationData(raw_output=EXAMPLE_SP_OUT)

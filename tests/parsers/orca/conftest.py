from pathlib import Path

import pytest

from calcflow.parsers import orca
from calcflow.parsers.orca.typing import _MutableCalculationData

# Load the example output file content once
ex_folder = Path(__file__).resolve().parents[3] / "data" / "calculations" / "examples" / "orca"
EXAMPLE_SP_OUT_PATH = ex_folder / "h2o" / "sp" / "sp.out"
EXAMPLE_OPT_OUT_PATH = ex_folder / "h2o" / "opt" / "opt.out"


@pytest.fixture(scope="module")
def parsed_sp_data() -> orca.CalculationData:
    """Fixture to parse the standard single point output file."""
    return orca.parse_orca_sp_output(EXAMPLE_SP_OUT_PATH.read_text())


@pytest.fixture(scope="module")
def parsed_opt_data() -> orca.OptimizationData:
    """
    Fixture to read and parse the opt.out file once for the module.
    """
    if not EXAMPLE_OPT_OUT_PATH.exists():
        pytest.skip(f"Test data file not found: {EXAMPLE_OPT_OUT_PATH}")
    content = EXAMPLE_OPT_OUT_PATH.read_text()
    # Assuming parse_optimization_output returns a dictionary or a data object
    data = orca.parse_orca_opt_output(content)
    if data is None:
        pytest.fail("Parsing failed to return data.")
    return data


@pytest.fixture
def mutable_data() -> _MutableCalculationData:
    """Provides a fresh mutable data object for testing."""
    return _MutableCalculationData(raw_output=EXAMPLE_SP_OUT_PATH.read_text())

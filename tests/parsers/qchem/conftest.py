from pathlib import Path

import pytest

from calcflow.parsers import qchem
from calcflow.parsers.qchem.typing import CalculationData, _MutableCalculationData

# Load the example output file content once
ex_folder = Path(__file__).resolve().parents[3] / "data" / "calculations" / "examples" / "qchem"
EXAMPLE_SP_OUT_PATH = ex_folder / "h2o" / "sp.out"


@pytest.fixture(scope="module")
def parsed_sp_data() -> CalculationData:
    """Fixture to parse the standard single point output file."""
    return qchem.parse_qchem_sp_output(EXAMPLE_SP_OUT_PATH.read_text())


@pytest.fixture
def mutable_data() -> _MutableCalculationData:
    """Provides a fresh mutable data object for testing."""
    return _MutableCalculationData(raw_output=EXAMPLE_SP_OUT_PATH.read_text())

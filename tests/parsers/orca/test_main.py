import math
from pathlib import Path

import pytest

from calcflow.parsers import orca

# Load the example output file content once
EXAMPLE_SP_OUT_PATH = Path(__file__).resolve().parents[3] / "data" / "calculations" / "examples" / "sp.out"
EXAMPLE_SP_OUT = EXAMPLE_SP_OUT_PATH.read_text()


@pytest.fixture(scope="module")
def parsed_sp_data() -> orca.CalculationData:
    """Fixture to parse the standard single point output file."""
    return orca.parse_orca_output(EXAMPLE_SP_OUT)


def test_parsing_success(parsed_sp_data: orca.CalculationData) -> None:
    """Verify that the parser runs without throwing critical errors on the example file."""
    assert parsed_sp_data is not None
    assert isinstance(parsed_sp_data, orca.CalculationData)


def test_termination_status(parsed_sp_data: orca.CalculationData) -> None:
    """Test that the termination status is correctly identified."""
    assert parsed_sp_data.termination_status == "NORMAL"


def test_final_energy(parsed_sp_data: orca.CalculationData) -> None:
    """Test that the final single point energy is parsed correctly."""
    assert parsed_sp_data.final_energy_eh is not None
    # Using math.isclose for float comparison
    assert math.isclose(parsed_sp_data.final_energy_eh, -75.313506060725, rel_tol=1e-9)

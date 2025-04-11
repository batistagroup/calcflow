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


def test_input_geometry(parsed_sp_data: orca.CalculationData) -> None:
    """Test that the input geometry is parsed correctly."""
    assert parsed_sp_data.input_geometry is not None
    geometry = parsed_sp_data.input_geometry
    assert len(geometry) == 3
    # Check the first atom (H)
    assert geometry[0].symbol == "H"
    assert math.isclose(geometry[0].x, 1.364990, rel_tol=1e-6)
    assert math.isclose(geometry[0].y, 1.693850, rel_tol=1e-6)
    assert math.isclose(geometry[0].z, -0.197480, rel_tol=1e-6)
    # Check the second atom (O)
    assert geometry[1].symbol == "O"
    assert math.isclose(geometry[1].x, 2.328770, rel_tol=1e-6)
    assert math.isclose(geometry[1].y, 1.562940, rel_tol=1e-6)
    assert math.isclose(geometry[1].z, -0.041680, rel_tol=1e-6)
    # Check the third atom (H)
    assert geometry[2].symbol == "H"
    assert math.isclose(geometry[2].x, 2.702440, rel_tol=1e-6)
    assert math.isclose(geometry[2].y, 1.311570, rel_tol=1e-6)
    assert math.isclose(geometry[2].z, -0.916650, rel_tol=1e-6)

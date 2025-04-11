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


def test_dipole_moment(parsed_sp_data: orca.CalculationData) -> None:
    """Test the parsed dipole moment data."""
    dipole = parsed_sp_data.dipole_moment
    assert dipole is not None
    assert isinstance(dipole, orca.DipoleMomentData)

    assert math.isclose(dipole.x_au, -0.319770911, rel_tol=1e-7)
    assert math.isclose(dipole.y_au, -0.065576153, rel_tol=1e-7)
    assert math.isclose(dipole.z_au, -0.559981644, rel_tol=1e-7)
    assert math.isclose(dipole.total_au, 0.648176757, rel_tol=1e-7)
    assert math.isclose(dipole.total_debye, 1.647534386, rel_tol=1e-7)


def test_dispersion_correction(parsed_sp_data: orca.CalculationData) -> None:
    """Test the parsed dispersion correction data."""
    disp = parsed_sp_data.dispersion_correction
    assert disp is not None
    assert isinstance(disp, orca.DispersionCorrectionData)

    assert "DFTD3" in disp.method  # Check if the method string contains DFTD3
    assert math.isclose(disp.energy_eh, -0.000001638, abs_tol=1e-9)

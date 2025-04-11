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

import math

from calcflow.parsers import orca


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

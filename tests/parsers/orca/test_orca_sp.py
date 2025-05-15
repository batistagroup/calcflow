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


def test_calculation_data_repr_minimal() -> None:
    """Test the repr of CalculationData with only mandatory fields."""
    minimal_data = orca.CalculationData(
        raw_output="",
        termination_status="NORMAL",
        final_energy_eh=-10.0,
        input_geometry=[orca.Atom(symbol="H", x=0.0, y=0.0, z=0.0)],
    )
    repr_str = repr(minimal_data)
    assert "CalculationData(" in repr_str
    assert "termination_status='NORMAL'" in repr_str
    assert "final_energy_eh=-10.00000000" in repr_str
    assert "input_geometry=(1 Atoms)" in repr_str
    assert "scf=None" not in repr_str  # Check optional fields are absent


def test_calculation_data_repr_with_optional_fields(parsed_sp_data: orca.CalculationData) -> None:
    """Test the repr of CalculationData includes summaries for present optional fields."""
    repr_str = repr(parsed_sp_data)
    assert "CalculationData(" in repr_str
    assert "termination_status='NORMAL'" in repr_str
    assert f"final_energy_eh={parsed_sp_data.final_energy_eh:.8f}" in repr_str
    assert f"input_geometry=({len(parsed_sp_data.input_geometry)} Atoms)" in repr_str  # type: ignore

    # Check for the presence of summaries, not their exact content
    assert "scf=ScfData" in repr_str
    assert "orbitals=OrbitalsSet" in repr_str
    assert "atomic_charges=[" in repr_str
    assert "dipole_moment=DipoleMomentData" in repr_str
    # Dispersion might not be in the default test file, check its presence conditionally
    if parsed_sp_data.dispersion_correction:
        assert "dispersion_correction=DispersionCorrectionData" in repr_str

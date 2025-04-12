import math

from calcflow.parsers import orca


def test_orbital_data_presence(parsed_sp_data: orca.CalculationData) -> None:
    """Verify that the OrbitalData object is present."""
    assert parsed_sp_data.orbitals is not None
    assert isinstance(parsed_sp_data.orbitals, orca.OrbitalData)


def test_orbital_energies(parsed_sp_data: orca.CalculationData) -> None:
    """Test the details of parsed orbital energies."""
    orbital_data = parsed_sp_data.orbitals
    assert orbital_data is not None
    orbitals = orbital_data.orbitals
    assert len(orbitals) == 7  # Based on the sp.out file

    # Check first orbital
    orb0 = orbitals[0]
    assert orb0.index == 0
    assert math.isclose(orb0.occupation, 2.0)
    assert math.isclose(orb0.energy_eh, -18.937331, rel_tol=1e-6)
    assert math.isclose(orb0.energy_ev, -515.3110, rel_tol=1e-4)

    # Check last occupied orbital (HOMO)
    homo = orbitals[4]
    assert homo.index == 4
    assert math.isclose(homo.occupation, 2.0)
    assert math.isclose(homo.energy_eh, -0.243811, rel_tol=1e-6)
    assert math.isclose(homo.energy_ev, -6.6344, rel_tol=1e-4)

    # Check first unoccupied orbital (LUMO)
    lumo = orbitals[5]
    assert lumo.index == 5
    assert math.isclose(lumo.occupation, 0.0)
    assert math.isclose(lumo.energy_eh, 0.434377, rel_tol=1e-6)
    assert math.isclose(lumo.energy_ev, 11.8200, rel_tol=1e-4)

    # Check HOMO/LUMO indices
    assert orbital_data.homo_index == 4
    assert orbital_data.lumo_index == 5

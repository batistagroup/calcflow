import math
from pathlib import Path

import pytest

from calcflow.parsers import orca

# Load the example output file content once
EXAMPLE_SP_OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "calculations" / "examples" / "sp.out"
EXAMPLE_SP_OUT = EXAMPLE_SP_OUT_PATH.read_text()


@pytest.fixture(scope="module")
def parsed_sp_data() -> orca.CalculationData:
    """Fixture to parse the standard single point output file."""
    return orca.parse_orca_output(EXAMPLE_SP_OUT)


# --- Phase 1: Basic Parsing and Top-Level Attributes ---


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


# --- Phase 2: SCF Section --- #


def test_scf_data_presence(parsed_sp_data: orca.CalculationData) -> None:
    """Verify that the ScfData object is present."""
    assert parsed_sp_data.scf is not None
    assert isinstance(parsed_sp_data.scf, orca.ScfData)


def test_scf_convergence(parsed_sp_data: orca.CalculationData) -> None:
    """Test the SCF convergence status and number of iterations."""
    scf_data = parsed_sp_data.scf
    assert scf_data is not None
    assert scf_data.converged is True
    assert scf_data.n_iterations == 8


def test_scf_energy(parsed_sp_data: orca.CalculationData) -> None:
    """Test the final SCF energy (should match last iteration)."""
    scf_data = parsed_sp_data.scf
    assert scf_data is not None
    assert math.isclose(scf_data.energy_eh, -75.31352786506064, rel_tol=1e-9)


def test_scf_energy_components(parsed_sp_data: orca.CalculationData) -> None:
    """Test the parsed SCF energy components."""
    components = parsed_sp_data.scf.components if parsed_sp_data.scf else None
    assert components is not None
    assert math.isclose(components.nuclear_repulsion_eh, 8.93764967318280, rel_tol=1e-9)
    assert math.isclose(components.electronic_eh, -84.25117753824344, rel_tol=1e-9)
    assert math.isclose(components.one_electron_eh, -121.92433818585613, rel_tol=1e-9)
    assert math.isclose(components.two_electron_eh, 37.67316064761268, rel_tol=1e-9)
    # XC energy might be None for non-DFT, but should be present here
    assert components.xc_eh is not None
    assert math.isclose(components.xc_eh, -6.561524953854, rel_tol=1e-9)


def test_scf_iteration_history(parsed_sp_data: orca.CalculationData) -> None:
    """Test the SCF iteration history details."""
    history = parsed_sp_data.scf.iteration_history if parsed_sp_data.scf else None
    assert history is not None
    assert len(history) == 8

    # Check first iteration (DIIS)
    first_iter = history[0]
    assert first_iter.iteration == 1
    assert math.isclose(first_iter.energy_eh, -75.2570665045658, rel_tol=1e-9)
    assert math.isclose(first_iter.delta_e_eh, 0.0, abs_tol=1e-9)  # Delta E is 0 for first iter
    assert first_iter.rmsdp is not None
    assert math.isclose(first_iter.rmsdp, 3.14e-02, rel_tol=1e-3)
    assert first_iter.maxdp is not None
    assert math.isclose(first_iter.maxdp, 8.08e-02, rel_tol=1e-3)
    assert first_iter.diis_error is not None
    assert math.isclose(first_iter.diis_error, 2.12e-01, rel_tol=1e-3)
    assert first_iter.max_gradient is None
    assert first_iter.damping is not None
    assert math.isclose(first_iter.damping, 0.700, rel_tol=1e-3)
    assert math.isclose(first_iter.time_sec, 2.6, rel_tol=1e-1)

    # Check transition iteration (SOSCF start)
    soscf_start_iter = history[5]  # Iteration 6
    assert soscf_start_iter.iteration == 6
    assert math.isclose(soscf_start_iter.energy_eh, -75.31352708163747, rel_tol=1e-9)
    assert math.isclose(soscf_start_iter.delta_e_eh, -1.77e-05, rel_tol=1e-3)
    assert soscf_start_iter.rmsdp is not None
    assert math.isclose(soscf_start_iter.rmsdp, 3.76e-04, rel_tol=1e-3)
    assert soscf_start_iter.maxdp is not None
    assert math.isclose(soscf_start_iter.maxdp, 1.26e-03, rel_tol=1e-3)
    assert soscf_start_iter.diis_error is None
    assert soscf_start_iter.max_gradient is not None
    assert math.isclose(soscf_start_iter.max_gradient, 7.37e-04, rel_tol=1e-3)
    assert soscf_start_iter.damping is None
    assert math.isclose(soscf_start_iter.time_sec, 10.7, rel_tol=1e-1)

    # Check last iteration
    last_iter = history[-1]
    assert last_iter.iteration == 8
    assert math.isclose(last_iter.energy_eh, -75.31352786506064, rel_tol=1e-9)
    assert math.isclose(last_iter.delta_e_eh, -1.11e-07, rel_tol=1e-3)
    assert last_iter.rmsdp is not None
    assert math.isclose(last_iter.rmsdp, 8.42e-05, rel_tol=1e-3)
    assert last_iter.maxdp is not None
    assert math.isclose(last_iter.maxdp, 2.23e-04, rel_tol=1e-3)
    assert last_iter.diis_error is None  # Should be SOSCF at the end
    assert last_iter.max_gradient is not None
    assert math.isclose(last_iter.max_gradient, 6.64e-05, rel_tol=1e-3)
    assert last_iter.damping is None
    assert math.isclose(last_iter.time_sec, 21.8, rel_tol=1e-1)


# --- Phase 3: Orbitals, Charges, Dipole, Dispersion --- #


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

import math

from calcflow.parsers import orca


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

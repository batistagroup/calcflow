import pytest

from calcflow.parsers.qchem.typing import (
    MomCalculationResult,
    ScfIteration,
)

# Tolerance for floating point comparisons
ENERGY_TOL = 1e-8
OVERLAP_TOL = 1e-3  # MOM overlaps are usually to 2-3 decimal places in output
S2_TOL = 1e-7 # S^2 values are often to 7 decimal places
SMD_ENERGY_TOL = 1e-4 # SMD energies in kcal/mol usually to 4 decimal places


def test_qchem_54_mom_job_splitting_and_raw_output(parsed_mom_smd_sp_data: MomCalculationResult) -> None:
    """Test that the MOM output is split into two jobs and raw output is stored for QChem 5.4."""
    assert parsed_mom_smd_sp_data is not None, "Fixture did not load/parse data"
    assert parsed_mom_smd_sp_data.job1 is not None, "Initial SCF job data is missing"
    assert parsed_mom_smd_sp_data.job2 is not None, "MOM SCF job data is missing"
    assert isinstance(parsed_mom_smd_sp_data.raw_output, str), "Raw output should be a string"
    assert len(parsed_mom_smd_sp_data.raw_output) > 0, "Raw output should not be empty"
    assert "Running Job 1 of 2" in parsed_mom_smd_sp_data.job1.raw_output
    assert "Running Job 2 of 2" in parsed_mom_smd_sp_data.job2.raw_output
    assert "Running Job 2 of 2" not in parsed_mom_smd_sp_data.job1.raw_output
    assert "Running Job 1 of 2" not in parsed_mom_smd_sp_data.job2.raw_output


def test_qchem_54_mom_initial_scf_job_results(parsed_mom_smd_sp_data: MomCalculationResult) -> None:
    """Test key results from the first (initial SCF) job for QChem 5.4."""
    job1_data = parsed_mom_smd_sp_data.job1
    assert job1_data.scf is not None, "SCF results missing in Job 1"

    assert job1_data.scf.converged is True, "Job 1 SCF should be converged"
    assert job1_data.scf.n_iterations == 7, "Job 1 SCF iteration count mismatch"
    assert job1_data.scf.energy == pytest.approx(-76.4538738687, abs=ENERGY_TOL)

    # Check a specific iteration if needed (e.g., the last one)
    last_iter_job1 = job1_data.scf.iterations[-1]
    assert last_iter_job1.iteration == 7
    assert last_iter_job1.energy == pytest.approx(-76.4562213331, abs=ENERGY_TOL) # Note: energy here might be before final basis set correction if parsed this way
    assert last_iter_job1.diis_error == pytest.approx(3.44e-06, abs=1e-8)

    # Assert MOM fields are None for Job 1 SCF cycles
    for iteration in job1_data.scf.iterations:
        assert iteration.mom_active is None, f"Job 1, Iter {iteration.iteration}: mom_active should be None"
        assert iteration.mom_method_type is None, f"Job 1, Iter {iteration.iteration}: mom_method_type should be None"
        assert iteration.mom_overlap_current is None, (
            f"Job 1, Iter {iteration.iteration}: mom_overlap_current should be None"
        )
        assert iteration.mom_overlap_target is None, (
            f"Job 1, Iter {iteration.iteration}: mom_overlap_target should be None"
        )

    # Check S^2 for Job 1
    assert job1_data.s_squared_final_scf == pytest.approx(0.000000000, abs=S2_TOL)
    assert job1_data.smd.g_enp_au == pytest.approx(-76.456221333, abs=ENERGY_TOL)
    assert job1_data.smd.g_cds_kcal_mol == pytest.approx(1.4731, abs=SMD_ENERGY_TOL)
    assert job1_data.smd.g_tot_au == pytest.approx(-76.453873869, abs=ENERGY_TOL)


def test_qchem_54_mom_driven_scf_job_results(parsed_mom_smd_sp_data: MomCalculationResult) -> None:
    """Test key results from the second (MOM-driven SCF) job for QChem 5.4."""
    job2_data = parsed_mom_smd_sp_data.job2
    assert job2_data.scf is not None, "SCF results missing in Job 2"

    assert job2_data.scf.converged is True, "Job 2 SCF should be converged"
    # From the output: 1 Roothaan + 7 DIIS iterations within the second SCF table
    assert job2_data.scf.n_iterations == 8, "Job 2 SCF iteration count mismatch"
    assert job2_data.scf.energy == pytest.approx(-76.1676313769, abs=ENERGY_TOL) # Final energy after basis set correction

    # Check S^2 for Job 2
    assert job2_data.s_squared_final_scf == pytest.approx(1.003786391, abs=S2_TOL)
    assert job2_data.smd.g_enp_au == pytest.approx(-76.169978841, abs=ENERGY_TOL)
    assert job2_data.smd.g_cds_kcal_mol == pytest.approx(1.4731, abs=SMD_ENERGY_TOL)
    assert job2_data.smd.g_tot_au == pytest.approx(-76.167631377, abs=ENERGY_TOL)


def test_qchem_54_specific_mom_scf_iterations_job2(parsed_mom_smd_sp_data: MomCalculationResult) -> None:
    """Check specific SCF iteration data from Job 2 including MOM details for QChem 5.4."""
    job2_scf_iterations = parsed_mom_smd_sp_data.job2.scf.iterations
    # 1 Roothaan + 7 DIIS iterations = 8 iterations
    assert len(job2_scf_iterations) == 8, "Incorrect number of SCF iterations in Job 2"

    expected_iterations_data = [
        # iter, energy, diis_error, mom_active, mom_method, mom_curr, mom_targ, step_type
        # Note: Iteration numbers in the output restart at 1 for the DIIS table after the Roothaan step.
        # The parser should handle this and provide a continuous 1-based iteration number across the job.
        # Adjusting expectations based on how the parser likely numbers iterations within the job.
        (1, -76.0833919817, 7.23e-03, True, "IMOM", 5.0, 5.0, "Roothaan Step"), # First iteration (Roothaan)
        (2, -76.1266046234, 6.45e-03, True, "IMOM", 4.99, 5.0, None), # First DIIS, Iter 1 in output table
        (3, -76.1420331889, 5.03e-03, True, "IMOM", 4.98, 5.0, None), # Second DIIS, Iter 2
        (4, -76.1698796165, 1.63e-04, True, "IMOM", 4.98, 5.0, None), # Third DIIS, Iter 3
        (5, -76.1700093835, 7.72e-05, True, "IMOM", 4.98, 5.0, None), # Fourth DIIS, Iter 4
        (6, -76.1706945744, 2.70e-04, True, "IMOM", 4.98, 5.0, None), # Fifth DIIS, Iter 5
        (7, -76.1699784580, 1.19e-05, True, "IMOM", 4.98, 5.0, None), # Sixth DIIS, Iter 6
        (8, -76.1699788412, 3.44e-06, None, None, None, None, None), # Seventh DIIS, Iter 7 (Converged)
    ]

    assert len(job2_scf_iterations) == len(expected_iterations_data), "Mismatch in number of Job 2 iterations"

    for i, expected in enumerate(expected_iterations_data):
        actual_iter: ScfIteration = job2_scf_iterations[i]
        (
            exp_iter_num,
            exp_energy,
            exp_diis,
            exp_mom_active,
            exp_mom_method,
            exp_mom_curr,
            exp_mom_targ,
            exp_step_type,
        ) = expected

        assert actual_iter.iteration == exp_iter_num, f"Job 2, Iter {i + 1}: Iteration number mismatch"
        assert actual_iter.energy == pytest.approx(exp_energy, abs=ENERGY_TOL), f"Job 2, Iter {i + 1}: Energy mismatch"
        assert actual_iter.diis_error == pytest.approx(exp_diis, abs=1e-5), f"Job 2, Iter {i + 1}: DIIS error mismatch"
        assert actual_iter.mom_active == exp_mom_active, f"Job 2, Iter {i + 1}: mom_active mismatch"
        assert actual_iter.mom_method_type == exp_mom_method, f"Job 2, Iter {i + 1}: mom_method_type mismatch"
        assert actual_iter.step_type == exp_step_type, f"Job 2, Iter {i + 1}: step_type mismatch"

        # Only check overlap if MOM is expected to be active
        if exp_mom_active is True:
            assert actual_iter.mom_overlap_current == pytest.approx(exp_mom_curr, abs=OVERLAP_TOL), (
                f"Job 2, Iter {i + 1}: mom_overlap_current mismatch"
            )
            assert actual_iter.mom_overlap_target == pytest.approx(exp_mom_targ, abs=OVERLAP_TOL), (
                f"Job 2, Iter {i + 1}: mom_overlap_target mismatch"
            )
        elif exp_mom_active is False:
             assert actual_iter.mom_overlap_current is None, f"Job 2, Iter {i + 1}: mom_overlap_current should be None"
             assert actual_iter.mom_overlap_target is None, f"Job 2, Iter {i + 1}: mom_overlap_target should be None"
        # If exp_mom_active is None, we don't assert anything about mom_overlap fields.

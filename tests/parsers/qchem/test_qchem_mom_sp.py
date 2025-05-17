import pytest

from calcflow.parsers.qchem.typing import (  # Assuming parsed_mom_sp_data fixture provides MomCalculationResult
    MomCalculationResult,
    ScfIteration,
)

# Tolerance for floating point comparisons
ENERGY_TOL = 1e-8
OVERLAP_TOL = 1e-3  # MOM overlaps are usually to 2-3 decimal places in output


def test_mom_job_splitting_and_raw_output(parsed_mom_sp_data: MomCalculationResult) -> None:
    """Test that the MOM output is split into two jobs and raw output is stored."""
    assert parsed_mom_sp_data is not None, "Fixture did not load/parse data"
    assert parsed_mom_sp_data.job1 is not None, "Initial SCF job data is missing"
    assert parsed_mom_sp_data.job2 is not None, "MOM SCF job data is missing"
    assert isinstance(parsed_mom_sp_data.raw_output, str), "Raw output should be a string"
    assert len(parsed_mom_sp_data.raw_output) > 0, "Raw output should not be empty"
    assert "Running Job 1 of 2" in parsed_mom_sp_data.job1.raw_output
    assert "Running Job 2 of 2" in parsed_mom_sp_data.job2.raw_output
    assert "Running Job 2 of 2" not in parsed_mom_sp_data.job1.raw_output
    assert "Running Job 1 of 2" not in parsed_mom_sp_data.job2.raw_output


def test_mom_initial_scf_job_results(parsed_mom_sp_data: MomCalculationResult) -> None:
    """Test key results from the first (initial SCF) job."""
    job1_data = parsed_mom_sp_data.job1
    assert job1_data.scf is not None, "SCF results missing in Job 1"

    assert job1_data.scf.converged is True, "Job 1 SCF should be converged"
    assert job1_data.scf.n_iterations == 6, "Job 1 SCF iteration count mismatch"
    assert job1_data.scf.energy == pytest.approx(-76.44125293, abs=ENERGY_TOL)

    # Check a specific iteration if needed (e.g., the last one)
    last_iter_job1 = job1_data.scf.iterations[-1]
    assert last_iter_job1.iteration == 6
    assert last_iter_job1.energy == pytest.approx(-76.4412529278, abs=ENERGY_TOL)
    assert last_iter_job1.diis_error == pytest.approx(9.39e-06, abs=1e-8)

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

    # <S^2> is not directly part of ScfResults in current structure, usually higher level.
    # If it were added to CalculationData.s_squared_final_scf for example:
    # assert job1_data.s_squared_final_scf == pytest.approx(0.0, abs=1e-7)
    # For now, we expect it to be None if not explicitly parsed into CalculationData by core parser
    # assert job1_data.s_squared_final_scf is None, "s_squared_final_scf should be None for Job 1 based on current parsing"


def test_mom_driven_scf_job_results(parsed_mom_sp_data: MomCalculationResult) -> None:
    """Test key results from the second (MOM-driven SCF) job."""
    job2_data = parsed_mom_sp_data.job2
    assert job2_data.scf is not None, "SCF results missing in Job 2"

    assert job2_data.scf.converged is True, "Job 2 SCF should be converged"
    assert job2_data.scf.n_iterations == 8, "Job 2 SCF iteration count mismatch"
    assert job2_data.scf.energy == pytest.approx(-76.16464101, abs=ENERGY_TOL)

    # <S^2> is not directly part of ScfResults in current structure.
    # If it were added to CalculationData.s_squared_final_scf for example:
    # assert job2_data.s_squared_final_scf == pytest.approx(1.003746412, abs=1e-7)
    # For now, we expect it to be None if not explicitly parsed into CalculationData by core parser
    # assert job2_data.s_squared_final_scf is None, "s_squared_final_scf should be None for Job 2 based on current parsing"


def test_mom_details_in_job2_scf_cycles(parsed_mom_sp_data: MomCalculationResult) -> None:
    """Test MOM-specific details parsed within Job 2 SCF cycles."""
    job2_scf_iterations = parsed_mom_sp_data.job2.scf.iterations
    assert len(job2_scf_iterations) == 8, "Incorrect number of SCF iterations in Job 2"

    # Cycle 1 assertions (MOM data present)
    iter1 = job2_scf_iterations[0]
    assert iter1.iteration == 1
    assert iter1.energy == pytest.approx(-76.0804398088, abs=ENERGY_TOL)
    assert iter1.mom_active is True, "Cycle 1: mom_active should be True"
    assert iter1.mom_method_type == "IMOM", "Cycle 1: mom_method_type mismatch"
    assert iter1.mom_overlap_current == pytest.approx(5.0, abs=OVERLAP_TOL)
    assert iter1.mom_overlap_target == pytest.approx(5.0, abs=OVERLAP_TOL)

    # Cycle 2 assertions (MOM data might be repeated or absent if only on change)
    # Based on output, MOM active signal is repeated, overlap changes.
    iter2 = job2_scf_iterations[1]
    assert iter2.iteration == 2
    assert iter2.energy == pytest.approx(-76.1166679058, abs=ENERGY_TOL)
    assert iter2.mom_active is True, "Cycle 2: mom_active should be True"
    assert iter2.mom_method_type == "IMOM", "Cycle 2: mom_method_type mismatch"
    assert iter2.mom_overlap_current == pytest.approx(
        4.98, abs=OVERLAP_TOL
    )  # Corrected: parser picks up the last MOM overlap line (4.98)
    assert iter2.mom_overlap_target == pytest.approx(5.0, abs=OVERLAP_TOL)  # Target is 5.0

    # Cycle 3 assertions
    iter3 = job2_scf_iterations[2]
    assert iter3.iteration == 3
    assert iter3.energy == pytest.approx(-76.1640019887, abs=ENERGY_TOL)
    assert iter3.mom_active is True, "Cycle 3: mom_active should be True"
    assert iter3.mom_method_type == "IMOM", "Cycle 3: mom_method_type mismatch"
    assert iter3.mom_overlap_current == pytest.approx(4.98, abs=OVERLAP_TOL)
    assert iter3.mom_overlap_target == pytest.approx(5.0, abs=OVERLAP_TOL)

    # Cycle 7 (second to last cycle) assertions
    iter8 = job2_scf_iterations[6]
    assert iter8.iteration == 7
    assert iter8.energy == pytest.approx(-76.1646460372, abs=ENERGY_TOL)
    assert iter8.mom_active is True, "Cycle 7: mom_active should be True"
    assert iter8.mom_method_type == "IMOM", "Cycle 7: mom_method_type mismatch"
    assert iter8.mom_overlap_current == pytest.approx(4.98, abs=OVERLAP_TOL)
    assert iter8.mom_overlap_target == pytest.approx(5.0, abs=OVERLAP_TOL)

    # Example: A cycle where MOM might not be active if logic was different (not the case here)
    # For this output, all cycles after first active signal have MOM data because signal is repeated.


def test_specific_mom_scf_iterations_job2(parsed_mom_sp_data: MomCalculationResult) -> None:
    """Check specific SCF iteration data from Job 2 including MOM details."""
    scf_iterations = parsed_mom_sp_data.job2.scf.iterations

    expected_iterations_data = [
        # iter, energy, diis_error, mom_active, mom_method, mom_curr, mom_targ
        (1, -76.0804398088, 7.45e-03, True, "IMOM", 5.0, 5.0),
        (
            2,
            -76.1166679058,
            6.80e-03,
            True,
            "IMOM",
            4.98,
            5.0,
        ),  # Corrected: parser picks up the last MOM overlap line (4.98)
        (3, -76.1640019887, 5.14e-04, True, "IMOM", 4.98, 5.0),
        (4, -76.1645744120, 9.39e-05, True, "IMOM", 4.98, 5.0),
        (5, -76.1646370016, 2.52e-05, True, "IMOM", 4.98, 5.0),
        (6, -76.1646527070, 3.48e-05, True, "IMOM", 4.98, 5.0),
        (7, -76.1646460372, 2.29e-05, True, "IMOM", 4.98, 5.0),
        (8, -76.1646410132, 1.15e-06, None, None, None, None),
    ]

    assert len(scf_iterations) == len(expected_iterations_data), "Mismatch in number of Job 2 iterations"

    for i, expected in enumerate(expected_iterations_data):
        actual_iter: ScfIteration = scf_iterations[i]
        exp_iter_num, exp_energy, exp_diis, exp_mom_active, exp_mom_method, exp_mom_curr, exp_mom_targ = expected

        assert actual_iter.iteration == exp_iter_num, f"Job 2, Iter {i + 1}: Iteration number mismatch"
        assert actual_iter.energy == pytest.approx(exp_energy, abs=ENERGY_TOL), f"Job 2, Iter {i + 1}: Energy mismatch"
        assert actual_iter.diis_error == pytest.approx(exp_diis, abs=1e-5), f"Job 2, Iter {i + 1}: DIIS error mismatch"
        assert actual_iter.mom_active == exp_mom_active, f"Job 2, Iter {i + 1}: mom_active mismatch"
        assert actual_iter.mom_method_type == exp_mom_method, f"Job 2, Iter {i + 1}: mom_method_type mismatch"
        if exp_mom_active:  # Only check overlap if MOM is expected to be active
            assert actual_iter.mom_overlap_current == pytest.approx(exp_mom_curr, abs=OVERLAP_TOL), (
                f"Job 2, Iter {i + 1}: mom_overlap_current mismatch"
            )
            assert actual_iter.mom_overlap_target == pytest.approx(exp_mom_targ, abs=OVERLAP_TOL), (
                f"Job 2, Iter {i + 1}: mom_overlap_target mismatch"
            )
        else:
            assert actual_iter.mom_overlap_current is None, f"Job 2, Iter {i + 1}: mom_overlap_current should be None"
            assert actual_iter.mom_overlap_target is None, f"Job 2, Iter {i + 1}: mom_overlap_target should be None"

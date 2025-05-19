import pytest

from calcflow.parsers.qchem.typing import (
    ExcitedStateDetailedAnalysis,
    ExcitedStateExcitonDifferenceDM,
    ExcitedStateProperties,
    ExcitonPropertiesSet,
    GroundStateReferenceAnalysis,
    MomCalculationResult,
    NTOStateAnalysis,
    OrbitalTransition,
    ScfIteration,
    TransitionDensityMatrixDetailedAnalysis,
)

# Tolerance for floating point comparisons
ENERGY_TOL = 1e-6
OS_STRENGTH_TOL = 1e-7
MOM_OVERLAP_TOL = 1e-3
EXCITATION_ENERGY_TOL = 1e-4
TRANS_MOM_TOL = 1e-4
AMPLITUDE_TOL = 1e-4
POPULATION_TOL = 1e-4
MULTIPOLE_TOL = 1e-4
RMS_SIZE_TOL = 1e-4
PR_NO_TOL = 1e-4  # For NO participation ratio
N_UNPAIRED_TOL = 1e-5  # For n_unpaired electrons in NO analysis
WEIGHT_PERCENT_TOL = 0.1  # For NTO weights


def test_mom_xas_job_splitting_and_raw_output(parsed_mom_smd_xas_data: MomCalculationResult) -> None:
    """Test that the MOM-XAS output is split into two jobs and raw output is stored."""
    assert parsed_mom_smd_xas_data is not None, "Fixture did not load/parse data"
    assert parsed_mom_smd_xas_data.job1 is not None, "Initial SCF job (Job 1) data is missing"
    assert parsed_mom_smd_xas_data.job2 is not None, "MOM-XAS SCF+TDDFT job (Job 2) data is missing"
    assert isinstance(parsed_mom_smd_xas_data.raw_output, str), "Raw output should be a string"
    assert len(parsed_mom_smd_xas_data.raw_output) > 0, "Raw output should not be empty"
    assert "Running Job 1 of 2" in parsed_mom_smd_xas_data.job1.raw_output
    assert "Running Job 2 of 2" in parsed_mom_smd_xas_data.job2.raw_output
    assert "Running Job 2 of 2" not in parsed_mom_smd_xas_data.job1.raw_output
    assert "Running Job 1 of 2" not in parsed_mom_smd_xas_data.job2.raw_output


def test_mom_xas_job1_scf_results(parsed_mom_smd_xas_data: MomCalculationResult) -> None:
    """Test key results from the first (initial SCF) job of MOM-XAS."""
    job1_data = parsed_mom_smd_xas_data.job1
    assert job1_data.scf is not None, "SCF results missing in Job 1"

    assert job1_data.scf.converged is True, "Job 1 SCF should be converged"
    assert job1_data.scf.n_iterations == 7, "Job 1 SCF iteration count mismatch"
    assert job1_data.scf.energy == pytest.approx(-76.5344743503, abs=ENERGY_TOL)  # E_SCF (including G_PCM)
    assert job1_data.final_energy == pytest.approx(-76.5344743503, abs=ENERGY_TOL)  # G(tot) = G_ENP + G_CDS

    last_iter_job1 = job1_data.scf.iterations[-1]
    assert last_iter_job1.iteration == 7
    assert last_iter_job1.energy == pytest.approx(-76.5368218146, abs=ENERGY_TOL)
    assert last_iter_job1.diis_error == pytest.approx(7.24e-06, abs=1e-8)

    for iteration in job1_data.scf.iterations:
        assert iteration.mom_active is None
        assert iteration.mom_method_type is None
        assert iteration.mom_overlap_current is None
        assert iteration.mom_overlap_target is None

    assert job1_data.smd is not None, "SMD results missing in Job 1"
    assert job1_data.smd.g_pcm_kcal_mol is None
    assert job1_data.smd.g_cds_kcal_mol == pytest.approx(1.4731, abs=MULTIPOLE_TOL)
    assert job1_data.smd.g_enp_au == pytest.approx(-76.536821815, abs=ENERGY_TOL)
    assert job1_data.smd.g_tot_au == pytest.approx(-76.5344743503, abs=ENERGY_TOL)


def test_mom_xas_job2_scf_results(parsed_mom_smd_xas_data: MomCalculationResult) -> None:
    """Test key SCF results from the second (MOM-driven + TDDFT) job."""
    job2_data = parsed_mom_smd_xas_data.job2
    assert job2_data.scf is not None, "SCF results missing in Job 2"

    assert job2_data.scf.converged is True, "Job 2 SCF should be converged"
    assert job2_data.scf.n_iterations == 9, "Job 2 SCF iteration count mismatch"
    assert job2_data.scf.energy == pytest.approx(-76.5344747817, abs=ENERGY_TOL)
    assert job2_data.final_energy == pytest.approx(-76.5344747817, abs=ENERGY_TOL)

    assert job2_data.smd is not None, "SMD results missing in Job 2"
    assert job2_data.smd.g_pcm_kcal_mol is None
    assert job2_data.smd.g_cds_kcal_mol == pytest.approx(1.4731, abs=MULTIPOLE_TOL)
    assert job2_data.smd.g_enp_au == pytest.approx(-76.536822246, abs=ENERGY_TOL)  # Changed value
    assert job2_data.smd.g_tot_au == pytest.approx(-76.534474782, abs=ENERGY_TOL)  # Changed value


def test_mom_xas_job2_scf_iterations(parsed_mom_smd_xas_data: MomCalculationResult) -> None:
    """Test MOM-specific details parsed within Job 2 SCF cycles for MOM-XAS."""
    job2_scf_iterations = parsed_mom_smd_xas_data.job2.scf.iterations
    assert len(job2_scf_iterations) == 9, "Incorrect number of SCF iterations in Job 2"

    expected_iterations_data = [
        (1, -76.5192748967, 7.73e-04, "Roothaan", True, "IMOM", 5.0, 5.0),  # Changed energy
        (1, -76.5328144078, 1.03e-03, None, True, "IMOM", 5.00, 5.0),
        (2, -76.5346554632, 1.11e-03, None, True, "IMOM", 5.00, 5.0),  # Changed energy
        (3, -76.5367667998, 1.55e-04, None, True, "IMOM", 5.00, 5.0),  # Changed energy
        (4, -76.5368198878, 3.46e-05, None, True, "IMOM", 5.00, 5.0),
        (5, -76.5368221812, 4.41e-06, None, True, "IMOM", 5.00, 5.0),
        (6, -76.5368222456, 3.78e-07, None, True, "IMOM", 5.00, 5.0),
        (7, -76.5368222461, 4.39e-08, None, True, "IMOM", 5.00, 5.0),
        (8, -76.5368222461, 9.85e-09, None, None, None, None, None),
    ]

    assert len(job2_scf_iterations) == len(expected_iterations_data)

    for i, expected in enumerate(expected_iterations_data):
        actual_iter: ScfIteration = job2_scf_iterations[i]
        (
            exp_iter_num,
            exp_energy,
            exp_diis,
            exp_step_type,
            exp_mom_active,
            exp_mom_method,
            exp_mom_curr,
            exp_mom_targ,
        ) = expected

        assert actual_iter.iteration == exp_iter_num
        assert actual_iter.energy == pytest.approx(exp_energy, abs=ENERGY_TOL)
        if actual_iter.diis_error is not None:
            assert actual_iter.diis_error == pytest.approx(exp_diis, abs=1e-5)
        assert actual_iter.step_type == exp_step_type
        assert actual_iter.mom_active == exp_mom_active
        assert actual_iter.mom_method_type == exp_mom_method
        if exp_mom_active:
            assert actual_iter.mom_overlap_current == pytest.approx(exp_mom_curr, abs=MOM_OVERLAP_TOL)
            assert actual_iter.mom_overlap_target == pytest.approx(exp_mom_targ, abs=MOM_OVERLAP_TOL)
        else:
            assert actual_iter.mom_overlap_current is None
            assert actual_iter.mom_overlap_target is None


def test_mom_xas_job2_tddft_tda_states(parsed_mom_smd_xas_data: MomCalculationResult) -> None:
    """Test TDDFT/TDA excited state results from Job 2 of MOM-XAS."""
    job2_data = parsed_mom_smd_xas_data.job2
    assert job2_data.tddft is not None, "TDDFT results missing in Job 2"
    tddft_results = job2_data.tddft

    assert tddft_results.tda_states is not None, "TDDFT/TDA states missing"
    assert len(tddft_results.tda_states) == 10, "Incorrect number of TDDFT/TDA states"

    tda_es1: ExcitedStateProperties = tddft_results.tda_states[0]
    assert tda_es1.state_number == 1
    assert tda_es1.excitation_energy_ev == pytest.approx(533.9298, abs=EXCITATION_ENERGY_TOL)  # Changed value
    assert tda_es1.total_energy_au == pytest.approx(-56.91291437, abs=ENERGY_TOL)  # Changed value
    assert tda_es1.multiplicity == "Triplet"  # Output <S**2> :  2.0000
    assert tda_es1.trans_moment_x == pytest.approx(-0.0000, abs=TRANS_MOM_TOL)
    assert tda_es1.oscillator_strength == pytest.approx(0.0000000000, abs=OS_STRENGTH_TOL)
    assert len(tda_es1.transitions) == 4
    # fmt:off
    assert OrbitalTransition(from_label="D", from_idx=1, to_label="V", to_idx=1, amplitude=pytest.approx(-0.6579, abs=AMPLITUDE_TOL), is_alpha_spin=True) in tda_es1.transitions
    assert OrbitalTransition(from_label="D", from_idx=1, to_label="V", to_idx=5, amplitude=pytest.approx(-0.1928, abs=AMPLITUDE_TOL), is_alpha_spin=True) in tda_es1.transitions
    assert OrbitalTransition(from_label="D", from_idx=1, to_label="V", to_idx=1, amplitude=pytest.approx(0.6579, abs=AMPLITUDE_TOL), is_alpha_spin=False) in tda_es1.transitions
    assert OrbitalTransition(from_label="D", from_idx=1, to_label="V", to_idx=5, amplitude=pytest.approx(0.1928, abs=AMPLITUDE_TOL), is_alpha_spin=False) in tda_es1.transitions
    # fmt:on

    tda_es3: ExcitedStateProperties = tddft_results.tda_states[2]
    assert tda_es3.state_number == 3
    assert tda_es3.excitation_energy_ev == pytest.approx(535.1010, abs=EXCITATION_ENERGY_TOL)
    assert tda_es3.multiplicity == "Singlet"  # Output <S**2> :  0.0000
    assert tda_es3.oscillator_strength == pytest.approx(0.0167642433, abs=OS_STRENGTH_TOL)  # Changed value
    assert len(tda_es3.transitions) == 2
    # fmt:off
    assert OrbitalTransition(from_label="D", from_idx=1, to_label="V", to_idx=1, amplitude=pytest.approx(0.6844, abs=AMPLITUDE_TOL), is_alpha_spin=True) in tda_es3.transitions
    assert OrbitalTransition(from_label="D", from_idx=1, to_label="V", to_idx=1, amplitude=pytest.approx(0.6844, abs=AMPLITUDE_TOL), is_alpha_spin=False) in tda_es3.transitions
    # fmt:on

    assert tddft_results.tddft_states is None  # MOM-XAS uses CIS (TDA) based on CIS_N_ROOTS


def test_mom_xas_job2_excited_state_analysis(parsed_mom_smd_xas_data: MomCalculationResult) -> None:
    """Test selected fields from the 'Excited State Analysis' section for Job 2."""
    job2_data = parsed_mom_smd_xas_data.job2
    assert job2_data.tddft is not None, "TDDFT results missing for state analysis"
    analyses = job2_data.tddft.excited_state_analyses
    assert analyses is not None, "Excited state analyses list is missing"
    assert len(analyses) == 10, "Incorrect number of excited state analyses"

    gs_analysis: GroundStateReferenceAnalysis | None = job2_data.gs_reference_analysis
    assert gs_analysis is not None, "Ground state reference analysis missing"
    assert gs_analysis.mulliken is not None
    assert len(gs_analysis.mulliken.populations) == 3
    assert gs_analysis.mulliken.populations[0].symbol == "H"
    assert gs_analysis.mulliken.populations[0].charge_e == pytest.approx(0.304422, abs=POPULATION_TOL)
    assert gs_analysis.mulliken.populations[0].spin_e == pytest.approx(0.000000, abs=POPULATION_TOL)
    assert gs_analysis.no_data_rks_or_spin_traced is not None
    assert gs_analysis.no_data_rks_or_spin_traced.n_unpaired_nl == pytest.approx(0.00000, abs=N_UNPAIRED_TOL)

    es1_analysis: ExcitedStateDetailedAnalysis = analyses[0]
    assert es1_analysis.state_number == 1
    assert es1_analysis.no_data is not None
    assert es1_analysis.no_data.n_unpaired_nl == pytest.approx(2.00000, abs=N_UNPAIRED_TOL)
    assert es1_analysis.no_data.pr_no == pytest.approx(2.000000, abs=PR_NO_TOL)
    assert es1_analysis.mulliken is not None
    assert len(es1_analysis.mulliken.populations) == 3

    assert es1_analysis.mulliken.populations[0].symbol == "H"
    assert es1_analysis.mulliken.populations[0].charge_e == pytest.approx(-0.143709, abs=POPULATION_TOL)
    assert es1_analysis.mulliken.populations[0].spin_e == pytest.approx(0.000002, abs=POPULATION_TOL)

    assert es1_analysis.multipole is not None
    assert es1_analysis.multipole.dipole_moment_debye is not None
    assert es1_analysis.multipole.dipole_moment_debye.magnitude == pytest.approx(
        0.854467, abs=MULTIPOLE_TOL
    )  # Changed value

    assert es1_analysis.exciton_difference_dm_analysis is not None
    exciton1_dm: ExcitedStateExcitonDifferenceDM = es1_analysis.exciton_difference_dm_analysis
    assert exciton1_dm.electron_hole_separation_ang == pytest.approx(0.694114, abs=RMS_SIZE_TOL)
    assert exciton1_dm.hole_size_ang == pytest.approx(0.122441, abs=RMS_SIZE_TOL)
    assert exciton1_dm.electron_size_ang == pytest.approx(1.607068, abs=RMS_SIZE_TOL)

    es3_analysis: ExcitedStateDetailedAnalysis = analyses[2]
    assert es3_analysis.state_number == 3
    assert es3_analysis.mulliken is not None
    assert es3_analysis.mulliken.populations[1].symbol == "O"
    assert es3_analysis.mulliken.populations[1].charge_e == pytest.approx(0.347950, abs=POPULATION_TOL)


def test_mom_xas_job2_transition_density_matrix_analysis(parsed_mom_smd_xas_data: MomCalculationResult) -> None:
    """Test selected fields from 'Transition Density Matrix Analysis' for Job 2."""
    job2_data = parsed_mom_smd_xas_data.job2
    assert job2_data.tddft is not None, "TDDFT results missing for transition DM analysis"
    trans_dm_analyses = job2_data.tddft.transition_dm_analyses
    assert trans_dm_analyses is not None, "Transition DM analyses list is missing"
    assert len(trans_dm_analyses) == 10, "Incorrect number of transition DM analyses"

    tdm1: TransitionDensityMatrixDetailedAnalysis = trans_dm_analyses[0]
    assert tdm1.state_number == 1
    assert tdm1.mulliken is None

    assert tdm1.ct_numbers is not None
    assert tdm1.ct_numbers.omega == pytest.approx(1.0000, abs=POPULATION_TOL)
    assert tdm1.ct_numbers.loc_a is None

    assert tdm1.exciton_analysis is not None
    exciton_tdm1_props: ExcitonPropertiesSet | None = tdm1.exciton_analysis.total_properties
    assert exciton_tdm1_props is not None
    assert exciton_tdm1_props.total_transition_dipole_moment == pytest.approx(0.000000, abs=TRANS_MOM_TOL)
    assert exciton_tdm1_props.hole_electron_distance_ang == pytest.approx(0.694114, abs=RMS_SIZE_TOL)

    tdm3: TransitionDensityMatrixDetailedAnalysis = trans_dm_analyses[2]
    assert tdm3.state_number == 3
    assert tdm3.mulliken is None
    assert tdm3.exciton_analysis is not None
    exciton_tdm3_props: ExcitonPropertiesSet | None = tdm3.exciton_analysis.total_properties
    assert exciton_tdm3_props is not None
    assert exciton_tdm3_props.total_transition_dipole_moment == pytest.approx(0.090892, abs=TRANS_MOM_TOL)


def test_mom_xas_job2_nto_analysis(parsed_mom_smd_xas_data: MomCalculationResult) -> None:
    """Test selected fields from 'SA-NTO Decomposition' for Job 2."""
    job2_data = parsed_mom_smd_xas_data.job2
    assert job2_data.tddft is not None, "TDDFT results missing for NTO analysis"
    nto_analyses = job2_data.tddft.nto_analyses
    assert nto_analyses is not None, "NTO analyses list is missing"
    assert len(nto_analyses) == 10, "Incorrect number of NTO analyses"

    nto1: NTOStateAnalysis = nto_analyses[0]
    assert nto1.state_number == 1
    assert nto1.contributions is not None
    # Alpha: H- 0 -> L+ 2: -0.6131 ( 37.6%); H- 0 -> L+ 1: -0.3502 ( 12.3%)
    # Beta:  H- 0 -> L+ 2:  0.6131 ( 37.6%); H- 0 -> L+ 1:  0.3502 ( 12.3%)
    # The parser creates separate NTOContribution objects for each line, with is_alpha_spin flag.
    # We expect 4 contributions in total for state 1 based on output format.
    assert len(nto1.contributions) == 4

    # Example check for first alpha contribution: H-0 -> L+2
    alpha_contrib1 = next(
        c for c in nto1.contributions if c.is_alpha_spin and c.hole_offset == 0 and c.electron_offset == 2
    )
    assert alpha_contrib1.hole_reference == "H"
    assert alpha_contrib1.electron_reference == "L"
    assert alpha_contrib1.coefficient == pytest.approx(-0.6131, abs=AMPLITUDE_TOL)
    assert alpha_contrib1.weight_percent == pytest.approx(37.6, abs=WEIGHT_PERCENT_TOL)

    # Example check for first beta contribution: H-0 -> L+2
    beta_contrib1 = next(
        c for c in nto1.contributions if not c.is_alpha_spin and c.hole_offset == 0 and c.electron_offset == 2
    )
    assert beta_contrib1.hole_reference == "H"
    assert beta_contrib1.electron_reference == "L"
    assert beta_contrib1.coefficient == pytest.approx(0.6131, abs=AMPLITUDE_TOL)
    assert beta_contrib1.weight_percent == pytest.approx(37.6, abs=WEIGHT_PERCENT_TOL)

    assert nto1.omega_alpha_percent == pytest.approx(50.0, abs=WEIGHT_PERCENT_TOL)
    assert nto1.omega_beta_percent == pytest.approx(50.0, abs=WEIGHT_PERCENT_TOL)

    nto3: NTOStateAnalysis = nto_analyses[2]
    assert nto3.state_number == 3
    assert nto3.contributions is not None
    assert len(nto3.contributions) == 4  # Expect 2 alpha, 2 beta
    # Alpha: H- 0 -> L+ 2:  0.5748 ( 33.0%)
    alpha_contrib_s3_1 = next(
        c for c in nto3.contributions if c.is_alpha_spin and c.hole_offset == 0 and c.electron_offset == 2
    )
    assert alpha_contrib_s3_1.coefficient == pytest.approx(0.5748, abs=AMPLITUDE_TOL)
    assert alpha_contrib_s3_1.weight_percent == pytest.approx(33.0, abs=WEIGHT_PERCENT_TOL)

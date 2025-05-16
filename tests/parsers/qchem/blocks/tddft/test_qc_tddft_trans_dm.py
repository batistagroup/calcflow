import pytest

from calcflow.parsers.qchem.typing import (
    CalculationData,
    TransitionDensityMatrixDetailedAnalysis,
)


def find_state_analysis(
    analyses: list[TransitionDensityMatrixDetailedAnalysis] | None, state_number: int, multiplicity: str
) -> TransitionDensityMatrixDetailedAnalysis | None:
    if analyses is None:
        return None
    for analysis in analyses:
        if analysis.state_number == state_number and analysis.multiplicity.lower() == multiplicity.lower():
            return analysis
    return None


def test_tddft_transition_dm_analysis_singlet_3(parsed_tddft_pc2_data: CalculationData) -> None:
    """Tests parsing of specific values for Singlet 3 in TDM analysis."""
    assert parsed_tddft_pc2_data.tddft is not None
    assert parsed_tddft_pc2_data.tddft.transition_dm_analyses is not None

    s3_analysis = find_state_analysis(
        parsed_tddft_pc2_data.tddft.transition_dm_analyses,
        state_number=3,
        multiplicity="Singlet",
    )
    assert s3_analysis is not None, "Singlet 3 analysis not found"

    # Test Mulliken Analysis
    mulliken = s3_analysis.mulliken
    assert mulliken is not None
    assert mulliken.populations is not None
    assert len(mulliken.populations) == 3
    # Atom 1 H (index 0)
    assert mulliken.populations[0].symbol == "H"
    assert mulliken.populations[0].atom_index == 0
    pytest.approx(mulliken.populations[0].transition_charge_e, 0.062445)
    pytest.approx(mulliken.populations[0].hole_charge_rks, 0.067398)
    pytest.approx(mulliken.populations[0].electron_charge_rks, -0.539695)
    pytest.approx(mulliken.populations[0].delta_charge_rks, -0.472297)
    # Sums
    assert mulliken.sum_abs_trans_charges_qta == pytest.approx(0.248846)
    assert mulliken.sum_sq_trans_charges_qt2 == pytest.approx(0.023222)

    # Test CT Numbers
    ct = s3_analysis.ct_numbers
    assert ct is not None
    assert ct.omega == pytest.approx(1.0019)
    assert ct.loc_a == pytest.approx(0.6119)
    assert ct.two_alpha_beta_overlap == pytest.approx(1.0019)
    assert ct.loc == pytest.approx(0.0014)
    assert ct.phe_overlap == pytest.approx(-0.0169)

    # Test Exciton Analysis
    exciton = s3_analysis.exciton_analysis.total_properties
    assert exciton is not None
    assert exciton.total_transition_dipole_moment == pytest.approx(1.686852)
    assert exciton.transition_dipole_moment_components is not None
    assert exciton.transition_dipole_moment_components[0] == pytest.approx(-0.835815)
    assert exciton.transition_dipole_moment_components[1] == pytest.approx(-0.169604)
    assert exciton.transition_dipole_moment_components[2] == pytest.approx(-1.455376)
    assert exciton.hole_size_ang == pytest.approx(0.855723)
    assert exciton.hole_size_ang_components is not None
    assert exciton.hole_size_ang_components[0] == pytest.approx(0.491784)
    assert exciton.hole_size_ang_components[1] == pytest.approx(0.375972)
    assert exciton.hole_size_ang_components[2] == pytest.approx(0.590808)
    assert exciton.correlation_coefficient == pytest.approx(0.000925)
    assert exciton.electron_size_ang == pytest.approx(1.969431)
    assert exciton.electron_size_ang_components is not None
    assert exciton.electron_size_ang_components[0] == pytest.approx(1.248228)
    assert exciton.electron_size_ang_components[1] == pytest.approx(1.043375)
    assert exciton.electron_size_ang_components[2] == pytest.approx(1.109935)
    assert exciton.center_of_mass_size_ang == pytest.approx(1.074016)


def test_tddft_transition_dm_analysis_singlet_7(parsed_tddft_pc2_data: CalculationData) -> None:
    """Tests parsing of specific values for Singlet 7 in TDM analysis."""
    assert parsed_tddft_pc2_data.tddft is not None
    assert parsed_tddft_pc2_data.tddft.transition_dm_analyses is not None

    s7_analysis = find_state_analysis(
        parsed_tddft_pc2_data.tddft.transition_dm_analyses,
        state_number=7,
        multiplicity="Singlet",
    )
    assert s7_analysis is not None, "Singlet 7 analysis not found"

    # Test Mulliken Analysis
    mulliken = s7_analysis.mulliken
    assert mulliken is not None
    assert mulliken.populations is not None
    assert len(mulliken.populations) == 3
    # Atom 2 O (index 1)
    assert mulliken.populations[1].symbol == "O"
    assert mulliken.populations[1].atom_index == 1
    pytest.approx(mulliken.populations[1].transition_charge_e, 0.000281)
    pytest.approx(mulliken.populations[1].hole_charge_rks, 0.967409)
    pytest.approx(mulliken.populations[1].electron_charge_rks, -0.239249)
    pytest.approx(mulliken.populations[1].delta_charge_rks, 0.728160)
    # Sums
    assert mulliken.sum_abs_trans_charges_qta == pytest.approx(0.000563)
    assert mulliken.sum_sq_trans_charges_qt2 == pytest.approx(0.000000)

    # Test CT Numbers
    ct = s7_analysis.ct_numbers
    assert ct is not None
    assert ct.loc == pytest.approx(0.0035)
    assert ct.phe_overlap == pytest.approx(-0.0129)
    assert ct.omega == pytest.approx(1.0020)
    assert ct.loc_a == pytest.approx(0.2509)

    # Test Exciton Analysis
    exciton = s7_analysis.exciton_analysis.total_properties
    assert exciton is not None
    assert exciton.transition_r_squared_au == pytest.approx(2.333932)
    assert exciton.electron_position_ang is not None
    assert exciton.electron_position_ang[0] == pytest.approx(2.022229)
    assert exciton.electron_position_ang[1] == pytest.approx(1.499302)
    assert exciton.electron_position_ang[2] == pytest.approx(-0.580968)
    assert exciton.electron_size_ang == pytest.approx(1.844460)
    assert exciton.electron_size_ang_components is not None
    assert exciton.electron_size_ang_components[0] == pytest.approx(1.073142)
    assert exciton.electron_size_ang_components[1] == pytest.approx(1.053041)
    assert exciton.electron_size_ang_components[2] == pytest.approx(1.068412)
    assert exciton.covariance_rh_re_ang_sq == pytest.approx(0.012279)
    assert exciton.hole_size_ang == pytest.approx(0.828215)  # Added based on output
    assert exciton.center_of_mass_size_ang == pytest.approx(1.013969)  # Added


def test_tddft_uks_transition_dm_analysis_exc_state_6(parsed_tddft_uks_pc2_data: CalculationData) -> None:
    """Tests parsing of specific values for Excited State 6 in TDM analysis from UKS output."""
    assert parsed_tddft_uks_pc2_data.tddft is not None
    assert parsed_tddft_uks_pc2_data.tddft.transition_dm_analyses is not None

    es6_analysis = find_state_analysis(
        parsed_tddft_uks_pc2_data.tddft.transition_dm_analyses,
        state_number=6,
        multiplicity="Excited State",  # UKS output uses this label
    )
    assert es6_analysis is not None, "Excited State 6 analysis not found"

    # Test Mulliken Analysis (UKS specific fields)
    mulliken = es6_analysis.mulliken
    assert mulliken is not None
    assert mulliken.populations is not None
    assert len(mulliken.populations) == 3
    # Atom 1 H (index 0)
    pop_h1 = mulliken.populations[0]
    assert pop_h1.symbol == "H"
    assert pop_h1.atom_index == 0
    assert pop_h1.transition_charge_e == pytest.approx(0.062447)
    assert pop_h1.hole_charge_alpha_uks == pytest.approx(0.033699)
    assert pop_h1.hole_charge_beta_uks == pytest.approx(0.033699)
    assert pop_h1.electron_charge_alpha_uks == pytest.approx(-0.269847)
    assert pop_h1.electron_charge_beta_uks == pytest.approx(-0.269847)
    # Sums
    assert mulliken.sum_abs_trans_charges_qta == pytest.approx(0.248854)
    assert mulliken.sum_sq_trans_charges_qt2 == pytest.approx(0.023223)

    # Test CT Numbers (UKS specific fields)
    ct = es6_analysis.ct_numbers
    assert ct is not None
    assert ct.omega == pytest.approx(1.0019)
    assert ct.omega_alpha == pytest.approx(0.5010)
    assert ct.omega_beta == pytest.approx(0.5010)
    assert ct.two_alpha_beta_overlap == pytest.approx(1.0019)
    assert ct.loc == pytest.approx(0.0014)
    assert ct.loc_alpha == pytest.approx(0.0007)
    assert ct.loc_beta == pytest.approx(0.0007)
    assert ct.loc_a == pytest.approx(0.6119)
    assert ct.loc_a_alpha == pytest.approx(0.3060)
    assert ct.loc_a_beta == pytest.approx(0.3060)
    assert ct.phe_overlap == pytest.approx(-0.0169)
    assert ct.phe_overlap_alpha == pytest.approx(-0.0169)
    assert ct.phe_overlap_beta == pytest.approx(-0.0169)

    # Test Exciton Analysis
    assert es6_analysis.exciton_analysis is not None
    exciton_total = es6_analysis.exciton_analysis.total_properties
    assert exciton_total is not None
    assert exciton_total.total_transition_dipole_moment == pytest.approx(1.686853)
    assert exciton_total.transition_dipole_moment_components == pytest.approx([-0.835816, -0.169604, -1.455377])
    assert exciton_total.transition_r_squared_au == pytest.approx(2.519291)
    assert exciton_total.transition_r_squared_au_components == pytest.approx([2.390912, 0.412279, -0.678521])
    assert exciton_total.hole_position_ang == pytest.approx([2.359689, 1.569308, 0.012609])
    assert exciton_total.electron_position_ang == pytest.approx([1.988725, 1.494574, -0.630774])
    assert exciton_total.hole_electron_distance_ang == pytest.approx(0.746418)
    assert exciton_total.hole_size_ang == pytest.approx(0.855722)
    assert exciton_total.hole_size_ang_components == pytest.approx([0.491784, 0.375972, 0.590808])
    assert exciton_total.electron_size_ang == pytest.approx(1.969433)
    assert exciton_total.electron_size_ang_components == pytest.approx([1.248229, 1.043376, 1.109936])
    assert exciton_total.rms_electron_hole_separation_ang == pytest.approx(2.272653)
    assert exciton_total.rms_electron_hole_separation_ang_components == pytest.approx([1.395836, 1.105771, 1.412042])
    assert exciton_total.covariance_rh_re_ang_sq == pytest.approx(0.001558)
    assert exciton_total.correlation_coefficient == pytest.approx(0.000925)
    assert exciton_total.center_of_mass_size_ang == pytest.approx(1.074016)
    assert exciton_total.center_of_mass_size_ang_components == pytest.approx([0.668788, 0.557412, 0.628908])

    exciton_alpha = es6_analysis.exciton_analysis.alpha_spin_properties
    assert exciton_alpha is not None
    assert exciton_alpha.total_transition_dipole_moment == pytest.approx(0.843427)
    assert exciton_alpha.transition_dipole_moment_components == pytest.approx([-0.417908, -0.084802, -0.727688])
    # For this specific state, many alpha/beta properties mirror the total ones used above for brevity in example
    # but in a real test, one might check more if they can differ significantly.
    assert exciton_alpha.hole_position_ang == pytest.approx([2.359689, 1.569308, 0.012609])
    assert exciton_alpha.electron_position_ang == pytest.approx([1.988725, 1.494574, -0.630774])

    exciton_beta = es6_analysis.exciton_analysis.beta_spin_properties
    assert exciton_beta is not None
    assert exciton_beta.total_transition_dipole_moment == pytest.approx(0.843427)
    assert exciton_beta.transition_dipole_moment_components == pytest.approx([-0.417908, -0.084802, -0.727688])
    assert exciton_beta.hole_position_ang == pytest.approx([2.359689, 1.569308, 0.012609])
    assert exciton_beta.electron_position_ang == pytest.approx([1.988725, 1.494574, -0.630774])

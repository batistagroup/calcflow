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

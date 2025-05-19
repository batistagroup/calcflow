from pytest import approx

from calcflow.parsers.qchem.typing import (
    CalculationData,
)


def test_parse_qchem_54_uks_tddft_state3(parsed_tddft_uks_pc2_data: CalculationData) -> None:
    """
    Tests parsing of Excited State 3 data from Q-Chem 5.4 UKS output.
    """
    data = parsed_tddft_uks_pc2_data
    assert data.tddft.excited_state_analyses is not None
    # Find state 3
    state3_data = None
    for state in data.tddft.excited_state_analyses:
        if state.state_number == 3:
            state3_data = state
            break

    assert state3_data is not None

    # NOs (spin-traced)
    assert state3_data.no_data is not None
    assert state3_data.no_data.frontier_occupations == approx([0.9996, 1.0003])
    assert state3_data.no_data.n_electrons == approx(10.000000)
    assert state3_data.no_data.n_unpaired == approx(2.00096)
    assert state3_data.no_data.n_unpaired_nl == approx(2.00000)
    assert state3_data.no_data.pr_no == approx(2.003299)

    # Mulliken Population Analysis
    assert state3_data.mulliken is not None
    assert len(state3_data.mulliken.populations) == 3
    # Atom 1 H
    assert state3_data.mulliken.populations[0].atom_index == 0
    assert state3_data.mulliken.populations[0].symbol == "H"
    assert state3_data.mulliken.populations[0].charge_e == approx(-0.291180)
    assert state3_data.mulliken.populations[0].spin_e == approx(0.000000)
    assert state3_data.mulliken.populations[0].hole_charge_alpha == approx(0.008492)
    assert state3_data.mulliken.populations[0].electron_charge_alpha == approx(-0.269360)
    assert state3_data.mulliken.populations[0].delta_charge == approx(
        2 * (-0.269360 - (0.008492))
    )  # Calc delta_charge = charge - hole + electron
    # Atom 2 O
    assert state3_data.mulliken.populations[1].atom_index == 1
    assert state3_data.mulliken.populations[1].symbol == "O"
    assert state3_data.mulliken.populations[1].charge_e == approx(0.588170)
    assert state3_data.mulliken.populations[1].spin_e == approx(-0.000000)  # Note the negative zero in output
    assert state3_data.mulliken.populations[1].hole_charge_alpha == approx(0.483579)
    assert state3_data.mulliken.populations[1].electron_charge_alpha == approx(0.040735)
    assert state3_data.mulliken.populations[1].delta_charge == approx(2 * (0.040735 - 0.483579))
    # Atom 3 H
    assert state3_data.mulliken.populations[2].atom_index == 2
    assert state3_data.mulliken.populations[2].symbol == "H"
    assert state3_data.mulliken.populations[2].charge_e == approx(-0.296990)
    assert state3_data.mulliken.populations[2].spin_e == approx(0.000000)
    assert state3_data.mulliken.populations[2].hole_charge_alpha == approx(0.008512)
    assert state3_data.mulliken.populations[2].electron_charge_alpha == approx(-0.271960)
    assert state3_data.mulliken.populations[2].delta_charge == approx(2 * (-0.271960 - 0.008512))

    # Multipole moment analysis
    assert state3_data.multipole is not None
    assert state3_data.multipole.molecular_charge == approx(0.000000)
    assert state3_data.multipole.n_electrons == approx(10.000000)
    assert state3_data.multipole.center_electronic_charge_ang == approx([2.254024, 1.547500, -0.173092])
    assert state3_data.multipole.center_nuclear_charge_ang == approx([2.269759, 1.550894, -0.144757])
    assert state3_data.multipole.dipole_moment_debye is not None
    assert state3_data.multipole.dipole_moment_debye.magnitude == approx(1.565289)
    assert state3_data.multipole.dipole_moment_debye.x == approx(0.755798)
    assert state3_data.multipole.dipole_moment_debye.y == approx(0.163020)
    assert state3_data.multipole.dipole_moment_debye.z == approx(1.361001)
    assert state3_data.multipole.rms_density_size_ang == approx(0.991391)  # Scalar value
    assert state3_data.multipole.rms_density_size_ang_comps == approx([0.658415, 0.465752, 0.576560])  # Components

    # Exciton analysis of the difference density matrix
    assert state3_data.exciton_difference_dm_analysis is not None  # Total
    assert state3_data.exciton_difference_dm_analysis.hole_center_ang == approx([2.311751, 1.559451, -0.071493])
    assert state3_data.exciton_difference_dm_analysis.electron_center_ang == approx([1.947497, 1.483232, -0.716414])
    assert state3_data.exciton_difference_dm_analysis.electron_hole_separation_ang == approx(0.744589)
    assert state3_data.exciton_difference_dm_analysis.hole_size_ang == approx(0.827979)  # Scalar
    assert state3_data.exciton_difference_dm_analysis.hole_size_ang_comps == approx(
        [0.390136, 0.616866, 0.390920]
    )  # Components
    assert state3_data.exciton_difference_dm_analysis.electron_size_ang == approx(2.119876)  # Scalar
    assert state3_data.exciton_difference_dm_analysis.electron_size_ang_comps == approx(
        [1.529608, 0.958548, 1.111467]
    )  # Components

    # Alpha spin Exciton analysis
    assert state3_data.exciton_difference_dm_analysis_alpha is not None
    assert state3_data.exciton_difference_dm_analysis_alpha.hole_center_ang == approx([2.311751, 1.559451, -0.071493])
    assert state3_data.exciton_difference_dm_analysis_alpha.electron_center_ang == approx(
        [1.947497, 1.483232, -0.716414]
    )
    assert state3_data.exciton_difference_dm_analysis_alpha.electron_hole_separation_ang == approx(0.744589)
    assert state3_data.exciton_difference_dm_analysis_alpha.hole_size_ang == approx(0.827979)  # Scalar
    assert state3_data.exciton_difference_dm_analysis_alpha.hole_size_ang_comps == approx(
        [0.390136, 0.616866, 0.390920]
    )  # Components
    assert state3_data.exciton_difference_dm_analysis_alpha.electron_size_ang == approx(2.119876)  # Scalar
    assert state3_data.exciton_difference_dm_analysis_alpha.electron_size_ang_comps == approx(
        [1.529608, 0.958548, 1.111467]
    )  # Components

    # Beta spin Exciton analysis
    assert state3_data.exciton_difference_dm_analysis_beta is not None
    assert state3_data.exciton_difference_dm_analysis_beta.hole_center_ang == approx([2.311751, 1.559451, -0.071493])
    assert state3_data.exciton_difference_dm_analysis_beta.electron_center_ang == approx(
        [1.947497, 1.483232, -0.716414]
    )
    assert state3_data.exciton_difference_dm_analysis_beta.electron_hole_separation_ang == approx(0.744589)
    assert state3_data.exciton_difference_dm_analysis_beta.hole_size_ang == approx(0.827979)  # Scalar
    assert state3_data.exciton_difference_dm_analysis_beta.hole_size_ang_comps == approx(
        [0.390136, 0.616866, 0.390920]
    )  # Components
    assert state3_data.exciton_difference_dm_analysis_beta.electron_size_ang == approx(2.119876)  # Scalar
    assert state3_data.exciton_difference_dm_analysis_beta.electron_size_ang_comps == approx(
        [1.529608, 0.958548, 1.111467]
    )  # Components


# Define test for state 7 similarly
def test_parse_qchem_54_uks_tddft_state7(parsed_tddft_uks_pc2_data: CalculationData) -> None:
    """
    Tests parsing of Excited State 7 data from Q-Chem 5.4 UKS output.
    """
    data = parsed_tddft_uks_pc2_data
    assert data.tddft.excited_state_analyses is not None
    # Find state 7
    state7_data = None
    for state in data.tddft.excited_state_analyses:
        if state.state_number == 7:
            state7_data = state
            break

    assert state7_data is not None

    # NOs (spin-traced)
    assert state7_data.no_data is not None
    assert state7_data.no_data.frontier_occupations == approx([0.9804, 1.0193])
    assert state7_data.no_data.n_electrons == approx(10.000000)
    assert state7_data.no_data.n_unpaired == approx(2.00541)
    assert state7_data.no_data.n_unpaired_nl == approx(2.00217)
    assert state7_data.no_data.pr_no == approx(2.090472)

    # Mulliken Population Analysis
    assert state7_data.mulliken is not None
    assert len(state7_data.mulliken.populations) == 3
    # Atom 1 H
    assert state7_data.mulliken.populations[0].atom_index == 0
    assert state7_data.mulliken.populations[0].symbol == "H"
    assert state7_data.mulliken.populations[0].charge_e == approx(-0.229283)
    assert state7_data.mulliken.populations[0].spin_e == approx(-0.000000)
    assert state7_data.mulliken.populations[0].hole_charge_alpha == approx(0.037057)
    assert state7_data.mulliken.populations[0].electron_charge_alpha == approx(-0.266976)
    assert state7_data.mulliken.populations[0].delta_charge == approx(
        2 * (-0.266976 - 0.037057)
    )  # Calc delta_charge = charge - hole + electron
    # Atom 2 O
    assert state7_data.mulliken.populations[1].atom_index == 1
    assert state7_data.mulliken.populations[1].symbol == "O"
    assert state7_data.mulliken.populations[1].charge_e == approx(0.466761)
    assert state7_data.mulliken.populations[1].spin_e == approx(0.000000)
    assert state7_data.mulliken.populations[1].hole_charge_alpha == approx(0.427360)
    assert state7_data.mulliken.populations[1].electron_charge_alpha == approx(0.036251)
    assert state7_data.mulliken.populations[1].delta_charge == approx(2 * (0.036251 - 0.427360))
    # Atom 3 H
    assert state7_data.mulliken.populations[2].atom_index == 2
    assert state7_data.mulliken.populations[2].symbol == "H"
    assert state7_data.mulliken.populations[2].charge_e == approx(-0.237478)
    assert state7_data.mulliken.populations[2].spin_e == approx(-0.000000)
    assert state7_data.mulliken.populations[2].hole_charge_alpha == approx(0.036935)
    assert state7_data.mulliken.populations[2].electron_charge_alpha == approx(-0.270627)
    assert state7_data.mulliken.populations[2].delta_charge == approx(2 * (-0.270627 - 0.036935))

    # Multipole moment analysis
    assert state7_data.multipole is not None
    assert state7_data.multipole.molecular_charge == approx(0.000000)
    assert state7_data.multipole.n_electrons == approx(10.000000)
    assert state7_data.multipole.center_electronic_charge_ang == approx([2.250575, 1.546716, -0.179492])
    assert state7_data.multipole.center_nuclear_charge_ang == approx([2.269759, 1.550894, -0.144757])
    assert state7_data.multipole.dipole_moment_debye is not None
    assert state7_data.multipole.dipole_moment_debye.magnitude == approx(1.916470)
    assert state7_data.multipole.dipole_moment_debye.x == approx(0.921424)
    assert state7_data.multipole.dipole_moment_debye.y == approx(0.200661)
    assert state7_data.multipole.dipole_moment_debye.z == approx(1.668403)
    assert state7_data.multipole.rms_density_size_ang == approx(0.978422)  # Scalar value
    assert state7_data.multipole.rms_density_size_ang_comps == approx([0.643398, 0.486988, 0.553345])  # Components

    # Exciton analysis of the difference density matrix
    assert state7_data.exciton_difference_dm_analysis is not None  # Total
    assert state7_data.exciton_difference_dm_analysis.hole_center_ang == approx([2.349369, 1.567221, -0.005281])
    assert state7_data.exciton_difference_dm_analysis.electron_center_ang == approx([1.951284, 1.483303, -0.713040])
    assert state7_data.exciton_difference_dm_analysis.electron_hole_separation_ang == approx(0.816355)
    assert state7_data.exciton_difference_dm_analysis.hole_size_ang == approx(0.860981)  # Scalar
    assert state7_data.exciton_difference_dm_analysis.hole_size_ang_comps == approx(
        [0.498346, 0.374083, 0.594139]
    )  # Components
    assert state7_data.exciton_difference_dm_analysis.electron_size_ang == approx(2.078160)  # Scalar
    assert state7_data.exciton_difference_dm_analysis.electron_size_ang_comps == approx(
        [1.498519, 0.937784, 1.092589]
    )  # Components

    # Alpha spin Exciton analysis
    assert state7_data.exciton_difference_dm_analysis_alpha is not None
    assert state7_data.exciton_difference_dm_analysis_alpha.hole_center_ang == approx([2.349369, 1.567221, -0.005281])
    assert state7_data.exciton_difference_dm_analysis_alpha.electron_center_ang == approx(
        [1.951284, 1.483303, -0.713040]
    )
    assert state7_data.exciton_difference_dm_analysis_alpha.electron_hole_separation_ang == approx(0.816355)
    assert state7_data.exciton_difference_dm_analysis_alpha.hole_size_ang == approx(0.860981)  # Scalar
    assert state7_data.exciton_difference_dm_analysis_alpha.hole_size_ang_comps == approx(
        [0.498346, 0.374083, 0.594139]
    )  # Components
    assert state7_data.exciton_difference_dm_analysis_alpha.electron_size_ang == approx(2.078160)  # Scalar
    assert state7_data.exciton_difference_dm_analysis_alpha.electron_size_ang_comps == approx(
        [1.498519, 0.937784, 1.092589]
    )  # Components

    # Beta spin Exciton analysis
    assert state7_data.exciton_difference_dm_analysis_beta is not None
    assert state7_data.exciton_difference_dm_analysis_beta.hole_center_ang == approx([2.349369, 1.567221, -0.005281])
    assert state7_data.exciton_difference_dm_analysis_beta.electron_center_ang == approx(
        [1.951284, 1.483303, -0.713040]
    )
    assert state7_data.exciton_difference_dm_analysis_beta.electron_hole_separation_ang == approx(0.816355)
    assert state7_data.exciton_difference_dm_analysis_beta.hole_size_ang == approx(0.860981)  # Scalar
    assert state7_data.exciton_difference_dm_analysis_beta.hole_size_ang_comps == approx(
        [0.498346, 0.374083, 0.594139]
    )  # Components
    assert state7_data.exciton_difference_dm_analysis_beta.electron_size_ang == approx(2.078160)  # Scalar
    assert state7_data.exciton_difference_dm_analysis_beta.electron_size_ang_comps == approx(
        [1.498519, 0.937784, 1.092589]
    )  # Components

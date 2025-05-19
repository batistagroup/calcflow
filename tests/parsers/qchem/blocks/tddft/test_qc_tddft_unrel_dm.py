import pytest

from calcflow.parsers.qchem.typing import (
    CalculationData,
    ExcitedStateDetailedAnalysis,
)


def get_state_analysis(
    data: CalculationData, state_number: int, multiplicity: str
) -> ExcitedStateDetailedAnalysis | None:
    """Helper to find a specific excited state analysis."""
    if not data.tddft or not data.tddft.excited_state_analyses:
        return None
    for analysis in data.tddft.excited_state_analyses:
        if analysis.state_number == state_number and analysis.multiplicity == multiplicity:
            return analysis
    return None


def test_unrelaxed_dm_singlet_4(parsed_tddft_pc2_data: CalculationData) -> None:
    """Test parsing of Unrelaxed Density Matrices for Singlet 4."""
    analysis = get_state_analysis(parsed_tddft_pc2_data, state_number=4, multiplicity="Singlet")
    assert analysis is not None, "Singlet 4 analysis not found"

    # NOs Data
    assert analysis.no_data is not None
    no_data = analysis.no_data
    assert no_data.frontier_occupations == pytest.approx([0.9982, 1.0016])
    assert no_data.n_electrons == pytest.approx(10.000000)
    assert no_data.n_unpaired == pytest.approx(2.00267)
    assert no_data.n_unpaired_nl == pytest.approx(2.00003)
    assert no_data.pr_no == pytest.approx(2.012158)

    # Mulliken Population Analysis
    assert analysis.mulliken is not None
    mulliken = analysis.mulliken
    assert len(mulliken.populations) == 3
    pop0 = mulliken.populations[0]
    assert pop0.atom_index == 0
    assert pop0.symbol == "H"
    assert pop0.charge_e == pytest.approx(-0.250530)
    assert pop0.hole_charge == pytest.approx(0.072631)
    assert pop0.electron_charge == pytest.approx(-0.553716)
    assert pop0.delta_charge == pytest.approx(-0.481084)

    pop1 = mulliken.populations[1]
    assert pop1.atom_index == 1
    assert pop1.symbol == "O"
    assert pop1.charge_e == pytest.approx(0.505834)
    assert pop1.hole_charge == pytest.approx(0.856237)
    assert pop1.electron_charge == pytest.approx(0.110057)
    assert pop1.delta_charge == pytest.approx(0.966294)

    pop2 = mulliken.populations[2]
    assert pop2.atom_index == 2
    assert pop2.symbol == "H"
    assert pop2.charge_e == pytest.approx(-0.255304)
    assert pop2.hole_charge == pytest.approx(0.072467)
    assert pop2.electron_charge == pytest.approx(-0.557676)
    assert pop2.delta_charge == pytest.approx(-0.485209)

    # Multipole moment analysis
    assert analysis.multipole is not None
    multipole = analysis.multipole
    assert multipole.molecular_charge == pytest.approx(-0.000000)
    assert multipole.n_electrons == pytest.approx(10.000000)
    assert multipole.center_electronic_charge_ang == pytest.approx([2.249186, 1.546547, -0.181386])
    assert multipole.center_nuclear_charge_ang == pytest.approx([2.269759, 1.550894, -0.144757])
    assert multipole.dipole_moment_debye is not None
    dipole = multipole.dipole_moment_debye
    assert dipole.magnitude == pytest.approx(2.028667)
    assert dipole.x == pytest.approx(0.988173)
    assert dipole.y == pytest.approx(0.208793)
    assert dipole.z == pytest.approx(1.759378)
    assert multipole.rms_density_size_ang_comps == pytest.approx([0.674938, 0.505847, 0.572431])

    # Exciton analysis
    assert analysis.exciton_difference_dm_analysis is not None
    exciton = analysis.exciton_difference_dm_analysis
    assert exciton.hole_center_ang == pytest.approx([2.349918, 1.567321, -0.004409])
    assert exciton.electron_center_ang == pytest.approx([1.937410, 1.481598, -0.732052])
    assert exciton.electron_hole_separation_ang == pytest.approx(0.840818)
    assert exciton.hole_size_ang == pytest.approx(0.860625)
    assert exciton.hole_size_ang_comps == pytest.approx([0.495363, 0.374054, 0.596133])
    assert exciton.electron_size_ang == pytest.approx(2.260846)
    assert exciton.electron_size_ang_comps == pytest.approx([1.628597, 1.033055, 1.179785])


def test_unrelaxed_dm_singlet_8(parsed_tddft_pc2_data: CalculationData) -> None:
    """Test parsing of Unrelaxed Density Matrices for Singlet 8."""
    analysis = get_state_analysis(parsed_tddft_pc2_data, state_number=8, multiplicity="Singlet")
    assert analysis is not None, "Singlet 8 analysis not found"

    # NOs Data
    assert analysis.no_data is not None
    no_data = analysis.no_data
    assert no_data.frontier_occupations == pytest.approx([0.8403, 1.1596])
    assert no_data.n_electrons == pytest.approx(10.000000)
    assert no_data.n_unpaired == pytest.approx(2.00221)
    assert no_data.n_unpaired_nl == pytest.approx(2.04660)
    assert no_data.pr_no == pytest.approx(2.754909)

    # Mulliken Population Analysis
    assert analysis.mulliken is not None
    mulliken = analysis.mulliken
    assert len(mulliken.populations) == 3
    pop0 = mulliken.populations[0]
    assert pop0.atom_index == 0
    assert pop0.symbol == "H"
    assert pop0.charge_e == pytest.approx(-0.083945)
    assert pop0.hole_charge == pytest.approx(0.165115)
    assert pop0.electron_charge == pytest.approx(-0.479614)
    assert pop0.delta_charge == pytest.approx(-0.314500)

    pop1 = mulliken.populations[1]
    assert pop1.atom_index == 1
    assert pop1.symbol == "O"
    assert pop1.charge_e == pytest.approx(0.172471)
    assert pop1.hole_charge == pytest.approx(0.670900)
    assert pop1.electron_charge == pytest.approx(-0.037970)
    assert pop1.delta_charge == pytest.approx(0.632931)

    pop2 = mulliken.populations[2]
    assert pop2.atom_index == 2
    assert pop2.symbol == "H"
    assert pop2.charge_e == pytest.approx(-0.088525)
    assert pop2.hole_charge == pytest.approx(0.165089)
    assert pop2.electron_charge == pytest.approx(-0.483520)
    assert pop2.delta_charge == pytest.approx(-0.318431)

    # Multipole moment analysis
    assert analysis.multipole is not None
    multipole = analysis.multipole
    assert multipole.molecular_charge == pytest.approx(-0.000000)
    assert multipole.n_electrons == pytest.approx(10.000000)
    assert multipole.center_electronic_charge_ang == pytest.approx([2.266041, 1.550010, -0.151939])
    assert multipole.center_nuclear_charge_ang == pytest.approx([2.269759, 1.550894, -0.144757])
    assert multipole.dipole_moment_debye is not None
    dipole = multipole.dipole_moment_debye
    assert dipole.magnitude == pytest.approx(0.390744)
    assert dipole.x == pytest.approx(0.178572)
    assert dipole.y == pytest.approx(0.042455)
    assert dipole.z == pytest.approx(0.344950)
    assert multipole.rms_density_size_ang_comps == pytest.approx([0.644320, 0.509829, 0.574419])

    # Exciton analysis
    assert analysis.exciton_difference_dm_analysis is not None
    exciton = analysis.exciton_difference_dm_analysis
    assert exciton.hole_center_ang == pytest.approx([2.235904, 1.544114, -0.203361])
    assert exciton.electron_center_ang == pytest.approx([1.991669, 1.492964, -0.637020])
    assert exciton.electron_hole_separation_ang == pytest.approx(0.500327)
    assert exciton.hole_size_ang == pytest.approx(0.881559)
    assert exciton.hole_size_ang_comps == pytest.approx([0.609250, 0.421281, 0.477999])
    assert exciton.electron_size_ang == pytest.approx(2.215156)
    assert exciton.electron_size_ang_comps == pytest.approx([1.548862, 1.070711, 1.166843])


def test_unrelaxed_dm_uks_excited_state_1(parsed_tddft_uks_pc2_data: CalculationData) -> None:
    """Test parsing of Unrelaxed Density Matrices for UKS Excited State 1."""
    analysis = get_state_analysis(parsed_tddft_uks_pc2_data, state_number=1, multiplicity="Excited State")
    assert analysis is not None, "UKS Excited State 1 analysis not found"

    # NOs Data (should be from spin-traced)
    assert analysis.no_data is not None
    no_data = analysis.no_data
    assert no_data.frontier_occupations == pytest.approx([0.9991, 1.0007])
    assert no_data.n_electrons == pytest.approx(10.000000)
    assert no_data.n_unpaired == pytest.approx(2.00012)
    assert no_data.n_unpaired_nl == pytest.approx(2.00000)
    assert no_data.pr_no == pytest.approx(2.003363)

    # Mulliken Population Analysis
    assert analysis.mulliken is not None
    mulliken = analysis.mulliken
    assert len(mulliken.populations) == 3
    pop0 = mulliken.populations[0]
    assert pop0.atom_index == 0
    assert pop0.symbol == "H"
    assert pop0.charge_e == pytest.approx(-0.272264)
    assert pop0.spin_e == pytest.approx(-0.000000)
    assert pop0.hole_charge == pytest.approx(0.008484 + 0.008484)  # alpha + beta
    assert pop0.electron_charge == pytest.approx(-0.259893 + -0.259893)  # alpha + beta
    # delta_charge = electron_charge - hole_charge
    assert pop0.delta_charge == pytest.approx((-0.259893 + -0.259893) - (0.008484 + 0.008484))

    pop1 = mulliken.populations[1]
    assert pop1.atom_index == 1
    assert pop1.symbol == "O"
    assert pop1.charge_e == pytest.approx(0.541008)
    assert pop1.spin_e == pytest.approx(0.000000)
    assert pop1.hole_charge == pytest.approx(0.483823 + 0.483823)
    assert pop1.electron_charge == pytest.approx(0.016911 + 0.016911)
    assert pop1.delta_charge == pytest.approx((0.016911 + 0.016911) - (0.483823 + 0.483823))

    pop2 = mulliken.populations[2]
    assert pop2.atom_index == 2
    assert pop2.symbol == "H"
    assert pop2.charge_e == pytest.approx(-0.268744)
    assert pop2.spin_e == pytest.approx(-0.000000)
    assert pop2.hole_charge == pytest.approx(0.008503 + 0.008503)
    assert pop2.electron_charge == pytest.approx(-0.257828 + -0.257828)
    assert pop2.delta_charge == pytest.approx((-0.257828 + -0.257828) - (0.008503 + 0.008503))

    # Multipole moment analysis
    assert analysis.multipole is not None
    multipole = analysis.multipole
    assert multipole.molecular_charge == pytest.approx(-0.000000)
    assert multipole.n_electrons == pytest.approx(10.000000)
    assert multipole.center_electronic_charge_ang == pytest.approx([2.256810, 1.548385, -0.166743])
    assert multipole.center_nuclear_charge_ang == pytest.approx([2.269759, 1.550894, -0.144757])
    assert multipole.dipole_moment_debye is not None
    dipole = multipole.dipole_moment_debye
    assert dipole.magnitude == pytest.approx(1.231477)
    assert dipole.x == pytest.approx(0.621980)
    assert dipole.y == pytest.approx(0.120499)
    assert dipole.z == pytest.approx(1.056010)
    assert multipole.rms_density_size_ang_comps == pytest.approx([0.583184, 0.469983, 0.557523])

    # Exciton analysis - Total
    assert analysis.exciton_difference_dm_analysis is not None
    exciton_total = analysis.exciton_difference_dm_analysis
    assert exciton_total.hole_center_ang == pytest.approx([2.311701, 1.559437, -0.071560])
    assert exciton_total.electron_center_ang == pytest.approx([1.975431, 1.492090, -0.652796])
    assert exciton_total.electron_hole_separation_ang == pytest.approx(0.674871)
    assert exciton_total.hole_size_ang == pytest.approx(0.827901)
    assert exciton_total.hole_size_ang_comps == pytest.approx([0.389995, 0.616871, 0.390887])
    assert exciton_total.electron_size_ang == pytest.approx(1.861781)
    assert exciton_total.electron_size_ang_comps == pytest.approx([1.192652, 0.979371, 1.041463])

    # Exciton analysis - Alpha
    assert analysis.exciton_difference_dm_analysis_alpha is not None
    exciton_alpha = analysis.exciton_difference_dm_analysis_alpha
    assert exciton_alpha.hole_center_ang == pytest.approx([2.311702, 1.559437, -0.071560])
    assert exciton_alpha.electron_center_ang == pytest.approx([1.975432, 1.492090, -0.652796])
    assert exciton_alpha.electron_hole_separation_ang == pytest.approx(0.674872)
    assert exciton_alpha.hole_size_ang == pytest.approx(0.827902)
    assert exciton_alpha.hole_size_ang_comps == pytest.approx([0.389995, 0.616872, 0.390887])
    assert exciton_alpha.electron_size_ang == pytest.approx(1.861782)
    assert exciton_alpha.electron_size_ang_comps == pytest.approx([1.192652, 0.979372, 1.041463])

    # Exciton analysis - Beta
    assert analysis.exciton_difference_dm_analysis_beta is not None
    exciton_beta = analysis.exciton_difference_dm_analysis_beta
    assert exciton_beta.hole_center_ang == pytest.approx([2.311703, 1.559437, -0.071560])
    assert exciton_beta.electron_center_ang == pytest.approx([1.975433, 1.492090, -0.652796])
    assert exciton_beta.electron_hole_separation_ang == pytest.approx(0.674873)
    assert exciton_beta.hole_size_ang == pytest.approx(0.827903)
    assert exciton_beta.hole_size_ang_comps == pytest.approx([0.389995, 0.616873, 0.390887])
    assert exciton_beta.electron_size_ang == pytest.approx(1.861783)
    assert exciton_beta.electron_size_ang_comps == pytest.approx([1.192652, 0.979373, 1.041463])

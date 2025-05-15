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
    assert multipole.rms_density_size_ang == pytest.approx([0.674938, 0.505847, 0.572431])

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
    assert multipole.rms_density_size_ang == pytest.approx([0.644320, 0.509829, 0.574419])

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

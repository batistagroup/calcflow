from pytest import approx

from calcflow.parsers.qchem.typing import DipoleMoment, ExcitedStateMultipole, GroundStateReferenceAnalysis


def test_ground_state_reference_analysis_exists(parsed_tddft_pc2_data):
    """Test that GroundStateReferenceAnalysis is parsed."""
    gs_ref = parsed_tddft_pc2_data.gs_reference_analysis
    assert gs_ref is not None
    assert isinstance(gs_ref, GroundStateReferenceAnalysis)


def test_gs_ref_no_data(parsed_tddft_pc2_data):
    """Test NOs data in GroundStateReferenceAnalysis."""
    gs_ref = parsed_tddft_pc2_data.gs_reference_analysis
    assert gs_ref is not None
    no_data = gs_ref.no_data_rks_or_spin_traced
    assert no_data is not None
    # Based on tddft-rks-pc2.out snippet
    # Occupation of frontier NOs: 0.0000   2.0000 -> Not stored in ExcitedStateNOData based on current fields
    # n_electrons=10.0, n_unpaired=-0.0, n_unpaired_nl=0.0
    assert no_data.n_electrons == approx(10.0)
    assert no_data.n_unpaired == approx(-0.0)
    assert no_data.n_unpaired_nl == approx(0.0)

    assert no_data.frontier_occupations is not None
    assert len(no_data.frontier_occupations) == 2
    assert no_data.frontier_occupations[0] == approx(0.0)
    assert no_data.frontier_occupations[1] == approx(2.0)
    assert no_data.pr_no is None


def test_gs_ref_mulliken_analysis(parsed_tddft_pc2_data):
    """Test Mulliken analysis data in GroundStateReferenceAnalysis."""
    gs_ref = parsed_tddft_pc2_data.gs_reference_analysis
    assert gs_ref is not None
    mulliken = gs_ref.mulliken
    assert mulliken is not None
    assert len(mulliken.populations) == 3
    # Atom      Charge (e)
    #  1 H        0.230554
    #  2 O       -0.460460
    #  3 H        0.229906
    pop1 = mulliken.populations[0]
    assert pop1.atom_index == 0
    assert pop1.symbol == "H"
    assert pop1.charge_e == approx(0.230554)
    assert pop1.hole_charge is None
    assert pop1.electron_charge is None
    assert pop1.delta_charge is None

    pop2 = mulliken.populations[1]
    assert pop2.atom_index == 1
    assert pop2.symbol == "O"
    assert pop2.charge_e == approx(-0.460460)

    pop3 = mulliken.populations[2]
    assert pop3.atom_index == 2
    assert pop3.symbol == "H"
    assert pop3.charge_e == approx(0.229906)


def test_gs_ref_multipole_analysis(parsed_tddft_pc2_data):
    """Test Multipole analysis data in GroundStateReferenceAnalysis."""
    gs_ref = parsed_tddft_pc2_data.gs_reference_analysis
    assert gs_ref is not None
    multipole = gs_ref.multipole
    assert multipole is not None
    assert isinstance(multipole, ExcitedStateMultipole)

    assert multipole.molecular_charge == approx(-0.0)
    assert multipole.n_electrons == approx(10.0)

    assert multipole.center_electronic_charge_ang is not None
    assert multipole.center_electronic_charge_ang[0] == approx(2.290492)
    assert multipole.center_electronic_charge_ang[1] == approx(1.555131)
    assert multipole.center_electronic_charge_ang[2] == approx(-0.108525)

    assert multipole.center_nuclear_charge_ang is not None
    assert multipole.center_nuclear_charge_ang[0] == approx(2.269759)
    assert multipole.center_nuclear_charge_ang[1] == approx(1.550894)
    assert multipole.center_nuclear_charge_ang[2] == approx(-0.144757)

    assert multipole.dipole_moment_debye is not None
    dipole = multipole.dipole_moment_debye
    assert isinstance(dipole, DipoleMoment)
    assert dipole.magnitude == approx(2.015379)
    assert dipole.x == approx(-0.995831)
    assert dipole.y == approx(-0.203503)
    assert dipole.z == approx(-1.740304)

    assert multipole.rms_density_size_ang is not None
    assert multipole.rms_density_size_ang[0] == approx(0.451777)
    assert multipole.rms_density_size_ang[1] == approx(0.403206)
    assert multipole.rms_density_size_ang[2] == approx(0.437437)


# --- Tests for Unrestricted (UKS) TDDFT Output ---


def test_gs_ref_uks_no_data(parsed_tddft_uks_pc2_data):
    """Test NOs data in GroundStateReferenceAnalysis for UKS output."""
    gs_ref = parsed_tddft_uks_pc2_data.gs_reference_analysis
    assert gs_ref is not None

    # Test NOs (alpha)
    no_alpha = gs_ref.no_data_alpha
    assert no_alpha is not None
    assert no_alpha.n_electrons == approx(5.0)
    assert no_alpha.frontier_occupations is not None
    assert len(no_alpha.frontier_occupations) == 2
    assert no_alpha.frontier_occupations[0] == approx(0.0)
    assert no_alpha.frontier_occupations[1] == approx(1.0)
    assert no_alpha.n_unpaired is None  # Not expected for alpha/beta specific blocks
    assert no_alpha.pr_no is None

    # Test NOs (beta)
    no_beta = gs_ref.no_data_beta
    assert no_beta is not None
    assert no_beta.n_electrons == approx(5.0)
    assert no_beta.frontier_occupations is not None
    assert len(no_beta.frontier_occupations) == 2
    assert no_beta.frontier_occupations[0] == approx(0.0)
    assert no_beta.frontier_occupations[1] == approx(1.0)
    assert no_beta.n_unpaired is None
    assert no_beta.pr_no is None

    # Test NOs (spin-traced) - this is what no_data_rks_or_spin_traced should hold for UKS
    no_spin_traced = gs_ref.no_data_rks_or_spin_traced
    assert no_spin_traced is not None
    assert no_spin_traced.n_electrons == approx(10.0)
    assert no_spin_traced.n_unpaired == approx(-0.0)  # QChem often prints -0.0
    assert no_spin_traced.n_unpaired_nl == approx(0.0)
    assert no_spin_traced.frontier_occupations is not None
    assert len(no_spin_traced.frontier_occupations) == 2
    assert no_spin_traced.frontier_occupations[0] == approx(0.0)
    assert no_spin_traced.frontier_occupations[1] == approx(2.0)
    assert no_spin_traced.pr_no is None  # Not present in the example output for spin-traced


def test_gs_ref_uks_mulliken_analysis(parsed_tddft_uks_pc2_data):
    """Test Mulliken analysis data in GroundStateReferenceAnalysis for UKS output."""
    gs_ref = parsed_tddft_uks_pc2_data.gs_reference_analysis
    assert gs_ref is not None
    mulliken = gs_ref.mulliken
    assert mulliken is not None
    assert len(mulliken.populations) == 3

    # Atom      Charge (e)        Spin (e)
    #  1 H        0.230554        0.000000
    #  2 O       -0.460460        0.000000
    #  3 H        0.229906        0.000000

    pop1 = mulliken.populations[0]
    assert pop1.atom_index == 0
    assert pop1.symbol == "H"
    assert pop1.charge_e == approx(0.230554)
    assert pop1.spin_e == approx(0.000000)

    pop2 = mulliken.populations[1]
    assert pop2.atom_index == 1
    assert pop2.symbol == "O"
    assert pop2.charge_e == approx(-0.460460)
    assert pop2.spin_e == approx(0.000000)

    pop3 = mulliken.populations[2]
    assert pop3.atom_index == 2
    assert pop3.symbol == "H"
    assert pop3.charge_e == approx(0.229906)
    assert pop3.spin_e == approx(0.000000)


def test_gs_ref_uks_multipole_analysis(parsed_tddft_uks_pc2_data):
    """Test Multipole analysis data in GroundStateReferenceAnalysis for UKS output."""
    gs_ref = parsed_tddft_uks_pc2_data.gs_reference_analysis
    assert gs_ref is not None
    multipole = gs_ref.multipole
    assert multipole is not None
    assert isinstance(multipole, ExcitedStateMultipole)

    # Values from tddft-uks-pc2.out
    assert multipole.molecular_charge == approx(-0.0)
    assert multipole.n_electrons == approx(10.0)

    assert multipole.center_electronic_charge_ang is not None
    assert multipole.center_electronic_charge_ang[0] == approx(2.290492)
    assert multipole.center_electronic_charge_ang[1] == approx(1.555131)
    assert multipole.center_electronic_charge_ang[2] == approx(-0.108525)

    assert multipole.center_nuclear_charge_ang is not None
    assert multipole.center_nuclear_charge_ang[0] == approx(2.269759)
    assert multipole.center_nuclear_charge_ang[1] == approx(1.550894)
    assert multipole.center_nuclear_charge_ang[2] == approx(-0.144757)

    assert multipole.dipole_moment_debye is not None
    dipole = multipole.dipole_moment_debye
    assert isinstance(dipole, DipoleMoment)
    assert dipole.magnitude == approx(2.015379)
    assert dipole.x == approx(-0.995831)
    assert dipole.y == approx(-0.203503)
    assert dipole.z == approx(-1.740304)

    assert multipole.rms_density_size_ang is not None
    assert multipole.rms_density_size_ang[0] == approx(0.451777)
    assert multipole.rms_density_size_ang[1] == approx(0.403206)
    assert multipole.rms_density_size_ang[2] == approx(0.437437)

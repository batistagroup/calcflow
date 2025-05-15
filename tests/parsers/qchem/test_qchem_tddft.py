from pytest import approx

from calcflow.parsers.qchem.typing import (
    Atom,
    CalculationData,
    DipoleMoment,
    HexadecapoleMoment,
    OctopoleMoment,
    OrbitalTransition,
    QuadrupoleMoment,
    ScfData,
    TddftData,
)


def test_parse_qchem_tddft_output_h2o(parsed_tddft_pc2_data: CalculationData) -> None:
    """
    Integration test for parsing the H2O TDDFT output file (tddft-rks-pc2.out).

    Verifies that the main fields extracted by parse_qchem_sp_output match
    the expected values.
    """
    data = parsed_tddft_pc2_data

    # --- Top-Level Checks ---
    assert data.termination_status == "NORMAL"
    # The final_energy should be the ground state SCF energy
    assert data.final_energy == approx(-76.44125314)
    assert data.nuclear_repulsion_eh == approx(8.93764808)

    # --- Metadata Checks ---
    assert data.metadata.qchem_version == "6.2"
    assert data.metadata.host == "login12"
    # assert data.metadata.run_date == "Sun May  4 12:07:16 2025" # Date is dynamic
    assert (
        data.metadata.calculation_method == "wb97x-d3"
    )  # From input, not explicitly in $rem block in output but parsed from there.
    assert data.metadata.basis_set == "pc-2"

    # --- Input Options ($rem section) ---
    # The parser stores these in CalculationData.metadata directly if they are standard
    # or they might be in a more generic input_options dict if that's implemented.
    # For now, checking metadata which derives from $rem.
    # If a separate data.input_options field exists and is populated, it should be checked.
    # Based on the current output structure, these are in metadata or implied.

    # --- Geometry Checks ---
    # Input Geometry ($molecule)
    assert data.input_geometry is not None
    assert len(data.input_geometry) == 3
    assert data.input_geometry[0] == Atom(symbol="H", x=1.36499, y=1.69385, z=-0.19748)
    assert data.input_geometry[1] == Atom(symbol="O", x=2.32877, y=1.56294, z=-0.04168)
    assert data.input_geometry[2] == Atom(symbol="H", x=2.70244, y=1.31157, z=-0.91665)

    # Standard Orientation Geometry
    assert data.standard_orientation_geometry is not None
    assert len(data.standard_orientation_geometry) == 3
    assert data.standard_orientation_geometry[0] == Atom(symbol="H", x=1.3649900000, y=1.6938500000, z=-0.1974800000)
    assert data.standard_orientation_geometry[1] == Atom(symbol="O", x=2.3287700000, y=1.5629400000, z=-0.0416800000)
    assert data.standard_orientation_geometry[2] == Atom(symbol="H", x=2.7024400000, y=1.3115700000, z=-0.9166500000)

    # --- SCF Checks ---
    assert data.scf is not None
    scf: ScfData = data.scf  # type: ignore
    assert scf.converged is True
    assert scf.energy == approx(-76.44125314)  # This is the ground state SCF energy
    assert scf.n_iterations == 10
    assert len(scf.iteration_history) == 10

    assert scf.iteration_history[0].iteration == 1
    assert scf.iteration_history[0].energy == approx(-76.3054390399)
    assert scf.iteration_history[0].diis_error == approx(5.31e-02)

    assert scf.iteration_history[-1].iteration == 10
    assert scf.iteration_history[-1].energy == approx(-76.4412531352)
    assert scf.iteration_history[-1].diis_error == approx(6.73e-09)

    # --- Orbital Checks ---
    assert data.orbitals is not None
    assert data.orbitals.beta_orbitals is None  # Restricted calculation
    assert data.orbitals.alpha_orbitals is not None
    alpha_orbitals = data.orbitals.alpha_orbitals
    assert len(alpha_orbitals) == 58  # 5 occupied + 53 virtual for pc-2 (from output file)

    # Check specific occupied orbitals
    assert (
        alpha_orbitals[0].index == 0
    )  # QChem output is 1-indexed, parser should convert to 0-indexed if that's the convention
    assert alpha_orbitals[0].energy == approx(-19.2346)
    assert alpha_orbitals[4].index == 4  # HOMO
    assert alpha_orbitals[4].energy == approx(-0.4147)

    # Check specific virtual orbitals
    assert alpha_orbitals[5].index == 5  # LUMO
    assert alpha_orbitals[5].energy == approx(0.0878)
    assert alpha_orbitals[6].index == 6  # LUMO+1
    assert alpha_orbitals[6].energy == approx(0.1389)

    # If state_label is not part of AtomicCharges, we might need another way to distinguish
    # For now, assuming the first Mulliken set IF state_analysis is off, or specific label if on.
    # The provided output has "Mulliken Population Analysis (State DM)" under "Ground State (Reference)"
    # This suggests the parser should ideally label it or it's the first one.
    # Let's assume the parser correctly identifies and stores the ground state Mulliken charges.
    # If not, this part of the test might need adjustment based on parser behavior.
    # From the output, the first Mulliken block is for the ground state.

    # --- Multipole Checks (Ground State) ---
    # These are typically for the ground state unless specified.
    assert data.multipole is not None
    assert data.multipole.charge_esu == approx(-0.0000)

    assert data.multipole.dipole is not None
    dipole: DipoleMoment = data.multipole.dipole  # type: ignore
    assert dipole.x == approx(-0.9958)  # From "Cartesian Multipole Moments"
    assert dipole.y == approx(-0.2035)
    assert dipole.z == approx(-1.7403)
    assert dipole.magnitude == approx(2.0154)

    assert data.multipole.quadrupole is not None
    quad: QuadrupoleMoment = data.multipole.quadrupole  # type: ignore
    assert quad.xx == approx(-9.3797)
    assert quad.xy == approx(-2.6489)
    assert quad.yy == approx(-8.0621)
    assert quad.xz == approx(-4.5863)
    assert quad.yz == approx(-2.1771)
    assert quad.zz == approx(-5.4667)

    assert data.multipole.octopole is not None
    octo: OctopoleMoment = data.multipole.octopole  # type: ignore
    assert octo.xxx == approx(-50.1854)
    assert octo.xxy == approx(-18.4076)
    # ... (rest of octopole moments - can be added if full coverage desired)
    assert octo.zzz == approx(0.8499)

    assert data.multipole.hexadecapole is not None
    hexa: HexadecapoleMoment = data.multipole.hexadecapole  # type: ignore
    assert hexa.xxxx == approx(-218.5681)
    assert hexa.xxxy == approx(-89.6390)
    # ... (rest of hexadecapole moments)
    assert hexa.zzzz == approx(-6.7076)

    # --- TDDFT Data Checks ---
    assert data.tddft_data is not None
    tddft_results: TddftData = data.tddft_data

    # --- TDA (CIS) Excited States ---
    assert tddft_results.tda_excited_states is not None
    assert len(tddft_results.tda_excited_states) == 10

    # Check State 1 (TDA)
    tda_es1 = tddft_results.tda_excited_states[0]
    assert tda_es1.state_number == 1
    assert tda_es1.excitation_energy_ev == approx(7.5954)
    assert tda_es1.total_energy_au == approx(-76.16212816)
    assert tda_es1.multiplicity == "Singlet"
    assert tda_es1.trans_moment_x == approx(-0.0869)
    assert tda_es1.trans_moment_y == approx(-0.5093)
    assert tda_es1.trans_moment_z == approx(0.1092)
    assert tda_es1.oscillator_strength == approx(0.0518861703)
    assert len(tda_es1.transitions) >= 1
    # fmt:off
    assert tda_es1.transitions[0] == OrbitalTransition(from_orbital_type="D", from_orbital_index=5, to_orbital_type="V", to_orbital_index=1, amplitude=approx(0.9957), is_alpha_spin=None) # Spin not specified for RKS
    # fmt:on

    # Check State 3 (TDA)
    tda_es3 = tddft_results.tda_excited_states[2]
    assert tda_es3.state_number == 3
    assert tda_es3.excitation_energy_ev == approx(9.8215)
    assert tda_es3.oscillator_strength == approx(0.1112256100)
    assert len(tda_es3.transitions) >= 1
    assert tda_es3.transitions[0] == OrbitalTransition(
        from_orbital_type="D",
        from_orbital_index=4,
        to_orbital_type="V",
        to_orbital_index=1,
        amplitude=approx(0.9914),
        is_alpha_spin=None,
    )

    # Check State 10 (TDA) - multiple transitions
    tda_es10 = tddft_results.tda_excited_states[9]
    assert tda_es10.state_number == 10
    assert tda_es10.excitation_energy_ev == approx(16.8451)
    assert tda_es10.oscillator_strength == approx(0.0194126086)
    assert len(tda_es10.transitions) == 4
    # fmt:off
    assert OrbitalTransition(from_orbital_type="D", from_orbital_index=3, to_orbital_type="V", to_orbital_index=2, amplitude=approx(-0.2130), is_alpha_spin=None) in tda_es10.transitions
    assert OrbitalTransition(from_orbital_type="D", from_orbital_index=4, to_orbital_type="V", to_orbital_index=4, amplitude=approx(0.6069), is_alpha_spin=None) in tda_es10.transitions
    assert OrbitalTransition(from_orbital_type="D", from_orbital_index=4, to_orbital_type="V", to_orbital_index=5, amplitude=approx(-0.4794), is_alpha_spin=None) in tda_es10.transitions
    assert OrbitalTransition(from_orbital_type="D", from_orbital_index=5, to_orbital_type="V", to_orbital_index=6, amplitude=approx(-0.5917), is_alpha_spin=None) in tda_es10.transitions
    # fmt:on

    # --- Full TDDFT (RPA) Excited States ---
    assert tddft_results.tddft_excited_states is not None
    assert len(tddft_results.tddft_excited_states) == 10

    # Check State 1 (TDDFT)
    tddft_es1 = tddft_results.tddft_excited_states[0]
    assert tddft_es1.state_number == 1
    assert tddft_es1.excitation_energy_ev == approx(7.5725)
    assert tddft_es1.total_energy_au == approx(-76.16297039)
    assert tddft_es1.multiplicity == "Singlet"
    assert tddft_es1.trans_moment_x == approx(-0.0870)
    assert tddft_es1.trans_moment_y == approx(-0.5098)
    assert tddft_es1.trans_moment_z == approx(0.1093)
    assert tddft_es1.oscillator_strength == approx(0.0518397380)
    assert len(tddft_es1.transitions) >= 1  # Output shows X: and Y: parts, parser needs to handle this
    # The parser currently stores all transitions under 'transitions'. QChem output has X: D(...) and Y: D(...)
    # Assuming parser collects all of them, or primarily X. For TDDFT, output format is "X: D( 5) --> V( 1) amplitude = 0.9957"
    # The OrbitalTransition type does not distinguish X/Y, so we expect a single list.
    # fmt:off
    assert tddft_es1.transitions[0] == OrbitalTransition(from_orbital_type="D", from_orbital_index=5, to_orbital_type="V", to_orbital_index=1, amplitude=approx(0.9957), is_alpha_spin=None)
    # fmt:on

    # Check State 3 (TDDFT)
    tddft_es3 = tddft_results.tddft_excited_states[2]
    assert tddft_es3.state_number == 3
    assert tddft_es3.excitation_energy_ev == approx(9.7830)
    assert tddft_es3.oscillator_strength == approx(0.1055651660)
    assert len(tddft_es3.transitions) >= 1
    # fmt:off
    assert tddft_es3.transitions[0] == OrbitalTransition(from_orbital_type="D", from_orbital_index=4, to_orbital_type="V", to_orbital_index=1, amplitude=approx(0.9934), is_alpha_spin=None)
    # fmt:on

    # Check State 10 (TDDFT) - multiple transitions
    tddft_es10 = tddft_results.tddft_excited_states[9]
    assert tddft_es10.state_number == 10
    assert tddft_es10.excitation_energy_ev == approx(16.7974)
    assert tddft_es10.oscillator_strength == approx(0.0160088859)
    assert len(tddft_es10.transitions) == 4
    # fmt:off
    assert OrbitalTransition(from_orbital_type="D", from_orbital_index=3, to_orbital_type="V", to_orbital_index=2, amplitude=approx(-0.2248), is_alpha_spin=None) in tddft_es10.transitions
    assert OrbitalTransition(from_orbital_type="D", from_orbital_index=4, to_orbital_type="V", to_orbital_index=4, amplitude=approx(0.5374), is_alpha_spin=None) in tddft_es10.transitions
    assert OrbitalTransition(from_orbital_type="D", from_orbital_index=4, to_orbital_type="V", to_orbital_index=5, amplitude=approx(-0.4757), is_alpha_spin=None) in tddft_es10.transitions
    assert OrbitalTransition(from_orbital_type="D", from_orbital_index=5, to_orbital_type="V", to_orbital_index=6, amplitude=approx(-0.6556), is_alpha_spin=None) in tddft_es10.transitions
    # fmt:on

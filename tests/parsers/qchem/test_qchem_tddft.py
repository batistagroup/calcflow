from pytest import approx

from calcflow.parsers.qchem.typing import (
    Atom,
    CalculationData,
    DipoleMomentData,
    HexadecapoleMoments,
    OctopoleMoments,
    OrbitalTransition,
    QuadrupoleMoments,
    ScfData,
    # TDDFT specific imports
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
    # The final_energy_eh should be the ground state SCF energy
    assert data.final_energy_eh == approx(-76.44125314)
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
    assert scf.energy_eh == approx(-76.44125314)  # This is the ground state SCF energy
    assert scf.n_iterations == 10
    assert len(scf.iteration_history) == 10

    assert scf.iteration_history[0].iteration == 1
    assert scf.iteration_history[0].energy_eh == approx(-76.3054390399)
    assert scf.iteration_history[0].diis_error == approx(5.31e-02)

    assert scf.iteration_history[-1].iteration == 10
    assert scf.iteration_history[-1].energy_eh == approx(-76.4412531352)
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
    assert alpha_orbitals[0].energy_eh == approx(-19.2346)
    assert alpha_orbitals[4].index == 4  # HOMO
    assert alpha_orbitals[4].energy_eh == approx(-0.4147)

    # Check specific virtual orbitals
    assert alpha_orbitals[5].index == 5  # LUMO
    assert alpha_orbitals[5].energy_eh == approx(0.0878)
    assert alpha_orbitals[6].index == 6  # LUMO+1
    assert alpha_orbitals[6].energy_eh == approx(0.1389)

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
    dipole: DipoleMomentData = data.multipole.dipole  # type: ignore
    assert dipole.x_debye == approx(-0.9958)  # From "Cartesian Multipole Moments"
    assert dipole.y_debye == approx(-0.2035)
    assert dipole.z_debye == approx(-1.7403)
    assert dipole.total_debye == approx(2.0154)

    assert data.multipole.quadrupole is not None
    quad: QuadrupoleMoments = data.multipole.quadrupole  # type: ignore
    assert quad.xx == approx(-9.3797)
    assert quad.xy == approx(-2.6489)
    assert quad.yy == approx(-8.0621)
    assert quad.xz == approx(-4.5863)
    assert quad.yz == approx(-2.1771)
    assert quad.zz == approx(-5.4667)

    assert data.multipole.octopole is not None
    octo: OctopoleMoments = data.multipole.octopole  # type: ignore
    assert octo.xxx == approx(-50.1854)
    assert octo.xxy == approx(-18.4076)
    # ... (rest of octopole moments - can be added if full coverage desired)
    assert octo.zzz == approx(0.8499)

    assert data.multipole.hexadecapole is not None
    hexa: HexadecapoleMoments = data.multipole.hexadecapole  # type: ignore
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
    assert tda_es1.transitions[0] == OrbitalTransition(
        from_orbital_type="D",
        from_orbital_index=5,
        to_orbital_type="V",
        to_orbital_index=1,
        amplitude=approx(0.9957),
        is_alpha_spin=None,
    )  # Spin not specified for RKS

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
    assert (
        OrbitalTransition(
            from_orbital_type="D",
            from_orbital_index=3,
            to_orbital_type="V",
            to_orbital_index=2,
            amplitude=approx(-0.2130),
            is_alpha_spin=None,
        )
        in tda_es10.transitions
    )
    assert (
        OrbitalTransition(
            from_orbital_type="D",
            from_orbital_index=4,
            to_orbital_type="V",
            to_orbital_index=4,
            amplitude=approx(0.6069),
            is_alpha_spin=None,
        )
        in tda_es10.transitions
    )
    assert (
        OrbitalTransition(
            from_orbital_type="D",
            from_orbital_index=4,
            to_orbital_type="V",
            to_orbital_index=5,
            amplitude=approx(-0.4794),
            is_alpha_spin=None,
        )
        in tda_es10.transitions
    )
    assert (
        OrbitalTransition(
            from_orbital_type="D",
            from_orbital_index=5,
            to_orbital_type="V",
            to_orbital_index=6,
            amplitude=approx(-0.5917),
            is_alpha_spin=None,
        )
        in tda_es10.transitions
    )

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
    assert tddft_es1.transitions[0] == OrbitalTransition(
        from_orbital_type="D",
        from_orbital_index=5,
        to_orbital_type="V",
        to_orbital_index=1,
        amplitude=approx(0.9957),
        is_alpha_spin=None,
    )

    # Check State 3 (TDDFT)
    tddft_es3 = tddft_results.tddft_excited_states[2]
    assert tddft_es3.state_number == 3
    assert tddft_es3.excitation_energy_ev == approx(9.7830)
    assert tddft_es3.oscillator_strength == approx(0.1055651660)
    assert len(tddft_es3.transitions) >= 1
    assert tddft_es3.transitions[0] == OrbitalTransition(
        from_orbital_type="D",
        from_orbital_index=4,
        to_orbital_type="V",
        to_orbital_index=1,
        amplitude=approx(0.9934),
        is_alpha_spin=None,
    )

    # Check State 10 (TDDFT) - multiple transitions
    tddft_es10 = tddft_results.tddft_excited_states[9]
    assert tddft_es10.state_number == 10
    assert tddft_es10.excitation_energy_ev == approx(16.7974)
    assert tddft_es10.oscillator_strength == approx(0.0160088859)
    assert len(tddft_es10.transitions) == 4
    assert (
        OrbitalTransition(
            from_orbital_type="D",
            from_orbital_index=3,
            to_orbital_type="V",
            to_orbital_index=2,
            amplitude=approx(-0.2248),
            is_alpha_spin=None,
        )
        in tddft_es10.transitions
    )
    assert (
        OrbitalTransition(
            from_orbital_type="D",
            from_orbital_index=4,
            to_orbital_type="V",
            to_orbital_index=4,
            amplitude=approx(0.5374),
            is_alpha_spin=None,
        )
        in tddft_es10.transitions
    )
    assert (
        OrbitalTransition(
            from_orbital_type="D",
            from_orbital_index=4,
            to_orbital_type="V",
            to_orbital_index=5,
            amplitude=approx(-0.4757),
            is_alpha_spin=None,
        )
        in tddft_es10.transitions
    )
    assert (
        OrbitalTransition(
            from_orbital_type="D",
            from_orbital_index=5,
            to_orbital_type="V",
            to_orbital_index=6,
            amplitude=approx(-0.6556),
            is_alpha_spin=None,
        )
        in tddft_es10.transitions
    )

    # # --- Detailed Excited State Analysis (for TDDFT states) ---
    # assert tddft_results.excited_state_analyses is not None
    # assert len(tddft_results.excited_state_analyses) == 10 # Should match number of TDDFT states if STATE_ANALYSIS is on

    # # Check Detailed Analysis for State 1 (TDDFT)
    # analysis_s1 = tddft_results.excited_state_analyses[0]
    # assert analysis_s1.state_number == 1
    # assert analysis_s1.multiplicity == "Singlet"

    # # NO Data for State 1
    # assert analysis_s1.no_data is not None
    # no_s1 = analysis_s1.no_data
    # assert no_s1.frontier_occupations == approx([0.9992, 1.0006])
    # assert no_s1.num_electrons == approx(10.000000)
    # assert no_s1.num_unpaired_electrons_nu == approx(1.99930)
    # assert no_s1.num_unpaired_electrons_nunl == approx(2.00000)
    # assert no_s1.participation_ratio_pr_no == approx(2.001426)

    # # Mulliken Analysis for State 1
    # assert analysis_s1.mulliken_analysis is not None
    # mulliken_s1 = analysis_s1.mulliken_analysis
    # assert len(mulliken_s1.populations) == 3
    # pop_s1_h1 = mulliken_s1.populations[0]
    # assert pop_s1_h1.atom_index == 0 # Assuming parser maps to 0-indexed
    # assert pop_s1_h1.symbol == "H"
    # assert pop_s1_h1.charge_e == approx(-0.283169)
    # assert pop_s1_h1.hole_charge == approx(0.016896)
    # assert pop_s1_h1.electron_charge == approx(-0.530620)
    # assert pop_s1_h1.delta_charge == approx(-0.513723)
    # # ... check other atoms if needed (O, H2)
    # pop_s1_o = mulliken_s1.populations[1]
    # assert pop_s1_o.symbol == "O"
    # assert pop_s1_o.charge_e == approx(0.562822)
    # pop_s1_h2 = mulliken_s1.populations[2]
    # assert pop_s1_h2.symbol == "H"
    # assert pop_s1_h2.charge_e == approx(-0.279653)

    # # Multipole Analysis for State 1
    # assert analysis_s1.multipole_analysis is not None
    # multipole_s1 = analysis_s1.multipole_analysis
    # assert multipole_s1.molecular_charge == approx(-0.000000)
    # assert multipole_s1.num_electrons == approx(10.000000)
    # assert multipole_s1.center_electronic_charge_ang == approx([2.254976,   1.548012,  -0.169953])
    # assert multipole_s1.center_nuclear_charge_ang == approx([2.269759,   1.550894,  -0.144757]) # This is from GS, should be consistent
    # assert multipole_s1.dipole_moment_debye is not None
    # dipole_s1: DipoleMomentData = multipole_s1.dipole_moment_debye # type: ignore
    # assert dipole_s1.x_debye == approx(0.710066)
    # assert dipole_s1.y_debye == approx(0.138452)
    # assert dipole_s1.z_debye == approx(1.210193)
    # assert dipole_s1.total_debye == approx(1.409939)
    # assert multipole_s1.rms_density_size_ang == approx([0.591568,   0.481035,   0.567111]) # RMS size of the density [Ang]: -> Cartesian components

    # # Exciton Difference DM Analysis for State 1
    # assert analysis_s1.exciton_difference_dm_analysis is not None
    # exciton_s1 = analysis_s1.exciton_difference_dm_analysis
    # assert exciton_s1.hole_center_ang == approx([2.311785,   1.559453,  -0.071430])
    # assert exciton_s1.electron_center_ang == approx([1.957004,   1.488335,  -0.685054])
    # assert exciton_s1.electron_hole_separation_ang == approx(0.712363)
    # assert exciton_s1.hole_size_ang == approx([0.389955,   0.616973,   0.390864]) # Hole size [Ang]: -> Cartesian components
    # assert exciton_s1.electron_size_ang == approx([1.229170,   1.031590,   1.077361]) # Electron size [Ang]: -> Cartesian components

    # # --- Transition Density Matrix Analysis (for TDDFT states) ---
    # assert tddft_results.transition_density_matrix_analyses is not None
    # assert len(tddft_results.transition_density_matrix_analyses) == 10

    # # Check Transition DM Analysis for State 1 (TDDFT)
    # trans_dm_s1 = tddft_results.transition_density_matrix_analyses[0]
    # assert trans_dm_s1.state_number == 1
    # assert trans_dm_s1.multiplicity == "Singlet"

    # # Mulliken for Transition DM State 1
    # assert trans_dm_s1.mulliken_analysis is not None
    # mulliken_trans_s1 = trans_dm_s1.mulliken_analysis
    # assert len(mulliken_trans_s1.populations) == 3
    # pop_trans_s1_h1 = mulliken_trans_s1.populations[0]
    # assert pop_trans_s1_h1.atom_index == 0
    # assert pop_trans_s1_h1.symbol == "H"
    # assert pop_trans_s1_h1.transition_charge_e == approx(-0.000020)
    # assert pop_trans_s1_h1.hole_charge == approx(0.016997)
    # assert pop_trans_s1_h1.electron_charge == approx(-0.530519)
    # assert pop_trans_s1_h1.delta_charge == approx(-0.513522)
    # assert mulliken_trans_s1.sum_abs_trans_charges_qta == approx(0.000053)
    # assert mulliken_trans_s1.sum_sq_trans_charges_qt2 == approx(0.000000)

    # # CT Numbers for Transition DM State 1
    # assert trans_dm_s1.ct_numbers is not None
    # ct_s1 = trans_dm_s1.ct_numbers
    # assert ct_s1.omega == approx(1.0011)
    # assert ct_s1.two_alpha_beta_overlap == approx(1.0011)
    # assert ct_s1.loc == approx(0.0010)
    # assert ct_s1.loc_a == approx(0.1492)
    # assert ct_s1.phe_overlap == approx(-0.0230)

    # # Exciton Analysis for Transition DM State 1
    # assert trans_dm_s1.exciton_analysis is not None
    # exciton_trans_s1 = trans_dm_s1.exciton_analysis
    # assert exciton_trans_s1.total_transition_dipole_moment == approx(1.343588) # Trans. dipole moment [D]
    # assert exciton_trans_s1.transition_dipole_moment_components == approx([0.221046,   1.295818,  -0.277890])
    # assert exciton_trans_s1.hole_position_ang == approx([2.311727,   1.559441,  -0.071529])
    # assert exciton_trans_s1.electron_position_ang == approx([1.957061,   1.488347,  -0.684955])
    # assert exciton_trans_s1.hole_electron_distance_ang == approx(0.712134)
    # assert exciton_trans_s1.hole_size_ang_components == approx([0.390126,   0.616966,   0.390990]) # Hole size [Ang]: Cartesian components
    # assert exciton_trans_s1.electron_size_ang_components == approx([1.229132,   1.031595,   1.077371]) # Electron size [Ang]: Cartesian components
    # assert exciton_trans_s1.rms_electron_hole_separation_ang == approx(2.219123)
    # assert exciton_trans_s1.correlation_coefficient == approx(0.001259)

    # # --- NTO Analysis (SA-NTO Decomposition) ---
    # assert tddft_results.nto_state_analyses is not None
    # assert len(tddft_results.nto_state_analyses) == 10

    # # Check NTO for State 1
    # nto_s1 = tddft_results.nto_state_analyses[0]
    # assert nto_s1.state_number == 1 # Parser needs to add this or identify by order
    # assert nto_s1.multiplicity == "Singlet" # Parser needs to add this
    # assert len(nto_s1.contributions) == 1
    # contrib1_s1 = nto_s1.contributions[0]
    # assert contrib1_s1.hole_nto_type == "H-" # Note: QChem output has H- 0, L+ 0 etc.
    # assert contrib1_s1.hole_nto_index == 0
    # assert contrib1_s1.electron_nto_type == "L+"
    # assert contrib1_s1.electron_nto_index == 0
    # assert contrib1_s1.coefficient == approx(-0.7067)
    # assert contrib1_s1.weight_percent == approx(99.9)
    # assert nto_s1.omega_percent == approx(100.1)

    # # Check NTO for State 7 (multiple contributions)
    # nto_s7 = tddft_results.nto_state_analyses[6]
    # assert nto_s7.state_number == 7
    # assert nto_s7.multiplicity == "Singlet"
    # assert len(nto_s7.contributions) == 2
    # assert NTOContribution(hole_nto_type="H-", hole_nto_index=0, electron_nto_type="L+", electron_nto_index=2, coefficient=approx(0.5097), weight_percent=approx(52.0)) in nto_s7.contributions
    # assert NTOContribution(hole_nto_type="H-", hole_nto_index=0, electron_nto_type="L+", electron_nto_index=4, coefficient=approx(0.4872), weight_percent=approx(47.5)) in nto_s7.contributions
    # assert nto_s7.omega_percent == approx(100.2)

    # # Check NTO for State 10 (multiple contributions)
    # nto_s10 = tddft_results.nto_state_analyses[9]
    # assert nto_s10.state_number == 10
    # assert nto_s10.multiplicity == "Singlet"
    # assert len(nto_s10.contributions) == 3
    # assert NTOContribution(hole_nto_type="H-", hole_nto_index=1, electron_nto_type="L+", electron_nto_index=2, coefficient=approx(-0.5072), weight_percent=approx(51.5)) in nto_s10.contributions
    # assert NTOContribution(hole_nto_type="H-", hole_nto_index=0, electron_nto_type="L+", electron_nto_index=5, coefficient=approx(0.4636), weight_percent=approx(43.0)) in nto_s10.contributions
    # assert NTOContribution(hole_nto_type="H-", hole_nto_index=2, electron_nto_type="L+", electron_nto_index=1, coefficient=approx(-0.1545), weight_percent=approx(4.8)) in nto_s10.contributions
    # assert nto_s10.omega_percent == approx(100.2)

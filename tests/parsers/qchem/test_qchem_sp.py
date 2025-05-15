import logging

from pytest import LogCaptureFixture, approx

from calcflow.parsers import qchem
from calcflow.parsers.qchem.typing import (
    Atom,
    CalculationData,
    DipoleMomentData,
    HexadecapoleMoments,
    OctopoleMoments,
    QuadrupoleMoments,
    ScfData,
)

# --- Success Case Test ---


def test_parse_qchem_sp_output_h2o(parsed_sp_sto_data: CalculationData) -> None:
    """
    Integration test for parsing the standard H2O SP output file.

    Verifies that the main fields extracted by parse_qchem_sp_output match
    the expected values from the data/calculations/examples/qchem/h2o/sp.out file.
    """
    data = parsed_sp_sto_data

    # --- Top-Level Checks ---
    assert data.termination_status == "NORMAL"
    assert data.final_energy_eh == approx(-75.31188446)
    assert data.nuclear_repulsion_eh == approx(8.93764808)

    # --- Metadata Checks ---
    assert data.metadata.qchem_version == "6.2"
    assert data.metadata.host == "login30"
    # assert data.metadata.run_date == "Sun May  4 14:52:42 2025" # Date is dynamic in example
    assert data.metadata.calculation_method == "wb97x-d3"
    assert data.metadata.basis_set == "sto-3g"

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
    scf: ScfData = data.scf
    assert scf.converged is True
    assert scf.energy_eh == approx(-75.31188446)
    assert scf.n_iterations == 7
    assert len(scf.iteration_history) == 7
    # assert scf.iteration_history[0] == ScfIteration(iteration=1, energy_eh=-75.0734525425, diis_error=approx(3.82e-01))
    assert scf.iteration_history[0].iteration == 1
    assert scf.iteration_history[0].energy_eh == approx(-75.0734525425)
    assert scf.iteration_history[0].diis_error == approx(3.82e-01)
    # assert scf.iteration_history[-1] == ScfIteration(iteration=7, energy_eh=-75.3118844639, diis_error=approx(5.35e-08))
    assert scf.iteration_history[-1].iteration == 7
    assert scf.iteration_history[-1].energy_eh == approx(-75.3118844639)
    assert scf.iteration_history[-1].diis_error == approx(5.35e-08)

    # --- Orbital Checks ---
    assert data.orbitals is not None
    assert data.orbitals.beta_orbitals is None  # Restricted calculation
    assert data.orbitals.alpha_orbitals is not None
    alpha_orbitals = data.orbitals.alpha_orbitals
    assert len(alpha_orbitals) == 7  # 5 occupied + 2 virtual

    # Check specific occupied orbitals
    # assert alpha_orbitals[0] == Orbital(index=0, energy_eh=approx(-18.9359), occ=None)  # HOMO-4
    assert alpha_orbitals[0].index == 0
    assert alpha_orbitals[0].energy == approx(-18.9359)
    assert alpha_orbitals[0].occ is None
    # assert alpha_orbitals[4] == Orbital(index=4, energy_eh=approx(-0.2436), occ=None)  # HOMO
    assert alpha_orbitals[4].index == 4
    assert alpha_orbitals[4].energy == approx(-0.2436)
    assert alpha_orbitals[4].occ is None

    # Check specific virtual orbitals
    # assert alpha_orbitals[5] == Orbital(index=5, energy_eh=approx(0.4341), occ=None)  # LUMO
    assert alpha_orbitals[5].index == 5
    assert alpha_orbitals[5].energy == approx(0.4341)
    assert alpha_orbitals[5].occ is None
    # assert alpha_orbitals[6] == Orbital(index=6, energy_eh=approx(0.5498), occ=None)  # LUMO+1
    assert alpha_orbitals[6].index == 6
    assert alpha_orbitals[6].energy == approx(0.5498)
    assert alpha_orbitals[6].occ is None

    # --- Atomic Charges Checks ---
    assert len(data.atomic_charges) == 1
    mulliken = data.atomic_charges[0]
    assert mulliken.method == "Mulliken"
    assert mulliken.charges[0] == approx(0.173083)  # Atom index 0 (H)
    assert mulliken.charges[1] == approx(-0.346478)  # Atom index 1 (O)
    assert mulliken.charges[2] == approx(0.173395)  # Atom index 2 (H)

    # --- Multipole Checks ---
    assert data.multipole is not None
    assert data.multipole.charge_esu == approx(-0.0000)

    assert data.multipole.dipole is not None
    dipole: DipoleMomentData = data.multipole.dipole
    assert dipole.x_debye == approx(-0.8135)
    assert dipole.y_debye == approx(-0.1666)
    assert dipole.z_debye == approx(-1.4237)
    assert dipole.total_debye == approx(1.6482)

    assert data.multipole.quadrupole is not None
    quad: QuadrupoleMoments = data.multipole.quadrupole
    assert quad.xx == approx(-8.3319)
    assert quad.xy == approx(-1.9775)
    assert quad.yy == approx(-6.5229)
    assert quad.xz == approx(-3.5103)
    assert quad.yz == approx(-1.9205)
    assert quad.zz == approx(-4.7954)

    assert data.multipole.octopole is not None
    octo: OctopoleMoments = data.multipole.octopole
    assert octo.xxx == approx(-44.7808)
    assert octo.xxy == approx(-15.2389)
    assert octo.xyy == approx(-17.7938)
    assert octo.yyy == approx(-29.1521)
    assert octo.xxz == approx(-8.0908)
    assert octo.xyz == approx(-4.6554)
    assert octo.yyz == approx(-1.6923)
    assert octo.xzz == approx(-10.5109)
    assert octo.yzz == approx(-7.6527)
    assert octo.zzz == approx(1.7889)

    assert data.multipole.hexadecapole is not None
    hexa: HexadecapoleMoments = data.multipole.hexadecapole
    assert hexa.xxxx == approx(-193.1184)
    assert hexa.xxxy == approx(-75.8577)
    assert hexa.xxyy == approx(-59.8295)
    assert hexa.xyyy == approx(-71.1532)
    assert hexa.yyyy == approx(-92.7422)
    assert hexa.xxxz == approx(-16.9182)
    assert hexa.xxyz == approx(-10.5382)
    assert hexa.xyyz == approx(-3.9580)
    assert hexa.yyyz == approx(0.3498)
    assert hexa.xxzz == approx(-25.0172)
    assert hexa.xyzz == approx(-16.7675)
    assert hexa.yyzz == approx(-13.8075)
    assert hexa.xzzz == approx(4.1195)
    assert hexa.yzzz == approx(2.6781)
    assert hexa.zzzz == approx(-5.5277)

    # --- Error/Warning Checks ---
    # Assumes the fixture creation succeeded without errors.
    # Accessing internal _MutableCalculationData would be needed for direct check.
    # For now, rely on fixture and successful parsing to imply no errors.


# --- Error Handling and Edge Case Tests ---

MALFORMED_FINAL_ENERGY = """
SCF time:   CPU 0.28s  wall 0.00s
SCF   energy =   -75.31188446
Total energy =   INVALID_VALUE
"""


def test_malformed_final_energy(caplog: LogCaptureFixture) -> None:
    """Test parsing handles non-numeric final energy line without error but marks status ERROR."""
    with caplog.at_level(logging.WARNING):
        data = qchem.parse_qchem_sp_output(MALFORMED_FINAL_ENERGY)

    assert data.final_energy_eh is None  # The malformed line is ignored
    assert data.termination_status == "ERROR"  # ERROR due to missing normal termination
    assert "Termination status unknown after parsing, assuming ERROR." in caplog.text
    # Check for other expected warnings due to minimal input
    assert "Standard orientation geometry block was not found or parsed." in caplog.text
    assert "Input geometry block ($molecule) was not found or parsed." in caplog.text


MALFORMED_NUC_REPULSION = """
             Standard Nuclear Orientation (Angstroms)
    I     Atom           X                Y                Z
 ----------------------------------------------------------------
    1      H       1.3649900000     1.6938500000    -0.1974800000
 ----------------------------------------------------------------
 Nuclear Repulsion Energy =           BAD_NUMBER hartrees
"""


def test_malformed_nuclear_repulsion(caplog: LogCaptureFixture) -> None:
    """Test parsing handles non-numeric nuclear repulsion line without specific warning."""
    with caplog.at_level(logging.WARNING):
        data = qchem.parse_qchem_sp_output(MALFORMED_NUC_REPULSION)

    # The specific warning isn't logged because the regex doesn't match the line
    assert "Could not parse Nuclear Repulsion from line" not in caplog.text
    assert data.nuclear_repulsion_eh is None  # Value is never assigned
    assert data.termination_status == "ERROR"  # ERROR due to missing normal termination
    # Check for other expected warnings due to minimal input
    assert "Input geometry block ($molecule) was not found or parsed." in caplog.text
    assert "Termination status unknown after parsing, assuming ERROR." in caplog.text


EXPLICIT_ERROR_TERMINATION = """
 SCF failed to converge. Error.
 Some other text.
"""


def test_explicit_error_termination() -> None:
    """Test detection of explicit error messages."""
    data = qchem.parse_qchem_sp_output(EXPLICIT_ERROR_TERMINATION)
    assert data.termination_status == "ERROR"


MISSING_NORMAL_TERMINATION = """
 Some calculation data...
 Total energy =   -10.0
 Final line, but not the standard termination message.
"""


def test_missing_normal_termination(caplog: LogCaptureFixture) -> None:
    """Test status defaults to ERROR if normal termination is missing."""
    with caplog.at_level(logging.WARNING):
        data = qchem.parse_qchem_sp_output(MISSING_NORMAL_TERMINATION)
    assert data.termination_status == "ERROR"
    assert "Termination status unknown after parsing, assuming ERROR." in caplog.text


MISSING_INPUT_GEOMETRY = """
$rem
JOBTYPE         sp
METHOD          wb97x-d3
BASIS           sto-3g
$end

             Standard Nuclear Orientation (Angstroms)
    I     Atom           X                Y                Z
 ----------------------------------------------------------------
    1      H       1.3649900000     1.6938500000    -0.1974800000
 ----------------------------------------------------------------
 Nuclear Repulsion Energy =           8.93764808 hartrees
 Total energy =   -75.31188446
        *************************************************************
        *                                                           *
        *  Thank you very much for using Q-Chem.  Have a nice day.  *
        *                                                           *
        *************************************************************
"""


def test_missing_input_geometry(caplog: LogCaptureFixture) -> None:
    """Test parsing proceeds with warnings if input geometry is missing."""
    with caplog.at_level(logging.WARNING):  # Error is logged at ERROR level, warning at WARNING
        data = qchem.parse_qchem_sp_output(MISSING_INPUT_GEOMETRY)
    assert data.input_geometry is None
    assert data.standard_orientation_geometry is not None  # Should still parse this
    assert data.termination_status == "NORMAL"  # Ends normally
    assert "Input geometry block ($molecule) was not found or parsed." in caplog.text


PREMATURE_END_IN_SCF = """
$molecule
0 1
H        1.36499000      1.69385000     -0.19748000
$end
$rem
JOBTYPE         sp
METHOD          wb97x-d3
BASIS           sto-3g
$end
---------------------------------------
  Cycle       Energy         DIIS error
 ---------------------------------------
    1     -75.0734525425      3.82e-01
    2     -75.2783922518      5.94e-02
    3     -75.3074084202      2.46e-02
    4     -75.3118637269      1.29e-03
# File ends here abruptly
"""


def test_premature_end_during_block_parsing(caplog: LogCaptureFixture) -> None:
    """Test parsing handles input ending before/during potential block (ScfParser doesn't match)."""
    # For this specific input, ScfParser.matches likely returns False.
    # Therefore, the specific StopIteration->ParsingError path in sp.py isn't triggered.
    # Parsing finishes, and status becomes ERROR due to missing normal termination.
    with caplog.at_level(logging.WARNING):
        data = qchem.parse_qchem_sp_output(PREMATURE_END_IN_SCF)

    assert data.termination_status == "ERROR"
    assert data.scf is None  # SCF block was never parsed
    # Check for expected warnings due to minimal input
    assert "Standard orientation geometry block was not found or parsed." in caplog.text
    assert "Termination status unknown after parsing, assuming ERROR." in caplog.text
    # Ensure the specific StopIteration error log is NOT present
    assert "unexpectedly consumed end of iterator" not in caplog.text


def test_empty_input(caplog: LogCaptureFixture) -> None:
    """Test parsing an empty string completes with ERROR status and logs warnings."""
    with caplog.at_level(logging.WARNING):
        data = qchem.parse_qchem_sp_output("")

    assert data.termination_status == "ERROR"
    assert data.final_energy_eh is None
    assert data.input_geometry is None
    assert data.standard_orientation_geometry is None
    assert "Standard orientation geometry block was not found or parsed." in caplog.text
    assert "Input geometry block ($molecule) was not found or parsed." in caplog.text
    assert "Termination status unknown after parsing, assuming ERROR." in caplog.text


MINIMAL_HEADER = """
                  Welcome to Q-Chem
     A Quantum Leap Into The Future Of Chemistry
"""


def test_minimal_header_input(caplog: LogCaptureFixture) -> None:
    """Test parsing input with only the Q-Chem header."""
    with caplog.at_level(logging.WARNING):
        data = qchem.parse_qchem_sp_output(MINIMAL_HEADER)
    assert data.termination_status == "ERROR"
    assert data.final_energy_eh is None
    assert data.input_geometry is None
    assert data.standard_orientation_geometry is None
    assert "Standard orientation geometry block was not found or parsed." in caplog.text
    assert "Input geometry block ($molecule) was not found or parsed." in caplog.text
    assert "Termination status unknown after parsing, assuming ERROR." in caplog.text


# --- Test for SP with SMD ---


def test_parse_qchem_sp_output_h2o_smd(parsed_sp_sto_smd_data: CalculationData) -> None:
    """
    Integration test for parsing the H2O SP output file with SMD solvation.

    Verifies that the main fields extracted by parse_qchem_sp_output match
    the expected values from the data/calculations/examples/qchem/h2o/sp-sto-smd.out file.
    """
    data = parsed_sp_sto_smd_data

    # --- Top-Level Checks ---
    assert data.termination_status == "NORMAL"
    assert data.final_energy_eh == approx(-75.31846024)  # This is G(tot) from SMD output
    assert data.nuclear_repulsion_eh == approx(8.93764808)

    # --- Metadata Checks ---
    assert data.metadata.qchem_version == "6.2"
    assert data.metadata.host == "login30"
    # assert data.metadata.run_date == "Sun May  4 14:52:50 2025" # Date is dynamic
    assert data.metadata.calculation_method == "wb97x-d3"
    assert data.metadata.basis_set == "sto-3g"
    assert data.metadata.solvent_method == "smd"
    assert data.metadata.solvent_name == "water"

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
    scf: ScfData = data.scf
    assert scf.converged is True
    # For SMD, the SCF energy reported in the block is E_SCF (including G_PCM)
    assert scf.energy_eh == approx(-75.32080770)
    assert scf.n_iterations == 7
    assert len(scf.iteration_history) == 7
    assert scf.iteration_history[0].iteration == 1
    assert scf.iteration_history[0].energy_eh == approx(-75.0734525440)
    assert scf.iteration_history[0].diis_error == approx(3.82e-01)
    assert scf.iteration_history[-1].iteration == 7
    assert scf.iteration_history[-1].energy_eh == approx(-75.3208077035)
    assert scf.iteration_history[-1].diis_error == approx(2.17e-08)

    # --- Orbital Checks ---
    assert data.orbitals is not None
    assert data.orbitals.beta_orbitals is None  # Restricted calculation
    assert data.orbitals.alpha_orbitals is not None
    alpha_orbitals = data.orbitals.alpha_orbitals
    assert len(alpha_orbitals) == 7  # 5 occupied + 2 virtual

    # Check specific occupied orbitals
    assert alpha_orbitals[0].index == 0
    assert alpha_orbitals[0].energy == approx(-18.9200)
    assert alpha_orbitals[0].occ is None
    assert alpha_orbitals[4].index == 4
    assert alpha_orbitals[4].energy == approx(-0.2422)  # HOMO
    assert alpha_orbitals[4].occ is None

    # Check specific virtual orbitals
    assert alpha_orbitals[5].index == 5
    assert alpha_orbitals[5].energy == approx(0.4520)  # LUMO
    assert alpha_orbitals[5].occ is None
    assert alpha_orbitals[6].index == 6
    assert alpha_orbitals[6].energy == approx(0.5636)  # LUMO+1
    assert alpha_orbitals[6].occ is None

    # --- Atomic Charges Checks ---
    assert len(data.atomic_charges) == 1
    mulliken = data.atomic_charges[0]
    assert mulliken.method == "Mulliken"
    assert mulliken.charges[0] == approx(0.193937)  # Atom index 0 (H)
    assert mulliken.charges[1] == approx(-0.388200)  # Atom index 1 (O)
    assert mulliken.charges[2] == approx(0.194263)  # Atom index 2 (H)

    # --- Multipole Checks ---
    assert data.multipole is not None
    assert data.multipole.charge_esu == approx(-0.0000)

    assert data.multipole.dipole is not None
    dipole: DipoleMomentData = data.multipole.dipole
    assert dipole.x_debye == approx(-0.8826)
    assert dipole.y_debye == approx(-0.1808)
    assert dipole.z_debye == approx(-1.5445)
    assert dipole.total_debye == approx(1.7880)

    assert data.multipole.quadrupole is not None
    quad: QuadrupoleMoments = data.multipole.quadrupole
    assert quad.xx == approx(-8.5235)
    assert quad.xy == approx(-2.1415)
    assert quad.yy == approx(-6.5392)
    assert quad.xz == approx(-3.8091)
    assert quad.yz == approx(-2.0882)
    assert quad.zz == approx(-4.6864)

    assert data.multipole.octopole is not None
    octo: OctopoleMoments = data.multipole.octopole
    assert octo.xxx == approx(-45.1424)
    assert octo.xxy == approx(-15.7146)
    assert octo.xyy == approx(-18.0848)
    assert octo.yyy == approx(-29.1344)
    assert octo.xxz == approx(-8.8897)
    assert octo.xyz == approx(-5.0592)
    assert octo.yyz == approx(-1.9524)
    assert octo.xzz == approx(-10.2351)
    assert octo.yzz == approx(-7.5112)
    assert octo.zzz == approx(1.6208)

    assert data.multipole.hexadecapole is not None
    hexa: HexadecapoleMoments = data.multipole.hexadecapole
    assert hexa.xxxx == approx(-193.4737)
    assert hexa.xxxy == approx(-76.9501)
    assert hexa.xxyy == approx(-60.7182)
    assert hexa.xyyy == approx(-71.6161)
    assert hexa.yyyy == approx(-92.5752)
    assert hexa.xxxz == approx(-19.1566)
    assert hexa.xxyz == approx(-11.6023)
    assert hexa.xyyz == approx(-4.5683)
    assert hexa.yyyz == approx(-0.0879)
    assert hexa.xxzz == approx(-24.2567)
    assert hexa.xyzz == approx(-16.4191)
    assert hexa.yyzz == approx(-13.5970)
    assert hexa.xzzz == approx(3.6825)
    assert hexa.yzzz == approx(2.4530)
    assert hexa.zzzz == approx(-5.3042)

    # Check SMD specific data
    assert data.smd_data is not None
    assert data.smd_data.g_pcm_kcal_mol == approx(-6.0201)
    assert data.smd_data.g_cds_kcal_mol == approx(1.4731)
    assert data.smd_data.g_enp_au == approx(-75.32080770)
    assert data.smd_data.g_tot_au == approx(-75.31846024)

    # --- Error/Warning Checks ---
    # For now, rely on fixture and successful parsing to imply no errors.


def test_parse_qchem_sp_output_h2o_tzvppd_smd(parsed_sp_tzvppd_smd_data: CalculationData) -> None:
    """
    Integration test for parsing the H2O SP output file with SMD solvation and def2-tzvppd basis.

    Verifies that the main fields extracted by parse_qchem_sp_output match
    the expected values from the data/calculations/examples/qchem/h2o/sp-tzvppd-smd.out file.
    """
    data = parsed_sp_tzvppd_smd_data

    # --- Top-Level Checks ---
    assert data.termination_status == "NORMAL"
    assert data.final_energy_eh == approx(-76.45372896)  # G(tot) from SMD output
    assert data.nuclear_repulsion_eh == approx(8.93764808)

    # --- Metadata Checks ---
    assert data.metadata.qchem_version == "6.2"
    assert data.metadata.host == "login30"
    # assert data.metadata.run_date == "Sun May  4 14:52:49 2025" # Date is dynamic
    assert data.metadata.calculation_method == "wb97x-d3"
    assert data.metadata.basis_set == "def2-tzvppd"
    assert data.metadata.solvent_method == "smd"
    assert data.metadata.solvent_name == "water"

    # --- Geometry Checks ---
    assert data.input_geometry is not None
    assert len(data.input_geometry) == 3
    assert data.input_geometry[0] == Atom(symbol="H", x=1.36499, y=1.69385, z=-0.19748)
    assert data.input_geometry[1] == Atom(symbol="O", x=2.32877, y=1.56294, z=-0.04168)
    assert data.input_geometry[2] == Atom(symbol="H", x=2.70244, y=1.31157, z=-0.91665)

    assert data.standard_orientation_geometry is not None
    assert len(data.standard_orientation_geometry) == 3
    assert data.standard_orientation_geometry[0] == Atom(symbol="H", x=1.3649900000, y=1.6938500000, z=-0.1974800000)
    assert data.standard_orientation_geometry[1] == Atom(symbol="O", x=2.3287700000, y=1.5629400000, z=-0.0416800000)
    assert data.standard_orientation_geometry[2] == Atom(symbol="H", x=2.7024400000, y=1.3115700000, z=-0.9166500000)

    # --- SCF Checks ---
    assert data.scf is not None
    scf: ScfData = data.scf
    assert scf.converged is True
    assert scf.energy_eh == approx(-76.45607642)  # E_SCF (including G_PCM)
    assert scf.n_iterations == 7
    assert len(scf.iteration_history) == 7
    assert scf.iteration_history[0].iteration == 1
    assert scf.iteration_history[0].energy_eh == approx(-76.3019247487)
    assert scf.iteration_history[0].diis_error == approx(4.14e-02)
    assert scf.iteration_history[-1].iteration == 7
    assert scf.iteration_history[-1].energy_eh == approx(-76.4560764224)
    assert scf.iteration_history[-1].diis_error == approx(5.94e-06)

    # --- Orbital Checks ---
    assert data.orbitals is not None
    assert data.orbitals.beta_orbitals is None  # Restricted calculation
    assert data.orbitals.alpha_orbitals is not None
    alpha_orbitals = data.orbitals.alpha_orbitals
    assert len(alpha_orbitals) == 74  # 5 occupied + 69 virtual for def2-tzvppd

    assert alpha_orbitals[0].index == 0
    assert alpha_orbitals[0].energy == approx(-19.2349)
    assert alpha_orbitals[0].occ is None
    assert alpha_orbitals[4].index == 4  # HOMO
    assert alpha_orbitals[4].energy == approx(-0.4206)
    assert alpha_orbitals[4].occ is None

    assert alpha_orbitals[5].index == 5  # LUMO
    assert alpha_orbitals[5].energy == approx(0.0815)
    assert alpha_orbitals[5].occ is None
    assert alpha_orbitals[6].index == 6  # LUMO+1
    assert alpha_orbitals[6].energy == approx(0.1133)
    assert alpha_orbitals[6].occ is None

    # --- Atomic Charges Checks ---
    assert len(data.atomic_charges) == 1
    mulliken = data.atomic_charges[0]
    assert mulliken.method == "Mulliken"
    assert mulliken.charges[0] == approx(0.258251)  # H
    assert mulliken.charges[1] == approx(-0.515681)  # O
    assert mulliken.charges[2] == approx(0.257430)  # H

    # --- Multipole Checks ---
    assert data.multipole is not None
    assert data.multipole.charge_esu == approx(-0.0000)

    assert data.multipole.dipole is not None
    dipole: DipoleMomentData = data.multipole.dipole
    assert dipole.x_debye == approx(-1.2166)
    assert dipole.y_debye == approx(-0.2484)
    assert dipole.z_debye == approx(-2.1256)
    assert dipole.total_debye == approx(2.4617)

    assert data.multipole.quadrupole is not None
    quad: QuadrupoleMoments = data.multipole.quadrupole
    assert quad.xx == approx(-10.2609)
    assert quad.xy == approx(-3.1695)
    assert quad.yy == approx(-8.4016)
    assert quad.xz == approx(-5.5077)
    assert quad.yz == approx(-2.7052)
    assert quad.zz == approx(-5.3504)

    assert data.multipole.octopole is not None
    octo: OctopoleMoments = data.multipole.octopole
    assert octo.xxx == approx(-53.2148)
    assert octo.xxy == approx(-20.3433)
    assert octo.xyy == approx(-24.4478)
    assert octo.yyy == approx(-37.4666)
    assert octo.xxz == approx(-14.1705)
    assert octo.xyz == approx(-6.9000)
    assert octo.yyz == approx(-2.6870)
    assert octo.xzz == approx(-11.5574)
    assert octo.yzz == approx(-8.8172)
    assert octo.zzz == approx(0.2702)

    assert data.multipole.hexadecapole is not None
    hexa: HexadecapoleMoments = data.multipole.hexadecapole
    assert hexa.xxxx == approx(-229.0175)
    assert hexa.xxxy == approx(-96.1184)
    assert hexa.xxyy == approx(-81.3512)
    assert hexa.xyyy == approx(-95.7993)
    assert hexa.yyyy == approx(-122.7738)
    assert hexa.xxxz == approx(-35.9972)
    assert hexa.xxyz == approx(-17.5148)
    assert hexa.xyyz == approx(-6.8408)
    assert hexa.yyyz == approx(-0.8927)
    assert hexa.xxzz == approx(-27.1713)
    assert hexa.xyzz == approx(-19.3476)
    assert hexa.yyzz == approx(-17.0816)
    assert hexa.xzzz == approx(-0.1908)
    assert hexa.yzzz == approx(0.8986)
    assert hexa.zzzz == approx(-6.6403)

    # Check SMD specific data
    assert data.smd_data is not None
    assert data.smd_data.g_pcm_kcal_mol == approx(-10.8438)
    assert data.smd_data.g_cds_kcal_mol == approx(1.4731)
    assert data.smd_data.g_enp_au == approx(-76.45607642)
    assert data.smd_data.g_tot_au == approx(-76.45372896)

    # --- Error/Warning Checks ---
    # For now, rely on fixture and successful parsing to imply no errors.

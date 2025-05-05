from pytest import approx

from calcflow.parsers.qchem.typing import (
    Atom,
    CalculationData,
    DipoleMomentData,
    HexadecapoleMoments,
    OctopoleMoments,
    QuadrupoleMoments,
    ScfData,
)


def test_parse_qchem_sp_output_h2o(parsed_sp_data: CalculationData) -> None:
    """
    Integration test for parsing the standard H2O SP output file.

    Verifies that the main fields extracted by parse_qchem_sp_output match
    the expected values from the data/calculations/examples/qchem/h2o/sp.out file.
    """
    data = parsed_sp_data

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
    assert scf.iteration_history[0].iteration == 1
    assert scf.iteration_history[0].energy_eh == approx(-75.0734525425)
    assert scf.iteration_history[0].diis_error == approx(3.82e-01)
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
    assert alpha_orbitals[0].index == 0
    assert alpha_orbitals[0].energy_eh == approx(-18.9359)
    assert alpha_orbitals[0].occupation is None
    assert alpha_orbitals[4].index == 4
    assert alpha_orbitals[4].energy_eh == approx(-0.2436)
    assert alpha_orbitals[4].occupation is None

    # Check specific virtual orbitals
    assert alpha_orbitals[5].index == 5
    assert alpha_orbitals[5].energy_eh == approx(0.4341)
    assert alpha_orbitals[5].occupation is None
    assert alpha_orbitals[6].index == 6
    assert alpha_orbitals[6].energy_eh == approx(0.5498)
    assert alpha_orbitals[6].occupation is None

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

    # --- Dispersion Correction Checks ---
    # NOTE: The parser doesn't currently extract this, but the info is in sp.out
    # Add assertions here if/when DispersionCorrectionData parsing is implemented.
    # assert data.dispersion_correction is not None
    # assert data.dispersion_correction.method == "D3(0)"
    # assert data.dispersion_correction.energy_eh == approx(-0.0000016383)

    # --- Error/Warning Checks ---
    # Assumes the fixture creation succeeded without errors.
    # Accessing internal _MutableCalculationData would be needed for direct check.
    # For now, rely on fixture and successful parsing to imply no errors.

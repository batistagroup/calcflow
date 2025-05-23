import re
from dataclasses import replace

import pytest

from calcflow.exceptions import ConfigurationError, ValidationError
from calcflow.geometry.static import AtomCoords, Geometry
from calcflow.inputs.qchem import QchemInput


def test_enable_mom_valid(default_qchem_input: QchemInput) -> None:
    """Test enabling MOM with valid methods."""
    inp_imom = default_qchem_input.enable_mom(method="IMOM")
    assert inp_imom.run_mom
    assert inp_imom.mom_method == "IMOM"
    assert inp_imom is not default_qchem_input

    inp_mom = default_qchem_input.enable_mom(method="MOM")
    assert inp_mom.run_mom
    assert inp_mom.mom_method == "MOM"

    inp_lower = default_qchem_input.enable_mom(method="imom")
    assert inp_lower.run_mom
    assert inp_lower.mom_method == "IMOM"


def test_enable_mom_invalid_method(default_qchem_input: QchemInput) -> None:
    """Test enabling MOM with an invalid method raises ValidationError."""
    with pytest.raises(ValidationError, match="MOM method must be either 'IMOM' or 'MOM'"):
        default_qchem_input.enable_mom(method="INVALID")


def test_set_mom_occupation_valid(mom_enabled_input: QchemInput) -> None:
    """Test setting valid MOM occupation strings."""
    inp = mom_enabled_input.set_mom_occupation(alpha_occ="1:4 6", beta_occ="1:5")
    assert inp.mom_alpha_occ == "1:4 6"
    assert inp.mom_beta_occ == "1:5"
    assert inp.mom_transition is None  # Should clear transition
    assert inp is not mom_enabled_input


def test_set_mom_occupation_before_enable(default_qchem_input: QchemInput) -> None:
    """Test setting MOM occupation before enabling MOM raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="MOM must be enabled first"):
        default_qchem_input.set_mom_occupation(alpha_occ="1:4 6", beta_occ="1:5")


@pytest.mark.parametrize(
    "occ_str, label",
    [
        ("1:a", "alpha"),
        ("1: 5", "alpha"),
        ("1-5", "alpha"),
        ("1:4:5", "alpha"),
        ("1 5:b", "beta"),
    ],
)
def test_set_mom_occupation_invalid_format(mom_enabled_input: QchemInput, occ_str: str, label: str) -> None:
    """Test setting MOM occupation with invalid Q-Chem format raises ValidationError."""
    with pytest.raises(ValidationError, match=f"Invalid {label} occupation"):
        if label == "alpha":
            mom_enabled_input.set_mom_occupation(alpha_occ=occ_str, beta_occ="1:5")
        else:
            mom_enabled_input.set_mom_occupation(alpha_occ="1:5", beta_occ=occ_str)


def test_set_mom_transition_valid(mom_enabled_input: QchemInput) -> None:
    """Test setting valid MOM transitions."""
    inp_homo_lumo = mom_enabled_input.set_mom_transition("HOMO->LUMO")
    assert inp_homo_lumo.mom_transition == "HOMO->LUMO"
    assert inp_homo_lumo.mom_alpha_occ is None  # Should clear direct occ
    assert inp_homo_lumo.mom_beta_occ is None
    assert inp_homo_lumo is not mom_enabled_input

    inp_offset = mom_enabled_input.set_mom_transition("HOMO-1 -> LUMO+2")
    assert inp_offset.mom_transition == "HOMO-1 -> LUMO+2"

    inp_integer = mom_enabled_input.set_mom_transition("5 -> 7")
    assert inp_integer.mom_transition == "5 -> 7"

    inp_mixed = mom_enabled_input.set_mom_transition("HOMO-2 -> 8")
    assert inp_mixed.mom_transition == "HOMO-2 -> 8"

    inp_case = mom_enabled_input.set_mom_transition("homo->lumo+1")
    assert inp_case.mom_transition == "homo->lumo+1"


def test_set_mom_transition_ionization(mom_enabled_input: QchemInput) -> None:
    """Test setting valid MOM ionization transitions."""
    inp_ionization = mom_enabled_input.set_mom_transition("5->vac")
    assert inp_ionization.mom_transition == "5->vac"
    assert inp_ionization.mom_alpha_occ is None
    assert inp_ionization.mom_beta_occ is None

    inp_homo_ionization = mom_enabled_input.set_mom_transition("HOMO->vac")
    assert inp_homo_ionization.mom_transition == "HOMO->vac"

    inp_spin_specific = mom_enabled_input.set_mom_transition("5(beta)->vac")
    assert inp_spin_specific.mom_transition == "5(beta)->vac"


def test_set_mom_transitions_multiple(mom_enabled_input: QchemInput) -> None:
    """Test setting multiple MOM transitions."""
    transitions = ["HOMO->LUMO", "5->vac"]
    inp = mom_enabled_input.set_mom_transitions(transitions)
    assert inp.mom_transition == "HOMO->LUMO; 5->vac"
    assert inp.mom_alpha_occ is None
    assert inp.mom_beta_occ is None

    # Test excitation followed by ionization
    transitions = ["HOMO-1->LUMO+1", "3(beta)->vac"]
    inp = mom_enabled_input.set_mom_transitions(transitions)
    assert inp.mom_transition == "HOMO-1->LUMO+1; 3(beta)->vac"


def test_set_mom_transition_before_enable(default_qchem_input: QchemInput) -> None:
    """Test setting MOM transition before enabling MOM raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="MOM must be enabled first"):
        default_qchem_input.set_mom_transition("HOMO->LUMO")


def test_set_mom_transitions_before_enable(default_qchem_input: QchemInput) -> None:
    """Test setting multiple MOM transitions before enabling MOM raises ConfigurationError."""
    with pytest.raises(
        ConfigurationError, match="MOM must be enabled first. Call enable_mom\\(\\) before set_mom_transitions\\(\\)."
    ):
        default_qchem_input.set_mom_transitions(["HOMO->LUMO", "5->vac"])


@pytest.mark.parametrize(
    "transition_str, error_msg",
    [
        ("HOMO->", "Invalid target orbital: ''"),
        ("->LUMO", "Invalid source orbital: ''"),
        ("HOMOLUMO", "Invalid transition format"),
        ("HOMO + 1 -> LUMO", "Invalid source orbital: 'HOMO + 1'. Must be HOMO[-n], LUMO[+n], or positive integer."),
        ("HOMO->LUMO-1", "Invalid target orbital: 'LUMO-1'"),
        ("HOMO+1->LUMO", "Invalid source orbital: 'HOMO+1'"),
        ("LU -> MO", "Invalid source orbital: 'LU'"),
        ("5->0", "Invalid target orbital: '0'. Orbital numbers must be positive."),
        ("0->6", "Invalid source orbital: '0'. Orbital numbers must be positive."),
        ("-1->6", "Invalid source orbital: '-1'. Must be HOMO[-n], LUMO[+n], or positive integer."),
        ("vac->5", "Invalid source orbital: 'vac'. Must be HOMO[-n], LUMO[+n], or positive integer."),
        ("5(gamma)->vac", "Invalid spin specification '(gamma)' in '5(gamma)'. Must be 'alpha' or 'beta'."),
    ],
)
def test_set_mom_transition_invalid_format(mom_enabled_input: QchemInput, transition_str: str, error_msg: str) -> None:
    """Test setting invalid MOM transition formats raises ValidationError."""
    with pytest.raises(ValidationError, match=re.escape(error_msg)):
        mom_enabled_input.set_mom_transition(transition_str)


def test_generate_occupied_block_not_mom(default_qchem_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test that the occupied block is empty if MOM is not enabled."""
    assert default_qchem_input._generate_occupied_block(h2o_geometry) == ""


def test_generate_occupied_block_not_unrestricted(mom_enabled_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test error if MOM enabled but calculation is restricted."""
    inp = replace(mom_enabled_input, unrestricted=False).set_mom_transition("HOMO->LUMO")
    with pytest.raises(ConfigurationError, match="MOM requires unrestricted=True"):
        inp._generate_occupied_block(h2o_geometry)


def test_generate_occupied_block_direct_occupation(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test occupied block generation with direct occupation strings."""
    inp = unrestricted_input.enable_mom().set_mom_occupation(alpha_occ="1:4 6", beta_occ="1:5")
    expected = """$occupied
1:4 6
1:5
$end"""
    assert inp._generate_occupied_block(h2o_geometry) == expected


def test_generate_occupied_block_homo_lumo_transition(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test occupied block generation for HOMO->LUMO transition (H2O = 10 electrons)."""
    inp = unrestricted_input.enable_mom().set_mom_transition("HOMO->LUMO")
    # H2O: 10 electrons -> 5 alpha, 5 beta. HOMO=5, LUMO=6
    # Alpha: 1:4 6, Beta: 1:5
    expected = """$occupied
1:4 6
1:5
$end"""
    assert inp._generate_occupied_block(h2o_geometry) == expected


def test_generate_occupied_block_offset_transition(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test occupied block generation for HOMO-1 -> LUMO+1 transition (H2O = 10 electrons)."""
    inp = unrestricted_input.enable_mom().set_mom_transition("HOMO-1 -> LUMO+1")
    # H2O: 10 electrons -> 5 alpha, 5 beta. HOMO=5, LUMO=6
    # Source = HOMO-1 = 4. Target = LUMO+1 = 7.
    # Alpha: 1:3 5 7, Beta: 1:5
    expected = """$occupied
1:3 5 7
1:5
$end"""
    assert inp._generate_occupied_block(h2o_geometry) == expected


def test_generate_occupied_block_integer_transition(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test occupied block generation for 4 -> 7 transition (H2O = 10 electrons)."""
    inp = unrestricted_input.enable_mom().set_mom_transition("4 -> 7")
    # H2O: 10 electrons -> 5 alpha, 5 beta. HOMO=5, LUMO=6
    # Source = 4. Target = 7.
    # Alpha: 1:3 5 7, Beta: 1:5
    expected = """$occupied
1:3 5 7
1:5
$end"""
    assert inp._generate_occupied_block(h2o_geometry) == expected


def test_generate_occupied_block_ionization(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test occupied block generation for ionization (H2O = 10 electrons)."""
    # Core ionization: remove electron from orbital 2 (beta channel)
    # This creates a cationic doublet state with 9 electrons
    inp = unrestricted_input.enable_mom().set_mom_transition("2(beta)->vac")
    inp = inp.set_mom_job2_charge(1).set_mom_job2_spin_multiplicity(2)  # +1 charge, doublet
    # H2O+: 9 electrons -> 5 alpha, 4 beta
    # Alpha: 1:5 (unchanged), Beta: 1 3:5 (orbital 2 removed)
    expected = """$occupied
1:5
1 3:5
$end"""
    assert inp._generate_occupied_block(h2o_geometry) == expected


def test_generate_occupied_block_multiple_operations(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test occupied block generation for multiple operations (H2O = 10 electrons)."""
    # Excitation followed by ionization
    inp = unrestricted_input.enable_mom().set_mom_transition("HOMO->LUMO; 3(beta)->vac")
    inp = inp.set_mom_job2_charge(1).set_mom_job2_spin_multiplicity(2)  # +1 charge, doublet after ionization
    # H2O+: 9 electrons -> 5 alpha, 4 beta
    # After HOMO->LUMO: Alpha: 1:4 6, Beta: 1:5
    # After 3(beta)->vac: Alpha: 1:4 6, Beta: 1:2 4:5
    expected = """$occupied
1:4 6
1:2 4:5
$end"""
    assert inp._generate_occupied_block(h2o_geometry) == expected


# --- Test Export with MOM ---


def test_export_input_file_mom_direct_occupation(h2o_geometry: Geometry, unrestricted_input: QchemInput) -> None:
    """Test exporting a two-job MOM input file with direct occupation."""
    inp = unrestricted_input.enable_mom().set_mom_occupation(alpha_occ="1:4 6", beta_occ="1:5")

    output = inp.export_input_file(h2o_geometry)
    jobs = output.split("\n\n@@@\n\n")
    assert len(jobs) == 2
    job1, job2 = jobs

    # --- Check Job 1 ---
    assert "$molecule" in job1
    assert f"{inp.charge} {inp.spin_multiplicity}" in job1  # Should be 0 1 from fixture
    assert "O        0.00000000" in job1  # Corrected spacing
    assert "$rem" in job1
    assert "METHOD          hf" in job1  # From fixture
    assert "BASIS           def2-svp" in job1  # From fixture
    assert "JOBTYPE         sp" in job1  # Default energy task
    assert "UNRESTRICTED    True" in job1  # From unrestricted_input fixture
    assert "MOM_START" not in job1
    assert "$occupied" not in job1

    # --- Check Job 2 ---
    assert job2.startswith("$molecule\n    read\n$end")
    assert "$rem" in job2
    assert "SCF_GUESS       read" in job2
    assert "MOM_START       1" in job2
    assert "MOM_METHOD      IMOM" in job2
    assert "UNRESTRICTED    True" in job2  # Should be forced True
    assert "JOBTYPE         sp" in job2  # Should be forced SP
    assert "METHOD          hf" in job2  # Should be same as job 1
    assert "BASIS           def2-svp" in job2  # Should be same as job 1
    assert "$occupied" in job2
    assert "1:4 6" in job2
    assert "1:5" in job2


def test_export_input_file_mom_transition(h2o_geometry: Geometry, unrestricted_input: QchemInput) -> None:
    """Test exporting a two-job MOM input file with HOMO->LUMO transition."""
    inp = unrestricted_input.enable_mom().set_mom_transition("HOMO->LUMO")

    output = inp.export_input_file(h2o_geometry)
    jobs = output.split("\n\n@@@\n\n")
    assert len(jobs) == 2
    job1, job2 = jobs

    # --- Check Job 1 ---
    # Similar checks as above, ensure it's a standard unrestricted calculation
    assert "$molecule" in job1
    assert "0 1" in job1  # H2O fixture
    assert "$rem" in job1
    assert "METHOD          hf" in job1
    assert "BASIS           def2-svp" in job1
    assert "JOBTYPE         sp" in job1
    assert "UNRESTRICTED    True" in job1
    assert "MOM_START" not in job1
    assert "$occupied" not in job1

    # --- Check Job 2 ---
    assert job2.startswith("$molecule\n    read\n$end")
    assert "$rem" in job2
    assert "SCF_GUESS       read" in job2
    assert "MOM_START       1" in job2
    assert "MOM_METHOD      IMOM" in job2
    assert "UNRESTRICTED    True" in job2
    assert "JOBTYPE         sp" in job2
    assert "$occupied" in job2
    # H2O HOMO=5, LUMO=6. Alpha: 1:4 6, Beta: 1:5
    assert "1:4 6" in job2
    assert "1:5" in job2


def test_export_input_file_mom_ionization(h2o_geometry: Geometry, unrestricted_input: QchemInput) -> None:
    """Test exporting a two-job MOM input file with ionization."""
    inp = unrestricted_input.enable_mom().set_mom_transition("2(beta)->vac")
    inp = inp.set_mom_job2_charge(1).set_mom_job2_spin_multiplicity(2)  # +1 charge, doublet after ionization

    output = inp.export_input_file(h2o_geometry)
    jobs = output.split("\n\n@@@\n\n")
    assert len(jobs) == 2
    job1, job2 = jobs

    # --- Check Job 2 has ionization occupation ---
    assert "$occupied" in job2
    assert "1:5" in job2  # Alpha unchanged
    assert "1 3:5" in job2  # Beta with orbital 2 removed


def test_export_input_file_mom_opt_then_sp(h2o_geometry: Geometry, unrestricted_input: QchemInput) -> None:
    """Test exporting MOM where the *first* job is an OPT, second is SP."""
    inp = replace(unrestricted_input, task="geometry").enable_mom().set_mom_transition("HOMO->LUMO")

    output = inp.export_input_file(h2o_geometry)
    jobs = output.split("\n\n@@@\n\n")
    assert len(jobs) == 2
    job1, job2 = jobs

    # --- Check Job 1 ---
    assert "$rem" in job1
    assert "JOBTYPE         opt" in job1  # Should be opt for job 1
    assert "UNRESTRICTED    True" in job1
    assert "MOM_START" not in job1

    # --- Check Job 2 ---
    assert "$rem" in job2
    assert "SCF_GUESS       read" in job2
    assert "MOM_START       1" in job2
    assert "JOBTYPE         sp" in job2  # Should be forced SP for job 2
    assert "UNRESTRICTED    True" in job2
    assert "$occupied" in job2


def test_export_input_file_mom_with_solvation(h2o_geometry: Geometry, unrestricted_input: QchemInput) -> None:
    """Test that solvation settings carry over to the second MOM job."""
    inp = unrestricted_input.enable_mom().set_mom_transition("HOMO->LUMO").set_solvation(model="pcm", solvent="water")

    output = inp.export_input_file(h2o_geometry)
    jobs = output.split("\n\n@@@\n\n")
    assert len(jobs) == 2
    job1, job2 = jobs

    # --- Check Job 1 ---
    assert "$rem" in job1
    assert "SOLVENT_METHOD    pcm" in job1
    assert "$solvent" in job1
    assert "SolventName           water" in job1

    # --- Check Job 2 ---
    assert "$rem" in job2
    assert "SOLVENT_METHOD    pcm" in job2  # Should still be present
    assert "$solvent" in job2  # Should still be present
    assert "SolventName           water" in job2
    assert "$occupied" in job2  # Ensure MOM stuff is also there


# --- Tests for set_mom_ground_state ---


def test_set_mom_ground_state_success(mom_enabled_input: QchemInput) -> None:
    """Test successful call to set_mom_ground_state."""
    inp = mom_enabled_input.set_mom_ground_state()
    assert inp.mom_transition == "GROUND_STATE"
    assert inp.mom_alpha_occ is None
    assert inp.mom_beta_occ is None
    assert inp.unrestricted is True  # Should be set by set_mom_ground_state
    assert inp is not mom_enabled_input  # Immutability


def test_set_mom_ground_state_before_enable(default_qchem_input: QchemInput) -> None:
    """Test calling set_mom_ground_state before enable_mom raises ConfigurationError."""
    with pytest.raises(
        ConfigurationError,
        match=r"MOM must be enabled \(run_mom=True\) via enable_mom\(\) before setting MOM to target the ground state\.",
    ):
        default_qchem_input.set_mom_ground_state()


# fmt:off
@pytest.mark.parametrize(
    "geometry_elements_coords, charge, multiplicity, expected_alpha, expected_beta, expected_error, error_match",
    [
        # Valid cases
        ((["O", "H", "H"], [[0, 0, 0], [0, 0, 1], [0, 1, 0]]), 0, 1, "1:5", "1:5", None, None),  # H2O, 10e-
        ((["He"], [[0, 0, 0]]), 0, 1, "1", "1", None, None),  # He, 2e-
        # Invalid system: odd total electrons for GROUND_STATE processing (QchemInput charge=0, mult=1)
        ((["Li"], [[0, 0, 0]]), 0, 1, None, None, ConfigurationError, "Expected an even number of total electrons"),
        # Invalid system: insufficient occupied orbitals
        (([], []), 0, 1, None, None, ConfigurationError, "System with 0 electrons .* insufficient occupied orbitals for 'GROUND_STATE' MOM"),
        # N atom (total_nuclear_charge=7) QchemInput(charge=0, mult=1) -> 7e-. Expect odd electron error.
        ((["N"], [[0, 0, 0]]), 0, 1, None, None, ConfigurationError, "Expected an even number of total electrons"),
    ],
) # fmt:on
def test_generate_occupied_block_mom_ground_state(
    unrestricted_input: QchemInput,
    geometry_elements_coords: tuple[list[str], list[list[float]]],
    charge: int,
    multiplicity: int,
    expected_alpha: str | None,
    expected_beta: str | None,
    expected_error: type[Exception] | None,
    error_match: str | None,
) -> None:
    """Test _generate_occupied_block with mom_transition='GROUND_STATE'."""
    elements, coords_list = geometry_elements_coords
    num_atoms = len(elements)

    mock_atoms: list[AtomCoords] = []
    for i, el_symbol in enumerate(elements):
        # Ensure coords_list has enough entries, or use dummy if not (though test data should be consistent)
        current_coords = tuple(coords_list[i]) if i < len(coords_list) else (0.0, 0.0, 0.0)
        mock_atoms.append((el_symbol, current_coords))  # type: ignore

    geom = Geometry(num_atoms=num_atoms, comment="Test geom", atoms=mock_atoms)
    # total_nuclear_charge will now be calculated by the Geometry object's property

    inp = replace(unrestricted_input, charge=charge, spin_multiplicity=multiplicity)
    inp = inp.enable_mom().set_mom_ground_state()  # unrestricted is True now

    if expected_error:
        with pytest.raises(expected_error, match=error_match):
            inp._generate_occupied_block(geom)
    else:
        block = inp._generate_occupied_block(geom)
        expected = f"""$occupied
{expected_alpha}
{expected_beta}
$end"""
        assert block == expected


def test_export_input_file_mom_ground_state(h2o_geometry: Geometry, unrestricted_input: QchemInput) -> None:
    """Test exporting a two-job MOM input file with mom_transition='GROUND_STATE'."""
    # H2O has 10 electrons. For charge 0, multiplicity 1, HOMO=5.
    # Ground state: alpha=1:5, beta=1:5
    inp = unrestricted_input.enable_mom().set_mom_ground_state()
    # Ensure h2o_geometry has total_nuclear_charge if it's not set by default by the fixture
    if not hasattr(h2o_geometry, "total_nuclear_charge") or h2o_geometry.total_nuclear_charge == 0:
        h2o_geometry.total_nuclear_charge = 10  # O (8) + H (1) + H (1)

    output = inp.export_input_file(h2o_geometry)
    jobs = output.split("\n\n@@@\n\n")
    assert len(jobs) == 2
    job1, job2 = jobs

    # --- Check Job 1 ---
    assert "$molecule" in job1
    assert f"{inp.charge} {inp.spin_multiplicity}" in job1  # Should be 0 1 from unrestricted_input
    assert "O        0.00000000" in job1
    assert "$rem" in job1
    assert "METHOD          hf" in job1
    assert "BASIS           def2-svp" in job1
    assert "JOBTYPE         sp" in job1
    assert "UNRESTRICTED    True" in job1  # From unrestricted_input and set_mom_ground_state
    assert "MOM_START" not in job1
    assert "$occupied" not in job1

    # --- Check Job 2 ---
    assert job2.startswith("$molecule\n    read\n$end")
    assert "$rem" in job2
    assert "SCF_GUESS       read" in job2
    assert "MOM_START       1" in job2
    assert f"MOM_METHOD      {inp.mom_method}" in job2  # Should be IMOM by default
    assert "UNRESTRICTED    True" in job2  # Should be forced True
    assert "JOBTYPE         sp" in job2  # Should be forced SP
    assert "$occupied" in job2
    assert "1:5" in job2  # For H2O ground state alpha and beta
    # More specific check for alpha and beta lines
    assert "\n1:5\n1:5\n" in job2  # Alpha occ \n Beta occ


# --- Tests for set_mom_job2_charge ---


def test_set_mom_job2_charge_success(mom_enabled_input: QchemInput) -> None:
    """Test setting specific charge for second MOM job."""
    inp = mom_enabled_input.set_mom_job2_charge(1)
    assert inp.mom_job2_charge == 1
    assert inp is not mom_enabled_input


def test_set_mom_job2_charge_before_enable(default_qchem_input: QchemInput) -> None:
    """Test calling set_mom_job2_charge before enable_mom raises ConfigurationError."""
    with pytest.raises(
        ConfigurationError,
        match=r"MOM must be enabled \(run_mom=True\) via enable_mom\(\) before setting a specific charge for the second MOM job\.",
    ):
        default_qchem_input.set_mom_job2_charge(1)


# --- Tests for set_mom_job2_spin_multiplicity ---


def test_set_mom_job2_spin_multiplicity_success(mom_enabled_input: QchemInput) -> None:
    """Test setting specific spin multiplicity for second MOM job."""
    inp = mom_enabled_input.set_mom_job2_spin_multiplicity(2)
    assert inp.mom_job2_spin_multiplicity == 2
    assert inp is not mom_enabled_input


def test_set_mom_job2_spin_multiplicity_before_enable(default_qchem_input: QchemInput) -> None:
    """Test calling set_mom_job2_spin_multiplicity before enable_mom raises ConfigurationError."""
    with pytest.raises(
        ConfigurationError,
        match=r"MOM must be enabled \(run_mom=True\) via enable_mom\(\) before setting a specific spin multiplicity for the second MOM job\.",
    ):
        default_qchem_input.set_mom_job2_spin_multiplicity(2)


def test_set_mom_job2_spin_multiplicity_invalid(mom_enabled_input: QchemInput) -> None:
    """Test setting invalid spin multiplicity raises ValidationError."""
    with pytest.raises(ValidationError, match="Spin multiplicity must be a positive integer"):
        mom_enabled_input.set_mom_job2_spin_multiplicity(0)
    
    with pytest.raises(ValidationError, match="Spin multiplicity must be a positive integer"):
        mom_enabled_input.set_mom_job2_spin_multiplicity(-1)


# --- Tests for MOM occupation validation ---


def test_mom_occupation_validation_valid_cases(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test that valid occupation strings pass validation."""
    # H2O: 10 electrons, charge=0, mult=1 -> 5 alpha, 5 beta
    inp = unrestricted_input.enable_mom().set_mom_occupation(alpha_occ="1:5", beta_occ="1:5")
    # Should not raise any exceptions
    inp._generate_occupied_block(h2o_geometry)
    
    # Core ionization example: charge=+1, mult=2 -> 4.5 alpha, 4.5 beta (rounds to 5 alpha, 4 beta)
    inp_ionized = replace(unrestricted_input, charge=1, spin_multiplicity=2)
    inp_ionized = inp_ionized.enable_mom().set_mom_occupation(alpha_occ="1:5", beta_occ="1:4")
    inp_ionized._generate_occupied_block(h2o_geometry)


def test_mom_occupation_validation_wrong_total_electrons(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test that wrong total electron count fails validation."""
    # H2O: 10 electrons, but specifying 11 electrons (6 alpha + 5 beta)
    inp = unrestricted_input.enable_mom().set_mom_occupation(alpha_occ="1:6", beta_occ="1:5")
    with pytest.raises(
        ValidationError, 
        match="Total electrons in occupation .* does not match expected .* from geometry and charge"
    ):
        inp._generate_occupied_block(h2o_geometry)


def test_mom_occupation_validation_wrong_spin_multiplicity(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test that inconsistent spin multiplicity fails validation."""
    # H2O: 10 electrons, mult=1 (should have equal alpha/beta), but 6 alpha + 4 beta
    inp = unrestricted_input.enable_mom().set_mom_occupation(alpha_occ="1:6", beta_occ="1:4")
    with pytest.raises(
        ValidationError,
        match="Spin multiplicity .* inconsistent with alpha/beta electron counts"
    ):
        inp._generate_occupied_block(h2o_geometry)


def test_mom_occupation_validation_fractional_electrons(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test that configurations requiring fractional electrons fail validation."""
    # H2O: 10 electrons, mult=4 -> would need 6.5 alpha, 3.5 beta (impossible)
    inp = replace(unrestricted_input, spin_multiplicity=4)
    inp = inp.enable_mom().set_mom_occupation(alpha_occ="1:6", beta_occ="1:4")
    with pytest.raises(
        ValidationError,
        match="Cannot achieve spin multiplicity .* with .* total electrons"
    ):
        inp._generate_occupied_block(h2o_geometry)


def test_core_ionization_beta_hole_example(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test core ionization with proper beta hole setup."""
    # Simulate removing core electron from beta channel
    # H2O: 10 electrons -> 9 electrons (charge +1), mult=2
    # Expected: 5 alpha, 4 beta electrons
    inp = replace(unrestricted_input, charge=1, spin_multiplicity=2)
    inp = inp.enable_mom().set_mom_occupation(
        alpha_occ="1:5",      # Keep all alpha electrons 
        beta_occ="1:4"        # Remove one beta electron (core hole)
    )
    
    block = inp._generate_occupied_block(h2o_geometry)
    expected = """$occupied
1:5
1:4
$end"""
    assert block == expected


def test_core_ionization_alpha_hole_invalid(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test that putting core hole in alpha channel with mult=2 fails validation."""
    # H2O: 10 electrons -> 9 electrons (charge +1), mult=2
    # Wrong: 4 alpha, 5 beta (would give mult=0, not 2)
    inp = replace(unrestricted_input, charge=1, spin_multiplicity=2)
    inp = inp.enable_mom().set_mom_occupation(
        alpha_occ="1:4",      # Wrong: hole in alpha
        beta_occ="1:5"        # Wrong: keep all beta
    )
    
    with pytest.raises(
        ValidationError,
        match="Spin multiplicity 2 inconsistent with alpha/beta electron counts"
    ):
        inp._generate_occupied_block(h2o_geometry)


def test_export_input_file_mom_job2_different_charge_and_multiplicity(h2o_geometry: Geometry, unrestricted_input: QchemInput) -> None:
    """Test exporting MOM input with different charge and spin multiplicity for job2."""
    # Job 1: neutral H2O (charge=0, mult=1)
    # Job 2: cation radical H2O+ (charge=+1, mult=2) with beta hole
    inp = unrestricted_input.enable_mom()
    inp = inp.set_mom_job2_charge(1).set_mom_job2_spin_multiplicity(2)
    inp = inp.set_mom_occupation(alpha_occ="1:5", beta_occ="1:4")
    
    output = inp.export_input_file(h2o_geometry)
    jobs = output.split("\n\n@@@\n\n")
    assert len(jobs) == 2
    job1, job2 = jobs
    
    # Job 1: neutral system
    assert "0 1" in job1  # charge=0, mult=1
    
    # Job 2: ionized system  
    assert "1 2" in job2  # charge=+1, mult=2
    assert "$occupied" in job2
    assert "1:5" in job2  # alpha electrons
    assert "1:4" in job2  # beta electrons (one less due to hole)


def test_count_electrons_in_occupation_string() -> None:
    """Test the helper function for counting electrons in Q-Chem occupation strings."""
    from calcflow.inputs.qchem import _count_electrons_in_qchem_occupation
    
    assert _count_electrons_in_qchem_occupation("1:5") == 5
    assert _count_electrons_in_qchem_occupation("1:4 6") == 5
    assert _count_electrons_in_qchem_occupation("1:3 5:7 9") == 7
    assert _count_electrons_in_qchem_occupation("1") == 1
    assert _count_electrons_in_qchem_occupation("5 7 9") == 3
    assert _count_electrons_in_qchem_occupation("") == 0


@pytest.mark.parametrize(
    "n_electrons, multiplicity, expected_alpha, expected_beta, should_fail",
    [
        # Valid cases
        (10, 1, 5, 5, False),    # Closed shell singlet
        (9, 2, 5, 4, False),     # Doublet (one unpaired electron)
        (8, 1, 4, 4, False),     # Closed shell singlet
        (8, 3, 5, 3, False),     # Triplet (two unpaired electrons)
        (11, 2, 6, 5, False),    # Doublet
        (11, 4, 7, 4, False),    # Quartet (three unpaired electrons)
        (9, 4, 6, 3, False),     # Quartet with 9 electrons
        (5, 6, 5, 0, False),     # Sextet with 5 electrons
        
        # Invalid cases (fractional electrons)
        (9, 1, None, None, True),   # Odd electrons can't be singlet
        (8, 2, None, None, True),   # Even electrons with doublet impossible
        (10, 6, None, None, True),  # Would need fractional electrons
    ],
)
def test_calculate_expected_electron_distribution(
    n_electrons: int, 
    multiplicity: int, 
    expected_alpha: int | None, 
    expected_beta: int | None, 
    should_fail: bool
) -> None:
    """Test calculation of expected alpha/beta electron distribution."""
    from calcflow.inputs.qchem import _calculate_expected_electron_distribution
    
    if should_fail:
        with pytest.raises(ValidationError):
            _calculate_expected_electron_distribution(n_electrons, multiplicity)
    else:
        alpha, beta = _calculate_expected_electron_distribution(n_electrons, multiplicity)
        assert alpha == expected_alpha
        assert beta == expected_beta
        assert alpha + beta == n_electrons
        assert alpha - beta == multiplicity - 1


# --- Tests for transition validation with new features ---


def test_set_mom_transition_spin_specification_validation(mom_enabled_input: QchemInput) -> None:
    """Test validation of spin specifications in transitions."""
    # Valid spin specifications
    inp = mom_enabled_input.set_mom_transition("5(alpha)->7")
    assert inp.mom_transition == "5(alpha)->7"
    
    inp = mom_enabled_input.set_mom_transition("HOMO(beta)->LUMO")
    assert inp.mom_transition == "HOMO(beta)->LUMO"
    
    # Invalid spin specifications should raise ValidationError
    with pytest.raises(ValidationError, match="Invalid spin specification"):
        mom_enabled_input.set_mom_transition("5(gamma)->7")


def test_set_mom_transition_complex_combinations(mom_enabled_input: QchemInput) -> None:
    """Test complex transition combinations."""
    # Multiple excitations
    inp = mom_enabled_input.set_mom_transition("HOMO->LUMO; HOMO-1->LUMO+1")
    assert inp.mom_transition == "HOMO->LUMO; HOMO-1->LUMO+1"
    
    # Excitation plus ionization
    inp = mom_enabled_input.set_mom_transition("HOMO->LUMO; 3(beta)->vac")
    assert inp.mom_transition == "HOMO->LUMO; 3(beta)->vac"
    
    # Multiple ionizations
    inp = mom_enabled_input.set_mom_transition("2->vac; 5(beta)->vac")
    assert inp.mom_transition == "2->vac; 5(beta)->vac"


def test_physical_validation_in_transitions(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test that transitions respect physical constraints."""
    # Try to excite from unoccupied orbital (should fail during generation)
    inp = unrestricted_input.enable_mom().set_mom_transition("7->8")  # 7 is LUMO for H2O
    with pytest.raises(ValidationError, match="Source orbital 7 must be occupied"):
        inp._generate_occupied_block(h2o_geometry)
    
    # Try to excite to occupied orbital (should fail during generation)
    inp = unrestricted_input.enable_mom().set_mom_transition("4->3")  # Both occupied for H2O
    with pytest.raises(ValidationError, match="Target orbital 3 must be unoccupied"):
        inp._generate_occupied_block(h2o_geometry)


def test_ionization_validation_physics(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test physical validation for ionization operations."""
    # Try to ionize from unoccupied orbital
    inp = unrestricted_input.enable_mom().set_mom_transition("7->vac")  # 7 is LUMO for H2O
    with pytest.raises(ValidationError, match="Cannot ionize from unoccupied orbital"):
        inp._generate_occupied_block(h2o_geometry)
    
    # Valid ionization should work
    inp = unrestricted_input.enable_mom().set_mom_transition("3->vac")  # 3 is occupied for H2O
    inp = inp.set_mom_job2_charge(1).set_mom_job2_spin_multiplicity(2)  # +1 charge, doublet after ionization
    block = inp._generate_occupied_block(h2o_geometry)
    assert "$occupied" in block


def test_generate_occupied_block_job2_charge_exceeds_nuclear(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test _generate_occupied_block when job2 charge makes electron count negative (exceeds nuclear)."""
    # H2O has nuclear charge 10. Setting charge to 11 means 10 - 11 = -1 electrons.
    inp = unrestricted_input.enable_mom().set_mom_ground_state() # Use ground state to trigger occupation generation
    inp = inp.set_mom_job2_charge(11)

    with pytest.raises(ConfigurationError, match=r"Invalid total electron count \(-1\) for 'GROUND_STATE' MOM"):
        inp._generate_occupied_block(h2o_geometry)


def test_apply_ionization_unoccupied_beta(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test that _apply_ionization raises error if trying to ionize unoccupied beta orbital."""
    # H2O has 10 electrons, HOMO=5. Orbital 7 is unoccupied.
    inp = unrestricted_input.enable_mom().set_mom_transition("7(beta)->vac") # Try to ionize unoccupied beta 7
    
    # _generate_occupied_block calls _convert_extended_transitions_to_occupations which calls _apply_single_operation
    # which calls _apply_ionization, so we test via the block generation.
    with pytest.raises(ValidationError, match="Cannot ionize from unoccupied beta orbital 7"):
        inp._generate_occupied_block(h2o_geometry)


def test_apply_excitation_unoccupied_beta_source(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test that _apply_excitation raises error if source beta orbital is unoccupied."""
    # H2O has 10 electrons, HOMO=5. Orbital 6 is LUMO.
    # Exciting from occupied alpha 5 to unoccupied alpha 6 is fine (tested elsewhere).
    # Exciting from unoccupied beta 6 to unoccupied beta 7 should fail.
    inp = unrestricted_input.enable_mom().set_mom_transition("6(beta)->7(beta)") # Try to excite from unoccupied beta 6

    # _generate_occupied_block calls _convert_extended_transitions_to_occupations which calls _apply_single_operation
    # which calls _apply_excitation, so we test via the block generation.
    with pytest.raises(ValidationError, match="Source orbital 6 must be occupied \(HOMO=5\)"):
        inp._generate_occupied_block(h2o_geometry)


def test_apply_excitation_unoccupied_beta_source_sequential(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test _apply_excitation for beta channel when source orbital becomes unoccupied mid-transition.
    This test targets the internal validation within _apply_excitation, not the
    primary validation in _validate_mom_occupations based on initial state.
    """
    # H2O: 5 beta electrons (1,2,3,4,5). Orbitals 6+ are initially unoccupied.
    # 1(b)->vac makes orbital 1(b) empty.
    # Then 1(b)->6(b) attempts to excite from the now-empty 1(b) to an initially empty 6(b).
    inp = unrestricted_input.enable_mom().set_mom_transition("1(beta)->vac; 1(beta)->6(beta)")
    # Must adjust charge/multiplicity for the ionization part to be valid overall,
    # so the error we are testing is not preempted by total electron count issues.
    inp = inp.set_mom_job2_charge(1).set_mom_job2_spin_multiplicity(2)

    # This error comes from _apply_excitation's internal check.
    with pytest.raises(ValidationError, match="Cannot excite from unoccupied beta orbital 1"):
        inp._generate_occupied_block(h2o_geometry)


def test_apply_excitation_unoccupied_alpha_source_sequential(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test _apply_excitation for alpha channel when source orbital becomes unoccupied mid-transition."""
    # H2O: 5 alpha electrons (1,2,3,4,5). Orbitals 6+ are initially unoccupied.
    # 1(a)->vac makes orbital 1(a) empty.
    # Then 1(a)->6(a) attempts to excite from the now-empty 1(a) to an initially empty 6(a).
    inp = unrestricted_input.enable_mom().set_mom_transition("1(alpha)->vac; 1(alpha)->6(alpha)")
    inp = inp.set_mom_job2_charge(1).set_mom_job2_spin_multiplicity(2) # Accommodate ionization

    with pytest.raises(ValidationError, match="Cannot excite from unoccupied alpha orbital 1"):
        inp._generate_occupied_block(h2o_geometry)


def test_apply_ionization_unoccupied_alpha_source_sequential(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test _apply_ionization for alpha channel when source orbital becomes unoccupied mid-transition."""
    # H2O: 5 alpha electrons (1,2,3,4,5). Orbitals 6+ are initially unoccupied.
    # 1(a)->vac makes orbital 1(a) empty.
    # Then 1(a)->vac attempts to ionize again from the now-empty 1(a).
    inp = unrestricted_input.enable_mom().set_mom_transition("1(alpha)->vac; 1(alpha)->vac")
    # Charge and multiplicity for two ionizations from H2O (10e) -> H2O++ (8e)
    # If singlet, 8e -> mult=1. If triplet, mult=3.
    # The first 1(a)->vac leads to 9e system (mult=2, e.g. 5a, 4b or 4a, 5b).
    # The second 1(a)->vac would lead to 8e system.
    # Let's set target for two ionizations leading to a singlet.
    inp = inp.set_mom_job2_charge(2).set_mom_job2_spin_multiplicity(1)


    with pytest.raises(ValidationError, match="Cannot ionize from unoccupied alpha orbital 1"):
        inp._generate_occupied_block(h2o_geometry)


def test_apply_excitation_unoccupied_beta_source_previously_emptied(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test excitation from beta orbital that was emptied by a previous operation."""
    # Initial state: H2O, 5 beta electrons (1,2,3,4,5). Orbital 6 is initially unoccupied.
    # Operation 1: Ionize from beta orbital 1 (1(beta)->vac). Beta occupied: {2,3,4,5}
    # Operation 2: Attempt to excite from beta orbital 1 (now empty) to beta orbital 6.
    inp = unrestricted_input.enable_mom().set_mom_transition("1(beta)->vac; 1(beta)->6(beta)")
    inp = inp.set_mom_job2_charge(1).set_mom_job2_spin_multiplicity(2) # For the single ionization
    with pytest.raises(ValidationError, match="Cannot excite from unoccupied beta orbital 1"):
        inp._generate_occupied_block(h2o_geometry)

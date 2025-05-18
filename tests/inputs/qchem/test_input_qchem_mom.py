import re
from dataclasses import replace

import pytest

from calcflow.exceptions import ConfigurationError, NotSupportedError, ValidationError
from calcflow.geometry.static import Geometry
from calcflow.inputs.qchem import QchemInput, _convert_transition_to_occupations


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
    assert inp_offset.mom_transition == "HOMO-1->LUMO+2"

    inp_integer = mom_enabled_input.set_mom_transition("5 -> 7")
    assert inp_integer.mom_transition == "5->7"

    inp_mixed = mom_enabled_input.set_mom_transition("HOMO-2 -> 8")
    assert inp_mixed.mom_transition == "HOMO-2->8"

    inp_case = mom_enabled_input.set_mom_transition("homo->lumo+1")
    assert inp_case.mom_transition == "homo->lumo+1"


def test_set_mom_transition_before_enable(default_qchem_input: QchemInput) -> None:
    """Test setting MOM transition before enabling MOM raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="MOM must be enabled first"):
        default_qchem_input.set_mom_transition("HOMO->LUMO")


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
        ("5->0", "Invalid target orbital: '0'. Must be HOMO[-n], LUMO[+n], or positive integer."),
        ("0->6", "Invalid source orbital: '0'. Must be HOMO[-n], LUMO[+n], or positive integer."),
        ("-1->6", "Invalid source orbital: '-1'. Must be HOMO[-n], LUMO[+n], or positive integer."),
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


def test_generate_occupied_block_non_singlet(mom_enabled_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test error if MOM enabled for non-closed-shell-singlet reference."""
    inp_charged = replace(mom_enabled_input, charge=1, unrestricted=True).set_mom_transition("HOMO->LUMO")
    with pytest.raises(NotSupportedError, match="MOM currently only supports closed-shell singlet reference states"):
        inp_charged._generate_occupied_block(h2o_geometry)

    inp_triplet = replace(mom_enabled_input, spin_multiplicity=3, unrestricted=True).set_mom_transition("HOMO->LUMO")
    with pytest.raises(NotSupportedError, match="MOM currently only supports closed-shell singlet reference states"):
        inp_triplet._generate_occupied_block(h2o_geometry)


def test_generate_occupied_block_no_occupation_set(unrestricted_input: QchemInput, h2o_geometry: Geometry) -> None:
    """Test error if MOM enabled but neither transition nor direct occupation is set."""
    inp = unrestricted_input.enable_mom()  # MOM enabled, unrestricted, singlet, but no occupation details
    with pytest.raises(
        ConfigurationError, match="Either mom_transition or both mom_alpha_occ and mom_beta_occ must be set"
    ):
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


# Generate the expected ALPHA string for HOMO-5 -> LUMO with 156 electrons (HOMO=78)
# Occupied set becomes {1..72} U {74..78} U {79}
# Compressed: 1:72 74:78 79
homo_minus_5_lumo_156_alpha_occ = "1:72 74:79"


# Define test cases using parametrize
# Format: (n_electrons, transition, expected_alpha, expected_beta)
valid_cases = [
    # --- Basic cases (10 electrons: HOMO=5, LUMO=6) ---
    (10, "HOMO->LUMO", "1:4 6", "1:5"),
    (10, "homo->lumo", "1:4 6", "1:5"),  # Test case insensitivity
    (10, " HOMO -> LUMO ", "1:4 6", "1:5"),  # Test whitespace robustness
    (10, "HOMO-1->LUMO", "1:3 5:6", "1:5"),  # Updated expected alpha
    (10, "HOMO->LUMO+1", "1:4 7", "1:5"),
    (10, "HOMO-2->LUMO+2", "1:2 4:5 8", "1:5"),  # Updated expected alpha
    # --- More complex cases (156 electrons: HOMO=78, LUMO=79) ---
    (156, "HOMO->LUMO", "1:77 79", "1:78"),
    (
        156,
        "HOMO-5->LUMO",
        homo_minus_5_lumo_156_alpha_occ,  # Corrected alpha
        "1:78",  # Beta is ground state
    ),
    (156, "HOMO->LUMO+10", "1:77 89", "1:78"),
    # --- Edge case (2 electrons: HOMO=1, LUMO=2) ---
    (2, "HOMO->LUMO", "2", "1"),
    # --- Explicit Integer Indices (156 electrons: HOMO=78, LUMO=79) ---
    (156, "6->LUMO", "1:5 7:79", "1:78"),  # Int source, LUMO target
    (156, "6 -> 79", "1:5 7:79", "1:78"),  # Int source, Int target (with space)
    (156, "HOMO->79", "1:77 79", "1:78"),  # HOMO source, Int target
    (156, "78->LUMO", "1:77 79", "1:78"),  # Int(HOMO) source, LUMO target
]

# Define test cases for invalid transitions
invalid_transition_cases = [
    # --- Invalid Targets (Target <= HOMO) ---
    (156, "HOMO->HOMO", "Target orbital must be unoccupied"),
    (156, "HOMO->78", "Target orbital must be unoccupied"),
    (156, "6->HOMO", "Target orbital must be unoccupied"),
    (156, "6->77", "Target orbital must be unoccupied"),
    # --- Invalid Sources (Source > HOMO) ---
    (156, "LUMO->LUMO+1", "Source orbital must be occupied"),
    (156, "LUMO->80", "Source orbital must be occupied"),
    (156, "79->LUMO+1", "Source orbital must be occupied"),
    (156, "80->81", "Source orbital must be occupied"),
    # --- Non-positive indices ---
    (10, "HOMO-5 -> LUMO", "Calculated source orbital index must be positive, got 0"),  # HOMO=5, HOMO-5=0
    (10, "5 -> 0", "Target orbital index must be positive, got '0'"),
    (10, "0 -> 6", "Source orbital index must be positive, got '0'"),
    # Negative indices are caught by set_mom_transition format check
]


@pytest.mark.parametrize("n_electrons, transition, expected_alpha, expected_beta", valid_cases)
def test_convert_transition_to_occupations_valid(
    n_electrons: int, transition: str, expected_alpha: str, expected_beta: str
) -> None:
    """Tests valid transition string conversions produce correct occupation strings."""
    alpha_occ, beta_occ = _convert_transition_to_occupations(transition, n_electrons)
    assert alpha_occ == expected_alpha
    assert beta_occ == expected_beta


def test_convert_transition_to_occupations_odd_electrons() -> None:
    """Tests that an odd number of electrons raises a ValidationError."""
    with pytest.raises(ValidationError, match="Expected even number of electrons"):
        _convert_transition_to_occupations("HOMO->LUMO", 11)


@pytest.mark.parametrize("n_electrons, transition, error_match", invalid_transition_cases)
def test_convert_transition_to_occupations_invalid_transitions(
    n_electrons: int, transition: str, error_match: str
) -> None:
    """Tests that invalid transitions (e.g., to occupied, from virtual) raise ValidationError."""
    with pytest.raises(ValidationError, match=error_match):
        _convert_transition_to_occupations(transition, n_electrons)


# Note: Validation errors for malformed transitions like "HOMO+1->LUMO" or
# "HOMO->LUMO-1" or invalid characters are expected to be caught by the
# calling function (set_mom_transition) before this helper is invoked.
# Therefore, we only test the core conversion logic and the electron count check here.

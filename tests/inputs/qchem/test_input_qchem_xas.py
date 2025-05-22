import logging

import pytest

from calcflow.exceptions import ConfigurationError, ValidationError
from calcflow.geometry.static import Geometry
from calcflow.inputs.qchem import QchemInput
from calcflow.utils import logger

logger.setLevel(logging.CRITICAL)


# Helper function for normalizing block content for comparison
def _normalize_block_content(content: str) -> str:
    return "\n".join(line.strip() for line in content.strip().split("\n"))


def test_set_reduced_excitation_space_valid(default_qchem_input: QchemInput) -> None:
    """Test setting valid reduced excitation space parameters."""
    # Enable TDDFT first
    inp_tddft = default_qchem_input.set_tddft(nroots=5)
    orbitals = [10, 8, 12, 8]  # Unsorted, with duplicates
    expected_orbitals = [8, 10, 12]

    inp_reduced = inp_tddft.set_reduced_excitation_space(initial_orbitals=orbitals)

    assert inp_reduced.run_tddft
    assert inp_reduced.reduced_excitation_space
    assert inp_reduced.initial_orbitals == expected_orbitals
    assert inp_reduced is not inp_tddft  # Immutability


def test_set_reduced_excitation_space_tddft_not_enabled(default_qchem_input: QchemInput) -> None:
    """Test error if setting reduced excitation space before TDDFT is enabled."""
    with pytest.raises(ConfigurationError, match="Reduced excitation space .TRNSS. requires TDDFT to be enabled"):
        default_qchem_input.set_reduced_excitation_space(initial_orbitals=[1, 2, 3])


def test_set_reduced_excitation_space_empty_orbitals(default_qchem_input: QchemInput) -> None:
    """Test error if initial_orbitals list is empty."""
    inp_tddft = default_qchem_input.set_tddft(nroots=3)
    with pytest.raises(ValidationError, match="The initial_orbitals list cannot be empty"):
        inp_tddft.set_reduced_excitation_space(initial_orbitals=[])


@pytest.mark.parametrize(
    "invalid_orbitals, error_msg",
    [
        ([1, 0, 2], "All initial_orbitals must be positive integers."),
        ([1, -5, 2], "All initial_orbitals must be positive integers."),
        ([1, 2.5, 3], "All initial_orbitals must be positive integers."),
        (["a", 2, 3], "All initial_orbitals must be positive integers."),
    ],
)
def test_set_reduced_excitation_space_invalid_orbital_values(
    default_qchem_input: QchemInput, invalid_orbitals: list, error_msg: str
) -> None:
    """Test error if initial_orbitals list contains non-positive or non-integer values."""
    inp_tddft = default_qchem_input.set_tddft(nroots=3)
    with pytest.raises(ValidationError, match=error_msg):
        inp_tddft.set_reduced_excitation_space(initial_orbitals=invalid_orbitals)


def test_get_rem_block_tddft_reduced_excitation(helpers, default_qchem_input: QchemInput) -> None:
    """Test $rem block with TDDFT and reduced excitation space."""
    inp = default_qchem_input.set_tddft(nroots=5).set_reduced_excitation_space(initial_orbitals=[8, 10, 11])

    expected_rem_dict = {
        # Base from default_qchem_input
        "METHOD": "hf",
        "BASIS": "sto-3g",
        "JOBTYPE": "sp",
        "UNRESTRICTED": "False",
        "SYMMETRY": "False",
        "SYM_IGNORE": "True",
        # TDDFT params
        "CIS_N_ROOTS": "5",
        "CIS_SINGLETS": "True",  # Default for set_tddft
        "CIS_TRIPLETS": "False",  # Default for set_tddft
        "STATE_ANALYSIS": "True",  # Default for set_tddft
        # TRNSS params
        "TRNSS": "True",
        "TRTYPE": "3",
        "N_SOL": "3",
    }

    actual_block_full_string = inp._get_rem_block()
    actual_block_content_lines = actual_block_full_string.strip().split("\n")
    if len(actual_block_content_lines) < 2:  # Should have $rem and $end
        raise AssertionError(f"REM block is malformed or too short: {actual_block_full_string}")

    actual_block_inner_content = "\n".join(actual_block_content_lines[1:-1])
    actual_rem_dict = helpers.parse_qchem_rem_section(actual_block_inner_content)

    assert actual_rem_dict == expected_rem_dict


def test_get_solute_block_active(default_qchem_input: QchemInput) -> None:
    """Test $solute block generation when TRNSS is active."""
    inp = default_qchem_input.set_tddft(nroots=3).set_reduced_excitation_space(initial_orbitals=[7, 8, 9])
    expected_block = """$solute
7 8 9
$end"""
    assert inp._get_solute_block() == expected_block


def test_get_solute_block_inactive(default_qchem_input: QchemInput) -> None:
    """Test $solute block is empty when TRNSS is not active."""
    assert default_qchem_input._get_solute_block() == ""  # TRNSS not set
    inp_tddft_only = default_qchem_input.set_tddft(nroots=5)
    assert inp_tddft_only._get_solute_block() == ""  # TDDFT enabled, but not TRNSS


def test_export_input_file_tddft_reduced_excitation(
    helpers, h2o_geometry: Geometry, default_qchem_input: QchemInput
) -> None:
    """Test exporting an input file with TDDFT and reduced excitation space."""
    inp = (
        default_qchem_input.set_tddft(nroots=3)
        .set_reduced_excitation_space(initial_orbitals=[4, 5])
        .set_basis("def2-svp")  # Use a slightly more realistic basis
    )

    actual_output_str = inp.export_input_file(h2o_geometry)
    actual_blocks = helpers.split_qchem_input_into_blocks(actual_output_str)

    # Check $molecule block
    expected_molecule_content = _normalize_block_content("""
0 1
O        0.00000000      0.00000000      0.11730000
H        0.00000000      0.75720000     -0.46920000
H        0.00000000     -0.75720000     -0.46920000
    """)
    assert "molecule" in actual_blocks
    assert _normalize_block_content(actual_blocks["molecule"]) == expected_molecule_content

    # Check $rem block
    expected_rem_params = {
        "METHOD": "hf",
        "BASIS": "def2-svp",
        "JOBTYPE": "sp",
        "UNRESTRICTED": "False",
        "SYMMETRY": "False",
        "SYM_IGNORE": "True",
        "CIS_N_ROOTS": "3",
        "CIS_SINGLETS": "True",
        "CIS_TRIPLETS": "False",
        "STATE_ANALYSIS": "True",
        "TRNSS": "True",
        "TRTYPE": "3",
        "N_SOL": "2",
    }
    assert "rem" in actual_blocks
    actual_rem_params = helpers.parse_qchem_rem_section(actual_blocks["rem"])
    assert actual_rem_params == expected_rem_params

    # Check $solute block
    expected_solute_content = _normalize_block_content("""
4 5
    """)
    assert "solute" in actual_blocks
    assert _normalize_block_content(actual_blocks["solute"]) == expected_solute_content

    # Check block order (basic)
    assert list(actual_blocks.keys()) == ["molecule", "rem", "solute"]


def test_export_input_file_mom_tddft_reduced_excitation(
    helpers, h2o_geometry: Geometry, unrestricted_input: QchemInput
) -> None:
    """Test exporting MOM with TDDFT (reduced excitation) in the second job."""
    inp = (
        unrestricted_input.enable_mom()
        .set_mom_transition("HOMO->LUMO")  # Sets up MOM for H2O (5 alpha, 5 beta)
        .set_tddft(nroots=2)  # Enable TDDFT for the second job
        .set_reduced_excitation_space(initial_orbitals=[4, 5])  # Reduce excitation space for TDDFT
        .set_basis("def2-tzvp")  # Use a different basis for clarity
    )

    output_str = inp.export_input_file(h2o_geometry)
    job_strs = output_str.split("\n\n@@@\n\n")
    assert len(job_strs) == 2
    job1_str, job2_str = job_strs

    # --- Check Job 1 (standard unrestricted calc) ---
    job1_blocks = helpers.split_qchem_input_into_blocks(job1_str)

    # $molecule block - Job 1
    expected_mol_j1_content = _normalize_block_content(f"""
{unrestricted_input.charge} {unrestricted_input.spin_multiplicity}
{h2o_geometry.get_coordinate_block().strip()}
    """)
    assert "molecule" in job1_blocks
    assert _normalize_block_content(job1_blocks["molecule"]) == expected_mol_j1_content

    # $rem block - Job 1
    expected_rem_j1 = {
        "METHOD": unrestricted_input.level_of_theory,  # hf from fixture
        "BASIS": "def2-tzvp",  # Overridden by .set_basis()
        "JOBTYPE": "sp",  # Default task for unrestricted_input if not MOM-opt
        "UNRESTRICTED": "True",
        "SYMMETRY": "False",
        "SYM_IGNORE": "True",
        # No MOM, TDDFT, or TRNSS in job 1
    }
    assert "rem" in job1_blocks
    actual_rem_j1 = helpers.parse_qchem_rem_section(job1_blocks["rem"])
    assert actual_rem_j1 == expected_rem_j1
    assert "solute" not in job1_blocks
    assert "occupied" not in job1_blocks
    assert list(job1_blocks.keys()) == ["molecule", "rem"]  # Expected order for Job 1

    # --- Check Job 2 (MOM + TDDFT with reduced excitation) ---
    job2_blocks = helpers.split_qchem_input_into_blocks(job2_str)

    # $molecule block - Job 2
    expected_mol_j2_content = _normalize_block_content("read")
    assert "molecule" in job2_blocks
    assert _normalize_block_content(job2_blocks["molecule"]) == expected_mol_j2_content

    # $rem block - Job 2
    expected_rem_j2 = {
        "METHOD": unrestricted_input.level_of_theory,  # hf from fixture
        "BASIS": "def2-tzvp",  # Carried over
        "JOBTYPE": "sp",  # Forced for MOM excited state part
        "UNRESTRICTED": "True",  # Forced for MOM
        "SYMMETRY": "False",
        "SYM_IGNORE": "True",
        "SCF_GUESS": "read",
        "MOM_START": "1",
        "MOM_METHOD": "IMOM",  # Default from enable_mom()
        "CIS_N_ROOTS": "2",
        "CIS_SINGLETS": "True",
        "CIS_TRIPLETS": "False",
        "STATE_ANALYSIS": "True",
        "TRNSS": "True",
        "TRTYPE": "3",
        "N_SOL": "2",
    }
    assert "rem" in job2_blocks
    actual_rem_j2 = helpers.parse_qchem_rem_section(job2_blocks["rem"])
    assert actual_rem_j2 == expected_rem_j2

    # $occupied block - Job 2
    # H2O: 10 electrons -> 5 alpha, 5 beta. HOMO=5. MOM: HOMO->LUMO (5->6)
    # Alpha: 1:4 6, Beta: 1:5
    expected_occupied_j2_content = _normalize_block_content("""
1:4 6
1:5
    """)
    assert "occupied" in job2_blocks
    assert _normalize_block_content(job2_blocks["occupied"]) == expected_occupied_j2_content

    # $solute block - Job 2
    expected_solute_j2_content = _normalize_block_content("""
4 5
    """)
    assert "solute" in job2_blocks
    assert _normalize_block_content(job2_blocks["solute"]) == expected_solute_j2_content

    # Check block order for Job 2
    assert list(job2_blocks.keys()) == ["molecule", "rem", "occupied", "solute"]

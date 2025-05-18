import logging
from dataclasses import replace

import pytest

from calcflow.exceptions import NotSupportedError, ValidationError
from calcflow.geometry.static import Geometry
from calcflow.inputs.qchem import QchemInput
from calcflow.utils import logger

logger.setLevel(logging.CRITICAL)


def test_get_molecule_block(h2o_geometry: Geometry) -> None:
    """Test the generation of the $molecule block."""
    inp = QchemInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="hf",
        basis_set="sto-3g",
    )
    geom_str = h2o_geometry.get_coordinate_block()
    expected_block = """$molecule
0 1
O        0.00000000      0.00000000      0.11730000
H        0.00000000      0.75720000     -0.46920000
H        0.00000000     -0.75720000     -0.46920000
$end"""
    assert inp._get_molecule_block(geom_str) == expected_block

    inp_charged = QchemInput(
        charge=1,
        spin_multiplicity=2,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g*",
    )
    expected_charged_block = """$molecule
1 2
O        0.00000000      0.00000000      0.11730000
H        0.00000000      0.75720000     -0.46920000
H        0.00000000     -0.75720000     -0.46920000
$end"""
    assert inp_charged._get_molecule_block(geom_str) == expected_charged_block


def test_get_rem_block_minimal(helpers, default_qchem_input: QchemInput) -> None:
    """Test minimal $rem block generation (energy, HF)."""
    inp = replace(default_qchem_input)
    expected_dict = {
        "METHOD": "hf",
        "BASIS": "sto-3g",
        "JOBTYPE": "sp",
        "UNRESTRICTED": "False",
        "SYMMETRY": "False",
        "SYM_IGNORE": "True",
    }
    actual_block_full_string = inp._get_rem_block()
    actual_block_content_lines = actual_block_full_string.strip().split("\n")
    if len(actual_block_content_lines) < 2:
        pytest.fail(f"REM block is too short: {actual_block_full_string}")
    actual_block_inner_content = "\n".join(actual_block_content_lines[1:-1])
    actual_dict = helpers.parse_qchem_rem_section(actual_block_inner_content)
    assert actual_dict == expected_dict


def test_get_rem_block_opt_dft_unrestricted(helpers, default_qchem_input: QchemInput) -> None:
    """Test $rem block for geometry optimization with DFT and unrestricted."""
    inp = replace(
        default_qchem_input,
        task="geometry",
        level_of_theory="b3lyp",
        basis_set="def2-svp",
        unrestricted=True,
        spin_multiplicity=2,  # Consistent with unrestricted=True
    )
    expected_dict = {
        "METHOD": "b3lyp",
        "BASIS": "def2-svp",
        "JOBTYPE": "opt",
        "UNRESTRICTED": "True",
        "SYMMETRY": "False",
        "SYM_IGNORE": "True",
    }
    actual_block_full_string = inp._get_rem_block()
    actual_block_content_lines = actual_block_full_string.strip().split("\n")
    if len(actual_block_content_lines) < 2:
        pytest.fail(f"REM block is too short: {actual_block_full_string}")
    actual_block_inner_content = "\n".join(actual_block_content_lines[1:-1])
    actual_dict = helpers.parse_qchem_rem_section(actual_block_inner_content)
    assert actual_dict == expected_dict


def test_get_rem_block_tddft(helpers, default_qchem_input: QchemInput) -> None:
    """Test $rem block generation with TDDFT enabled."""
    inp = default_qchem_input.set_tddft(nroots=5, triplets=True, state_analysis=False)
    expected_dict = {
        "METHOD": "hf",
        "BASIS": "sto-3g",
        "JOBTYPE": "sp",
        "UNRESTRICTED": "False",
        "SYMMETRY": "False",
        "SYM_IGNORE": "True",
        "CIS_N_ROOTS": "5",
        "CIS_SINGLETS": "True",
        "CIS_TRIPLETS": "True",
        "STATE_ANALYSIS": "False",
    }
    actual_block_full_string = inp._get_rem_block()
    actual_block_content_lines = actual_block_full_string.strip().split("\n")
    if len(actual_block_content_lines) < 2:
        pytest.fail(f"REM block is too short: {actual_block_full_string}")
    actual_block_inner_content = "\n".join(actual_block_content_lines[1:-1])
    actual_dict = helpers.parse_qchem_rem_section(actual_block_inner_content)
    assert actual_dict == expected_dict


def test_get_rem_block_tddft_rpa(helpers, default_qchem_input: QchemInput) -> None:
    """Test $rem block generation with TDDFT and RPA enabled."""
    inp = default_qchem_input.set_tddft(nroots=3).set_rpa(enable=True)
    expected_dict = {
        "METHOD": "hf",
        "BASIS": "sto-3g",
        "JOBTYPE": "sp",
        "UNRESTRICTED": "False",
        "SYMMETRY": "False",
        "SYM_IGNORE": "True",
        "CIS_N_ROOTS": "3",
        "CIS_SINGLETS": "True",
        "CIS_TRIPLETS": "False",
        "STATE_ANALYSIS": "True",
        "RPA": "True",
    }
    actual_block_full_string = inp._get_rem_block()
    actual_block_content_lines = actual_block_full_string.strip().split("\n")
    if len(actual_block_content_lines) < 2:
        pytest.fail(f"REM block is too short: {actual_block_full_string}")
    actual_block_inner_content = "\n".join(actual_block_content_lines[1:-1])
    actual_dict = helpers.parse_qchem_rem_section(actual_block_inner_content)
    assert actual_dict == expected_dict


def test_get_rem_block_solvation(helpers, default_qchem_input: QchemInput) -> None:
    """Test $rem block generation with implicit solvation."""
    inp = default_qchem_input.set_solvation(model="pcm", solvent="water")
    expected_dict = {
        "METHOD": "hf",
        "BASIS": "sto-3g",
        "JOBTYPE": "sp",
        "UNRESTRICTED": "False",
        "SYMMETRY": "False",
        "SYM_IGNORE": "True",
        "SOLVENT_METHOD": "pcm",
    }
    actual_block_full_string = inp._get_rem_block()
    actual_block_content_lines = actual_block_full_string.strip().split("\n")
    if len(actual_block_content_lines) < 2:
        pytest.fail(f"REM block is too short: {actual_block_full_string}")
    actual_block_inner_content = "\n".join(actual_block_content_lines[1:-1])
    actual_dict = helpers.parse_qchem_rem_section(actual_block_inner_content)
    assert actual_dict == expected_dict


def test_get_rem_block_dict_basis(helpers, default_qchem_input: QchemInput) -> None:
    """Test $rem block uses BASIS=gen for dictionary basis sets."""
    inp = default_qchem_input.set_basis({"O": "6-31g*", "H": "sto-3g"})
    expected_dict = {
        "METHOD": "hf",
        "BASIS": "gen",
        "JOBTYPE": "sp",
        "UNRESTRICTED": "False",
        "SYMMETRY": "False",
        "SYM_IGNORE": "True",
    }
    actual_block_full_string = inp._get_rem_block()
    actual_block_content_lines = actual_block_full_string.strip().split("\n")
    if len(actual_block_content_lines) < 2:
        pytest.fail(f"REM block is too short: {actual_block_full_string}")
    actual_block_inner_content = "\n".join(actual_block_content_lines[1:-1])
    actual_dict = helpers.parse_qchem_rem_section(actual_block_inner_content)
    assert actual_dict == expected_dict


def test_get_rem_block_invalid_task(default_qchem_input: QchemInput) -> None:
    """Test that an unsupported task raises NotSupportedError."""
    inp = replace(default_qchem_input, task="frequency")  # type: ignore
    with pytest.raises(NotSupportedError, match="Task type 'frequency' not currently supported"):
        inp._get_rem_block()


def test_get_rem_block_invalid_method(default_qchem_input: QchemInput) -> None:
    """Test that an unsupported method raises ValidationError."""
    inp = replace(default_qchem_input, level_of_theory="ccsdt")
    with pytest.raises(ValidationError, match="Unsupported or unrecognized level_of_theory"):
        inp._get_rem_block()


# Note: _get_basis_block tests rely on basis_sets.registry functionality
# These might need more setup or mocking if registry interaction is complex
def test_get_basis_block_dict(default_qchem_input: QchemInput) -> None:
    """Test $basis block generation for a dictionary basis set."""
    basis_dict = {"O": "6-31g*", "H": "sto-3g"}
    inp = default_qchem_input.set_basis(basis_dict)
    # Basic check, assumes registry provides simple names for these common sets
    expected = """$basis
O 0
6-31g*
****
H 0
sto-3g
****
$end"""
    assert inp._get_basis_block() == expected


def test_get_basis_block_string(default_qchem_input: QchemInput) -> None:
    """Test that $basis block is empty for a string basis set."""
    inp = default_qchem_input.set_basis("def2-svp")
    assert inp._get_basis_block() == ""


def test_get_solvent_block_pcm(helpers, default_qchem_input: QchemInput) -> None:
    """Test $solvent block generation for PCM."""
    inp = default_qchem_input.set_solvation(model="pcm", solvent="water")
    expected_dict = {"SOLVENTNAME": "water"}

    actual_block_full_string = inp._get_solvent_block()
    if not actual_block_full_string:
        actual_dict = {}
    else:
        actual_block_content_lines = actual_block_full_string.strip().split("\n")
        if len(actual_block_content_lines) < 2:  # Should have $section and $end
            pytest.fail(f"Solvent block is malformed: {actual_block_full_string}")
        actual_block_inner_content = "\n".join(actual_block_content_lines[1:-1])
        actual_dict = helpers.parse_qchem_rem_section(actual_block_inner_content)
    assert actual_dict == expected_dict


def test_get_solvent_block_non_pcm(default_qchem_input: QchemInput) -> None:
    """Test $solvent block is empty for non-PCM models or no solvation."""
    inp_smd = default_qchem_input.set_solvation(model="smd", solvent="toluene")
    assert inp_smd._get_solvent_block() == ""
    assert default_qchem_input._get_solvent_block() == ""  # No solvation


def test_get_smx_block_smd(helpers, default_qchem_input: QchemInput) -> None:
    """Test $smx block generation for SMD."""
    inp = default_qchem_input.set_solvation(model="smd", solvent="Acetonitrile")  # Test mixed case
    expected_dict = {"SOLVENT": "acetonitrile"}

    actual_block_full_string = inp._get_smx_block()
    if not actual_block_full_string:
        actual_dict = {}
    else:
        actual_block_content_lines = actual_block_full_string.strip().split("\n")
        if len(actual_block_content_lines) < 2:  # Should have $section and $end
            pytest.fail(f"SMX block is malformed: {actual_block_full_string}")
        actual_block_inner_content = "\n".join(actual_block_content_lines[1:-1])
        actual_dict = helpers.parse_qchem_rem_section(actual_block_inner_content)
    assert actual_dict == expected_dict


def test_get_smx_block_non_smd(default_qchem_input: QchemInput) -> None:
    """Test $smx block is empty for non-SMD models or no solvation."""
    inp_pcm = default_qchem_input.set_solvation(model="pcm", solvent="water")
    assert inp_pcm._get_smx_block() == ""
    assert default_qchem_input._get_smx_block() == ""  # No solvation

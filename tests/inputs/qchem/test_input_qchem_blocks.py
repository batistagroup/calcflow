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


def test_get_rem_block_minimal(default_qchem_input: QchemInput) -> None:
    """Test minimal $rem block generation (energy, HF)."""
    # Use replace to ensure defaults are tested correctly
    inp = replace(default_qchem_input)
    expected = """$rem
    METHOD          hf
    BASIS           sto-3g
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
$end"""
    # Simple comparison, ignoring whitespace differences for robustness
    assert set(inp._get_rem_block().split()) == set(expected.split())


def test_get_rem_block_opt_dft_unrestricted(default_qchem_input: QchemInput) -> None:
    """Test $rem block for geometry optimization with DFT and unrestricted."""
    inp = replace(
        default_qchem_input,
        task="geometry",
        level_of_theory="b3lyp",
        basis_set="def2-svp",
        unrestricted=True,
        spin_multiplicity=2,  # Consistent with unrestricted=True
    )
    expected = """$rem
    METHOD          b3lyp
    BASIS           def2-svp
    JOBTYPE         opt
    UNRESTRICTED    True
    SYMMETRY        False
    SYM_IGNORE      True
$end"""
    assert set(inp._get_rem_block().split()) == set(expected.split())


def test_get_rem_block_tddft(default_qchem_input: QchemInput) -> None:
    """Test $rem block generation with TDDFT enabled."""
    inp = default_qchem_input.set_tddft(nroots=5, triplets=True, state_analysis=False)
    expected = """$rem
    METHOD          hf
    BASIS           sto-3g
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
    CIS_N_ROOTS     5
    CIS_SINGLETS    True
    CIS_TRIPLETS    True
    STATE_ANALYSIS  False
$end"""
    assert set(inp._get_rem_block().split()) == set(expected.split())


def test_get_rem_block_tddft_rpa(default_qchem_input: QchemInput) -> None:
    """Test $rem block generation with TDDFT and RPA enabled."""
    inp = default_qchem_input.set_tddft(nroots=3).set_rpa(enable=True)
    expected = """$rem
    METHOD          hf
    BASIS           sto-3g
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
    CIS_N_ROOTS     3
    CIS_SINGLETS    True
    CIS_TRIPLETS    False
    STATE_ANALYSIS  True
    RPA             True
$end"""
    assert set(inp._get_rem_block().split()) == set(expected.split())


def test_get_rem_block_solvation(default_qchem_input: QchemInput) -> None:
    """Test $rem block generation with implicit solvation."""
    inp = default_qchem_input.set_solvation(model="pcm", solvent="water")
    expected = """$rem
    METHOD          hf
    BASIS           sto-3g
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
    SOLVENT_METHOD  pcm
$end"""
    assert set(inp._get_rem_block().split()) == set(expected.split())


def test_get_rem_block_dict_basis(default_qchem_input: QchemInput) -> None:
    """Test $rem block uses BASIS=gen for dictionary basis sets."""
    inp = default_qchem_input.set_basis({"O": "6-31g*", "H": "sto-3g"})
    expected = """$rem
    METHOD          hf
    BASIS           gen
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
$end"""
    assert set(inp._get_rem_block().split()) == set(expected.split())


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


def test_get_solvent_block_pcm(default_qchem_input: QchemInput) -> None:
    """Test $solvent block generation for PCM."""
    inp = default_qchem_input.set_solvation(model="pcm", solvent="water")
    expected = """$solvent
    SolventName           water
$end"""
    assert inp._get_solvent_block() == expected


def test_get_solvent_block_non_pcm(default_qchem_input: QchemInput) -> None:
    """Test $solvent block is empty for non-PCM models or no solvation."""
    inp_smd = default_qchem_input.set_solvation(model="smd", solvent="toluene")
    assert inp_smd._get_solvent_block() == ""
    assert default_qchem_input._get_solvent_block() == ""  # No solvation


def test_get_smx_block_smd(default_qchem_input: QchemInput) -> None:
    """Test $smx block generation for SMD."""
    inp = default_qchem_input.set_solvation(model="smd", solvent="Acetonitrile")  # Test mixed case
    expected = """$smx
    solvent    acetonitrile
$end"""
    assert inp._get_smx_block() == expected


def test_get_smx_block_non_smd(default_qchem_input: QchemInput) -> None:
    """Test $smx block is empty for non-SMD models or no solvation."""
    inp_pcm = default_qchem_input.set_solvation(model="pcm", solvent="water")
    assert inp_pcm._get_smx_block() == ""
    assert default_qchem_input._get_smx_block() == ""  # No solvation


# --- Test Public Export Methods ---


def test_export_input_file_minimal(h2o_geometry: Geometry, default_qchem_input: QchemInput) -> None:
    """Test exporting a minimal input file (HF/sto-3g energy)."""
    inp = default_qchem_input
    expected_output = """$molecule
0 1
O      0.00000000      0.00000000      0.11730000
H      0.00000000      0.75720000     -0.46920000
H      0.00000000     -0.75720000     -0.46920000
$end

$rem
    METHOD          hf
    BASIS           sto-3g
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
$end
"""
    actual_output = inp.export_input_file(h2o_geometry)
    # Compare ignoring potential subtle whitespace differences
    assert set(actual_output.split()) == set(expected_output.split())


def test_export_input_file_opt_dft_pcm(h2o_geometry: Geometry, default_qchem_input: QchemInput) -> None:
    """Test exporting an input file for B3LYP/def2-svp Opt with PCM water."""
    inp = replace(
        default_qchem_input,
        task="geometry",
        level_of_theory="b3lyp",
        basis_set="def2-svp",
    ).set_solvation(model="pcm", solvent="water")

    expected_output = """$molecule
0 1
O      0.00000000      0.00000000      0.11730000
H      0.00000000      0.75720000     -0.46920000
H      0.00000000     -0.75720000     -0.46920000
$end

$rem
    METHOD          b3lyp
    BASIS           def2-svp
    JOBTYPE         opt
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
    SOLVENT_METHOD  pcm
$end

$solvent
    SolventName           water
$end
"""
    actual_output = inp.export_input_file(h2o_geometry)
    assert set(actual_output.split()) == set(expected_output.split())


def test_export_input_file_tddft_smd(h2o_geometry: Geometry, default_qchem_input: QchemInput) -> None:
    """Test exporting an input file for TDDFT (triplets) with SMD ethanol."""
    inp = (
        default_qchem_input.set_tddft(nroots=8, singlets=False, triplets=True)
        .set_solvation(model="smd", solvent="ethanol")
        .set_rpa(True)
    )

    expected_output = """$molecule
0 1
O      0.00000000      0.00000000      0.11730000
H      0.00000000      0.75720000     -0.46920000
H      0.00000000     -0.75720000     -0.46920000
$end

$rem
    METHOD          hf
    BASIS           sto-3g
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
    CIS_N_ROOTS     8
    CIS_SINGLETS    False
    CIS_TRIPLETS    True
    STATE_ANALYSIS  True
    RPA             True
    SOLVENT_METHOD  smd
$end

$smx
    solvent    ethanol
$end
"""
    actual_output = inp.export_input_file(h2o_geometry)
    assert set(actual_output.split()) == set(expected_output.split())


def test_export_input_file_dict_basis(h2o_geometry: Geometry, default_qchem_input: QchemInput) -> None:
    """Test exporting an input file with a dictionary basis set."""
    basis_dict = {"O": "6-31g*", "H": "sto-3g"}
    inp = default_qchem_input.set_basis(basis_dict)

    expected_output = """$molecule
0 1
O      0.00000000      0.00000000      0.11730000
H      0.00000000      0.75720000     -0.46920000
H      0.00000000     -0.75720000     -0.46920000
$end

$rem
    METHOD          hf
    BASIS           gen
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
$end

$basis
O 0
6-31g*
****
H 0
sto-3g
****
$end
"""
    actual_output = inp.export_input_file(h2o_geometry)
    assert set(actual_output.split()) == set(expected_output.split())


def test_export_input_file_from_geometry(h2o_geometry: Geometry, default_qchem_input: QchemInput) -> None:
    """Test exporting directly from a Geometry object."""
    inp = default_qchem_input
    # Expected output is the same as test_export_input_file_minimal
    expected_output = """$molecule
0 1
O      0.00000000      0.00000000      0.11730000
H      0.00000000      0.75720000     -0.46920000
H      0.00000000     -0.75720000     -0.46920000
$end

$rem
    METHOD          hf
    BASIS           sto-3g
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
$end
"""
    actual_output = inp.export_input_file(h2o_geometry)
    assert set(actual_output.split()) == set(expected_output.split())


def test_export_input_file_from_geometry_dict_basis_validation(
    h2o_geometry: Geometry, default_qchem_input: QchemInput
) -> None:
    """Test validation failure when dict basis misses elements from geometry."""
    basis_dict_missing_h = {"O": "6-31g*"}  # Missing H
    inp = default_qchem_input.set_basis(basis_dict_missing_h)

    with pytest.raises(ValidationError, match="Custom basis set dictionary is missing definitions for elements"):
        inp.export_input_file(h2o_geometry)


def test_export_input_file_from_geometry_dict_basis_valid(
    h2o_geometry: Geometry, default_qchem_input: QchemInput
) -> None:
    """Test valid export from geometry with complete dictionary basis set."""
    basis_dict_complete = {"O": "6-31g*", "H": "sto-3g"}
    inp = default_qchem_input.set_basis(basis_dict_complete)

    # Expected output is the same as test_export_input_file_dict_basis
    expected_output = """$molecule
0 1
O      0.00000000      0.00000000      0.11730000
H      0.00000000      0.75720000     -0.46920000
H      0.00000000     -0.75720000     -0.46920000
$end

$rem
    METHOD          hf
    BASIS           gen
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
$end

$basis
O 0
6-31g*
****
H 0
sto-3g
****
$end
"""
    actual_output = inp.export_input_file(h2o_geometry)
    assert set(actual_output.split()) == set(expected_output.split())


def test_export_input_file_custom_mixed_basis(h2o_geometry: Geometry, default_qchem_input: QchemInput) -> None:
    """Test exporting an input file with a mixed custom/standard dictionary basis set."""
    # From src/calcflow/basis_sets/qchem/pcX.py
    pcx2_o_def = """S   1   1.00
      0.168149D+05           1.0000000
S   1   1.00
      0.252216D+04           1.0000000
S   1   1.00
      0.574121D+03           1.0000000
S   1   1.00
      0.162513D+03           1.0000000
S   1   1.00
      0.526845D+02           1.0000000
S   1   1.00
      0.186061D+02           1.0000000
S   1   1.00
      0.682448D+01           1.0000000
S   1   1.00
      0.193105D+01           1.0000000
S   1   1.00
      0.743149D+00           1.0000000
S   1   1.00
      0.247793D+00           1.0000000
P   1   1.00
      0.684929D+02           1.0000000
P   1   1.00
      0.158224D+02           1.0000000
P   1   1.00
      0.473591D+01           1.0000000
P   1   1.00
      0.161501D+01           1.0000000
P   1   1.00
      0.540907D+00           1.0000000
P   1   1.00
      0.163898D+00           1.0000000
D   1   1.00
      0.248193D+01           1.0000000
D   1   1.00
      0.721110D+00           1.0000000
F   1   1.00
      0.114891D+01           1.0000000"""
    basis_dict = {"O": "pcX-2", "H": "sto-3g"}  # Use custom pcX2 for O, standard sto-3g for H
    inp = replace(default_qchem_input, level_of_theory="b3lyp").set_basis(basis_dict)

    expected_output = f"""$molecule
0 1
O        0.00000000      0.00000000      0.11730000
H        0.00000000      0.75720000     -0.46920000
H        0.00000000     -0.75720000     -0.46920000
$end

$rem
    METHOD          b3lyp
    BASIS           gen
    JOBTYPE         sp
    UNRESTRICTED    False
    SYMMETRY        False
    SYM_IGNORE      True
$end

$basis
O 0
{pcx2_o_def}
****
H 0
sto-3g
****
$end
"""
    # We need to register the basis set for the test to work
    # importing qchem triggers the registration

    actual_output = inp.export_input_file(h2o_geometry)
    assert set(actual_output.split()) == set(expected_output.split())

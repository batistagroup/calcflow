import logging
from dataclasses import replace

import pytest

from calcflow.exceptions import ValidationError
from calcflow.geometry.static import Geometry
from calcflow.inputs.qchem import QchemInput
from calcflow.utils import logger

logger.setLevel(logging.CRITICAL)


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

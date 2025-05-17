import logging

import pytest

from calcflow.exceptions import ValidationError
from calcflow.inputs.qchem import QchemInput
from calcflow.utils import logger

logger.setLevel(logging.CRITICAL)


def test_set_solvation_valid(default_qchem_input: QchemInput) -> None:
    """Test setting valid solvation parameters."""
    inp = default_qchem_input.set_solvation(model="pcm", solvent="water")
    assert inp.implicit_solvation_model == "pcm"
    assert inp.solvent == "water"
    assert inp is not default_qchem_input  # Ensure immutability


def test_set_solvation_disable(default_qchem_input: QchemInput) -> None:
    """Test disabling solvation by setting model and solvent to None."""
    # First set it, then disable
    inp1 = default_qchem_input.set_solvation(model="smd", solvent="ethanol")
    assert inp1.implicit_solvation_model == "smd"
    assert inp1.solvent == "ethanol"

    inp2 = inp1.set_solvation(model=None, solvent=None)
    assert inp2.implicit_solvation_model is None
    assert inp2.solvent is None
    assert inp2 is not inp1


def test_set_solvation_invalid_model(default_qchem_input: QchemInput) -> None:
    """Test setting an invalid solvation model raises ValidationError."""
    with pytest.raises(ValidationError, match="Solvation model 'cosmo' not recognized."):
        default_qchem_input.set_solvation(model="cosmo", solvent="water")


def test_set_solvation_inconsistent_args(default_qchem_input: QchemInput) -> None:
    """Test setting model or solvent alone raises ValidationError."""
    with pytest.raises(ValidationError, match="Both `model` and `solvent` must be provided together, or neither."):
        default_qchem_input.set_solvation(model="pcm", solvent=None)

    with pytest.raises(ValidationError, match="Both `model` and `solvent` must be provided together, or neither."):
        default_qchem_input.set_solvation(model=None, solvent="water")


def test_set_tddft_valid(default_qchem_input: QchemInput) -> None:
    """Test setting valid TDDFT parameters."""
    inp = default_qchem_input.set_tddft(nroots=10, singlets=False, triplets=True)
    assert inp.run_tddft
    assert inp.tddft_nroots == 10
    assert not inp.tddft_singlets
    assert inp.tddft_triplets
    assert inp.tddft_state_analysis  # Default
    assert inp is not default_qchem_input


def test_set_tddft_invalid_nroots(default_qchem_input: QchemInput) -> None:
    """Test setting invalid nroots in set_tddft raises ValidationError."""
    with pytest.raises(ValidationError, match="tddft_nroots must be a positive integer."):
        default_qchem_input.set_tddft(nroots=0)
    with pytest.raises(ValidationError, match="tddft_nroots must be a positive integer."):
        default_qchem_input.set_tddft(nroots=-1)


def test_set_tddft_no_states(default_qchem_input: QchemInput) -> None:
    """Test setting no states in set_tddft raises ValidationError."""
    with pytest.raises(ValidationError, match="At least one of singlets or triplets must be True for TDDFT."):
        default_qchem_input.set_tddft(nroots=5, singlets=False, triplets=False)


def test_set_rpa(default_qchem_input: QchemInput) -> None:
    """Test setting the rpa flag."""
    assert not default_qchem_input.rpa  # Default is False

    inp_true = default_qchem_input.set_rpa(enable=True)
    assert inp_true.rpa
    assert inp_true is not default_qchem_input

    inp_false = inp_true.set_rpa(enable=False)
    assert not inp_false.rpa
    assert inp_false is not inp_true


def test_set_basis_string(default_qchem_input: QchemInput) -> None:
    """Test setting basis set using a string."""
    inp = default_qchem_input.set_basis("def2-svp")
    assert inp.basis_set == "def2-svp"
    assert inp is not default_qchem_input


def test_set_basis_dict(default_qchem_input: QchemInput) -> None:
    """Test setting basis set using a dictionary."""
    basis_dict = {"O": "def2-tzvp", "H": "def2-svp"}
    inp = default_qchem_input.set_basis(basis_dict)
    assert inp.basis_set == basis_dict
    assert inp is not default_qchem_input

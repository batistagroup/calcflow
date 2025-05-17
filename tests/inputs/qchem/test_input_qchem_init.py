import logging

import pytest

from calcflow.exceptions import ValidationError
from calcflow.inputs.qchem import QchemInput
from calcflow.utils import logger

logger.setLevel(logging.CRITICAL)


# --- Test __post_init__ Validation ---


def test_qchem_input_post_init_valid_defaults(default_qchem_input: QchemInput) -> None:
    """Test that default QchemInput initializes without errors."""
    assert default_qchem_input.program == "qchem"
    assert default_qchem_input.task == "energy"
    assert default_qchem_input.n_cores == 1


def test_qchem_input_post_init_negative_cores() -> None:
    """Test that initialization fails with negative n_cores."""
    with pytest.raises(ValidationError, match="Number of cores must be a positive integer."):
        QchemInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            n_cores=0,
        )
    with pytest.raises(ValidationError, match="Number of cores must be a positive integer."):
        QchemInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            n_cores=-1,
        )


def test_qchem_input_post_init_tddft_missing_nroots() -> None:
    """Test validation error if run_tddft is True but tddft_nroots is missing."""
    with pytest.raises(ValidationError, match="If run_tddft is True, tddft_nroots must be a positive integer."):
        QchemInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            run_tddft=True,
            tddft_nroots=None,
        )


def test_qchem_input_post_init_tddft_invalid_nroots() -> None:
    """Test validation error if run_tddft is True but tddft_nroots is not positive."""
    with pytest.raises(ValidationError, match="If run_tddft is True, tddft_nroots must be a positive integer."):
        QchemInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            run_tddft=True,
            tddft_nroots=0,
        )
    with pytest.raises(ValidationError, match="If run_tddft is True, tddft_nroots must be a positive integer."):
        QchemInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            run_tddft=True,
            tddft_nroots=-5,
        )


def test_qchem_input_post_init_tddft_no_states() -> None:
    """Test validation error if run_tddft is True but both singlets and triplets are False."""
    with pytest.raises(
        ValidationError,
        match="If run_tddft is True, at least one of tddft_singlets or tddft_triplets must be True.",
    ):
        QchemInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            run_tddft=True,
            tddft_nroots=5,
            tddft_singlets=False,
            tddft_triplets=False,
        )


def test_qchem_input_post_init_valid_tddft() -> None:
    """Test valid TDDFT parameters pass initialization."""
    inp = QchemInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="hf",
        basis_set="sto-3g",
        run_tddft=True,
        tddft_nroots=5,
        tddft_singlets=True,
        tddft_triplets=False,
    )
    assert inp.run_tddft
    assert inp.tddft_nroots == 5
    assert inp.tddft_singlets
    assert not inp.tddft_triplets

    inp_triplets = QchemInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="hf",
        basis_set="sto-3g",
        run_tddft=True,
        tddft_nroots=3,
        tddft_singlets=False,
        tddft_triplets=True,
    )
    assert inp_triplets.run_tddft
    assert inp_triplets.tddft_nroots == 3
    assert not inp_triplets.tddft_singlets
    assert inp_triplets.tddft_triplets

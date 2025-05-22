import logging

import pytest

from calcflow.exceptions import ConfigurationError, ValidationError
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


def test_qchem_input_post_init_trnss_without_tddft() -> None:
    """Test validation error if reduced_excitation_space is True but run_tddft is False."""
    with pytest.raises(
        ConfigurationError, match="reduced_excitation_space \(TRNSS\) requires TDDFT \(run_tddft\) to be enabled."
    ):
        QchemInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            reduced_excitation_space=True,
            run_tddft=False,  # Explicitly set to False
            initial_orbitals=[1, 2],  # Provide valid orbitals to isolate the TDDFT dependency check
        )


@pytest.mark.parametrize(
    "initial_orbitals, expected_error_match",
    [
        ([], "The initial_orbitals list cannot be empty for reduced excitation space."),
        ([1, 0], "All initial_orbitals must be positive integers."),
        ([1, -2], "All initial_orbitals must be positive integers."),
        ([1.5], "All initial_orbitals must be positive integers."),  # Test non-integer
    ],
)
def test_qchem_input_post_init_trnss_invalid_solute_orbitals(initial_orbitals, expected_error_match) -> None:
    """Test validation error if reduced_excitation_space is True but initial_orbitals are invalid."""
    # TDDFT must be enabled for TRNSS check to be reached
    with pytest.raises(ValidationError, match=expected_error_match):
        QchemInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            run_tddft=True,  # TDDFT enabled to reach initial_orbitals validation
            tddft_nroots=5,
            reduced_excitation_space=True,
            initial_orbitals=initial_orbitals,
        )


def test_qchem_input_copy_method(default_qchem_input: QchemInput) -> None:
    """Test the copy() method for QchemInput to ensure deep copies."""
    from dataclasses import replace

    # Make the original input a bit more complex
    original_input = replace(
        default_qchem_input,
        n_cores=2,
        run_tddft=True,
        tddft_nroots=3,
        reduced_excitation_space=True,
        initial_orbitals=[10, 20, 30],
        basis_set={"C": "6-31g*", "H": "sto-3g"},  # Use a dict basis for deepcopy check
    )

    copied_input = original_input.copy()

    # 1. Check they are different instances
    assert copied_input is not original_input, "Copied input should be a new instance."

    # 2. Check they are equal in value (dataclasses implement __eq__)
    assert copied_input == original_input, "Copied input should be equal in value to the original."

    # 3. Test deep copy behavior for mutable attributes

    # Test with initial_orbitals (list)
    assert copied_input.initial_orbitals is not None
    assert original_input.initial_orbitals is not None
    copied_input.initial_orbitals.append(40)
    assert original_input.initial_orbitals == [10, 20, 30], (
        "Modifying initial_orbitals in copied input should not affect original."
    )
    assert copied_input.initial_orbitals == [10, 20, 30, 40]

    # Test with basis_set (dict)
    assert isinstance(copied_input.basis_set, dict)
    assert isinstance(original_input.basis_set, dict)
    # The type ignore is because basis_set can be str | dict, but we've ensured it's dict here.
    copied_input.basis_set["O"] = "def2-svp"  # type: ignore

    assert "O" not in original_input.basis_set, "Modifying basis_set in copied input should not affect original."
    assert copied_input.basis_set["O"] == "def2-svp"  # type: ignore

    # Ensure other attributes are still the same after modifications to mutable fields of the copy
    assert copied_input.n_cores == original_input.n_cores
    assert copied_input.run_tddft == original_input.run_tddft

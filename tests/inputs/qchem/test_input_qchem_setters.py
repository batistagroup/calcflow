import logging

import pytest

from calcflow.exceptions import ConfigurationError, ValidationError
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


def test_set_tddft_full_config(default_qchem_input: QchemInput) -> None:
    """Test setting all TDDFT parameters at once."""
    inp = default_qchem_input.set_tddft(nroots=10, singlets=False, triplets=True, state_analysis=False)
    assert inp.run_tddft
    assert inp.tddft_nroots == 10
    assert not inp.tddft_singlets
    assert inp.tddft_triplets
    assert not inp.tddft_state_analysis
    assert inp is not default_qchem_input


def test_set_tddft_partial_update(default_qchem_input: QchemInput) -> None:
    """Test updating individual TDDFT parameters while preserving others."""
    # First setup initial TDDFT configuration
    inp1 = default_qchem_input.set_tddft(nroots=5, singlets=False, triplets=True, state_analysis=False)

    # Update only nroots, should preserve other settings
    inp2 = inp1.set_tddft(nroots=10)
    assert inp2.run_tddft
    assert inp2.tddft_nroots == 10
    assert not inp2.tddft_singlets  # Preserved
    assert inp2.tddft_triplets  # Preserved
    assert not inp2.tddft_state_analysis  # Preserved
    assert inp2 is not inp1

    # Update only state_analysis
    inp3 = inp2.set_tddft(state_analysis=True)
    assert inp3.tddft_nroots == 10  # Preserved
    assert not inp3.tddft_singlets  # Preserved
    assert inp3.tddft_triplets  # Preserved
    assert inp3.tddft_state_analysis  # Updated


def test_set_tddft_no_params_returns_unchanged(default_qchem_input: QchemInput) -> None:
    """Test that calling set_tddft with no parameters returns the same instance."""
    inp = default_qchem_input.set_tddft()
    assert inp is default_qchem_input


def test_set_tddft_invalid_nroots(default_qchem_input: QchemInput) -> None:
    """Test setting invalid nroots raises ValidationError during object creation."""
    with pytest.raises(ValidationError, match="tddft_nroots must be a positive integer"):
        default_qchem_input.set_tddft(nroots=0)
    with pytest.raises(ValidationError, match="tddft_nroots must be a positive integer"):
        default_qchem_input.set_tddft(nroots=-1)


def test_set_tddft_no_states(default_qchem_input: QchemInput) -> None:
    """Test setting no states raises ValidationError during object creation."""
    with pytest.raises(ValidationError, match="at least one of tddft_singlets or tddft_triplets must be True"):
        default_qchem_input.set_tddft(nroots=5, singlets=False, triplets=False)


def test_set_tddft_update_to_invalid_state(default_qchem_input: QchemInput) -> None:
    """Test updating TDDFT to an invalid state raises ValidationError."""
    # Setup valid initial state
    inp1 = default_qchem_input.set_tddft(nroots=5, singlets=True, triplets=False)

    # Try to update to invalid state (no singlets or triplets)
    with pytest.raises(ValidationError, match="at least one of tddft_singlets or tddft_triplets must be True"):
        inp1.set_tddft(singlets=False, triplets=False)


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


@pytest.mark.parametrize(
    "initial_orbitals, expected_orbitals, expected_error",
    [
        ([3, 1, 2, 3], [1, 2, 3], None),
        ([], None, ValidationError),
        ([1, 0, -2], None, ValidationError),
    ],
)
def test_set_reduced_excitation_space(
    default_qchem_input: QchemInput, initial_orbitals, expected_orbitals, expected_error
) -> None:
    """Test configuring reduced excitation space for TDDFT."""

    # Test case where TDDFT is NOT enabled first (should raise ConfigurationError)
    with pytest.raises(ConfigurationError, match="requires TDDFT"):
        default_qchem_input.set_reduced_excitation_space(initial_orbitals=[1])  # Use a dummy list

    # Enable TDDFT first
    tddft_input = default_qchem_input.set_tddft(nroots=5)

    if expected_error:
        with pytest.raises(expected_error):
            tddft_input.set_reduced_excitation_space(initial_orbitals=initial_orbitals)
    else:
        inp = tddft_input.set_reduced_excitation_space(initial_orbitals=initial_orbitals)
        assert inp.reduced_excitation_space is True
        assert inp.initial_orbitals == expected_orbitals
        assert inp is not tddft_input  # Ensure immutability

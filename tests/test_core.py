"""Tests for core.py functionality."""

import logging
from dataclasses import dataclass
from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture

from calcflow.core import CalculationInput
from calcflow.exceptions import ValidationError

# fmt:off

@dataclass(frozen=True)
class SimpleInput(CalculationInput):
    """Simple concrete implementation of CalculationInput for testing."""

    def export_input_file(self, geom: str) -> str:
        """Basic implementation that just returns the geometry."""
        # Basic check to ensure required fields are used
        assert self.charge is not None
        assert self.spin_multiplicity is not None
        assert self.level_of_theory
        assert self.basis_set
        return f"# Simple input file for testing\ncharge: {self.charge}\nspin: {self.spin_multiplicity}\nmethod: {self.level_of_theory}\nbasis: {self.basis_set}\ngeom:\n{geom}"

@pytest.fixture
def base_input() -> SimpleInput:
    """Provides a default valid SimpleInput instance."""
    return SimpleInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="B3LYP",
        basis_set="6-31G*",
    )


def test_simple_input_valid_creation(base_input: SimpleInput) -> None:
    """Test valid SimpleInput creation using the fixture."""
    assert base_input.charge == 0
    assert base_input.spin_multiplicity == 1
    assert base_input.task == "energy"
    assert base_input.level_of_theory == "B3LYP"
    assert base_input.basis_set == "6-31G*"
    assert base_input.memory_mb == 2000 # Default value
    assert base_input.memory_per_core_mb == 2000 # Default value
    assert base_input.implicit_solvation_model is None
    assert base_input.solvent is None
    assert not base_input.unrestricted


@pytest.mark.parametrize("spin_mult", [-1, 0])
def test_creation_invalid_spin_multiplicity(spin_mult: int) -> None:
    """Test that invalid spin multiplicities raise ValidationError during creation."""
    with pytest.raises(ValidationError, match="Spin multiplicity must be a positive integer"):
        SimpleInput(charge=0, spin_multiplicity=spin_mult, task="energy",
                   level_of_theory="B3LYP", basis_set="6-31G*")


@pytest.mark.parametrize("charge", ["0", 1.5, "invalid", "-1", "+1"])
def test_creation_invalid_charge_type(charge: Any) -> None:
    """Test that non-integer charges raise ValidationError during creation."""
    # Using Any for charge type as we are explicitly testing invalid types
    with pytest.raises(ValidationError, match="Charge must be an integer"):
        SimpleInput(charge=charge, spin_multiplicity=1, task="energy",
                   level_of_theory="B3LYP", basis_set="6-31G*")

@pytest.mark.parametrize("inv_model", ["invalid_model", "solvent"])
def test_creation_invalid_solvation_model(inv_model: str) -> None:
    """Test that invalid solvation models raise ValidationError during creation if solvent is also provided."""
    # Need to provide a solvent, otherwise the model check isn't reached due to the consistency check
    with pytest.raises(ValidationError, match="Both `implicit_solvation_model` and `solvent` must be provided together"):
        SimpleInput(charge=0, spin_multiplicity=1, task="energy", level_of_theory="B3LYP", basis_set="6-31G*",
                    implicit_solvation_model=inv_model, solvent=None) # type: ignore[arg-type] # Testing invalid model string

    # Check the consistency error when model is None but solvent is not
    with pytest.raises(ValidationError, match="Both `implicit_solvation_model` and `solvent` must be provided together"):
        SimpleInput(charge=0, spin_multiplicity=1, task="energy", level_of_theory="B3LYP", basis_set="6-31G*",
                    implicit_solvation_model=None, solvent="water")

@pytest.mark.parametrize("model", ["pcm", "smd", "cpcm"])
def test_creation_requires_solvent_with_model(model: str) -> None:
    """Test that specifying solvation model without solvent raises ValidationError."""
    with pytest.raises(ValidationError, match="Both `implicit_solvation_model` and `solvent` must be provided together"):
        SimpleInput(charge=0, spin_multiplicity=1, task="energy", level_of_theory="B3LYP", basis_set="6-31G*",
                    implicit_solvation_model=model, solvent=None) # type: ignore[arg-type] # Testing model without solvent

@pytest.mark.parametrize("solvent", ["water", "ethanol", "benzene"])
def test_creation_valid_solvation_model_with_solvent(solvent: str) -> None:
    """Test that specifying valid solvation model with solvent works."""
    instance = SimpleInput(charge=0, spin_multiplicity=1, task="energy", level_of_theory="B3LYP", basis_set="6-31G*",
                implicit_solvation_model="pcm", solvent=solvent)
    assert instance.implicit_solvation_model == "pcm"
    assert instance.solvent == solvent.lower() # Check normalization

@pytest.mark.parametrize("model,solvent", [
    ("pcm", "water"),
    ("smd", "ethanol"),
    ("cpcm", "benzene"),
    ("PCM", "WATER"), # Test case insensitivity
])
def test_set_solvation(base_input: SimpleInput, model: str, solvent: str) -> None:
    """Test set_solvation method adds, removes, and validates solvation."""
    solvated = base_input.set_solvation(model, solvent) # type: ignore[arg-type] # Testing case normalization handled by method
    assert solvated.implicit_solvation_model == model.lower()
    assert solvated.solvent == solvent.lower()
    assert solvated is not base_input # Ensure immutability

    # Remove solvation
    dry = solvated.set_solvation(None, None)
    assert dry.implicit_solvation_model is None
    assert dry.solvent is None
    assert dry is not solvated # Ensure immutability

    # Invalid: mixed None/value
    with pytest.raises(ValidationError, match="Both `model` and `solvent` must be provided together"):
        base_input.set_solvation("pcm", None)
    with pytest.raises(ValidationError, match="Both `model` and `solvent` must be provided together"):
        base_input.set_solvation(None, "water")

    # Invalid: unrecognized model
    with pytest.raises(ValidationError, match="Solvation model 'invalid_model' not recognized"):
        base_input.set_solvation("invalid_model", "water") # type: ignore[arg-type] # Testing invalid model string recognition

def test_set_memory(base_input: SimpleInput, caplog: LogCaptureFixture) -> None:
    """Test set_memory method."""
    assert base_input.memory_mb == 2000 # Default

    # Set new memory
    high_mem = base_input.set_memory(8000)
    assert high_mem.memory_mb == 8000
    assert high_mem is not base_input # Ensure immutability

    # Test low memory warning
    caplog.set_level(logging.WARNING)
    low_mem = base_input.set_memory(200)
    assert low_mem.memory_mb == 200
    assert "Memory allocation seems low (< 256 MB)" in caplog.text

def test_set_memory_per_core(base_input: SimpleInput, caplog: LogCaptureFixture) -> None:
    """Test set_memory_per_core method."""
    assert base_input.memory_per_core_mb == 2000 # Default

    # Set new memory per core
    high_mem_pc = base_input.set_memory_per_core(4000)
    assert high_mem_pc.memory_per_core_mb == 4000
    assert high_mem_pc is not base_input # Ensure immutability

    # Test low memory per core warning
    caplog.set_level(logging.WARNING)
    low_mem_pc = base_input.set_memory_per_core(100)
    assert low_mem_pc.memory_per_core_mb == 100
    assert "Memory per core allocation seems low (< 256 MB)" in caplog.text

def test_post_init_warnings(caplog: LogCaptureFixture) -> None:
    """Test warnings generated during __post_init__."""
    caplog.set_level(logging.WARNING)

    # Low memory warning
    SimpleInput(charge=0, spin_multiplicity=1, task="energy", level_of_theory="B3LYP", basis_set="6-31G*", memory_mb=500)
    assert "Memory allocation seems low (< 512 MB)" in caplog.text
    caplog.clear()

    # Low memory per core warning
    SimpleInput(charge=0, spin_multiplicity=1, task="energy", level_of_theory="B3LYP", basis_set="6-31G*", memory_per_core_mb=200)
    assert "Memory per core allocation seems low (< 256 MB)" in caplog.text
    caplog.clear()

    # Spin multiplicity > 1 without unrestricted warning
    SimpleInput(charge=0, spin_multiplicity=2, task="energy", level_of_theory="B3LYP", basis_set="6-31G*", unrestricted=False)
    assert "Spin multiplicity > 1 but you didn't choose an unrestricted calculation" in caplog.text
    caplog.clear()

    # Unrestricted singlet warning
    SimpleInput(charge=0, spin_multiplicity=1, task="energy", level_of_theory="B3LYP", basis_set="6-31G*", unrestricted=True)
    assert "unrestricted calculation with a singlet spin multiplicity" in caplog.text
    caplog.clear()

def test_export_input_file(base_input: SimpleInput) -> None:
    """Test the basic export_input_file implementation."""
    geom_str = "H 0 0 0\nO 0 0 1"
    expected_output = (
        "# Simple input file for testing\n"
        "charge: 0\n"
        "spin: 1\n"
        "method: B3LYP\n"
        "basis: 6-31G*\n"
        "geom:\n"
        "H 0 0 0\n"
        "O 0 0 1"
    )
    assert base_input.export_input_file(geom_str) == expected_output

# fmt:on

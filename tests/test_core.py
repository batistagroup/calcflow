"""Tests for core.py functionality."""

from dataclasses import dataclass

import pytest

from calcflow.core import CalculationInput

# fmt:off

@dataclass(frozen=True)
class SimpleInput(CalculationInput):  
    """Simple concrete implementation of CalculationInput for testing."""

    def export_input_file(self, geom: str) -> str:
        """Basic implementation that just returns the geometry."""
        return f"# Simple input file for testing\ncharge: {self.charge}\nspin: {self.spin_multiplicity}\nmethod: {self.level_of_theory}\nbasis: {self.basis_set}\ngeom:\n{geom}"

def test_valid_input() -> None:
    """Test valid input parameters."""
    valid = SimpleInput(charge=0, spin_multiplicity=1, task="energy", 
                       level_of_theory="B3LYP", basis_set="6-31G*")
    assert valid.charge == 0
    assert valid.spin_multiplicity == 1


@pytest.mark.parametrize("spin_mult", [-1, 0])
def test_invalid_spin_multiplicity(spin_mult: int) -> None:
    """Test that invalid spin multiplicities raise ValueError."""
    with pytest.raises(ValueError):
        SimpleInput(charge=0, spin_multiplicity=spin_mult, task="energy",
                   level_of_theory="B3LYP", basis_set="6-31G*")


@pytest.mark.parametrize("charge", ["0", 1.5, "invalid", "-1", "+1"])
def test_invalid_charge(charge: object) -> None:
    """Test that non-integer charges raise ValueError."""
    with pytest.raises(ValueError):
        SimpleInput(charge=charge, spin_multiplicity=1, task="energy",  # type: ignore
                   level_of_theory="B3LYP", basis_set="6-31G*")

@pytest.mark.parametrize("inv_model", ["invalid_model", "solvent"])
def test_invalid_solvation_model(inv_model: str) -> None:
    """Test that invalid solvation models raise ValueError."""
    with pytest.raises(ValueError):
        SimpleInput(charge=0, spin_multiplicity=1, task="energy", level_of_theory="B3LYP", basis_set="6-31G*",
                    implicit_solvation_model=inv_model) # type: ignore

@pytest.mark.parametrize("model", ["pcm", "smd", "cpcm"])
def test_valid_solvation_model_without_solvent(model: str) -> None:
    """Test that specifying solvation model without solvent raises ValueError."""
    with pytest.raises(ValueError):
        SimpleInput(charge=0, spin_multiplicity=1, task="energy", level_of_theory="B3LYP", basis_set="6-31G*",
                    implicit_solvation_model=model) # type: ignore

@pytest.mark.parametrize("solvent", ["water", "ethanol", "benzene"])
def test_valid_solvation_model_with_solvent(solvent: str) -> None:
    """Test that specifying solvation model with solvent does not raise ValueError."""
    SimpleInput(charge=0, spin_multiplicity=1, task="energy", level_of_theory="B3LYP", basis_set="6-31G*",
                implicit_solvation_model="pcm", solvent=solvent)

@pytest.mark.parametrize("model,solvent", [
    ("pcm", "water"),
    ("smd", "ethanol"), 
    ("cpcm", "benzene")
])
def test_set_solvation(model: str, solvent: str) -> None:
    """Test set_solvation method."""
    base = SimpleInput(charge=0, spin_multiplicity=1, task="energy", 
                        level_of_theory="B3LYP", basis_set="6-31G*")

    solvated = base.set_solvation(model, solvent) # type: ignore
    assert solvated.implicit_solvation_model == model
    assert solvated.solvent == solvent

    # Remove solvation
    dry = solvated.set_solvation(None, None)
    assert dry.implicit_solvation_model is None
    assert dry.solvent is None

    # Invalid: mixed None/value
    with pytest.raises(ValueError):
        base.set_solvation("pcm", None)

    # Invalid: mixed None/value in other order
    with pytest.raises(ValueError):
        base.set_solvation(None, "water")

    # Invalid: unrecognized model
    with pytest.raises(ValueError):
        base.set_solvation("invalid_model", "water")  # type: ignore

    # Valid choice
    assert base.set_solvation(model, solvent).implicit_solvation_model == model # type: ignore
    assert base.set_solvation(model, solvent).solvent == solvent # type: ignore

def test_set_memory() -> None:
    """Test set_memory method."""
    # Start with a basic instance
    base = SimpleInput(charge=0, spin_multiplicity=1, task="energy",
                level_of_theory="B3LYP", basis_set="6-31G*")

    # default memory
    assert base.memory_mb == 2000

    # Set new memory
    high_mem = base.set_memory(8000)
    assert high_mem.memory_mb == 8000

"""Tests for core.py functionality."""

from dataclasses import dataclass

import pytest

from calcflow.core import CalculationInput


# We need a concrete implementation to test the abstract base class
@dataclass(frozen=True)
class SimpleInput(CalculationInput):
    """Simple concrete implementation of CalculationInput for testing."""

    def export_input_file(self, geom: str) -> str:
        """Basic implementation that just returns the geometry."""
        return f"# Simple input file for testing\ncharge: {self.charge}\nspin: {self.spin_multiplicity}\nmethod: {self.level_of_theory}\nbasis: {self.basis_set}\ngeom:\n{geom}"


def test_calculation_input_validation() -> None:
    """Test validation rules in CalculationInput."""
    # Valid instance
    valid = SimpleInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="B3LYP",
        basis_set="6-31G*",
    )
    assert valid.charge == 0
    assert valid.spin_multiplicity == 1

    # Invalid: negative spin multiplicity
    with pytest.raises(ValueError):
        SimpleInput(
            charge=0,
            spin_multiplicity=-1,  # Invalid
            task="energy",
            level_of_theory="B3LYP",
            basis_set="6-31G*",
        )

    # Invalid: non-integer charge
    with pytest.raises(ValueError):
        SimpleInput(
            charge="0",  # type: ignore # Invalid
            spin_multiplicity=1,
            task="energy",
            level_of_theory="B3LYP",
            basis_set="6-31G*",
        )

    # Invalid: solvation model without solvent
    with pytest.raises(ValueError):
        SimpleInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="B3LYP",
            basis_set="6-31G*",
            implicit_solvation_model="pcm",  # Missing solvent
        )

    # Invalid: solvent without solvation model
    with pytest.raises(ValueError):
        SimpleInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="B3LYP",
            basis_set="6-31G*",
            solvent="water",  # Missing solvation model
        )


def test_set_solvation() -> None:
    """Test set_solvation method."""
    # Start with a basic instance
    base = SimpleInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="B3LYP",
        basis_set="6-31G*",
    )

    # Add solvation
    solvated = base.set_solvation("pcm", "water")
    assert solvated.implicit_solvation_model == "pcm"
    assert solvated.solvent == "water"

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
        base.set_solvation("invalid_model", "water")


def test_set_memory() -> None:
    """Test set_memory method."""
    # Start with a basic instance
    base = SimpleInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="B3LYP",
        basis_set="6-31G*",
    )

    # Default memory
    assert base.memory_mb == 2000

    # Set new memory
    high_mem = base.set_memory(8000)
    assert high_mem.memory_mb == 8000

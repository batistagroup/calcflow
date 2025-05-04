"""Tests for core.py functionality."""

import logging
from dataclasses import dataclass
from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture

from calcflow.core import CalculationInput
from calcflow.exceptions import ValidationError
from calcflow.geometry.static import Geometry
from calcflow.utils import logger

logger.setLevel(logging.INFO)

# fmt:off

@dataclass(frozen=True)
class SimpleInput(CalculationInput):
    """Simple concrete implementation of CalculationInput for testing."""

    def export_input_file(self, geom: Geometry) -> str:
        """Basic implementation that just returns the geometry."""
        # Basic check to ensure required fields are used
        assert self.charge is not None
        assert self.spin_multiplicity is not None
        assert self.level_of_theory
        assert self.basis_set
        return f"# Simple input file for testing\ncharge: {self.charge}\nspin: {self.spin_multiplicity}\nmethod: {self.level_of_theory}\nbasis: {self.basis_set}\ngeom:\n{geom.get_coordinate_block()}"

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
    geom = Geometry(num_atoms=2, comment="", atoms=[("O", (0, 0, 0)), ("H", (0, 0, 1))])
    expected_output = (
        "# Simple input file for testing\n"
        "charge: 0\n"
        "spin: 1\n"
        "method: B3LYP\n"
        "basis: 6-31G*\n"
        "geom:\n"
        "O        0.00000000      0.00000000      0.00000000\n"
        "H        0.00000000      0.00000000      1.00000000"
    )
    assert base_input.export_input_file(geom) == expected_output

# fmt:on

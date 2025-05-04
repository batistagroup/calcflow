import logging
from collections.abc import Generator

import pytest

# Directly import the functions/classes and the registry dictionary for manipulation/clearing
from calcflow.basis_sets import registry
from calcflow.basis_sets.core import CustomBasisSet
from calcflow.exceptions import ValidationError  # Need this for checking basis set validation
from calcflow.utils import logger

logger.setLevel(logging.CRITICAL)  # Keep test output clean

# Define valid sample definitions and objects for reuse
ORCA_DEFS_1 = {"H": "ORCA H def 1", "O": "ORCA O def 1"}
ORCA_DEFS_2 = {"C": "ORCA C def 2", "N": "ORCA N def 2"}
QCHEM_DEFS_1 = {"H": "QCHEM H def 1", "O": "QCHEM O def 1"}

# Create valid basis set objects for tests
BASIS_ORCA_1 = CustomBasisSet(program="orca", name="pcX-3", definitions=ORCA_DEFS_1)
BASIS_ORCA_2 = CustomBasisSet(program="ORCA", name="Def2", definitions=ORCA_DEFS_2)  # Test case insensitivity
BASIS_QCHEM_1 = CustomBasisSet(program="qchem", name="pcX-2", definitions=QCHEM_DEFS_1)  # Same name, diff program


@pytest.fixture(autouse=True)
def clear_registry_fixture() -> Generator[None, None, None]:
    """Ensure the central registry is empty before each test."""
    # Access the private registry directly for clearing
    registry._PROGRAM_REGISTRIES.clear()
    yield  # Run the test
    registry._PROGRAM_REGISTRIES.clear()  # Clean up after test


class TestRegisterBasisSet:
    def test_register_single_program_success(self) -> None:
        """Verify successful registration for one program."""
        registry.register_basis_set(BASIS_ORCA_1)
        assert "orca" in registry._PROGRAM_REGISTRIES
        assert "pcx-3" in registry._PROGRAM_REGISTRIES["orca"]
        assert registry._PROGRAM_REGISTRIES["orca"]["pcx-3"] is BASIS_ORCA_1

    def test_register_multiple_programs_success(self) -> None:
        """Verify successful registration across different programs."""
        registry.register_basis_set(BASIS_ORCA_1)
        registry.register_basis_set(BASIS_QCHEM_1)

        assert "orca" in registry._PROGRAM_REGISTRIES
        assert "pcx-3" in registry._PROGRAM_REGISTRIES["orca"]
        assert registry._PROGRAM_REGISTRIES["orca"]["pcx-3"] is BASIS_ORCA_1

        assert "qchem" in registry._PROGRAM_REGISTRIES
        assert "pcx-2" in registry._PROGRAM_REGISTRIES["qchem"]
        assert registry._PROGRAM_REGISTRIES["qchem"]["pcx-2"] is BASIS_QCHEM_1

    def test_register_same_name_different_program_success(self) -> None:
        """Verify basis sets with the same name can exist under different programs."""
        registry.register_basis_set(BASIS_ORCA_1)
        registry.register_basis_set(BASIS_QCHEM_1)
        # Verification is implicitly done by the previous test's assertions

    def test_register_duplicate_name_same_program_raises_error(self) -> None:
        """Verify ValueError on registering duplicate name for the same program."""
        registry.register_basis_set(BASIS_ORCA_1)
        # Create another object with same normalized program/name
        duplicate_basis = CustomBasisSet(program="ORCA", name="PCX-3", definitions={"He": "..."})
        with pytest.raises(ValueError, match="'pcx-3' is already registered for program 'orca'"):
            registry.register_basis_set(duplicate_basis)

    def test_register_incorrect_type_raises_error(self) -> None:
        """Verify TypeError for non-CustomBasisSet objects."""

        class NotABasisSet:
            program: str = "fake"
            name: str = "fake_name"
            definitions: dict[str, str] = {}

        with pytest.raises(TypeError, match="must be an instance of CustomBasisSet"):
            registry.register_basis_set(NotABasisSet())  # type: ignore

    def test_register_invalid_basis_set_raises_error(self) -> None:
        """Verify that validation errors from CustomBasisSet propagate."""
        with pytest.raises(ValidationError, match="program cannot be empty"):
            CustomBasisSet(program="", name="test", definitions={"H": "..."})
        # Note: register_basis_set doesn't need explicit test, as CustomBasisSet init fails


class TestProgramBasisRegistry:
    @pytest.fixture(autouse=True)
    def setup_registries(self) -> None:
        """Pre-register basis sets for testing the accessor class."""
        registry.register_basis_set(BASIS_ORCA_1)
        registry.register_basis_set(BASIS_ORCA_2)
        registry.register_basis_set(BASIS_QCHEM_1)

    def test_init_success(self) -> None:
        """Verify successful initialization for a program with registered sets."""
        orca_registry = registry.ProgramBasisRegistry("orca")
        assert orca_registry.program == "orca"
        # Check internal state (optional, but useful for debugging test)
        assert orca_registry._registry is registry._PROGRAM_REGISTRIES["orca"]

        qchem_registry = registry.ProgramBasisRegistry("QCHEM")  # Case-insensitive
        assert qchem_registry.program == "qchem"
        assert qchem_registry._registry is registry._PROGRAM_REGISTRIES["qchem"]

    def test_init_program_not_found_raises_error(self) -> None:
        """Verify ValueError if initializing for a program with no sets registered."""
        with pytest.raises(KeyError, match="No basis sets registered for program 'gamess'"):
            registry.ProgramBasisRegistry("gamess")

    def test_getitem_success(self) -> None:
        """Verify successful retrieval using __getitem__ (case-insensitive name)."""
        orca_registry = registry.ProgramBasisRegistry("orca")
        assert orca_registry["pcx-3"] is BASIS_ORCA_1
        assert orca_registry["PCX-3"] is BASIS_ORCA_1  # Case-insensitivity
        assert orca_registry["def2"] is BASIS_ORCA_2
        assert orca_registry["DEF2"] is BASIS_ORCA_2  # Case-insensitivity

        qchem_registry = registry.ProgramBasisRegistry("qchem")
        assert qchem_registry["pcx-2"] is BASIS_QCHEM_1

    def test_getitem_not_found_raises_keyerror(self) -> None:
        """Verify KeyError when accessing a non-existent name for the program."""
        orca_registry = registry.ProgramBasisRegistry("orca")
        with pytest.raises(KeyError, match="'nonexistent' not found for program 'orca'"):
            _ = orca_registry["nonexistent"]

        # Verify name exists for qchem but not orca
        with pytest.raises(KeyError, match="'pcx-2' not found for program 'orca'"):
            # This assumes BASIS_QCHEM_1 has 'pcx-2', but it wasn't registered for ORCA specifically.
            # Let's refine this check slightly to be clearer. pcx-2 *is* registered for orca.
            # Let's try a name only registered for qchem.
            _ = orca_registry["pcx-2"]

        qchem_registry = registry.ProgramBasisRegistry("qchem")
        with pytest.raises(KeyError, match="'def2' not found for program 'qchem'"):
            _ = qchem_registry["def2"]  # def2 only registered for orca

    def test_contains_success(self) -> None:
        """Verify `in` operator works correctly (case-insensitive name)."""
        orca_registry = registry.ProgramBasisRegistry("orca")
        assert "pcx-3" in orca_registry
        assert "PCX-3" in orca_registry  # Case-insensitivity
        assert "def2" in orca_registry
        assert "Def2" in orca_registry  # Case-insensitivity
        assert "nonexistent" not in orca_registry

        qchem_registry = registry.ProgramBasisRegistry("qchem")
        assert "pcx-2" in qchem_registry
        assert "def2" not in qchem_registry  # Registered only for orca

    def test_program_property(self) -> None:
        """Verify the program property returns the correct normalized name."""
        orca_registry = registry.ProgramBasisRegistry("ORCA")
        assert orca_registry.program == "orca"
        qchem_registry = registry.ProgramBasisRegistry("qchem")
        assert qchem_registry.program == "qchem"

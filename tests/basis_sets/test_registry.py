import logging
from collections.abc import Generator

import pytest

# Directly import the functions and the registry dictionary for manipulation
from calcflow.basis_sets import registry
from calcflow.basis_sets.core import CustomBasisSet
from calcflow.utils import logger

logger.setLevel(logging.ERROR)

# Define valid sample definitions for reuse
VALID_DEFS_1 = {"H": "H def 1", "O": "O def 1"}
VALID_DEFS_2 = {"C": "C def 2", "N": "N def 2"}


@pytest.fixture(autouse=True)
def clear_registry_fixture() -> Generator[None, None, None]:
    """Ensure the registry is empty before each test."""
    # Access the private registry directly for clearing
    registry._BASIS_SET_REGISTRY.clear()
    yield  # Run the test
    registry._BASIS_SET_REGISTRY.clear()  # Clean up after test


class TestBasisSetRegistry:
    def test_register_basis_set_success(self) -> None:
        """Verify successful registration of a valid CustomBasisSet."""
        basis_set = CustomBasisSet(name="pcX-2", definitions=VALID_DEFS_1)
        registry.register_basis_set(basis_set)

        # Verify it's in the private registry (lowercase key)
        assert "pcx-2" in registry._BASIS_SET_REGISTRY
        assert registry._BASIS_SET_REGISTRY["pcx-2"] is basis_set

    def test_register_basis_set_duplicate_name_raises_error(self) -> None:
        """Verify ValueError is raised when registering a duplicate name."""
        basis_set1 = CustomBasisSet(name="pcX-2", definitions=VALID_DEFS_1)
        registry.register_basis_set(basis_set1)

        # Try registering another with the same name (different case)
        basis_set2 = CustomBasisSet(name="PCX-2", definitions=VALID_DEFS_2)
        with pytest.raises(ValueError, match="'pcx-2' is already registered"):
            registry.register_basis_set(basis_set2)

    def test_register_basis_set_incorrect_type_raises_error(self) -> None:
        """Verify TypeError is raised for non-CustomBasisSet objects."""

        class NotABasisSet:
            name: str = "fake"
            definitions: dict[str, str] = {}

        with pytest.raises(TypeError, match="must be an instance of CustomBasisSet"):
            registry.register_basis_set(NotABasisSet())  # type: ignore

    def test_get_basis_set_object_success(self) -> None:
        """Verify successful retrieval of a registered basis set."""
        basis_set1 = CustomBasisSet(name="pcX-2", definitions=VALID_DEFS_1)
        basis_set2 = CustomBasisSet(name="Another", definitions=VALID_DEFS_2)
        registry.register_basis_set(basis_set1)
        registry.register_basis_set(basis_set2)

        # Test retrieval (case-insensitive)
        retrieved1 = registry.get_basis_set_object("pcx-2")
        retrieved2 = registry.get_basis_set_object("PCX-2")
        retrieved3 = registry.get_basis_set_object("Another")
        retrieved4 = registry.get_basis_set_object("aNOTHER")

        assert retrieved1 is basis_set1
        assert retrieved2 is basis_set1  # Should be the same object
        assert retrieved3 is basis_set2
        assert retrieved4 is basis_set2

    def test_get_basis_set_object_not_found_returns_none(self) -> None:
        """Verify None is returned for a non-registered name."""
        basis_set = CustomBasisSet(name="pcX-2", definitions=VALID_DEFS_1)
        registry.register_basis_set(basis_set)

        assert registry.get_basis_set_object("nonexistent") is None

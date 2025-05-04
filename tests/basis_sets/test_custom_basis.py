import pytest

from calcflow.basis_sets.core import CustomBasisSet
from calcflow.exceptions import ValidationError

# Define valid sample definitions for reuse
VALID_DEFS = {"H": "H 0\nS 3 1.00\n...", "O": "O 0\nS 5 1.00\n..."}


class TestCustomBasisSet:
    def test_initialization_success(self) -> None:
        """Verify successful initialization with valid data."""
        basis_set = CustomBasisSet(name="pcX-2", definitions=VALID_DEFS)
        assert basis_set.name == "pcx-2"  # Check lowercase normalization
        assert basis_set.supported_elements == {"H", "O"}

    def test_initialization_empty_name_raises_error(self) -> None:
        """Verify ValidationError is raised for an empty name."""
        with pytest.raises(ValidationError, match="name cannot be empty"):
            CustomBasisSet(name="", definitions=VALID_DEFS)

    def test_initialization_empty_definitions_raises_error(self) -> None:
        """Verify ValidationError is raised for empty definitions."""
        with pytest.raises(ValidationError, match="must have definitions"):
            CustomBasisSet(name="pcX-2", definitions={})

    def test_post_init_normalizes_name_case(self) -> None:
        """Verify the name is consistently stored in lowercase."""
        basis_set = CustomBasisSet(name="MyBasis_Set", definitions=VALID_DEFS)
        assert basis_set.name == "mybasis_set"

    def test_get_definition_for_element_success(self) -> None:
        """Verify correct definition retrieval for a supported element."""
        basis_set = CustomBasisSet(name="TestBasis", definitions=VALID_DEFS)
        assert basis_set.get_definition_for_element("H") == VALID_DEFS["H"]
        assert basis_set.get_definition_for_element("h") == VALID_DEFS["H"]  # Case-insensitivity

    def test_get_definition_for_element_not_found_raises_error(self) -> None:
        """Verify ValueError is raised for an unsupported element."""
        basis_set = CustomBasisSet(name="TestBasis", definitions=VALID_DEFS)
        with pytest.raises(ValueError, match="Element 'He' not found"):
            basis_set.get_definition_for_element("He")

    def test_supports_element(self) -> None:
        """Verify correct reporting of element support."""
        basis_set = CustomBasisSet(name="TestBasis", definitions=VALID_DEFS)
        assert basis_set.supports_element("H") is True
        assert basis_set.supports_element("o") is True  # Case-insensitivity
        assert basis_set.supports_element("He") is False

    def test_supported_elements_property(self) -> None:
        """Verify the supported_elements property returns the correct set."""
        basis_set = CustomBasisSet(name="TestBasis", definitions=VALID_DEFS)
        assert basis_set.supported_elements == {"H", "O"}

        # Test with different elements
        more_defs = {"C": "...", "N": "..."}
        basis_set_more = CustomBasisSet(name="TestBasis2", definitions=more_defs)
        assert basis_set_more.supported_elements == {"C", "N"}

import pytest

from calcflow.basis_sets.custom_basis import CustomBasisSet
from calcflow.exceptions import ValidationError

# Define valid sample definitions for reuse
VALID_DEFS = {"H": "H 0\nS 3 1.00\n...", "O": "O 0\nS 5 1.00\n..."}
VALID_PROGRAM = "orca"


class TestCustomBasisSet:
    def test_initialization_success(self) -> None:
        """Verify successful initialization with valid data."""
        basis_set = CustomBasisSet(program=VALID_PROGRAM, name="pcX-2", definitions=VALID_DEFS)
        assert basis_set.program == "orca"  # Check lowercase normalization
        assert basis_set.name == "pcx-2"  # Check lowercase normalization
        assert basis_set.supported_elements == {"H", "O"}

    def test_initialization_empty_program_raises_error(self) -> None:
        """Verify ValidationError is raised for an empty program."""
        with pytest.raises(ValidationError, match="program cannot be empty"):
            CustomBasisSet(program="", name="pcX-2", definitions=VALID_DEFS)

    def test_initialization_empty_name_raises_error(self) -> None:
        """Verify ValidationError is raised for an empty name."""
        with pytest.raises(ValidationError, match="name cannot be empty"):
            CustomBasisSet(program=VALID_PROGRAM, name="", definitions=VALID_DEFS)

    def test_initialization_empty_definitions_raises_error(self) -> None:
        """Verify ValidationError is raised for empty definitions."""
        with pytest.raises(ValidationError, match="must have definitions"):
            CustomBasisSet(program=VALID_PROGRAM, name="pcX-2", definitions={})

    def test_post_init_normalizes_program_and_name_case(self) -> None:
        """Verify program and name are consistently stored in lowercase."""
        basis_set = CustomBasisSet(program="ORCA", name="MyBasis_Set", definitions=VALID_DEFS)
        assert basis_set.program == "orca"
        assert basis_set.name == "mybasis_set"

    def test_getitem_success(self) -> None:
        """Verify correct definition retrieval using __getitem__."""
        basis_set = CustomBasisSet(program=VALID_PROGRAM, name="TestBasis", definitions=VALID_DEFS)
        assert basis_set["H"] == VALID_DEFS["H"]
        assert basis_set["h"] == VALID_DEFS["H"]  # Case-insensitivity

    def test_getitem_not_found_raises_keyerror(self) -> None:
        """Verify KeyError is raised for an unsupported element via __getitem__."""
        basis_set = CustomBasisSet(program=VALID_PROGRAM, name="TestBasis", definitions=VALID_DEFS)
        with pytest.raises(KeyError, match="Element 'He' not found"):
            _ = basis_set["He"]

    def test_contains_success(self) -> None:
        """Verify correct reporting of element support via __contains__."""
        basis_set = CustomBasisSet(program=VALID_PROGRAM, name="TestBasis", definitions=VALID_DEFS)
        assert "H" in basis_set
        assert "o" in basis_set  # Case-insensitivity
        assert "He" not in basis_set

    def test_supported_elements_property(self) -> None:
        """Verify the supported_elements property returns the correct set."""
        basis_set = CustomBasisSet(program=VALID_PROGRAM, name="TestBasis", definitions=VALID_DEFS)
        assert basis_set.supported_elements == {"H", "O"}

        # Test with different elements
        more_defs = {"C": "...", "N": "..."}
        basis_set_more = CustomBasisSet(program=VALID_PROGRAM, name="TestBasis2", definitions=more_defs)
        assert basis_set_more.supported_elements == {"C", "N"}

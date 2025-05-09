import pytest

from calcflow.exceptions import ValidationError
from calcflow.inputs.qchem import _convert_transition_to_occupations

# Generate the expected ALPHA string for HOMO-5 -> LUMO with 156 electrons (HOMO=78)
# Occupied set becomes {1..72} U {74..78} U {79}
# Compressed: 1:72 74:78 79
homo_minus_5_lumo_156_alpha_occ = "1:72 74:79"


# Define test cases using parametrize
# Format: (n_electrons, transition, expected_alpha, expected_beta)
valid_cases = [
    # --- Basic cases (10 electrons: HOMO=5, LUMO=6) ---
    (10, "HOMO->LUMO", "1:4 6", "1:5"),
    (10, "homo->lumo", "1:4 6", "1:5"),  # Test case insensitivity
    (10, " HOMO -> LUMO ", "1:4 6", "1:5"),  # Test whitespace robustness
    (10, "HOMO-1->LUMO", "1:3 5:6", "1:5"),  # Updated expected alpha
    (10, "HOMO->LUMO+1", "1:4 7", "1:5"),
    (10, "HOMO-2->LUMO+2", "1:2 4:5 8", "1:5"),  # Updated expected alpha
    # --- More complex cases (156 electrons: HOMO=78, LUMO=79) ---
    (156, "HOMO->LUMO", "1:77 79", "1:78"),
    (
        156,
        "HOMO-5->LUMO",
        homo_minus_5_lumo_156_alpha_occ,  # Corrected alpha
        "1:78",  # Beta is ground state
    ),
    (156, "HOMO->LUMO+10", "1:77 89", "1:78"),
    # --- Edge case (2 electrons: HOMO=1, LUMO=2) ---
    (2, "HOMO->LUMO", "2", "1"),
    # --- Explicit Integer Indices (156 electrons: HOMO=78, LUMO=79) ---
    (156, "6->LUMO", "1:5 7:79", "1:78"),  # Int source, LUMO target
    (156, "6 -> 79", "1:5 7:79", "1:78"),  # Int source, Int target (with space)
    (156, "HOMO->79", "1:77 79", "1:78"),  # HOMO source, Int target
    (156, "78->LUMO", "1:77 79", "1:78"),  # Int(HOMO) source, LUMO target
]

# Define test cases for invalid transitions
invalid_transition_cases = [
    # --- Invalid Targets (Target <= HOMO) ---
    (156, "HOMO->HOMO", "Target orbital must be unoccupied"),
    (156, "HOMO->78", "Target orbital must be unoccupied"),
    (156, "6->HOMO", "Target orbital must be unoccupied"),
    (156, "6->77", "Target orbital must be unoccupied"),
    # --- Invalid Sources (Source > HOMO) ---
    (156, "LUMO->LUMO+1", "Source orbital must be occupied"),
    (156, "LUMO->80", "Source orbital must be occupied"),
    (156, "79->LUMO+1", "Source orbital must be occupied"),
    (156, "80->81", "Source orbital must be occupied"),
    # --- Non-positive indices ---
    (10, "HOMO-5 -> LUMO", "Calculated source orbital index must be positive, got 0"),  # HOMO=5, HOMO-5=0
    (10, "5 -> 0", "Target orbital index must be positive, got '0'"),
    (10, "0 -> 6", "Source orbital index must be positive, got '0'"),
    # Negative indices are caught by set_mom_transition format check
]


@pytest.mark.parametrize("n_electrons, transition, expected_alpha, expected_beta", valid_cases)
def test_convert_transition_to_occupations_valid(
    n_electrons: int, transition: str, expected_alpha: str, expected_beta: str
) -> None:
    """Tests valid transition string conversions produce correct occupation strings."""
    alpha_occ, beta_occ = _convert_transition_to_occupations(transition, n_electrons)
    assert alpha_occ == expected_alpha
    assert beta_occ == expected_beta


def test_convert_transition_to_occupations_odd_electrons() -> None:
    """Tests that an odd number of electrons raises a ValidationError."""
    with pytest.raises(ValidationError, match="Expected even number of electrons"):
        _convert_transition_to_occupations("HOMO->LUMO", 11)


@pytest.mark.parametrize("n_electrons, transition, error_match", invalid_transition_cases)
def test_convert_transition_to_occupations_invalid_transitions(
    n_electrons: int, transition: str, error_match: str
) -> None:
    """Tests that invalid transitions (e.g., to occupied, from virtual) raise ValidationError."""
    with pytest.raises(ValidationError, match=error_match):
        _convert_transition_to_occupations(transition, n_electrons)


# Note: Validation errors for malformed transitions like "HOMO+1->LUMO" or
# "HOMO->LUMO-1" or invalid characters are expected to be caught by the
# calling function (set_mom_transition) before this helper is invoked.
# Therefore, we only test the core conversion logic and the electron count check here.

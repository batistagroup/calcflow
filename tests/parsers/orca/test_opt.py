from collections.abc import Sequence

import pytest

# Assuming your parser lives in electronic_structure.parsers.orca
# and has a function parse_optimization or similar.
# Adjust the import path and function name as necessary.
# from electronic_structure.parsers.orca import parse_optimization_output # No longer needed if using fixture
from calcflow.parsers.orca.opt import OptimizationData  # Keep this for type hinting

# Remove fixture definition as it's moved to conftest.py

# --- Helper Function for list comparison --- #


def assert_allclose_list(
    actual: Sequence[float | Sequence[float]],
    expected: Sequence[float | Sequence[float]],
    atol: float,
    msg: str = "",
) -> None:
    """Asserts that two sequences (potentially nested) are close within a tolerance."""
    assert len(actual) == len(expected), f"Length mismatch: {len(actual)} vs {len(expected)}. {msg}"
    for i, (a_item, e_item) in enumerate(zip(actual, expected, strict=False)):
        if isinstance(a_item, (list | tuple)) and isinstance(e_item, (list | tuple)):
            # Recursively check nested sequences
            assert_allclose_list(a_item, e_item, atol, msg=f"Element {i} mismatch. {msg}")
        elif isinstance(a_item, (int | float)) and isinstance(e_item, (int | float)):
            assert abs(a_item - e_item) <= atol, f"Value mismatch at index {i}: {a_item} vs {e_item}. {msg}"
        else:
            pytest.fail(f"Type mismatch at index {i}: {type(a_item)} vs {type(e_item)}. {msg}")


# Use the fixture name from conftest.py ('parsed_opt_data')
def test_convergence_status(parsed_opt_data: OptimizationData) -> None:
    """
    Verify that the optimization successfully converged.
    """
    # Arrange
    expected_status = "CONVERGED"  # Status is a string

    # Act
    # Access attribute directly
    actual_status = parsed_opt_data.termination_status

    # Assert
    assert actual_status is not None, "Termination status not found in parsed data"
    assert actual_status == expected_status


# Use the fixture name from conftest.py ('parsed_opt_data')
def test_number_of_cycles(parsed_opt_data: OptimizationData) -> None:
    """
    Verify the total number of optimization cycles performed.
    """
    # Arrange
    expected_cycles = 4

    # Act
    # Access attribute directly
    actual_cycles = parsed_opt_data.n_cycles

    # Assert
    assert actual_cycles is not None, "Number of cycles not found in parsed data"
    assert actual_cycles == expected_cycles


# # Use the fixture name from conftest.py ('parsed_opt_data')
# def test_final_energy(parsed_opt_data: OptimizationData) -> None:
#     """
#     Verify the final total energy (including dispersion) after optimization.
#     """
#     # Arrange
#     # FINAL SINGLE POINT ENERGY       -75.317802959907
#     expected_energy_eh = -75.317802959907
#     tolerance = 1e-8  # Tolerance in Hartree

#     # Act
#     # Access attribute directly
#     actual_energy_eh = parsed_opt_data.final_energy_eh

#     # Assert
#     assert actual_energy_eh is not None, "Final energy not found in parsed data"
#     assert isinstance(actual_energy_eh, (float))
#     # Ensure it's not None before comparison
#     assert abs(actual_energy_eh - expected_energy_eh) < tolerance


# # Use the fixture name from conftest.py ('parsed_opt_data')
# def test_final_geometry_coordinates(parsed_opt_data: OptimizationData) -> None:
#     """
#     Verify the Cartesian coordinates of the final optimized geometry.
#     """
#     # Arrange
#     # CARTESIAN COORDINATES (ANGSTROEM)
#     # H      1.367583    1.686996   -0.227375
#     # O      2.353544    1.568144    0.002277
#     # H      2.675073    1.313220   -0.930712
#     expected_coords_angstrom: list[list[float]] = [
#         [1.367583, 1.686996, -0.227375],
#         [2.353544, 1.568144, 0.002277],
#         [2.675073, 1.313220, -0.930712],
#     ]
#     tolerance = 1e-6  # Tolerance in Angstrom

#     # Act
#     # Access attribute directly, checking for None
#     final_geometry = parsed_opt_data.final_geometry
#     assert final_geometry is not None, "Final geometry object not found"
#     # Assume coordinates are stored as a list of lists or similar sequence
#     actual_coords_angstrom_any = getattr(final_geometry, "coordinates_angstrom", None)
#     assert actual_coords_angstrom_any is not None, "Final coordinates not found in geometry object"
#     assert isinstance(actual_coords_angstrom_any, list), "Coordinates should be a list"
#     actual_coords_angstrom: list[list[float]] = actual_coords_angstrom_any  # Type hint for clarity

#     # Assert
#     assert len(actual_coords_angstrom) == len(expected_coords_angstrom), "Mismatch in number of atoms"
#     assert_allclose_list(actual_coords_angstrom, expected_coords_angstrom, atol=tolerance)


# # Use the fixture name from conftest.py ('parsed_opt_data')
# def test_final_geometry_symbols(parsed_opt_data: OptimizationData) -> None:
#     """
#     Verify the atomic symbols of the final optimized geometry.
#     """
#     # Arrange
#     expected_symbols = ["H", "O", "H"]

#     # Act
#     # Access attribute directly, checking for None
#     final_geometry = parsed_opt_data.final_geometry
#     assert final_geometry is not None, "Final geometry object not found"
#     actual_symbols_any = getattr(final_geometry, "symbols", None)
#     assert actual_symbols_any is not None, "Atomic symbols not found in geometry object"
#     assert isinstance(actual_symbols_any, list), "Symbols should be a list"
#     actual_symbols: list[str] = actual_symbols_any

#     # Assert
#     assert actual_symbols == expected_symbols


# # Use the fixture name from conftest.py ('parsed_opt_data')
# def test_dipole_moment(parsed_opt_data: OptimizationData) -> None:
#     """
#     Verify the final total dipole moment magnitude.
#     """
#     # Arrange
#     # Magnitude (Debye)      :      1.628902382
#     expected_dipole_debye = 1.628902382
#     tolerance = 1e-6  # Tolerance in Debye

#     # Act
#     # Access attribute directly, checking for None
#     final_dipole = parsed_opt_data.final_dipole
#     assert final_dipole is not None, "Final dipole object not found in parsed data"
#     # Assuming the dipole object has a 'magnitude_debye' attribute
#     actual_dipole_debye_any = getattr(final_dipole, "magnitude_debye", None)
#     assert actual_dipole_debye_any is not None, "Dipole magnitude not found in dipole object"
#     assert isinstance(actual_dipole_debye_any, float), "Dipole magnitude should be float"
#     actual_dipole_debye: float = actual_dipole_debye_any

#     # Assert
#     assert abs(actual_dipole_debye - expected_dipole_debye) < tolerance


# # Use the fixture name from conftest.py ('parsed_opt_data')
# def test_rotational_constants(parsed_opt_data: OptimizationData) -> None:
#     """
#     Verify the final rotational constants.
#     """
#     # Arrange
#     # Rotational constants in MHz : 623216.120429 427803.063571 253671.645242
#     expected_rot_const_mhz: list[float] = [623216.120429, 427803.063571, 253671.645242]
#     tolerance = 1e-3  # Tolerance in MHz

#     # Act
#     # Rotational constants are often part of the dipole moment block parsing
#     final_dipole = parsed_opt_data.final_dipole
#     assert final_dipole is not None, "Final dipole object not found in parsed data"
#     # Assuming the dipole object has a 'rotational_constants_mhz' attribute
#     actual_rot_const_mhz_any = getattr(final_dipole, "rotational_constants_mhz", None)
#     assert actual_rot_const_mhz_any is not None, "Rotational constants not found in dipole object"
#     assert isinstance(actual_rot_const_mhz_any, list), "Rotational constants should be a list"
#     actual_rot_const_mhz: list[float] = actual_rot_const_mhz_any

#     # Assert
#     assert len(actual_rot_const_mhz) == len(expected_rot_const_mhz), "Mismatch in number of rotational constants"
#     assert_allclose_list(actual_rot_const_mhz, expected_rot_const_mhz, atol=tolerance)


# Add more tests as needed for other parsed data, e.g., Mulliken charges,
# energies/gradients per cycle, basis set info, method info, etc.

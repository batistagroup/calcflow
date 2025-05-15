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


# --- Input Geometry Tests ---


# Use the fixture name from conftest.py ('parsed_opt_data')
def test_input_geometry_coordinates(parsed_opt_data: OptimizationData) -> None:
    """
    Verify the Cartesian coordinates of the input geometry.
    """
    # Arrange
    # From Input File Block / Cycle 1 Start
    # H      1.364990    1.693850   -0.197480
    # O      2.328770    1.562940   -0.041680
    # H      2.702440    1.311570   -0.916650
    expected_coords = [(1.36499, 1.69385, -0.19748), (2.32877, 1.56294, -0.04168), (2.70244, 1.31157, -0.91665)]
    tolerance = 1e-6  # Tolerance in Angstrom

    # Act
    input_geometry = parsed_opt_data.input_geometry
    assert input_geometry is not None, "Input geometry object not found"

    actual_coords = [(atom.x, atom.y, atom.z) for atom in input_geometry]

    # Assert
    assert len(actual_coords) == len(expected_coords), "Mismatch in number of atoms for input geometry"
    for actual, expected in zip(actual_coords, expected_coords, strict=False):
        for a, e in zip(actual, expected, strict=False):
            assert abs(a - e) < tolerance, "Input Geometry Coordinates mismatch"


# Use the fixture name from conftest.py ('parsed_opt_data')
def test_input_geometry_symbols(parsed_opt_data: OptimizationData) -> None:
    """
    Verify the atomic symbols of the input geometry.
    """
    # Arrange
    expected_symbols = ["H", "O", "H"]

    # Act
    input_geometry = parsed_opt_data.input_geometry
    assert input_geometry is not None, "Input geometry object not found"

    actual_symbols = [atom.symbol for atom in input_geometry]

    # Assert
    assert actual_symbols == expected_symbols, "Input Geometry Symbols"


# --- Optimization Cycle Data Tests ---


# Use the fixture name from conftest.py ('parsed_opt_data')
def test_cycle_1_energy(parsed_opt_data: OptimizationData) -> None:
    """
    Verify the SCF energy calculated during the first optimization cycle.
    """
    # Arrange
    expected_energy = -75.31350443847158
    tolerance = 1e-8

    # Act
    assert len(parsed_opt_data.cycles) > 0, "No optimization cycles found in parsed data"
    cycle_1_data = parsed_opt_data.cycles[0]
    assert cycle_1_data is not None, "Cycle 1 data not found"
    assert cycle_1_data.scf_data is not None, "SCF data not found for cycle 1"
    actual_energy = cycle_1_data.scf_data.energy  # Access energy from scf_data

    # Assert
    assert actual_energy is not None, "Energy for cycle 1 not found"
    assert isinstance(actual_energy, float)
    assert abs(actual_energy - expected_energy) < tolerance, "Energy mismatch for cycle 1"


# Use the fixture name from conftest.py ('parsed_opt_data')
def test_cycle_2_energy(parsed_opt_data: OptimizationData) -> None:
    """
    Verify the SCF energy calculated during the second optimization cycle.
    """
    # Arrange
    expected_energy = -75.31760090240802
    tolerance = 1e-8

    # Act
    assert len(parsed_opt_data.cycles) > 1, "Less than 2 optimization cycles found"
    cycle_2_data = parsed_opt_data.cycles[1]
    assert cycle_2_data is not None, "Cycle 2 data not found"
    assert cycle_2_data.scf_data is not None, "SCF data not found for cycle 2"
    actual_energy = cycle_2_data.scf_data.energy  # Access energy from scf_data

    # Assert
    assert actual_energy is not None, "Energy for cycle 2 not found"
    assert isinstance(actual_energy, float)
    assert abs(actual_energy - expected_energy) < tolerance, "Energy mismatch for cycle 2"


# Use the fixture name from conftest.py ('parsed_opt_data')
def test_final_energy(parsed_opt_data: OptimizationData) -> None:
    """
    Verify the final total energy (including dispersion) after optimization.
    """
    # Arrange
    # FINAL SINGLE POINT ENERGY       -75.317802959907
    expected_energy = -75.317802959907
    tolerance = 1e-8  # Tolerance in Hartree

    # Act
    # Access attribute directly
    actual_energy = parsed_opt_data.final_energy
    assert actual_energy is not None, "Final energy not found in parsed data"
    assert parsed_opt_data.final_dispersion is not None, "Dispersion correction not found in parsed data"
    actual_energy += parsed_opt_data.final_dispersion.energy

    # Assert
    assert actual_energy is not None, "Final energy not found in parsed data"
    assert isinstance(actual_energy, (float))
    # Ensure it's not None before comparison
    assert abs(actual_energy - expected_energy) < tolerance


# Use the fixture name from conftest.py ('parsed_opt_data')
def test_final_geometry_coordinates(parsed_opt_data: OptimizationData) -> None:
    """
    Verify the Cartesian coordinates of the final optimized geometry.
    """
    # Arrange
    # CARTESIAN COORDINATES (ANGSTROEM)
    # H      1.367583    1.686996   -0.227375
    # O      2.353544    1.568144    0.002277
    # H      2.675073    1.313220   -0.930712
    expected_coords: list[list[float]] = [
        [1.367583, 1.686996, -0.227375],
        [2.353544, 1.568144, 0.002277],
        [2.675073, 1.313220, -0.930712],
    ]
    tolerance = 1e-6  # Tolerance in Angstrom

    # Act
    # Access attribute directly, checking for None
    final_geometry = parsed_opt_data.final_geometry
    assert final_geometry is not None, "Final geometry object not found"

    actual_coords = [(atom.x, atom.y, atom.z) for atom in final_geometry]

    # Assert
    assert len(actual_coords) == len(expected_coords), "Mismatch in number of atoms for input geometry"
    for actual, expected in zip(actual_coords, expected_coords, strict=False):
        for a, e in zip(actual, expected, strict=False):
            assert abs(a - e) < tolerance, "Final Geometry Coordinates mismatch"


# # Use the fixture name from conftest.py ('parsed_opt_data')
def test_final_geometry_symbols(parsed_opt_data: OptimizationData) -> None:
    """
    Verify the atomic symbols of the final optimized geometry.
    """
    # Arrange
    expected_symbols = ["H", "O", "H"]

    final_geometry = parsed_opt_data.final_geometry
    assert final_geometry is not None, "Final geometry object not found"
    actual_symbols = [atom.symbol for atom in final_geometry]
    assert actual_symbols == expected_symbols


def test_dipole_moment(parsed_opt_data: OptimizationData) -> None:
    """
    Verify the final total dipole moment magnitude.
    """
    # Arrange
    # Magnitude (Debye)      :      1.628902382
    expected_dipole_debye = 1.628902382
    tolerance = 1e-6  # Tolerance in Debye

    # Act
    # Access attribute directly, checking for None
    final_dipole = parsed_opt_data.final_dipole
    assert final_dipole is not None, "Final dipole object not found in parsed data"
    actual_dipole_debye = final_dipole.magnitude
    assert actual_dipole_debye is not None, "Dipole magnitude not found in dipole object"
    assert isinstance(actual_dipole_debye, float), "Dipole magnitude should be float"

    assert abs(actual_dipole_debye - expected_dipole_debye) < tolerance

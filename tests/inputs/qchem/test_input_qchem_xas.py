import logging

import pytest

from calcflow.exceptions import ConfigurationError, ValidationError
from calcflow.geometry.static import Geometry
from calcflow.inputs.qchem import QchemInput
from calcflow.utils import logger

logger.setLevel(logging.CRITICAL)


def test_set_reduced_excitation_space_valid(default_qchem_input: QchemInput) -> None:
    """Test setting valid reduced excitation space parameters."""
    # Enable TDDFT first
    inp_tddft = default_qchem_input.set_tddft(nroots=5)
    orbitals = [10, 8, 12, 8]  # Unsorted, with duplicates
    expected_orbitals = [8, 10, 12]

    inp_reduced = inp_tddft.set_reduced_excitation_space(solute_orbitals=orbitals)

    assert inp_reduced.run_tddft
    assert inp_reduced.reduced_excitation_space
    assert inp_reduced.solute_orbitals == expected_orbitals
    assert inp_reduced is not inp_tddft  # Immutability


def test_set_reduced_excitation_space_tddft_not_enabled(default_qchem_input: QchemInput) -> None:
    """Test error if setting reduced excitation space before TDDFT is enabled."""
    with pytest.raises(ConfigurationError, match="Reduced excitation space .TRNSS. requires TDDFT to be enabled"):
        default_qchem_input.set_reduced_excitation_space(solute_orbitals=[1, 2, 3])


def test_set_reduced_excitation_space_empty_orbitals(default_qchem_input: QchemInput) -> None:
    """Test error if solute_orbitals list is empty."""
    inp_tddft = default_qchem_input.set_tddft(nroots=3)
    with pytest.raises(ValidationError, match="The solute_orbitals list cannot be empty"):
        inp_tddft.set_reduced_excitation_space(solute_orbitals=[])


@pytest.mark.parametrize(
    "invalid_orbitals, error_msg",
    [
        ([1, 0, 2], "All solute_orbitals must be positive integers."),
        ([1, -5, 2], "All solute_orbitals must be positive integers."),
        ([1, 2.5, 3], "All solute_orbitals must be positive integers."),
        (["a", 2, 3], "All solute_orbitals must be positive integers."),
    ],
)
def test_set_reduced_excitation_space_invalid_orbital_values(
    default_qchem_input: QchemInput, invalid_orbitals: list, error_msg: str
) -> None:
    """Test error if solute_orbitals list contains non-positive or non-integer values."""
    inp_tddft = default_qchem_input.set_tddft(nroots=3)
    with pytest.raises(ValidationError, match=error_msg):
        inp_tddft.set_reduced_excitation_space(solute_orbitals=invalid_orbitals)


def test_get_rem_block_tddft_reduced_excitation(default_qchem_input: QchemInput) -> None:
    """Test $rem block with TDDFT and reduced excitation space."""
    inp = default_qchem_input.set_tddft(nroots=5).set_reduced_excitation_space(solute_orbitals=[8, 10, 11])
    rem_block = inp._get_rem_block()

    assert "TRNSS             True" in rem_block
    assert "TRTYPE            3" in rem_block
    assert "N_SOL             3" in rem_block
    # Ensure other TDDFT params are still there
    assert "CIS_N_ROOTS       5" in rem_block


def test_get_solute_block_active(default_qchem_input: QchemInput) -> None:
    """Test $solute block generation when TRNSS is active."""
    inp = default_qchem_input.set_tddft(nroots=3).set_reduced_excitation_space(solute_orbitals=[7, 8, 9])
    expected_block = """$solute
7 8 9
$end"""
    assert inp._get_solute_block() == expected_block


def test_get_solute_block_inactive(default_qchem_input: QchemInput) -> None:
    """Test $solute block is empty when TRNSS is not active."""
    assert default_qchem_input._get_solute_block() == ""  # TRNSS not set
    inp_tddft_only = default_qchem_input.set_tddft(nroots=5)
    assert inp_tddft_only._get_solute_block() == ""  # TDDFT enabled, but not TRNSS


def test_export_input_file_tddft_reduced_excitation(h2o_geometry: Geometry, default_qchem_input: QchemInput) -> None:
    """Test exporting an input file with TDDFT and reduced excitation space."""
    inp = (
        default_qchem_input.set_tddft(nroots=3)
        .set_reduced_excitation_space(solute_orbitals=[4, 5])
        .set_basis("def2-svp")  # Use a slightly more realistic basis
    )

    expected_output_parts = [
        "$molecule",
        "0 1",
        "O        0.00000000      0.00000000      0.11730000",
        "$rem",
        "METHOD            hf",
        "BASIS             def2-svp",
        "JOBTYPE           sp",
        "CIS_N_ROOTS       3",
        "TRNSS             True",
        "TRTYPE            3",
        "N_SOL             2",
        "$solute",
        "4 5",
        "$end",
    ]

    actual_output = inp.export_input_file(h2o_geometry)

    for part in expected_output_parts:
        assert part in actual_output

    # Check block order roughly: molecule -> rem -> solute
    assert actual_output.find("$molecule") < actual_output.find("$rem")
    assert actual_output.find("$rem") < actual_output.find("$solute")


def test_export_input_file_mom_tddft_reduced_excitation(h2o_geometry: Geometry, unrestricted_input: QchemInput) -> None:
    """Test exporting MOM with TDDFT (reduced excitation) in the second job."""
    inp = (
        unrestricted_input.enable_mom()
        .set_mom_transition("HOMO->LUMO")  # Sets up MOM for H2O (5 alpha, 5 beta)
        .set_tddft(nroots=2)  # Enable TDDFT for the second job
        .set_reduced_excitation_space(solute_orbitals=[4, 5])  # Reduce excitation space for TDDFT
        .set_basis("def2-tzvp")  # Use a different basis for clarity
    )

    output = inp.export_input_file(h2o_geometry)
    jobs = output.split("\n\n@@@\n\n")
    assert len(jobs) == 2
    job1, job2 = jobs

    # --- Check Job 1 (standard unrestricted calc) ---
    assert "$solute" not in job1
    assert "TRNSS" not in job1
    assert "BASIS             def2-tzvp" in job1  # Check basis propagates

    # --- Check Job 2 (MOM + TDDFT with reduced excitation) ---
    assert "$rem" in job2
    assert "SCF_GUESS         read" in job2
    assert "MOM_START         1" in job2
    assert "CIS_N_ROOTS       2" in job2
    assert "TRNSS             True" in job2
    assert "TRTYPE            3" in job2
    assert "N_SOL             2" in job2
    assert "BASIS             def2-tzvp" in job2  # Ensure basis is consistent in job 2

    assert "$occupied" in job2
    assert "1:4 6" in job2  # From HOMO->LUMO for H2O
    assert "1:5" in job2

    assert "$solute" in job2
    assert "\n4 5\n" in job2  # Ensure specific orbitals are listed
    assert job2.find("$occupied") < job2.find("$solute")  # Check solute block is after occupied if both present

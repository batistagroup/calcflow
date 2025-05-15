import math
from typing import Any, NoReturn

import pytest
from _pytest.logging import LogCaptureFixture

from calcflow.exceptions import ParsingError
from calcflow.parsers import orca
from calcflow.parsers.orca.blocks.orbitals import OrbitalsParser
from calcflow.parsers.orca.typing import Orbital, _MutableCalculationData


def test_orbital_data_presence(parsed_sp_data: orca.CalculationData) -> None:
    """Verify that the OrbitalsSet object is present."""
    assert parsed_sp_data.orbitals is not None
    assert isinstance(parsed_sp_data.orbitals, orca.OrbitalsSet)


def test_orbital_energies(parsed_sp_data: orca.CalculationData) -> None:
    """Test the details of parsed orbital energies."""
    orbital_data = parsed_sp_data.orbitals
    assert orbital_data is not None
    orbitals = orbital_data.orbitals
    assert len(orbitals) == 7  # Based on the sp.out file

    # Check first orbital
    orb0 = orbitals[0]
    assert orb0.index == 0
    assert math.isclose(orb0.occupation, 2.0)
    assert math.isclose(orb0.energy_eh, -18.937331, rel_tol=1e-6)
    assert math.isclose(orb0.energy_ev, -515.3110, rel_tol=1e-4)

    # Check last occupied orbital (HOMO)
    homo = orbitals[4]
    assert homo.index == 4
    assert math.isclose(homo.occupation, 2.0)
    assert math.isclose(homo.energy_eh, -0.243811, rel_tol=1e-6)
    assert math.isclose(homo.energy_ev, -6.6344, rel_tol=1e-4)

    # Check first unoccupied orbital (LUMO)
    lumo = orbitals[5]
    assert lumo.index == 5
    assert math.isclose(lumo.occupation, 0.0)
    assert math.isclose(lumo.energy_eh, 0.434377, rel_tol=1e-6)
    assert math.isclose(lumo.energy_ev, 11.8200, rel_tol=1e-4)

    # Check HOMO/LUMO indices
    assert orbital_data.homo_index == 4
    assert orbital_data.lumo_index == 5


# Helper to run the parser on text lines
def _run_orbital_parser(lines: list[str]) -> _MutableCalculationData:
    results = _MutableCalculationData(raw_output="")
    parser = OrbitalsParser()
    iterator = iter(lines)
    start_line = next(iterator)  # Assume first line is the start trigger
    parser.parse(iterator, start_line, results)
    return results


def test_parse_empty_block() -> None:
    """Test parsing when the block is found but contains no orbital lines."""
    lines = [
        "                       ORBITAL ENERGIES",
        "                       ----------------",
        "",
        "--------",
    ]
    with pytest.raises(ParsingError, match="no orbitals could be parsed"):
        _run_orbital_parser(lines)


def test_parse_malformed_line_stops_parsing() -> None:
    """Test that parsing stops gracefully if a malformed line is encountered after valid ones."""
    lines = [
        "                       ORBITAL ENERGIES",
        "                       ----------------",
        "",
        "   0   2.0000  -18.937331   -515.3110",
        "   1   2.0000   -1.028610    -27.9900",
        "   THIS IS NOT AN ORBITAL LINE",
        "   2   2.0000   -0.577060    -15.7027",  # Should not be parsed
        "--------",
    ]
    results = _run_orbital_parser(lines)
    assert results.orbitals is not None
    assert len(results.orbitals.orbitals) == 2
    assert results.orbitals.orbitals[0].index == 0
    assert results.orbitals.orbitals[1].index == 1
    assert results.orbitals.homo_index == 1  # Last successfully parsed occupied orbital
    assert results.orbitals.lumo_index is None  # Parsing stopped before LUMO


def test_parse_no_occupied_orbitals() -> None:
    """Test parsing when all orbitals have zero occupation (e.g., cation calc)."""
    lines = [
        "                       ORBITAL ENERGIES",
        "                       ----------------",
        "",
        "   0   0.0000   0.100000      2.7211",
        "   1   0.0000   0.200000      5.4423",
        "   2   0.0000   0.300000      8.1634",
        "--------",
    ]
    results = _run_orbital_parser(lines)
    assert results.orbitals is not None
    assert len(results.orbitals.orbitals) == 3
    assert results.orbitals.homo_index is None
    assert results.orbitals.lumo_index is None  # Cannot determine LUMO without HOMO


def test_parse_all_occupied_orbitals() -> None:
    """Test parsing when all orbitals are occupied (edge case)."""
    lines = [
        "                       ORBITAL ENERGIES",
        "                       ----------------",
        "",
        "   0   2.0000  -18.937331   -515.3110",
        "   1   2.0000   -1.028610    -27.9900",
        "   2   2.0000   -0.577060    -15.7027",
        "--------",
    ]
    results = _run_orbital_parser(lines)
    assert results.orbitals is not None
    assert len(results.orbitals.orbitals) == 3
    assert results.orbitals.homo_index == 2  # Last index is HOMO
    assert results.orbitals.lumo_index is None  # No orbital after HOMO


def test_parse_lumo_index_mismatch_warning(caplog: LogCaptureFixture) -> None:
    """Test that a warning is logged if the orbital index after HOMO is not sequential."""
    lines = [
        "                       ORBITAL ENERGIES",
        "                       ----------------",
        "",
        "   0   2.0000  -18.937331   -515.3110",  # HOMO
        "   2   0.0000    0.434377     11.8200",  # Index jumps from 0 to 2 (LUMO candidate)
        "   3   0.0000    0.500000     13.6057",
        "--------",
    ]
    results = _run_orbital_parser(lines)
    assert results.orbitals is not None
    assert len(results.orbitals.orbitals) == 3
    assert results.orbitals.homo_index == 0
    assert results.orbitals.lumo_index is None  # Mismatch means we don't assign LUMO index
    assert "LUMO index mismatch. Expected 1, found 2" in caplog.text


def test_parse_virtual_orbitals_printed_stops_parsing() -> None:
    """Test that parsing stops at the 'Virtual orbitals printed' line."""
    lines = [
        "                       ORBITAL ENERGIES",
        "                       ----------------",
        "",
        "   0   2.0000  -18.937331   -515.3110",
        "   1   2.0000   -1.028610    -27.9900",
        "   2   2.0000   -0.577060    -15.7027",  # HOMO
        "* Virtual orbitals printed *",
        "   3   0.0000    0.434377     11.8200",  # Should not be parsed
        "--------",
    ]
    results = _run_orbital_parser(lines)
    assert results.orbitals is not None
    assert len(results.orbitals.orbitals) == 3
    assert results.orbitals.homo_index == 2
    assert results.orbitals.lumo_index is None  # Parsing stopped before LUMO


def test_matches_method() -> None:
    """Test the 'matches' logic of the parser."""
    parser = OrbitalsParser()
    results_not_parsed = _MutableCalculationData(raw_output="")
    results_parsed = _MutableCalculationData(raw_output="", parsed_orbitals=True)

    matching_line = "--- ORBITAL ENERGIES ---"
    non_matching_line = "--- GEOMETRY OPTIMIZATION ---"

    # Should match: Correct line, not parsed yet
    assert parser.matches(matching_line, results_not_parsed) is True

    # Should not match: Incorrect line
    assert parser.matches(non_matching_line, results_not_parsed) is False

    # Should not match: Correct line, but already parsed
    assert parser.matches(matching_line, results_parsed) is False

    # Should not match: Incorrect line, already parsed
    assert parser.matches(non_matching_line, results_parsed) is False


# --- Exception Handling Tests ---


def test_unexpected_exception_during_loop(monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture) -> None:
    """Test handling of an unexpected exception during the orbital line loop."""
    lines = [
        "                       ORBITAL ENERGIES",
        "                       ----------------",
        "",
        "   0   2.0000  -18.937331   -515.3110",
        "   1   2.0000   -1.028610    -27.9900",  # Error will be raised here
        "   2   2.0000   -0.577060    -15.7027",
        "--------",
    ]
    results = _MutableCalculationData(raw_output="")
    parser = OrbitalsParser()
    iterator = iter(lines)
    start_line = next(iterator)

    # Mock Orbital creation to raise an error on the second call (orbital index 1)
    call_count = 0
    original_orbital_init = Orbital.__init__

    def mock_orbital_init(*args: Any, **kwargs: Any) -> NoReturn | None:
        nonlocal call_count
        if call_count == 1:
            # Use RuntimeError to bypass the specific ValueError handler
            raise RuntimeError("Simulated unexpected error")
        call_count += 1
        # Ensure the original __init__ is called correctly for the first orbital
        # Pass 'self' (the first arg) along with the others
        original_orbital_init(args[0], **kwargs)  # Corrected call
        return None  # __init__ should return None

    monkeypatch.setattr(Orbital, "__init__", mock_orbital_init)

    # Parse should catch the exception, log it, and mark as attempted
    parser.parse(iterator, start_line, results)

    assert results.orbitals is None  # Parsing aborted before final data creation
    assert results.parsed_orbitals is True  # Marked as attempted
    assert "Unexpected error parsing orbital lines: Simulated unexpected error" in caplog.text


def test_unexpected_exception_post_loop(monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture) -> None:
    """Test handling of an unexpected exception after the loop (e.g., in OrbitalsSet creation)."""
    lines = [
        "                       ORBITAL ENERGIES",
        "                       ----------------",
        "",
        "   0   2.0000  -18.937331   -515.3110",
        "   1   2.0000   -1.028610    -27.9900",
        "--------",
    ]
    results = _MutableCalculationData(raw_output="")
    parser = OrbitalsParser()
    iterator = iter(lines)
    start_line = next(iterator)

    # Mock OrbitalsSet creation to raise an error
    def mock_orbital_data_init(*args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError("Simulated data processing error")

    monkeypatch.setattr(
        "calcflow.parsers.orca.blocks.orbitals.OrbitalsSet", mock_orbital_data_init
    )  # Use string path for patching

    # Parse should catch the exception, log it, and mark as attempted
    parser.parse(iterator, start_line, results)

    assert results.orbitals is None  # Final data creation failed
    assert results.parsed_orbitals is True  # Marked as attempted
    assert "Error processing found orbitals or determining HOMO/LUMO: Simulated data processing error" in caplog.text

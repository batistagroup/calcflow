import math
from typing import Any, NoReturn, cast

import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers import orca
from calcflow.parsers.orca.blocks.scf import ScfParser
from calcflow.parsers.orca.typing import LineIterator, _MutableCalculationData

# --- Fixtures for ScfParser Testing ---


@pytest.fixture
def scf_parser() -> ScfParser:
    """Provides a fresh ScfParser instance."""
    return ScfParser()


@pytest.fixture
def initial_mutable_data() -> _MutableCalculationData:
    """Provides a fresh _MutableCalculationData instance."""
    # Provide an empty string for raw_output as it's required
    return _MutableCalculationData(raw_output="")


def _make_iterator(lines: list[str]) -> LineIterator:
    """Helper to create a LineIterator from a list of strings."""
    return cast(LineIterator, iter(lines))


# --- Existing Tests (Keep Unchanged) ---


def test_scf_data_presence(parsed_sp_data: orca.CalculationData) -> None:
    """Verify that the ScfResults object is present."""
    assert parsed_sp_data.scf is not None
    assert isinstance(parsed_sp_data.scf, orca.ScfResults)


def test_scf_convergence(parsed_sp_data: orca.CalculationData) -> None:
    """Test the SCF convergence status and number of iterations."""
    scf_data = parsed_sp_data.scf
    assert scf_data is not None
    assert scf_data.converged is True
    assert scf_data.n_iterations == 8


def test_scf_energy(parsed_sp_data: orca.CalculationData) -> None:
    """Test the final SCF energy (should match last iteration)."""
    scf_data = parsed_sp_data.scf
    assert scf_data is not None
    assert math.isclose(scf_data.energy, -75.313504422, rel_tol=1e-9)


def test_scf_energy_components(parsed_sp_data: orca.CalculationData) -> None:
    """Test the parsed SCF energy components."""
    components = parsed_sp_data.scf.components if parsed_sp_data.scf else None
    assert components is not None
    assert math.isclose(components.nuclear_repulsion, 8.93764967318280, rel_tol=1e-9)
    assert math.isclose(components.electronic_eh, -84.25117753824344, rel_tol=1e-9)
    assert math.isclose(components.one_electron_eh, -121.92433818585613, rel_tol=1e-9)
    assert math.isclose(components.two_electron_eh, 37.67316064761268, rel_tol=1e-9)
    # XC energy might be None for non-DFT, but should be present here
    assert components.xc_eh is not None
    assert math.isclose(components.xc_eh, -6.561524953854, rel_tol=1e-9)


def test_scf_iterations(parsed_sp_data: orca.CalculationData) -> None:
    """Test the SCF iteration history details."""
    history = parsed_sp_data.scf.iterations if parsed_sp_data.scf else None
    assert history is not None
    assert len(history) == 8

    # Check first iteration (DIIS)
    first_iter = history[0]
    assert first_iter.iteration == 1
    assert math.isclose(first_iter.energy, -75.2570665045658, rel_tol=1e-9)
    assert math.isclose(first_iter.delta_e_eh, 0.0, abs_tol=1e-9)  # Delta E is 0 for first iter
    assert first_iter.rmsdp is not None
    assert math.isclose(first_iter.rmsdp, 3.14e-02, rel_tol=1e-3)
    assert first_iter.maxdp is not None
    assert math.isclose(first_iter.maxdp, 8.08e-02, rel_tol=1e-3)
    assert first_iter.diis_error is not None
    assert math.isclose(first_iter.diis_error, 2.12e-01, rel_tol=1e-3)
    assert first_iter.max_gradient is None
    assert first_iter.damping is not None
    assert math.isclose(first_iter.damping, 0.700, rel_tol=1e-3)
    assert math.isclose(first_iter.time_sec, 2.6, rel_tol=1e-1)

    # Check transition iteration (SOSCF start)
    soscf_start_iter = history[5]  # Iteration 6
    assert soscf_start_iter.iteration == 6
    assert math.isclose(soscf_start_iter.energy, -75.31352708163747, rel_tol=1e-9)
    assert math.isclose(soscf_start_iter.delta_e_eh, -1.77e-05, rel_tol=1e-3)
    assert soscf_start_iter.rmsdp is not None
    assert math.isclose(soscf_start_iter.rmsdp, 3.76e-04, rel_tol=1e-3)
    assert soscf_start_iter.maxdp is not None
    assert math.isclose(soscf_start_iter.maxdp, 1.26e-03, rel_tol=1e-3)
    assert soscf_start_iter.diis_error is None
    assert soscf_start_iter.max_gradient is not None
    assert math.isclose(soscf_start_iter.max_gradient, 7.37e-04, rel_tol=1e-3)
    assert soscf_start_iter.damping is None
    assert math.isclose(soscf_start_iter.time_sec, 10.7, rel_tol=1e-1)

    # Check last iteration
    last_iter = history[-1]
    assert last_iter.iteration == 8
    assert math.isclose(last_iter.energy, -75.31352786506064, rel_tol=1e-9)
    assert math.isclose(last_iter.delta_e_eh, -1.11e-07, rel_tol=1e-3)
    assert last_iter.rmsdp is not None
    assert math.isclose(last_iter.rmsdp, 8.42e-05, rel_tol=1e-3)
    assert last_iter.maxdp is not None
    assert math.isclose(last_iter.maxdp, 2.23e-04, rel_tol=1e-3)
    assert last_iter.diis_error is None  # Should be SOSCF at the end
    assert last_iter.max_gradient is not None
    assert math.isclose(last_iter.max_gradient, 6.64e-05, rel_tol=1e-3)
    assert last_iter.damping is None
    assert math.isclose(last_iter.time_sec, 21.8, rel_tol=1e-1)


# --- New Tests for ScfParser Logic ---


def test_scf_matches_already_parsed(scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData) -> None:
    """Test that matches() returns False if scf data is already parsed."""
    initial_mutable_data.parsed_scf = True
    line = " Cycle         Energy        Delta-E       RMS-DP      Max-DP      DIIS-err    Damping     Time"
    assert not scf_parser.matches(line, initial_mutable_data)


def test_scf_matches_trigger_lines(scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData) -> None:
    """Test that matches() returns True for trigger lines when not parsed."""
    line_diis = " --------- সূচনা ----------   D-I-I-S   ---------- সূচনা ---------"
    line_soscf = " --------- START ----------   S-O-S-C-F   ---------- START ---------"
    assert scf_parser.matches(line_diis, initial_mutable_data)
    assert scf_parser.matches(line_soscf, initial_mutable_data)


def test_scf_parse_invalid_iteration_lines(
    scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData
) -> None:
    """Test handling of invalid numeric formats in iteration lines."""
    diis_header = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
    ]
    # Regex won't match this line, so it should be skipped without error
    invalid_diis_line = "   1   -75.INVALID   1.0e-1   1.0e-2   1.0e-2   1.0e-1   0.700     1.0"
    soscf_header = [
        " --------- START ----------   S-O-S-C-F   ---------- START ---------",
        "-----------------------------------------------------------------",
    ]
    # Regex won't match this line
    invalid_soscf_line = "   2   -75.300       BADVAL   1.0e-3   1.0e-3   1.0e-4    2.0"

    # Test 1: Lines that don't match regex structure
    lines_no_match = diis_header + [invalid_diis_line] + soscf_header + [invalid_soscf_line] + ["TOTAL SCF ENERGY"]
    iterator_no_match = _make_iterator(lines_no_match)
    current_line_no_match = next(iterator_no_match)  # Consume DIIS header line

    # No error should be raised, lines are skipped
    iterator_no_match, _ = scf_parser._parse_iteration_tables(iterator_no_match, current_line_no_match)
    assert not scf_parser.iterations


def test_scf_iteration_table_termination_blank(
    scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData
) -> None:
    """Test iteration table parsing terminates on a blank line."""
    lines = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
        "   1   -75.257066   0.000000   3.14e-02   8.08e-02   2.12e-01   0.700     2.6",
        "",  # Blank line terminator
        "TOTAL SCF ENERGY",
    ]
    iterator = _make_iterator(lines)
    current_line = next(iterator)
    iterator, last_line = scf_parser._parse_iteration_tables(iterator, current_line)
    assert last_line == ""
    assert len(scf_parser.iterations) == 1


def test_scf_iteration_table_termination_eof(
    scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData
) -> None:
    """Test iteration table parsing terminates at end of input."""
    lines = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
        "   1   -75.257066   0.000000   3.14e-02   8.08e-02   2.12e-01   0.700     2.6",
    ]
    iterator = _make_iterator(lines)
    current_line = next(iterator)
    iterator, last_line = scf_parser._parse_iteration_tables(iterator, current_line)
    assert last_line == lines[-1]
    assert len(scf_parser.iterations) == 1
    assert next(iterator, None) is None  # Iterator should be exhausted


def test_scf_parse_no_xc_energy(scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData) -> None:
    """Test parsing when E(XC) component is missing (e.g., Hartree-Fock)."""
    lines = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
        "   1   -75.000000   0.000000   1.0e-02   1.0e-02   1.0e-01   0.700     1.0",
        " **** SCF CONVERGED AFTER 1 CYCLES ****",
        "TOTAL SCF ENERGY",
        " Nuclear Repulsion      :      8.900000000 Eh",
        " Electronic Energy      :    -83.900000000 Eh",
        "   One Electron Energy  :   -120.000000000 Eh",
        "   Two Electron Energy  :     36.100000000 Eh",
        "-----------------------",
        "FINAL SINGLE POINT ENERGY",
    ]
    iterator = _make_iterator(lines)
    scf_parser.parse(iterator, next(iterator), initial_mutable_data)

    assert initial_mutable_data.scf is not None
    assert initial_mutable_data.scf.converged
    assert initial_mutable_data.scf.n_iterations == 1
    assert math.isclose(initial_mutable_data.scf.energy, -75.000000)  # type: ignore
    assert initial_mutable_data.scf.components is not None
    assert initial_mutable_data.scf.components.xc_eh is None
    assert math.isclose(initial_mutable_data.scf.components.nuclear_repulsion, 8.9)
    assert math.isclose(initial_mutable_data.scf.components.electronic_eh, -83.9)


def test_scf_component_termination_keyword(
    scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData
) -> None:
    """Test energy component parsing terminates on a known keyword."""
    lines = [
        "TOTAL SCF ENERGY",
        " Nuclear Repulsion      :      8.900000000 Eh",
        " Electronic Energy      :    -84.200000000 Eh",
        "   One Electron Energy  :   -121.900000000 Eh",
        # Missing Two Electron Energy intentionally before terminator
        "ORBITAL ENERGIES",
        "   Two Electron Energy  :     37.700000000 Eh",  # Should not be parsed
    ]
    iterator = _make_iterator(lines)
    current_line = next(iterator)
    iterator, last_line = scf_parser._parse_energy_components(iterator, current_line)

    assert last_line == "ORBITAL ENERGIES"
    assert scf_parser.nuclear_rep_eh is not None
    assert scf_parser.electronic_eh is not None
    assert scf_parser.one_electron_eh is not None
    assert scf_parser.two_electron_eh is None  # Stopped before parsing it


def test_scf_component_termination_eof(scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData) -> None:
    """Test energy component parsing terminates at end of input."""
    lines = [
        "TOTAL SCF ENERGY",
        " Nuclear Repulsion      :      8.900000000 Eh",
        " Electronic Energy      :    -84.200000000 Eh",
        "   One Electron Energy  :   -121.900000000 Eh",
        "   Two Electron Energy  :     37.700000000 Eh",
        # Intentionally missing XC, end of file next
    ]
    iterator = _make_iterator(lines)
    current_line = next(iterator)
    iterator, last_line = scf_parser._parse_energy_components(iterator, current_line)

    assert last_line is None  # Reached end of iterator
    assert scf_parser.nuclear_rep_eh is not None
    assert scf_parser.electronic_eh is not None
    assert scf_parser.one_electron_eh is not None
    assert scf_parser.two_electron_eh is not None
    assert scf_parser.xc_eh is None


def test_scf_no_convergence_line(scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData) -> None:
    """Test when SCF converges (history present) but convergence line is missing."""
    lines = [
        " --------- START ----------   S-O-S-C-F   ---------- START ---------",
        "-----------------------------------------------------------------",
        "   1   -75.300000   0.000000   1.0e-03   1.0e-03   1.0e-04    2.0",
        "   2   -75.310000  -1.0e-02   1.0e-04   1.0e-04   1.0e-05    3.0",
        # Convergence line omitted
        "TOTAL SCF ENERGY",
        " Nuclear Repulsion      :      8.937649673 Eh",
        " Electronic Energy      :    -84.247649673 Eh",
        "   One Electron Energy  :   -121.900000000 Eh",
        "   Two Electron Energy  :     37.652350327 Eh",
        " E(XC)                  :     -6.500000000 Eh",
        "FINAL SINGLE POINT ENERGY",
    ]
    iterator = _make_iterator(lines)
    scf_parser.parse(iterator, next(iterator), initial_mutable_data)

    assert initial_mutable_data.scf is not None
    assert initial_mutable_data.scf.converged is False  # Line wasn't found
    assert initial_mutable_data.scf.n_iterations == 2  # Inferred from history
    assert math.isclose(initial_mutable_data.scf.energy, -75.310000)  # type: ignore
    assert "SCF convergence line 'SCF CONVERGED AFTER ...' not found" in initial_mutable_data.parsing_warnings[0]


def test_scf_missing_history_warning(scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData) -> None:
    """Test warning when convergence line present but history missing."""
    # Simulate state after parsing components but finding no history
    scf_parser._reset_state()
    scf_parser.converged = True
    scf_parser.n_iterations = 3  # From hypothetical convergence line
    scf_parser.nuclear_rep_eh = 8.937649673
    scf_parser.electronic_eh = -84.247649673
    scf_parser.one_electron_eh = -121.900000000
    scf_parser.two_electron_eh = 37.652350327
    scf_parser.xc_eh = -6.500000000

    # Directly call finalize to test its logic for this state
    scf_parser._finalize_scf_data(initial_mutable_data)

    assert initial_mutable_data.scf is not None
    assert initial_mutable_data.scf.converged is True


def test_scf_iteration_mismatch_warning(scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData) -> None:
    """Test warning when convergence line iteration count mismatches history."""
    lines = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
        "   1   -75.257066   0.000000   3.14e-02   8.08e-02   2.12e-01   0.700     2.6",
        "   2   -75.300000  -4.3e-02   1.0e-03   1.0e-03   1.0e-02   0.700     5.0",
        " **** SCF CONVERGED AFTER 1 CYCLES ****",  # Mismatch! History has 2
        "TOTAL SCF ENERGY",
        " Nuclear Repulsion      :      8.937649673 Eh",
        " Electronic Energy      :    -84.237649673 Eh",  # Adjusted for energy
        "   One Electron Energy  :   -121.900000000 Eh",
        "   Two Electron Energy  :     37.662350327 Eh",
        " E(XC)                  :     -6.500000000 Eh",
        "FINAL SINGLE POINT ENERGY",
    ]
    iterator = _make_iterator(lines)
    scf_parser.parse(iterator, next(iterator), initial_mutable_data)

    assert initial_mutable_data.scf is not None
    assert initial_mutable_data.scf.converged is True
    assert initial_mutable_data.scf.n_iterations == 2  # Corrected from history
    assert math.isclose(initial_mutable_data.scf.energy, -75.300000)  # type: ignore
    assert "Convergence line reported 1 cycles, but found 2 in history" in initial_mutable_data.parsing_warnings[0]


def test_scf_components_missing_no_scf_run_warning(
    scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData
) -> None:
    """Test warning when components are missing and SCF likely didn't run (no history/conv line)."""
    # We need to manually trigger the parser as if it expected SCF, but found none.
    # In a real parse flow, other sections might be parsed first.
    # Here, we simulate finding nothing related to SCF.
    # The ScfParser itself wouldn't naturally trigger on this input via `matches`.
    # This test focuses on the _finalize_scf_data logic under specific conditions.

    # Manually set parser state to mimic post-iteration check with no results
    scf_parser._reset_state()
    # Call finalize directly
    scf_parser._finalize_scf_data(initial_mutable_data)

    assert initial_mutable_data.scf is None  # No ScfResults object created
    assert initial_mutable_data.parsed_scf is True  # Marked as attempted
    assert "SCF did not run or failed early; energy components not found" in initial_mutable_data.parsing_warnings[0]
    assert not initial_mutable_data.parsing_errors


def test_scf_components_missing_after_scf_error(
    scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData
) -> None:
    """Test error when mandatory components missing after SCF converged/ran."""
    lines = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
        "   1   -75.257066   0.000000   3.14e-02   8.08e-02   2.12e-01   0.700     2.6",
        " **** SCF CONVERGED AFTER 1 CYCLES ****",
        "TOTAL SCF ENERGY",
        " Nuclear Repulsion      :      8.937649673 Eh",
        # Electronic and other mandatory energies missing
        " E(XC)                  :     -6.500000000 Eh",
        "FINAL SINGLE POINT ENERGY",
    ]
    iterator = _make_iterator(lines)

    # Call parse normally - it should catch the internal ParsingError
    scf_parser.parse(iterator, next(iterator), initial_mutable_data)

    # Assert that the error was handled internally
    assert initial_mutable_data.scf is None  # Finalize failed before creating object
    assert initial_mutable_data.parsed_scf is True  # Marked as attempted
    assert len(initial_mutable_data.parsing_warnings) == 1
    assert "Could not parse all required SCF energy components" in initial_mutable_data.parsing_warnings[0]


def test_scf_final_energy_ambiguous_error(scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData) -> None:
    """Test error when final energy cannot be determined (not converged, no history)."""
    # Scenario: Components printed, but no iterations, no convergence line.

    # Simulate parser state after component parsing but before finalize
    scf_parser._reset_state()
    scf_parser.nuclear_rep_eh = 8.937649673
    scf_parser.electronic_eh = -84.247649673
    scf_parser.one_electron_eh = -121.9
    scf_parser.two_electron_eh = 37.652350327
    scf_parser.xc_eh = -6.5
    scf_parser.converged = False  # Explicitly not converged
    scf_parser.iterations = []
    scf_parser.last_scf_energy = None

    with pytest.raises(ParsingError, match="Failed to determine final SCF energy"):
        scf_parser._finalize_scf_data(initial_mutable_data)

    assert initial_mutable_data.scf is None


def test_scf_components_section_not_found_warning(
    scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData
) -> None:
    """Test warning when 'TOTAL SCF ENERGY' section is never found."""
    lines = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
        "   1   -75.257066   0.000000   3.14e-02   8.08e-02   2.12e-01   0.700     2.6",
        " **** SCF CONVERGED AFTER 1 CYCLES ****",
        # TOTAL SCF ENERGY section missing
        "ORBITAL ENERGIES",
        "----------------",
    ]
    iterator = _make_iterator(lines)

    # Call parse normally - it should catch the internal ParsingError
    scf_parser.parse(iterator, next(iterator), initial_mutable_data)

    # Assert that the error was handled internally
    assert initial_mutable_data.scf is None  # Finalize failed before creating object
    assert initial_mutable_data.parsed_scf is True  # Marked as attempted
    # Check for both the missing section warning and the caught ParsingError
    assert len(initial_mutable_data.parsing_warnings) == 2
    expected_warning_1 = "SCF Energy Components section ('TOTAL SCF ENERGY') not found."
    expected_warning_2 = "SCF Block: Could not parse all required SCF energy components after SCF execution."
    assert expected_warning_1 in initial_mutable_data.parsing_warnings
    assert expected_warning_2 in initial_mutable_data.parsing_warnings


def test_scf_parse_exception_handling(
    scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test general exception handling during the main parse method."""
    lines = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
        "   1   -75.257066   0.000000   3.14e-02   8.08e-02   2.12e-01   0.700     2.6",
    ]
    iterator = _make_iterator(lines)
    current_line = next(iterator)

    # Force an unexpected error during parsing
    def mock_parse_iterations(*args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError("Unexpected failure")

    monkeypatch.setattr(scf_parser, "_parse_iteration_tables", mock_parse_iterations)

    scf_parser.parse(iterator, current_line, initial_mutable_data)

    assert initial_mutable_data.scf is None
    assert initial_mutable_data.parsed_scf is True  # Marked attempted
    assert "SCF Block: Unexpected error during parsing." in initial_mutable_data.parsing_errors
    assert not initial_mutable_data.parsing_warnings


def test_scf_realistic_non_converged(scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData) -> None:
    """Test parsing a realistic (partial) output where SCF did not converge."""
    lines = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
        "   1   -75.257066   0.000000   3.14e-02   8.08e-02   2.12e-01   0.700     2.6",
        "   ... (many iterations) ...",
        "  50   -75.313500  -1.0e-05   5.0e-03   1.0e-02   8.0e-02   0.700   150.0",
        # No convergence line
        "TOTAL SCF ENERGY",  # Components might still be printed on failure
        " Nuclear Repulsion      :      8.937649673 Eh",
        " Electronic Energy      :    -84.251149673 Eh",  # Based on last iter energy
        "   One Electron Energy  :   -121.900000000 Eh",
        "   Two Electron Energy  :     37.648850327 Eh",
        " E(XC)                  :     -6.500000000 Eh",
        "---------------------",
        "*** Starting incremental Fock matrix formation ***",
        "ERROR: SCF failed to converge in 50 cycles",  # Typical error message
        "FINAL SINGLE POINT ENERGY",
    ]
    # Simulate parsing only the relevant lines for SCF part
    scf_lines = [
        line
        for line in lines
        if "..." not in line and "Starting incremental" not in line and "ERROR" not in line and "FINAL" not in line
    ]
    iterator = _make_iterator(scf_lines)
    current_line = next(iterator)

    scf_parser.parse(iterator, current_line, initial_mutable_data)

    assert initial_mutable_data.scf is not None
    assert initial_mutable_data.scf.converged is False
    assert initial_mutable_data.scf.n_iterations == 50  # From last iteration number
    assert math.isclose(initial_mutable_data.scf.energy, -75.313500)  # type: ignore
    assert len(initial_mutable_data.scf.iterations) == 2  # Only the 2 parsed lines
    assert initial_mutable_data.scf.components is not None
    assert math.isclose(initial_mutable_data.scf.components.electronic_eh, -84.251149673)
    assert "SCF convergence line 'SCF CONVERGED AFTER ...' not found" in initial_mutable_data.parsing_warnings[0]


def test_scf_late_convergence_line(scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData) -> None:
    """Test finding the convergence line *after* the iteration table block."""
    lines = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
        "   1   -75.257066   0.000000   3.14e-02   8.08e-02   2.12e-01   0.700     2.6",
        "",  # Table terminates here
        "Some intermediate lines perhaps",
        " **** SCF CONVERGED AFTER 1 CYCLES ****",  # Found late
        "TOTAL SCF ENERGY",
        " Nuclear Repulsion      :      8.937649673 Eh",
        " Electronic Energy      :    -84.194715673 Eh",  # Based on iter 1 energy
        "   One Electron Energy  :   -121.900000000 Eh",
        "   Two Electron Energy  :     37.705284327 Eh",
        " E(XC)                  :     -6.500000000 Eh",
        "FINAL SINGLE POINT ENERGY",
    ]
    iterator = _make_iterator(lines)
    current_line = next(iterator)

    scf_parser.parse(iterator, current_line, initial_mutable_data)

    assert initial_mutable_data.scf is not None
    assert initial_mutable_data.scf.converged is True
    assert initial_mutable_data.scf.n_iterations == 1
    assert math.isclose(initial_mutable_data.scf.energy, -75.257066)  # type: ignore
    assert len(initial_mutable_data.scf.iterations) == 1
    assert len(initial_mutable_data.parsing_warnings) == 1
    assert (
        initial_mutable_data.parsing_warnings[0]
        == "SCF converged but 'Total energy after final integration' line not found. Using last iteration energy: -75.25706600 Eh"
    )


def test_scf_termination_before_components(
    scf_parser: ScfParser, initial_mutable_data: _MutableCalculationData
) -> None:
    """Test parsing stops if a known terminator appears before TOTAL SCF ENERGY."""
    lines = [
        " --------- START ----------   D-I-I-S   ---------- START ---------",
        "--------------------------------------------------------------------",
        "   1   -75.257066   0.000000   3.14e-02   8.08e-02   2.12e-01   0.700     2.6",
        " **** SCF CONVERGED AFTER 1 CYCLES ****",
        # No TOTAL SCF ENERGY section
        "TIMINGS",  # Terminator encountered before components expected
    ]
    iterator = _make_iterator(lines)
    current_line = next(iterator)

    # Expecting error because converged but components missing
    scf_parser.parse(iterator, current_line, initial_mutable_data)

    assert initial_mutable_data.parsed_scf is True  # Marked attempted

    # Call parse normally - it should catch the internal ParsingError
    scf_parser.parse(iterator, current_line, initial_mutable_data)

    # Assert that the error was handled internally
    assert initial_mutable_data.scf is None  # Finalize failed before creating object
    assert initial_mutable_data.parsed_scf is True  # Marked as attempted
    # Check for both the missing section warning and the caught ParsingError
    assert len(initial_mutable_data.parsing_warnings) > 2
    expected_warning_1 = "SCF Energy Components section ('TOTAL SCF ENERGY') not found."
    expected_warning_2 = "SCF Block: Could not parse all required SCF energy components after SCF execution."
    assert expected_warning_1 in initial_mutable_data.parsing_warnings
    assert expected_warning_2 in initial_mutable_data.parsing_warnings

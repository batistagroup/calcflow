import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.blocks.scf import ScfParser
from calcflow.parsers.qchem.typing import ScfData, ScfIteration, _MutableCalculationData


@pytest.fixture
def parser() -> ScfParser:
    """Fixture for the ScfParser."""
    return ScfParser()


@pytest.fixture
def initial_data() -> _MutableCalculationData:
    """Provides a fresh _MutableCalculationData instance for each test."""
    # Provide an empty string for raw_output as it's a required field
    return _MutableCalculationData(raw_output="")


@pytest.fixture
def sample_scf_block() -> list[str]:
    """Provides lines from a typical converged SCF block."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        "  Eric Jon Sundstrom, Paul Horn, Yuezhi Mao, Dmitri Zuev, Alec White,",
        # ... potentially other header lines ...
        " Exchange:     0.1957 Hartree-Fock + 1.0000 wB97X-D3 + LR-HF",
        " Correlation:  1.0000 wB97X-D3",
        " Using SG-2 standard quadrature grid",
        " Dispersion:   Grimme D3",
        " A restricted SCF calculation will be",
        " performed using DIIS",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525425      3.82e-01  ",
        "    2     -75.2783922518      5.94e-02  ",
        "    3     -75.3074084202      2.46e-02  ",
        "    4     -75.3118637269      1.29e-03  ",
        "    5     -75.3118840796      1.40e-04  ",
        "    6     -75.3118844501      2.68e-05  ",
        "    7     -75.3118844639      5.35e-08  Convergence criterion met",
        " ---------------------------------------",
        " SCF time:   CPU 0.28s  wall 0.00s ",
        " SCF   energy =   -75.31188446",
        " Total energy =   -75.31188446",
        " ",
        " --------------------------------------------------------------",
        " ",
        "                    Orbital Energies (a.u.)",  # Next block start
        " --------------------------------------------------------------",
    ]


def test_matches_start_of_scf_block(parser: ScfParser, initial_data: _MutableCalculationData) -> None:
    """Verify that the parser correctly identifies the SCF start line."""
    line = "  General SCF calculation program by"
    assert parser.matches(line, initial_data) is True


def test_does_not_match_other_lines(parser: ScfParser, initial_data: _MutableCalculationData) -> None:
    """Verify that the parser does not match unrelated lines."""
    lines = [
        " Exchange:     0.1957 Hartree-Fock + 1.0000 wB97X-D3 + LR-HF",
        "    1     -75.0734525425      3.82e-01  ",
        " SCF   energy =   -75.31188446",
        "                    Orbital Energies (a.u.)",
        "",
    ]
    for line in lines:
        assert parser.matches(line, initial_data) is False


def test_does_not_match_if_already_parsed(parser: ScfParser, initial_data: _MutableCalculationData) -> None:
    """Verify that the parser does not match if scf data is already present."""
    line = "  General SCF calculation program by"
    initial_data.parsed_scf = True  # Simulate already parsed
    initial_data.scf = ScfData(
        converged=True, energy_eh=-1.0, n_iterations=1, iteration_history=[]
    )  # Fixed: Added empty history
    assert parser.matches(line, initial_data) is False


def test_parse_converged_scf(
    parser: ScfParser, sample_scf_block: list[str], initial_data: _MutableCalculationData
) -> None:
    """Test parsing a standard converged SCF block."""
    line_iter = iter(sample_scf_block)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True
    assert results.scf.n_iterations == 7
    assert results.scf.energy_eh == pytest.approx(-75.31188446)
    assert len(results.scf.iteration_history) == 7
    assert results.scf.iteration_history[-1] == ScfIteration(iteration=7, energy_eh=-75.3118844639, diis_error=5.35e-08)


def test_parse_non_converged_scf(parser: ScfParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing when SCF does not converge (no convergence line, no final energy)."""
    non_converged_block = [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        # ... header ...
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525425      3.82e-01  ",
        "    2     -75.2783922518      5.94e-02  ",
        # Assume it stops here without converging
        " ---------------------------------------",
        # No "SCF energy =" line
        "                    Orbital Energies (a.u.)",
    ]
    line_iter = iter(non_converged_block)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is False
    assert results.scf.n_iterations == 2
    # Should use the energy from the last iteration when final is missing
    assert results.scf.energy_eh == pytest.approx(-75.2783922518)
    assert len(results.scf.iteration_history) == 2


def test_parse_converged_missing_final_energy(parser: ScfParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing converged SCF but missing the 'SCF energy =' line."""
    missing_final_energy_block = [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        # ... header ...
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525425      3.82e-01  ",
        "    2     -75.2783922518      5.94e-02  Convergence criterion met",
        " ---------------------------------------",
        # No "SCF energy =" line
        "                    Orbital Energies (a.u.)",
    ]
    line_iter = iter(missing_final_energy_block)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True
    assert results.scf.n_iterations == 2
    # Should use the energy from the last iteration
    assert results.scf.energy_eh == pytest.approx(-75.2783922518)
    assert len(results.scf.iteration_history) == 2


def test_parse_unexpected_end_during_iterations(parser: ScfParser, initial_data: _MutableCalculationData) -> None:
    """Test handling of file ending abruptly during SCF iterations."""
    abrupt_end_block = [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        # ... header ...
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525425      3.82e-01  ",
        "    2     -75.2783922518      5.94e-02  ",
        # File ends here
    ]
    line_iter = iter(abrupt_end_block)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    with pytest.raises(ParsingError, match="Unexpected end of file in SCF iteration block"):
        parser.parse(line_iter, start_line, results)

    # Check that partial results were stored before raising
    assert results.parsed_scf is True  # Marked as attempted
    assert results.scf is not None
    assert results.scf.converged is False  # Assumed False due to abrupt end
    assert results.scf.n_iterations == 2
    assert results.scf.energy_eh == pytest.approx(-75.2783922518)


def test_parse_unexpected_end_after_iterations(parser: ScfParser, initial_data: _MutableCalculationData) -> None:
    """Test handling of file ending after iterations but before finding final energy."""
    abrupt_end_block = [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        # ... header ...
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525425      3.82e-01  ",
        "    2     -75.2783922518      5.94e-02  Convergence criterion met",
        " ---------------------------------------",
        " SCF time:   CPU 0.28s  wall 0.00s ",
        # File ends here before "SCF energy =" line
    ]
    line_iter = iter(abrupt_end_block)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # This case should *not* raise an error, but log a warning and use the last iteration energy
    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True  # Convergence was marked
    assert results.scf.n_iterations == 2
    assert results.scf.energy_eh == pytest.approx(-75.2783922518)  # Used last iteration energy


def test_parse_no_iterations_found(parser: ScfParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing when the SCF header is found but no iteration lines follow."""
    no_iterations_block = [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        # ... header ...
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        # No "Cycle Energy DIIS error" header or lines
        " ---------------------------------------",
        "                    Orbital Energies (a.u.)",
    ]
    line_iter = iter(no_iterations_block)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # Should raise ParsingError because the iteration table header is missing
    with pytest.raises(ParsingError, match="Unexpected end of file while searching for SCF iteration table."):
        parser.parse(line_iter, start_line, results)

    # Ensure no partial SCF data was stored incorrectly
    assert results.parsed_scf is False  # Should not be marked as parsed
    assert results.scf is None  # No data stored

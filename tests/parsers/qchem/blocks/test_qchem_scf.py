import logging

import pytest

from calcflow.parsers.qchem.blocks.scf import ScfParser
from calcflow.parsers.qchem.typing import ScfIteration, ScfResults, _MutableCalculationData


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


@pytest.fixture
def sample_scf_block_with_smd() -> list[str]:
    """Provides lines from an SCF block with an SMD summary."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        "  Eric Jon Sundstrom, Paul Horn, Yuezhi Mao, Dmitri Zuev, Alec White,",
        " Exchange:     0.1957 Hartree-Fock + 1.0000 wB97X-D3 + LR-HF",
        " Correlation:  1.0000 wB97X-D3",
        " Using SG-2 standard quadrature grid",
        " Dispersion:   Grimme D3",
        " Using the SMD solvation model",
        " Solvent: water",
        " C-PCM solvent model [f = (eps-1)/eps], solve by matrix inversion",
        " A restricted SCF calculation will be",
        " performed using DIIS",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525440      3.82e-01  ",
        "    2     -75.2973764631      4.85e-02  ",
        "    3     -75.3188175820      1.63e-02  ",
        "    4     -75.3207889318      1.19e-03  ",
        "    5     -75.3208073942      1.25e-04  ",
        "    6     -75.3208076933      2.28e-05  ",
        "    7     -75.3208077035      2.17e-08  Convergence criterion met",
        " ---------------------------------------",
        " SCF time:   CPU 0.32s  wall 0.00s ",
        " Summary of SMD free energies:",
        " \tG_PCM  =    -6.0201 kcal/mol (polarization energy)",
        " \tG_CDS  =     1.4731 kcal/mol (non-electrostatic energy)",
        " \tG_ENP  =   -75.32080770 a.u. = E_SCF (including G_PCM)",
        " \tG(tot) =   -75.31846024 a.u. = G_ENP + G_CDS",
        " SCF   energy =   -75.32080770",  # This is G_ENP
        " Total energy =   -75.31846024",  # This is G(tot)
        " ",
        " --------------------------------------------------------------",
        " ",
        "                    Orbital Energies (a.u.)",  # Next block start
        " --------------------------------------------------------------",
    ]


@pytest.fixture
def sample_scf_block_smd_mismatched_genp() -> list[str]:
    """SCF block with SMD summary where G_ENP mismatches explicit SCF energy."""
    return [
        # ... similar to sample_scf_block_with_smd up to iterations ...
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        "  Eric Jon Sundstrom, Paul Horn, Yuezhi Mao, Dmitri Zuev, Alec White,",
        " Exchange:     0.1957 Hartree-Fock + 1.0000 wB97X-D3 + LR-HF",
        " Correlation:  1.0000 wB97X-D3",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525440      3.82e-01  ",
        "    2     -75.3208077035      2.17e-08  Convergence criterion met",
        " ---------------------------------------",
        " SCF time:   CPU 0.32s  wall 0.00s ",
        " Summary of SMD free energies:",
        " \tG_PCM  =    -6.0201 kcal/mol (polarization energy)",
        " \tG_CDS  =     1.4731 kcal/mol (non-electrostatic energy)",
        " \tG_ENP  =   -75.32080770 a.u. = E_SCF (including G_PCM)",  # G_ENP from summary
        " \tG(tot) =   -75.31846024 a.u. = G_ENP + G_CDS",
        " SCF   energy =   -75.32082770",  # Changed: Was -75.32080788, to make difference > 1e-6
        " Total energy =   -75.31846024",
        "                    Orbital Energies (a.u.)",
    ]


@pytest.fixture
def sample_scf_block_smd_partial_summary() -> list[str]:
    """SCF block with an incomplete SMD summary."""
    return [
        # ... similar to sample_scf_block_with_smd up to iterations ...
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        "  Eric Jon Sundstrom, Paul Horn, Yuezhi Mao, Dmitri Zuev, Alec White,",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525440      3.82e-01  ",
        "    2     -75.3208077035      2.17e-08  Convergence criterion met",
        " ---------------------------------------",
        " SCF time:   CPU 0.32s  wall 0.00s ",
        " Summary of SMD free energies:",
        " \tG_PCM  =    -6.0201 kcal/mol (polarization energy)",
        # G_CDS, G_ENP, G_TOT are missing
        " SCF   energy =   -75.32080770",
        " Total energy =   -75.31846024",
        "                    Orbital Energies (a.u.)",
    ]


@pytest.fixture
def sample_scf_block_smd_malformed_summary() -> list[str]:
    """SCF block with a malformed SMD summary after the start line."""
    return [
        # ... similar to sample_scf_block_with_smd up to iterations ...
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        "  Eric Jon Sundstrom, Paul Horn, Yuezhi Mao, Dmitri Zuev, Alec White,",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525440      3.82e-01  ",
        "    2     -75.3208077035      2.17e-08  Convergence criterion met",
        " ---------------------------------------",
        " SCF time:   CPU 0.32s  wall 0.00s ",
        " Summary of SMD free energies:",
        " \tThis is not a valid G_PCM line.",
        " \tAnother unexpected line.",
        " SCF   energy =   -75.32080770",
        " Total energy =   -75.31846024",
        "                    Orbital Energies (a.u.)",
    ]


@pytest.fixture
def scf_block_ends_after_iter_separator() -> list[str]:
    """SCF block that ends exactly after the iteration table separator."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        " Exchange:     HF",
        " Correlation:  None",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525425      3.82e-01  ",
        "    2     -75.2783922518      5.94e-02  Convergence criterion met",
        " ---------------------------------------",
        # File ends here
    ]


@pytest.fixture
def scf_block_multiple_segments() -> list[str]:
    """SCF block with two distinct iteration tables."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        " Exchange:     HF",
        " Correlation:  None",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",  # First segment header
        " ---------------------------------------",
        "    1     -75.01      1.0e-01  ",
        "    2     -75.02      1.0e-02  Convergence criterion met",
        " ---------------------------------------",
        # Start of second segment
        "  Cycle       Energy         DIIS error",  # Second segment header
        " ---------------------------------------",
        "    1     -76.01      2.0e-01  ",
        "    2     -76.02      2.0e-02  Convergence criterion met",
        " ---------------------------------------",
        " SCF   energy =   -76.02",  # Final energy from second segment
        "                    Orbital Energies (a.u.)",
    ]


@pytest.fixture
def scf_block_ends_with_normal_termination() -> list[str]:
    """SCF block where iteration table is followed by normal termination."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        " Exchange:     HF",
        " Correlation:  None",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525425      3.82e-01  ",
        "    2     -75.2783922518      5.94e-02  Convergence criterion met",
        " ---------------------------------------",
        " Thank you very much for using Q-Chem.",  # Normal termination
    ]


@pytest.fixture
def scf_block_unrecognized_line_after_iters() -> list[str]:
    """SCF block with an unrecognized non-blank line after iterations."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        " Exchange:     HF",
        " Correlation:  None",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525425      3.82e-01  ",
        "    2     -75.2783922518      5.94e-02  Convergence criterion met",
        " ---------------------------------------",
        " This is some unexpected but relevant text.",  # Unrecognized line
        " SCF   energy =   -75.2783922518",  # To ensure final energy parsing is still attempted
        "                    Orbital Energies (a.u.)",
    ]


@pytest.fixture
def scf_fixture_inner_break_new_cycle() -> list[str]:
    """Fixture to test inner loop break when a new 'Cycle...' header is encountered directly."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        " Exchange:     HF",
        " Correlation:  None",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",  # First segment header
        " ---------------------------------------",
        "    1     -75.01      1.0e-01  ",  # Iteration of first segment
        # NO table end '---' for first segment.
        # Instead, directly the header of the next segment.
        "  Cycle       Energy         DIIS error",  # Second segment header, should break inner loop for segment 1
        " ---------------------------------------",
        "    1     -76.01      2.0e-01  ",  # Iteration of second segment
        "    2     -76.02      2.0e-02  Convergence criterion met",
        " ---------------------------------------",
        " SCF   energy =   -76.02",
        "                    Orbital Energies (a.u.)",
    ]


@pytest.fixture
def scf_fixture_inner_break_heuristic() -> list[str]:
    """Fixture to test inner loop break on an end-of-block heuristic line."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        " Exchange:     HF",
        " Correlation:  None",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.01      1.0e-01  ",
        "    2     -75.02      2.0e-02  Convergence criterion met",
        # NO table end '---'. Directly a heuristic line.
        "                    Orbital Energies (a.u.)",  # Heuristic line
        # Minimal content after to ensure parsing can finish if it tries.
        " SCF   energy =   -75.02",  # Should be picked up later
    ]


@pytest.fixture
def scf_fixture_inner_break_unrecognized() -> list[str]:
    """Fixture to test inner loop break on an unrecognized non-blank line."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        " Exchange:     HF",
        " Correlation:  None",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.01      1.0e-01  ",
        "    2     -75.02      2.0e-02  Convergence criterion met",
        # NO table end '---'. Directly an unrecognized line.
        "This is an unrecognized but non-blank line.",  # Unrecognized line
        " SCF   energy =   -75.02",  # Should be picked up later by post-processing
        "                    Orbital Energies (a.u.)",  # To allow clean finish of post-processing
    ]


@pytest.mark.parametrize(
    "heuristic_line, expected_log_content",
    [
        (
            "                    Orbital Energies (a.u.)",
            "SCF block end heuristic encountered: Orbital Energies (a.u.)",
        ),
        (
            " Thank you very much for using Q-Chem.",
            "SCF block end heuristic encountered: Thank you very much for using Q-Chem.",
        ),
    ],
)
def test_inner_loop_breaks_on_heuristic(
    parser: ScfParser,
    initial_data: _MutableCalculationData,
    caplog,
    heuristic_line: str,
    expected_log_content: str,
) -> None:
    """Test that the inner SCF iteration loop breaks correctly on a heuristic line."""
    base_fixture = [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        " Exchange:     HF",
        " Correlation:  None",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.01      1.0e-01  ",
        "    2     -75.02      2.0e-02  Convergence criterion met",
        # Heuristic line will be injected here by the test
    ]
    test_specific_fixture = base_fixture + [
        heuristic_line,
        " SCF   energy =   -75.02",
    ]  # Add final energy line for robustness

    line_iter = iter(test_specific_fixture)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    with caplog.at_level(logging.DEBUG):  # Capture DEBUG messages
        parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True  # From iter line
    assert results.scf.n_iterations == 2
    assert results.scf.energy == pytest.approx(-75.02)  # Explicit line or fallback

    assert ScfIteration(iteration=1, energy=-75.01, diis_error=1.0e-01, step_type=None) in results.scf.iterations
    assert ScfIteration(iteration=2, energy=-75.02, diis_error=2.0e-02, step_type=None) in results.scf.iterations


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
    initial_data.scf = ScfResults(
        converged=True, energy=-1.0, n_iterations=1, iterations=[]
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
    assert results.scf.energy == pytest.approx(-75.31188446)
    assert len(results.scf.iterations) == 7
    assert results.scf.iterations[-1] == ScfIteration(
        iteration=7, energy=-75.3118844639, diis_error=5.35e-08, step_type=None
    )


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
    assert results.scf.energy == pytest.approx(-75.2783922518)
    assert len(results.scf.iterations) == 2


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
    assert results.scf.energy == pytest.approx(-75.2783922518)
    assert len(results.scf.iterations) == 2


def test_parse_unexpected_end_during_iterations(
    parser: ScfParser, initial_data: _MutableCalculationData, caplog
) -> None:
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

    parser.parse(line_iter, start_line, results)

    assert "File ended unexpectedly while parsing SCF iterations in a segment." in caplog.text
    # Check that partial results were stored
    assert results.parsed_scf is True  # Marked as attempted
    assert results.scf is not None
    assert results.scf.converged is False  # Assumed False due to abrupt end
    assert results.scf.n_iterations == 2
    assert results.scf.energy == pytest.approx(-75.2783922518)
    assert len(results.scf.iterations) == 2
    # Ensure the error log about missing final energy is also present due to how the parser now flows
    assert "File ended before finding final SCF energy line (or after SMD summary)." in caplog.text


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
    assert results.scf.energy == pytest.approx(-75.2783922518)  # Used last iteration energy


def test_parse_no_iterations_found(parser: ScfParser, initial_data: _MutableCalculationData, caplog) -> None:
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

    parser.parse(line_iter, start_line, results)

    assert "No 'Cycle Energy DIIS error' header found after SCF block start." in caplog.text
    assert results.parsed_scf is True  # Parser attempted
    assert results.scf is None  # No SCF data could be formed


def test_parse_scf_with_smd_summary(
    parser: ScfParser, sample_scf_block_with_smd: list[str], initial_data: _MutableCalculationData
) -> None:
    """Test parsing an SCF block that includes a full SMD summary."""
    line_iter = iter(sample_scf_block_with_smd)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True
    assert results.scf.n_iterations == 7
    # SCF energy should be G_ENP from the explicit "SCF energy =" line
    assert results.scf.energy == pytest.approx(-75.32080770)

    # Check parsed SMD values
    assert results.smd_g_pcm_kcal_mol == pytest.approx(-6.0201)
    assert results.smd_g_cds_kcal_mol == pytest.approx(1.4731)
    assert results.smd_g_enp_au == pytest.approx(-75.32080770)
    assert results.smd_g_tot_au == pytest.approx(-75.31846024)
    assert not results.parsing_warnings  # No warnings for a clean parse


def test_parse_smd_summary_mismatched_genp(
    parser: ScfParser, sample_scf_block_smd_mismatched_genp: list[str], initial_data: _MutableCalculationData, caplog
) -> None:
    """Test warning for mismatch between SMD G_ENP and explicit SCF energy."""
    line_iter = iter(sample_scf_block_smd_mismatched_genp)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    # ScfResults.energy should take the value from the "SCF   energy =" line
    assert results.scf.energy == pytest.approx(-75.32082770)
    # SMD parsed values should still be from the summary block
    assert results.smd_g_enp_au == pytest.approx(-75.32080770)
    assert "Mismatch between G_ENP from SMD summary" in caplog.text
    assert "Using explicit SCF energy." in caplog.text


def test_parse_smd_partial_summary(
    parser: ScfParser, sample_scf_block_smd_partial_summary: list[str], initial_data: _MutableCalculationData, caplog
) -> None:
    """Test parsing an incomplete SMD summary, expecting a warning."""
    line_iter = iter(sample_scf_block_smd_partial_summary)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.smd_g_pcm_kcal_mol == pytest.approx(-6.0201)
    assert results.smd_g_cds_kcal_mol is None
    assert results.smd_g_enp_au is None
    assert results.smd_g_tot_au is None
    assert "SMD summary block identified, but some energy components (G_ENP, G_TOT) were not parsed." in caplog.text


def test_parse_smd_malformed_summary(
    parser: ScfParser, sample_scf_block_smd_malformed_summary: list[str], initial_data: _MutableCalculationData, caplog
) -> None:
    """Test parsing a malformed SMD summary, expecting warnings and graceful handling."""
    line_iter = iter(sample_scf_block_smd_malformed_summary)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    # Check that values that couldn't be parsed are None
    assert results.smd_g_pcm_kcal_mol is None
    assert results.smd_g_cds_kcal_mol is None
    assert results.smd_g_enp_au is None
    assert results.smd_g_tot_au is None
    # The explicit SCF energy line should still be parsed for ScfResults
    assert results.scf.energy == pytest.approx(-75.32080770)
    # Expect a warning about missing G_ENP, G_TOT from the summary if the summary block was started
    assert "SMD summary block identified, but some energy components (G_ENP, G_TOT) were not parsed." in caplog.text


def test_parse_smd_summary_no_explicit_scf_energy(
    parser: ScfParser, initial_data: _MutableCalculationData, caplog
) -> None:
    """Test SMD summary parsing when the 'SCF   energy =' line is missing."""
    scf_block_smd_no_final_scf = [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525440      3.82e-01  ",
        "    2     -75.3208077035      2.17e-08  Convergence criterion met",
        " ---------------------------------------",
        " SCF time:   CPU 0.32s  wall 0.00s ",
        " Summary of SMD free energies:",
        " \tG_PCM  =    -6.0201 kcal/mol (polarization energy)",
        " \tG_CDS  =     1.4731 kcal/mol (non-electrostatic energy)",
        " \tG_ENP  =   -75.32080770 a.u. = E_SCF (including G_PCM)",
        " \tG(tot) =   -75.31846024 a.u. = G_ENP + G_CDS",
        # No "SCF   energy =" line here
        " Total energy =   -75.31846024",
        "                    Orbital Energies (a.u.)",
    ]
    line_iter = iter(scf_block_smd_no_final_scf)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True  # Converged based on iteration line
    # energy_to_store should fall back to the last iteration energy
    assert results.scf.energy == pytest.approx(-75.3208077035)
    assert results.smd_g_pcm_kcal_mol == pytest.approx(-6.0201)
    assert results.smd_g_cds_kcal_mol == pytest.approx(1.4731)
    assert results.smd_g_enp_au == pytest.approx(-75.32080770)
    assert results.smd_g_tot_au == pytest.approx(-75.31846024)
    # Check for the specific warning about using last iteration's energy
    assert (
        "Using energy from last SCF iteration as final SCF energy; explicit 'SCF energy =' line not found or parsed."
        in caplog.text
    )


def test_parse_file_ends_after_iteration_separator(
    parser: ScfParser, scf_block_ends_after_iter_separator: list[str], initial_data: _MutableCalculationData, caplog
) -> None:
    """Test parsing when the file ends immediately after an SCF iteration table separator."""
    line_iter = iter(scf_block_ends_after_iter_separator)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert "File ended immediately after SCF table separator." in caplog.text
    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True  # Converged based on last iteration line
    assert results.scf.n_iterations == 2
    assert results.scf.energy == pytest.approx(-75.2783922518)  # Uses last iteration energy
    # Check for the warning about using last iteration's energy because the post-processing loop for explicit SCF energy wasn't entered.
    assert (
        "Using energy from last SCF iteration as final SCF energy; explicit 'SCF energy =' line not found or parsed."
        in caplog.text
    )


def test_parse_multiple_scf_segments(
    parser: ScfParser, scf_block_multiple_segments: list[str], initial_data: _MutableCalculationData
) -> None:
    """Test parsing an SCF block with multiple iteration tables (segments)."""
    line_iter = iter(scf_block_multiple_segments)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    # The final convergence and energy should be from the *last* segment
    assert results.scf.converged is True
    assert results.scf.energy == pytest.approx(-76.02)
    # Iterations from ALL segments should be collected
    assert results.scf.n_iterations == 4
    assert len(results.scf.iterations) == 4
    # Check first iteration of first segment
    assert results.scf.iterations[0] == ScfIteration(iteration=1, energy=-75.01, diis_error=1.0e-01, step_type=None)
    # Check last iteration of last segment
    assert results.scf.iterations[3] == ScfIteration(iteration=2, energy=-76.02, diis_error=2.0e-02, step_type=None)


def test_parse_ends_with_normal_termination_heuristic(
    parser: ScfParser, scf_block_ends_with_normal_termination: list[str], initial_data: _MutableCalculationData, caplog
) -> None:
    """Test parsing when SCF block ends due to normal termination message."""
    line_iter = iter(scf_block_ends_with_normal_termination)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True
    assert results.scf.n_iterations == 2
    assert results.scf.energy == pytest.approx(-75.2783922518)  # From last iter, no explicit SCF line
    # This warning confirms the heuristic correctly stopped the search for a final SCF energy line.
    assert "Final SCF energy line not found. Stopped at: Thank you very much for using Q-Chem." in caplog.text
    assert "Using energy from last SCF iteration as final SCF energy" in caplog.text


def test_parse_unrecognized_line_ends_iterations(
    parser: ScfParser, scf_block_unrecognized_line_after_iters: list[str], initial_data: _MutableCalculationData, caplog
) -> None:
    """Test parsing when an unrecognized non-blank line terminates the iteration table."""
    line_iter = iter(scf_block_unrecognized_line_after_iters)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True  # Based on last iteration line before unrecognized one
    assert results.scf.n_iterations == 2
    # The explicit "SCF energy" line after the unrecognized line should be found and used
    assert results.scf.energy == pytest.approx(-75.2783922518)

    # Verify the first iteration to ensure they were collected before the unrecognized line
    assert (
        ScfIteration(iteration=1, energy=-75.0734525425, diis_error=3.82e-01, step_type=None) in results.scf.iterations
    )


def test_inner_loop_breaks_on_new_cycle_header(
    parser: ScfParser, scf_fixture_inner_break_new_cycle: list[str], initial_data: _MutableCalculationData, caplog
) -> None:
    """Test that the inner SCF iteration loop breaks correctly on a new 'Cycle...' header."""
    line_iter = iter(scf_fixture_inner_break_new_cycle)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    with caplog.at_level(logging.DEBUG):  # Capture DEBUG messages for this test
        parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True  # Based on the second segment
    assert results.scf.energy == pytest.approx(-76.02)
    assert results.scf.n_iterations == 3  # 1 from first segment, 2 from second

    # Ensure the first iteration of the first segment was captured
    assert ScfIteration(iteration=1, energy=-75.01, diis_error=1.0e-01, step_type=None) in results.scf.iterations
    # Ensure iterations from the second segment were also captured
    assert ScfIteration(iteration=1, energy=-76.01, diis_error=2.0e-01, step_type=None) in results.scf.iterations
    assert ScfIteration(iteration=2, energy=-76.02, diis_error=2.0e-02, step_type=None) in results.scf.iterations


def test_inner_loop_breaks_on_unrecognized_line(
    parser: ScfParser, scf_fixture_inner_break_unrecognized: list[str], initial_data: _MutableCalculationData, caplog
) -> None:
    """Test that the inner SCF iteration loop breaks correctly on an unrecognized non-blank line."""
    line_iter = iter(scf_fixture_inner_break_unrecognized)
    start_line = next(line for line in line_iter if parser.matches(line, initial_data))
    results = initial_data

    # No need to set DEBUG for this one, as it logs a WARNING
    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True  # Based on the iter line before unrecognized
    assert results.scf.n_iterations == 2
    # The explicit "SCF energy" line after the unrecognized line should be found and used
    assert results.scf.energy == pytest.approx(-75.02)

    # Verify that the specific warning for an unrecognized line was logged.
    unrecognized_line_warning_logged = any(
        record.levelname == "WARNING"
        and "Unrecognized non-blank line, assuming end of SCF tables: This is an unrecognized but non-blank line."
        in record.message
        for record in caplog.records
    )
    assert unrecognized_line_warning_logged, "The expected WARNING for an unrecognized line was not found in logs."

    # Verify the iterations were collected before the unrecognized line
    assert ScfIteration(iteration=1, energy=-75.01, diis_error=1.0e-01, step_type=None) in results.scf.iterations
    assert ScfIteration(iteration=2, energy=-75.02, diis_error=2.0e-02, step_type=None) in results.scf.iterations

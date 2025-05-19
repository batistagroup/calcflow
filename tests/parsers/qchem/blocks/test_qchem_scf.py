import pytest

from calcflow.parsers.qchem.blocks.scf import ScfParser
from calcflow.parsers.qchem.typing import ScfIteration, _MutableCalculationData


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


@pytest.fixture
def sample_scf_block_unconverged() -> list[str]:
    """Provides lines from a typical unconverged SCF block."""
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
        "    2     -75.2783922518      5.94e-02  ",
        "    3     -75.3074084202      2.46e-02  ",
        "    4     -75.3118637269      1.29e-03  ",
        # Block ends without meeting convergence criteria and without final energy line
        "                    Orbital Energies (a.u.)",
    ]


@pytest.fixture
def sample_smd_h2o_sp_sto() -> list[str]:
    """Provides lines from the specific SMD block from data/calculations/examples/qchem/h2o/sp-sto-smd.out."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        "  Eric Jon Sundstrom, Paul Horn, Yuezhi Mao, Dmitri Zuev, Alec White,",
        "  David Stuck, Shaama M.S., Shane Yost, Joonho Lee, David Small,",
        "  Daniel Levine, Susi Lehtola, Hugh Burton, Evgeny Epifanovsky,",
        "  Bang C. Huynh",
        " -----------------------------------------------------------------------",
        " Exchange:     0.1957 Hartree-Fock + 1.0000 wB97X-D3 + LR-HF",
        " Correlation:  1.0000 wB97X-D3",
        " Using SG-2 standard quadrature grid",
        " Dispersion:   Grimme D3",
        " ",
        " Using the SMD solvation model",
        " Solvent: water",
        " ",
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
        " SCF   energy =   -75.32080770",
        " Total energy =   -75.31846024",
        " ",
        " --------------------------------------------------------------",
        " ",
        "                    Orbital Energies (a.u.)",
        " --------------------------------------------------------------",
    ]


def test_parse_converged_scf(
    parser: ScfParser, sample_scf_block: list[str], initial_data: _MutableCalculationData
) -> None:
    """Test parsing a standard converged SCF block."""
    line_iter = iter(sample_scf_block)
    # Find the start line using the parser's logic
    start_line = next(line for line in sample_scf_block if parser.matches(line, initial_data))
    # Recreate iterator starting from the line after the start line
    line_iter = iter(sample_scf_block[sample_scf_block.index(start_line) + 1 :])
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True
    assert results.scf.n_iterations == 7
    assert results.scf.energy == pytest.approx(-75.31188446)
    assert len(results.scf.iterations) == 7
    assert results.scf.iterations[0] == ScfIteration(
        iteration=1, energy=-75.0734525425, diis_error=3.82e-01, step_type=None
    )
    assert results.scf.iterations[-1] == ScfIteration(
        iteration=7, energy=-75.3118844639, diis_error=5.35e-08, step_type=None
    )


def test_parse_unconverged_scf(
    parser: ScfParser, sample_scf_block_unconverged: list[str], initial_data: _MutableCalculationData
) -> None:
    """Test parsing an unconverged SCF block."""
    line_iter = iter(sample_scf_block_unconverged)
    # Find the start line using the parser's logic
    start_line = next(line for line in sample_scf_block_unconverged if parser.matches(line, initial_data))
    # Recreate iterator starting from the line after the start line
    line_iter = iter(sample_scf_block_unconverged[sample_scf_block_unconverged.index(start_line) + 1 :])
    results = initial_data

    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is False  # Should be false if convergence line is missing
    assert results.scf.n_iterations == 4
    # Should use the energy from the last iteration when final energy line is missing
    assert results.scf.energy == pytest.approx(-75.3118637269)
    assert len(results.scf.iterations) == 4
    assert results.scf.iterations[0] == ScfIteration(
        iteration=1, energy=-75.0734525425, diis_error=3.82e-01, step_type=None
    )
    assert results.scf.iterations[-1] == ScfIteration(
        iteration=4, energy=-75.3118637269, diis_error=1.29e-03, step_type=None
    )


def test_parse_scf_with_smd_summary(
    parser: ScfParser, sample_scf_block_with_smd: list[str], initial_data: _MutableCalculationData
) -> None:
    """Test parsing an SCF block that includes a full SMD summary."""
    line_iter = iter(sample_scf_block_with_smd)
    # Find the start line using the parser's logic
    start_line = next(line for line in sample_scf_block_with_smd if parser.matches(line, initial_data))
    # Recreate iterator starting from the line after the start line
    line_iter = iter(sample_scf_block_with_smd[sample_scf_block_with_smd.index(start_line) + 1 :])
    results = initial_data

    results.qchem_version = "6.2"
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


def test_parse_smd_summary_mismatched_genp(
    parser: ScfParser, sample_scf_block_smd_mismatched_genp: list[str], initial_data: _MutableCalculationData, caplog
) -> None:
    """Test warning for mismatch between SMD G_ENP and explicit SCF energy."""
    line_iter = iter(sample_scf_block_smd_mismatched_genp)
    # Find the start line using the parser's logic
    start_line = next(line for line in sample_scf_block_smd_mismatched_genp if parser.matches(line, initial_data))
    # Recreate iterator starting from the line after the start line
    line_iter = iter(sample_scf_block_smd_mismatched_genp[sample_scf_block_smd_mismatched_genp.index(start_line) + 1 :])
    results = initial_data

    results.qchem_version = "6.2"
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
    # Find the start line using the parser's logic
    start_line = next(line for line in sample_scf_block_smd_partial_summary if parser.matches(line, initial_data))
    # Recreate iterator starting from the line after the start line
    line_iter = iter(sample_scf_block_smd_partial_summary[sample_scf_block_smd_partial_summary.index(start_line) + 1 :])
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
    # Find the start line using the parser's logic
    start_line = next(line for line in sample_scf_block_smd_malformed_summary if parser.matches(line, initial_data))
    # Recreate iterator starting from the line after the start line
    line_iter = iter(
        sample_scf_block_smd_malformed_summary[sample_scf_block_smd_malformed_summary.index(start_line) + 1 :]
    )
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
    # Find the start line using the parser's logic
    start_line = next(line for line in scf_block_smd_no_final_scf if parser.matches(line, initial_data))
    # Recreate iterator starting from the line after the start line
    line_iter = iter(scf_block_smd_no_final_scf[scf_block_smd_no_final_scf.index(start_line) + 1 :])
    results = initial_data
    results.qchem_version = "6.2"
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


def test_parse_multiple_scf_segments(
    parser: ScfParser, scf_block_multiple_segments: list[str], initial_data: _MutableCalculationData
) -> None:
    """Test parsing an SCF block with multiple iteration tables (segments)."""
    line_iter = iter(scf_block_multiple_segments)
    # Find the start line using the parser's logic
    start_line = next(line for line in scf_block_multiple_segments if parser.matches(line, initial_data))
    # Recreate iterator starting from the line after the start line
    line_iter = iter(scf_block_multiple_segments[scf_block_multiple_segments.index(start_line) + 1 :])
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


def test_parse_sample_smd_h2o_sp_sto(
    parser: ScfParser, sample_smd_h2o_sp_sto: list[str], initial_data: _MutableCalculationData
) -> None:
    """Test parsing the specific SMD H2O block from example output and verify all values."""
    line_iter = iter(sample_smd_h2o_sp_sto)
    # Find the start line using the parser's logic
    start_line = next(line for line in sample_smd_h2o_sp_sto if parser.matches(line, initial_data))
    # Recreate iterator starting from the line after the start line
    line_iter = iter(sample_smd_h2o_sp_sto[sample_smd_h2o_sp_sto.index(start_line) + 1 :])
    results = initial_data
    results.qchem_version = "6.2"
    parser.parse(line_iter, start_line, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True
    assert results.scf.n_iterations == 7
    assert results.scf.energy == pytest.approx(-75.32080770)  # This should be G_ENP from the explicit line

    assert len(results.scf.iterations) == 7
    # Verify specific iteration values
    assert results.scf.iterations[0] == ScfIteration(
        iteration=1, energy=-75.0734525440, diis_error=3.82e-01, step_type=None
    )
    assert results.scf.iterations[1] == ScfIteration(
        iteration=2, energy=-75.2973764631, diis_error=4.85e-02, step_type=None
    )
    assert results.scf.iterations[2] == ScfIteration(
        iteration=3, energy=-75.3188175820, diis_error=1.63e-02, step_type=None
    )
    assert results.scf.iterations[3] == ScfIteration(
        iteration=4, energy=-75.3207889318, diis_error=1.19e-03, step_type=None
    )
    assert results.scf.iterations[4] == ScfIteration(
        iteration=5, energy=-75.3208073942, diis_error=1.25e-04, step_type=None
    )
    assert results.scf.iterations[5] == ScfIteration(
        iteration=6, energy=-75.3208076933, diis_error=2.28e-05, step_type=None
    )
    assert results.scf.iterations[6] == ScfIteration(
        iteration=7, energy=-75.3208077035, diis_error=2.17e-08, step_type=None
    )

    # Verify SMD specific values
    assert results.smd_g_pcm_kcal_mol == pytest.approx(-6.0201)
    assert results.smd_g_cds_kcal_mol == pytest.approx(1.4731)
    assert results.smd_g_enp_au == pytest.approx(-75.32080770)
    assert results.smd_g_tot_au == pytest.approx(-75.31846024)
    assert results.final_energy == pytest.approx(-75.31846024)  # Verify the 'Total energy' line is also parsed

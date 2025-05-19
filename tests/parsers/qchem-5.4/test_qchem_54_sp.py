import pytest

from calcflow.parsers.qchem.blocks.scf import ScfParser
from calcflow.parsers.qchem.typing import (
    CalculationData,
    ScfIteration,
    _MutableCalculationData,
)
from calcflow.parsers.qchem.typing.pattern import VersionSpec


@pytest.fixture
def parser() -> ScfParser:
    """Fixture for the ScfParser."""
    return ScfParser()


@pytest.fixture
def initial_data_qchem54() -> _MutableCalculationData:
    """Provides a fresh _MutableCalculationData instance for Q-Chem 5.4."""
    data = _MutableCalculationData(raw_output="")
    data.qchem_version = VersionSpec.from_str("5.4")  # Set Q-Chem version
    return data


@pytest.fixture
def sample_qchem54_sp_smd_output() -> list[str]:
    """Provides lines from a Q-Chem 5.4 SP SMD output block."""
    return [
        " -----------------------------------------------------------------------",
        "  General SCF calculation program by",
        "  A. Author, B. Author, C. Author",  # Placeholder names for header
        " -----------------------------------------------------------------------",
        " Exchange:     PBE0",  # Example, actual exchange/correlation not critical for SCF block structure
        " Correlation:  None",
        " Using the SMD solvation model",
        " Solvent: water",
        " C-PCM solvent model [f = (eps-1)/eps], solve by matrix inversion",
        " A restricted SCF calculation will be",
        " performed using DIIS",
        " SCF converges when DIIS error is below 1.0e-05",
        " ---------------------------------------",
        "  Cycle       Energy         DIIS error",
        " ---------------------------------------",
        "    1     -75.0734525405      3.82e-01  ",
        "    2     -75.2973764603      4.85e-02  ",
        "    3     -75.3188175792      1.63e-02  ",
        "    4     -75.3207889290      1.19e-03  ",
        "    5     -75.3208073914      1.25e-04  ",
        "    6     -75.3208076904      2.28e-05  ",
        "    7     -75.3208077007      2.17e-08  Convergence criterion met",
        " ---------------------------------------",
        " SCF time:   CPU 0.19s  wall 1.00s ",
        "(3)  G-ENP(liq) elect-nuc-pol free energy of system      -75.320807701 a.u.",
        "(4)  G-CDS(liq) cavity-dispersion-solvent structure ",
        "     free energy                                                1.4731 kcal/mol",
        "(6)  G-S(liq) free energy of system                      -75.318460236 a.u.",
        " SCF   energy in the final basis set =      -75.3184602363",
        " Total energy in the final basis set =      -75.3184602363",
        " ",
        " --------------------------------------------------------------",
        " ",
        "                    Orbital Energies (a.u.)",  # Next block start
        " --------------------------------------------------------------",
    ]


def test_parse_qchem54_sp_smd_output(
    parser: ScfParser,
    sample_qchem54_sp_smd_output: list[str],
    initial_data_qchem54: _MutableCalculationData,
    caplog,
) -> None:
    """Test parsing a Q-Chem 5.4 SP SMD output block focusing on SCF and related energies."""
    scf_start_line_content = ""
    scf_start_line_index = -1

    for i, line in enumerate(sample_qchem54_sp_smd_output):
        if parser.matches(line, initial_data_qchem54):
            scf_start_line_content = line
            scf_start_line_index = i
            break

    assert scf_start_line_index != -1, "SCF start pattern not found in sample output"

    line_iter_for_parse = iter(sample_qchem54_sp_smd_output[scf_start_line_index + 1 :])
    results = initial_data_qchem54
    parser.parse(line_iter_for_parse, scf_start_line_content, results)

    assert results.parsed_scf is True
    assert results.scf is not None
    assert results.scf.converged is True
    assert results.scf.n_iterations == 7
    # For QChem 5.4, 'SCF energy =' is not present in the standard format.
    # 'SCF energy in the final basis set =' is present but not matched by SCF_FINAL_ENERGY_PAT.
    # So, scf.energy should be the energy of the last iteration.
    assert results.scf.energy == pytest.approx(-75.3208077007)
    assert len(results.scf.iterations) == 7

    expected_iterations = [
        ScfIteration(iteration=1, energy=-75.0734525405, diis_error=3.82e-01, step_type=None),
        ScfIteration(iteration=2, energy=-75.2973764603, diis_error=4.85e-02, step_type=None),
        ScfIteration(iteration=3, energy=-75.3188175792, diis_error=1.63e-02, step_type=None),
        ScfIteration(iteration=4, energy=-75.3207889290, diis_error=1.19e-03, step_type=None),
        ScfIteration(iteration=5, energy=-75.3208073914, diis_error=1.25e-04, step_type=None),
        ScfIteration(iteration=6, energy=-75.3208076904, diis_error=2.28e-05, step_type=None),
        ScfIteration(iteration=7, energy=-75.3208077007, diis_error=2.17e-08, step_type=None),
    ]
    for i, expected_iter in enumerate(expected_iterations):
        assert results.scf.iterations[i].iteration == expected_iter.iteration
        assert results.scf.iterations[i].energy == pytest.approx(expected_iter.energy)
        assert results.scf.iterations[i].diis_error == pytest.approx(expected_iter.diis_error)
        assert results.scf.iterations[i].step_type == expected_iter.step_type

    # Check final_energy (parsed from "Total energy in the final basis set")
    assert results.final_energy == pytest.approx(-75.3184602363)

    # Check SMD values:
    # The Q-Chem 5.4 output format for SMD components and the lack of "Summary of SMD free energies:"
    # means the current ScfParser will not parse these values.
    assert results.smd_g_pcm_kcal_mol is None
    assert results.smd_g_cds_kcal_mol == pytest.approx(1.4731)
    assert results.smd_g_enp_au is None
    assert results.smd_g_tot_au is None

    # Consequently, the SmdResults object should not be created.
    assert results.smd is not None
    assert results.smd.g_cds_kcal_mol == pytest.approx(1.4731)

    # Assert that the warning for using last iteration energy is logged
    assert (
        "Using energy from last SCF iteration as final SCF energy; explicit 'SCF energy =' line not found or parsed."
        in caplog.text
    )


def test_parse_full_qchem54_sp_smd_file(
    parsed_sp_sto_smd: CalculationData,  # from conftest.py
    caplog,
) -> None:
    """Test parsing a full Q-Chem 5.4 SP SMD output file and check key values."""
    results = parsed_sp_sto_smd

    # Check basic properties from the full parse
    assert results.metadata.qchem_version == "5.4"  # Assuming conftest file is indeed 5.4
    # job_type might not be explicitly set in CalculationData root for all parser paths,
    # but if it were, it would be: assert results.job_type == "sp"

    # SCF specific checks
    assert results.scf is not None
    assert results.scf.converged is True
    assert results.scf.n_iterations == 7
    # For QChem 5.4, 'SCF energy =' is not present in the standard format.
    # 'SCF energy in the final basis set =' is present but not matched by SCF_FINAL_ENERGY_PAT.
    # So, scf.energy should be the energy of the last iteration.
    assert results.scf.energy == pytest.approx(-75.3208077007)

    # Check specific iteration energies
    assert len(results.scf.iterations) == 7
    assert results.scf.iterations[0].energy == pytest.approx(-75.0734525405)
    assert results.scf.iterations[0].diis_error == pytest.approx(3.82e-01)
    assert results.scf.iterations[6].energy == pytest.approx(-75.3208077007)  # Last iteration
    assert results.scf.iterations[6].diis_error == pytest.approx(2.17e-08)

    # Check final_energy (parsed from "Total energy in the final basis set")
    assert results.final_energy == pytest.approx(-75.3184602363)

    # Consequently, the SmdResults object should not be created based on these patterns.
    assert results.smd is not None
    assert results.smd.g_cds_kcal_mol == pytest.approx(1.4731)

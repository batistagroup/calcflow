from pathlib import Path

import pytest

from calcflow.parsers import qchem
from calcflow.parsers.qchem.typing import CalculationData, MomCalculationResult, _MutableCalculationData

# Load the example output file content once
ex_folder = Path(__file__).resolve().parents[3] / "data" / "calculations" / "examples" / "qchem"
EXAMPLE_SP_OUT_PATH = ex_folder / "h2o" / "sp-sto.out"


@pytest.fixture(scope="module")
def parsed_sp_sto_data() -> CalculationData:
    """Fixture to parse the standard single point output file."""
    return qchem.parse_qchem_sp_output(EXAMPLE_SP_OUT_PATH.read_text())


@pytest.fixture(scope="module")
def parsed_sp_sto_smd() -> CalculationData:
    """Fixture to parse the standard single point output file."""
    return qchem.parse_qchem_sp_output((ex_folder / "h2o" / "sp-sto-smd.out").read_text())


@pytest.fixture(scope="module")
def parsed_sp_tzvppd_smd() -> CalculationData:
    """Fixture to parse the standard single point output file."""
    return qchem.parse_qchem_sp_output((ex_folder / "h2o" / "sp-tzvppd-smd.out").read_text())


@pytest.fixture(scope="module")
def parsed_tddft_pc2_data() -> CalculationData:
    """Fixture to parse the standard single point output file."""
    return qchem.parse_qchem_tddft_output((ex_folder / "h2o" / "tddft-rks-pc2.out").read_text())


@pytest.fixture(scope="module")
def parsed_tddft_uks_pc2_data() -> CalculationData:
    """Fixture to parse the UKS TDDFT output file for H2O (pc2)."""
    return qchem.parse_qchem_tddft_output((ex_folder / "h2o" / "tddft-uks-pc2.out").read_text())


@pytest.fixture(scope="module")
def parsed_mom_sp_data() -> MomCalculationResult:
    """Fixture to parse the MOM output file for H2O."""
    return qchem.parse_qchem_mom_output((ex_folder / "h2o" / "mom-sp.out").read_text())


@pytest.fixture(scope="module")
def parsed_mom_smd_sp_data() -> MomCalculationResult:
    """Fixture to parse the MOM output file for H2O."""
    return qchem.parse_qchem_mom_output((ex_folder / "h2o" / "mom-smd-sp.out").read_text())


@pytest.fixture
def mutable_data() -> _MutableCalculationData:
    """Provides a fresh mutable data object for testing."""
    return _MutableCalculationData(raw_output=EXAMPLE_SP_OUT_PATH.read_text())

import logging
from dataclasses import replace

import pytest

from calcflow.geometry.static import Geometry
from calcflow.inputs.qchem import QchemInput
from calcflow.utils import logger

logger.setLevel(logging.CRITICAL)


@pytest.fixture
def h2o_geometry() -> Geometry:
    """Fixture for a simple water molecule Geometry object."""
    atoms = [
        ("O", (0.0000, 0.0000, 0.1173)),
        ("H", (0.0000, 0.7572, -0.4692)),
        ("H", (0.0000, -0.7572, -0.4692)),
    ]
    return Geometry(num_atoms=len(atoms), comment="Water molecule", atoms=atoms)


@pytest.fixture
def default_qchem_input() -> QchemInput:
    """Fixture for a default QchemInput instance."""
    return QchemInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="hf",
        basis_set="sto-3g",
    )


@pytest.fixture
def mom_enabled_input(default_qchem_input: QchemInput) -> QchemInput:
    """Fixture for a QchemInput with MOM enabled (IMOM method)."""
    return default_qchem_input.enable_mom(method="IMOM").set_basis("def2-svp")  # Needs a real basis


@pytest.fixture
def unrestricted_input(default_qchem_input: QchemInput) -> QchemInput:
    """Fixture for a QchemInput that is unrestricted."""
    return replace(default_qchem_input, unrestricted=True, basis_set="def2-svp")

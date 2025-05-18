import logging
import re
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


class Helpers:
    @staticmethod
    def parse_qchem_rem_section(section_string: str) -> dict[str, str]:
        """
        Parses a Q-Chem input section (like $rem, $solvent, $smx) into a dict.
        Assumes section_string is the content *between* the $section_name and $end lines,
        with leading/trailing whitespace already stripped from the overall section content.
        """
        parsed_data: dict[str, str] = {}
        lines = section_string.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Split only on the first occurrence of one or more whitespace chars
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts) == 2:
                key = parts[0].upper()
                # Keep original case for values (e.g., True/False, solvent names, basis set names)
                # The QchemInput class itself handles desired casing for output.
                value = parts[1]
                parsed_data[key] = value
            # If a line doesn't split into two parts (e.g. a malformed line or a standalone keyword
            # without a value, which is uncommon for the sections this parser targets),
            # it's ignored. This behavior can be adjusted if needed.
        return parsed_data


@pytest.fixture
def helpers() -> Helpers:
    """Fixture for the Helpers class."""
    return Helpers

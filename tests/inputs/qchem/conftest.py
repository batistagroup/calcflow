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

    @staticmethod
    def split_qchem_input_into_blocks(input_str: str) -> dict[str, str]:
        """
        Splits a Q-Chem input string (single job) into a dictionary of blocks.
        Keys are block names (lowercase, e.g., "molecule", "rem").
        Values are the inner content of the blocks, stripped.
        """
        blocks: dict[str, str] = {}
        # Regex to find blocks like $blockname ... $end
        # (?is) flags: i for case-insensitive block names, s for . to match newlines
        pattern = re.compile(
            r"^\$(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*?\n(?P<content>.*?)\n^\$end", re.MULTILINE | re.DOTALL
        )

        for match in pattern.finditer(input_str):
            block_name = match.group("name").lower()
            block_content = match.group("content").strip()
            blocks[block_name] = block_content
        return blocks

    @staticmethod
    def compare_qchem_input_files(actual_input_str: str, expected_input_str: str) -> None:
        """Compares two Q-Chem input file strings block by block."""
        actual_blocks = Helpers.split_qchem_input_into_blocks(actual_input_str)
        expected_blocks = Helpers.split_qchem_input_into_blocks(expected_input_str)

        # Compare block names
        actual_block_names = set(actual_blocks.keys())
        expected_block_names = set(expected_blocks.keys())
        assert actual_block_names == expected_block_names, (
            f"Mismatched block names. Actual: {sorted(list(actual_block_names))}, Expected: {sorted(list(expected_block_names))}"
        )

        # Compare content of each block
        for name in expected_block_names:
            actual_content = actual_blocks[name]
            expected_content = expected_blocks[name]

            if name in {"rem", "solvent", "smx", "occupied"}:  # Use parse_qchem_rem_section for these
                actual_parsed = Helpers.parse_qchem_rem_section(actual_content)
                expected_parsed = Helpers.parse_qchem_rem_section(expected_content)
                assert actual_parsed == expected_parsed, (
                    f"Mismatched parsed content in ${name} block.\nActual: {actual_parsed}\nExpected: {expected_parsed}"
                )
            elif name == "basis":
                # Special handling for basis block: check line by line after stripping
                actual_lines = [line.strip() for line in actual_content.splitlines() if line.strip()]
                expected_lines = [line.strip() for line in expected_content.splitlines() if line.strip()]
                # Normalize wildcard lines like **** which can have varying whitespace
                actual_lines = [re.sub(r"\s*", "", line) if line.strip() == "****" else line for line in actual_lines]
                expected_lines = [
                    re.sub(r"\s*", "", line) if line.strip() == "****" else line for line in expected_lines
                ]

                assert actual_lines == expected_lines, (
                    f"Mismatched lines in ${name} block (line-by-line comparison).\nActual lines:\n{actual_content}\nExpected lines:\n{expected_content}"
                )
            else:  # Default to line-by-line comparison for other blocks like $molecule, $solute
                # Strip leading/trailing whitespace from each line
                actual_lines = [line.strip() for line in actual_content.splitlines()]
                expected_lines = [line.strip() for line in expected_content.splitlines()]

                # Remove entirely empty lines for comparison robustness
                actual_lines = [line for line in actual_lines if line]
                expected_lines = [line for line in expected_lines if line]

                # For molecule block coordinates, split by whitespace for flexibility
                if name == "molecule" and len(actual_lines) > 1 and len(expected_lines) > 1:
                    # Compare the first line (charge multiplicity) directly
                    assert actual_lines[0] == expected_lines[0], (
                        f"Mismatched first line in ${name} block.\nActual: {actual_lines[0]}\nExpected: {expected_lines[0]}"
                    )

                    # Compare coordinate lines by splitting on whitespace
                    actual_coords = [line.split() for line in actual_lines[1:]]
                    expected_coords = [line.split() for line in expected_lines[1:]]

                    assert actual_coords == expected_coords, (
                        f"Mismatched coordinate lines in ${name} block.\nActual:\n{actual_content}\nExpected:\n{expected_content}"
                    )
                else:
                    # Fallback to simple line comparison for other blocks or simple molecule blocks
                    assert actual_lines == expected_lines, (
                        f"Mismatched lines in ${name} block (line-by-line comparison).\nActual lines:\n{actual_content}\nExpected lines:\n{expected_content}"
                    )


@pytest.fixture
def helpers() -> Helpers:
    """Fixture for the Helpers class."""
    return Helpers

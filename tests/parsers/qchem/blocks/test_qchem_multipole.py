import logging

import pytest
from pytest import LogCaptureFixture

from calcflow.parsers.qchem.blocks.multipole import MultipoleParser
from calcflow.parsers.qchem.typing import (
    DipoleMoment,
    HexadecapoleMoment,
    OctopoleMoment,
    QuadrupoleMoment,
    _MutableCalculationData,
)


@pytest.fixture
def parser() -> MultipoleParser:
    """Fixture for the MultipoleParser."""
    return MultipoleParser()


@pytest.fixture
def results() -> _MutableCalculationData:
    """Fixture for the mutable calculation data."""
    return _MutableCalculationData(raw_output="")


# --- Test Data ---
MULTIPOLE_SECTION_H2O = """
 -----------------------------------------------------------------
                    Cartesian Multipole Moments
 -----------------------------------------------------------------
    Charge (ESU x 10^10)
                -0.0000
    Dipole Moment (Debye)
         X      -0.8135      Y      -0.1666      Z      -1.4237
       Tot       1.6482
    Quadrupole Moments (Debye-Ang)
        XX      -8.3319     XY      -1.9775     YY      -6.5229
        XZ      -3.5103     YZ      -1.9205     ZZ      -4.7954
    Octopole Moments (Debye-Ang^2)
       XXX     -44.7808    XXY     -15.2389    XYY     -17.7938
       YYY     -29.1521    XXZ      -8.0908    XYZ      -4.6554
       YYZ      -1.6923    XZZ     -10.5109    YZZ      -7.6527
       ZZZ       1.7889
    Hexadecapole Moments (Debye-Ang^3)
      XXXX    -193.1184   XXXY     -75.8577   XXYY     -59.8295
      XYYY     -71.1532   YYYY     -92.7422   XXXZ     -16.9182
      XXYZ     -10.5382   XYYZ      -3.9580   YYYZ       0.3498
      XXZZ     -25.0172   XYZZ     -16.7675   YYZZ     -13.8075
      XZZZ       4.1195   YZZZ       2.6781   ZZZZ      -5.5277
 -----------------------------------------------------------------
 Some other line
""".strip().splitlines()

MULTIPOLE_SECTION_MISSING_HEXA = """
 -----------------------------------------------------------------
                    Cartesian Multipole Moments
 -----------------------------------------------------------------
    Charge (ESU x 10^10)
                -0.0000
    Dipole Moment (Debye)
         X      -0.8135      Y      -0.1666      Z      -1.4237
       Tot       1.6482
    Quadrupole Moments (Debye-Ang)
        XX      -8.3319     XY      -1.9775     YY      -6.5229
        XZ      -3.5103     YZ      -1.9205     ZZ      -4.7954
    Octopole Moments (Debye-Ang^2)
       XXX     -44.7808    XXY     -15.2389    XYY     -17.7938
       YYY     -29.1521    XXZ      -8.0908    XYZ      -4.6554
       YYZ      -1.6923    XZZ     -10.5109    YZZ      -7.6527
       ZZZ       1.7889
 -----------------------------------------------------------------
 Some other line
""".strip().splitlines()

MULTIPOLE_SECTION_INVALID_DIPOLE = """
 -----------------------------------------------------------------
                    Cartesian Multipole Moments
 -----------------------------------------------------------------
    Charge (ESU x 10^10)
                -0.0000
    Dipole Moment (Debye)
         X      -0.8135      Y      INVALID    Z      -1.4237
       Tot       1.6482
 -----------------------------------------------------------------
 Some other line
""".strip().splitlines()

MULTIPOLE_SECTION_CHARGE_ONLY = """
 -----------------------------------------------------------------
                    Cartesian Multipole Moments
 -----------------------------------------------------------------
    Charge (ESU x 10^10)
                -0.0000
 -----------------------------------------------------------------
 Some other line
""".strip().splitlines()


# --- Test Cases ---


def test_multipole_matches_start(parser: MultipoleParser, results: _MutableCalculationData) -> None:
    """Verify matches method identifies the start line."""
    line = "                    Cartesian Multipole Moments"
    assert parser.matches(line, results) is True


def test_multipole_matches_already_parsed(parser: MultipoleParser, results: _MutableCalculationData) -> None:
    """Verify matches method returns False if already parsed."""
    line = "                    Cartesian Multipole Moments"
    results.parsed_multipole = True
    assert parser.matches(line, results) is False


def test_multipole_matches_no_match(parser: MultipoleParser, results: _MutableCalculationData) -> None:
    """Verify matches method returns False for non-matching lines."""
    line = "SCF   energy =   -75.31188446"
    assert parser.matches(line, results) is False


def test_multipole_parse_full_section(parser: MultipoleParser, results: _MutableCalculationData) -> None:
    """Test parsing a complete multipole section."""
    lines = iter(MULTIPOLE_SECTION_H2O)
    next(lines)  # Consume the '---' line
    multipole_header = next(lines)  # Consume the 'Cartesian...' line
    parser.parse(lines, multipole_header, results)

    assert results.parsed_multipole is True
    assert results.multipole is not None
    assert results.multipole.charge_esu == -0.0000

    assert results.multipole.dipole == DipoleMoment(x=-0.8135, y=-0.1666, z=-1.4237, magnitude=1.6482)
    assert results.multipole.quadrupole == QuadrupoleMoment(
        xx=-8.3319, xy=-1.9775, yy=-6.5229, xz=-3.5103, yz=-1.9205, zz=-4.7954
    )
    assert results.multipole.octopole == OctopoleMoment(
        xxx=-44.7808,
        xxy=-15.2389,
        xyy=-17.7938,
        yyy=-29.1521,
        xxz=-8.0908,
        xyz=-4.6554,
        yyz=-1.6923,
        xzz=-10.5109,
        yzz=-7.6527,
        zzz=1.7889,
    )
    assert results.multipole.hexadecapole == HexadecapoleMoment(
        xxxx=-193.1184,
        xxxy=-75.8577,
        xxyy=-59.8295,
        xyyy=-71.1532,
        yyyy=-92.7422,
        xxxz=-16.9182,
        xxyz=-10.5382,
        xyyz=-3.9580,
        yyyz=0.3498,
        xxzz=-25.0172,
        xyzz=-16.7675,
        yyzz=-13.8075,
        xzzz=4.1195,
        yzzz=2.6781,
        zzzz=-5.5277,
    )
    assert results.parsing_errors == []

    # Check that the iterator stopped at the correct line
    assert next(lines) == " Some other line"


def test_multipole_parse_missing_hexadecapole(parser: MultipoleParser, results: _MutableCalculationData) -> None:
    """Test parsing when hexadecapole moments are missing."""
    lines = iter(MULTIPOLE_SECTION_MISSING_HEXA)
    next(lines)
    multipole_header = next(lines)
    parser.parse(lines, multipole_header, results)

    assert results.parsed_multipole is True
    assert results.multipole is not None
    assert results.multipole.charge_esu == -0.0000
    assert results.multipole.dipole is not None
    assert results.multipole.quadrupole is not None
    assert results.multipole.octopole is not None
    assert results.multipole.hexadecapole is None  # Should be None
    assert results.parsing_errors == []
    assert next(lines) == " Some other line"


def test_multipole_parse_invalid_value(
    parser: MultipoleParser, results: _MutableCalculationData, caplog: LogCaptureFixture
) -> None:
    """Test parsing with an invalid numeric value."""
    lines = iter(MULTIPOLE_SECTION_INVALID_DIPOLE)
    next(lines)
    multipole_header = next(lines)

    with caplog.at_level(logging.WARNING):
        parser.parse(lines, multipole_header, results)

    assert results.parsed_multipole is True  # Still marked as parsed
    assert results.multipole is not None
    assert results.multipole.charge_esu == -0.0000
    assert results.multipole.dipole is None  # Dipole parsing failed
    assert results.multipole.quadrupole is None  # Subsequent sections not parsed
    assert results.multipole.octopole is None
    assert results.multipole.hexadecapole is None
    assert "Could not parse dipole moment values." in caplog.text
    # The exact error logged by the float conversion might vary, so we check the parser warning.
    # We expect no parsing errors added to the list, as the code logs warnings currently.
    assert results.parsing_errors == []
    assert next(lines) == " Some other line"


def test_multipole_parse_stop_iteration(
    parser: MultipoleParser, results: _MutableCalculationData, caplog: LogCaptureFixture
) -> None:
    """Test handling of StopIteration if file ends prematurely."""
    # Only provide the start and charge sections
    incomplete_section = MULTIPOLE_SECTION_H2O[:5]
    lines = iter(incomplete_section)
    next(lines)
    multipole_header = next(lines)

    with caplog.at_level(logging.WARNING):
        parser.parse(lines, multipole_header, results)

    assert results.parsed_multipole is True  # Still marked parsed if charge found
    assert results.multipole is not None
    assert results.multipole.charge_esu == -0.0000
    assert results.multipole.dipole is None
    assert results.multipole.quadrupole is None
    assert results.multipole.octopole is None
    assert results.multipole.hexadecapole is None
    assert "File ended unexpectedly while parsing Multipole Moments section." in caplog.text
    assert results.parsing_errors == []


def test_multipole_parse_charge_only(parser: MultipoleParser, results: _MutableCalculationData) -> None:
    """Test parsing when only the charge is present."""
    lines = iter(MULTIPOLE_SECTION_CHARGE_ONLY)
    next(lines)
    multipole_header = next(lines)
    parser.parse(lines, multipole_header, results)

    assert results.parsed_multipole is True
    assert results.multipole is not None
    assert results.multipole.charge_esu == -0.0000
    assert results.multipole.dipole is None
    assert results.multipole.quadrupole is None
    assert results.multipole.octopole is None
    assert results.multipole.hexadecapole is None
    assert results.parsing_errors == []
    assert next(lines) == " Some other line"


def test_multipole_parse_no_data_found(
    parser: MultipoleParser, results: _MutableCalculationData, caplog: LogCaptureFixture
) -> None:
    """Test parsing an empty or non-conforming section."""
    empty_section = [
        " -----------------------------------------------------------------",
        "                    Cartesian Multipole Moments",
        " -----------------------------------------------------------------",
        " Some other line",
    ]
    lines = iter(empty_section)
    next(lines)
    multipole_header = next(lines)

    with caplog.at_level(logging.WARNING):
        parser.parse(lines, multipole_header, results)

    assert results.parsed_multipole is False  # Not marked parsed if nothing found
    assert results.multipole is None
    assert "Did not find any multipole moment data in the section." in caplog.text
    assert results.parsing_errors == []

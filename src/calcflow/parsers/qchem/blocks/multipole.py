import re

from calcflow.parsers.qchem.typing import (
    DipoleMomentData,
    HexadecapoleMoments,
    LineIterator,
    MultipoleData,
    OctopoleMoments,
    QuadrupoleMoments,
    SectionParser,
    _MutableCalculationData,
)
from calcflow.utils import logger

# --- Regex Patterns ---
MULTIPOLE_START_PAT = re.compile(r"^\s*Cartesian Multipole Moments\s*$")
CHARGE_PAT = re.compile(r"^\s*Charge \(ESU x 10\^10\)\s*$")
CHARGE_VALUE_PAT = re.compile(r"^\s*(-?\d+\.\d+)\s*$")
DIPOLE_HEADER_PAT = re.compile(r"^\s*Dipole Moment \(Debye\)\s*$")
DIPOLE_VALUES_PAT = re.compile(r"^\s*X\s+(-?\d+\.\d+)\s+Y\s+(-?\d+\.\d+)\s+Z\s+(-?\d+\.\d+)\s*$")
TOTAL_DIPOLE_PAT = re.compile(r"^\s*Tot\s+(-?\d+\.\d+)\s*$")
QUAD_HEADER_PAT = re.compile(r"^\s*Quadrupole Moments \(Debye-Ang\)\s*$")
QUAD_XX_XY_YY_PAT = re.compile(r"^\s*XX\s+(-?\d+\.\d+)\s+XY\s+(-?\d+\.\d+)\s+YY\s+(-?\d+\.\d+)\s*$")
QUAD_XZ_YZ_ZZ_PAT = re.compile(r"^\s*XZ\s+(-?\d+\.\d+)\s+YZ\s+(-?\d+\.\d+)\s+ZZ\s+(-?\d+\.\d+)\s*$")
OCTO_HEADER_PAT = re.compile(r"^\s*Octopole Moments \(Debye-Ang\^2\)\s*$")
OCTO_XXX_XXY_XYY_PAT = re.compile(r"^\s*XXX\s+(-?\d+\.\d+)\s+XXY\s+(-?\d+\.\d+)\s+XYY\s+(-?\d+\.\d+)\s*$")
OCTO_YYY_XXZ_XYZ_PAT = re.compile(r"^\s*YYY\s+(-?\d+\.\d+)\s+XXZ\s+(-?\d+\.\d+)\s+XYZ\s+(-?\d+\.\d+)\s*$")
OCTO_YYZ_XZZ_YZZ_PAT = re.compile(r"^\s*YYZ\s+(-?\d+\.\d+)\s+XZZ\s+(-?\d+\.\d+)\s+YZZ\s+(-?\d+\.\d+)\s*$")
OCTO_ZZZ_PAT = re.compile(r"^\s*ZZZ\s+(-?\d+\.\d+)\s*$")
HEXA_HEADER_PAT = re.compile(r"^\s*Hexadecapole Moments \(Debye-Ang\^3\)\s*$")
HEXA_XXXX_XXXY_XXYY_PAT = re.compile(r"^\s*XXXX\s+(-?\d+\.\d+)\s+XXXY\s+(-?\d+\.\d+)\s+XXYY\s+(-?\d+\.\d+)\s*$")
HEXA_XYYY_YYYY_XXXZ_PAT = re.compile(r"^\s*XYYY\s+(-?\d+\.\d+)\s+YYYY\s+(-?\d+\.\d+)\s+XXXZ\s+(-?\d+\.\d+)\s*$")
HEXA_XXYZ_XYYZ_YYYZ_PAT = re.compile(r"^\s*XXYZ\s+(-?\d+\.\d+)\s+XYYZ\s+(-?\d+\.\d+)\s+YYYZ\s+(-?\d+\.\d+)\s*$")
HEXA_XXZZ_XYZZ_YYZZ_PAT = re.compile(r"^\s*XXZZ\s+(-?\d+\.\d+)\s+XYZZ\s+(-?\d+\.\d+)\s+YYZZ\s+(-?\d+\.\d+)\s*$")
HEXA_XZZZ_YZZZ_ZZZZ_PAT = re.compile(r"^\s*XZZZ\s+(-?\d+\.\d+)\s+YZZZ\s+(-?\d+\.\d+)\s+ZZZZ\s+(-?\d+\.\d+)\s*$")
END_PAT = re.compile(r"^-{20,}")  # End of the multipole section


class MultipoleParser(SectionParser):
    """Parses the Cartesian Multipole Moments section from Q-Chem output."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line marks the start of the multipole section."""
        if current_data.parsed_multipole:
            return False
        return MULTIPOLE_START_PAT.search(line) is not None

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Extracts charge, dipole, quadrupole, octopole, and hexadecapole moments."""
        logger.debug("Entering Multipole Moments section.")

        charge: float | None = None
        dipole: DipoleMomentData | None = None
        quadrupole: QuadrupoleMoments | None = None
        octopole: OctopoleMoments | None = None
        hexadecapole: HexadecapoleMoments | None = None

        try:
            while True:
                current_line = next(iterator)  # Start reading lines within the block

                # End Condition
                if END_PAT.search(current_line):
                    logger.debug("Found end of Multipole Moments section.")
                    break

                # --- Charge ---
                if CHARGE_PAT.search(current_line):
                    line_val = next(iterator)
                    match = CHARGE_VALUE_PAT.search(line_val)
                    if match:
                        charge = float(match.group(1))
                    else:
                        logger.warning(f"Could not parse charge value from: {line_val.strip()}")
                    continue

                # --- Dipole --- #
                if DIPOLE_HEADER_PAT.search(current_line):
                    line_xyz = next(iterator)
                    match_xyz = DIPOLE_VALUES_PAT.search(line_xyz)
                    line_tot = next(iterator)
                    match_tot = TOTAL_DIPOLE_PAT.search(line_tot)
                    if match_xyz and match_tot:
                        dipole = DipoleMomentData(
                            x_debye=float(match_xyz.group(1)),
                            y_debye=float(match_xyz.group(2)),
                            z_debye=float(match_xyz.group(3)),
                            total_debye=float(match_tot.group(1)),
                        )
                    else:
                        logger.warning("Could not parse dipole moment values.")
                    continue

                # --- Quadrupole --- #
                if QUAD_HEADER_PAT.search(current_line):
                    line1 = next(iterator)
                    match1 = QUAD_XX_XY_YY_PAT.search(line1)
                    line2 = next(iterator)
                    match2 = QUAD_XZ_YZ_ZZ_PAT.search(line2)
                    if match1 and match2:
                        quadrupole = QuadrupoleMoments(
                            xx=float(match1.group(1)),
                            xy=float(match1.group(2)),
                            yy=float(match1.group(3)),
                            xz=float(match2.group(1)),
                            yz=float(match2.group(2)),
                            zz=float(match2.group(3)),
                        )
                    else:
                        logger.warning("Could not parse quadrupole moment values.")
                    continue

                # --- Octopole --- #
                if OCTO_HEADER_PAT.search(current_line):
                    line1 = next(iterator)
                    match1 = OCTO_XXX_XXY_XYY_PAT.search(line1)
                    line2 = next(iterator)
                    match2 = OCTO_YYY_XXZ_XYZ_PAT.search(line2)
                    line3 = next(iterator)
                    match3 = OCTO_YYZ_XZZ_YZZ_PAT.search(line3)
                    line4 = next(iterator)
                    match4 = OCTO_ZZZ_PAT.search(line4)
                    if match1 and match2 and match3 and match4:
                        octopole = OctopoleMoments(
                            xxx=float(match1.group(1)),
                            xxy=float(match1.group(2)),
                            xyy=float(match1.group(3)),
                            yyy=float(match2.group(1)),
                            xxz=float(match2.group(2)),
                            xyz=float(match2.group(3)),
                            yyz=float(match3.group(1)),
                            xzz=float(match3.group(2)),
                            yzz=float(match3.group(3)),
                            zzz=float(match4.group(1)),
                        )
                    else:
                        logger.warning("Could not parse octopole moment values.")
                    continue

                # --- Hexadecapole --- #
                if HEXA_HEADER_PAT.search(current_line):
                    line1 = next(iterator)
                    match1 = HEXA_XXXX_XXXY_XXYY_PAT.search(line1)
                    line2 = next(iterator)
                    match2 = HEXA_XYYY_YYYY_XXXZ_PAT.search(line2)
                    line3 = next(iterator)
                    match3 = HEXA_XXYZ_XYYZ_YYYZ_PAT.search(line3)
                    line4 = next(iterator)
                    match4 = HEXA_XXZZ_XYZZ_YYZZ_PAT.search(line4)
                    line5 = next(iterator)
                    match5 = HEXA_XZZZ_YZZZ_ZZZZ_PAT.search(line5)
                    if match1 and match2 and match3 and match4 and match5:
                        hexadecapole = HexadecapoleMoments(
                            xxxx=float(match1.group(1)),
                            xxxy=float(match1.group(2)),
                            xxyy=float(match1.group(3)),
                            xyyy=float(match2.group(1)),
                            yyyy=float(match2.group(2)),
                            xxxz=float(match2.group(3)),
                            xxyz=float(match3.group(1)),
                            xyyz=float(match3.group(2)),
                            yyyz=float(match3.group(3)),
                            xxzz=float(match4.group(1)),
                            xyzz=float(match4.group(2)),
                            yyzz=float(match4.group(3)),
                            xzzz=float(match5.group(1)),
                            yzzz=float(match5.group(2)),
                            zzzz=float(match5.group(3)),
                        )
                    else:
                        logger.warning("Could not parse hexadecapole moment values.")
                    continue

        except StopIteration:
            # Reached end of file before finding the end pattern
            logger.warning("File ended unexpectedly while parsing Multipole Moments section.")
            # Continue with whatever was parsed

        except (ValueError, IndexError) as e:
            logger.error(f"Error converting multipole value: {e}", exc_info=True)
            results.parsing_errors.append(f"Value error during Multipole Moment parsing: {e}")
            # Continue with whatever was parsed, but mark error

        except Exception as e:
            logger.error(f"Unexpected error parsing Multipole Moments: {e}", exc_info=True)
            results.parsing_errors.append(f"Unexpected error in Multipole Parser: {e}")
            # Re-raise critical unexpected errors? For now, just log and continue

        # --- Store Results --- #
        if dipole or quadrupole or octopole or hexadecapole:
            results.multipole = MultipoleData(
                charge_esu=charge,
                dipole=dipole,
                quadrupole=quadrupole,
                octopole=octopole,
                hexadecapole=hexadecapole,
            )
            results.parsed_multipole = True
            logger.debug(f"Parsed Multipole Data: {results.multipole}")
        elif charge is not None:  # Handle case where only charge is printed (unlikely but possible)
            results.multipole = MultipoleData(charge_esu=charge)
            results.parsed_multipole = True
            logger.debug(f"Parsed Multipole Data (charge only): {results.multipole}")
        else:
            logger.warning("Did not find any multipole moment data in the section.")

        logger.debug("Exiting Multipole Moments section.")

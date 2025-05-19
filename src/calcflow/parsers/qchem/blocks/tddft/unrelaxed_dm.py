import re
from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from re import Pattern

from calcflow.parsers.qchem.typing import (
    DipoleMoment,
    ExcitedStateAtomPopulation,
    ExcitedStateDetailedAnalysis,
    ExcitedStateExcitonDifferenceDM,
    ExcitedStateMulliken,
    ExcitedStateMultipole,
    ExcitedStateNOData,
    LineIterator,
    _MutableCalculationData,
)
from calcflow.parsers.qchem.typing.pattern import PatternDefinition
from calcflow.utils import logger


class UnrelaxedDmSubSectionType(Enum):
    NOS_SPIN_TRACED_OR_RKS = "nos_spin_traced_or_rks"
    NOS_ALPHA_BETA_SKIP = "nos_alpha_beta_skip"
    MULLIKEN = "mulliken"
    MULTIPOLE = "multipole"
    EXCITON_MAIN_BLOCK = "exciton_main_block"


# Separate registries for RKS and UKS
UNRELAXED_DM_SECTION_PATTERNS_RKS = [
    PatternDefinition(
        field_name="no_data",
        description="NOs (RKS)",
        versioned_patterns=[
            (re.compile(r"^NOs$"), "6.2.0", None),
        ],
    ),
    PatternDefinition(
        field_name="mulliken",
        description="Mulliken Population Analysis (State/Difference DM)",
        versioned_patterns=[
            (re.compile(r"^Mulliken Population Analysis$"), "5.4.0", None),
            (re.compile(r"^Mulliken Population Analysis \(State/Difference DM\)$"), "6.2.0", None),
        ],
    ),
    PatternDefinition(
        field_name="multipole",
        description="Multipole moment analysis of the density matrix",
        versioned_patterns=[
            (re.compile(r"^Multipole moment analysis of the density matrix$"), "6.2.0", None),
        ],
    ),
    PatternDefinition(
        field_name="exciton_difference_dm_analysis",
        description="Exciton analysis of the difference density matrix",
        versioned_patterns=[
            (re.compile(r"^Exciton analysis of the difference density matrix$"), "6.2.0", None),
        ],
    ),
]

UNRELAXED_DM_SECTION_PATTERNS_UKS = [
    PatternDefinition(
        field_name="no_data",
        description="NOs (spin-traced) (UKS)",
        versioned_patterns=[
            (re.compile(r"^NOs \(spin-traced\)$"), "6.2.0", None),
        ],
    ),
    PatternDefinition(
        field_name="mulliken",
        description="Mulliken Population Analysis (State/Difference DM)",
        versioned_patterns=[
            (re.compile(r"^Mulliken Population Analysis$"), "5.4.0", None),
            (re.compile(r"^Mulliken Population Analysis \(State/Difference DM\)$"), "6.2.0", None),
        ],
    ),
    PatternDefinition(
        field_name="multipole",
        description="Multipole moment analysis of the density matrix",
        versioned_patterns=[
            (re.compile(r"^Multipole moment analysis of the density matrix$"), "6.2.0", None),
        ],
    ),
    PatternDefinition(
        field_name="exciton_difference_dm_analysis",
        description="Exciton analysis of the difference density matrix",
        versioned_patterns=[
            (re.compile(r"^Exciton analysis of the difference density matrix$"), "6.2.0", None),
        ],
    ),
]


class UnrelaxedExcitedStatePropertiesParser:
    """
    Parses the "Analysis of Unrelaxed Density Matrices" section from Q-Chem output.
    This section contains detailed properties for each excited state, such as
    Natural Orbital occupations, Mulliken populations, multipole moments, and
    exciton analysis of the difference density matrix.
    """

    SECTION_START_TOKEN: str = "Analysis of Unrelaxed Density Matrices"
    SECTION_END_TOKEN: str = "Transition Density Matrix Analysis"
    STATE_HEADER_PATTERN: Pattern[str] = re.compile(r"^\s*(Singlet|Triplet|Excited State)\s+(\d+)\s*:")
    XYZ_FLOAT_TUPLE_PATTERN: Pattern[str] = re.compile(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]")

    # Keywords that indicate the start of a new sub-section or end of the current one
    MULLIKEN_SUBSECTION_KEYWORDS: list[str] = [
        "Mulliken Population Analysis",
        "Multipole moment analysis",
        "Exciton analysis",
    ]
    MULTIPOLE_SUBSECTION_KEYWORDS: list[str] = ["Exciton analysis"]
    # NOs, Mulliken, Multipole, Exciton are handled by direct checks for their specific headers in the main parse loop

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Checks if the line marks the beginning of the unrelaxed DM analysis section."""
        return self.SECTION_START_TOKEN in line and not current_data.parsed_unrelaxed_excited_state_properties

    def _get_next_line(self, iterator: LineIterator, results: _MutableCalculationData) -> str | None:
        """Safely gets the next line, respecting the buffer. Returns None if StopIteration."""
        try:
            line = results.buffered_line or next(iterator)
            results.buffered_line = None
            return line
        except StopIteration:
            return None

    def _parse_xyz_float_tuple(self, line: str) -> tuple[float, float, float] | None:
        """Parses a [x, y, z] float tuple from a line."""
        match = self.XYZ_FLOAT_TUPLE_PATTERN.search(line)
        if match:
            try:
                return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
            except ValueError:  # pragma: no cover
                logger.warning(f"Failed to parse XYZ float tuple from: {line}")  # pragma: no cover
        return None  # pragma: no cover

    def _parse_cartesian_components_next_line(
        self, iterator: LineIterator, results: _MutableCalculationData, component_line_prefix: str
    ) -> tuple[float, float, float] | None:
        """
        Parses Cartesian components from the line immediately following a scalar value line.
        Example: "Cartesian components [Ang]:        [  0.591568,   0.481035,   0.567111]"
        """
        next_line = self._get_next_line(iterator, results)
        if next_line is None:
            logger.warning(
                f"EOF while expecting Cartesian components starting with '{component_line_prefix}'."
            )  # pragma: no cover
            return None  # pragma: no cover

        stripped_next_line = next_line.strip()
        if stripped_next_line.startswith(component_line_prefix):
            components = self._parse_xyz_float_tuple(stripped_next_line)
            if components:
                return components
            else:
                logger.warning(f"Could not parse Cartesian components from: {stripped_next_line}")  # pragma: no cover
        else:
            logger.warning(
                f"Expected Cartesian components line starting with '{component_line_prefix}', got: {stripped_next_line}"
            )  # pragma: no cover
            results.buffered_line = next_line  # pragma: no cover
        return None

    def _is_sub_parser_boundary(self, stripped_line: str, additional_keywords: list[str] | None = None) -> bool:
        """Checks if the line indicates an end of the current sub-parsing block."""
        keywords_to_check = additional_keywords or []
        if self.STATE_HEADER_PATTERN.match(stripped_line) or self.SECTION_END_TOKEN in stripped_line:
            return True
        return any(kw in stripped_line for kw in keywords_to_check)

    def _parse_nos_data(self, iterator: LineIterator, results: _MutableCalculationData) -> ExcitedStateNOData | None:
        """Parses the NOs subsection for an excited state."""
        frontier_occs: list[float] = []
        n_electrons: float | None = None
        n_u: float | None = None
        n_unl: float | None = None
        pr_no: float | None = None
        found_data = False

        while True:
            line = self._get_next_line(iterator, results)
            if line is None:
                break
            stripped_line = line.strip()

            if not stripped_line:
                if found_data:
                    break
                else:
                    continue

            if self._is_sub_parser_boundary(stripped_line, self.MULLIKEN_SUBSECTION_KEYWORDS):
                results.buffered_line = line
                break

            if "Occupation of frontier NOs:" in stripped_line:
                try:
                    data_line = (results.buffered_line or next(iterator)).strip()
                    results.buffered_line = None
                    frontier_occs.extend(map(float, data_line.split()))
                    found_data = True
                except StopIteration:  # pragma: no cover
                    logger.warning("EOF while expecting frontier NO occupations.")  # pragma: no cover
                    break  # pragma: no cover
                except ValueError:  # pragma: no cover
                    logger.warning(f"Could not parse frontier NO occupations from: {data_line}")  # pragma: no cover
            elif stripped_line.startswith("Number of electrons:"):
                try:
                    n_electrons = float(stripped_line.split()[-1])
                    found_data = True
                except (IndexError, ValueError):  # pragma: no cover
                    logger.warning(f"Could not parse number of electrons from: {stripped_line}")  # pragma: no cover
            elif stripped_line.startswith("Number of unpaired electrons:"):
                match = re.search(r"n_u\s*=\s*([\d.]+),\s*n_u,nl\s*=\s*([\d.]+)", stripped_line)
                if match:
                    try:
                        n_u = float(match.group(1))
                        n_unl = float(match.group(2))
                        found_data = True
                    except ValueError:  # pragma: no cover
                        logger.warning(f"Could not parse n_u/n_unl from: {stripped_line}")  # pragma: no cover
            elif stripped_line.startswith("NO participation ratio (PR_NO):"):
                try:
                    pr_no = float(stripped_line.split()[-1])
                    found_data = True
                except (IndexError, ValueError):  # pragma: no cover
                    logger.warning(f"Could not parse PR_NO from: {stripped_line}")  # pragma: no cover
            else:  # Unrecognized line within NOs
                logger.debug(f"Unrecognized line in NOs section, assuming end: '{stripped_line}'")
                results.buffered_line = line
                break

        if not found_data:
            return None
        return ExcitedStateNOData(
            frontier_occupations=frontier_occs if frontier_occs else None,
            n_electrons=n_electrons,
            n_unpaired=n_u,
            n_unpaired_nl=n_unl,
            pr_no=pr_no,
        )

    def _parse_mulliken_populations(
        self, iterator: LineIterator, results: _MutableCalculationData
    ) -> ExcitedStateMulliken | None:
        """Parses Mulliken population analysis for an excited state."""
        populations: list[ExcitedStateAtomPopulation] = []
        is_uks_format = False

        # Read and analyze the header line to determine format (RKS vs UKS)
        header_line = self._get_next_line(iterator, results)
        if header_line is None:
            logger.warning("EOF unexpected while reading Mulliken header line.")  # pragma: no cover
            return None  # pragma: no cover

        stripped_header = header_line.strip()
        if "Spin (e)" in stripped_header and "h+ (alpha)" in stripped_header:  # UKS specific columns
            is_uks_format = True
            logger.debug("Detected UKS Mulliken format.")
        elif "Charge (e)" in stripped_header and "Del q" in stripped_header:  # RKS specific column
            is_uks_format = False
            logger.debug("Detected RKS Mulliken format.")
        else:
            logger.warning(
                f"Unrecognized Mulliken header format: '{stripped_header}'. Assuming RKS and attempting to proceed."
            )  # pragma: no cover
            results.buffered_line = header_line  # Buffer this line as it might not be a header
            # For now, assume it was a header and we proceed to separator
            # No, if it's not recognized, better to stop this sub-parser or buffer and return.
            # Buffering and returning allows main loop to see it.
            return None  # pragma: no cover

        separator_line = self._get_next_line(iterator, results)
        if separator_line is None or not separator_line.strip().startswith("---"):
            logger.warning(
                f"Expected Mulliken separator line, got: '{separator_line.strip() if separator_line else 'EOF'}'"
            )  # pragma: no cover
            results.buffered_line = separator_line  # pragma: no cover
            return None  # pragma: no cover

        found_data = False
        while True:
            line = self._get_next_line(iterator, results)
            if line is None:
                break
            stripped_line = line.strip()

            if not stripped_line:
                if found_data:
                    break
                else:
                    continue

            if stripped_line.startswith("----") or "Sum:" in stripped_line:  # End of table
                if "Sum:" not in stripped_line:  # If it's just "----", check next line for "Sum:"
                    next_line_check = self._get_next_line(iterator, results)
                    if next_line_check is not None and "Sum:" not in next_line_check:
                        results.buffered_line = next_line_check
                break  # Consumed sum line or determined it's not there.

            if self._is_sub_parser_boundary(stripped_line, self.MULLIKEN_SUBSECTION_KEYWORDS):
                results.buffered_line = line
                break

            parts = stripped_line.split()
            try:
                atom_idx = int(parts[0]) - 1  # QChem is 1-indexed
                atom_symbol = parts[1]
                charge_e = float(parts[2])
                spin_e: float | None = None
                hole_charge: float | None = None
                electron_charge: float | None = None
                delta_charge: float | None = None

                if is_uks_format:
                    if len(parts) < 8:
                        logger.warning(
                            f"UKS Mulliken population line has too few parts: '{stripped_line}'"
                        )  # pragma: no cover
                        results.buffered_line = line  # pragma: no cover
                        break  # pragma: no cover
                    spin_e = float(parts[3])
                    h_alpha = float(parts[4])
                    h_beta = float(parts[5])
                    e_alpha = float(parts[6])
                    e_beta = float(parts[7])
                    hole_charge = h_alpha + h_beta
                    electron_charge = e_alpha + e_beta
                    # For UKS, delta_charge is typically not printed, so we calculate it
                    delta_charge = electron_charge - hole_charge
                else:  # RKS format
                    if len(parts) < 6:
                        logger.warning(
                            f"RKS Mulliken population line has too few parts: '{stripped_line}'"
                        )  # pragma: no cover
                        results.buffered_line = line  # pragma: no cover
                        break  # pragma: no cover
                    hole_charge = float(parts[3])
                    electron_charge = float(parts[4])
                    delta_charge = float(parts[5])

                populations.append(
                    ExcitedStateAtomPopulation(
                        atom_index=atom_idx,
                        symbol=atom_symbol,
                        charge_e=charge_e,
                        hole_charge=hole_charge,
                        electron_charge=electron_charge,
                        delta_charge=delta_charge,
                        spin_e=spin_e,  # Will be None for RKS
                        hole_charge_alpha=h_alpha if is_uks_format else None,  # For UKS
                        hole_charge_beta=h_beta if is_uks_format else None,  # For UKS
                        electron_charge_alpha=e_alpha if is_uks_format else None,  # For UKS
                        electron_charge_beta=e_beta if is_uks_format else None,  # For UKS
                    )
                )
                found_data = True
            except (IndexError, ValueError) as e:  # pragma: no cover
                logger.warning(
                    f"Could not parse Mulliken population line: '{stripped_line}'. Error: {e}"
                )  # pragma: no cover
                results.buffered_line = line  # End of Mulliken data due to error
                break  # pragma: no cover

        if not found_data:
            return None
        return ExcitedStateMulliken(populations=populations)

    def _parse_multipole_analysis(
        self, iterator: LineIterator, results: _MutableCalculationData
    ) -> ExcitedStateMultipole | None:
        """Parses multipole moment analysis for an excited state."""
        mol_charge: float | None = None
        n_electrons: float | None = None
        center_elec_chg: tuple[float, float, float] | None = None
        center_nucl_chg: tuple[float, float, float] | None = None
        dipole: DipoleMoment | None = None
        rms_density_size_cartesian: tuple[float, float, float] | None = None  # Renamed for clarity
        found_data = False

        while True:
            line = self._get_next_line(iterator, results)
            if line is None:
                break
            stripped_line = line.strip()

            if not stripped_line:
                if found_data:
                    break
                else:
                    continue

            if self._is_sub_parser_boundary(stripped_line, self.MULTIPOLE_SUBSECTION_KEYWORDS):
                results.buffered_line = line
                break

            if stripped_line.startswith("Molecular charge:"):
                try:
                    mol_charge = float(stripped_line.split()[-1])
                    found_data = True
                except (ValueError, IndexError):  # pragma: no cover
                    logger.warning(f"Failed to parse mol_charge: {stripped_line}")  # pragma: no cover
            elif stripped_line.startswith("Number of electrons:"):
                try:
                    n_electrons = float(stripped_line.split()[-1])
                    found_data = True
                except (ValueError, IndexError):  # pragma: no cover
                    logger.warning(f"Failed to parse n_electrons: {stripped_line}")  # pragma: no cover
            elif "Center of electronic charge [Ang]:" in stripped_line:
                center_elec_chg = self._parse_xyz_float_tuple(stripped_line)
                if center_elec_chg:
                    found_data = True
            elif stripped_line.startswith("Total nuclear charge:"):
                found_data = True
                continue
            elif "Center of nuclear charge [Ang]:" in stripped_line:
                center_nucl_chg = self._parse_xyz_float_tuple(stripped_line)
                if center_nucl_chg:
                    found_data = True
            elif stripped_line.startswith("Dipole moment [D]:") and not dipole:
                try:
                    total_d = float(stripped_line.split()[-1])
                    cart_components = self._parse_cartesian_components_next_line(
                        iterator, results, "Cartesian components [D]:"
                    )
                    if cart_components:
                        dipole = DipoleMoment(
                            x=cart_components[0],
                            y=cart_components[1],
                            z=cart_components[2],
                            magnitude=total_d,
                        )
                        found_data = True
                    else:  # _parse_cartesian_components_next_line already logs and buffers if needed
                        logger.warning(
                            f"Could not parse dipole Cartesian components following: {stripped_line}"
                        )  # pragma: no cover
                except (ValueError, IndexError):  # For float(stripped_line.split()[-1]) # pragma: no cover
                    logger.warning(f"Failed to parse total dipole moment from: {stripped_line}")  # pragma: no cover

            elif stripped_line.startswith("RMS size of the density [Ang]:") and not rms_density_size_cartesian:
                # Scalar RMS size is on this line, components on the next
                # We primarily care about the components for the dataclass.
                # The scalar value is not explicitly stored in ExcitedStateMultipole currently.
                # If it needs to be, the dataclass and this parsing would need adjustment.
                # total_rms = float(stripped_line.split()[-1])
                cart_components = self._parse_cartesian_components_next_line(
                    iterator, results, "Cartesian components [Ang]:"
                )
                if cart_components:
                    rms_density_size_cartesian = cart_components
                    found_data = True
                else:
                    logger.warning(
                        f"Could not parse RMS density Cartesian components following: {stripped_line}"
                    )  # pragma: no cover
            else:
                logger.debug(f"Unrecognized line in Multipole section, assuming end: '{stripped_line}'")
                results.buffered_line = line
                break

        if not found_data:
            return None
        return ExcitedStateMultipole(
            molecular_charge=mol_charge,
            n_electrons=n_electrons,
            center_electronic_charge_ang=center_elec_chg,
            center_nuclear_charge_ang=center_nucl_chg,
            dipole_moment_debye=dipole,
            rms_density_size_ang=rms_density_size_cartesian,  # Use the parsed cartesian components
        )

    def _parse_exciton_analysis_block(
        self, iterator: LineIterator, results: _MutableCalculationData
    ) -> ExcitedStateExcitonDifferenceDM | None:
        """Parses a single block of exciton analysis (Total, Alpha, or Beta)."""
        hole_center: tuple[float, float, float] | None = None
        elec_center: tuple[float, float, float] | None = None
        separation: float | None = None
        hole_size_scalar: float | None = None
        hole_size_comps: tuple[float, float, float] | None = None
        elec_size_scalar: float | None = None
        elec_size_comps: tuple[float, float, float] | None = None
        found_data = False

        while True:
            line = self._get_next_line(iterator, results)
            if line is None:
                break
            stripped_line = line.strip()

            if not stripped_line:
                # Blank lines within an exciton block are usually just spacing.
                # If found_data is true and we hit a blank line, it might signify end of a sub-block for RKS
                # but for UKS, specific headers "Alpha spin:", "Beta spin:" are primary delimiters.
                # Let's rely on _is_sub_parser_boundary or specific spin headers.
                logger.debug(f"Skipping blank line in exciton analysis block: '{line.rstrip()}'")
                continue

            # End conditions for this specific block (e.g., start of next spin block, or overall section end)
            if (
                self.STATE_HEADER_PATTERN.match(stripped_line)
                or self.SECTION_END_TOKEN in stripped_line
                or stripped_line == "Alpha spin:"
                or stripped_line == "Beta spin:"
            ):
                results.buffered_line = line
                break

            if stripped_line.startswith("<r_h> [Ang]:"):
                hole_center = self._parse_xyz_float_tuple(stripped_line)
                if hole_center:
                    found_data = True
                else:
                    logger.warning(f"Failed to parse <r_h>: {stripped_line}")  # pragma: no cover
                    results.buffered_line = line  # pragma: no cover
            elif stripped_line.startswith("<r_e> [Ang]:"):
                elec_center = self._parse_xyz_float_tuple(stripped_line)
                if elec_center:
                    found_data = True
                else:
                    logger.warning(f"Failed to parse <r_e>: {stripped_line}")  # pragma: no cover
                    results.buffered_line = line  # pragma: no cover
            elif stripped_line.startswith("|<r_e - r_h>| [Ang]:"):
                try:
                    separation = float(stripped_line.split()[-1])
                    found_data = True
                except (ValueError, IndexError):  # pragma: no cover
                    logger.warning(f"Failed to parse |<r_e - r_h>|: {stripped_line}")  # pragma: no cover
            elif (
                stripped_line.startswith("Hole size [Ang]:") and not hole_size_comps
            ):  # ensure components not already parsed
                try:
                    hole_size_scalar = float(stripped_line.split()[-1])  # Store the scalar value
                    comps = self._parse_cartesian_components_next_line(iterator, results, "Cartesian components [Ang]:")
                    if comps:
                        hole_size_comps = comps
                        found_data = True
                    else:
                        logger.warning(
                            f"Could not parse Hole size Cartesian components from: {stripped_line}"
                        )  # pragma: no cover
                        results.buffered_line = stripped_line  # pragma: no cover
                except (ValueError, IndexError, StopIteration):  # pragma: no cover
                    logger.warning(f"Failed to parse Hole size: {stripped_line}")  # pragma: no cover
            elif stripped_line.startswith("Electron size [Ang]:") and not elec_size_comps:
                try:
                    elec_size_scalar = float(stripped_line.split()[-1])  # Store the scalar value
                    comps = self._parse_cartesian_components_next_line(iterator, results, "Cartesian components [Ang]:")
                    if comps:
                        elec_size_comps = comps
                        found_data = True
                    else:
                        logger.warning(
                            f"Could not parse Electron size Cartesian components from: {stripped_line}"
                        )  # pragma: no cover
                        results.buffered_line = stripped_line  # pragma: no cover
                except (ValueError, IndexError, StopIteration):  # pragma: no cover
                    logger.warning(f"Failed to parse Electron size: {stripped_line}")  # pragma: no cover
            else:
                logger.debug(f"Unrecognized line in Exciton analysis, assuming end: '{stripped_line}'")
                results.buffered_line = line
                break

        if not found_data:
            return None

        # The ExcitedStateExcitonDifferenceDM wants scalar for hole_size_ang and electron_size_ang
        # but the example output shows a scalar line then a component line.
        # The dataclass seems to expect the components for *_size_ang. Let's use components.
        return ExcitedStateExcitonDifferenceDM(
            hole_center_ang=hole_center,
            electron_center_ang=elec_center,
            electron_hole_separation_ang=separation,
            hole_size_ang=hole_size_scalar,
            electron_size_ang=elec_size_scalar,
            hole_size_ang_comps=hole_size_comps,
            electron_size_ang_comps=elec_size_comps,
        )

    def _parse_optional_spin_block(
        self,
        iterator: LineIterator,
        results: _MutableCalculationData,
        spin_block_header: str,
        parser_func: Callable[[LineIterator, _MutableCalculationData], ExcitedStateExcitonDifferenceDM | None],
    ) -> ExcitedStateExcitonDifferenceDM | None:
        """Helper to parse optional Alpha or Beta spin blocks for exciton analysis."""
        data: ExcitedStateExcitonDifferenceDM | None = None
        # Peek at the current buffered line without consuming it from the main iterator yet,
        # as _get_next_line would if called directly here.
        line_to_check = results.buffered_line

        if line_to_check and line_to_check.strip() == spin_block_header:
            results.buffered_line = None  # Consume the header that was in buffer
            logger.debug(f"UKS Exciton: Parsing '{spin_block_header}' block")
            data = parser_func(iterator, results)
        elif line_to_check:  # If buffer has something else
            logger.debug(
                f"UKS Exciton: Expected '{spin_block_header}', but found '{line_to_check.strip()}'. No {spin_block_header.split()[0].lower()} data."
            )
        else:  # Buffer is empty, try to get next line to check
            next_line_peek = self._get_next_line(iterator, results)
            if next_line_peek and next_line_peek.strip() == spin_block_header:
                # Already consumed by _get_next_line, so no need to nullify buffer again
                logger.debug(f"UKS Exciton: Parsing '{spin_block_header}' block")
                data = parser_func(iterator, results)
            elif next_line_peek:
                logger.debug(
                    f"UKS Exciton: Expected '{spin_block_header}', but found '{next_line_peek.strip()}'. No {spin_block_header.split()[0].lower()} data."
                )
                results.buffered_line = next_line_peek  # Put it back
            else:  # EOF
                logger.debug(
                    f"UKS Exciton: EOF while checking for '{spin_block_header}'. No {spin_block_header.split()[0].lower()} data."
                )
        return data

    def _skip_current_sub_block(self, iterator: LineIterator, results: _MutableCalculationData) -> None:
        """Skips lines until a known major delimiter or next state is found."""
        logger.debug("Skipping current sub-block.")
        while True:
            line = self._get_next_line(iterator, results)
            if line is None:
                logger.debug("EOF reached while skipping sub-block.")
                break

            stripped_line = line.strip()

            # Check for main section end or new state header first
            if self.SECTION_END_TOKEN in stripped_line or self.STATE_HEADER_PATTERN.match(stripped_line):
                results.buffered_line = line  # Put back the delimiter
                logger.debug(f"Skipped until main delimiter: '{stripped_line[:50]}...'")
                break

            # Check for known next sub-section headers that signal end of current unknown/skipped block
            known_next_headers = [
                "NOs",
                "NOs (alpha)",
                "NOs (beta)",
                "NOs (spin-traced)",
                "Mulliken Population Analysis (State/Difference DM)",
                "Multipole moment analysis of the density matrix",
                "Exciton analysis of the difference density matrix",  # This one is tricky, as it's also the start of what we might skip TO
            ]
            # If we are skipping FROM "Exciton analysis...", then matching it again means we found the next one we *would* parse.
            # This logic is primarily for skipping things like "NOs (alpha)" when we only want "NOs (spin-traced)".
            if any(header == stripped_line for header in known_next_headers):
                results.buffered_line = line  # Put back the header
                logger.debug(f"Skipped until next known sub-header: '{stripped_line}'")
                break

            logger.debug(f"Skipping line: '{stripped_line[:50]}...'")

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """
        Parses the entire "Analysis of Unrelaxed Density Matrices" section.
        `current_line` is the line that matched `SECTION_START_TOKEN`.
        """
        logger.info(f"Parsing section: {self.SECTION_START_TOKEN}")

        # The current_line (SECTION_START_TOKEN) is consumed by virtue of being passed.
        # Skip any immediate blank lines or separators after the main section header.
        try:
            while True:
                line_after_header = self._get_next_line(iterator, results)
                if line_after_header is None:  # EOF
                    logger.warning(f"EOF immediately after '{self.SECTION_START_TOKEN}' header.")  # pragma: no cover
                    results.parsed_unrelaxed_excited_state_properties = True  # pragma: no cover
                    return  # pragma: no cover
                if line_after_header.strip() and not line_after_header.strip().startswith("---"):
                    results.buffered_line = line_after_header  # Put back the first significant line
                    break
        except StopIteration:  # pragma: no cover
            logger.warning(
                f"EOF immediately after '{self.SECTION_START_TOKEN}' header (caught StopIteration)."
            )  # pragma: no cover
            results.parsed_unrelaxed_excited_state_properties = True  # pragma: no cover
            return  # pragma: no cover

        active_state_props: ExcitedStateDetailedAnalysis | None = None

        qchem_version = getattr(results, "qchem_version", None)

        while True:
            try:
                line = self._get_next_line(iterator, results)
                if line is None:  # EOF
                    if active_state_props:
                        results.excited_state_detailed_analyses_list.append(active_state_props)
                        logger.debug(f"Added last active state {active_state_props.state_number} at EOF.")
                    logger.info(f"EOF reached while parsing {self.SECTION_START_TOKEN}.")
                    break

                stripped_line = line.strip()

                if self.SECTION_END_TOKEN in stripped_line:
                    if active_state_props:
                        results.excited_state_detailed_analyses_list.append(active_state_props)
                        logger.debug(
                            f"Added last active state {active_state_props.state_number} before {self.SECTION_END_TOKEN}."
                        )
                    results.buffered_line = line
                    logger.info(f"Found '{self.SECTION_END_TOKEN}'. Stopping parser.")
                    break

                state_match = self.STATE_HEADER_PATTERN.match(stripped_line)
                if state_match:
                    if active_state_props:  # Store previously accumulated state data
                        results.excited_state_detailed_analyses_list.append(active_state_props)
                        logger.debug(f"Stored state {active_state_props.state_number}. New state found.")

                    matched_multiplicity_or_type = state_match.group(1)  # "Singlet", "Triplet", or "Excited State"
                    state_number = int(state_match.group(2))
                    active_state_props = ExcitedStateDetailedAnalysis(
                        state_number=state_number,
                        multiplicity=matched_multiplicity_or_type,  # Use the matched group directly
                        no_data=None,
                        mulliken=None,
                        multipole=None,
                        exciton_difference_dm_analysis=None,
                    )
                    logger.debug(f"Parsing new state: {matched_multiplicity_or_type} {state_number}")
                    continue  # Continue to parse this state's content

                if active_state_props:
                    is_uks_like_state = "Excited State" in active_state_props.multiplicity
                    matched_action_this_line = False
                    # Select the correct registry for the current state
                    section_patterns = (
                        UNRELAXED_DM_SECTION_PATTERNS_UKS if is_uks_like_state else UNRELAXED_DM_SECTION_PATTERNS_RKS
                    )
                    for pattern_def in section_patterns:
                        versioned_pattern = pattern_def.get_matching_pattern(qchem_version)
                        if versioned_pattern and versioned_pattern.pattern.match(stripped_line):
                            field = pattern_def.field_name
                            if field == "no_data":
                                logger.debug(
                                    f"Parsing target NOs data for state {active_state_props.state_number}, header: '{stripped_line}'"
                                )
                                no_data = self._parse_nos_data(iterator, results)
                                active_state_props = replace(active_state_props, no_data=no_data)
                                matched_action_this_line = True
                                break
                            elif field == "mulliken":
                                logger.debug(f"Parsing Mulliken for state {active_state_props.state_number}")
                                mulliken_data = self._parse_mulliken_populations(iterator, results)
                                active_state_props = replace(active_state_props, mulliken=mulliken_data)
                                matched_action_this_line = True
                                break
                            elif field == "multipole":
                                logger.debug(f"Parsing Multipole for state {active_state_props.state_number}")
                                multipole_data = self._parse_multipole_analysis(iterator, results)
                                active_state_props = replace(active_state_props, multipole=multipole_data)
                                matched_action_this_line = True
                                break
                            elif field == "exciton_difference_dm_analysis":
                                logger.debug(f"Parsing Exciton for state {active_state_props.state_number}")
                                next_line_peek = self._get_next_line(iterator, results)
                                if next_line_peek is None:
                                    logger.warning("EOF after 'Exciton analysis' header.")
                                    break
                                results.buffered_line = next_line_peek
                                total_exciton_data = None
                                alpha_exciton_data = None
                                beta_exciton_data = None
                                if next_line_peek.strip() == "Total:":
                                    results.buffered_line = None
                                    logger.debug("UKS Exciton: Parsing 'Total:' block")
                                    total_exciton_data = self._parse_exciton_analysis_block(iterator, results)
                                    alpha_exciton_data = self._parse_optional_spin_block(
                                        iterator, results, "Alpha spin:", self._parse_exciton_analysis_block
                                    )
                                    beta_exciton_data = self._parse_optional_spin_block(
                                        iterator, results, "Beta spin:", self._parse_exciton_analysis_block
                                    )
                                else:
                                    logger.debug("RKS-like Exciton: Parsing single block")
                                    total_exciton_data = self._parse_exciton_analysis_block(iterator, results)
                                active_state_props = replace(
                                    active_state_props,
                                    exciton_difference_dm_analysis=total_exciton_data,
                                    exciton_difference_dm_analysis_alpha=alpha_exciton_data,
                                    exciton_difference_dm_analysis_beta=beta_exciton_data,
                                )
                                matched_action_this_line = True
                                break
                    if matched_action_this_line:
                        continue
                    # Always skip NOs (alpha) and NOs (beta)
                    if stripped_line in {"NOs (alpha)", "NOs (beta)"}:
                        logger.debug(
                            f"Skipping non-target NOs block: '{stripped_line}' for state {active_state_props.state_number} (fallback)"
                        )
                        self._skip_current_sub_block(iterator, results)
                        continue

                if not stripped_line or stripped_line.startswith("----"):
                    continue

                logger.warning(
                    f"Unexpected line in '{self.SECTION_START_TOKEN}' section: '{stripped_line}'. Current state: {active_state_props.state_number if active_state_props else 'None'}."
                )  # pragma: no cover
                if active_state_props:  # pragma: no cover
                    results.excited_state_detailed_analyses_list.append(active_state_props)  # pragma: no cover
                results.buffered_line = line  # pragma: no cover
                logger.warning(
                    f"Stopping {self.SECTION_START_TOKEN} parsing due to unrecognized line."
                )  # pragma: no cover
                break  # pragma: no cover

            except StopIteration:  # pragma: no cover
                if active_state_props:  # pragma: no cover
                    results.excited_state_detailed_analyses_list.append(active_state_props)  # pragma: no cover
                    logger.debug(
                        f"Added last active state {active_state_props.state_number} at EOF."
                    )  # pragma: no cover
                logger.info(f"EOF reached while parsing {self.SECTION_START_TOKEN}.")  # pragma: no cover
                break  # pragma: no cover

        results.parsed_unrelaxed_excited_state_properties = True
        count = len(results.excited_state_detailed_analyses_list)
        if count > 0:
            logger.info(f"Successfully parsed {count} states from '{self.SECTION_START_TOKEN}'.")
        elif not results.buffered_line or self.SECTION_END_TOKEN not in results.buffered_line:
            logger.warning(f"Parsed '{self.SECTION_START_TOKEN}' but found no detailed state analyses.")

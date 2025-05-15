import re
from dataclasses import replace

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
from calcflow.utils import logger


class UnrelaxedExcitedStatePropertiesParser:
    """
    Parses the "Analysis of Unrelaxed Density Matrices" section from Q-Chem output.
    This section contains detailed properties for each excited state, such as
    Natural Orbital occupations, Mulliken populations, multipole moments, and
    exciton analysis of the difference density matrix.
    """

    SECTION_START_TOKEN = "Analysis of Unrelaxed Density Matrices"
    SECTION_END_TOKEN = "Transition Density Matrix Analysis"
    STATE_HEADER_PATTERN = re.compile(r"^\s*(Singlet|Triplet)\s+(\d+)\s*:")

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Checks if the line marks the beginning of the unrelaxed DM analysis section."""
        return self.SECTION_START_TOKEN in line and not current_data.parsed_unrelaxed_excited_state_properties

    def _parse_nos_data(self, iterator: LineIterator, results: _MutableCalculationData) -> ExcitedStateNOData | None:
        """Parses the NOs subsection for an excited state."""
        frontier_occs: list[float] = []
        n_electrons: float | None = None
        n_u: float | None = None
        n_unl: float | None = None
        pr_no: float | None = None
        found_data = False

        while True:
            line = results.buffered_line or next(iterator)
            results.buffered_line = None
            stripped_line = line.strip()

            if not stripped_line:  # Blank line can signify end or just spacing
                if found_data:  # If we've parsed something, blank line is end.
                    break
                else:  # Else, it's just spacing, continue.
                    continue

            if any(
                kw in stripped_line
                for kw in [
                    "Mulliken Population Analysis",
                    "Multipole moment analysis",
                    "Exciton analysis",
                    self.SECTION_END_TOKEN,
                ]
            ) or self.STATE_HEADER_PATTERN.match(stripped_line):
                results.buffered_line = line
                break

            if "Occupation of frontier NOs:" in stripped_line:
                try:
                    data_line = (results.buffered_line or next(iterator)).strip()
                    results.buffered_line = None
                    frontier_occs.extend(map(float, data_line.split()))
                    found_data = True
                except StopIteration:
                    logger.warning("EOF while expecting frontier NO occupations.")
                    break
                except ValueError:
                    logger.warning(f"Could not parse frontier NO occupations from: {data_line}")
            elif stripped_line.startswith("Number of electrons:"):
                try:
                    n_electrons = float(stripped_line.split()[-1])
                    found_data = True
                except (IndexError, ValueError):
                    logger.warning(f"Could not parse number of electrons from: {stripped_line}")
            elif stripped_line.startswith("Number of unpaired electrons:"):
                match = re.search(r"n_u\s*=\s*([\d.]+),\s*n_u,nl\s*=\s*([\d.]+)", stripped_line)
                if match:
                    try:
                        n_u = float(match.group(1))
                        n_unl = float(match.group(2))
                        found_data = True
                    except ValueError:
                        logger.warning(f"Could not parse n_u/n_unl from: {stripped_line}")
            elif stripped_line.startswith("NO participation ratio (PR_NO):"):
                try:
                    pr_no = float(stripped_line.split()[-1])
                    found_data = True
                except (IndexError, ValueError):
                    logger.warning(f"Could not parse PR_NO from: {stripped_line}")
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
        # Skip header lines "Atom Charge (e) h+ e- Del q" and "----"
        try:
            _ = next(iterator)  # Header
            _ = next(iterator)  # Separator
        except StopIteration:
            logger.warning("EOF unexpected while skipping Mulliken headers.")
            return None

        found_data = False
        while True:
            line = results.buffered_line or next(iterator)
            results.buffered_line = None
            stripped_line = line.strip()

            if not stripped_line:
                if found_data:
                    break
                else:
                    continue

            if stripped_line.startswith("----") or "Sum:" in stripped_line:  # End of table
                # Consume sum line if present
                if "Sum:" in stripped_line:
                    pass  # Sum line is just for verification, not stored in this object.
                else:  # Separator line implies sum line might be next or end.
                    try:
                        next_line_check = results.buffered_line or next(iterator)
                        results.buffered_line = None
                        if "Sum:" not in next_line_check:
                            results.buffered_line = next_line_check  # Put it back if not sum
                    except StopIteration:
                        pass  # EOF after separator is fine
                break

            if any(
                kw in stripped_line
                for kw in [
                    "Multipole moment analysis",
                    "Exciton analysis",
                    self.SECTION_END_TOKEN,
                ]
            ) or self.STATE_HEADER_PATTERN.match(stripped_line):
                results.buffered_line = line
                break

            parts = stripped_line.split()
            try:
                if len(parts) < 6:
                    logger.warning(f"Mulliken population line has too few parts to parse: '{stripped_line}'")
                    results.buffered_line = line
                    break  # Stop parsing this Mulliken table

                atom_idx = int(parts[0]) - 1  # QChem is 1-indexed
                atom_symbol = parts[1]

                populations.append(
                    ExcitedStateAtomPopulation(
                        atom_index=atom_idx,
                        symbol=atom_symbol,
                        charge_e=float(parts[2]),
                        hole_charge=float(parts[3]),
                        electron_charge=float(parts[4]),
                        delta_charge=float(parts[5]),
                    )
                )
                found_data = True
            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse Mulliken population line: '{stripped_line}'. Error: {e}")
                results.buffered_line = line  # End of Mulliken data due to error
                break

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
        rms_density_size: tuple[float, float, float] | None = None
        found_data = False

        while True:
            line = results.buffered_line or next(iterator)
            results.buffered_line = None
            stripped_line = line.strip()

            if not stripped_line:
                if found_data:
                    break
                else:
                    continue

            if any(
                kw in stripped_line for kw in ["Exciton analysis", self.SECTION_END_TOKEN]
            ) or self.STATE_HEADER_PATTERN.match(stripped_line):
                results.buffered_line = line
                break

            if stripped_line.startswith("Molecular charge:"):
                try:
                    mol_charge = float(stripped_line.split()[-1])
                    found_data = True
                except (ValueError, IndexError):
                    logger.warning(f"Failed to parse mol_charge: {stripped_line}")
            elif stripped_line.startswith("Number of electrons:"):
                try:
                    n_electrons = float(stripped_line.split()[-1])
                    found_data = True
                except (ValueError, IndexError):
                    logger.warning(f"Failed to parse n_electrons: {stripped_line}")
            elif "Center of electronic charge [Ang]:" in stripped_line:
                match = re.search(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]", stripped_line)
                if match:
                    try:
                        center_elec_chg = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
                        found_data = True
                    except ValueError:
                        logger.warning(f"Failed to parse center_elec_chg: {stripped_line}")
            elif stripped_line.startswith("Total nuclear charge:"):
                # This line provides context but its value isn't stored directly in ExcitedStateMultipole.
                # We consume it to allow parsing of the subsequent "Center of nuclear charge" line.
                found_data = True  # Mark that we are still in a valid part of the section
                continue  # Explicitly continue to the next line
            elif (
                "Center of nuclear charge [Ang]:" in stripped_line
            ):  # Note: QChem example has "Total nuclear charge" then "Center..."
                match = re.search(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]", stripped_line)
                if match:
                    try:
                        center_nucl_chg = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
                        found_data = True
                    except ValueError:
                        logger.warning(f"Failed to parse center_nucl_chg: {stripped_line}")
            elif (
                stripped_line.startswith("Dipole moment [D]:") and not dipole
            ):  # Ensure we don't overwrite if Cartesian comes first
                try:
                    total_d = float(stripped_line.split()[-1])
                    # Cartesian components will be parsed next
                    next_line_cart = (results.buffered_line or next(iterator)).strip()
                    results.buffered_line = None
                    cart_match = re.search(
                        r"Cartesian components \[D\]:\s*\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]", next_line_cart
                    )
                    if cart_match:
                        dipole = DipoleMoment(
                            x=float(cart_match.group(1)),
                            y=float(cart_match.group(2)),
                            z=float(cart_match.group(3)),
                            magnitude=total_d,
                        )
                        found_data = True
                    else:
                        logger.warning(f"Could not parse dipole Cartesian components from: {next_line_cart}")
                        results.buffered_line = next_line_cart  # Put it back if not parsable as dipole
                except (ValueError, IndexError, StopIteration):
                    logger.warning(f"Failed to parse dipole moment: {stripped_line}")
            elif stripped_line.startswith("RMS size of the density [Ang]:") and not rms_density_size:
                # Example: "RMS size of the density [Ang]:         0.950243"
                # "Cartesian components [Ang]:        [  0.591568,   0.481035,   0.567111]"
                # For now, only storing the cartesian components as per ExcitedStateMultipole structure
                try:
                    # We are interested in the Cartesian components from the next line
                    next_line_cart_rms = (results.buffered_line or next(iterator)).strip()
                    results.buffered_line = None
                    cart_rms_match = re.search(
                        r"Cartesian components \[Ang\]:\s*\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]",
                        next_line_cart_rms,
                    )
                    if cart_rms_match:
                        rms_density_size = (
                            float(cart_rms_match.group(1)),
                            float(cart_rms_match.group(2)),
                            float(cart_rms_match.group(3)),
                        )
                        found_data = True
                    else:
                        logger.warning(f"Could not parse RMS density Cartesian components from: {next_line_cart_rms}")
                        results.buffered_line = next_line_cart_rms
                except (ValueError, IndexError, StopIteration):
                    logger.warning(f"Failed to parse RMS density size: {stripped_line}")
            else:  # Unrecognized line
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
            rms_density_size_ang=rms_density_size,
        )

    def _parse_exciton_analysis(
        self, iterator: LineIterator, results: _MutableCalculationData
    ) -> ExcitedStateExcitonDifferenceDM | None:
        """Parses exciton analysis of the difference density matrix."""
        hole_center: tuple[float, float, float] | None = None
        elec_center: tuple[float, float, float] | None = None
        separation: float | None = None
        hole_size_scalar: float | None = None  # The single value for "Hole size [Ang]:"
        hole_size_comps: tuple[float, float, float] | None = None
        elec_size_scalar: float | None = None  # The single value for "Electron size [Ang]:"
        elec_size_comps: tuple[float, float, float] | None = None
        found_data = False

        while True:
            line = results.buffered_line or next(iterator)
            results.buffered_line = None
            stripped_line = line.strip()

            if not stripped_line:
                if found_data:
                    break
                else:
                    continue

            # End conditions: next state, or end of main section
            if self.STATE_HEADER_PATTERN.match(stripped_line) or self.SECTION_END_TOKEN in stripped_line:
                results.buffered_line = line
                break

            if stripped_line.startswith("<r_h> [Ang]:"):
                match = re.search(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]", stripped_line)
                if match:
                    g1, g2, g3 = match.groups()
                    hole_center = (float(g1), float(g2), float(g3))
                    found_data = True
                else:
                    logger.warning(f"Failed to parse <r_h>: {stripped_line}")
            elif stripped_line.startswith("<r_e> [Ang]:"):
                match = re.search(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]", stripped_line)
                if match:
                    g1, g2, g3 = match.groups()
                    elec_center = (float(g1), float(g2), float(g3))
                    found_data = True
                else:
                    logger.warning(f"Failed to parse <r_e>: {stripped_line}")
            elif stripped_line.startswith("|<r_e - r_h>| [Ang]:"):
                try:
                    separation = float(stripped_line.split()[-1])
                    found_data = True
                except (ValueError, IndexError):
                    logger.warning(f"Failed to parse |<r_e - r_h>|: {stripped_line}")
            elif (
                stripped_line.startswith("Hole size [Ang]:") and not hole_size_comps
            ):  # ensure components not already parsed
                try:
                    hole_size_scalar = float(stripped_line.split()[-1])  # Store the scalar value
                    next_line_comps = (results.buffered_line or next(iterator)).strip()
                    results.buffered_line = None
                    comp_match = re.search(
                        r"Cartesian components \[Ang\]:\s*\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]",
                        next_line_comps,
                    )
                    if comp_match:
                        g1, g2, g3 = comp_match.groups()
                        hole_size_comps = (float(g1), float(g2), float(g3))
                        found_data = True
                    else:
                        logger.warning(f"Could not parse Hole size Cartesian components from: {next_line_comps}")
                        results.buffered_line = next_line_comps
                except (ValueError, IndexError, StopIteration):
                    logger.warning(f"Failed to parse Hole size: {stripped_line}")
            elif stripped_line.startswith("Electron size [Ang]:") and not elec_size_comps:
                try:
                    elec_size_scalar = float(stripped_line.split()[-1])  # Store the scalar value
                    next_line_comps = (results.buffered_line or next(iterator)).strip()
                    results.buffered_line = None
                    comp_match = re.search(
                        r"Cartesian components \[Ang\]:\s*\[\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]",
                        next_line_comps,
                    )
                    if comp_match:
                        g1, g2, g3 = comp_match.groups()
                        elec_size_comps = (float(g1), float(g2), float(g3))
                        found_data = True
                    else:
                        logger.warning(f"Could not parse Electron size Cartesian components from: {next_line_comps}")
                        results.buffered_line = next_line_comps
                except (ValueError, IndexError, StopIteration):
                    logger.warning(f"Failed to parse Electron size: {stripped_line}")
            else:  # Unrecognized line
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
            hole_size_ang_comps=hole_size_comps,  # Using components based on example parsing needs
            electron_size_ang_comps=elec_size_comps,  # Using components
        )

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
                line_after_header = results.buffered_line or next(iterator)
                results.buffered_line = None
                if line_after_header.strip() and not line_after_header.strip().startswith("---"):
                    results.buffered_line = line_after_header  # Put back the first significant line
                    break
        except StopIteration:
            logger.warning(f"EOF immediately after '{self.SECTION_START_TOKEN}' header.")
            results.parsed_unrelaxed_excited_state_properties = True
            return

        active_state_props: ExcitedStateDetailedAnalysis | None = None

        while True:
            try:
                line = results.buffered_line or next(iterator)
                results.buffered_line = None
            except StopIteration:
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

                multiplicity = state_match.group(1)
                state_number = int(state_match.group(2))
                active_state_props = ExcitedStateDetailedAnalysis(
                    state_number=state_number,
                    multiplicity=multiplicity,
                    no_data=None,
                    mulliken=None,
                    multipole=None,
                    exciton_difference_dm_analysis=None,
                )
                logger.debug(f"Parsing new state: {multiplicity} {state_number}")
                continue  # Continue to parse this state's content

            if active_state_props:
                # We are inside a state's block, look for sub-section headers
                # Sub-parsers will consume their lines and handle their own `results.buffered_line`
                if stripped_line == "NOs":
                    logger.debug(f"Parsing NOs for state {active_state_props.state_number}")
                    no_data = self._parse_nos_data(iterator, results)
                    active_state_props = replace(active_state_props, no_data=no_data)
                    continue
                elif stripped_line == "Mulliken Population Analysis (State/Difference DM)":
                    logger.debug(f"Parsing Mulliken for state {active_state_props.state_number}")
                    mulliken_data = self._parse_mulliken_populations(iterator, results)
                    active_state_props = replace(active_state_props, mulliken=mulliken_data)
                    continue
                elif stripped_line == "Multipole moment analysis of the density matrix":
                    logger.debug(f"Parsing Multipole for state {active_state_props.state_number}")
                    multipole_data = self._parse_multipole_analysis(iterator, results)
                    active_state_props = replace(active_state_props, multipole=multipole_data)
                    continue
                elif stripped_line == "Exciton analysis of the difference density matrix":
                    logger.debug(f"Parsing Exciton for state {active_state_props.state_number}")
                    exciton_data = self._parse_exciton_analysis(iterator, results)
                    active_state_props = replace(active_state_props, exciton_difference_dm_analysis=exciton_data)
                    # Typically, after exciton data, the state block is complete.
                    # The main loop will then look for a new state_header or section_end.
                    continue

            if not stripped_line or stripped_line.startswith("----"):
                # Skip blank lines or general separators if they are not part of a sub-section's structure.
                # Sub-parsers should handle their specific separators.
                continue

            # If line is not a known header, not blank, and not handled by a sub-parser context:
            logger.warning(
                f"Unexpected line in '{self.SECTION_START_TOKEN}' section: '{stripped_line}'. Current state: {active_state_props.state_number if active_state_props else 'None'}."
            )
            # As per spec for unrecognized line, buffer and return (end this section's parsing)
            if active_state_props:  # Store any partially parsed state
                results.excited_state_detailed_analyses_list.append(active_state_props)
            results.buffered_line = line
            logger.warning(f"Stopping {self.SECTION_START_TOKEN} parsing due to unrecognized line.")
            break  # Stop parsing this section

        results.parsed_unrelaxed_excited_state_properties = True
        count = len(results.excited_state_detailed_analyses_list)
        if count > 0:
            logger.info(f"Successfully parsed {count} states from '{self.SECTION_START_TOKEN}'.")
        elif not results.buffered_line or self.SECTION_END_TOKEN not in results.buffered_line:
            # Only warn if we didn't stop because we immediately found the next section
            logger.warning(f"Parsed '{self.SECTION_START_TOKEN}' but found no detailed state analyses.")

import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pprint import pformat
from re import Pattern
from typing import Literal, Protocol, TypeAlias

# Assuming exceptions.py is in the same parent directory or PYTHONPATH
from calcflow.exceptions import ParsingError  # Relative import

# Configure logging
logger = logging.getLogger(__name__)
# Example basic config - ideally configured at application entry point
# logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')


OCC_THRESHOLD = 0.1  # Threshold to consider an orbital occupied
# --- Data Structures (Copied from prd/res_data.md) --- #

# --- Core Geometry --- #


@dataclass(frozen=True)
class Atom:
    """Represents an atom in the molecular geometry."""

    symbol: str
    x: float  # Angstrom
    y: float  # Angstrom
    z: float  # Angstrom


# --- Result Components --- #


@dataclass(frozen=True)
class ScfIteration:
    """Holds data for a single SCF iteration."""

    iteration: int
    energy_eh: float
    delta_e_eh: float
    time_sec: float
    rmsdp: float | None = None  # RMS Density Change
    maxdp: float | None = None  # Max Density Change
    diis_error: float | None = None  # DIIS error (may be None in SOSCF)
    max_gradient: float | None = None  # Max Gradient (may be None in DIIS)
    damping: float | None = None  # Damping factor (may be None)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(iteration={self.iteration}, energy_eh={self.energy_eh:.8f}, delta_e_eh={self.delta_e_eh:.8f}, time_sec={self.time_sec:.2f})"


@dataclass(frozen=True)
class ScfEnergyComponents:
    """Holds the components of the raw SCF energy."""

    nuclear_repulsion_eh: float
    electronic_eh: float
    one_electron_eh: float
    two_electron_eh: float
    xc_eh: float | None = None  # Exchange-Correlation energy for DFT


@dataclass(frozen=True)
class ScfData:
    """Holds results specific to the SCF calculation step."""

    converged: bool
    energy_eh: float  # Converged SCF energy (or last energy if not converged)
    components: ScfEnergyComponents
    n_iterations: int
    iteration_history: Sequence[ScfIteration]  # Added field

    def __repr__(self) -> str:
        # Create a copy of the dict and convert iteration_history to strings
        dict_copy = self.__dict__.copy()
        dict_copy["iteration_history"] = [str(it) for it in self.iteration_history]
        return f"{self.__class__.__name__}(\n{pformat(dict_copy, indent=2)[1:-1]}\n)"

    def __str__(self) -> str:
        """Return a concise representation of the SCF data."""
        conv_status = "Converged" if self.converged else "Not Converged"
        energy_str = f"{self.energy_eh:.8f}"
        return (
            f"{type(self).__name__}(status='{conv_status}', energy_eh={energy_str}, n_iterations={self.n_iterations})"
        )


@dataclass(frozen=True)
class Orbital:
    """Represents a single molecular orbital."""

    index: int  # 0-based index
    occupation: float
    energy_eh: float  # Energy in Hartrees
    energy_ev: float  # Energy in electron Volts


@dataclass(frozen=True)
class OrbitalData:
    """Holds information about molecular orbitals."""

    orbitals: Sequence[Orbital]
    homo_index: int | None = None  # Index within the orbitals list
    lumo_index: int | None = None  # Index within the orbitals list

    def __repr__(self) -> str:
        n_orbitals = len(self.orbitals)
        homo_energy = self.orbitals[self.homo_index].energy_eh if self.homo_index is not None else None
        lumo_energy = self.orbitals[self.lumo_index].energy_eh if self.lumo_index is not None else None
        homo_str = f"{homo_energy:.6f}" if homo_energy is not None else "None"
        lumo_str = f"{lumo_energy:.6f}" if lumo_energy is not None else "None"
        return (
            f"{type(self).__name__}(n_orbitals={n_orbitals}, "
            f"homo_index={self.homo_index}, lumo_index={self.lumo_index}, "
            f"homo_energy_eh={homo_str}, "
            f"lumo_energy_eh={lumo_str})"
        )


@dataclass(frozen=True)
class AtomicCharges:
    """Stores atomic charges from a specific population analysis method."""

    method: Literal["Mulliken", "Loewdin", "Mayer"]
    charges: Mapping[int, float]  # Atom index (0-based) to charge

    def __repr__(self) -> str:
        n_charges = len(self.charges)
        charge_summary = "{...}"  # Avoid printing all charges by default
        if n_charges < 6:  # Show charges if few atoms
            charge_summary = str(self.charges)
        return f"{type(self).__name__}(method='{self.method}', n_atoms={n_charges}, charges={charge_summary})"


@dataclass(frozen=True)
class DipoleMomentData:
    """Represents the molecular dipole moment."""

    x_au: float
    y_au: float
    z_au: float
    total_au: float
    total_debye: float


@dataclass(frozen=True)
class DispersionCorrectionData:
    """Holds details of the empirical dispersion correction."""

    method: str  # e.g., "D30", "D3BJ", "D4"
    energy_eh: float


# --- Top-Level Container --- #


# Using a mutable class during parsing simplifies updates
@dataclass
class _MutableCalculationData:
    """Mutable version of CalculationData used internally during parsing."""

    raw_output: str
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"] = "UNKNOWN"
    input_geometry: Sequence[Atom] | None = None
    final_energy_eh: float | None = None
    scf: ScfData | None = None
    orbitals: OrbitalData | None = None
    atomic_charges: list[AtomicCharges] = field(default_factory=list)
    dipole_moment: DipoleMomentData | None = None
    dispersion_correction: DispersionCorrectionData | None = None
    # Flags to track if sections have been parsed (for sections expected once)
    parsed_geometry: bool = False
    parsed_scf: bool = False
    parsed_orbitals: bool = False
    parsed_dipole: bool = False
    parsed_dispersion: bool = False


@dataclass(frozen=True)
class CalculationData:
    """Top-level container for parsed ORCA calculation results."""

    raw_output: str = field(repr=False)
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"]
    input_geometry: Sequence[Atom] | None = None
    final_energy_eh: float | None = None
    scf: ScfData | None = None
    orbitals: OrbitalData | None = None
    atomic_charges: list[AtomicCharges] = field(default_factory=list)
    dipole_moment: DipoleMomentData | None = None
    dispersion_correction: DispersionCorrectionData | None = None

    @classmethod
    def from_mutable(cls, mutable_data: _MutableCalculationData) -> "CalculationData":
        """Creates an immutable CalculationData from the mutable version."""
        return cls(
            raw_output=mutable_data.raw_output,
            termination_status=mutable_data.termination_status,
            input_geometry=mutable_data.input_geometry,
            final_energy_eh=mutable_data.final_energy_eh,
            scf=mutable_data.scf,
            orbitals=mutable_data.orbitals,
            atomic_charges=list(mutable_data.atomic_charges),  # Ensure list copy
            dipole_moment=mutable_data.dipole_moment,
            dispersion_correction=mutable_data.dispersion_correction,
        )

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        lines.append(f"  termination_status='{self.termination_status}',")
        lines.append(
            f"  final_energy_eh={self.final_energy_eh:.8f},"
            if self.final_energy_eh is not None
            else "  final_energy_eh=None,"
        )
        n_atoms = len(self.input_geometry) if self.input_geometry else 0
        lines.append(f"  input_geometry=({n_atoms} Atoms),")

        # Add summaries for optional components if they exist
        if self.scf:
            lines.append(f"  scf={str(self.scf)},")
        if self.orbitals:
            lines.append(f"  orbitals={repr(self.orbitals)},")
        if self.atomic_charges:
            lines.append(f"  atomic_charges={repr(self.atomic_charges)},")
        if self.dipole_moment:
            lines.append(f"  dipole_moment={repr(self.dipole_moment)},")
        if self.dispersion_correction:
            lines.append(f"  dispersion_correction={repr(self.dispersion_correction)},")

        # Clean up trailing comma on last line if any optional fields were added
        if len(lines) > 4 and lines[-1].endswith(","):
            lines[-1] = lines[-1].rstrip(",")

        lines.append(")")
        return "\n".join(lines)


# --- Parsing Logic --- #

LineIterator: TypeAlias = Iterator[str]


class SectionParser(Protocol):
    """Protocol for section parsers."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line triggers this parser."""
        ...

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parse the section, consuming lines from the iterator and updating results."""
        ...


# --- Regex Patterns (Grouped by section) --- #

# Patterns not tied to a specific block parser
FINAL_ENERGY_PAT = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
NORMAL_TERM_PAT = re.compile(r"\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*")
ERROR_TERM_PAT = re.compile(r"(TERMINATING THE PROGRAM|ORCA finished with error)")

FLOAT_PAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"  # Common float pattern


# --- Geometry Parser --- #
GEOMETRY_START_PAT = re.compile(r"CARTESIAN COORDINATES \(ANGSTROEM\)")
GEOMETRY_LINE_PAT = re.compile(r"^\s*([A-Za-z]{1,3})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


class GeometryParser:
    """Parses the input geometry block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return not current_data.parsed_geometry and bool(GEOMETRY_START_PAT.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting geometry block parsing.")
        # Consume header lines (assumes at least one separator/blank line)
        next(iterator, None)
        geometry: list[Atom] = []
        try:
            for line in iterator:
                line_stripped = line.strip()
                if not line_stripped:
                    break  # Blank line marks the end

                match = GEOMETRY_LINE_PAT.match(line_stripped)
                if match:
                    symbol, x, y, z = match.groups()
                    geometry.append(Atom(symbol=symbol, x=float(x), y=float(y), z=float(z)))
                else:
                    logger.warning(f"Unexpected line format in geometry block: {line_stripped}")
                    break
            if not geometry:
                raise ParsingError("Geometry block found but no atoms could be parsed.")
            results.input_geometry = tuple(geometry)
            results.parsed_geometry = True
            logger.debug(f"Successfully parsed {len(geometry)} atoms.")
        except Exception as e:
            logger.error(f"Error parsing geometry block: {e}", exc_info=True)
            raise ParsingError("Failed to parse geometry block.") from e


# --- SCF Parser --- #
SCF_CONVERGED_LINE_PAT = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
SCF_DIIS_ITER_PAT = re.compile(
    rf"^\s*(\d+)\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}"
)
SCF_SOSCF_ITER_PAT = re.compile(
    rf"^\s*(\d+)\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}"
)
SCF_ENERGY_COMPONENTS_START_PAT = re.compile(r"TOTAL SCF ENERGY")
SCF_NUCLEAR_REP_PAT = re.compile(r"^\s*Nuclear Repulsion\s*:\s*(-?\d+\.\d+)")
SCF_ELECTRONIC_PAT = re.compile(r"^\s*Electronic Energy\s*:\s*(-?\d+\.\d+)")
SCF_ONE_ELECTRON_PAT = re.compile(r"^\s*One Electron Energy\s*:\s*(-?\d+\.\d+)")
SCF_TWO_ELECTRON_PAT = re.compile(r"^\s*Two Electron Energy\s*:\s*(-?\d+\.\d+)")
SCF_XC_PAT = re.compile(r"^\s*E\(XC\)\s*:\s*(-?\d+\.\d+)")


# Moved _parse_scf_block logic here
class ScfParser:
    """Parses the SCF calculation block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        # Trigger on the iteration table headers, only if SCF not already parsed
        return not current_data.parsed_scf and ("D-I-I-S" in line or "S-O-S-C-F" in line)

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug(f"Starting SCF block parsing triggered by: {current_line.strip()}")
        converged = False
        n_iterations = 0
        last_scf_energy_eh: float | None = None
        nuclear_rep_eh: float | None = None
        electronic_eh: float | None = None
        one_electron_eh: float | None = None
        two_electron_eh: float | None = None
        xc_eh: float | None = None
        latest_iter_num = 0
        iteration_history: list[ScfIteration] = []

        in_iteration_table = True
        current_table_type = None
        in_scf_component_section = False
        found_all_mandatory_components = False

        # Determine table type from the initial trigger line
        if "D-I-I-S" in current_line:
            current_table_type = "DIIS"
            next(iterator, None)  # Consume dashed line
        elif "S-O-S-C-F" in current_line:
            current_table_type = "SOSCF"
            next(iterator, None)  # Consume dashed line
        else:
            # This case should not happen due to `matches` logic
            logger.error("SCF Parser called with unexpected initial line.")
            return

        try:
            # Use loop_iterator = iter([current_line] + list(iterator)) ? No, iterator is stateful
            # Need to process lines *from* the iterator
            for line in iterator:
                line_stripped = line.strip()

                # --- State 1: Parsing Iteration Tables --- #
                if in_iteration_table:
                    if "D-I-I-S" in line:
                        current_table_type = "DIIS"
                        next(iterator, None)  # Header
                        next(iterator, None)  # Dashes
                        continue
                    if "S-O-S-C-F" in line:
                        current_table_type = "SOSCF"
                        next(iterator, None)  # Header
                        next(iterator, None)  # Dashes
                        continue

                    iter_data: ScfIteration | None = None
                    if current_table_type == "DIIS":
                        diis_match = SCF_DIIS_ITER_PAT.match(line_stripped)
                        if diis_match:
                            vals = diis_match.groups()
                            try:
                                iter_data = ScfIteration(
                                    iteration=int(vals[0]),
                                    energy_eh=float(vals[1]),
                                    delta_e_eh=float(vals[2]),
                                    rmsdp=float(vals[3]),
                                    maxdp=float(vals[4]),
                                    diis_error=float(vals[5]),
                                    damping=float(vals[6]),
                                    time_sec=float(vals[7]),
                                )
                            except (ValueError, IndexError) as e:
                                raise ParsingError(f"Could not parse DIIS iteration: {line_stripped}") from e
                    elif current_table_type == "SOSCF":
                        soscf_match = SCF_SOSCF_ITER_PAT.match(line_stripped)
                        if soscf_match:
                            vals = soscf_match.groups()
                            try:
                                iter_data = ScfIteration(
                                    iteration=int(vals[0]),
                                    energy_eh=float(vals[1]),
                                    delta_e_eh=float(vals[2]),
                                    rmsdp=float(vals[3]),
                                    maxdp=float(vals[4]),
                                    max_gradient=float(vals[5]),
                                    time_sec=float(vals[6]),
                                )
                            except (ValueError, IndexError) as e:
                                raise ParsingError(f"Could not parse SOSCF iteration: {line_stripped}") from e

                    if iter_data:
                        iteration_history.append(iter_data)
                        latest_iter_num = max(latest_iter_num, iter_data.iteration)
                        last_scf_energy_eh = iter_data.energy_eh
                        continue
                    elif line_stripped.startswith(("*", " ")) and "SCF CONVERGED AFTER" in line:
                        conv_match = SCF_CONVERGED_LINE_PAT.search(line)
                        if conv_match:
                            converged = True
                            n_iterations = int(conv_match.group(1))
                            in_iteration_table = False
                            # Don't continue here, might be the start of components immediately
                        # else: just a comment line, continue
                        continue
                    elif not line_stripped:  # Blank line can end the table
                        in_iteration_table = False
                        continue  # Process next line normally
                    elif SCF_ENERGY_COMPONENTS_START_PAT.search(line):
                        in_iteration_table = False
                        in_scf_component_section = True
                        # Fall through to component parsing for this line

                # --- State 2: Parsing SCF Energy Components --- #
                if in_scf_component_section:  # Use 'if' not 'elif' to catch fall-through
                    parsed_this_line = False
                    nr_match = SCF_NUCLEAR_REP_PAT.search(line)
                    if nr_match:
                        nuclear_rep_eh = float(nr_match.group(1))
                        parsed_this_line = True

                    el_match = SCF_ELECTRONIC_PAT.search(line)
                    if el_match:
                        electronic_eh = float(el_match.group(1))
                        parsed_this_line = True

                    one_el_match = SCF_ONE_ELECTRON_PAT.search(line)
                    if one_el_match:
                        try:
                            one_electron_eh = float(one_el_match.group(1))
                            parsed_this_line = True
                        except ValueError:
                            logger.error(
                                f"Could not convert One Electron Energy: {one_el_match.group(1)}", exc_info=True
                            )

                    two_el_match = SCF_TWO_ELECTRON_PAT.search(line)
                    if two_el_match:
                        try:
                            two_electron_eh = float(two_el_match.group(1))
                            parsed_this_line = True
                        except ValueError:
                            logger.error(
                                f"Could not convert Two Electron Energy: {two_el_match.group(1)}", exc_info=True
                            )

                    xc_match = SCF_XC_PAT.search(line)
                    if xc_match:
                        try:
                            xc_eh = float(xc_match.group(1))
                            parsed_this_line = True
                        except ValueError:
                            logger.error(f"Could not convert XC Energy: {xc_match.group(1)}", exc_info=True)

                    if not found_all_mandatory_components and None not in [
                        nuclear_rep_eh,
                        electronic_eh,
                        one_electron_eh,
                        two_electron_eh,
                    ]:
                        found_all_mandatory_components = True

                    # Check for termination conditions
                    if not parsed_this_line or found_all_mandatory_components:
                        strict_terminators = (
                            "SCF CONVERGENCE",
                            "ORBITAL ENERGIES",
                            "MULLIKEN POPULATION ANALYSIS",
                            "LOEWDIN POPULATION ANALYSIS",
                            "MAYER POPULATION ANALYSIS",
                            "DFT DISPERSION CORRECTION",
                            "FINAL SINGLE POINT ENERGY",
                            "TIMINGS",
                            "------",
                        )
                        is_terminator = False
                        if any(term in line for term in strict_terminators):
                            if line.strip().startswith("------") and not found_all_mandatory_components:
                                pass  # Ignore dashes before components found
                            else:
                                is_terminator = True

                        if is_terminator:
                            logger.debug(f"SCF component parsing finished due to terminator: {line.strip()}")
                            in_scf_component_section = False
                            break  # Exit SCF block parsing loop

                    continue  # Continue processing lines in component section

                # --- State 3: After iterations, before/after components --- #
                else:
                    # Check again if components start here
                    if SCF_ENERGY_COMPONENTS_START_PAT.search(line):
                        in_scf_component_section = True
                        continue  # Go back to component parsing state for this line

                    # Check for late convergence message if not found earlier
                    if not converged:
                        conv_match_late = SCF_CONVERGED_LINE_PAT.search(line)
                        if conv_match_late:
                            converged = True
                            n_iterations = int(conv_match_late.group(1))
                            continue  # Found convergence, process next line

                    # Check for terminators that signal the end of anything related to SCF
                    terminators_post_scf = (
                        "ORBITAL ENERGIES",
                        "MULLIKEN POPULATION ANALYSIS",
                        "LOEWDIN POPULATION ANALYSIS",
                        "MAYER POPULATION ANALYSIS",
                        "DFT DISPERSION CORRECTION",
                        "FINAL SINGLE POINT ENERGY",
                        "TIMINGS",
                    )
                    if any(term in line for term in terminators_post_scf):
                        logger.debug(f"SCF parsing loop finished due to post-SCF terminator: {line.strip()}")
                        break  # Exit SCF block parsing loop

        except Exception as e:
            logger.error(f"Error during SCF block parsing: {e}", exc_info=True)
            raise ParsingError("Failed during SCF block parsing.") from e

        # --- Final Checks and Object Creation --- #
        if not converged:
            logger.warning("SCF convergence line 'SCF CONVERGED AFTER ...' not found. Assuming not converged.")

        if n_iterations == 0 and latest_iter_num > 0:
            n_iterations = latest_iter_num
        elif n_iterations > 0 and latest_iter_num > n_iterations:
            logger.warning(
                f"Convergence line reported {n_iterations} cycles, but found {latest_iter_num} in history. Using history count."
            )
            n_iterations = latest_iter_num
        elif n_iterations == 0 and not iteration_history:
            logger.warning("No SCF iterations found or parsed, and convergence line not found.")

        if converged and not iteration_history and n_iterations > 0:
            logger.warning(f"SCF convergence line reported {n_iterations} cycles, but no iteration history parsed.")
            # Allow continuing if energy components are found, but log heavily.

        # Check mandatory components only if SCF seems to have run
        if converged or n_iterations > 0 or iteration_history:
            if not found_all_mandatory_components:
                missing_comps = [
                    comp
                    for comp, val in zip(
                        ["nuclear_rep_eh", "electronic_eh", "one_electron_eh", "two_electron_eh"],
                        [nuclear_rep_eh, electronic_eh, one_electron_eh, two_electron_eh],
                        strict=False,
                    )
                    if val is None
                ]
                logger.error(f"Missing mandatory SCF energy components after SCF ran/converged: {missing_comps}")
                raise ParsingError("Could not parse all required SCF energy components after SCF execution.")
        else:
            # If SCF didn't run, components might be missing, which is okay, but log it.
            if not found_all_mandatory_components:
                logger.warning("SCF did not run or failed immediately; energy components not found.")
            # We cannot create ScfData without components if it didn't run. Return without setting results.scf
            results.parsed_scf = True  # Mark as parsed/attempted
            logger.warning("SCF block parsing finished without finding components (likely SCF did not run).")
            return

        # Assertions only valid if components were expected and found
        assert nuclear_rep_eh is not None
        assert electronic_eh is not None
        assert one_electron_eh is not None
        assert two_electron_eh is not None

        components = ScfEnergyComponents(
            nuclear_repulsion_eh=nuclear_rep_eh,
            electronic_eh=electronic_eh,
            one_electron_eh=one_electron_eh,
            two_electron_eh=two_electron_eh,
            xc_eh=xc_eh,
        )

        final_scf_energy: float
        if iteration_history:
            final_scf_energy = iteration_history[-1].energy_eh
        elif last_scf_energy_eh is not None:
            # This case should ideally be covered by components being present if SCF converged
            final_scf_energy = last_scf_energy_eh
            logger.warning(
                f"Using last parsed SCF energy {final_scf_energy} as iteration history is missing/incomplete."
            )
        else:
            # If converged but somehow no energy found (unlikely given checks)
            if converged:
                raise ParsingError("SCF converged but failed to extract final energy.")
            else:  # Not converged, no history, no components
                raise ParsingError("Failed to determine final SCF energy (calculation may have failed early).")

        scf_result = ScfData(
            converged=converged,
            energy_eh=final_scf_energy,
            components=components,
            n_iterations=n_iterations,
            iteration_history=tuple(iteration_history),
        )
        results.scf = scf_result
        results.parsed_scf = True
        logger.debug(f"Successfully parsed SCF data: {str(scf_result)}")


# --- Orbitals Parser --- #
ORBITALS_START_PAT = re.compile(r"ORBITAL ENERGIES")
ORBITAL_LINE_PAT = re.compile(r"^\s*(\d+)\s+([\d\.]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


class OrbitalsParser:
    """Parses the orbital energies block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return not current_data.parsed_orbitals and bool(ORBITALS_START_PAT.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting orbital energies block parsing.")
        # Consume header lines (---, blank)
        next(iterator, None)
        next(iterator, None)
        orbitals: list[Orbital] = []
        homo_index: int | None = None
        lumo_index: int | None = None
        last_occupied_index = -1

        try:
            for line in iterator:
                line_stripped = line.strip()
                if not line_stripped or "*Virtual orbitals printed" in line_stripped or "--------" in line_stripped:
                    break

                match = ORBITAL_LINE_PAT.match(line_stripped)
                if match:
                    try:
                        idx_str, occ_str, eh_str, ev_str = match.groups()
                        idx = int(idx_str)
                        occ = float(occ_str)
                        eh = float(eh_str)
                        ev = float(ev_str)
                        orbitals.append(Orbital(index=idx, occupation=occ, energy_eh=eh, energy_ev=ev))
                        if occ > OCC_THRESHOLD:
                            last_occupied_index = idx
                    except ValueError as e:
                        raise ParsingError(f"Could not parse orbital line values: {line_stripped}") from e
                elif orbitals:  # Stop if format breaks after finding some orbitals
                    break

            if not orbitals:
                raise ParsingError("Orbital energies block found but no orbitals could be parsed.")

            if last_occupied_index != -1:
                homo_index = last_occupied_index
                if homo_index + 1 < len(orbitals):
                    # Check index consistency
                    if orbitals[homo_index + 1].index == homo_index + 1:
                        lumo_index = homo_index + 1
                    else:
                        logger.warning(
                            f"LUMO index mismatch. Expected {homo_index + 1}, found {orbitals[homo_index + 1].index}"
                        )

            orbital_data = OrbitalData(orbitals=tuple(orbitals), homo_index=homo_index, lumo_index=lumo_index)
            results.orbitals = orbital_data
            results.parsed_orbitals = True
            logger.debug(f"Successfully parsed orbital data: {repr(orbital_data)}")

        except Exception as e:
            logger.error(f"Error parsing orbital energies block: {e}", exc_info=True)
            # Allow parsing to continue for other sections
            results.parsed_orbitals = True  # Mark as attempted


# --- Charges Parsers --- #
MULLIKEN_CHARGES_START_PAT = re.compile(r"MULLIKEN ATOMIC CHARGES")
LOEWDIN_CHARGES_START_PAT = re.compile(r"LOEWDIN ATOMIC CHARGES")
CHARGE_LINE_PAT = re.compile(r"^\s*(\d+)\s+[A-Za-z]{1,3}\s+:\s+(-?\d+\.\d+)")


class ChargesParser:
    """Parses atomic charges blocks (Mulliken, Loewdin)."""

    def __init__(self, method: Literal["Mulliken", "Loewdin"], start_pattern: Pattern[str]):
        self.method = method
        self.start_pattern = start_pattern

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        # Allow multiple charge blocks, but only parse each *type* once if needed
        # has_parsed_this_method = any(c.method == self.method for c in current_data.atomic_charges)
        # For now, let's allow reparsing if the block appears again (might overwrite)
        # If strict single parse is needed, use: `and not has_parsed_this_method`
        return bool(self.start_pattern.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug(f"Starting {self.method} charges block parsing.")
        # Consume header lines (----)
        next(iterator, None)
        charges: dict[int, float] = {}
        try:
            for line in iterator:
                line_stripped = line.strip()
                if not line_stripped:
                    break

                match = CHARGE_LINE_PAT.match(line_stripped)
                if match:
                    try:
                        idx_str, charge_str = match.groups()
                        atom_index = int(idx_str)
                        charge = float(charge_str)
                        charges[atom_index] = charge
                    except ValueError as e:
                        raise ParsingError(f"Could not parse {self.method} charge line values: {line_stripped}") from e
                elif charges:  # Stop if format breaks
                    break

            if not charges:
                raise ParsingError(f"{self.method} charges block found but no charges could be parsed.")

            charge_data = AtomicCharges(method=self.method, charges=charges)
            # Append or replace? Let's append for now.
            results.atomic_charges.append(charge_data)
            logger.debug(f"Successfully parsed {self.method} charges: {repr(charge_data)}")

        except Exception as e:
            logger.error(f"Error parsing {self.method} charges block: {e}", exc_info=True)
            # Allow parsing to continue


# --- Dipole Moment Parser --- #
DIPOLE_START_PAT = re.compile(r"DIPOLE MOMENT")
DIPOLE_TOTAL_LINE_PAT = re.compile(r"Total Dipole Moment\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
DIPOLE_MAG_AU_PAT = re.compile(r"Magnitude \(a\.u\.\)\s+:\s+(\d+\.\d+)")
DIPOLE_MAG_DEBYE_PAT = re.compile(r"Magnitude \(Debye\)\s+:\s+(\d+\.\d+)")


class DipoleParser:
    """Parses the dipole moment block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return not current_data.parsed_dipole and bool(DIPOLE_START_PAT.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting dipole moment block parsing.")
        # Consume header lines (----)
        next(iterator, None)
        x_au: float | None = None
        y_au: float | None = None
        z_au: float | None = None
        total_au: float | None = None
        total_debye: float | None = None

        try:
            for line in iterator:
                total_match = DIPOLE_TOTAL_LINE_PAT.search(line)
                if total_match:
                    try:
                        x_au, y_au, z_au = map(float, total_match.groups())
                    except (ValueError, IndexError) as e:
                        raise ParsingError(f"Could not parse dipole components: {line.strip()}") from e
                    continue

                mag_au_match = DIPOLE_MAG_AU_PAT.search(line)
                if mag_au_match:
                    try:
                        total_au = float(mag_au_match.group(1))
                    except (ValueError, IndexError) as e:
                        raise ParsingError(f"Could not parse dipole magnitude (a.u.): {line.strip()}") from e
                    continue

                mag_debye_match = DIPOLE_MAG_DEBYE_PAT.search(line)
                if mag_debye_match:
                    try:
                        total_debye = float(mag_debye_match.group(1))
                    except (ValueError, IndexError) as e:
                        raise ParsingError(f"Could not parse dipole magnitude (Debye): {line.strip()}") from e
                    break  # Found all info

                if "Rotational spectrum" in line or "--------" in line:
                    # If we haven't found all components yet, log a warning but break
                    if None in [x_au, y_au, z_au, total_au, total_debye]:
                        logger.warning(f"Exiting dipole block prematurely due to terminator: '{line.strip()}'")
                    break  # End of dipole section

            if None in [x_au, y_au, z_au, total_au, total_debye]:
                missing = [
                    name
                    for name, val in zip(
                        ["X", "Y", "Z", "Mag(au)", "Mag(Debye)"],
                        [x_au, y_au, z_au, total_au, total_debye],
                        strict=False,
                    )
                    if val is None
                ]
                raise ParsingError(f"Dipole Moment block found but could not parse components: {', '.join(missing)}")

            assert x_au is not None and y_au is not None and z_au is not None
            assert total_au is not None and total_debye is not None

            dipole_data = DipoleMomentData(x_au=x_au, y_au=y_au, z_au=z_au, total_au=total_au, total_debye=total_debye)
            results.dipole_moment = dipole_data
            results.parsed_dipole = True
            logger.debug(f"Successfully parsed dipole moment: {repr(dipole_data)}")

        except Exception as e:
            logger.error(f"Error parsing dipole moment block: {e}", exc_info=True)
            results.parsed_dipole = True  # Mark as attempted


# --- Dispersion Correction Parser --- #
DISPERSION_START_PAT = re.compile(r"DFT DISPERSION CORRECTION")
DISPERSION_METHOD_PAT = re.compile(r"(DFTD\d V\d.*?)(?:\n|$)")  # Basic capture
DISPERSION_ENERGY_PAT = re.compile(r"Dispersion correction\s+(-?\d+\.\d+)")


class DispersionParser:
    """Parses the dispersion correction block."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        return not current_data.parsed_dispersion and bool(DISPERSION_START_PAT.search(line))

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        logger.debug("Starting dispersion correction block parsing.")
        # Consume header lines (----)
        next(iterator, None)
        method: str | None = None
        energy_eh: float | None = None

        try:
            for line in iterator:
                method_match = DISPERSION_METHOD_PAT.search(line)
                if method_match and method is None:  # Capture first method line encountered
                    method = method_match.group(1).strip()
                    # TODO: Refine method extraction (e.g., 'D3BJ', 'D4')
                    continue

                energy_match = DISPERSION_ENERGY_PAT.search(line)
                if energy_match:
                    try:
                        energy_eh = float(energy_match.group(1))
                    except (ValueError, IndexError) as e:
                        raise ParsingError(f"Could not parse dispersion energy: {line.strip()}") from e
                    break  # Found energy, assume done

                # Define terminators
                if "FINAL SINGLE POINT ENERGY" in line or "TIMINGS" in line or line.strip() == "-" * 60:
                    # If we haven't found energy yet, log warning before breaking
                    if energy_eh is None:
                        logger.warning(f"Exiting dispersion block prematurely due to terminator: '{line.strip()}'")
                    break

            if method is None or energy_eh is None:
                missing = [n for n, v in [("Method", method), ("Energy", energy_eh)] if v is None]
                raise ParsingError(f"Dispersion Correction block found but could not parse: {', '.join(missing)}")

            dispersion_data = DispersionCorrectionData(method=method, energy_eh=energy_eh)
            results.dispersion_correction = dispersion_data
            results.parsed_dispersion = True
            logger.debug(f"Successfully parsed dispersion correction: {repr(dispersion_data)}")

        except Exception as e:
            logger.error(f"Error parsing dispersion correction block: {e}", exc_info=True)
            results.parsed_dispersion = True  # Mark as attempted


# --- Parser Registry --- #
# Order might matter if sections can be nested or ambiguous, but generally shouldn't
# Place more specific/complex parsers (like SCF) earlier if needed.
PARSER_REGISTRY: Sequence[SectionParser] = [
    GeometryParser(),
    ScfParser(),  # SCF must come before orbitals/charges which might appear inside guess prints
    OrbitalsParser(),
    ChargesParser("Mulliken", MULLIKEN_CHARGES_START_PAT),
    ChargesParser("Loewdin", LOEWDIN_CHARGES_START_PAT),
    # Add MayerParser here if implemented
    DipoleParser(),
    DispersionParser(),
]


# --- Main Parsing Function --- #
def parse_orca_output(output: str) -> CalculationData:
    """
    Parses the text output of an ORCA calculation using a registry of section parsers.

    Args:
        output: The string content of the ORCA output file.

    Returns:
        A CalculationData object containing the parsed results.

    Raises:
        ParsingError: If essential components (like geometry) cannot be parsed,
                      or if critical errors occur during parsing.
    """
    lines = output.splitlines()
    # Use PeekingIterator if lookahead is needed, but avoid for now
    line_iterator = iter(lines)
    results = _MutableCalculationData(raw_output=output)  # Start with mutable data

    current_line_num = 0
    try:
        while True:  # Use explicit break or StopIteration
            try:
                line = next(line_iterator)
                current_line_num += 1
            except StopIteration:
                logger.debug("Reached end of input.")
                break

            # Check registered block parsers first
            parser_found = False
            for parser in PARSER_REGISTRY:
                try:
                    if parser.matches(line, results):
                        parser.parse(line_iterator, line, results)
                        parser_found = True
                        break  # Only one parser should handle the start of a block
                except ParsingError as e:
                    # Propagate critical errors, log others based on parser logic
                    logger.error(f"Parser {type(parser).__name__} failed critically: {e}", exc_info=True)
                    raise  # Re-raise critical errors
                except Exception as e:
                    # Catch unexpected errors within a parser's logic
                    logger.error(f"Unexpected error in {type(parser).__name__}: {e}", exc_info=True)
                    # Decide whether to raise or try to continue
                    # For robustness, let's try to continue, but log it as an error.
                    # Mark the relevant section as attempted if possible (depends on parser impl)

            if parser_found:
                continue  # Move to the next line after block parsing consumed lines

            # --- Handle Line-Specific Information (if not handled by a block parser) ---
            match_final_energy = FINAL_ENERGY_PAT.search(line)
            if match_final_energy:
                try:
                    # Overwrite if found multiple times, last one is usually relevant
                    results.final_energy_eh = float(match_final_energy.group(1))
                    logger.debug(f"Found Final Single Point Energy: {results.final_energy_eh}")
                except (ValueError, IndexError) as e:
                    logger.error(f"Could not parse final energy value from line: {line.strip()}", exc_info=True)
                    # Decide if this is critical - likely yes?
                    raise ParsingError("Failed to parse final energy value.") from e
                continue

            # --- Termination Status ---
            if NORMAL_TERM_PAT.search(line):
                results.termination_status = "NORMAL"
                logger.debug("Found Normal Termination.")
                # Don't break, continue parsing until the end
            elif ERROR_TERM_PAT.search(line) and results.termination_status == "UNKNOWN":
                results.termination_status = "ERROR"
                logger.debug("Found Error Termination.")

    except ParsingError:
        # Re-raise critical parsing errors immediately
        raise
    except Exception as e:
        # Catch unexpected errors during the main loop iteration
        logger.critical(f"Unexpected error in main parsing loop at line ~{current_line_num}: {e}", exc_info=True)
        # Preserve original traceback if possible?
        # traceback.print_exc() # Avoid direct print
        results.termination_status = "ERROR"  # Mark as error if unexpected exception happens
        # Optionally re-raise as a ParsingError
        raise ParsingError(f"An unexpected error occurred during parsing: {e}") from e

    # --- Final Checks ---
    if results.input_geometry is None:
        logger.error("Input geometry block was not found or parsed.")
        raise ParsingError("Input geometry block was not found in the output file.")

    # Refine termination status if still UNKNOWN
    if results.termination_status == "UNKNOWN":
        logger.warning("Termination status unknown after parsing, assuming ERROR.")
        results.termination_status = "ERROR"

    # Final check for SCF data if expected (depends on calculation type?)
    # For now, just a warning if it wasn't parsed successfully.
    if results.scf is None and results.parsed_scf:  # Check if parsing was attempted but failed
        logger.warning("SCF data block parsing was attempted but failed or did not find required components.")
    elif not results.parsed_scf:  # Check if the block wasn't even found
        logger.warning("SCF data block was not found.")

    logger.info(f"ORCA parsing finished. Status: {results.termination_status}, Final Energy: {results.final_energy_eh}")
    return CalculationData.from_mutable(results)  # Convert back to immutable dataclass

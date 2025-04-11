import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pprint import pformat
from re import Match
from typing import Literal

# Assuming exceptions.py is in the same parent directory or PYTHONPATH
from calcflow.exceptions import ParsingError  # Relative import

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
        return (
            f"{type(self).__name__}(n_orbitals={n_orbitals}, "
            f"homo_index={self.homo_index}, lumo_index={self.lumo_index}, "
            f"homo_energy_eh={homo_energy:.6f}, lumo_energy_eh={lumo_energy:.6f})"
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


@dataclass(frozen=True)
class CalculationData:
    """Top-level container for parsed ORCA calculation results."""

    raw_output: str = field(repr=False)  # Store the original full output first
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"]
    input_geometry: Sequence[Atom] | None = None  # Allow None initially
    final_energy_eh: float | None = None

    # Optional result components
    scf: ScfData | None = None
    orbitals: OrbitalData | None = None
    atomic_charges: list[AtomicCharges] = field(default_factory=list)
    dipole_moment: DipoleMomentData | None = None
    dispersion_correction: DispersionCorrectionData | None = None

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
            # Use default list repr which will call our custom repr for AtomicCharges
            lines.append(f"  atomic_charges={repr(self.atomic_charges)},")
        if self.dipole_moment:
            lines.append(f"  dipole_moment={repr(self.dipole_moment)},")
        if self.dispersion_correction:
            lines.append(f"  dispersion_correction={repr(self.dispersion_correction)},")

        # Clean up trailing comma on last line if any optional fields were added
        if len(lines) > 4:
            lines[-1] = lines[-1].rstrip(",")

        lines.append(")")
        return "\n".join(lines)


# --- Parsing Logic --- #

# Regex Patterns (pre-compile for efficiency)
GEOMETRY_START_PAT = re.compile(r"CARTESIAN COORDINATES \(ANGSTROEM\)")
GEOMETRY_LINE_PAT = re.compile(r"^\s*([A-Za-z]{1,3})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
FINAL_ENERGY_PAT = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
NORMAL_TERM_PAT = re.compile(r"\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*")
ERROR_TERM_PAT = re.compile(r"(TERMINATING THE PROGRAM|ORCA finished with error)")  # Add more error patterns

# --- SCF Patterns --- #
# Removed SCF_START_PAT
# Removed SCF_CONVERGED_PAT
SCF_CONVERGED_LINE_PAT = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")  # Pattern for the specific line
SCF_STEP_PAT = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)")  # Basic Iteration Energy (fallback)
# Define a robust float pattern
FLOAT_PAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
# DIIS Iteration Line: Iter Energy Delta-E RMSDP MaxDP DIISErr Damp Time
SCF_DIIS_ITER_PAT = re.compile(
    rf"^\s*(\d+)\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}"
)
# SOSCF Iteration Line: Iter Energy Delta-E RMSDP MaxDP MaxGrad Time
SCF_SOSCF_ITER_PAT = re.compile(
    rf"^\s*(\d+)\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}\s+{FLOAT_PAT}"
)
SCF_ENERGY_COMPONENTS_START_PAT = re.compile(r"TOTAL SCF ENERGY")
SCF_NUCLEAR_REP_PAT = re.compile(r"^\s*Nuclear Repulsion\s*:\s*(-?\d+\.\d+)")
SCF_ELECTRONIC_PAT = re.compile(r"^\s*Electronic Energy\s*:\s*(-?\d+\.\d+)")
SCF_ONE_ELECTRON_PAT = re.compile(r"^\s*One Electron Energy\s*:\s*(-?\d+\.\d+)")
SCF_TWO_ELECTRON_PAT = re.compile(r"^\s*Two Electron Energy\s*:\s*(-?\d+\.\d+)")
SCF_XC_PAT = re.compile(r"^\s*E\(XC\)\s*:\s*(-?\d+\.\d+)")

# --- Orbital Patterns --- #
ORBITALS_START_PAT = re.compile(r"ORBITAL ENERGIES")
ORBITAL_LINE_PAT = re.compile(r"^\s*(\d+)\s+([\d\.]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")  # NO OCC E(Eh) E(eV)

# --- Population Analysis Patterns --- #
MULLIKEN_CHARGES_START_PAT = re.compile(r"MULLIKEN ATOMIC CHARGES")
LOEWDIN_CHARGES_START_PAT = re.compile(r"LOEWDIN ATOMIC CHARGES")
# Note: Mayer charges are often within a different table structure, may need separate logic if added.
CHARGE_LINE_PAT = re.compile(r"^\s*(\d+)\s+[A-Za-z]{1,3}\s+:\s+(-?\d+\.\d+)")  # e.g., 0 H :  0.172827

# --- Dipole Moment Patterns --- #
DIPOLE_START_PAT = re.compile(r"DIPOLE MOMENT")
DIPOLE_TOTAL_LINE_PAT = re.compile(
    r"Total Dipole Moment\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)  # X Y Z (a.u.)
DIPOLE_MAG_AU_PAT = re.compile(r"Magnitude \(a\.u\.\)\s+:\s+(\d+\.\d+)")
DIPOLE_MAG_DEBYE_PAT = re.compile(r"Magnitude \(Debye\)\s+:\s+(\d+\.\d+)")

# --- Dispersion Correction Patterns --- #
DISPERSION_START_PAT = re.compile(r"DFT DISPERSION CORRECTION")
# Simple capture for now, refinement might be needed for specific D3/D4 variants
DISPERSION_METHOD_PAT = re.compile(r"(DFTD\d V\d.*?)(?:\n|$)")  # Capture DFTD3/4 line
DISPERSION_ENERGY_PAT = re.compile(r"Dispersion correction\s+(-?\d+\.\d+)")


def _parse_geometry_block(line_iterator: Iterator[str]) -> Sequence[Atom]:
    """
    Parses the geometry block, consuming lines from the iterator.
    Assumes the iterator is positioned *after* the header line.
    """
    geometry: list[Atom] = []
    for line in line_iterator:
        line_stripped = line.strip()
        if not line_stripped:
            break  # Blank line marks the end

        match = GEOMETRY_LINE_PAT.match(line_stripped)
        if match:
            symbol, x, y, z = match.groups()
            geometry.append(Atom(symbol=symbol, x=float(x), y=float(y), z=float(z)))
        else:
            # Should ideally not happen if block structure is consistent
            # Could raise error or log a warning
            break  # Stop parsing this block on unexpected format

    if not geometry:
        raise ParsingError("Geometry block found but no atoms could be parsed.")
    return tuple(geometry)


# Placeholder for SCF parser
def _parse_scf_block(line_iterator: Iterator[str], initial_line: str) -> ScfData:
    """
    Parses the SCF block, consuming lines from the iterator.
    Assumes the iterator is positioned *after* the header line that triggered the call.
    Searches for convergence message, iteration count, final energy, and components.

    Args:
        line_iterator: The iterator over the output file lines.
        initial_line: The line that triggered the call (e.g., DIIS/SOSCF header).

    Returns:
        A ScfData object containing the parsed SCF data.

    Raises:
        ParsingError: If there is an error during parsing.
    """
    converged = False
    n_iterations = 0
    last_scf_energy_eh: float | None = None
    nuclear_rep_eh: float | None = None
    electronic_eh: float | None = None
    one_electron_eh: float | None = None
    two_electron_eh: float | None = None
    xc_eh: float | None = None
    latest_iter_num = 0
    iteration_history: list[ScfIteration] = []  # Initialize history list

    in_iteration_table = True  # Start assuming we are in a table based on trigger
    current_table_type = None
    in_scf_component_section = False  # Flag for component parsing state
    found_all_mandatory_components = False  # Flag to break early if needed

    # Determine table type from the initial trigger line
    if "D-I-I-S" in initial_line:
        current_table_type = "DIIS"
        next(line_iterator, None)
    elif "S-O-S-C-F" in initial_line:
        current_table_type = "SOSCF"
        next(line_iterator, None)
    else:
        raise ParsingError("_parse_scf_block called with unexpected initial line")

    try:  # Add try block to catch errors within SCF parsing
        for line in line_iterator:
            line_stripped = line.strip()

            # --- State 1: Parsing Iteration Tables --- #
            if in_iteration_table:
                if "D-I-I-S" in line:
                    current_table_type = "DIIS"
                    next(line_iterator, None)
                    next(line_iterator, None)
                    continue
                if "S-O-S-C-F" in line:
                    current_table_type = "SOSCF"
                    next(line_iterator, None)
                    next(line_iterator, None)
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
                            raise ParsingError(f"Could not parse DIIS iteration: {line_stripped} - {e}") from e
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
                            raise ParsingError(f"Could not parse SOSCF iteration: {line_stripped} - {e}") from e

                if iter_data:
                    iteration_history.append(iter_data)
                    latest_iter_num = max(latest_iter_num, iter_data.iteration)
                    last_scf_energy_eh = iter_data.energy_eh
                    continue
                elif line_stripped.startswith(("*", " ")) and "SCF CONVERGED AFTER" in line:
                    conv_match: Match[str] | None = SCF_CONVERGED_LINE_PAT.search(line)
                    if conv_match:
                        converged = True
                        n_iterations = int(conv_match.group(1))
                        in_iteration_table = False
                        continue
                    else:
                        continue
                elif not line_stripped:
                    in_iteration_table = False
                    continue
                elif SCF_ENERGY_COMPONENTS_START_PAT.search(line):
                    in_iteration_table = False
                    in_scf_component_section = True
                    # Don't consume line here

            # --- State 2: Parsing SCF Energy Components --- #
            elif in_scf_component_section:
                # Attempt to parse all relevant component lines
                parsed_this_line = False
                nr_match = SCF_NUCLEAR_REP_PAT.search(line)
                if nr_match:
                    nuclear_rep_eh = float(nr_match.group(1))
                    parsed_this_line = True
                    # print(f"DEBUG: Parsed Nuclear Repulsion: {nuclear_rep_eh}")

                el_match = SCF_ELECTRONIC_PAT.search(line)
                if el_match:
                    electronic_eh = float(el_match.group(1))
                    parsed_this_line = True
                    # print(f"DEBUG: Parsed Electronic Energy: {electronic_eh}")

                one_el_match = SCF_ONE_ELECTRON_PAT.search(line)
                if one_el_match:
                    try:
                        one_electron_eh = float(one_el_match.group(1))
                        parsed_this_line = True
                    except ValueError as e:
                        print(f"ERROR: Could not convert One Electron Energy: {one_el_match.group(1)} - {e}")  # DEBUG

                two_el_match = SCF_TWO_ELECTRON_PAT.search(line)
                if two_el_match:
                    try:
                        two_electron_eh = float(two_el_match.group(1))
                        parsed_this_line = True
                    except ValueError as e:
                        print(f"ERROR: Could not convert Two Electron Energy: {two_el_match.group(1)} - {e}")  # DEBUG

                xc_match = SCF_XC_PAT.search(line)
                if xc_match:
                    # print(f"DEBUG: Checking XC Energy: '{line_stripped}', Match: {xc_match}") # DEBUG
                    try:
                        xc_eh = float(xc_match.group(1))
                        parsed_this_line = True
                        # print(f"DEBUG: Parsed XC Energy: {xc_eh}")
                    except ValueError as e:
                        print(f"ERROR: Could not convert XC Energy: {xc_match.group(1)} - {e}")  # DEBUG

                # Update mandatory component check *after* all parse attempts for the line
                if not found_all_mandatory_components and None not in [
                    nuclear_rep_eh,
                    electronic_eh,
                    one_electron_eh,
                    two_electron_eh,
                ]:
                    found_all_mandatory_components = True

                # Check for termination conditions ONLY if not a component line, OR after mandatory components found
                # This prevents terminating on intermediate lines like "Components:"
                if not parsed_this_line or found_all_mandatory_components:
                    is_terminator = False
                    # Define strict terminators that signify the end of the SCF details
                    strict_terminators = (
                        "SCF CONVERGENCE",  # Start of convergence details table
                        "ORBITAL ENERGIES",
                        "MULLIKEN POPULATION ANALYSIS",
                        "LOEWDIN POPULATION ANALYSIS",
                        "MAYER POPULATION ANALYSIS",
                        "DFT DISPERSION CORRECTION",
                        "FINAL SINGLE POINT ENERGY",
                        "TIMINGS",
                        "------",
                    )
                    if any(term in line for term in strict_terminators):
                        # Require mandatory components before terminating on ambiguous lines like ------
                        if line.strip().startswith("------") and not found_all_mandatory_components:
                            pass  # Ignore dashes before components found
                        else:
                            is_terminator = True

                    if is_terminator:
                        in_scf_component_section = False
                        # Break the _parse_scf_block loop
                        break
                # else: If it was a component line or not a terminator, just continue the loop
                continue  # Continue processing lines in component section

            # --- State 3: After SCF iterations & components --- #
            else:
                if SCF_ENERGY_COMPONENTS_START_PAT.search(line):
                    in_scf_component_section = True
                    continue

                if not converged:
                    conv_match_late: Match[str] | None = SCF_CONVERGED_LINE_PAT.search(line)
                    if conv_match_late:
                        converged = True
                        n_iterations = int(conv_match_late.group(1))
                        continue

                # Break if next major section starts
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
                    break

    except Exception as e:
        raise ParsingError(f"Error during SCF block parsing: {e}") from e

    # --- Final Checks and Object Creation --- #
    if not converged:
        print("Warning: SCF convergence line 'SCF CONVERGED AFTER ...' not found. Assuming not converged.")
        converged = False

    if n_iterations == 0 and latest_iter_num > 0:
        n_iterations = latest_iter_num
    elif n_iterations > 0 and latest_iter_num > n_iterations:
        print(
            f"Warning: Convergence line reported {n_iterations} cycles, but found {latest_iter_num} in history. Using history count."
        )
        n_iterations = latest_iter_num
    elif n_iterations == 0 and not iteration_history:
        print("Warning: No SCF iterations found or parsed, and convergence line not found.")
        n_iterations = 0

    if converged and not iteration_history and n_iterations > 0:
        print(f"Warning: SCF convergence line reported {n_iterations} cycles, but no iteration history parsed.")
        if last_scf_energy_eh is None:
            raise ParsingError(f"SCF converged after {n_iterations} cycles, but no iteration history or energy parsed.")

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
            print(f"ERROR: Missing mandatory SCF energy components after SCF ran/converged: {missing_comps}")
            raise ParsingError("Could not parse all required SCF energy components after SCF execution.")

    elif not found_all_mandatory_components:
        print("Warning: SCF did not run or failed immediately; energy components not found.")
        raise ParsingError("SCF energy components not found, calculation likely failed before SCF completion.")

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
        final_scf_energy = last_scf_energy_eh
        print(f"Warning: Using fallback SCF energy {final_scf_energy} as iteration history is missing.")
    else:
        if converged:
            raise ParsingError("SCF converged but failed to extract final energy.")
        raise ParsingError("Failed to determine final SCF energy (calculation may have failed early).")

    scf_result = ScfData(
        converged=converged,
        energy_eh=final_scf_energy,
        components=components,
        n_iterations=n_iterations,
        iteration_history=tuple(iteration_history),
    )
    return scf_result


# Placeholder for Orbitals parser
def _parse_orbitals_block(line_iterator: Iterator[str]) -> OrbitalData:
    """
    Parses the ORBITAL ENERGIES block.
    Assumes the iterator is positioned *after* the header line.
    """
    orbitals: list[Orbital] = []
    homo_index: int | None = None
    lumo_index: int | None = None
    last_occupied_index = -1

    for line in line_iterator:
        line_stripped = line.strip()
        if not line_stripped or "*Virtual orbitals printed" in line_stripped:
            break  # End of table

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
                raise ParsingError(f"Could not parse orbital line values: {line_stripped} - {e}") from e
        elif orbitals:  # If we already parsed some orbitals and hit a non-matching line
            break  # Assume end of table

    if not orbitals:
        raise ParsingError("Orbital energies block found but no orbitals could be parsed.")

    # Determine HOMO/LUMO based on the last occupied index found
    if last_occupied_index != -1:
        homo_index = last_occupied_index
        # LUMO is the next orbital, if it exists
        if homo_index + 1 < len(orbitals):
            # Check if the index matches - ORCA numbers orbitals from 0
            if orbitals[homo_index + 1].index == homo_index + 1:
                lumo_index = homo_index + 1
            else:
                # This case should be unlikely if orbitals are contiguous
                print(
                    f"Warning: LUMO index mismatch. Expected {homo_index + 1}, found {orbitals[homo_index + 1].index}"
                )

    return OrbitalData(orbitals=tuple(orbitals), homo_index=homo_index, lumo_index=lumo_index)


# Placeholder for Charges parser
def _parse_charges_block(line_iterator: Iterator[str], method: Literal["Mulliken", "Loewdin"]) -> AtomicCharges:
    """
    Parses a block of atomic charges (Mulliken or Loewdin).
    Assumes the iterator is positioned *after* the header and dashed lines.
    """
    charges: dict[int, float] = {}
    for line in line_iterator:
        line_stripped = line.strip()
        if not line_stripped:
            break  # Blank line signals end

        match = CHARGE_LINE_PAT.match(line_stripped)
        if match:
            try:
                idx_str, charge_str = match.groups()
                atom_index = int(idx_str)  # 0-based index from ORCA output
                charge = float(charge_str)
                charges[atom_index] = charge
            except ValueError as e:
                raise ParsingError(f"Could not parse charge line values: {line_stripped} - {e}") from e
        elif charges:  # If we already parsed some charges and hit a non-matching line
            break  # Assume end of block

    if not charges:
        raise ParsingError(f"{method} charges block found but no charges could be parsed.")

    return AtomicCharges(method=method, charges=charges)


# Placeholder for Dipole parser
def _parse_dipole_block(line_iterator: Iterator[str]) -> DipoleMomentData:
    """
    Parses the DIPOLE MOMENT block.
    Assumes the iterator is positioned *after* the header line.
    """
    x_au: float | None = None
    y_au: float | None = None
    z_au: float | None = None
    total_au: float | None = None
    total_debye: float | None = None

    for line in line_iterator:
        total_match = DIPOLE_TOTAL_LINE_PAT.search(line)
        if total_match:
            try:
                x_au = float(total_match.group(1))
                y_au = float(total_match.group(2))
                z_au = float(total_match.group(3))
            except (ValueError, IndexError) as e:
                raise ParsingError(f"Could not parse dipole components: {line.strip()} - {e}") from e
            continue  # Keep looking for magnitude lines

        mag_au_match = DIPOLE_MAG_AU_PAT.search(line)
        if mag_au_match:
            try:
                total_au = float(mag_au_match.group(1))
            except (ValueError, IndexError) as e:
                raise ParsingError(f"Could not parse dipole magnitude (a.u.): {line.strip()} - {e}") from e
            continue

        mag_debye_match = DIPOLE_MAG_DEBYE_PAT.search(line)
        if mag_debye_match:
            try:
                total_debye = float(mag_debye_match.group(1))
            except (ValueError, IndexError) as e:
                raise ParsingError(f"Could not parse dipole magnitude (Debye): {line.strip()} - {e}") from e
            # Once Debye is found, assume block is done for basic dipole
            break

        # Break if we hit unrelated sections like Rotational spectrum
        if "Rotational spectrum" in line:
            break

    # Check if all components were found
    if None in [x_au, y_au, z_au, total_au, total_debye]:
        missing = [
            name
            for name, val in zip(
                ["X", "Y", "Z", "Mag(au)", "Mag(Debye)"], [x_au, y_au, z_au, total_au, total_debye], strict=False
            )
            if val is None
        ]
        raise ParsingError(f"Dipole Moment block found but could not parse components: {', '.join(missing)}")

    # Assert types for the checker after the None check
    assert x_au is not None
    assert y_au is not None
    assert z_au is not None
    assert total_au is not None
    assert total_debye is not None

    return DipoleMomentData(x_au=x_au, y_au=y_au, z_au=z_au, total_au=total_au, total_debye=total_debye)


# Placeholder for Dispersion parser
def _parse_dispersion_block(line_iterator: Iterator[str]) -> DispersionCorrectionData:
    """
    Parses the DFT DISPERSION CORRECTION block.
    Assumes the iterator is positioned *after* the header line.
    """
    method: str | None = None
    energy_eh: float | None = None

    for line in line_iterator:
        method_match = DISPERSION_METHOD_PAT.search(line)
        if method_match:
            # Capture the full method line for now
            # TODO: Refine this to extract canonical names like 'D30', 'D3BJ', 'D4'
            method = method_match.group(1).strip()
            continue  # Keep looking for energy

        energy_match = DISPERSION_ENERGY_PAT.search(line)
        if energy_match:
            try:
                energy_eh = float(energy_match.group(1))
            except (ValueError, IndexError) as e:
                raise ParsingError(f"Could not parse dispersion energy: {line.strip()} - {e}") from e
            # Found energy, assume block is done for basic dispersion info
            break

        # Add break conditions if necessary, e.g., hitting FINAL SINGLE POINT ENERGY
        if "FINAL SINGLE POINT ENERGY" in line:
            break

    if method is None or energy_eh is None:
        missing = []
        if method is None:
            missing.append("Method")
        if energy_eh is None:
            missing.append("Energy")
        raise ParsingError(f"Dispersion Correction block found but could not parse: {', '.join(missing)}")

    return DispersionCorrectionData(method=method, energy_eh=energy_eh)


def parse_orca_output(output: str) -> CalculationData:
    """
    Parses the text output of an ORCA calculation using a single pass.

    Args:
        output: The string content of the ORCA output file.

    Returns:
        A CalculationData object containing the parsed results.

    Raises:
        ParsingError: If essential components cannot be parsed or structural
                      errors are detected in the output.
    """
    lines = output.splitlines()
    line_iterator = iter(lines)  # Create an iterator for single pass

    # Initialize result fields
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"] = "UNKNOWN"
    input_geometry: Sequence[Atom] | None = None
    final_energy_eh: float | None = None
    scf_data: ScfData | None = None
    orbital_data: OrbitalData | None = None
    atomic_charges_list: list[AtomicCharges] = []
    dipole_data: DipoleMomentData | None = None
    dispersion_data: DispersionCorrectionData | None = None

    try:
        for line in line_iterator:
            # print(f"DEBUG Main Loop: Processing line: '{line[:80]}'") # Very verbose DEBUG
            # --- Block Detection ---
            if GEOMETRY_START_PAT.search(line):
                # Consume header lines (assumes 2 lines: dashes, blank)
                next(line_iterator, None)
                # next(line_iterator, None) # The loop condition handles blank lines now
                if input_geometry is None:  # Parse only the *first* geometry block encountered
                    # print("DEBUG Main Loop: Found Geometry block")  # DEBUG
                    input_geometry = _parse_geometry_block(line_iterator)
                # Add logic here if multiple geometry blocks need handling (e.g., opt steps)
                continue  # Move to next line after block parsing

            # Detect SCF iteration table start (DIIS or SOSCF)
            # Check this *before* other blocks that might appear within SCF (like orbitals in guess)
            if scf_data is None and ("D-I-I-S" in line or "S-O-S-C-F" in line):
                try:
                    # Pass the iterator and the trigger line itself
                    scf_data = _parse_scf_block(line_iterator, line)
                except ParsingError as e:
                    # Log warning? Or re-raise depending on strictness
                    print(f"Warning: Could not fully parse SCF block: {e}")
                    # Continue parsing other sections if possible
                continue  # Important to continue to next line after parsing block

            # Detect Orbital block start
            if ORBITALS_START_PAT.search(line):
                if orbital_data is None:  # Parse only the first block
                    # print("DEBUG Main Loop: Found Orbitals block")  # DEBUG
                    # Consume header lines (assumes ---, blank)
                    next(line_iterator, None)
                    next(line_iterator, None)
                    try:
                        orbital_data = _parse_orbitals_block(line_iterator)
                    except ParsingError as e:
                        print(f"Warning: Could not fully parse Orbital Energies block: {e}")
                continue

            # Detect Mulliken Charges block start
            if MULLIKEN_CHARGES_START_PAT.search(line):
                # Consume header lines (----)
                next(line_iterator, None)
                # print("DEBUG Main Loop: Found Mulliken block")  # DEBUG
                try:
                    # Avoid parsing duplicates if block appears multiple times
                    if not any(c.method == "Mulliken" for c in atomic_charges_list):
                        atomic_charges_list.append(_parse_charges_block(line_iterator, "Mulliken"))
                except ParsingError as e:
                    print(f"Warning: Could not fully parse Mulliken Charges block: {e}")
                continue

            # Detect Loewdin Charges block start
            if LOEWDIN_CHARGES_START_PAT.search(line):
                # Consume header lines (----)
                next(line_iterator, None)
                # print("DEBUG Main Loop: Found Loewdin block")  # DEBUG
                try:
                    # Avoid parsing duplicates
                    if not any(c.method == "Loewdin" for c in atomic_charges_list):
                        atomic_charges_list.append(_parse_charges_block(line_iterator, "Loewdin"))
                except ParsingError as e:
                    print(f"Warning: Could not fully parse Loewdin Charges block: {e}")
                continue

            # Placeholder: Add detection for Mayer charges (different structure)
            # ...

            # Detect Dipole Moment block start
            if DIPOLE_START_PAT.search(line):
                if dipole_data is None:  # Parse only first instance
                    # print("DEBUG Main Loop: Found Dipole block")  # DEBUG
                    # Consume header lines (----)
                    next(line_iterator, None)
                    try:
                        dipole_data = _parse_dipole_block(line_iterator)
                    except ParsingError as e:
                        print(f"Warning: Could not fully parse Dipole Moment block: {e}")
                continue

            # Detect Dispersion block start
            if DISPERSION_START_PAT.search(line):
                if dispersion_data is None:  # Parse first instance
                    # print("DEBUG Main Loop: Found Dispersion block")  # DEBUG
                    # Consume header lines (----)
                    next(line_iterator, None)
                    try:
                        dispersion_data = _parse_dispersion_block(line_iterator)
                    except ParsingError as e:
                        print(f"Warning: Could not fully parse Dispersion Correction block: {e}")
                continue

            # --- Line-Specific Information ---
            match_final_energy = FINAL_ENERGY_PAT.search(line)
            if match_final_energy:
                # print("DEBUG Main Loop: Found Final Energy line")  # DEBUG
                try:
                    # Overwrite if found multiple times, last one usually most relevant
                    final_energy_eh = float(match_final_energy.group(1))
                except (ValueError, IndexError) as e:
                    raise ParsingError(f"Could not parse final energy value from line: {line.strip()}") from e
                continue  # Move to next line

            # --- Termination Status (Checked Primarily Here, Refined at End) ---
            if NORMAL_TERM_PAT.search(line):
                # print("DEBUG Main Loop: Found Normal Termination line")  # DEBUG
                termination_status = "NORMAL"
                # Typically near the end, could potentially break early if needed,
                # but parsing whole file ensures all data captured.
            elif ERROR_TERM_PAT.search(line) and termination_status == "UNKNOWN":
                # Only set error if not already marked as normal
                # print("DEBUG Main Loop: Found Error Termination line")  # DEBUG
                termination_status = "ERROR"

    except StopIteration:
        # Iterator exhausted, expected behavior
        # print("DEBUG Main Loop: Reached end of input")  # DEBUG
        pass
    except Exception as e:
        # Catch unexpected errors during parsing
        print(f"ERROR: Unexpected error in main parsing loop: {e}")  # DEBUG
        import traceback

        traceback.print_exc()
        raise ParsingError(f"An unexpected error occurred during parsing: {e}") from e

    # Final check: Geometry is essential
    if input_geometry is None:
        # print("ERROR: Input geometry block was not found.")  # DEBUG
        raise ParsingError("Input geometry block was not found in the output file.")

    # Refine termination status: If loop finished and it's still UNKNOWN, it likely failed or is incomplete.
    if termination_status == "UNKNOWN":
        # print("Warning: Termination status unknown after parsing, assuming ERROR.")  # DEBUG
        # Decide if this should be "ERROR" or remain "UNKNOWN"
        # For now, let's assume it's an error if not explicitly normal.
        termination_status = "ERROR"

    # Final sanity check for SCF data existence
    if scf_data is None:
        print("Warning: SCF data block was not found or parsed successfully.")  # DEBUG

    # print("DEBUG Main Loop: Constructing final CalculationData object")  # DEBUG
    return CalculationData(
        termination_status=termination_status,
        input_geometry=input_geometry,
        final_energy_eh=final_energy_eh,
        scf=scf_data,
        orbitals=orbital_data,
        atomic_charges=atomic_charges_list,
        dipole_moment=dipole_data,
        dispersion_correction=dispersion_data,
        raw_output=output,  # Store the original full output
    )

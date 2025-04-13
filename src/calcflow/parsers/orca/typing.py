from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pprint import pformat
from typing import Literal, Protocol


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
    # Track errors/warnings during parsing
    parsing_errors: list[str] = field(default_factory=list)
    parsing_warnings: list[str] = field(default_factory=list)
    # Flags to track if sections have been parsed (for sections expected once)
    parsed_geometry: bool = False
    parsed_scf: bool = False
    parsed_orbitals: bool = False
    parsed_dipole: bool = False
    parsed_dispersion: bool = False


@dataclass(frozen=True)
class GradientData:
    """Holds Cartesian gradient information for a specific optimization cycle."""

    gradients: Mapping[int, tuple[float, float, float]]  # Atom index -> (Gx, Gy, Gz)
    norm: float
    rms: float
    max: float


@dataclass(frozen=True)
class RelaxationStepData:
    """Holds geometry relaxation step details and convergence criteria."""

    energy_change: float | None = None
    rms_gradient: float | None = None
    max_gradient: float | None = None
    rms_step: float | None = None
    max_step: float | None = None
    converged_items: Mapping[str, bool] = field(default_factory=dict)  # e.g., {"Energy": True, "Gradient": False}
    trust_radius: float | None = None


@dataclass
class OptimizationCycleData:
    """Stores parsed data for a single geometry optimization cycle."""

    cycle_number: int
    geometry: Sequence[Atom] | None = None  # Geometry *at the start* of this cycle's calculation
    energy_eh: float | None = None  # Usually the SCF energy for this cycle
    scf_data: ScfData | None = None
    dispersion: DispersionCorrectionData | None = None
    gradient: GradientData | None = None
    relaxation_step: RelaxationStepData | None = None
    # Add other per-cycle properties if needed (e.g., orbitals, charges for each step)


@dataclass
class _MutableOptData:
    """Internal mutable container for accumulating optimization results during parsing."""

    raw_output: str
    termination_status: Literal["CONVERGED", "NOT_CONVERGED", "ERROR", "UNKNOWN"] = "UNKNOWN"
    input_geometry: Sequence[Atom] | None = None
    cycles: list[OptimizationCycleData] = field(default_factory=list)
    final_geometry: Sequence[Atom] | None = None  # Converged geometry
    final_energy_eh: float | None = None
    final_scf: ScfData | None = None
    final_orbitals: OrbitalData | None = None
    final_charges: list[AtomicCharges] = field(default_factory=list)
    final_dipole: DipoleMomentData | None = None
    final_dispersion: DispersionCorrectionData | None = None
    n_cycles: int = 0  # Track number of cycles parsed

    # --- Flags to manage parsing state ---
    # These might need adjustment as we add cycle logic
    parsed_input_geometry: bool = False
    # parsed_final_geometry: bool = False # Replaced by in_final_evaluation state
    # We'll need flags/logic for tracking state *within* cycles vs final eval
    # Add flags to track if specific components were parsed in the final block
    parsed_final_scf: bool = False
    parsed_final_orbitals: bool = False
    parsed_final_charges: bool = False
    parsed_final_dipole: bool = False
    parsed_final_dispersion: bool = False
    # Keep track of whether normal termination pattern was seen
    normal_termination_found: bool = False


# --- Parsing Logic --- #

type LineIterator = Iterator[str]


class SectionParser(Protocol):
    """Protocol for section parsers."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line triggers this parser."""
        ...

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parse the section, consuming lines from the iterator and updating results."""
        ...

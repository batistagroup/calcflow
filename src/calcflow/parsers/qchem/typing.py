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
    """Holds data for a single Q-Chem SCF iteration."""

    iteration: int
    energy_eh: float
    diis_error: float  # DIIS error magnitude


@dataclass(frozen=True)
class ScfEnergyComponents:
    """Holds the components of the raw SCF energy (adapt as needed for Q-Chem)."""

    nuclear_repulsion_eh: float | None = None
    # Add other components if found (Electronic, One-electron, Two-electron, XC)
    # Q-Chem output structure for these might differ or require calculation


@dataclass(frozen=True)
class ScfData:
    """Holds results specific to the Q-Chem SCF calculation step."""

    converged: bool
    energy_eh: float  # Converged SCF energy (or last energy if not converged)
    # components: ScfEnergyComponents | None = None # Optional for now
    n_iterations: int
    iteration_history: Sequence[ScfIteration]

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
    """Represents a single molecular orbital in Q-Chem."""

    index: int  # 0-based index
    energy_eh: float  # Energy in Hartrees
    occupation: float | None = None  # Q-Chem might require inference for occupation


@dataclass(frozen=True)
class OrbitalData:
    """Holds information about molecular orbitals from Q-Chem."""

    # Q-Chem separates Alpha and Beta orbitals
    alpha_orbitals: Sequence[Orbital] | None = None
    beta_orbitals: Sequence[Orbital] | None = None
    # HOMO/LUMO indices might need calculation based on occupations
    alpha_homo_index: int | None = None
    alpha_lumo_index: int | None = None
    beta_homo_index: int | None = None
    beta_lumo_index: int | None = None

    def __repr__(self) -> str:
        n_alpha = len(self.alpha_orbitals) if self.alpha_orbitals else 0
        n_beta = len(self.beta_orbitals) if self.beta_orbitals else 0
        # Add HOMO/LUMO details if available later
        return f"{type(self).__name__}(n_alpha_orbitals={n_alpha}, n_beta_orbitals={n_beta})"


@dataclass(frozen=True)
class AtomicCharges:
    """Stores atomic charges from a specific population analysis method (e.g., Mulliken)."""

    method: str  # e.g., "Mulliken", "Loewdin", "Hirshfeld"
    charges: Mapping[int, float]  # Atom index (0-based) to charge

    def __repr__(self) -> str:
        n_charges = len(self.charges)
        charge_summary = "{...}"  # Avoid printing all charges by default
        if n_charges < 6:  # Show charges if few atoms
            charge_summary = str(self.charges)
        return f"{type(self).__name__}(method='{self.method}', n_atoms={n_charges}, charges={charge_summary})"


@dataclass(frozen=True)
class DipoleMomentData:
    """Represents the molecular dipole moment from Q-Chem."""

    x_debye: float
    y_debye: float
    z_debye: float
    total_debye: float


@dataclass(frozen=True)
class DispersionCorrectionData:
    """Holds details of the empirical dispersion correction (e.g., D3)."""

    method: str  # e.g., "D3(0)", "D3(BJ)"
    energy_eh: float


@dataclass(frozen=True)
class CalculationMetadata:
    """Holds metadata about the Q-Chem calculation."""

    qchem_version: str | None = None
    host: str | None = None
    run_date: str | None = None  # Keep as string for now
    calculation_method: str | None = None
    basis_set: str | None = None
    # Add other relevant metadata: SCF type, symmetry, etc.


# Using a mutable class during parsing simplifies updates
@dataclass
class _MutableCalculationData:
    """Mutable version of CalculationData used internally during Q-Chem parsing."""

    raw_output: str
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"] = "UNKNOWN"
    # Removed: metadata: CalculationMetadata = field(default_factory=CalculationMetadata)
    # Add individual metadata fields
    qchem_version: str | None = None
    host: str | None = None
    run_date: str | None = None
    # Add rem block data
    rem: dict[str, str | bool | int | float] = field(default_factory=dict)  # Added rem field
    calculation_method: str | None = None
    basis_set: str | None = None

    input_geometry: Sequence[Atom] | None = None  # From $molecule block
    standard_orientation_geometry: Sequence[Atom] | None = None  # From 'Standard Nuclear Orientation'
    final_energy_eh: float | None = None  # Typically the 'Total energy' after SCF
    nuclear_repulsion_eh: float | None = None
    scf: ScfData | None = None
    orbitals: OrbitalData | None = None
    atomic_charges: list[AtomicCharges] = field(default_factory=list)
    dipole_moment: DipoleMomentData | None = None
    dispersion_correction: DispersionCorrectionData | None = None
    # Track errors/warnings during parsing
    parsing_errors: list[str] = field(default_factory=list)
    parsing_warnings: list[str] = field(default_factory=list)
    # Flags to track if sections have been parsed (for sections expected once)
    parsed_input_geometry: bool = False
    parsed_standard_geometry: bool = False
    parsed_scf: bool = False
    parsed_orbitals: bool = False
    parsed_mulliken_charges: bool = False  # Be specific if needed
    parsed_dipole: bool = False
    parsed_dispersion: bool = False
    # Add flags for metadata components
    parsed_meta_version: bool = False
    parsed_meta_host: bool = False
    parsed_meta_run_date: bool = False
    parsed_meta_method: bool = False
    parsed_meta_basis: bool = False
    # Add more flags as needed


@dataclass(frozen=True)
class CalculationData:
    """Immutable top-level container for parsed Q-Chem calculation results."""

    raw_output: str = field(repr=False)
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"]
    metadata: CalculationMetadata
    input_geometry: Sequence[Atom] | None = None
    standard_orientation_geometry: Sequence[Atom] | None = None
    final_energy_eh: float | None = None
    nuclear_repulsion_eh: float | None = None
    scf: ScfData | None = None
    orbitals: OrbitalData | None = None
    atomic_charges: list[AtomicCharges] = field(default_factory=list)
    dipole_moment: DipoleMomentData | None = None
    dispersion_correction: DispersionCorrectionData | None = None

    @classmethod
    def from_mutable(cls, mutable_data: _MutableCalculationData) -> "CalculationData":
        """Creates an immutable CalculationData from the mutable version."""
        # Construct metadata from individual fields
        metadata = CalculationMetadata(
            qchem_version=mutable_data.qchem_version,
            host=mutable_data.host,
            run_date=mutable_data.run_date,
            calculation_method=mutable_data.calculation_method,
            basis_set=mutable_data.basis_set,
        )
        return cls(
            raw_output=mutable_data.raw_output,
            termination_status=mutable_data.termination_status,
            metadata=metadata,  # Use the constructed metadata object
            input_geometry=mutable_data.input_geometry,
            standard_orientation_geometry=mutable_data.standard_orientation_geometry,
            final_energy_eh=mutable_data.final_energy_eh,
            nuclear_repulsion_eh=mutable_data.nuclear_repulsion_eh,
            scf=mutable_data.scf,
            orbitals=mutable_data.orbitals,
            atomic_charges=list(mutable_data.atomic_charges),  # Ensure list copy
            dipole_moment=mutable_data.dipole_moment,
            dispersion_correction=mutable_data.dispersion_correction,
        )

    # Add __repr__ or __str__ if desired for concise representation


# --- Parsing Logic Abstractions --- #

type LineIterator = Iterator[str]


class SectionParser(Protocol):
    """Protocol for Q-Chem section parsers."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line triggers this parser."""
        ...

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parse the section, consuming lines from the iterator and updating results."""
        ...

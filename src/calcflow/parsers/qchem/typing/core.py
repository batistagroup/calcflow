from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from calcflow.parsers.qchem.typing.orbitals import OrbitalsSet
from calcflow.parsers.qchem.typing.properties import (
    AtomicCharges,
    DispersionCorrectionData,
    MultipoleResults,
)
from calcflow.parsers.qchem.typing.scf import ScfResults, SmdResults
from calcflow.parsers.qchem.typing.tddft import (
    ExcitedStateDetailedAnalysis,
    ExcitedStateProperties,
    GroundStateReferenceAnalysis,
    NTOStateAnalysis,
    TddftData,
    TransitionDensityMatrixDetailedAnalysis,
)


@dataclass(frozen=True)
class Atom:
    """Represents an atom in the molecular geometry."""

    symbol: str
    x: float  # Angstrom
    y: float  # Angstrom
    z: float  # Angstrom


# --- Main Data Structures --- #
LineIterator = Iterator[str]  # Type alias for line iterators


# --- Mutable Data Structure (used during parsing) --- #
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
    # SMD specific fields for mutable data
    solvent_method: str | None = None
    solvent_name: str | None = None
    smd_g_pcm_kcal_mol: float | None = None
    smd_g_cds_kcal_mol: float | None = None
    smd_g_enp_au: float | None = None
    smd_g_tot_au: float | None = None

    input_geometry: Sequence[Atom] | None = None  # From $molecule block
    standard_orientation_geometry: Sequence[Atom] | None = None  # From 'Standard Nuclear Orientation'
    final_energy: float | None = None  # Typically the 'Total energy' after SCF
    nuclear_repulsion: float | None = None
    scf: ScfResults | None = None
    orbitals: OrbitalsSet | None = None
    atomic_charges: list[AtomicCharges] = field(default_factory=list)
    multipole: MultipoleResults | None = None
    dispersion_correction: DispersionCorrectionData | None = None
    smd_data: SmdResults | None = None  # Add SmdResults field
    # Track errors/warnings during parsing
    parsing_errors: list[str] = field(default_factory=list)
    parsing_warnings: list[str] = field(default_factory=list)
    # Flags to track if sections have been parsed (for sections expected once)
    parsed_input_geometry: bool = False
    parsed_standard_geometry: bool = False
    parsed_scf: bool = False
    parsed_orbitals: bool = False
    parsed_mulliken_charges: bool = False  # Be specific if needed
    parsed_multipole: bool = False
    parsed_dispersion: bool = False
    # Add flags for metadata components
    parsed_meta_version: bool = False
    parsed_meta_host: bool = False
    parsed_meta_run_date: bool = False
    parsed_meta_method: bool = False
    parsed_meta_basis: bool = False
    # Add more flags as needed
    parsed_unrelaxed_excited_state_properties: bool = False
    parsed_tddft_transition_dm_analysis: bool = False  # Flag for Transition Density Matrix Analysis

    # TDDFT data
    tda_excited_states_list: list[ExcitedStateProperties] = field(default_factory=list)
    tddft_excited_states_list: list[ExcitedStateProperties] = field(default_factory=list)
    excited_state_detailed_analyses_list: list[ExcitedStateDetailedAnalysis] = field(default_factory=list)
    transition_density_matrix_detailed_analyses_list: list[TransitionDensityMatrixDetailedAnalysis] = field(
        default_factory=list
    )
    nto_state_analyses_list: list[NTOStateAnalysis] = field(
        default_factory=list
    )  # Renamed from nto_decompositions_list

    # Flags for TDDFT sections
    parsed_tda_excitations: bool = False
    parsed_tddft_excitations: bool = False
    parsed_sa_nto_decomposition: bool = False  # Added flag for SA-NTO Decomposition
    # Flags for analysis sections are more complex as they are per-state
    # We might manage this state within the respective parsers or add more granular flags if needed.

    # Ground state reference data from within Excited State Analysis block
    ground_state_reference_analysis: GroundStateReferenceAnalysis | None = None

    # Buffer for a line that was read ahead by a parser and needs to be re-processed by the main loop
    buffered_line: str | None = None


@dataclass(frozen=True)
class CalculationMetadata:
    """Holds metadata about the Q-Chem calculation."""

    qchem_version: str | None = None
    host: str | None = None
    run_date: str | None = None  # Keep as string for now
    calculation_method: str | None = None
    basis_set: str | None = None
    solvent_method: str | None = None
    solvent_name: str | None = None


@dataclass(frozen=True)
class CalculationData:
    """Immutable top-level container for parsed Q-Chem calculation results."""

    raw_output: str = field(repr=False)
    termination_status: Literal["NORMAL", "ERROR", "UNKNOWN"]
    metadata: CalculationMetadata
    input_geometry: Sequence[Atom] | None = None
    standard_orientation_geometry: Sequence[Atom] | None = None
    final_energy: float | None = None
    nuclear_repulsion: float | None = None
    scf: ScfResults | None = None
    orbitals: OrbitalsSet | None = None
    atomic_charges: list[AtomicCharges] = field(default_factory=list)
    multipole: MultipoleResults | None = None
    dispersion_correction: DispersionCorrectionData | None = None
    smd_data: SmdResults | None = None  # Add SmdResults field
    tddft_data: TddftData | None = None  # Added TDDFT data container
    ground_state_reference_analysis: GroundStateReferenceAnalysis | None = None  # GS Ref from ESA block

    def __repr__(self) -> str:
        # Basic representation, can be expanded
        return (
            f"{type(self).__name__}(method='{self.metadata.calculation_method}', "
            f"basis='{self.metadata.basis_set}', status='{self.termination_status}')"
        )

    @classmethod
    def from_mutable(cls, mutable_data: _MutableCalculationData) -> "CalculationData":
        """Converts mutable parsing data to immutable CalculationData."""
        metadata = CalculationMetadata(
            qchem_version=mutable_data.qchem_version,
            host=mutable_data.host,
            run_date=mutable_data.run_date,
            calculation_method=mutable_data.calculation_method,
            basis_set=mutable_data.basis_set,
            solvent_method=mutable_data.solvent_method,
            solvent_name=mutable_data.solvent_name,
        )

        smd_data_instance: SmdResults | None = None
        if (
            mutable_data.smd_g_pcm_kcal_mol is not None
            or mutable_data.smd_g_cds_kcal_mol is not None
            or mutable_data.smd_g_enp_au is not None
            or mutable_data.smd_g_tot_au is not None
        ):
            smd_data_instance = SmdResults(
                g_pcm_kcal_mol=mutable_data.smd_g_pcm_kcal_mol,
                g_cds_kcal_mol=mutable_data.smd_g_cds_kcal_mol,
                g_enp_au=mutable_data.smd_g_enp_au,
                g_tot_au=mutable_data.smd_g_tot_au,
            )

        tddft_data_instance: TddftData | None = None
        if (
            mutable_data.tda_excited_states_list
            or mutable_data.tddft_excited_states_list
            or mutable_data.excited_state_detailed_analyses_list
            or mutable_data.transition_density_matrix_detailed_analyses_list
            or mutable_data.nto_state_analyses_list  # Renamed from nto_decompositions_list
        ):
            tddft_data_instance = TddftData(
                tda_excited_states=list(mutable_data.tda_excited_states_list)
                if mutable_data.tda_excited_states_list
                else None,
                tddft_excited_states=list(mutable_data.tddft_excited_states_list)
                if mutable_data.tddft_excited_states_list
                else None,
                excited_state_analyses=list(mutable_data.excited_state_detailed_analyses_list)
                if mutable_data.excited_state_detailed_analyses_list
                else None,
                transition_density_matrix_analyses=list(mutable_data.transition_density_matrix_detailed_analyses_list)
                if mutable_data.transition_density_matrix_detailed_analyses_list
                else None,
                nto_state_analyses=list(mutable_data.nto_state_analyses_list)  # Renamed from nto_decompositions_list
                if mutable_data.nto_state_analyses_list  # Renamed from nto_decompositions_list
                else None,
            )

        gs_ref_analysis_instance: GroundStateReferenceAnalysis | None = mutable_data.ground_state_reference_analysis

        return cls(
            raw_output=mutable_data.raw_output,
            termination_status=mutable_data.termination_status,
            metadata=metadata,  # Use the constructed metadata object
            input_geometry=mutable_data.input_geometry,
            standard_orientation_geometry=mutable_data.standard_orientation_geometry,
            final_energy=mutable_data.final_energy,
            nuclear_repulsion=mutable_data.nuclear_repulsion,
            scf=mutable_data.scf,
            orbitals=mutable_data.orbitals,
            atomic_charges=list(mutable_data.atomic_charges),  # Ensure list copy
            multipole=mutable_data.multipole,
            dispersion_correction=mutable_data.dispersion_correction,
            smd_data=smd_data_instance,  # Assign the new SmdResults instance
            tddft_data=tddft_data_instance,  # Assign the new TddftData instance
            ground_state_reference_analysis=gs_ref_analysis_instance,  # Assign GS Ref
        )

    # Add __repr__ or __str__ if desired for concise representation


class SectionParser(Protocol):
    """Protocol for Q-Chem section parsers."""

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line triggers this parser."""
        ...

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """Parse the section, consuming lines from the iterator and updating results."""
        ...

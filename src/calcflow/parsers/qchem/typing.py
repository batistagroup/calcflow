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
class QuadrupoleMoments:
    """Stores Cartesian quadrupole moments (Debye-Ang)."""

    xx: float
    xy: float
    yy: float
    xz: float
    yz: float
    zz: float


@dataclass(frozen=True)
class OctopoleMoments:
    """Stores Cartesian octopole moments (Debye-Ang^2)."""

    xxx: float
    xxy: float
    xyy: float
    yyy: float
    xxz: float
    xyz: float
    yyz: float
    xzz: float
    yzz: float
    zzz: float


@dataclass(frozen=True)
class HexadecapoleMoments:
    """Stores Cartesian hexadecapole moments (Debye-Ang^3)."""

    xxxx: float
    xxxy: float
    xxyy: float
    xyyy: float
    yyyy: float
    xxxz: float
    xxyz: float
    xyyz: float
    yyyz: float
    xxzz: float
    xyzz: float
    yyzz: float
    xzzz: float
    yzzz: float
    zzzz: float


@dataclass(frozen=True)
class MultipoleData:
    """Container for various electric multipole moments."""

    charge_esu: float | None = None
    dipole: DipoleMomentData | None = None
    quadrupole: QuadrupoleMoments | None = None
    octopole: OctopoleMoments | None = None
    hexadecapole: HexadecapoleMoments | None = None

    def __repr__(self) -> str:
        parts = []
        if self.charge_esu is not None:
            parts.append(f"charge_esu={self.charge_esu:.4f}")
        if self.dipole:
            # Use the dipole's own repr for detail
            parts.append(f"dipole={self.dipole!r}")
        if self.quadrupole:
            parts.append("quadrupole=QuadrupoleMoments(...)")
        if self.octopole:
            parts.append("octopole=OctopoleMoments(...)")
        if self.hexadecapole:
            parts.append("hexadecapole=HexadecapoleMoments(...)")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        """Return a concise representation showing presence of moments."""
        summary = []
        if self.charge_esu is not None:
            summary.append(f"Charge={self.charge_esu:.4f}")
        if self.dipole:
            summary.append(f"Dipole={self.dipole.total_debye:.4f} D")
        has_higher_moments = self.quadrupole or self.octopole or self.hexadecapole
        if has_higher_moments:
            summary.append("Higher moments present")
        if not summary:
            return f"{type(self).__name__}(Empty)"
        return f"{type(self).__name__}({', '.join(summary)})"


@dataclass(frozen=True)
class DispersionCorrectionData:
    """Holds details of the empirical dispersion correction (e.g., D3)."""

    method: str  # e.g., "D3(0)", "D3(BJ)"
    energy_eh: float
    basis_set: str | None = None
    # Add other relevant metadata: SCF type, symmetry, etc.


@dataclass(frozen=True)
class SmdData:
    """Holds results specific to the SMD solvation model."""

    g_pcm_kcal_mol: float | None = None  # Polarization energy component
    g_cds_kcal_mol: float | None = None  # Non-electrostatic (CDS) energy component
    g_enp_au: float | None = None  # E_SCF including G_PCM (SCF energy in solvent)
    g_tot_au: float | None = None  # Total free energy in solution (G_ENP + G_CDS)

    def __str__(self) -> str:
        parts = []
        if self.g_enp_au is not None:
            parts.append(f"G_ENP={self.g_enp_au:.8f} au")
        if self.g_tot_au is not None:
            parts.append(f"G(tot)={self.g_tot_au:.8f} au")
        if self.g_pcm_kcal_mol is not None:
            parts.append(f"G_PCM={self.g_pcm_kcal_mol:.4f} kcal/mol")
        if self.g_cds_kcal_mol is not None:
            parts.append(f"G_CDS={self.g_cds_kcal_mol:.4f} kcal/mol")
        if not parts:
            return f"{type(self).__name__}(Empty)"
        return f"{type(self).__name__}({', '.join(parts)})"


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


# --- TDDFT Specific Data Structures --- #


@dataclass(frozen=True)
class OrbitalTransition:
    """Represents a single orbital transition contributing to an excited state."""

    from_orbital_type: Literal["D", "V", "Unknown"]  # Donor or Virtual
    from_orbital_index: int  # 1-indexed from Q-Chem output
    to_orbital_type: Literal["D", "V", "Unknown"]
    to_orbital_index: int  # 1-indexed
    amplitude: float
    is_alpha_spin: bool | None = None  # True for alpha, False for beta, None if not specified

    def __repr__(self) -> str:
        spin = ""
        if self.is_alpha_spin is not None:
            spin = " (alpha)" if self.is_alpha_spin else " (beta)"
        return (
            f"{self.from_orbital_type}({self.from_orbital_index}) -> "
            f"{self.to_orbital_type}({self.to_orbital_index}) "
            f"(amp={self.amplitude:.4f}){spin}"
        )


@dataclass(frozen=True)
class ExcitedStateProperties:
    """Basic properties of an excited state from TDDFT/TDA output."""

    state_number: int
    excitation_energy_ev: float
    total_energy_au: float
    multiplicity: str  # e.g., "Singlet", "Triplet"
    trans_moment_x: float | None = None  # Transition dipole moment X (Debye or a.u. - check QChem)
    trans_moment_y: float | None = None
    trans_moment_z: float | None = None
    oscillator_strength: float | None = None  # Unitless
    transitions: Sequence[OrbitalTransition] = field(default_factory=list)

    def __repr__(self) -> str:
        parts = [
            f"state_number={self.state_number}",
            f"excitation_energy_ev={self.excitation_energy_ev:.4f} eV",
            f"total_energy_au={self.total_energy_au:.6f} au",
            f"multiplicity='{self.multiplicity}'",
        ]
        if self.oscillator_strength is not None:
            parts.append(f"oscillator_strength={self.oscillator_strength:.4f}")

        if self.trans_moment_x is not None and self.trans_moment_y is not None and self.trans_moment_z is not None:
            parts.append(
                f"trans_moment=[{self.trans_moment_x:.2f}, {self.trans_moment_y:.2f}, {self.trans_moment_z:.2f}]"
            )

        parts.append(f"transitions_count={len(self.transitions)}")

        body = ",\n    ".join(parts)
        return f"{type(self).__name__}(\n    {body}\n)"


@dataclass(frozen=True)
class ExcitedStateNOData:
    """Natural Orbital (NO) analysis for an excited state."""

    frontier_occupations: Sequence[float] | None = None  # e.g., [0.9992, 1.0006]
    num_electrons: float | None = None
    num_unpaired_electrons_nu: float | None = None  # n_u
    num_unpaired_electrons_nunl: float | None = None  # n_u,nl
    participation_ratio_pr_no: float | None = None


@dataclass(frozen=True)
class ExcitedStateAtomPopulation:
    """Mulliken or other population analysis for a single atom in an excited state."""

    atom_index: int  # 0-indexed, maps to original geometry
    symbol: str  # Atom symbol
    charge_e: float  # Net charge on the atom in the excited state
    hole_charge: float | None = None  # Contribution from hole (h+)
    electron_charge: float | None = None  # Contribution from electron (e-)
    delta_charge: float | None = None  # Change in charge (Del q)


@dataclass(frozen=True)
class ExcitedStateMulliken:
    """Mulliken population analysis for an excited state (State/Difference DM)."""

    populations: Sequence[ExcitedStateAtomPopulation] = field(default_factory=list)


@dataclass(frozen=True)
class ExcitedStateMultipole:
    """Multipole moment analysis for an excited state's density matrix."""

    molecular_charge: float | None = None
    num_electrons: float | None = None
    center_electronic_charge_ang: tuple[float, float, float] | None = None
    center_nuclear_charge_ang: tuple[float, float, float] | None = None
    dipole_moment_debye: DipoleMomentData | None = None  # Reusing existing DipoleMomentData
    rms_density_size_ang: tuple[float, float, float] | None = None  # Cartesian components


@dataclass(frozen=True)
class ExcitedStateExcitonDifferenceDM:
    """Exciton analysis of an excited state's difference density matrix."""

    hole_center_ang: tuple[float, float, float] | None = None  # <r_h>
    electron_center_ang: tuple[float, float, float] | None = None  # <r_e>
    electron_hole_separation_ang: float | None = None  # |<r_e - r_h>|
    hole_size_ang: float | None = None
    hole_size_ang_comps: tuple[float, float, float] | None = None  # Cartesian components
    electron_size_ang: float | None = None
    electron_size_ang_comps: tuple[float, float, float] | None = None  # Cartesian components


@dataclass(frozen=True)
class TransitionDMAtomPopulation:
    """Mulliken population analysis for a single atom in the transition density matrix."""

    atom_index: int
    symbol: str
    transition_charge_e: float  # Trans. (e)
    hole_charge: float | None = None  # h+
    electron_charge: float | None = None  # e-
    delta_charge: float | None = None  # Del q


@dataclass(frozen=True)
class TransitionDMMulliken:
    """Mulliken population analysis for the transition density matrix."""

    populations: Sequence[TransitionDMAtomPopulation] = field(default_factory=list)
    sum_abs_trans_charges_qta: float | None = None
    sum_sq_trans_charges_qt2: float | None = None


@dataclass(frozen=True)
class TransitionDMCTNumbers:
    """Charge Transfer (CT) numbers from Mulliken analysis of transition DM."""

    omega: float | None = None
    two_alpha_beta_overlap: float | None = None  # 2<alpha|beta>
    loc: float | None = None
    loc_a: float | None = None  # LOCa
    phe_overlap: float | None = None  # <Phe>


@dataclass(frozen=True)
class ExcitonAnalysisTransitionDM:
    """Exciton analysis of the transition density matrix, matching parser output."""

    total_transition_dipole_moment: float | None = None
    transition_dipole_moment_components: tuple[float, float, float] | None = None
    transition_r_squared_au: float | None = None
    transition_r_squared_au_components: tuple[float, float, float] | None = None
    hole_position_ang: tuple[float, float, float] | None = None
    electron_position_ang: tuple[float, float, float] | None = None
    hole_electron_distance_ang: float | None = None
    hole_size_ang: float | None = None
    hole_size_ang_components: tuple[float, float, float] | None = None
    electron_size_ang: float | None = None
    electron_size_ang_components: tuple[float, float, float] | None = None
    rms_electron_hole_separation_ang: float | None = None
    rms_electron_hole_separation_ang_components: tuple[float, float, float] | None = None
    covariance_rh_re_ang_sq: float | None = None
    correlation_coefficient: float | None = None
    center_of_mass_size_ang: float | None = None
    center_of_mass_size_ang_components: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class NTOContribution:
    """A single Natural Transition Orbital (NTO) contribution."""

    hole_reference: Literal["H", "V"]  # always 'H' for HOMO‐offset or 'V' if you ever see V- type
    hole_offset: int  # signed offset, e.g. -2 for H-2
    electron_reference: Literal["L", "V"]  # 'L' for LUMO‐offset
    electron_offset: int  # e.g. +3 for L+3
    coefficient: float
    weight_percent: float  # e.g., 99.9 for 99.9%


@dataclass(frozen=True)
class NTOStateAnalysis:  # Renamed from NTOData
    """Natural Transition Orbital (NTO) decomposition for a single excited state."""

    state_number: int
    multiplicity: str  # Added for consistency with other state-specific analyses
    contributions: Sequence[NTOContribution] = field(default_factory=list)
    omega_percent: float | None = None  # Overall omega for the decomposition


@dataclass(frozen=True)
class ExcitedStateDetailedAnalysis:
    """Comprehensive analysis for a single excited state."""

    state_number: int
    multiplicity: str  # e.g., "Singlet"
    no_data: ExcitedStateNOData | None = None
    mulliken_analysis: ExcitedStateMulliken | None = None
    multipole_analysis: ExcitedStateMultipole | None = None
    exciton_difference_dm_analysis: ExcitedStateExcitonDifferenceDM | None = None


@dataclass(frozen=True)
class GroundStateReferenceAnalysis:
    """Analysis data for the ground state reference within TDDFT output."""

    no_data: ExcitedStateNOData | None = None
    mulliken_analysis: ExcitedStateMulliken | None = None
    multipole_analysis: ExcitedStateMultipole | None = None


@dataclass(frozen=True)
class TransitionDensityMatrixDetailedAnalysis:
    """Comprehensive analysis of the transition density matrix for a single excited state."""

    state_number: int
    multiplicity: str  # e.g., "Singlet"
    mulliken_analysis: TransitionDMMulliken | None = None
    ct_numbers: TransitionDMCTNumbers | None = None
    exciton_analysis: ExcitonAnalysisTransitionDM | None = None


@dataclass(frozen=True)
class TddftData:
    """Container for all TDDFT related parsed data."""

    # QChem output can have both TDA and full TDDFT results
    tda_excited_states: Sequence[ExcitedStateProperties] | None = None
    tddft_excited_states: Sequence[ExcitedStateProperties] | None = None

    # Detailed analysis sections, usually correspond to tddft_excited_states
    excited_state_analyses: Sequence[ExcitedStateDetailedAnalysis] | None = None
    transition_density_matrix_analyses: Sequence[TransitionDensityMatrixDetailedAnalysis] | None = None
    nto_state_analyses: Sequence[NTOStateAnalysis] | None = None  # Renamed from nto_decompositions: Sequence[NTOData]

    def __repr__(self) -> str:
        parts = []
        if self.tda_excited_states:
            parts.append(f"tda_excited_states={len(self.tda_excited_states)} states")
        else:
            parts.append("tda_excited_states=None")

        if self.tddft_excited_states:
            parts.append(f"tddft_excited_states={len(self.tddft_excited_states)} states")
        else:
            parts.append("tddft_excited_states=None")

        if self.excited_state_analyses:
            parts.append(f"excited_state_analyses={len(self.excited_state_analyses)} analyses")
        else:
            parts.append("excited_state_analyses=None")

        if self.transition_density_matrix_analyses:
            parts.append(f"transition_density_matrix_analyses={len(self.transition_density_matrix_analyses)} analyses")
        else:
            parts.append("transition_density_matrix_analyses=None")

        if self.nto_state_analyses:
            parts.append(f"nto_state_analyses={len(self.nto_state_analyses)} analyses")
        else:
            parts.append("nto_state_analyses=None")

        return f"{type(self).__name__}({', '.join(parts)})"


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
    final_energy_eh: float | None = None  # Typically the 'Total energy' after SCF
    nuclear_repulsion_eh: float | None = None
    scf: ScfData | None = None
    orbitals: OrbitalData | None = None
    atomic_charges: list[AtomicCharges] = field(default_factory=list)
    multipole: MultipoleData | None = None
    dispersion_correction: DispersionCorrectionData | None = None
    smd_data: SmdData | None = None  # Add SmdData field
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
    multipole: MultipoleData | None = None
    dispersion_correction: DispersionCorrectionData | None = None
    smd_data: SmdData | None = None  # Add SmdData field
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

        smd_data_instance: SmdData | None = None
        if (
            mutable_data.smd_g_pcm_kcal_mol is not None
            or mutable_data.smd_g_cds_kcal_mol is not None
            or mutable_data.smd_g_enp_au is not None
            or mutable_data.smd_g_tot_au is not None
        ):
            smd_data_instance = SmdData(
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
            final_energy_eh=mutable_data.final_energy_eh,
            nuclear_repulsion_eh=mutable_data.nuclear_repulsion_eh,
            scf=mutable_data.scf,
            orbitals=mutable_data.orbitals,
            atomic_charges=list(mutable_data.atomic_charges),  # Ensure list copy
            multipole=mutable_data.multipole,
            dispersion_correction=mutable_data.dispersion_correction,
            smd_data=smd_data_instance,  # Assign the new SmdData instance
            tddft_data=tddft_data_instance,  # Assign the new TddftData instance
            ground_state_reference_analysis=gs_ref_analysis_instance,  # Assign GS Ref
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

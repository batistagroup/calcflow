from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from calcflow.parsers.qchem.typing.properties import DipoleMoment


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
    dipole_moment_debye: DipoleMoment | None = None  # Reusing existing DipoleMoment
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

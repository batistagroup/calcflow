from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Orbital:
    """Represents a single molecular orbital in Q-Chem."""

    index: int  # 0-based index
    energy: float  # in Ha
    # TODO: we need to infer the occupation from the Q-Chem output
    occ: float | None = None


@dataclass(frozen=True)
class OrbitalsSet:
    """Holds information about molecular orbitals from Q-Chem."""

    # Q-Chem separates Alpha and Beta orbitals
    alpha_orbitals: Sequence[Orbital] | None = None
    beta_orbitals: Sequence[Orbital] | None = None
    # HOMO/LUMO indices might need calculation based on occupations
    alpha_homo_idx: int | None = None
    alpha_lumo_idx: int | None = None
    beta_homo_idx: int | None = None
    beta_lumo_idx: int | None = None

    def __repr__(self) -> str:
        n_alpha = len(self.alpha_orbitals) if self.alpha_orbitals else 0
        n_beta = len(self.beta_orbitals) if self.beta_orbitals else 0
        # Add HOMO/LUMO details if available later
        return f"{type(self).__name__}(n_alpha_orbitals={n_alpha}, n_beta_orbitals={n_beta})"

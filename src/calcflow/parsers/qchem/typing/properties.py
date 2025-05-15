from collections.abc import Mapping
from dataclasses import dataclass


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
class DipoleMoment:
    """Stores dipole moment and components in Debye."""

    x: float
    y: float
    z: float
    magnitude: float


@dataclass(frozen=True)
class QuadrupoleMoment:
    """Stores Cartesian quadrupole moments (Debye-Ang)."""

    xx: float
    xy: float
    yy: float
    xz: float
    yz: float
    zz: float


@dataclass(frozen=True)
class OctopoleMoment:
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
class HexadecapoleMoment:
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
class MultipoleResults:
    """Container for various electric multipole moments."""

    charge_esu: float | None = None
    dipole: DipoleMoment | None = None
    quadrupole: QuadrupoleMoment | None = None
    octopole: OctopoleMoment | None = None
    hexadecapole: HexadecapoleMoment | None = None

    def __repr__(self) -> str:
        parts = []
        if self.charge_esu is not None:
            parts.append(f"charge_esu={self.charge_esu:.4f}")
        if self.dipole:
            # Use the dipole's own repr for detail
            parts.append(f"dipole={self.dipole!r}")
        if self.quadrupole:
            parts.append("quadrupole=QuadrupoleMoment(...)")
        if self.octopole:
            parts.append("octopole=OctopoleMoment(...)")
        if self.hexadecapole:
            parts.append("hexadecapole=HexadecapoleMoment(...)")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        """Return a concise representation showing presence of moments."""
        summary = []
        if self.charge_esu is not None:
            summary.append(f"Charge={self.charge_esu:.4f}")
        if self.dipole:
            summary.append(f"Dipole={self.dipole.magnitude:.4f} D")
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
    energy: float
    basis_set: str | None = None
    # Add other relevant metadata: SCF type, symmetry, etc.

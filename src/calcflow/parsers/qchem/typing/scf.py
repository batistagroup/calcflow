from collections.abc import Sequence
from dataclasses import dataclass
from pprint import pformat


@dataclass(frozen=True)
class ScfIteration:
    """Holds data for a single Q-Chem SCF iteration."""

    iteration: int
    energy: float
    diis_error: float  # DIIS error magnitude
    mom_active: bool | None = None
    mom_method_type: str | None = None
    mom_overlap_current: float | None = None
    mom_overlap_target: float | None = None


@dataclass(frozen=True)
class ScfEnergyComponents:
    """Holds the components of the raw SCF energy (adapt as needed for Q-Chem)."""

    nuclear_repulsion: float | None = None


@dataclass(frozen=True)
class ScfResults:
    """Holds results specific to the Q-Chem SCF calculation step."""

    converged: bool
    energy: float  # Converged SCF energy (or last energy if not converged)
    # components: ScfEnergyComponents | None = None # Optional for now
    n_iterations: int
    iterations: Sequence[ScfIteration]

    def __repr__(self) -> str:
        # Create a copy of the dict and convert iterations to strings
        dict_copy = self.__dict__.copy()
        dict_copy["iterations"] = [str(it) for it in self.iterations]
        return f"{self.__class__.__name__}(\n{pformat(dict_copy, indent=2)[1:-1]}\n)"

    def __str__(self) -> str:
        """Return a concise representation of the SCF data."""
        conv_status = "Converged" if self.converged else "Not Converged"
        energy_str = f"{self.energy:.8f}"
        return f"{type(self).__name__}(status='{conv_status}', energy={energy_str}, n_iterations={self.n_iterations})"


@dataclass(frozen=True)
class SmdResults:
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

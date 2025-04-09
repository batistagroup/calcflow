from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Literal, TypeVar

from calcflow.utils import logger

# Generic TypeVar for fluent methods in subclasses
T_CalcInput = TypeVar("T_CalcInput", bound="CalculationInput")


@dataclass(frozen=True)
class CalculationInput(ABC):
    """
    Abstract base class for defining quantum chemistry calculation inputs.

    Focuses on common chemical concepts rather than program-specific keywords.
    Subclasses are responsible for translating these concepts into the
    required format for a specific QM program (QChem, ORCA, etc.).

    Attributes:
        charge: The net charge of the molecule.
        spin_multiplicity: The spin multiplicity (2S+1).
        task: The primary calculation task (e.g., single point energy, geometry optimization).
        level_of_theory: The electronic structure method (e.g., "B3LYP", "MP2").
        basis_set: The basis set specification (string name or dictionary for general basis).
        memory_mb: Requested memory in megabytes.
        implicit_solvation_model: The implicit solvation model to use (e.g., "pcm", "smd").
        solvent: The solvent to use with the implicit solvation model.
    """

    charge: int
    spin_multiplicity: int
    task: Literal["energy", "geometry"]
    level_of_theory: str
    basis_set: str | dict[str, str]

    unrestricted: bool = False

    # --- Optional Common Concepts ---
    memory_mb: int = 2000  # Default memory in MB
    memory_per_core_mb: int = 2000  # Default memory per core in MB
    implicit_solvation_model: Literal["pcm", "smd", "cpcm"] | None = None
    solvent: str | None = None

    def __post_init__(self) -> None:
        """Basic validation common to all calculation types."""
        # Hard Errors
        if not isinstance(self.spin_multiplicity, int) or self.spin_multiplicity < 1:
            raise ValueError("Spin multiplicity must be a positive integer (e.g., 1 for singlet, 2 for doublet).")
        if not isinstance(self.charge, int):
            raise ValueError("Charge must be an integer.")
        if (self.implicit_solvation_model is not None) != (self.solvent is not None):
            raise ValueError("Both `implicit_solvation_model` and `solvent` must be provided together, or neither.")

        # Soft Warnings
        if self.memory_mb <= 512:
            logger.warning("Memory allocation seems low (< 512 MB), please specify in MB (e.g., 2000 for 2GB).")
        if self.memory_per_core_mb <= 256:
            logger.warning(
                "Memory per core allocation seems low (< 256 MB), please specify in MB (e.g., 2000 for 2GB)."
            )

        if self.spin_multiplicity > 1 and not self.unrestricted:
            logger.warning(
                "Spin multiplicity > 1 but you didn't choose an unrestricted calculation. Please abort if this was not intentional."
            )

        if self.spin_multiplicity == 1 and self.unrestricted:
            logger.warning(
                "Just so you know, you're running an unrestricted calculation with a singlet spin multiplicity."
            )

        # Subclasses should add specific validation in their __post_init__
        # and ideally call super().__post_init__()

    @abstractmethod
    def export_input_file(self, geom: str) -> str:
        """
        Generates the complete input file string for the specific QM program.

        Args:
            geom: A string containing the molecular geometry in XYZ or relevant format.
                  (Format details might be handled by subclass or conventions).

        Returns:
            The formatted input file content as a string.
        """
        pass

    # --- Fluent API methods for common concepts ---

    def set_solvation(
        self: T_CalcInput, model: Literal["pcm", "smd", "cpcm"] | None, solvent: str | None
    ) -> T_CalcInput:
        """Returns a new instance with updated solvation parameters."""
        if (model is not None) != (solvent is not None):
            raise ValueError("Both `model` and `solvent` must be provided together, or neither.")
        # Basic normalization
        solvent_lower = solvent.lower() if solvent else None
        model_lower = model.lower() if model else None

        # Model-specific validation can go in subclass __post_init__ or here if universal
        allowed_models = {"pcm", "smd", "cpcm"}
        if model_lower is not None and model_lower not in allowed_models:
            raise ValueError(f"Solvation model '{model}' not recognized. Allowed: {allowed_models}")

        return replace(self, implicit_solvation_model=model_lower, solvent=solvent_lower)  # type: ignore

    def set_memory(self: T_CalcInput, memory_mb: int) -> T_CalcInput:
        """Returns a new instance with updated memory allocation (in MB)."""
        if memory_mb < 256:
            logger.warning("Memory allocation seems low (< 256 MB), please specify in MB.")
        return replace(self, memory_mb=memory_mb)

    def set_memory_per_core(self: T_CalcInput, memory_per_core_mb: int) -> T_CalcInput:
        """Returns a new instance with updated memory per core allocation (in MB)."""
        if memory_per_core_mb < 256:
            logger.warning("Memory per core allocation seems low (< 256 MB), please specify in MB.")
        return replace(self, memory_per_core_mb=memory_per_core_mb)

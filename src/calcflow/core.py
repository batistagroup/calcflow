from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Literal, TypeVar

from calcflow.utils import logger

# Generic TypeVar for fluent methods in subclasses
T_CalcInput = TypeVar("T_CalcInput", bound="CalculationInput")


@dataclass(frozen=True)
class CalculationInput(ABC):
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
        pass

    # --- Fluent API methods for common concepts ---

    def set_solvation(
        self: T_CalcInput, model: Literal["pcm", "smd", "cpcm"] | None, solvent: str | None
    ) -> T_CalcInput:
        if (model is not None) != (solvent is not None):
            raise ValueError("Both `model` and `solvent` must be provided together, or neither.")
        # Basic normalization
        solvent_lower = solvent.lower() if solvent else None
        model_lower = model.lower() if model else None

        allowed_models = {"pcm", "smd", "cpcm"}
        if model_lower is not None and model_lower not in allowed_models:
            raise ValueError(f"Solvation model '{model}' not recognized. Allowed: {allowed_models}")

        return replace(self, implicit_solvation_model=model_lower, solvent=solvent_lower)  # type: ignore

    def set_memory(self: T_CalcInput, memory_mb: int) -> T_CalcInput:
        if memory_mb < 256:
            logger.warning("Memory allocation seems low (< 256 MB), please specify in MB.")
        return replace(self, memory_mb=memory_mb)

    def set_memory_per_core(self: T_CalcInput, memory_per_core_mb: int) -> T_CalcInput:
        if memory_per_core_mb < 256:
            logger.warning("Memory per core allocation seems low (< 256 MB), please specify in MB.")
        return replace(self, memory_per_core_mb=memory_per_core_mb)

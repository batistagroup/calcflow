from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Literal, TypeVar

from calcflow.exceptions import ValidationError
from calcflow.utils import logger

# Generic TypeVar for fluent methods in subclasses
T_CalcInput = TypeVar("T_CalcInput", bound="CalculationInput")


@dataclass(frozen=True)
class CalculationInput(ABC):
    """
    Abstract base class for all calculation input classes.

    This class defines the common attributes and methods that all calculation input classes should have.
    It is designed to be subclassed by program-specific input classes (e.g., QChemInput, OrcaInput).

    Attributes:
        charge (int): Molecular charge.
        spin_multiplicity (int): Spin multiplicity (1 for singlet, 2 for doublet, etc.). Must be a positive integer.
        task (Literal["energy", "geometry"]): Type of calculation to perform.
        level_of_theory (str): Electronic structure method/functional (e.g., "B3LYP", "CCSD(T)").
        basis_set (str | dict[str, str]): Basis set to use. Can be a string for standard basis sets
            or a dictionary for program-specific basis set specifications.
        unrestricted (bool): Whether to use an unrestricted method (e.g., UB3LYP vs B3LYP). Defaults to False.
        memory_mb (int): Total memory to allocate for the calculation in MB. Defaults to 2000 MB.
        memory_per_core_mb (int): Memory to allocate per core in MB. Defaults to 2000 MB.
        implicit_solvation_model (Literal["pcm", "smd", "isosvp", "cpcm"] | None): Implicit solvation model to use (e.g., "pcm"). Defaults to None.
        solvent (str | None): Solvent to use for implicit solvation (e.g., "water"). Must be provided if `implicit_solvation_model` is specified. Defaults to None.
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

    def __post_init__(self) -> None:
        """
        Performs basic validation common to all calculation types after initialization.

        Raises:
            ValidationError: If spin_multiplicity is not a positive integer.
            ValidationError: If charge is not an integer.
            ValidationError: If `implicit_solvation_model` and `solvent` are not both provided or both None.

        Warns:
            UserWarning: If `memory_mb` is less than or equal to 512 MB.
            UserWarning: If `memory_per_core_mb` is less than or equal to 256 MB.
            UserWarning: If `spin_multiplicity` > 1 and `unrestricted` is False.
            UserWarning: If `spin_multiplicity` == 1 and `unrestricted` is True.
        """
        # Hard Errors
        if not isinstance(self.spin_multiplicity, int) or self.spin_multiplicity < 1:
            raise ValidationError("Spin multiplicity must be a positive integer (e.g., 1 for singlet, 2 for doublet).")
        if not isinstance(self.charge, int):
            raise ValidationError("Charge must be an integer.")

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
        Abstract method to export the input file content as a string.

        Subclasses must implement this method to generate the program-specific input file format.

        Args:
            geom (str): Molecular geometry in a program-agnostic format (e.g., XYZ string).

        Returns:
            str: Input file content as a string.
        """
        pass

    # --- Fluent API methods for common concepts ---

    def set_memory(self: T_CalcInput, memory_mb: int) -> T_CalcInput:
        """
        Set the total memory for the calculation.

        Args:
            memory_mb (int): Total memory to allocate in MB.

        Returns:
            T_CalcInput: A new instance of the CalculationInput subclass with the memory updated.

        Warns:
            UserWarning: If `memory_mb` is less than 256 MB.
        """
        if memory_mb < 256:
            logger.warning("Memory allocation seems low (< 256 MB), please specify in MB.")
        return replace(self, memory_mb=memory_mb)

    def set_memory_per_core(self: T_CalcInput, memory_per_core_mb: int) -> T_CalcInput:
        """
        Set the memory per core for the calculation.

        Args:
            memory_per_core_mb (int): Memory per core to allocate in MB.

        Returns:
            T_CalcInput: A new instance of the CalculationInput subclass with the memory per core updated.

        Warns:
            UserWarning: If `memory_per_core_mb` is less than 256 MB.
        """
        if memory_per_core_mb < 256:
            logger.warning("Memory per core allocation seems low (< 256 MB), please specify in MB.")
        return replace(self, memory_per_core_mb=memory_per_core_mb)

    def set_unrestricted(self: T_CalcInput) -> T_CalcInput:
        """Set whether the calculation should be unrestricted.

        Args:
            unrestricted (bool): True to use an unrestricted method, False otherwise.

        Returns:
            T_CalcInput: A new instance of the CalculationInput subclass with the unrestricted setting updated.
        """
        # The __post_init__ method will handle warnings related to spin multiplicity mismatch.
        logger.debug(f"Setting unrestricted to: {True}")
        return replace(self, unrestricted=True)

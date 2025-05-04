from dataclasses import dataclass, replace
from typing import ClassVar, Literal, TypeVar, cast, get_args

from calcflow.basis_sets import registry as basis_registry
from calcflow.core import CalculationInput
from calcflow.exceptions import InputGenerationError, NotSupportedError, ValidationError
from calcflow.geometry.static import Geometry
from calcflow.utils import logger

# Define allowed models specifically for Q-Chem
QCHEM_ALLOWED_SOLVATION_MODELS = Literal["pcm", "smd", "isosvp", "cpcm"]


T_QchemInput = TypeVar("T_QchemInput", bound="QchemInput")

# fmt:off
SUPPORTED_FUNCTIONALS = {"b3lyp", "pbe0", "m06", "cam-b3lyp", "wb97x", "wb97x-d3" }
# fmt:on


@dataclass(frozen=True)
class QchemInput(CalculationInput):
    """Input parameters specific to Q-Chem calculations.

    Inherits common parameters and validation logic from CalculationInput.
    Implements the export_input_file method to generate Q-Chem input files.

    Attributes:
        program: The program name, always "qchem".
        n_cores: Number of CPU cores to use (influences $omp block). Defaults to 1.
        geom_mode: Geometry input mode for Q-Chem. Defaults to "xyz".
        run_tddft: Flag to enable TDDFT calculation. Defaults to False.
        tddft_nroots: Number of roots to calculate in TDDFT. Required if run_tddft is True.
        tddft_singlets: Flag to include singlet states in TDDFT calculation. Defaults to True.
        tddft_triplets: Flag to include triplet states in TDDFT calculation. Defaults to False.
        tddft_state_analysis: Flag to enable state analysis. Defaults to True.
        rpa: Controls whether to use RPA (True TDDFT) instead of CIS/TDA.
    """

    program: ClassVar[str] = "qchem"

    n_cores: int = 1
    geom_mode: str = "xyz"  # Q-Chem doesn't use this directly in input, but consistent with base

    run_tddft: bool = False
    tddft_nroots: int | None = None
    tddft_singlets: bool = True
    tddft_triplets: bool = False
    tddft_state_analysis: bool = True
    rpa: bool = False  # Controls whether to use RPA (True TDDFT) instead of CIS/TDA

    implicit_solvation_model: QCHEM_ALLOWED_SOLVATION_MODELS | None = None
    solvent: str | None = None

    def __post_init__(self) -> None:
        """Performs Q-Chem-specific validation after initialization."""
        super().__post_init__()

        if self.task == "geometry" and self.basis_set == "def2-svp":
            logger.warning(
                "Using def2-svp for geometry optimization might yield less accurate results. "
                "Consider using def2-tzvp or larger basis sets for final geometries."
            )

        if self.n_cores < 1:
            raise ValidationError("Number of cores must be a positive integer.")

        if self.run_tddft:
            if self.tddft_nroots is None or self.tddft_nroots < 1:
                raise ValidationError("If run_tddft is True, tddft_nroots must be a positive integer.")
            if not self.tddft_singlets and not self.tddft_triplets:
                raise ValidationError(
                    "If run_tddft is True, at least one of tddft_singlets or tddft_triplets must be True."
                )

        # Use the Q-Chem specific literal for validation
        if self.implicit_solvation_model and self.implicit_solvation_model not in get_args(
            QCHEM_ALLOWED_SOLVATION_MODELS
        ):
            raise ValidationError(
                f"Solvation model '{self.implicit_solvation_model}' not recognized. Allowed: {get_args(QCHEM_ALLOWED_SOLVATION_MODELS)}"
            )

    def set_solvation(self: T_QchemInput, model: str | None, solvent: str | None) -> T_QchemInput:
        """
        Set the implicit solvation model and solvent.

        Both `model` and `solvent` must be provided together, or neither.

        Args:
            model (QCHEM_ALLOWED_SOLVATION_MODELS | None): Implicit solvation model to use (e.g., "pcm").
                If None, solvation is disabled.
            solvent (str | None): Solvent to use for implicit solvation (e.g., "water").
                If None, solvation is disabled.

        Returns:
            T_QchemInput: A new instance of the QchemInput subclass with the solvation settings updated.
        Raises:
            ValidationError: If `model` and `solvent` are not consistently provided (both or neither).
            ValidationError: If the provided `model` is not one of the allowed models ("pcm", "smd", "isosvp", "cpcm").
        """
        if (model is not None) != (solvent is not None):
            raise ValidationError("Both `model` and `solvent` must be provided together, or neither.")
        # Basic normalization
        solvent_lower = solvent.lower() if solvent else None
        model_lower = model.lower() if model else None

        # Validate against the Q-Chem specific Literal
        if model_lower is not None and model_lower not in get_args(QCHEM_ALLOWED_SOLVATION_MODELS):
            raise ValidationError(
                f"Solvation model '{model}' not recognized. Allowed: {get_args(QCHEM_ALLOWED_SOLVATION_MODELS)}"
            )
        casted = cast(QCHEM_ALLOWED_SOLVATION_MODELS, model_lower)

        return replace(self, implicit_solvation_model=casted, solvent=solvent_lower)  # type: ignore

    def set_tddft(
        self: T_QchemInput,
        nroots: int,
        singlets: bool = True,
        triplets: bool = False,
        state_analysis: bool = True,
    ) -> T_QchemInput:
        """Configure and enable TDDFT calculation.

        Args:
            nroots: Number of excited states to calculate. Must be positive.
            singlets: Whether to include singlet states. Defaults to True.
            triplets: Whether to include triplet states. Defaults to False.
            state_analysis: Whether to perform state analysis. Defaults to True.

        Returns:
            A new QchemInput instance with TDDFT enabled and configured.

        Raises:
            ValidationError: If input parameters are invalid (e.g., nroots <= 0).
        """
        if nroots <= 0:
            raise ValidationError("tddft_nroots must be a positive integer.")
        if not singlets and not triplets:
            raise ValidationError("At least one of singlets or triplets must be True for TDDFT.")

        logger.debug(
            f"Setting TDDFT: nroots={nroots}, singlets={singlets}, triplets={triplets}, state_analysis={state_analysis}"
        )
        return replace(
            self,
            run_tddft=True,
            tddft_nroots=nroots,
            tddft_singlets=singlets,
            tddft_triplets=triplets,
            tddft_state_analysis=state_analysis,
        )

    def set_rpa(self: T_QchemInput, enable: bool = True) -> T_QchemInput:
        """Enable or disable the RPA keyword for TDDFT calculations.

        Args:
            enable: Set to True to enable RPA, False to disable. Defaults to True.

        Returns:
            A new QchemInput instance with the updated RPA setting.
        """
        logger.debug(f"Setting RPA keyword to: {enable}")
        return replace(self, rpa=enable)

    def set_basis(self: T_QchemInput, basis: str | dict[str, str]) -> T_QchemInput:
        """Set the basis set for the calculation.

        Args:
            basis: The basis set specification. Can be a string (e.g., "def2-svp")
                   or a dictionary mapping element symbols to basis set names
                   (e.g., {"C": "6-31g*", "H": "sto-3g"}).

        Returns:
            A new QchemInput instance with the updated basis set.
        """
        # Add validation if specific formats are required or common errors checked
        logger.debug(f"Setting basis set to: {basis}")
        return replace(self, basis_set=basis)

    def _get_molecule_block(self, geom: str) -> str:
        """Generates the $molecule block.

        Args:
            geom: Molecular geometry in XYZ format (multiline string, excluding header lines).

        Returns:
            String containing the formatted $molecule block.
        """
        lines = [
            "$molecule",
            f"{self.charge} {self.spin_multiplicity}",
            geom.strip(),
            "$end",
        ]
        return "\n".join(lines)

    def _get_rem_block(self) -> str:
        """Generates the $rem block.

        Returns:
            String containing the formatted $rem block.

        Raises:
            ValidationError: If level_of_theory is invalid or unsupported.
            InputGenerationError: If basis_set type is unsupported.
        """
        rem_vars: dict[str, str | bool | int] = {}

        # --- Job Type --- #
        if self.task == "energy":
            rem_vars["JOBTYPE"] = "sp"
        elif self.task == "geometry":
            rem_vars["JOBTYPE"] = "opt"
        else:
            # Q-Chem supports other job types like freq, tss, etc.
            # Add handling here if needed in the future.
            raise NotSupportedError(f"Task type '{self.task}' not currently supported for Q-Chem export.")

        # --- Level of Theory --- #
        method = self.level_of_theory.lower()
        if method in SUPPORTED_FUNCTIONALS or method == "hf":
            rem_vars["METHOD"] = method
        else:
            # Q-Chem supports many methods. Add validation/mapping as needed.
            raise ValidationError(f"Unsupported or unrecognized level_of_theory for Q-Chem: '{self.level_of_theory}'")

        # --- Basis Set --- #
        if isinstance(self.basis_set, str):
            rem_vars["BASIS"] = self.basis_set
        elif isinstance(self.basis_set, dict):
            # Use a general basis set keyword when a dictionary is provided
            rem_vars["BASIS"] = "gen"
        else:
            raise InputGenerationError(f"Unsupported basis_set type: {type(self.basis_set)}")

        # --- Spin/Symmetry --- #
        rem_vars["UNRESTRICTED"] = self.unrestricted
        rem_vars["SYMMETRY"] = False  # Often disabled for robustness
        rem_vars["SYM_IGNORE"] = True

        # --- TDDFT --- #
        if self.run_tddft:
            if self.tddft_nroots is None:  # Should be caught by __post_init__, but defensive check
                raise InputGenerationError("tddft_nroots must be set when run_tddft is True.")
            rem_vars["CIS_N_ROOTS"] = self.tddft_nroots
            rem_vars["CIS_SINGLETS"] = self.tddft_singlets
            rem_vars["CIS_TRIPLETS"] = self.tddft_triplets
            rem_vars["STATE_ANALYSIS"] = self.tddft_state_analysis
            # Q-Chem uses CIS_N_ROOTS > 0 to trigger TDDFT/CIS calculations.
            # The METHOD keyword determines DFT vs HF-based excitation.
            if self.rpa:
                rem_vars["RPA"] = True

        # --- Implicit Solvation --- #
        # Signal solvation implicitly via $solvent or $smx block, not directly in $rem
        if self.implicit_solvation_model:
            if not self.solvent:
                # This case should be caught by set_solvation/post_init, but defensive check
                raise InputGenerationError(
                    f"Solvation model '{self.implicit_solvation_model}' requires a solvent to be specified."
                )
            rem_vars["SOLVENT_METHOD"] = self.implicit_solvation_model
            logger.warning(
                f"Setting SOLVENT_METHOD={self.implicit_solvation_model}. "
                f"Ensure any required parameters are set, possibly via a manual $solvent block."
            )

        # --- Formatting --- #
        lines = ["$rem"]
        max_key_len = max(len(k) for k in rem_vars) if rem_vars else 0
        for key, value in rem_vars.items():
            # Format boolean values as True/False for Q-Chem
            if isinstance(value, bool):
                value_str = str(value)
            else:
                value_str = str(value)
            lines.append(f"    {key:<{max_key_len}}    {value_str}")
        lines.append("$end")
        return "\n".join(lines)

    def _get_basis_block(self) -> str:
        """Generates the $basis block for mixed basis set definitions.

        Returns:
            String containing the formatted $basis block if self.basis_set is a dictionary,
            otherwise an empty string.
        """
        if not isinstance(self.basis_set, dict):
            return ""

        lines = ["$basis"]
        for element, basis_name in self.basis_set.items():
            if not isinstance(element, str) or not isinstance(basis_name, str):
                logger.warning(
                    f"Skipping invalid entry in basis_set dictionary: {{{element}: {basis_name}}}. Both key and value must be strings."
                )
                continue
            lines.append(f"{element.capitalize()} 0")  # Q-Chem expects element symbol followed by 0
            if (custom_basis := basis_registry.get_basis_set_object(basis_name)) is not None:
                lines.append(custom_basis.get_definition_for_element(element))
            else:
                lines.append(basis_name)
            lines.append("****")  # Separator
        lines.append("$end")
        return "\n".join(lines)

    def _get_solvent_block(self) -> str:
        """Generates the $solvent block for PCM calculations."""
        if self.implicit_solvation_model != "pcm" or not self.solvent:
            return ""

        solvent_lower = self.solvent.lower()
        lines = [
            "$solvent",
            f"    SolventName           {solvent_lower}",
            "$end",
        ]
        return "\n".join(lines)

    def _get_smx_block(self) -> str:
        """Generates the $smx block for SMD calculations."""
        if self.implicit_solvation_model != "smd" or not self.solvent:
            return ""

        # Q-Chem expects solvent names directly in $smx block. Case sensitivity might matter.
        # We pass the user-provided name, assuming they know the correct Q-Chem identifier.
        lines = [
            "$smx",
            f"    solvent    {self.solvent}",
            "$end",
        ]
        logger.debug(f"Generating $smx block for solvent: {self.solvent}")
        return "\n".join(lines)

    def export_input_file(self, geom: str) -> str:
        """Generates the Q-Chem input file content.

        Args:
            geom: Molecular geometry in XYZ format (multiline string, excluding header lines).

        Returns:
            String containing the formatted Q-Chem input file.

        Raises:
            InputGenerationError: If basis_set type is unsupported.
            ValidationError: If validation fails for settings.
            NotSupportedError: If requested task is not supported.
        """
        input_blocks: list[str] = [
            self._get_molecule_block(geom),
            self._get_rem_block(),
            self._get_basis_block(),
            self._get_solvent_block(),  # Add PCM block if needed
            self._get_smx_block(),  # Add SMD block if needed
            # Add other blocks like $external_charges if needed later
        ]

        final_input = "\n\n".join(block for block in input_blocks if block)

        return final_input.strip() + "\n"

    def export_input_file_from_geometry(self, geometry: "Geometry") -> str:
        """Generates the Q-Chem input file content from a Geometry object, validating basis sets.

        Checks if a dictionary basis set covers all elements in the geometry before
        generating the input file.

        Args:
            geometry: A Geometry object containing the molecular structure.

        Returns:
            String containing the formatted Q-Chem input file.

        Raises:
            ValidationError: If basis_set is a dictionary and does not contain entries
                             for all elements present in the geometry.
            InputGenerationError: If basis_set type is unsupported by the underlying export method.
            NotSupportedError: If the requested task is not supported by the underlying export method.
        """
        # Perform basis set validation if a dictionary is used
        if isinstance(self.basis_set, dict):
            geom_elements = geometry.unique_elements
            basis_elements = {element.upper() for element in self.basis_set}
            missing_elements = geom_elements - basis_elements
            if missing_elements:
                raise ValidationError(
                    f"Custom basis set dictionary is missing definitions for elements found in geometry: {missing_elements}. "
                    f"Geometry elements: {geom_elements}. Basis elements: {basis_elements}."
                )

        # Get the coordinate block string from the Geometry object
        geom_str = geometry.get_coordinate_block()

        # Call the original method that accepts the string
        return self.export_input_file(geom=geom_str)

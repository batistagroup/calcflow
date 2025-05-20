from copy import deepcopy
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, TypeVar, cast, get_args

from calcflow.core import CalculationInput
from calcflow.exceptions import InputGenerationError, NotSupportedError, ValidationError
from calcflow.geometry.static import Geometry
from calcflow.utils import logger

T_OrcaInput = TypeVar("T_OrcaInput", bound="OrcaInput")

# fmt:off
SUPPORTED_FUNCTIONALS = {"b3lyp", "pbe0", "m06", "cam-b3lyp", "wb97x", "wb97x-d3" }
# fmt:on

# Define allowed models specifically for ORCA
ORCA_ALLOWED_SOLVATION_MODELS = Literal["smd", "cpcm"]


@dataclass(frozen=True)
class OrcaInput(CalculationInput):
    """Input parameters specific to ORCA calculations.

    Inherits common parameters and validation logic from CalculationInput.
    Implements the export_input_file method to generate ORCA input files.

    Attributes:
        program: The program name, always "orca".
        task: The type of calculation task. Defaults to "energy".
            TDDFT tasks are controlled via `run_tddft` flag.
        n_cores: Number of CPU cores to use for the calculation. Defaults to 1.
        geom_mode: Geometry input mode for ORCA. Defaults to "xyz".
        ri_approx: RI approximation to use, e.g., "RIJCOSX", "RIJK". Defaults to None.
        aux_basis: Auxiliary basis set for RI approximation, e.g., "def2/J". Defaults to None.
        run_tddft: Flag to enable TDDFT calculation. Defaults to False.
        tddft_nroots: Number of roots to calculate in TDDFT. Defaults to 6 if TDDFT is enabled.
        tddft_iroot: Specific root to target in TDDFT geometry optimization. Defaults to None.
        tddft_triplets: Flag to include triplet states in TDDFT calculation. Defaults to False.
        tddft_use_tda: Flag to use the Tamm-Dancoff Approximation for TDDFT. Defaults to True.
        output_verbosity: Level of output verbosity. Defaults to "Normal".
        print_mos: Flag to control printing of Molecular Orbitals (MOs) and Overlap matrix.
            Defaults to False.
        recalc_hess_freq: Frequency for Hessian recalculation. Defaults to None.
        optimize_hydrogens_only: Flag to optimize only hydrogens in geometry optimization. Defaults to False.
    """

    program: ClassVar[str] = "orca"

    n_cores: int = 1
    geom_mode: str = "xyz"

    ri_approx: str | None = None
    aux_basis: str | None = None

    run_tddft: bool = False
    tddft_nroots: int | None = None
    tddft_iroot: int | None = None
    tddft_triplets: bool = False
    tddft_use_tda: bool = True

    implicit_solvation_model: ORCA_ALLOWED_SOLVATION_MODELS | None = None
    solvent: str | None = None

    output_verbosity: Literal["Normal", "Verbose", "Mini"] = "Normal"
    print_mos: bool = False
    recalc_hess_freq: int | None = None
    optimize_hydrogens_only: bool = False

    def __post_init__(self) -> None:
        """Performs ORCA-specific validation after initialization.

        Validates input parameters specific to ORCA calculations, including:
        - Basis set recommendations for geometry optimizations
        - RI approximation and auxiliary basis set consistency
        - TDDFT parameter validation
        - Solvation model compatibility

        Raises:
            ValidationError: If any ORCA-specific validation fails.
        """
        super().__post_init__()

        if self.task == "geometry" and self.basis_set == "def2-svp":
            logger.warning(
                "Using def2-svp for geometry optimization might yield less accurate results. "
                "Consider using def2-tzvp or larger basis sets for final geometries."
            )

        if isinstance(self.basis_set, dict):
            raise NotSupportedError(
                "Dictionary basis sets are not yet fully implemented in export_input_file for ORCA."
            )

        if self.implicit_solvation_model and self.implicit_solvation_model not in get_args(
            ORCA_ALLOWED_SOLVATION_MODELS
        ):
            raise NotSupportedError(
                f"ORCA's primary solvation models are CPCM and SMD. Model '{self.implicit_solvation_model}' is not directly supported."
                f"Allowed: {get_args(ORCA_ALLOWED_SOLVATION_MODELS)}"
            )

        if self.n_cores < 1:
            raise ValidationError("Number of cores must be a positive integer.")

        if self.ri_approx and not self.aux_basis:
            raise ValidationError(
                "An auxiliary basis set (aux_basis) must be provided when using an RI approximation (ri_approx)."
            )
        if not self.ri_approx and self.aux_basis:
            logger.warning(
                "An auxiliary basis set (aux_basis) was provided, but no RI approximation (ri_approx) was specified."
            )

        if self.run_tddft:
            if self.tddft_nroots is None and self.tddft_iroot is None:
                raise ValidationError("If run_tddft is True, either tddft_nroots or tddft_iroot must be specified.")
            if self.tddft_nroots is not None and self.tddft_iroot is not None:
                raise ValidationError(
                    "Specify either tddft_nroots (for multiple roots) or tddft_iroot (for a single root), not both."
                )
            if self.tddft_nroots is not None and self.tddft_nroots < 1:
                raise ValidationError("tddft_nroots must be a positive integer.")
            if self.tddft_iroot is not None and self.tddft_iroot < 1:
                raise ValidationError("tddft_iroot must be a positive integer.")

        if (self.implicit_solvation_model is not None) != (self.solvent is not None):
            raise ValidationError(
                "Both `implicit_solvation_model` and `solvent` must be provided together, or neither."
            )

        if self.recalc_hess_freq is not None:
            if self.task != "geometry":
                raise ValidationError(
                    "Hessian recalculation (recalc_hess_freq) is only applicable for 'geometry' tasks."
                )
            if self.recalc_hess_freq < 1:
                raise ValidationError("recalc_hess_freq must be a positive integer.")

        if self.optimize_hydrogens_only and self.task != "geometry":
            raise ValidationError("Optimizing only hydrogens is only applicable for 'geometry' tasks.")

    def set_solvation(self: T_OrcaInput, model: str | None, solvent: str | None) -> T_OrcaInput:
        """
        Set the implicit solvation model and solvent.

        Both `model` and `solvent` must be provided together, or neither.

        Args:
            model (ORCA_ALLOWED_SOLVATION_MODELS | None): Implicit solvation model to use (e.g., "smd").
                If None, solvation is disabled.
            solvent (str | None): Solvent to use for implicit solvation (e.g., "water").
                If None, solvation is disabled.

        Returns:
            T_OrcaInput: A new instance of the OrcaInput subclass with the solvation settings updated.
        Raises:
            ValidationError: If `model` and `solvent` are not consistently provided (both or neither).
            ValidationError: If the provided `model` is not one of the allowed models ("pcm", "smd", "isosvp", "cpcm").
        """
        if (model is not None) != (solvent is not None):
            raise ValidationError("Both `model` and `solvent` must be provided together, or neither.")
        solvent_lower = solvent.lower() if solvent else None
        model_lower = model.lower() if model else None

        # Validate against ORCA allowed models
        if model_lower is not None and model_lower not in get_args(ORCA_ALLOWED_SOLVATION_MODELS):
            raise ValidationError(
                f"Solvation model '{model}' not recognized for ORCA. Allowed: {get_args(ORCA_ALLOWED_SOLVATION_MODELS)}"
            )
        casted = cast(ORCA_ALLOWED_SOLVATION_MODELS, model_lower)

        return replace(self, implicit_solvation_model=casted, solvent=solvent_lower)  # type: ignore

    def enable_ri(self: T_OrcaInput, approx: str, aux_basis: str) -> T_OrcaInput:
        """Enable RI approximation with a given auxiliary basis set.

        Args:
            approx: The RI approximation method to use (e.g. "RIJCOSX", "RIJK")
            aux_basis: The auxiliary basis set to use (e.g. "def2/J")

        Returns:
            A new OrcaInput instance with RI enabled
        """
        logger.info(f"Enabling RI approximation: {approx} with aux basis {aux_basis}")
        return replace(self, ri_approx=approx, aux_basis=aux_basis)

    def enable_print_mos(self: T_OrcaInput) -> T_OrcaInput:
        """Enable printing of MOs and Overlap matrix in the output.

        Returns:
            A new OrcaInput instance with MO printing enabled
        """
        logger.info("Enabling MO and Overlap matrix printing.")
        return replace(self, print_mos=True)

    def set_tddft(
        self: T_OrcaInput,
        nroots: int | None = None,
        iroot: int | None = None,
        triplets: bool = False,
        use_tda: bool = True,
    ) -> T_OrcaInput:
        """Configure and enable TDDFT calculation.

        Args:
            nroots: Number of excited states to calculate. Defaults to 6 if neither nroots nor iroot specified.
            iroot: Specific excited state to target. Cannot be used with nroots.
            triplets: Whether to include triplet states. Defaults to False.
            use_tda: Whether to use the Tamm-Dancoff Approximation. Defaults to True.

        Returns:
            A new OrcaInput instance with TDDFT enabled and configured

        Notes:
            Either nroots or iroot must be specified, but not both.
            If neither is specified, defaults to calculating 6 roots.
        """
        if nroots is None and iroot is None:
            nroots = 5
            logger.info(f"Setting TDDFT with default nroots={nroots}, triplets={triplets}, use_tda={use_tda}")
        elif iroot is not None:
            logger.info(f"Setting TDDFT for iroot={iroot}, triplets={triplets}, use_tda={use_tda}")
        else:
            logger.info(f"Setting TDDFT for nroots={nroots}, triplets={triplets}, use_tda={use_tda}")

        return replace(
            self,
            run_tddft=True,
            tddft_nroots=nroots,
            tddft_iroot=iroot,
            tddft_triplets=triplets,
            tddft_use_tda=use_tda,
        )

    def set_hessian_recalculation(self: T_OrcaInput, frequency: int) -> T_OrcaInput:
        """Enable and configure Hessian recalculation during geometry optimization.

        Args:
            frequency: The frequency (number of steps) at which to recalculate the Hessian.

        Returns:
            A new OrcaInput instance with Hessian recalculation configured.

        Raises:
            ValidationError: If the task is not 'geometry' or frequency is not positive.
        """
        if self.task != "geometry":
            raise ValidationError("Hessian recalculation is only applicable for 'geometry' tasks.")
        if frequency < 1:
            raise ValidationError("Hessian recalculation frequency must be a positive integer.")

        logger.info(f"Setting Hessian recalculation frequency to every {frequency} steps.")
        return replace(self, recalc_hess_freq=frequency)

    def enable_optimize_hydrogens_only(self: T_OrcaInput) -> T_OrcaInput:
        """Enable optimizing only hydrogen atoms during geometry optimization.

        Returns:
            A new OrcaInput instance with hydrogen-only optimization enabled.

        Raises:
            ValidationError: If the task is not 'geometry'.
        """
        if self.task != "geometry":
            raise ValidationError("Optimizing only hydrogens is only applicable for 'geometry' tasks.")

        logger.info("Enabling optimization of hydrogen atoms only.")
        return replace(self, optimize_hydrogens_only=True)

    def copy(self: T_OrcaInput) -> T_OrcaInput:
        """Create a deep copy of the OrcaInput instance.

        Returns:
            A new, fully independent OrcaInput instance with the same parameters.
        """
        return deepcopy(self)

    def _handle_level_of_theory(self) -> list[str]:
        keywords: list[str] = []
        raw_method = self.level_of_theory.lower()
        method: str

        # Handle HF method variants
        hf_variants = {"uhf": "UHF", "rhf": "RHF", "hf": "UHF" if self.unrestricted else "RHF"}
        mp2_variants = {"ump2": "UMP2", "mp2": "MP2", "ri-mp2": "RI-MP2", "ri-ump2": "RI-UMP2"}
        mp2_variants["mp2"] = "UMP2" if self.unrestricted else "MP2"
        mp2_variants["ri-mp2"] = "RI-UMP2" if self.unrestricted else "RI-MP2"
        cc_variants = {"ccsd": "CCSD", "ccsd(t)": "CCSD(T)"}

        if "hf" in raw_method:
            if raw_method not in hf_variants:
                raise ValidationError(f"Unrecognized HF method: {raw_method}")

            method = hf_variants[raw_method]

            if method == "UHF" and not self.unrestricted:
                logger.warning("Requested method UHF but unrestricted was not set to True. Assuming UHF was intended.")
            elif method == "RHF" and self.unrestricted:
                raise ValidationError("Requested method RHF but unrestricted was set to True.")
            keywords.append(method)

        elif raw_method in SUPPORTED_FUNCTIONALS:
            if self.unrestricted:
                keywords.append("UKS")
            else:
                keywords.append("RKS")
            keywords.append(raw_method)

        elif raw_method in ["rks", "uks", "ks"]:
            raise ValidationError(
                "If you want to run DFT, please specify a functional as the level_of_theory. RKS or UKS will be assumed from the setting of unrestricted."
            )

        elif "mp2" in raw_method:
            if raw_method not in mp2_variants:
                raise ValidationError(f"Unrecognized MP2 method: {raw_method}")
            method = mp2_variants[raw_method]
            print(f"Got {method=} for {raw_method=} and {self.unrestricted=}")
            if method in ["UMP2", "RI-UMP2"] and not self.unrestricted:
                logger.warning(
                    "Requested method UMP2 but unrestricted was not set to True. Assuming UMP2 was intended."
                )
            keywords.append(method)

        elif raw_method in cc_variants:
            method = cc_variants[raw_method]
            keywords.append(method)
            if method == "CCSD(T)" and not self.unrestricted:
                logger.warning(
                    "Requested method CCSD(T) but unrestricted was not set to True. Assuming CCSD(T) was intended."
                )
            elif method == "CCSD" and self.unrestricted:
                raise ValidationError("Requested method CCSD but unrestricted was set to True.")
        else:
            raise ValidationError(f"Unsupported or unrecognized level_of_theory: '{self.level_of_theory}'")

        return keywords

    def _get_keywords_line(self) -> str:
        """Generates the main ORCA keywords line (!).

        Returns:
            String containing space-separated ORCA keywords prefixed with "!"

        Raises:
            InputGenerationError: If basis_set type is unsupported
            ValidationError: If RI approximation specified without auxiliary basis
        """
        keywords: list[str] = []

        if self.task == "geometry":
            keywords.append("Opt")
        elif self.task == "energy":
            keywords.append("SP")

        keywords.extend(self._handle_level_of_theory())

        if isinstance(self.basis_set, str):
            keywords.append(self.basis_set)
        elif isinstance(self.basis_set, dict):
            raise InputGenerationError("Dictionary basis sets require a %basis block, not yet implemented.")
        else:
            raise InputGenerationError(f"Unsupported basis_set type: {type(self.basis_set)}")

        if self.ri_approx:
            keywords.append(self.ri_approx)
            if self.aux_basis:
                keywords.append(self.aux_basis)
            else:
                raise ValidationError("RI approximation specified without auxiliary basis set.")

        if self.implicit_solvation_model:
            solvent_str = f'("{self.solvent}")' if self.solvent else ""
            model_lower = self.implicit_solvation_model.lower()
            if model_lower == "cpcm":
                keywords.append(f"CPCM{solvent_str}")
            elif model_lower == "smd":
                pass
            else:
                raise ValidationError(
                    f"Unrecognized solvation model: {self.implicit_solvation_model}"
                )  # pragma: no cover

        return f"! {' '.join(keywords)}"

    def _get_procs_line(self) -> str:
        """Generates the processor allocation line (%pal).

        Returns:
            String containing %pal block if n_cores > 1, empty string otherwise
        """
        if self.n_cores > 1:
            return f"%pal nprocs {self.n_cores} end"
        return ""

    def _get_mem_line(self) -> str:
        """Generates the memory specification line (%maxcore).

        Returns:
            String specifying memory per core in MB
        """
        return f"%maxcore {self.memory_per_core_mb}"

    def _get_solvent_block(self) -> str:
        """Generates the %cpcm block for SMD solvation.

        Returns:
            String containing %cpcm block if SMD solvation enabled, empty string otherwise

        Raises:
            ValidationError: If solvent name missing for SMD solvation
        """
        if not self.implicit_solvation_model or self.implicit_solvation_model.lower() == "cpcm":
            return ""
        elif self.implicit_solvation_model.lower() == "smd":
            if not self.solvent:
                raise ValidationError("Solvent name is required for SMD solvation.")
            lines = ["%cpcm"]
            lines.append("    smd true")
            lines.append(f'    SMDsolvent "{self.solvent}"')
            lines.append("end")
            return "\n".join(lines)
        raise ValidationError(f"Unrecognized solvation model: {self.implicit_solvation_model}")

    def _get_tddft_block(self) -> str:
        """Generates the %tddft block for TDDFT calculations.

        Returns:
            String containing %tddft block if TDDFT enabled, empty string otherwise

        Raises:
            ValidationError: If TDDFT enabled but neither nroots nor iroot specified
        """
        if not self.run_tddft:
            return ""

        lines = ["%tddft"]
        if self.tddft_iroot is not None:
            lines.append("    NRoots 1")
            lines.append(f"    IRoot {self.tddft_iroot}")
            logger.debug(f"Generating TDDFT block for iroot {self.tddft_iroot}")
        elif self.tddft_nroots is not None:
            lines.append(f"    NRoots {self.tddft_nroots}")
            logger.debug(f"Generating TDDFT block for nroots {self.tddft_nroots}")
        else:
            raise ValidationError("TDDFT requested but neither nroots nor iroot specified.")

        lines.append(f"    Triplets {str(self.tddft_triplets).lower()}")
        lines.append(f"    TDA {str(self.tddft_use_tda).lower()}")
        lines.append("end")
        return "\n".join(lines)

    def _get_output_block(self) -> str:
        """Generates the %output block for controlling print levels.

        Returns:
            String containing %output block if print_mos enabled, empty string otherwise
        """
        lines: list[str] = []
        if self.print_mos:
            lines.append("%output")
            lines.append("    Print[ P_MOs ] 1")
            lines.append("    Print[ P_Overlap ] 1")
            lines.append("end")

        return "\n".join(lines)

    def _get_geom_block(self) -> str:
        """Generates the %geom block for Hessian recalculation during geometry optimization.

        Returns:
            String containing %geom block if recalc_hess_freq is set or optimize_hydrogens_only is true, empty string otherwise.
        """
        if self.task == "geometry" and (self.recalc_hess_freq is not None or self.optimize_hydrogens_only):
            lines = ["%geom"]
            if self.recalc_hess_freq is not None:
                lines.append("    Calc_Hess true")
                lines.append(f"    Recalc_Hess {self.recalc_hess_freq}")
            if self.optimize_hydrogens_only:
                lines.append("    OptimizeHydrogens true")
            lines.append("end")
            return "\n".join(lines)
        return ""

    def export_input_file(self, geometry: "Geometry") -> str:
        """Generates the ORCA input file content.

        Args:
            geom: Molecular geometry in XYZ format (multiline string, excluding header lines)

        Returns:
            String containing the formatted ORCA input file

        Raises:
            InputGenerationError: If dictionary basis sets provided or basis_set type unsupported
            ValidationError: If validation fails for settings (e.g., missing solvent for SMD,
                inconsistent RI/aux basis, inconsistent TDDFT roots)
        """
        input_blocks: list[str] = [
            self._get_keywords_line(),
            self._get_procs_line(),
            self._get_mem_line(),
            self._get_solvent_block(),
            self._get_tddft_block(),
            self._get_output_block(),
            self._get_geom_block(),
            f"* {self.geom_mode} {self.charge} {self.spin_multiplicity}",
            geometry.get_coordinate_block(),
            "*",
        ]

        final_input = "\n".join(block for block in input_blocks if block)

        return final_input.strip() + "\n"

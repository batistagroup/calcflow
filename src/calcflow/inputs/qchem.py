import re
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, TypeVar, cast, get_args

from calcflow.basis_sets import registry as basis_registry
from calcflow.core import CalculationInput
from calcflow.exceptions import ConfigurationError, InputGenerationError, NotSupportedError, ValidationError
from calcflow.geometry.static import Geometry
from calcflow.utils import logger

# Define allowed models specifically for Q-Chem
QCHEM_ALLOWED_SOLVATION_MODELS = Literal["pcm", "smd", "isosvp", "cpcm"]

# Allowed MOM methods
QCHEM_ALLOWED_MOM_METHODS = Literal["IMOM", "MOM"]

T_QchemInput = TypeVar("T_QchemInput", bound="QchemInput")

# fmt:off
SUPPORTED_FUNCTIONALS = {"b3lyp", "pbe0", "m06", "cam-b3lyp", "wb97x", "wb97x-d3", "src1-r1"}
# fmt:on

QCHEM_BASIS_REGISTRY = basis_registry.ProgramBasisRegistry("qchem")


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
        run_mom: Flag to enable MOM calculation. Defaults to False.
        mom_method: Method for MOM calculation. Defaults to "IMOM".
        mom_alpha_occ: Alpha orbital occupation string for MOM (e.g., "1:77 79").
        mom_beta_occ: Beta orbital occupation string for MOM.
        mom_transition: Symbolic transition specification (e.g., "HOMO->LUMO").
        reduced_excitation_space: Flag to enable reduced excitation space for TDDFT calculations.
        initial_orbitals: List of molecular orbitals from which excitations are allowed when TRNSS is enabled.
        mom_job2_charge: Charge for the second job in a MOM calculation
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

    # --- TDDFT Reduced Excitation Space ---
    reduced_excitation_space: bool = False
    initial_orbitals: list[int] | None = None

    # --- MOM Specific Attributes ---
    run_mom: bool = False
    mom_method: str = "IMOM"  # Default to IMOM if run_mom is True
    mom_alpha_occ: str | None = None
    mom_beta_occ: str | None = None
    mom_transition: str | None = None  # Stores symbolic transition if using that mode
    mom_job2_charge: int | None = None  # Charge for the second job in a MOM calculation

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

        # Ensure reduced_excitation_space is only true if TDDFT is enabled
        if self.reduced_excitation_space:
            if not self.run_tddft:
                raise ConfigurationError("reduced_excitation_space (TRNSS) requires TDDFT (run_tddft) to be enabled.")
            # Validate initial_orbitals if reduced_excitation_space is enabled
            if not self.initial_orbitals:
                raise ValidationError("The initial_orbitals list cannot be empty for reduced excitation space.")
            if not all(isinstance(orb, int) and orb > 0 for orb in self.initial_orbitals):
                raise ValidationError("All initial_orbitals must be positive integers.")

        # Use the Q-Chem specific literal for validation
        if self.implicit_solvation_model and self.implicit_solvation_model not in get_args(
            QCHEM_ALLOWED_SOLVATION_MODELS
        ):
            raise ValidationError(
                f"Solvation model '{self.implicit_solvation_model}' not recognized. Allowed: {get_args(QCHEM_ALLOWED_SOLVATION_MODELS)}"
            )

    def copy(self: T_QchemInput) -> T_QchemInput:
        """Create a deep copy of the QchemInput instance.

        Returns:
            A new QchemInput instance with the same parameters.
        """
        return deepcopy(self)

    def enable_mom(self: T_QchemInput, method: str = "IMOM") -> T_QchemInput:
        """Enable MOM calculation with specified method.

        Args:
            method: The MOM method to use. Must be either "IMOM" or "MOM". Defaults to "IMOM".

        Returns:
            A new QchemInput instance with MOM enabled.

        Raises:
            ValidationError: If method is not "IMOM" or "MOM".
        """
        method_upper = method.upper()
        if method_upper not in {"IMOM", "MOM"}:
            raise ValidationError(f"MOM method must be either 'IMOM' or 'MOM', got '{method}'")

        return replace(self, run_mom=True, mom_method=method_upper)

    def set_mom_occupation(self: T_QchemInput, alpha_occ: str, beta_occ: str) -> T_QchemInput:
        """Set orbital occupation strings for MOM calculation using Q-Chem style ranges.

        Args:
            alpha_occ: Q-Chem style occupation string for alpha orbitals (e.g., "1:77 79").
            beta_occ: Q-Chem style occupation string for beta orbitals (e.g., "1:77 78").

        Returns:
            A new QchemInput instance with the specified occupations.

        Raises:
            ConfigurationError: If MOM is not enabled (call enable_mom first).
            ValidationError: If the occupation strings are not in valid Q-Chem format.
        """
        if not self.run_mom:
            raise ConfigurationError("MOM must be enabled first. Call enable_mom() before set_mom_occupation().")

        # Basic validation of the Q-Chem range format
        for occ, label in [(alpha_occ, "alpha"), (beta_occ, "beta")]:
            parts = occ.split()
            for part in parts:
                if ":" in part:
                    split_parts = part.split(":")
                    # Ensure exactly two parts for a range and both are digits
                    if len(split_parts) != 2 or not (split_parts[0].isdigit() and split_parts[1].isdigit()):
                        raise ValidationError(
                            f"Invalid {label} occupation range '{part}'. Must be in format 'start:end' with integers."
                        )
                elif not part.isdigit():
                    raise ValidationError(f"Invalid {label} occupation number '{part}'. Must be an integer.")

        return replace(self, mom_alpha_occ=alpha_occ, mom_beta_occ=beta_occ, mom_transition=None)

    def set_mom_transition(self: T_QchemInput, transition: str) -> T_QchemInput:
        """Set MOM occupations using a symbolic HOMO/LUMO transition.

        For closed-shell singlet ground states only. The transition will be applied
        to both alpha and beta electrons identically.

        Args:
            transition: Transition specification (e.g., "HOMO->LUMO", "HOMO-1->LUMO+2").
                       Only single transitions supported.

        Returns:
            A new QchemInput instance with the transition stored.

        Raises:
            ConfigurationError: If MOM is not enabled.
            ValidationError: If the transition specification is invalid.
        """
        if not self.run_mom:
            raise ConfigurationError("MOM must be enabled first. Call enable_mom() before set_mom_transition().")

        # Parse and validate the transition format
        if "->" not in transition:
            raise ValidationError(f"Invalid transition format: '{transition}'. Expected 'SourceOrb->TargetOrb'.")

        source, target = [x.strip().upper() for x in transition.split("->")]

        # Validate HOMO/LUMO format with optional +/- offset OR positive integer
        pattern = re.compile(r"(HOMO|LUMO)(?:([+-])(\d+))?")

        def _is_valid_specifier(spec: str) -> bool:
            if spec.isdigit() and int(spec) > 0:
                return True  # Positive integer
            match = pattern.fullmatch(spec)
            if not match:
                return False  # Neither integer nor HOMO/LUMO format
            # Additional checks for HOMO/LUMO format
            if match.group(1) == "HOMO" and match.group(2) == "+":
                raise ValidationError(f"Invalid source orbital: '{spec}'. Cannot use HOMO+n.")
            if match.group(1) == "LUMO" and match.group(2) == "-":
                raise ValidationError(f"Invalid target orbital: '{spec}'. Cannot use LUMO-n.")
            return True

        if not _is_valid_specifier(source):
            raise ValidationError(
                f"Invalid source orbital: '{source}'. Must be HOMO[-n], LUMO[+n], or positive integer."
            )
        if not _is_valid_specifier(target):
            raise ValidationError(
                f"Invalid target orbital: '{target}'. Must be HOMO[-n], LUMO[+n], or positive integer."
            )

        # Store the validated transition string (keep original case for potential user preference)
        original_source, original_target = [x.strip() for x in transition.split("->")]
        validated_transition = f"{original_source}->{original_target}"

        return replace(self, mom_transition=validated_transition, mom_alpha_occ=None, mom_beta_occ=None)

    def set_mom_ground_state(self: T_QchemInput) -> T_QchemInput:
        """Sets the MOM calculation to target the ground state electronic configuration.

        This method is intended for scenarios where MOM is enabled (run_mom=True),
        but the calculation should target the ground state. This allows using
        MOM's convergence algorithms or specific MOM methods (like IMOM) for
        obtaining the ground state, followed by potential post-processing like TDDFT
        on this MOM-converged ground state.

        This method marks the transition as "GROUND_STATE". The actual occupation
        strings will be generated later during input file export, based on the
        molecule's total electrons, charge, and spin multiplicity.

        Crucially, this method requires that the system is intended to be a
        closed-shell singlet (charge=0, spin_multiplicity=1) for the
        "GROUND_STATE" marker to be correctly interpreted. The validation for this
        occurs during the generation of the $occupied block.

        This will also ensure `unrestricted=True` is set, as MOM calculations
        typically require it, and clears any explicit `mom_alpha_occ` and `mom_beta_occ`
        to avoid conflicts with the "GROUND_STATE" marker.

        Returns:
            A new QchemInput instance configured for a MOM ground state calculation.

        Raises:
            ConfigurationError: If MOM is not enabled (run_mom=False).
            NotSupportedError: If the system is not a closed-shell singlet
                               (charge != 0 or spin_multiplicity != 1), checked
                               during input generation.
        """
        if not self.run_mom:
            raise ConfigurationError(
                "MOM must be enabled (run_mom=True) via enable_mom() before setting MOM to target the ground state."
            )

        # Defer electron count validation until geometry is available.
        # The check for charge and spin_multiplicity for this specific mode
        # will happen in _generate_occupied_block when "GROUND_STATE" is processed.

        logger.info(
            "Setting MOM to target ground state reference. "
            "Occupation will be determined at input generation. "
            "mom_transition='GROUND_STATE'. Unrestricted will be True."
        )

        return replace(
            self,
            mom_transition="GROUND_STATE",
            mom_alpha_occ=None,
            mom_beta_occ=None,
            unrestricted=True,
        )

    def _generate_occupied_block(self, geometry: "Geometry") -> str:
        """Generates the $occupied block for MOM calculations.

        Args:
            geometry: The Geometry object for the molecule.

        Returns:
            String containing the formatted $occupied block.

        Raises:
            ConfigurationError: If MOM is enabled but occupations are not set.
            NotSupportedError: If the system is not closed-shell singlet in ground state.
        """
        if not self.run_mom:
            return ""

        if not self.unrestricted:
            raise ConfigurationError("MOM requires unrestricted=True")

        # Determine the charge to use for electron counting in this block
        # If mom_job2_charge is set, it means we are generating the $occupied block
        # for the *second* job of a MOM calculation, which might have a different charge.
        effective_charge = self.mom_job2_charge if self.mom_job2_charge is not None else self.charge

        # For MOM targeting ground state or excited states, we currently only support
        # closed-shell singlet reference states for determining occupations.
        # This validation now uses the effective_charge for the current job.
        if self.mom_transition is not None and (effective_charge != 0 or self.spin_multiplicity != 1):
            raise NotSupportedError(
                "MOM occupation determination via symbolic transition or 'GROUND_STATE' marker "
                f"currently only supports a closed-shell singlet reference state for the job where occupations are determined. "
                f"Effective charge for this job = {effective_charge}, multiplicity = {self.spin_multiplicity}."
            )

        total_electrons = geometry.total_nuclear_charge - effective_charge

        if self.mom_transition == "GROUND_STATE":
            if total_electrons < 0:
                raise ConfigurationError(
                    f"Invalid total electron count ({total_electrons}) for 'GROUND_STATE' MOM. "
                    f"(Nuclear charge: {geometry.total_nuclear_charge}, System charge: {self.charge})."
                )
            if total_electrons % 2 != 0:
                raise ConfigurationError(
                    f"Expected an even number of total electrons ({total_electrons}) for 'GROUND_STATE' MOM "
                    f"with a closed-shell singlet reference."
                )
            n_occupied_per_spin = total_electrons // 2
            if n_occupied_per_spin <= 0:
                # This case also covers total_electrons == 0
                raise ConfigurationError(
                    f"System with {total_electrons} electrons ({n_occupied_per_spin} per spin) has insufficient "
                    f"occupied orbitals for 'GROUND_STATE' MOM. At least 1 occupied orbital per spin required."
                )
            alpha_occ = f"1:{n_occupied_per_spin}" if n_occupied_per_spin > 1 else "1"
            beta_occ = alpha_occ  # Closed-shell singlet ground state
        elif self.mom_transition is not None:
            # This path handles symbolic transitions like HOMO->LUMO
            alpha_occ, beta_occ = _convert_transition_to_occupations(self.mom_transition, total_electrons)
        elif self.mom_alpha_occ is None or self.mom_beta_occ is None:
            raise ConfigurationError(
                "If mom_transition is not 'GROUND_STATE' or a symbolic transition, "
                "then both mom_alpha_occ and mom_beta_occ must be explicitly set."
            )
        else:
            # Use explicitly provided occupation strings
            alpha_occ = self.mom_alpha_occ
            beta_occ = self.mom_beta_occ

        # Format the block
        lines = [
            "$occupied",
            alpha_occ,
            beta_occ,
            "$end",
        ]
        return "\n".join(lines)

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

    def set_reduced_excitation_space(self: T_QchemInput, initial_orbitals: list[int]) -> T_QchemInput:
        """Configure a reduced excitation space for TDDFT calculations.

        This setting activates Q-Chem's TRNSS capability, restricting excitations
        to originate from a specified subset of molecular orbitals. It sets
        TRNSS = TRUE, TRTYPE = 3, and N_SOL = len(initial_orbitals) in the $rem block,
        and requires a $solute block specifying the active orbitals.

        Args:
            initial_orbitals: A list of positive integers representing the molecular orbitals
                             from which excitations are allowed. The list will be
                             stored internally sorted and with duplicates removed.

        Returns:
            A new QchemInput instance with reduced excitation space configured.

        Raises:
            ConfigurationError: If TDDFT (run_tddft) is not enabled.
            ValidationError: If the initial_orbitals list is empty or contains
                             non-positive integers.
        """
        if not self.run_tddft:
            raise ConfigurationError(
                "Reduced excitation space (TRNSS) requires TDDFT to be enabled. "
                "Ensure run_tddft is True before calling this method."
            )
        if not initial_orbitals:
            raise ValidationError("The initial_orbitals list cannot be empty for reduced excitation space.")
        if not all(isinstance(orb, int) and orb > 0 for orb in initial_orbitals):
            raise ValidationError("All initial_orbitals must be positive integers.")

        # Store unique, sorted orbitals
        processed_orbitals = sorted(list(set(initial_orbitals)))

        logger.debug(
            f"Setting reduced excitation space with orbitals: {processed_orbitals}. "
            f"This will set TRNSS=True, TRTYPE=3, N_SOL={len(processed_orbitals)}."
        )
        return replace(
            self,
            reduced_excitation_space=True,
            initial_orbitals=processed_orbitals,
        )

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

    def _get_molecule_block(self, geom: str, charge_override: int | None = None) -> str:
        """Generates the $molecule block.

        Args:
            geom: Molecular geometry in XYZ format (multiline string, excluding header lines).
            charge_override: Optional integer to override the instance's charge for this block.

        Returns:
            String containing the formatted $molecule block.
        """
        charge_to_use = charge_override if charge_override is not None else self.charge
        lines = [
            "$molecule",
            f"{charge_to_use} {self.spin_multiplicity}",
            geom.strip(),
            "$end",
        ]
        return "\n".join(lines)

    def _get_rem_block(
        self,
        scf_guess: str | None = None,
        mom_start: bool = False,
        force_unrestricted: bool = False,
        skip_optimization: bool = False,
    ) -> str:
        """Generates the $rem block with optional MOM-specific settings."""
        rem_vars: dict[str, str | bool | int] = {}

        # --- Job Type --- #
        if skip_optimization and self.task == "geometry" or self.task == "energy":
            rem_vars["JOBTYPE"] = "sp"
        elif self.task == "geometry":
            rem_vars["JOBTYPE"] = "opt"
        else:
            raise NotSupportedError(f"Task type '{self.task}' not currently supported for Q-Chem export.")

        # --- Level of Theory --- #
        method = self.level_of_theory.lower()
        if method in SUPPORTED_FUNCTIONALS or method == "hf":
            rem_vars["METHOD"] = method
        else:
            raise ValidationError(f"Unsupported or unrecognized level_of_theory for Q-Chem: '{self.level_of_theory}'")

        # --- Basis Set --- #
        if isinstance(self.basis_set, str):
            rem_vars["BASIS"] = self.basis_set
        elif isinstance(self.basis_set, dict):
            rem_vars["BASIS"] = "gen"
        else:
            raise InputGenerationError(f"Unsupported basis_set type: {type(self.basis_set)}")

        # --- Spin/Symmetry --- #
        rem_vars["UNRESTRICTED"] = force_unrestricted or self.unrestricted
        rem_vars["SYMMETRY"] = False
        rem_vars["SYM_IGNORE"] = True

        # --- SCF Settings --- #
        if scf_guess:
            rem_vars["SCF_GUESS"] = scf_guess
        if mom_start:
            rem_vars["MOM_START"] = 1
            rem_vars["MOM_METHOD"] = self.mom_method

        # --- TDDFT --- #
        if self.run_tddft:  # noqa: SIM102 for clarity
            # TDDFT keywords should only be added if:
            # 1. This is the second job of a MOM calculation (mom_start = True)
            # OR
            # 2. This is NOT a MOM calculation (self.run_mom = False), so it's a single job.
            if mom_start or not self.run_mom:
                if self.tddft_nroots is None:
                    raise InputGenerationError("tddft_nroots must be set when run_tddft is True.")
                rem_vars["CIS_N_ROOTS"] = self.tddft_nroots
                rem_vars["CIS_SINGLETS"] = self.tddft_singlets
                rem_vars["CIS_TRIPLETS"] = self.tddft_triplets
                rem_vars["STATE_ANALYSIS"] = self.tddft_state_analysis
                if self.rpa:
                    rem_vars["RPA"] = True

                # --- Reduced Excitation Space for TDDFT --- (This logic is now nested)
                if self.reduced_excitation_space:
                    # The outer condition (mom_start or not self.run_mom) already ensures this is the correct job context.
                    if not self.initial_orbitals:  # Should be caught by setter, defensive check
                        raise InputGenerationError(
                            "initial_orbitals must be set when reduced_excitation_space is True."
                        )
                    rem_vars["TRNSS"] = True
                    rem_vars["TRTYPE"] = 3
                    rem_vars["N_SOL"] = len(self.initial_orbitals)

        # --- Implicit Solvation --- #
        if self.implicit_solvation_model:
            if not self.solvent:
                raise InputGenerationError(
                    f"Solvation model '{self.implicit_solvation_model}' requires a solvent to be specified."
                )
            rem_vars["SOLVENT_METHOD"] = self.implicit_solvation_model

        # --- Formatting --- #
        lines = ["$rem"]
        max_key_len = max(len(k) for k in rem_vars) if rem_vars else 0
        for key, value in rem_vars.items():
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
            if basis_name in QCHEM_BASIS_REGISTRY:
                lines.append(QCHEM_BASIS_REGISTRY[basis_name][element])
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

    def _get_solute_block(self) -> str:
        """Generates the $solute block for reduced excitation space TDDFT.

        This block lists the molecular orbitals from which excitations are allowed
        when TRNSS is enabled.

        Returns:
            String containing the formatted $solute block if reduced_excitation_space
            is True and initial_orbitals are provided, otherwise an empty string.
        """
        if not self.reduced_excitation_space or not self.initial_orbitals:
            return ""

        lines = ["$solute"]
        lines.append(" ".join(str(orb) for orb in self.initial_orbitals))
        lines.append("$end")
        return "\n".join(lines)

    def _pre_export_validation(self, geometry: "Geometry") -> None:
        if isinstance(self.basis_set, dict):
            geom_elements = geometry.unique_elements
            basis_elements = {element.upper() for element in self.basis_set}
            missing_elements = geom_elements - basis_elements
            if missing_elements:
                raise ValidationError(
                    f"Custom basis set dictionary is missing definitions for elements found in geometry: {missing_elements}. "
                    f"Geometry elements: {geom_elements}. Basis elements: {basis_elements}."
                )

    def export_input_file(self, geometry: "Geometry") -> str:
        """Generates the Q-Chem input file content.

        For MOM calculations, generates a two-job input file with the second job
        reading orbitals from the first and applying the MOM procedure.

        Args:
            geometry: The Geometry object containing the molecular structure.

        Returns:
            String containing the formatted Q-Chem input file.

        Raises:
            ValidationError: If basis_set validation fails.
            ConfigurationError: If MOM settings are invalid.
            NotSupportedError: If requested features are not supported.
        """
        self._pre_export_validation(geometry)

        # Generate first job blocks
        first_job_blocks = [
            self._get_molecule_block(geometry.get_coordinate_block()),
            self._get_rem_block(),
            self._get_basis_block(),
            self._get_solvent_block(),
            self._get_smx_block(),
        ]
        if not self.run_mom:
            first_job_blocks.append(self._get_solute_block())

        first_job = "\n\n".join(block for block in first_job_blocks if block)

        if not self.run_mom:
            return first_job.strip() + "\n"

        # For MOM, add second job
        second_job_molecule_block: str
        if self.mom_job2_charge is not None:
            # If a specific charge is set for job 2, generate a full $molecule block
            logger.info(f"Generating $molecule block for MOM job 2 with charge: {self.mom_job2_charge}")
            second_job_molecule_block = self._get_molecule_block(
                geometry.get_coordinate_block(), charge_override=self.mom_job2_charge
            )
        else:
            # Default behavior: read geometry from the first job
            second_job_molecule_block = "$molecule\n    read\n$end"

        second_job_blocks = [
            second_job_molecule_block,
            self._get_rem_block(
                scf_guess="read",
                mom_start=True,
                force_unrestricted=True,
                skip_optimization=True,
            ),
            self._get_solvent_block(),  # Keep solvation consistent
            self._get_basis_block(),
            self._get_smx_block(),
            self._generate_occupied_block(geometry),
            self._get_solute_block(),
        ]
        second_job = "\n\n".join(block for block in second_job_blocks if block)

        return f"{first_job.strip()}\n\n@@@\n\n{second_job.strip()}\n"

    def set_mom_job2_charge(self: T_QchemInput, charge: int) -> T_QchemInput:
        """Set a specific charge for the second job in a MOM calculation.

        This charge will be used in the $molecule block of the second job,
        allowing it to differ from the primary charge of the QchemInput object.
        The spin multiplicity remains the same as the primary setting.

        Args:
            charge: The integer charge for the second MOM job.

        Returns:
            A new QchemInput instance with the second job charge set.

        Raises:
            ConfigurationError: If MOM is not enabled (run_mom=False).
        """
        if not self.run_mom:
            raise ConfigurationError(
                "MOM must be enabled (run_mom=True) via enable_mom() before setting a specific charge for the second MOM job."
            )
        logger.info(
            f"Setting charge for second MOM job to: {charge}. This will affect the $molecule block and electron count for job 2."
        )
        return replace(self, mom_job2_charge=charge)


def _convert_transition_to_occupations(transition: str, n_electrons: int) -> tuple[str, str]:
    """Converts a symbolic transition to Q-Chem occupation strings.

    Args:
        transition: String containing the transition to convert.
        n_electrons: Total number of electrons in the system.

    Returns:
        Tuple of (alpha_occ, beta_occ) strings in Q-Chem format.

    Raises:
        ConfigurationError: If mom_transition is not set.
        ValidationError: If the transition format is invalid.
    """
    # For closed shell singlet, each spin channel has n_electrons/2 electrons
    if n_electrons % 2 != 0:
        raise ValidationError(f"Expected even number of electrons, got {n_electrons}.")

    n_per_spin = n_electrons // 2
    homo_idx = n_per_spin  # 1-based indexing in Q-Chem
    lumo_idx = homo_idx + 1

    # Ground state occupation for beta (closed shell reference)
    beta_occ = str(homo_idx) if homo_idx == 1 else f"1:{homo_idx}"

    # --- Parse Source and Target --- #
    source_str, target_str = [x.strip().upper() for x in transition.split("->")]
    pattern = re.compile(r"(HOMO|LUMO)(?:([+-])(\d+))?")

    def _parse_specifier(spec: str, name: str) -> int:
        if spec.isdigit():
            idx = int(spec)
            if idx <= 0:
                raise ValidationError(f"{name.capitalize()} orbital index must be positive, got '{spec}'")
            return idx

        match = pattern.fullmatch(spec)
        # This should ideally not happen due to validation in set_mom_transition, but check defensively
        if not match:
            raise ValidationError(f"Invalid {name} orbital format: '{spec}'")

        base = match.group(1)
        op = match.group(2)
        offset = int(match.group(3)) if match.group(3) else 0

        if base == "HOMO":
            idx = homo_idx
            if op == "-":
                idx -= offset
            # op == "+" case already caught by set_mom_transition validation
        else:  # LUMO
            idx = lumo_idx
            if op == "+":
                idx += offset
            # op == "-" case already caught by set_mom_transition validation

        if idx <= 0:
            raise ValidationError(f"Calculated {name} orbital index must be positive, got {idx} from '{spec}'")
        return idx

    source_idx = _parse_specifier(source_str, "source")
    target_idx = _parse_specifier(target_str, "target")

    # --- Validate Transition Physics --- #
    if source_idx > homo_idx:
        raise ValidationError(
            f"Source orbital must be occupied (index <= HOMO={homo_idx}), got {source_idx} from '{source_str}'"
        )
    if target_idx <= homo_idx:
        raise ValidationError(
            f"Target orbital must be unoccupied (index > HOMO={homo_idx}), got {target_idx} from '{target_str}'"
        )

    # --- Generate Occupation Strings --- #
    # (Existing logic for generating occ string based on source_idx/target_idx follows)
    # Get source orbital index
    # source_base = source_match.group(1)
    # ... (old logic removed) ...

    # Get target orbital index
    # target_base = target_match.group(1)
    # ... (old logic removed) ...

    # Generate Q-Chem occupation string for alpha
    if source_idx == homo_idx and target_idx == lumo_idx:
        if homo_idx == 1:
            occ = str(lumo_idx)  # Special case for 1-electron per spin (HOMO=1)
        else:
            occ = f"1:{homo_idx - 1} {lumo_idx}"
    else:
        # For more complex transitions, we need to build the string carefully
        occupied = set(range(1, homo_idx + 1))
        occupied.remove(source_idx)
        occupied.add(target_idx)
        occupied_list = sorted(list(occupied))

        if not occupied_list:
            occ = ""  # Handle case with zero occupied orbitals if it ever occurs
        else:
            parts = []
            start = occupied_list[0]
            end = start
            for i in range(1, len(occupied_list)):
                if occupied_list[i] == end + 1:
                    end = occupied_list[i]
                else:
                    if start == end:
                        parts.append(str(start))
                    else:
                        parts.append(f"{start}:{end}")
                    start = occupied_list[i]
                    end = start
            # Append the last range/number
            if start == end:
                parts.append(str(start))
            else:
                parts.append(f"{start}:{end}")
            occ = " ".join(parts)

    # Alpha occupation is modified, beta remains ground state
    alpha_occ = occ

    return alpha_occ, beta_occ

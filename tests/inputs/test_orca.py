import logging
from dataclasses import replace

import pytest

from calcflow.exceptions import NotSupportedError, ValidationError
from calcflow.geometry.static import Geometry
from calcflow.inputs.orca import OrcaInput
from calcflow.utils import logger

logger.setLevel(logging.INFO)


@pytest.fixture
def default_geom() -> Geometry:
    """Provides a default geometry object."""
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.0)),
        ("H", (0.0, 1.0, 0.0)),
    ]
    return Geometry(num_atoms=3, comment="Default water geometry", atoms=atoms)


@pytest.fixture
def minimal_orca_input(default_geom: Geometry) -> OrcaInput:
    """Provides a minimal valid OrcaInput instance."""
    return OrcaInput(
        task="energy",
        level_of_theory="hf",
        basis_set="sto-3g",
        charge=0,
        spin_multiplicity=1,
    )


class TestOrcaInputInitValidation:
    """Tests for OrcaInput initialization and __post_init__ validation."""

    def test_minimal_initialization(self, minimal_orca_input: OrcaInput) -> None:
        """Test initialization with minimal valid parameters."""
        assert minimal_orca_input.program == "orca"
        assert minimal_orca_input.task == "energy"
        assert minimal_orca_input.level_of_theory == "hf"
        assert minimal_orca_input.basis_set == "sto-3g"
        assert minimal_orca_input.n_cores == 1
        assert minimal_orca_input.ri_approx is None
        assert minimal_orca_input.aux_basis is None
        assert not minimal_orca_input.run_tddft
        assert minimal_orca_input.tddft_nroots is None
        assert minimal_orca_input.tddft_iroot is None
        assert not minimal_orca_input.tddft_triplets
        assert minimal_orca_input.tddft_use_tda  # Defaults to True
        assert minimal_orca_input.implicit_solvation_model is None
        assert minimal_orca_input.solvent is None

    def test_override_defaults(self) -> None:
        """Test overriding default parameters during initialization."""
        instance = OrcaInput(
            task="geometry",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            charge=-1,
            spin_multiplicity=2,
            n_cores=4,
            ri_approx="RIJCOSX",
            aux_basis="def2/J",
            run_tddft=True,
            tddft_nroots=10,
            tddft_triplets=True,
            tddft_use_tda=True,  # Explicitly True, but also default
            implicit_solvation_model="cpcm",
            solvent="water",
        )
        assert instance.task == "geometry"
        assert instance.n_cores == 4
        assert instance.ri_approx == "RIJCOSX"
        assert instance.aux_basis == "def2/J"
        assert instance.run_tddft
        assert instance.tddft_nroots == 10
        assert instance.tddft_iroot is None
        assert instance.tddft_triplets
        assert instance.tddft_use_tda
        assert instance.implicit_solvation_model == "cpcm"
        assert instance.solvent == "water"

    def test_ri_requires_aux_basis_validation(self) -> None:
        """Test validation fails if ri_approx is set without aux_basis."""
        with pytest.raises(ValidationError, match="An auxiliary basis set .* must be provided"):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                ri_approx="RIJCOSX",
                aux_basis=None,  # Missing aux_basis
            )

    def test_aux_basis_without_ri_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning if aux_basis is set without ri_approx."""
        OrcaInput(
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            charge=0,
            spin_multiplicity=1,
            ri_approx=None,  # Missing ri_approx
            aux_basis="def2/J",
        )
        assert "auxiliary basis set (aux_basis) was provided, but no RI approximation" in caplog.text

    def test_negative_cores_validation(self) -> None:
        """Test validation fails for non-positive n_cores."""
        with pytest.raises(ValidationError, match="Number of cores must be a positive integer"):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                n_cores=0,
            )

    def test_dict_basis_set_not_supported_validation(self) -> None:
        """Test validation fails for dictionary basis sets (currently unsupported)."""
        with pytest.raises(
            NotSupportedError,
            match="Dictionary basis sets are not yet fully implemented in export_input_file for ORCA.",
        ):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set={"O": "def2-svp", "H": "sto-3g"},  # type: ignore
                charge=0,
                spin_multiplicity=1,
            )

    def test_unsupported_solvation_model_validation(self) -> None:
        """Test validation fails for unsupported implicit solvation models."""
        with pytest.raises(NotSupportedError, match="ORCA's primary solvation models are CPCM and SMD"):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                implicit_solvation_model="xyz",  # type: ignore[arg-type] # Intentionally invalid for testing validation
                solvent="water",
            )

    # --- TDDFT Validation Tests ---
    def test_tddft_requires_nroots_or_iroot_validation(self) -> None:
        """Test validation fails if run_tddft=True but nroots/iroot are None."""
        with pytest.raises(
            ValidationError, match="If run_tddft is True, either tddft_nroots or tddft_iroot must be specified"
        ):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                run_tddft=True,
                tddft_nroots=None,
                tddft_iroot=None,
            )

    def test_tddft_nroots_and_iroot_exclusive_validation(self) -> None:
        """Test validation fails if both tddft_nroots and tddft_iroot are specified."""
        with pytest.raises(ValidationError, match="Specify either tddft_nroots.*or tddft_iroot.*not both"):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                run_tddft=True,
                tddft_nroots=5,
                tddft_iroot=1,
            )

    @pytest.mark.parametrize("invalid_nroots", [0, -1])
    def test_tddft_invalid_nroots_validation(self, invalid_nroots: int) -> None:
        """Test validation fails for non-positive tddft_nroots."""
        with pytest.raises(ValidationError, match="tddft_nroots must be a positive integer"):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                run_tddft=True,
                tddft_nroots=invalid_nroots,
            )

    @pytest.mark.parametrize("invalid_iroot", [0, -1])
    def test_tddft_invalid_iroot_validation(self, invalid_iroot: int) -> None:
        """Test validation fails for non-positive tddft_iroot."""
        with pytest.raises(ValidationError, match="tddft_iroot must be a positive integer"):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                run_tddft=True,
                tddft_nroots=None,  # Explicitly set nroots to None
                tddft_iroot=invalid_iroot,
            )

    def test_geometry_def2svp_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning for using def2-svp with task='geometry'."""
        OrcaInput(
            task="geometry",
            level_of_theory="hf",
            basis_set="def2-svp",  # Basis set that triggers warning
            charge=0,
            spin_multiplicity=1,
        )
        assert "Using def2-svp for geometry optimization might yield less accurate results" in caplog.text

    # --- Solvation Validation Tests ---
    def test_init_requires_solvent_with_model(self) -> None:
        """Test validation fails if implicit_solvation_model is set without solvent during init."""
        with pytest.raises(
            ValidationError, match="Both `implicit_solvation_model` and `solvent` must be provided together"
        ):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                implicit_solvation_model="cpcm",  # Model provided
                solvent=None,  # Solvent missing
            )

    def test_init_requires_model_with_solvent(self) -> None:
        """Test validation fails if solvent is set without implicit_solvation_model during init."""
        with pytest.raises(
            ValidationError, match="Both `implicit_solvation_model` and `solvent` must be provided together"
        ):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                implicit_solvation_model=None,  # Model missing
                solvent="water",  # Solvent provided
            )

    def test_init_valid_orca_solvation_models(self) -> None:
        """Test successful initialization with supported ORCA solvation models."""
        instance_cpcm = OrcaInput(
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            charge=0,
            spin_multiplicity=1,
            implicit_solvation_model="cpcm",
            solvent="water",
        )
        assert instance_cpcm.implicit_solvation_model == "cpcm"
        assert instance_cpcm.solvent == "water"

        instance_smd = OrcaInput(
            task="energy",
            level_of_theory="hf",
            basis_set="sto-3g",
            charge=0,
            spin_multiplicity=1,
            implicit_solvation_model="smd",
            solvent="toluene",
        )
        assert instance_smd.implicit_solvation_model == "smd"
        assert instance_smd.solvent == "toluene"

    def test_init_unsupported_base_solvation_model_error(self) -> None:
        """Test NotSupportedError for a model valid in base class but not ORCA."""
        with pytest.raises(NotSupportedError, match="ORCA's primary solvation models are CPCM and SMD"):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                implicit_solvation_model="pcm",  # type: ignore[arg-type] # Intentionally invalid for ORCA validation
                solvent="water",
            )

    # --- Hessian Recalculation and Optimize Hydrogens Init Validation ---
    def test_init_recalc_hess_freq_wrong_task(self) -> None:
        """Test __post_init__ validation for recalc_hess_freq with non-geometry task."""
        with pytest.raises(ValidationError, match="Hessian recalculation .* only applicable for 'geometry' tasks"):
            OrcaInput(
                task="energy",  # Non-geometry task
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                recalc_hess_freq=10,
            )

    @pytest.mark.parametrize("invalid_freq", [0, -1])
    def test_init_recalc_hess_freq_invalid_value(self, invalid_freq: int) -> None:
        """Test __post_init__ validation for non-positive recalc_hess_freq."""
        with pytest.raises(ValidationError, match="recalc_hess_freq must be a positive integer"):
            OrcaInput(
                task="geometry",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                recalc_hess_freq=invalid_freq,
            )

    def test_init_optimize_hydrogens_only_wrong_task(self) -> None:
        """Test __post_init__ validation for optimize_hydrogens_only with non-geometry task."""
        with pytest.raises(ValidationError, match="Optimizing only hydrogens is only applicable for 'geometry' tasks"):
            OrcaInput(
                task="energy",  # Non-geometry task
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                optimize_hydrogens_only=True,
            )

    # --- Frequency Calculation Validation Tests ---
    def test_init_frequency_task_with_run_frequency_flag_error(self) -> None:
        """Test __post_init__ validation fails when both task='frequency' and run_frequency=True."""
        with pytest.raises(
            ValidationError,
            match="Cannot set both task='frequency' and run_frequency=True. Use either task='frequency' for frequency-only calculations or run_frequency=True to combine with other tasks.",
        ):
            OrcaInput(
                task="frequency",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                run_frequency=True,
            )

    def test_init_invalid_basis_set_type_error(self) -> None:
        """Test __post_init__ validation fails for non-string basis_set types."""
        with pytest.raises(ValidationError, match="basis_set must be a string, got"):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set=123,  # type: ignore[arg-type]
                charge=0,
                spin_multiplicity=1,
            )

    def test_init_invalid_aux_basis_type_error(self) -> None:
        """Test __post_init__ validation fails for non-string aux_basis when ri_approx is set."""
        with pytest.raises(ValidationError, match="aux_basis must be a string when ri_approx is set, got"):
            OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set="sto-3g",
                charge=0,
                spin_multiplicity=1,
                ri_approx="RIJCOSX",
                aux_basis=456,  # type: ignore[arg-type]
            )


class TestOrcaInputMethods:
    """Tests for methods modifying OrcaInput instances."""

    def test_enable_ri(self, minimal_orca_input: OrcaInput) -> None:
        """Test the enable_ri method."""
        modified_input = minimal_orca_input.enable_ri(approx="RIJCOSX", aux_basis="def2/J")
        assert modified_input is not minimal_orca_input  # Ensure immutability
        assert modified_input.ri_approx == "RIJCOSX"
        assert modified_input.aux_basis == "def2/J"
        # Check other fields remain unchanged
        assert modified_input.level_of_theory == minimal_orca_input.level_of_theory
        assert not modified_input.run_tddft

    def test_enable_print_mos(self, minimal_orca_input: OrcaInput) -> None:
        """Test the enable_print_mos method."""
        modified_input = minimal_orca_input.enable_print_mos()
        assert modified_input is not minimal_orca_input
        assert modified_input.print_mos
        # Check other fields remain unchanged
        assert modified_input.level_of_theory == minimal_orca_input.level_of_theory
        assert modified_input.n_cores == minimal_orca_input.n_cores

    def test_set_tddft_defaults(self, minimal_orca_input: OrcaInput) -> None:
        """Test set_tddft raises ValidationError if nroots and iroot are None."""
        with pytest.raises(ValidationError, match="Either nroots or iroot must be specified"):
            minimal_orca_input.set_tddft()

    def test_set_tddft_with_nroots(self, minimal_orca_input: OrcaInput) -> None:
        """Test set_tddft specifying nroots."""
        modified_input = minimal_orca_input.set_tddft(nroots=10, triplets=True, use_tda=False)
        assert modified_input.run_tddft
        assert modified_input.tddft_nroots == 10
        assert modified_input.tddft_iroot is None
        assert modified_input.tddft_triplets
        assert not modified_input.tddft_use_tda

    def test_set_tddft_with_iroot(self, minimal_orca_input: OrcaInput) -> None:
        """Test set_tddft specifying iroot."""
        modified_input = minimal_orca_input.set_tddft(iroot=3, triplets=False, use_tda=True)
        assert modified_input.run_tddft
        assert modified_input.tddft_nroots is None  # nroots should be None if iroot specified
        assert modified_input.tddft_iroot == 3
        assert not modified_input.tddft_triplets
        assert modified_input.tddft_use_tda

    def test_set_tddft_overwrites_previous(self, minimal_orca_input: OrcaInput) -> None:
        """Test that subsequent calls to set_tddft overwrite previous settings."""
        input1 = minimal_orca_input.set_tddft(nroots=5, use_tda=False)  # Explicitly set to False for change
        input2 = input1.set_tddft(iroot=2, triplets=True, use_tda=True)  # Default TDA

        assert input2.run_tddft
        assert input2.tddft_nroots is None
        assert input2.tddft_iroot == 2
        assert input2.tddft_triplets
        assert input2.tddft_use_tda  # Default use_tda

    # Note: Validation for conflicting nroots/iroot happens in __post_init__,
    # not directly in set_tddft, so we don't test that specific error here.

    def test_set_hessian_recalculation_valid(self, minimal_orca_input: OrcaInput) -> None:
        """Test set_hessian_recalculation with valid frequency on a geometry task."""
        geom_input = replace(minimal_orca_input, task="geometry")
        modified_input = geom_input.set_hessian_recalculation(frequency=10)
        assert modified_input is not geom_input
        assert modified_input.recalc_hess_freq == 10
        assert modified_input.task == "geometry"

    def test_set_hessian_recalculation_wrong_task(self, minimal_orca_input: OrcaInput) -> None:
        """Test set_hessian_recalculation raises error if task is not geometry."""
        assert minimal_orca_input.task == "energy"  # Precondition
        with pytest.raises(ValidationError, match="Hessian recalculation is only applicable for 'geometry' tasks"):
            minimal_orca_input.set_hessian_recalculation(frequency=10)

    @pytest.mark.parametrize("invalid_freq", [0, -5])
    def test_set_hessian_recalculation_invalid_frequency(
        self, minimal_orca_input: OrcaInput, invalid_freq: int
    ) -> None:
        """Test set_hessian_recalculation raises error for non-positive frequency."""
        geom_input = replace(minimal_orca_input, task="geometry")
        with pytest.raises(ValidationError, match="Hessian recalculation frequency must be a positive integer"):
            geom_input.set_hessian_recalculation(frequency=invalid_freq)

    def test_enable_optimize_hydrogens_only_valid(self, minimal_orca_input: OrcaInput) -> None:
        """Test enable_optimize_hydrogens_only on a geometry task."""
        geom_input = replace(minimal_orca_input, task="geometry")
        modified_input = geom_input.enable_optimize_hydrogens_only()
        assert modified_input is not geom_input
        assert modified_input.optimize_hydrogens_only
        assert modified_input.task == "geometry"

    def test_enable_optimize_hydrogens_only_wrong_task(self, minimal_orca_input: OrcaInput) -> None:
        """Test enable_optimize_hydrogens_only raises error if task is not geometry."""
        assert minimal_orca_input.task == "energy"  # Precondition
        with pytest.raises(ValidationError, match="Optimizing only hydrogens is only applicable for 'geometry' tasks"):
            minimal_orca_input.enable_optimize_hydrogens_only()

    def test_copy_method(self, minimal_orca_input: OrcaInput) -> None:
        """Test the copy method for deep copying."""
        original_input = replace(minimal_orca_input, ri_approx="RI", aux_basis="def2/J")  # Add a mutable-like field
        copied_input = original_input.copy()

        assert copied_input is not original_input
        assert copied_input == original_input  # Dataclass equality should hold

        # Test deepcopy by attempting to modify a field (even if immutable here, it's good practice for dicts)
        # If basis_set could be a dict: original_input.basis_set["new_key"] = "new_value"
        # For now, let's show they are independent by replacing one
        modified_original = replace(original_input, n_cores=100)
        assert copied_input.n_cores == original_input.n_cores  # Copied should still have original n_cores
        assert modified_original.n_cores == 100

        # More direct test for a mutable attribute if we had one that's commonly a dict.
        # For example, if 'basis_set' was guaranteed to be a dict for this test:
        # original_input_with_dict_basis = replace(minimal_orca_input, basis_set={"C": "sto-3g"})
        # copied_with_dict_basis = original_input_with_dict_basis.copy()
        # copied_with_dict_basis.basis_set["H"] = "6-31g"
        # assert "H" not in original_input_with_dict_basis.basis_set # Check original is untouched

    def test_enable_rijcosx_valid(self, minimal_orca_input: OrcaInput) -> None:
        """Test enable_rijcosx with a valid auxiliary basis."""
        modified_input = minimal_orca_input.enable_ri(approx="RIJCOSX", aux_basis="def2/J")
        assert modified_input is not minimal_orca_input
        assert modified_input.ri_approx == "RIJCOSX"
        assert modified_input.aux_basis == "def2/J"

    def test_enable_ri_invalid_approximation_error(self, minimal_orca_input: OrcaInput) -> None:
        """Test enable_ri raises ValidationError for unrecognized RI approximation."""
        with pytest.raises(ValidationError, match="RI approximation 'INVALID_RI' not recognized for ORCA. Allowed:"):
            minimal_orca_input.enable_ri(approx="INVALID_RI", aux_basis="def2/J")

    def test_enable_frequency_valid(self, minimal_orca_input: OrcaInput) -> None:
        """Test enable_frequency method with valid task."""
        modified_input = minimal_orca_input.enable_frequency()
        assert modified_input is not minimal_orca_input
        assert modified_input.run_frequency
        assert modified_input.task == minimal_orca_input.task  # Original task preserved
        assert not minimal_orca_input.run_frequency  # Original unchanged

    def test_enable_frequency_with_geometry_task(self, minimal_orca_input: OrcaInput) -> None:
        """Test enable_frequency with geometry task."""
        geom_input = replace(minimal_orca_input, task="geometry")
        modified_input = geom_input.enable_frequency()
        assert modified_input.run_frequency
        assert modified_input.task == "geometry"

    def test_enable_frequency_already_frequency_task_error(self, minimal_orca_input: OrcaInput) -> None:
        """Test enable_frequency raises error when task is already 'frequency'."""
        freq_input = replace(minimal_orca_input, task="frequency")
        with pytest.raises(
            ValidationError, match="Task is already 'frequency'. Use task='frequency' for frequency-only calculations."
        ):
            freq_input.enable_frequency()


class TestOrcaInputSetSolvation:
    """Tests for the set_solvation method."""

    @pytest.mark.parametrize(
        "model, solvent, expected_model, expected_solvent",
        [
            ("cpcm", "water", "cpcm", "water"),
            ("smd", "ethanol", "smd", "ethanol"),
            ("CPCM", "WATER", "cpcm", "water"),  # Test case normalization
            ("SmD", "eThAnOl", "smd", "ethanol"),  # Test case normalization
        ],
    )
    def test_set_solvation_add_valid(
        self, minimal_orca_input: OrcaInput, model: str, solvent: str, expected_model: str, expected_solvent: str
    ) -> None:
        """Test adding valid ORCA solvation models using set_solvation."""
        solvated = minimal_orca_input.set_solvation(model, solvent)
        assert solvated.implicit_solvation_model == expected_model
        assert solvated.solvent == expected_solvent
        assert solvated is not minimal_orca_input  # Ensure immutability

        # Ensure other fields are unchanged
        assert solvated.task == minimal_orca_input.task
        assert solvated.level_of_theory == minimal_orca_input.level_of_theory

    def test_set_solvation_remove(self, minimal_orca_input: OrcaInput) -> None:
        """Test removing solvation using set_solvation(None, None)."""
        # First add solvation, then remove it
        solvated = minimal_orca_input.set_solvation("cpcm", "water")
        dry = solvated.set_solvation(None, None)

        assert dry.implicit_solvation_model is None
        assert dry.solvent is None
        assert dry is not solvated  # Ensure immutability
        # Ensure other fields are unchanged
        assert dry.task == minimal_orca_input.task

    def test_set_solvation_requires_both_or_neither(self, minimal_orca_input: OrcaInput) -> None:
        """Test set_solvation raises ValidationError if model/solvent are inconsistent."""
        with pytest.raises(ValidationError, match="Both `model` and `solvent` must be provided together"):
            minimal_orca_input.set_solvation("cpcm", None)
        with pytest.raises(ValidationError, match="Both `model` and `solvent` must be provided together"):
            minimal_orca_input.set_solvation(None, "water")

    def test_set_solvation_unsupported_model_error(self, minimal_orca_input: OrcaInput) -> None:
        """Test set_solvation raises ValidationError for unsupported ORCA models."""
        with pytest.raises(ValidationError, match="Solvation model 'pcm' not recognized for ORCA"):
            minimal_orca_input.set_solvation("pcm", "water")  # type: ignore[arg-type] # Testing invalid model string
        with pytest.raises(ValidationError, match="Solvation model 'invalid' not recognized for ORCA"):
            minimal_orca_input.set_solvation("invalid", "water")  # type: ignore[arg-type] # Testing invalid model string


class TestOrcaInputExport:
    """Tests for the export_input_file method."""

    def _assert_keywords_match(self, output: str, expected_keywords_set: set[str]) -> None:
        """Helper to assert that the keyword line matches the expected set (order independent)."""
        actual_lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        actual_keywords_line = ""
        for line in actual_lines:
            if line.startswith("!"):
                actual_keywords_line = line
                break
        assert actual_keywords_line, "Keyword line starting with '!' not found in output"
        actual_keywords_set = set(actual_keywords_line.split())
        assert actual_keywords_set == expected_keywords_set, (
            f"Keyword mismatch: Expected {expected_keywords_set}, Got {actual_keywords_set}"
        )

    def test_export_minimal_energy(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export for a minimal single-point energy calculation."""
        output = minimal_orca_input.export_input_file(default_geom)
        actual_lines = [line.strip() for line in output.strip().split("\n") if line.strip()]

        # --- Keyword Line Check (Order Independent) ---
        expected_keywords = {"!", "RHF", "sto-3g", "SP"}
        actual_keywords_line = ""
        for line in actual_lines:
            if line.startswith("!"):
                actual_keywords_line = line
                break
        assert actual_keywords_line, "Keyword line starting with '!' not found in output"
        actual_keywords_set = set(actual_keywords_line.split())
        assert actual_keywords_set == expected_keywords, (
            f"Keyword mismatch: Expected {expected_keywords}, Got {actual_keywords_set}"
        )

        # --- Other Lines Check (Order Dependent) ---
        expected_other_lines = [
            "%maxcore 2000",  # Correct default memory
            "* xyz 0 1",
            # Add individual lines from the geometry
            *default_geom.get_coordinate_block().strip().split("\n"),
            "*",
        ]
        # Filter out the keyword line from actual_lines for this check
        actual_other_lines = [line for line in actual_lines if not line.startswith("!")]

        assert actual_other_lines == expected_other_lines, "Mismatch in non-keyword lines or their order"

        # --- Block Presence Check --- Explicitly check private methods return empty strings
        assert minimal_orca_input._get_procs_line() == ""  # Line 264
        assert minimal_orca_input._get_solvent_block() == ""  # Line 274
        assert minimal_orca_input._get_tddft_block() == ""  # Line 309
        assert minimal_orca_input._get_output_block() == ""  # Line 338
        assert "%pal" not in output  # n_cores=1
        assert "%cpcm" not in output
        assert "%tddft" not in output
        assert "%output" not in output

        # Check the final '*' exists after the geometry
        assert output.strip().endswith("*")

    def test_export_geometry_optimization(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export for a geometry optimization."""
        geom_opt_input = replace(minimal_orca_input, task="geometry")
        output = geom_opt_input.export_input_file(default_geom)

        # --- Keyword Line Check (Order Independent) ---
        expected_keywords = {"!", "RHF", "sto-3g", "Opt"}
        actual_lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        actual_keywords_line = ""
        for line in actual_lines:
            if line.startswith("!"):
                actual_keywords_line = line
                break
        assert actual_keywords_line, "Keyword line starting with '!' not found in output"
        actual_keywords_set = set(actual_keywords_line.split())
        assert actual_keywords_set == expected_keywords, (
            f"Keyword mismatch: Expected {expected_keywords}, Got {actual_keywords_set}"
        )

        # --- Geometry Block Check ---
        assert "* xyz 0 1" in output
        # Ensure all lines of the geometry are present in the output
        geom_lines = default_geom.get_coordinate_block().strip().split("\n")
        assert all(line in output for line in geom_lines)
        # Check the final '*' exists after the geometry
        assert output.strip().endswith("*")

    @pytest.mark.parametrize(
        "level_of_theory, unrestricted, expected_keywords_str",
        [
            ("hf", False, "! RHF sto-3g SP"),
            ("hf", True, "! UHF sto-3g SP"),
            ("rhf", False, "! RHF sto-3g SP"),
            ("uhf", True, "! UHF sto-3g SP"),
            ("b3lyp", False, "! RKS b3lyp sto-3g SP"),
            ("b3lyp", True, "! UKS b3lyp sto-3g SP"),
            ("pbe0", False, "! RKS pbe0 sto-3g SP"),
            ("mp2", False, "! MP2 sto-3g SP"),
            ("mp2", True, "! UMP2 sto-3g SP"),
            ("ri-mp2", False, "! RI-MP2 sto-3g SP"),
            ("ri-mp2", True, "! RI-UMP2 sto-3g SP"),
            ("ccsd(t)", True, "! CCSD(T) sto-3g SP"),  # Assume unrestricted OK here
            ("ccsd", False, "! CCSD sto-3g SP"),
        ],
    )
    def test_export_level_of_theory_variants(
        self,
        minimal_orca_input: OrcaInput,
        default_geom: Geometry,
        level_of_theory: str,
        unrestricted: bool,
        expected_keywords_str: str,
    ) -> None:
        """Test various levels of theory and unrestricted flag handling."""
        test_input = replace(
            minimal_orca_input,
            level_of_theory=level_of_theory,
            unrestricted=unrestricted,
        )
        output = test_input.export_input_file(default_geom)
        expected_keywords_set = set(expected_keywords_str.split())
        self._assert_keywords_match(output, expected_keywords_set)
        # Add basic geometry check for completeness
        assert "* xyz 0 1" in output

    @pytest.mark.parametrize(
        "level_of_theory, unrestricted, warning_msg",
        [
            ("uhf", False, "Requested method UHF but unrestricted was not set to True"),
            ("ump2", False, "Requested method UMP2 but unrestricted was not set to True"),
            ("ccsd(t)", False, "Requested method CCSD(T) but unrestricted was not set to True"),
        ],
    )
    def test_export_level_of_theory_warnings(
        self,
        minimal_orca_input: OrcaInput,
        default_geom: Geometry,
        level_of_theory: str,
        unrestricted: bool,
        warning_msg: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warnings for specific level_of_theory/unrestricted combinations."""
        caplog.set_level(logging.WARNING)  # Ensure warnings are captured
        test_input = replace(
            minimal_orca_input,
            level_of_theory=level_of_theory,
            unrestricted=unrestricted,
        )
        test_input.export_input_file(default_geom)
        assert warning_msg in caplog.text

    @pytest.mark.parametrize(
        "level_of_theory, unrestricted, error_match",
        [
            ("rhf", True, "Requested method RHF but unrestricted was set to True"),
            ("rks", False, "specify a functional as the level_of_theory"),
            ("uks", True, "specify a functional as the level_of_theory"),
            ("bad_method", False, "Unsupported or unrecognized level_of_theory:.*bad_method"),  # Corrected regex
        ],
    )
    def test_export_invalid_level_of_theory(
        self,
        minimal_orca_input: OrcaInput,
        default_geom: Geometry,
        level_of_theory: str,
        unrestricted: bool,
        error_match: str,
    ) -> None:
        """Test invalid combinations of level_of_theory and unrestricted flag."""
        test_input = replace(
            minimal_orca_input,
            level_of_theory=level_of_theory,
            unrestricted=unrestricted,
        )
        with pytest.raises(ValidationError, match=error_match):
            test_input.export_input_file(default_geom)

    def test_export_unrecognized_hf_method_error(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export raises ValidationError for unrecognized HF method variants."""
        test_input = replace(minimal_orca_input, level_of_theory="invalid_hf")
        with pytest.raises(ValidationError, match="Unrecognized HF method: invalid_hf"):
            test_input.export_input_file(default_geom)

    def test_export_unrecognized_mp2_method_error(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export raises ValidationError for unrecognized MP2 method variants."""
        test_input = replace(minimal_orca_input, level_of_theory="invalid_mp2")
        with pytest.raises(ValidationError, match="Unrecognized MP2 method: invalid_mp2"):
            test_input.export_input_file(default_geom)

    def test_export_ccsd_unrestricted_error(
        self,
        minimal_orca_input: OrcaInput,
        default_geom: Geometry,
    ) -> None:
        """Test error for CCSD with unrestricted=True."""
        test_input = replace(
            minimal_orca_input,
            level_of_theory="ccsd",
            unrestricted=True,
        )
        with pytest.raises(ValidationError, match="Requested method CCSD but unrestricted was set to True."):
            test_input.export_input_file(default_geom)

    def test_export_frequency_only_task(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export for frequency-only calculation."""
        freq_input = replace(minimal_orca_input, task="frequency")
        output = freq_input.export_input_file(default_geom)
        expected_keywords_set = {"!", "RHF", "sto-3g", "Freq"}
        self._assert_keywords_match(output, expected_keywords_set)
        assert "* xyz 0 1" in output

    def test_export_geometry_with_frequency(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export for geometry optimization with frequency calculation."""
        geom_freq_input = replace(minimal_orca_input, task="geometry").enable_frequency()
        output = geom_freq_input.export_input_file(default_geom)
        expected_keywords_set = {"!", "RHF", "sto-3g", "Opt", "Freq"}
        self._assert_keywords_match(output, expected_keywords_set)
        assert "* xyz 0 1" in output

    def test_export_energy_with_frequency(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export for single-point energy with frequency calculation."""
        energy_freq_input = minimal_orca_input.enable_frequency()
        output = energy_freq_input.export_input_file(default_geom)
        expected_keywords_set = {"!", "RHF", "sto-3g", "SP", "Freq"}
        self._assert_keywords_match(output, expected_keywords_set)
        assert "* xyz 0 1" in output

    @pytest.mark.parametrize(
        "task, run_frequency, expected_keywords",
        [
            ("energy", False, {"!", "RHF", "sto-3g", "SP"}),
            ("energy", True, {"!", "RHF", "sto-3g", "SP", "Freq"}),
            ("geometry", False, {"!", "RHF", "sto-3g", "Opt"}),
            ("geometry", True, {"!", "RHF", "sto-3g", "Opt", "Freq"}),
            ("frequency", False, {"!", "RHF", "sto-3g", "Freq"}),
        ],
    )
    def test_export_frequency_combinations(
        self,
        minimal_orca_input: OrcaInput,
        default_geom: Geometry,
        task: str,
        run_frequency: bool,
        expected_keywords: set[str],
    ) -> None:
        """Test various combinations of task and run_frequency flag."""
        test_input = replace(minimal_orca_input, task=task, run_frequency=run_frequency)  # type: ignore[arg-type]
        output = test_input.export_input_file(default_geom)
        self._assert_keywords_match(output, expected_keywords)

    def test_export_with_ri(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export with RI approximation enabled."""
        ri_input = minimal_orca_input.enable_ri(approx="RIJCOSX", aux_basis="def2/J")
        output = ri_input.export_input_file(default_geom)
        expected_keywords_set = {"!", "RHF", "sto-3g", "SP", "RIJCOSX", "def2/J"}
        self._assert_keywords_match(output, expected_keywords_set)

    def test_export_with_procs_and_mem(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export with multiple processors and custom memory."""
        mem_proc_input = replace(minimal_orca_input, n_cores=8, memory_per_core_mb=2000)
        output = mem_proc_input.export_input_file(default_geom)
        assert "%pal nprocs 8 end" in output
        assert "%maxcore 2000" in output

    def test_export_with_cpcm_solvation(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export with CPCM implicit solvation."""
        cpcm_input = replace(minimal_orca_input, implicit_solvation_model="cpcm", solvent="water")
        output = cpcm_input.export_input_file(default_geom)
        assert 'CPCM("water")' in output
        assert "%cpcm" not in output  # No separate block for CPCM keyword

    def test_export_with_smd_solvation(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export with SMD implicit solvation (requires %cpcm block)."""
        smd_input = replace(minimal_orca_input, implicit_solvation_model="smd", solvent="toluene")
        output = smd_input.export_input_file(default_geom)
        expected_keywords_set = {"!", "RHF", "sto-3g", "SP"}  # SMD is not added to keywords line
        self._assert_keywords_match(output, expected_keywords_set)
        expected_block = """%cpcm
    smd true
    SMDsolvent "toluene"
end"""
        assert expected_block in output

    def test_export_smd_missing_solvent_error(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export fails if SMD is used without specifying a solvent."""
        # Error should be raised during initialization via replace due to base class validation
        with pytest.raises(
            ValidationError, match="Both `implicit_solvation_model` and `solvent` must be provided together"
        ):
            replace(minimal_orca_input, implicit_solvation_model="smd", solvent=None)

    def test_export_with_tddft_nroots(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export with TDDFT enabled using nroots."""
        # Expect TDA False here
        tddft_input = minimal_orca_input.set_tddft(nroots=5, triplets=True, use_tda=False)
        # We need to re-validate after replace/set_tddft for the export checks
        tddft_input_validated = replace(tddft_input)
        output = tddft_input_validated.export_input_file(default_geom)
        expected_block = """%tddft
    NRoots 5
    Triplets true
    TDA false
end"""
        assert expected_block in output

    def test_export_with_tddft_iroot(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export with TDDFT enabled using iroot."""
        # Expect TDA True (default)
        tddft_input = minimal_orca_input.set_tddft(iroot=2, triplets=False, use_tda=True)
        # Re-validate after replace/set_tddft
        tddft_input_validated = replace(tddft_input)
        output = tddft_input_validated.export_input_file(default_geom)
        expected_block = """%tddft
    NRoots 1
    IRoot 2
    Triplets false
    TDA true
end"""
        assert expected_block in output

    def test_export_with_print_mos(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export with MO printing enabled."""
        print_mo_input = minimal_orca_input.enable_print_mos()
        # Re-validate after replace/enable_print_mos
        print_mo_input_validated = replace(print_mo_input)
        output = print_mo_input_validated.export_input_file(default_geom)
        expected_block = """%output
    Print[ P_MOs ] 1
    Print[ P_Overlap ] 1
end"""
        # Check if the expected block is present in the output string
        assert expected_block in output
        # Explicitly check the return value of the private method
        assert print_mo_input_validated._get_output_block() == expected_block  # Line 338

    def test_export_full_example(self, default_geom: Geometry) -> None:
        """Test export with multiple features enabled simultaneously."""
        full_input = OrcaInput(
            task="geometry",
            level_of_theory="cam-b3lyp",
            basis_set="def2-tzvp",
            charge=1,
            spin_multiplicity=2,
            unrestricted=True,
            n_cores=16,
            memory_per_core_mb=4000,
            ri_approx="RIJCOSX",
            aux_basis="def2/J",
            implicit_solvation_model="smd",
            solvent="chcl3",
            run_tddft=True,
            tddft_iroot=3,
            tddft_triplets=True,
            tddft_use_tda=True,  # TDA is true
            print_mos=True,
        )
        output = full_input.export_input_file(default_geom)

        # Check keywords line using helper
        expected_keywords_set = {"!", "UKS", "cam-b3lyp", "def2-tzvp", "Opt", "RIJCOSX", "def2/J"}
        self._assert_keywords_match(output, expected_keywords_set)

        # Check blocks presence
        assert "%pal nprocs 16 end" in output
        assert "%maxcore 4000" in output
        assert 'SMDsolvent "chcl3"' in output
        assert "%tddft" in output
        assert "IRoot 3" in output
        assert "Triplets true" in output
        assert "TDA true" in output
        assert "%output" in output
        assert "Print[ P_MOs ] 1" in output
        # Check geometry section
        assert f"* xyz {full_input.charge} {full_input.spin_multiplicity}" in output
        assert default_geom.get_coordinate_block().strip() in output
        assert output.strip().endswith("*")

    def test_export_geom_recalc_hess(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export with Hessian recalculation enabled."""
        geom_input = replace(minimal_orca_input, task="geometry")
        hess_input = geom_input.set_hessian_recalculation(frequency=15)
        output = hess_input.export_input_file(default_geom)
        expected_geom_block = """%geom
    Calc_Hess true
    Recalc_Hess 15
end"""
        assert expected_geom_block in output
        # Ensure Opt keyword is still present
        expected_keywords_set = {"!", "RHF", "sto-3g", "Opt"}
        self._assert_keywords_match(output, expected_keywords_set)

    def test_export_geom_optimize_hydrogens(self, minimal_orca_input: OrcaInput, default_geom: Geometry) -> None:
        """Test export with optimize_hydrogens_only enabled."""
        geom_input = replace(minimal_orca_input, task="geometry")
        hydro_opt_input = geom_input.enable_optimize_hydrogens_only()
        output = hydro_opt_input.export_input_file(default_geom)
        expected_geom_block = """%geom
    OptimizeHydrogens true
end"""
        assert expected_geom_block in output
        # Ensure Opt keyword is still present
        expected_keywords_set = {"!", "RHF", "sto-3g", "Opt"}
        self._assert_keywords_match(output, expected_keywords_set)

    def test_export_geom_recalc_hess_and_optimize_hydrogens(
        self, minimal_orca_input: OrcaInput, default_geom: Geometry
    ) -> None:
        """Test export with both Hessian recalculation and optimize_hydrogens_only enabled."""
        geom_input = replace(minimal_orca_input, task="geometry")
        combined_input = geom_input.set_hessian_recalculation(frequency=20).enable_optimize_hydrogens_only()
        output = combined_input.export_input_file(default_geom)
        # Order within the block might vary if not enforced, but content should be there.
        # For now, let's check for individual lines if order is not guaranteed by _get_geom_block for multiple options.
        # However, current _get_geom_block has a fixed order.
        expected_geom_block = """%geom
    Calc_Hess true
    Recalc_Hess 20
    OptimizeHydrogens true
end"""
        assert expected_geom_block in output
        expected_keywords_set = {"!", "RHF", "sto-3g", "Opt"}
        self._assert_keywords_match(output, expected_keywords_set)


# --- Add more test classes below ---

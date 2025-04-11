from dataclasses import replace

import pytest

from calcflow.exceptions import NotSupportedError, ValidationError
from calcflow.inputs.orca import OrcaInput


@pytest.fixture
def default_geom() -> str:
    """Provides a default geometry string."""
    return """O 0.0 0.0 0.0
H 0.0 0.0 1.0
H 0.0 1.0 0.0"""


@pytest.fixture
def minimal_orca_input(default_geom: str) -> OrcaInput:
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
        assert minimal_orca_input.tddft_method == "TDDFT"

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
            tddft_method="TDA",
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
        assert instance.tddft_method == "TDA"
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
        """Test set_tddft with default arguments (should set nroots=5)."""
        modified_input = minimal_orca_input.set_tddft()
        assert modified_input is not minimal_orca_input
        assert modified_input.run_tddft
        assert modified_input.tddft_nroots == 5  # Default when nroots/iroot are None
        assert modified_input.tddft_iroot is None
        assert not modified_input.tddft_triplets
        assert modified_input.tddft_method == "TDA"  # Default method in set_tddft

    def test_set_tddft_with_nroots(self, minimal_orca_input: OrcaInput) -> None:
        """Test set_tddft specifying nroots."""
        modified_input = minimal_orca_input.set_tddft(nroots=10, triplets=True, method="TDDFT")
        assert modified_input.run_tddft
        assert modified_input.tddft_nroots == 10
        assert modified_input.tddft_iroot is None
        assert modified_input.tddft_triplets
        assert modified_input.tddft_method == "TDDFT"

    def test_set_tddft_with_iroot(self, minimal_orca_input: OrcaInput) -> None:
        """Test set_tddft specifying iroot."""
        modified_input = minimal_orca_input.set_tddft(iroot=3, triplets=False, method="TDA")
        assert modified_input.run_tddft
        assert modified_input.tddft_nroots is None  # nroots should be None if iroot specified
        assert modified_input.tddft_iroot == 3
        assert not modified_input.tddft_triplets
        assert modified_input.tddft_method == "TDA"

    def test_set_tddft_overwrites_previous(self, minimal_orca_input: OrcaInput) -> None:
        """Test that subsequent calls to set_tddft overwrite previous settings."""
        input1 = minimal_orca_input.set_tddft(nroots=5)
        input2 = input1.set_tddft(iroot=2, triplets=True)

        assert input2.run_tddft
        assert input2.tddft_nroots is None
        assert input2.tddft_iroot == 2
        assert input2.tddft_triplets
        assert input2.tddft_method == "TDA"  # Default method

    # Note: Validation for conflicting nroots/iroot happens in __post_init__,
    # not directly in set_tddft, so we don't test that specific error here.


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

    def test_export_minimal_energy(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
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
            *default_geom.strip().split("\n"),
            "*",
        ]
        # Filter out the keyword line from actual_lines for this check
        actual_other_lines = [line for line in actual_lines if not line.startswith("!")]

        assert actual_other_lines == expected_other_lines, "Mismatch in non-keyword lines or their order"

        # --- Block Presence Check ---
        assert "%pal" not in output  # n_cores=1
        assert "%cpcm" not in output
        assert "%tddft" not in output
        assert "%output" not in output

        # Check the final '*' exists after the geometry
        assert output.strip().endswith("*")

    def test_export_geometry_optimization(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
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
        geom_lines = default_geom.strip().split("\n")
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
            ("ccsd(t)", True, "! CCSD(T) sto-3g SP"),  # Assume unrestricted OK here
            ("ccsd", False, "! CCSD sto-3g SP"),
        ],
    )
    def test_export_level_of_theory_variants(
        self,
        minimal_orca_input: OrcaInput,
        default_geom: str,
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
        default_geom: str,
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

    def test_export_with_ri(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
        """Test export with RI approximation enabled."""
        ri_input = minimal_orca_input.enable_ri(approx="RIJCOSX", aux_basis="def2/J")
        output = ri_input.export_input_file(default_geom)
        expected_keywords_set = {"!", "RHF", "sto-3g", "SP", "RIJCOSX", "def2/J"}
        self._assert_keywords_match(output, expected_keywords_set)

    def test_export_with_procs_and_mem(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
        """Test export with multiple processors and custom memory."""
        mem_proc_input = replace(minimal_orca_input, n_cores=8, memory_per_core_mb=2000)
        output = mem_proc_input.export_input_file(default_geom)
        assert "%pal nprocs 8 end" in output
        assert "%maxcore 2000" in output

    def test_export_with_cpcm_solvation(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
        """Test export with CPCM implicit solvation."""
        cpcm_input = replace(minimal_orca_input, implicit_solvation_model="cpcm", solvent="water")
        output = cpcm_input.export_input_file(default_geom)
        assert 'CPCM("water")' in output
        assert "%cpcm" not in output  # No separate block for CPCM keyword

    def test_export_with_smd_solvation(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
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

    def test_export_smd_missing_solvent_error(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
        """Test export fails if SMD is used without specifying a solvent."""
        # Error should be raised during initialization via replace due to base class validation
        with pytest.raises(
            ValidationError, match="Both `implicit_solvation_model` and `solvent` must be provided together"
        ):
            replace(minimal_orca_input, implicit_solvation_model="smd", solvent=None)

    def test_export_with_tddft_nroots(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
        """Test export with TDDFT enabled using nroots."""
        tddft_input = minimal_orca_input.set_tddft(nroots=5, triplets=True, method="TDDFT")
        # We need to re-validate after replace/set_tddft for the export checks
        tddft_input_validated = replace(tddft_input)
        output = tddft_input_validated.export_input_file(default_geom)
        expected_block = """%tddft
    NRoots 5
    Triplets true
    Method TDDFT
end"""
        assert expected_block in output

    def test_export_with_tddft_iroot(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
        """Test export with TDDFT enabled using iroot."""
        tddft_input = minimal_orca_input.set_tddft(iroot=2, triplets=False, method="TDA")
        # Re-validate after replace/set_tddft
        tddft_input_validated = replace(tddft_input)
        output = tddft_input_validated.export_input_file(default_geom)
        expected_block = """%tddft
    NRoots 1
    IRoot 2
    Triplets false
    Method TDA
end"""
        assert expected_block in output

    def test_export_with_print_mos(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
        """Test export with MO printing enabled."""
        print_mo_input = minimal_orca_input.enable_print_mos()
        # Re-validate after replace/enable_print_mos
        print_mo_input_validated = replace(print_mo_input)
        output = print_mo_input_validated.export_input_file(default_geom)
        expected_block = """%output
    Print[ P_MOs ] 1
    Print[ P_Overlap ] 1
end"""
        assert expected_block in output

    def test_export_full_example(self, default_geom: str) -> None:
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
            tddft_method="TDA",
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
        assert "Method TDA" in output
        assert "%output" in output
        assert "Print[ P_MOs ] 1" in output
        # Check geometry section
        assert "* xyz 1 2" in output
        assert default_geom.strip() in output
        assert output.strip().endswith("*")

    def test_export_dict_basis_error(self, minimal_orca_input: OrcaInput, default_geom: str) -> None:
        """Test export raises error for dictionary basis sets."""
        with pytest.raises(
            NotSupportedError,
            match="Dictionary basis sets are not yet fully implemented in export_input_file for ORCA.",
        ):
            dict_basis_input = OrcaInput(
                task="energy",
                level_of_theory="hf",
                basis_set={"O": "def2-svp", "H": "sto-3g"},  # type: ignore
                charge=0,
                spin_multiplicity=1,
            )
            dict_basis_input.export_input_file(default_geom)


# --- Add more test classes below ---

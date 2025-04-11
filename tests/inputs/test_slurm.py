from dataclasses import FrozenInstanceError, replace

import pytest

from calcflow.inputs.slurm import SlurmArgs


# fmt:off
@pytest.fixture
def base_slurm_args() -> SlurmArgs:
    """Provides a basic SlurmArgs instance for testing."""
    return SlurmArgs(
        exec_fname="test_job",
        time="01:00:00",
        n_cores=4,
    )


def test_slurmargs_initialization_minimal(base_slurm_args: SlurmArgs) -> None:
    """Test basic initialization with required arguments."""
    assert base_slurm_args.exec_fname == "test_job"
    assert base_slurm_args.time == "01:00:00"
    assert base_slurm_args.n_cores == 4
    # Check defaults for optional args
    assert base_slurm_args.memory_mb is None
    assert base_slurm_args.software is None
    assert base_slurm_args.partition is None
    assert base_slurm_args.constraint is None
    assert base_slurm_args.account is None
    assert base_slurm_args.queue is None
    assert base_slurm_args.parallelism is None
    assert base_slurm_args.modules is None


def test_slurmargs_initialization_full() -> None:
    """Test initialization with all arguments."""
    args = SlurmArgs(
        exec_fname="full_job",
        time="10:00:00",
        n_cores=16,
        memory_mb=8000,
        software="orca", # Should be normalized to lower
        partition="compute",
        constraint="intel",
        account="test_acc",
        queue="high",
        parallelism="mpi",
        modules=["gcc/10.2.0", "openmpi/4.1.1"],
    )
    assert args.exec_fname == "full_job"
    assert args.time == "10:00:00"
    assert args.n_cores == 16
    assert args.memory_mb == 8000
    assert args.software == "orca"
    assert args.partition == "compute"
    assert args.constraint == "intel"
    assert args.account == "test_acc"
    assert args.queue == "high"
    assert args.parallelism == "mpi"
    # Note: Initial modules are used directly, software module is not added automatically on init
    assert args.modules == ["gcc/10.2.0", "openmpi/4.1.1"]


def test_slurmargs_immutability(base_slurm_args: SlurmArgs) -> None:
    """Test that SlurmArgs instances are immutable (frozen dataclass)."""
    with pytest.raises(FrozenInstanceError):
        base_slurm_args.n_cores = 8 # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        base_slurm_args.time = "02:00:00" # type: ignore[misc]


# --- Test Setters (Fluent Interface) ---

def test_set_software_valid(base_slurm_args: SlurmArgs) -> None:
    """Test setting valid software adds module and handles case."""
    args_orca = base_slurm_args.set_software("orca")
    assert args_orca.software == "orca"
    assert args_orca.modules == ["orca"], "Module should be added when none exist"
    assert args_orca is not base_slurm_args, "Setter must return a new instance"
    assert base_slurm_args.software is None, "Original instance should remain unchanged"
    assert base_slurm_args.modules is None, "Original instance should remain unchanged"

    args_qchem = base_slurm_args.set_software("QChem") # Test case insensitivity
    assert args_qchem.software == "qchem", "Software name should be lowercased"
    assert args_qchem.modules == ["qchem"], "Module should be added when none exist"
    assert args_qchem is not base_slurm_args, "Setter must return a new instance"

    # Test adding software when modules already exist
    args_with_modules = base_slurm_args.add_modules(["existing/module"])
    args_with_modules_and_orca = args_with_modules.set_software("orca")
    assert args_with_modules_and_orca.software == "orca"
    assert args_with_modules_and_orca.modules == ["existing/module", "orca"], "Software module should append"
    assert args_with_modules_and_orca is not args_with_modules, "Setter must return a new instance"


def test_set_software_invalid(base_slurm_args: SlurmArgs) -> None:
    """Test setting invalid software raises ValueError."""
    with pytest.raises(ValueError, match=r"Software must be one of \{'(orca|qchem)', '(qchem|orca)'\}"):
        base_slurm_args.set_software("invalid_software")


def test_set_time(base_slurm_args: SlurmArgs) -> None:
    """Test setting time."""
    new_time = "05:30:00"
    args = base_slurm_args.set_time(new_time)
    assert args.time == new_time
    assert args is not base_slurm_args, "Setter must return a new instance"
    assert base_slurm_args.time == "01:00:00", "Original instance should remain unchanged"


def test_set_memory(base_slurm_args: SlurmArgs, caplog: pytest.LogCaptureFixture) -> None:
    """Test setting memory, including low memory warning."""
    mem_mb = 4096
    args = base_slurm_args.set_memory(mem_mb)
    assert args.memory_mb == mem_mb
    assert args is not base_slurm_args, "Setter must return a new instance"
    assert base_slurm_args.memory_mb is None, "Original instance should remain unchanged"

    # Test low memory warning
    low_mem_mb = 128
    with caplog.at_level("WARNING", logger="calcflow"):
        args_low_mem = base_slurm_args.set_memory(low_mem_mb)
    assert args_low_mem.memory_mb == low_mem_mb
    assert "Memory allocation seems low (< 256 MB)" in caplog.text


def test_set_partition(base_slurm_args: SlurmArgs) -> None:
    """Test setting partition."""
    partition = "gpu"
    args = base_slurm_args.set_partition(partition)
    assert args.partition == partition
    assert args is not base_slurm_args, "Setter must return a new instance"
    assert base_slurm_args.partition is None, "Original instance should remain unchanged"


def test_set_constraint(base_slurm_args: SlurmArgs) -> None:
    """Test setting constraint."""
    constraint = "amd"
    args = base_slurm_args.set_constraint(constraint)
    assert args.constraint == constraint
    assert args is not base_slurm_args, "Setter must return a new instance"
    assert base_slurm_args.constraint is None, "Original instance should remain unchanged"


def test_set_queue(base_slurm_args: SlurmArgs) -> None:
    """Test setting queue."""
    queue = "debug"
    args = base_slurm_args.set_queue(queue)
    assert args.queue == queue
    assert args is not base_slurm_args, "Setter must return a new instance"
    assert base_slurm_args.queue is None, "Original instance should remain unchanged"


def test_set_parallelism_valid(base_slurm_args: SlurmArgs) -> None:
    """Test setting valid parallelism options."""
    args_mpi = base_slurm_args.set_parallelism("mpi")
    assert args_mpi.parallelism == "mpi"
    assert args_mpi is not base_slurm_args, "Setter must return a new instance"
    assert base_slurm_args.parallelism is None, "Original instance should remain unchanged"

    args_omp = base_slurm_args.set_parallelism("openmp")
    assert args_omp.parallelism == "openmp"
    assert args_omp is not base_slurm_args, "Setter must return a new instance"

def test_set_parallelism_invalid(base_slurm_args: SlurmArgs) -> None:
    """Test setting invalid parallelism raises ValueError."""
    # Match the set representation in the error message, allowing for different order
    with pytest.raises(ValueError, match=r"Parallelism must be one of \{'(mpi|openmp)', '(openmp|mpi)'\}"):
        base_slurm_args.set_parallelism("invalid") # type: ignore[arg-type]


def test_add_modules(base_slurm_args: SlurmArgs) -> None:
    """Test adding modules (appending and overwriting)."""
    assert base_slurm_args.modules is None, "Base args should have no modules initially"

    # Add initial modules
    modules1 = ["mod1/v1", "mod2/v2"]
    args1 = base_slurm_args.add_modules(modules1)
    assert args1.modules == modules1
    assert args1 is not base_slurm_args, "Setter must return a new instance"

    # Add more modules (append - default behavior)
    modules2 = ["mod3/v3"]
    args2 = args1.add_modules(modules2)
    assert args2.modules == modules1 + modules2
    assert args2 is not args1, "Setter must return a new instance"

    # Overwrite modules
    modules3 = ["new_mod/v1"]
    args3 = args2.add_modules(modules3, overwrite=True)
    assert args3.modules == modules3
    assert args3 is not args2, "Setter must return a new instance"


# --- Test Helper Methods ---

def test_get_temp_variables(base_slurm_args: SlurmArgs) -> None:
    """Test get_temp_variables (currently returns empty string)."""
    # This method currently has static behavior, just confirm it.
    assert base_slurm_args.get_temp_variables() == ""


def test_get_modules_string_generation(base_slurm_args: SlurmArgs) -> None:
    """Test generation of the 'module load' string block."""
    assert base_slurm_args.get_modules() == "", "Should be empty if no modules"

    args_one = base_slurm_args.add_modules(["one/1.0"])
    assert args_one.get_modules() == "module load one/1.0\n"

    args_multi = base_slurm_args.add_modules(["two/2.0", "three/3.0"])
    expected = """module load two/2.0
module load three/3.0
"""
    assert args_multi.get_modules() == expected


def test_get_launch_cmd_valid_configs(base_slurm_args: SlurmArgs) -> None:
    """Test launch command generation for different valid software/parallelism configs."""
    # Orca (no parallelism option in SlurmArgs)
    args_orca = base_slurm_args.set_software("orca")
    assert args_orca.get_launch_cmd() == "$(which orca) test_job.inp > test_job.out", "Orca launch command"

    # QChem - default (no parallelism explicitly set)
    args_qchem = base_slurm_args.set_software("qchem")
    assert args_qchem.get_launch_cmd() == "qchem test_job.in test_job.out", "QChem default launch command"

    # QChem - OpenMP
    args_qchem_omp = args_qchem.set_parallelism("openmp")
    assert args_qchem_omp.get_launch_cmd() == "qchem -nt 4 test_job.in test_job.out", "QChem OpenMP launch command"

    # QChem - MPI
    args_qchem_mpi = args_qchem.set_parallelism("mpi")
    assert args_qchem_mpi.get_launch_cmd() == "qchem -np 4 test_job.in test_job.out", "QChem MPI launch command"


# --- Test Pre-Submit Check ---

def test_pre_submit_check_ok(base_slurm_args: SlurmArgs) -> None:
    """Test pre_submit_check passes with valid software set via setter."""
    args_orca = base_slurm_args.set_software("orca")
    args_orca.pre_submit_check() # Should not raise

    args_qchem = base_slurm_args.set_software("qchem")
    args_qchem.pre_submit_check() # Should not raise


def test_pre_submit_check_no_software(base_slurm_args: SlurmArgs) -> None:
    """Test pre_submit_check fails when software is not set."""
    with pytest.raises(ValueError, match="Software not specified"):
        base_slurm_args.pre_submit_check()


def test_pre_submit_check_invalid_software() -> None:
    """Test pre_submit_check fails with invalid software set during init (bypassing setter)."""
    # Need to bypass the setter validation for this test case by setting directly in init
    args_invalid = SlurmArgs(
        exec_fname="test_job",
        time="01:00:00",
        n_cores=4,
        software="invalid_sw" # Manually set invalid software, bypassing setter validation
    )
    with pytest.raises(ValueError, match=r"Software must be one of \{'(orca|qchem)', '(qchem|orca)'\}"):
        args_invalid.pre_submit_check()


# --- Test Submit Script Creation ---

def test_create_submit_script_minimal_orca(base_slurm_args: SlurmArgs) -> None:
    """Test script creation with minimal args for ORCA."""
    args = base_slurm_args.set_software("orca")
    script = args.create_submit_script("my_orca_job")
    expected = """#!/bin/sh
#SBATCH --output sbatch.out
#SBATCH --error sbatch.err
#SBATCH -J my_orca_job
#SBATCH --time 01:00:00
#SBATCH --ntasks 4
#SBATCH --nodes 1

module load orca


$(which orca) test_job.inp > test_job.out
"""
    assert script == expected


def test_create_submit_script_minimal_qchem_default(base_slurm_args: SlurmArgs) -> None:
    """Test script creation with minimal args for QChem (no parallelism)."""
    args = base_slurm_args.set_software("qchem")
    script = args.create_submit_script("my_qchem_job")
    expected = """#!/bin/sh
#SBATCH --output sbatch.out
#SBATCH --error sbatch.err
#SBATCH -J my_qchem_job
#SBATCH --time 01:00:00

module load qchem


qchem test_job.in test_job.out
"""
    # Note: Q-Chem default doesn't add ntasks/cpus-per-task itself
    assert script == expected


def test_create_submit_script_qchem_openmp(base_slurm_args: SlurmArgs) -> None:
    """Test script creation for QChem with OpenMP."""
    args = base_slurm_args.set_software("qchem").set_parallelism("openmp")
    script = args.create_submit_script("my_qchem_omp_job")
    expected = """#!/bin/sh
#SBATCH --output sbatch.out
#SBATCH --error sbatch.err
#SBATCH -J my_qchem_omp_job
#SBATCH --time 01:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4

module load qchem


qchem -nt 4 test_job.in test_job.out
"""
    assert script == expected


def test_create_submit_script_qchem_mpi(base_slurm_args: SlurmArgs) -> None:
    """Test script creation for QChem with MPI."""
    args = base_slurm_args.set_software("qchem").set_parallelism("mpi")
    script = args.create_submit_script("my_qchem_mpi_job")
    expected = """#!/bin/sh
#SBATCH --output sbatch.out
#SBATCH --error sbatch.err
#SBATCH -J my_qchem_mpi_job
#SBATCH --time 01:00:00
#SBATCH -ntasks 4

module load qchem


qchem -np 4 test_job.in test_job.out
"""
    assert script == expected


def test_create_submit_script_full_options(base_slurm_args: SlurmArgs) -> None:
    """Test script creation with all optional Slurm directives."""
    args_temp = (
        base_slurm_args.set_software("orca")
        .set_memory(8192)
        .set_partition("high_mem")
        .set_constraint("skylake")
        .set_queue("priority")
        .add_modules(["extra/module", "another/one"]) # Adds after 'orca'
    )
    # Set account using replace as there's no dedicated setter
    args = replace(args_temp, account="my_account")

    script = args.create_submit_script("full_job")
    expected = """#!/bin/sh
#SBATCH --output sbatch.out
#SBATCH --error sbatch.err
#SBATCH -J full_job
#SBATCH --time 01:00:00
#SBATCH --constraint=skylake
#SBATCH --partition=high_mem
#SBATCH --qos=priority
#SBATCH --account=my_account
#SBATCH --mem=8192
#SBATCH --ntasks 4
#SBATCH --nodes 1

module load orca
module load extra/module
module load another/one


$(which orca) test_job.inp > test_job.out
"""
    assert script == expected

def test_create_submit_script_pre_submit_check_fail(base_slurm_args: SlurmArgs) -> None:
    """Test that create_submit_script fails if pre_submit_check fails (e.g., no software)."""
    # No software set on base_slurm_args
    with pytest.raises(ValueError, match="Software not specified"):
        base_slurm_args.create_submit_script("fail_job")

# fmt:on

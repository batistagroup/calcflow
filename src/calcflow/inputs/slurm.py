from dataclasses import dataclass, replace
from typing import Literal

from calcflow.utils import logger


@dataclass(frozen=True)
class SlurmArgs:
    """Container for Slurm arguments.

    Attributes:
        exec_fname (str): Name of the executable file (without extension).
        time (str): Time limit for the job in SLURM format (e.g., "12:00:00" for 12 hours).
        n_cores (int): Number of CPU cores to request.
        memory_mb (int | None): Memory to request in MB. Defaults to None.
        software (str | None): Software to be used (e.g., "orca", "qchem"). Defaults to None.
        partition (str | None): Slurm partition to use. Defaults to None.
        constraint (str | None): Slurm constraint to use. Defaults to None.
        account (str | None): Slurm account to use. Defaults to None.
        queue (str | None): Slurm queue (QOS) to use. Defaults to None.
        parallelism (Literal["mpi", "openmp"] | None): Parallelism type ("mpi" or "openmp"). Defaults to None.
        modules (list[str] | None): List of modules to load. Defaults to None.

    Raises:
        ValueError: If software or parallelism is not in allowed sets.
    """

    exec_fname: str
    time: str
    n_cores: int
    memory_mb: int | None = None
    software: str | None = None
    partition: str | None = None
    constraint: str | None = None
    account: str | None = None
    queue: str | None = None
    parallelism: Literal["mpi", "openmp"] | None = None
    modules: list[str] | None = None

    ALLOWED_SOFTWARE = {"orca", "qchem"}
    ALLOWED_PARALLELISM = {"mpi", "openmp"}

    def set_software(self, software: str) -> "SlurmArgs":
        """Set the software to be used.

        Args:
            software (str): Software name (e.g., "orca", "qchem").

        Returns:
            SlurmArgs: A new SlurmArgs instance with updated software.

        Raises:
            ValueError: If software is not in ALLOWED_SOFTWARE.
        """
        if software.lower() not in self.ALLOWED_SOFTWARE:
            raise ValueError(f"Software must be one of {self.ALLOWED_SOFTWARE}")
        if self.modules is None:
            modules = [software.lower()]
        else:
            modules = self.modules + [software.lower()]
        return replace(self, software=software.lower(), modules=modules)

    def set_time(self, time: str) -> "SlurmArgs":
        """Set the time limit for the job.

        Args:
            time (str): Time limit in SLURM format (e.g., "12:00:00").

        Returns:
            SlurmArgs: A new SlurmArgs instance with updated time.
        """
        return replace(self, time=time)

    def set_memory(self, memory_mb: int) -> "SlurmArgs":
        """Set the memory to request for the job.

        Args:
            memory_mb (int): Memory in MB.

        Returns:
            SlurmArgs: A new SlurmArgs instance with updated memory.

        Raises:
            ValueError: If memory_mb is less than 256.
        """
        if memory_mb < 256:
            logger.warning("Memory allocation seems low (< 256 MB), please specify in MB.")
        return replace(self, memory_mb=memory_mb)

    def set_partition(self, partition: str) -> "SlurmArgs":
        """Set the Slurm partition to use.

        Args:
            partition (str): Partition name.

        Returns:
            SlurmArgs: A new SlurmArgs instance with updated partition.
        """
        return replace(self, partition=partition)

    def set_constraint(self, constraint: str) -> "SlurmArgs":
        """Set the Slurm constraint to use.

        Args:
            constraint (str): Constraint string.

        Returns:
            SlurmArgs: A new SlurmArgs instance with updated constraint.
        """
        return replace(self, constraint=constraint)

    def set_queue(self, queue: str) -> "SlurmArgs":
        """Set the Slurm queue (QOS) to use.

        Args:
            queue (str): Queue name.

        Returns:
            SlurmArgs: A new SlurmArgs instance with updated queue.
        """
        return replace(self, queue=queue)

    def set_parallelism(self, parallelism: Literal["mpi", "openmp"]) -> "SlurmArgs":
        """Set the parallelism type.

        Args:
            parallelism (Literal["mpi", "openmp"]): Parallelism type ("mpi" or "openmp").

        Returns:
            SlurmArgs: A new SlurmArgs instance with updated parallelism.

        Raises:
            ValueError: If parallelism is not in ALLOWED_PARALLELISM.
        """
        if parallelism not in self.ALLOWED_PARALLELISM:
            raise ValueError(f"Parallelism must be one of {self.ALLOWED_PARALLELISM}")
        return replace(self, parallelism=parallelism)

    def add_modules(self, modules: list[str], overwrite: bool = False) -> "SlurmArgs":
        """Add modules to the list of modules to load.

        Args:
            modules (list[str]): List of module names to add.
            overwrite (bool, optional): If True, overwrite existing modules. Defaults to False.

        Returns:
            SlurmArgs: A new SlurmArgs instance with updated modules.
        """
        if overwrite or self.modules is None:
            return replace(self, modules=modules)
        else:
            return replace(self, modules=self.modules + modules)

    def get_temp_variables(self) -> str:
        """Get temporary environment variables to set in the SLURM script.

        Returns:
            str: Empty string as no temp variables are needed for now.
        """
        return ""

    def get_modules(self) -> str:
        """Get the 'module load' commands as a string.

        Returns:
            str: String containing 'module load' commands, one per line.
                 Returns an empty string if no modules are specified.
        """
        if not self.modules:
            return ""
        modules_str = ""
        for module in self.modules:
            modules_str += f"module load {module}\n"
        return modules_str

    def get_launch_cmd(self) -> str:
        """Get the command to launch the calculation.

        Returns:
            str: Command string to launch the specified software.

        Raises:
            ValueError: If software is not supported.
        """
        assert self.software is not None

        if self.software == "orca":
            return f"$(which orca) {self.exec_fname}.inp > {self.exec_fname}.out"
        elif self.software == "qchem":
            if self.parallelism == "mpi":
                return f"qchem -np {self.n_cores} {self.exec_fname}.in {self.exec_fname}.out"
            elif self.parallelism == "openmp":
                return f"qchem -nt {self.n_cores} {self.exec_fname}.in {self.exec_fname}.out"
            else:
                return f"qchem {self.exec_fname}.in {self.exec_fname}.out"
        else:
            raise ValueError(f"Software {self.software} not supported")

    def pre_submit_check(self) -> None:
        """Perform pre-submission checks to ensure required arguments are set.

        Raises:
            ValueError: If software is not specified or not in ALLOWED_SOFTWARE.
        """
        if self.software is None:
            raise ValueError("Software not specified")
        if self.software not in self.ALLOWED_SOFTWARE:
            raise ValueError(f"Software must be one of {self.ALLOWED_SOFTWARE}")

    def create_submit_script(self, job_name: str) -> str:
        """Create the SLURM submission script.

        Args:
            job_name (str): Name of the job.

        Returns:
            str: String containing the full SLURM submission script.
        """
        self.pre_submit_check()
        script = f"""#!/bin/sh
#SBATCH --output sbatch.out
#SBATCH --error sbatch.err
#SBATCH -J {job_name}
"""
        if self.constraint is not None:
            script += f"#SBATCH --constraint={self.constraint}\n"
        if self.partition is not None:
            script += f"#SBATCH --partition={self.partition}\n"
        if self.queue is not None:
            script += f"#SBATCH --qos={self.queue}\n"
        if self.account is not None:
            script += f"#SBATCH --account={self.account}\n"
        if self.memory_mb is not None:
            script += f"#SBATCH --mem={self.memory_mb}\n"
        if self.software == "orca":
            script += f"#SBATCH --ntasks {self.n_cores}\n"
            script += "#SBATCH --nodes 1\n"

        elif self.software == "qchem":
            if self.parallelism == "openmp":
                script += "#SBATCH --ntasks 1\n"
                script += f"#SBATCH --cpus-per-task {self.n_cores}\n"
            elif self.parallelism == "mpi":
                script += f"#SBATCH -ntasks {self.n_cores}\n"
        script += "\n"
        script += self.get_modules() + "\n"
        script += self.get_temp_variables() + "\n"
        script += self.get_launch_cmd() + "\n"
        return script

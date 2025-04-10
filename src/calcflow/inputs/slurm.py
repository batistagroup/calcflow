from dataclasses import dataclass, replace
from typing import Literal

from calcflow.utils import logger


@dataclass(frozen=True)
class SlurmArgs:
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
        if software.lower() not in self.ALLOWED_SOFTWARE:
            raise ValueError(f"Software must be one of {self.ALLOWED_SOFTWARE}")
        if self.modules is None:
            modules = [software.lower()]
        else:
            modules = self.modules + [software.lower()]
        return replace(self, software=software.lower(), modules=modules)

    def set_time(self, time: str) -> "SlurmArgs":
        return replace(self, time=time)

    def set_memory(self, memory_mb: int) -> "SlurmArgs":
        if memory_mb < 256:
            logger.warning("Memory allocation seems low (< 256 MB), please specify in MB.")
        return replace(self, memory_mb=memory_mb)

    def set_partition(self, partition: str) -> "SlurmArgs":
        return replace(self, partition=partition)

    def set_constraint(self, constraint: str) -> "SlurmArgs":
        return replace(self, constraint=constraint)

    def set_queue(self, queue: str) -> "SlurmArgs":
        return replace(self, queue=queue)

    def set_parallelism(self, parallelism: Literal["mpi", "openmp"]) -> "SlurmArgs":
        if parallelism not in self.ALLOWED_PARALLELISM:
            raise ValueError(f"Parallelism must be one of {self.ALLOWED_PARALLELISM}")
        return replace(self, parallelism=parallelism)

    def add_modules(self, modules: list[str], overwrite: bool = False) -> "SlurmArgs":
        if overwrite or self.modules is None:
            return replace(self, modules=modules)
        else:
            return replace(self, modules=self.modules + modules)

    def get_temp_variables(self) -> str:
        return ""

    def get_modules(self) -> str:
        if not self.modules:
            return ""
        modules_str = ""
        for module in self.modules:
            modules_str += f"module load {module}\n"
        return modules_str

    def get_launch_cmd(self) -> str:
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
        if self.software is None:
            raise ValueError("Software not specified")
        if self.software not in self.ALLOWED_SOFTWARE:
            raise ValueError(f"Software must be one of {self.ALLOWED_SOFTWARE}")

    def create_submit_script(self, job_name: str) -> str:
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

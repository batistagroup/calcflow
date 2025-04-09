from dataclasses import dataclass, replace

from calcflow.utils import logger


@dataclass(frozen=True)
class SlurmArgs:
    exec_fname: str
    time: str
    n_cores: int
    memory_mb: int
    software: str | None = None

    ALLOWED_SOFTWARE = {"orca", "qchem"}

    def _check_software(self) -> None:
        if self.software is None:
            raise ValueError("Software not specified")
        if self.software not in self.ALLOWED_SOFTWARE:
            raise ValueError(f"Software must be one of {self.ALLOWED_SOFTWARE}")

    def set_software(self, software: str) -> "SlurmArgs":
        allowed = {"orca", "qchem"}
        if software.lower() not in allowed:
            raise ValueError(f"Software must be one of {allowed}")
        return replace(self, software=software.lower())

    def set_time(self, time: str) -> "SlurmArgs":
        return replace(self, time=time)

    def set_memory(self, memory_mb: int) -> "SlurmArgs":
        if memory_mb < 256:
            logger.warning("Memory allocation seems low (< 256 MB), please specify in MB.")
        return replace(self, memory_mb=memory_mb)

    def get_temp_variables(self) -> str:
        self._check_software()
        assert self.software is not None  # for mypy
        temp_vars = {
            "orca": "",
            "qchem": f"""export OMP_NUM_THREADS={self.n_cores}
export MKL_NUM_THREADS={self.n_cores}
export OPENBLAS_NUM_THREADS={self.n_cores}
export NUMEXPR_NUM_THREADS={self.n_cores}""",
        }
        return temp_vars[self.software]

    def get_modules(self) -> str:
        self._check_software()
        assert self.software is not None  # for mypy
        modules = {
            "orca": "module load ORCA",
            "qchem": "module load Q-Chem",
        }
        return modules[self.software]

    def get_launch_cmd(self) -> str:
        self._check_software()
        assert self.software is not None  # for mypy
        commands = {
            "orca": f"$(which orca) {self.exec_fname}.inp > {self.exec_fname}.out",
            "qchem": f"qchem -np {self.n_cores} {self.exec_fname}.in {self.exec_fname}.out",
        }
        return commands[self.software]

    def export_submit(self, job_name: str) -> str:
        if self.software is None:
            raise ValueError("Software not specified")

        if "-" in self.time:
            time_spec = f"#SBATCH -p week\n#SBATCH -t {self.time}:00"
        else:
            time_spec = f"#SBATCH -t {self.time}:00"

        script = f"""#!/bin/sh
{time_spec}
#SBATCH -o sbatch.out
#SBATCH -e sbatch.err
#SBATCH -J {job_name}
#SBATCH -n {self.n_cores}
#SBATCH -N 1
#SBATCH --mem={self.memory_mb}

{self.get_temp_variables()}
{self.get_modules()}
{self.get_launch_cmd()}
"""
        return script

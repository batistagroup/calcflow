from pathlib import Path

from calcflow.geometry.static import Geometry
from calcflow.inputs.orca import OrcaInput
from calcflow.parsers.orca import parse_orca_output
from meoq.slurm import NerscSlurmArgs

data_path = Path(__file__).resolve().parents[1] / "data"
calcs_path = data_path / "calculations"


xyz_1h2o = Geometry.from_xyz_file(data_path / "geometries" / "1h2o.xyz")


clc_folder = calcs_path / "working" / "h2o" / "sp-svp"
clc_folder.mkdir(parents=True, exist_ok=True)

# fmt:off
base_args = NerscSlurmArgs(
    exec_fname="opt", time="01:00:00", n_cores=16,
    constraint="cpu", account="m410", queue="premium")

job = OrcaInput(
    charge=0, spin_multiplicity=1, task="geometry", level_of_theory="wB97X-D3", basis_set="def2-tzvp", n_cores=16
)
# fmt:on

run = {
    "create": False,
    "parse": True,
}

if run["create"]:
    with (clc_folder / "submit.sh").open("w") as f:
        f.write(base_args.set_software("orca").create_submit_script("h2o-opt"))

    with (clc_folder / "opt.inp").open("w") as f:
        f.write(job.export_input_file(xyz_1h2o.get_coordinate_block()))

if run["parse"]:
    out = (clc_folder / "opt.out").read_text()
    calc_data = parse_orca_output(out)
    print(calc_data)

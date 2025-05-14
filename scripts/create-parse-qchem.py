from pathlib import Path

from calcflow.geometry.static import Geometry
from calcflow.inputs.qchem import QchemInput
from calcflow.inputs.slurm import SlurmArgs
from calcflow.parsers.qchem import parse_qchem_sp_output

data_path = Path(__file__).resolve().parents[1] / "data"
calcs_path = data_path / "calculations"


xyz_1h2o = Geometry.from_xyz_file(data_path / "geometries" / "1h2o.xyz")


clc_folder = calcs_path / "examples" / "qchem" / "h2o"
clc_folder.mkdir(parents=True, exist_ok=True)

# fmt:off
base_args = SlurmArgs(
    exec_fname="sp", time="01:00:00", n_cores=16,
    constraint="cpu", account="m410", queue="premium")

job = QchemInput(
    charge=0, spin_multiplicity=1, task="energy", level_of_theory="wB97X-D3", basis_set="sto-3g", n_cores=16
)
# fmt:on

run = {
    "create": False,
    "parse": True,
}

if run["create"]:
    with (clc_folder / "submit.sh").open("w") as f:
        f.write(base_args.set_software("qchem").create_submit_script(f"h2o-{base_args.exec_fname}"))

    with (clc_folder / f"{base_args.exec_fname}.inp").open("w") as f:
        f.write(job.export_input_file(xyz_1h2o))

if run["parse"]:
    sp_sto = parse_qchem_sp_output((clc_folder / "sp-sto.out").read_text())
    print(sp_sto.metadata)
    print(sp_sto.final_energy_eh)
    # sp_tzvppd = parse_qchem_sp_output((clc_folder / "sp-tzvppd.out").read_text())
    sp_sto_smd = parse_qchem_sp_output((clc_folder / "sp-sto-smd.out").read_text())
    print(sp_sto_smd.metadata)
    print(sp_sto_smd.final_energy_eh)
    print(sp_sto_smd.smd_data)
    # breakpoint()

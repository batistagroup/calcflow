import logging
from pathlib import Path

from calcflow.geometry.static import Geometry
from calcflow.inputs.qchem import QchemInput
from calcflow.inputs.slurm import SlurmArgs
from calcflow.parsers.qchem import parse_qchem_mom_output, parse_qchem_sp_output, parse_qchem_tddft_output
from calcflow.utils import logger

# logger.setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)

data_path = Path(__file__).resolve().parents[1] / "data"
calcs_path = data_path / "calculations"


xyz_1h2o = Geometry.from_xyz_file(data_path / "geometries" / "1h2o.xyz")


clc_folder = calcs_path / "examples" / "qchem" / "h2o"
clc_54_folder = calcs_path / "examples" / "qchem" / "h2o-5.4"
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
    "sp-create": False,
    "sp-parse": False,
    "tddft-create": False,
    "tddft-parse": False,
    "mom-parse": False,
    "mom-xas-parse": False,
    "sp-parse-5.4": False,
    "tddft-uks-parse-5.4": False,
    "mom-sp-parse-5.4": True,
}

if run["sp-create"]:
    with (clc_folder / "submit.sh").open("w") as f:
        f.write(base_args.set_software("qchem").create_submit_script(f"h2o-{base_args.exec_fname}"))

    with (clc_folder / f"{base_args.exec_fname}.inp").open("w") as f:
        f.write(job.export_input_file(xyz_1h2o))


if run["sp-parse"]:
    print("------- SP STO-3G ---------")
    sp_sto = parse_qchem_sp_output((clc_folder / "sp-sto.out").read_text())
    print(sp_sto.metadata)
    print(sp_sto.final_energy)
    print("------- SP STO-3G-SMD ---------")
    sp_sto_smd = parse_qchem_sp_output((clc_folder / "sp-sto-smd.out").read_text())
    print(sp_sto_smd.metadata)
    print(sp_sto_smd.final_energy)
    print(sp_sto_smd.smd)
    print("------- SP TZVPPD-D3 ---------")
    sp_tzvppd = parse_qchem_sp_output((clc_folder / "sp-tzvppd.out").read_text())
    print(sp_tzvppd.metadata)
    print(sp_tzvppd.final_energy)
    print("------- SP TZVPPD-SMD ---------")
    sp_tzvppd_smd = parse_qchem_sp_output((clc_folder / "sp-tzvppd-smd.out").read_text())
    print(sp_tzvppd_smd.metadata)
    print(sp_tzvppd_smd.final_energy)
    print(sp_tzvppd_smd.smd)

    print("------- SP TZVPPD-SMD ---------")
    sp_tzvppd_smd = parse_qchem_sp_output((clc_folder / "sp-tzvppd-smd.out").read_text())
    print(sp_tzvppd_smd.metadata)
    print(sp_tzvppd_smd.final_energy)
    print(sp_tzvppd_smd.smd)

if run["tddft-parse"]:
    print("------- TDDFT STO-3G ---------")
    tddft_pc2 = parse_qchem_tddft_output((clc_folder / "tddft-uks-pc2.out").read_text())
    print(tddft_pc2)
    print(tddft_pc2.tddft)
    # print(tddft_pc2.gs_reference_analysis)
    assert tddft_pc2.tddft is not None
    assert tddft_pc2.tddft.excited_state_analyses is not None
    print(len(tddft_pc2.tddft.excited_state_analyses))

    assert tddft_pc2.tddft.transition_dm_analyses is not None
    print(len(tddft_pc2.tddft.transition_dm_analyses))
    assert tddft_pc2.tddft.transition_dm_analyses[0].exciton_analysis is not None
    print(tddft_pc2.tddft.transition_dm_analyses[0].exciton_analysis.total_properties)

    # print(tddft_pc2.tddft.nto_analyses[-2])
    print(tddft_pc2.orbitals)

if run["mom-parse"]:
    mom_pc2 = parse_qchem_mom_output((clc_folder / "mom-sp.out").read_text())
    assert mom_pc2.job1 is not None
    assert mom_pc2.job2 is not None
    assert mom_pc2.job1.scf is not None
    assert mom_pc2.job2.scf is not None
    j1 = mom_pc2.job1.scf.energy
    j2 = mom_pc2.job2.scf.energy
    ev = (j2 - j1) * 27.21138602
    print(f"E(H2O) = {ev:.6f} eV")

    mom_pc2 = parse_qchem_mom_output((clc_folder / "mom-smd-sp.out").read_text())
    assert mom_pc2.job1 is not None
    assert mom_pc2.job2 is not None
    assert mom_pc2.job1.scf is not None
    assert mom_pc2.job2.scf is not None
    j1 = mom_pc2.job1.scf.energy
    j2 = mom_pc2.job2.scf.energy
    ev = (j2 - j1) * 27.21138602
    print(f"E(H2O) = {ev:.6f} eV")
    print(mom_pc2.job2.scf)

if run["mom-xas-parse"]:
    mom_pc2 = parse_qchem_mom_output((clc_folder / "mom-smd-xas.out").read_text())
    print(mom_pc2.job1)
    print(mom_pc2.job2)
    print(mom_pc2.job2.scf)
    print(mom_pc2.job2.tddft)

if run["sp-parse-5.4"]:
    sp_sto = parse_qchem_sp_output((clc_54_folder / "sp-sto-smd.out").read_text())
    print(sp_sto)
    print(sp_sto.metadata)
    print(sp_sto.final_energy)
    print(sp_sto.smd)

if run["tddft-uks-parse-5.4"]:
    tddft_pc2 = parse_qchem_tddft_output((clc_54_folder / "tddft-uks-pc2.out").read_text())
    print(tddft_pc2)
    print(tddft_pc2.tddft)

if run["mom-sp-parse-5.4"]:
    mom_pc2 = parse_qchem_mom_output((clc_54_folder / "mom-smd-sp.out").read_text())
    print(mom_pc2.job1)
    print(mom_pc2.job2)
    assert mom_pc2.job1.scf is not None
    assert mom_pc2.job2.scf is not None
    ev = (mom_pc2.job2.scf.energy - mom_pc2.job1.scf.energy) * 27.21138602
    print(f"E(H2O) = {ev:.6f} eV")

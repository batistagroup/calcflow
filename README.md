# CalcFlow

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/batistagroup/calcflow/blob/main/LICENSE)
[![image](https://img.shields.io/pypi/v/calcflow.svg)](https://pypi.org/project/calcflow/)
[![codecov](https://codecov.io/gh/batistagroup/calcflow/graph/badge.svg?token=bO5X75J8li)](https://codecov.io/gh/batistagroup/calcflow)
![codequal](https://github.com/batistagroup/calcflow/actions/workflows/quality.yml/badge.svg)

**CalcFlow: Quantum Chemistry Calculation I/O. Done Right.**

`CalcFlow` provides a robust, Pythonic interface for preparing inputs and parsing outputs for quantum chemistry software like Q-Chem and ORCA. It has **zero external dependencies** and is built for clarity and reliability. Get your calculations set up and results processed without the usual boilerplate.

WARNING: Package is in pre-release alpha stage. May introduce backwards-incompatible changes. Contributions & suggestions & advice are welcome.

## Key Features

what you care about:

- **Zero Dependencies**: is true now, will be true forever. Integrate into any project without worrying about whether it matches your numpy or rdkit version.
- **Intuitive Pythonic Interface**: specify calculation parameters how you think about them and calcflow will handle the translation into what QChem or ORCA expects.
- **Proactive Input Validation**: Minimizes runtime errors by rigorously validating all calculation parameters against both general and program-specific constraints.
- **Comprehensive Parsing**: stop wasting time on extracting data you need from a txt out file and start doing meaningful work (analysis).

what's nice under the hood:

- **Immutable & Fluent API**: ensures data integrity and predictable state transitions via `frozen` dataclasses and an expressive, chainable API.
- **Fully Type Annotated**: code is more robust and easier to understand.
- **Extensible Core Architecture**: designed with clear abstract base classes, simplifying the integration of support for additional quantum chemistry programs.
- **Assured Reliability via Comprehensive Testing**: so you don't have to worry if the parser is working correctly.

## Quick Start: Q-Chem TDDFT Example

Set up a Q-Chem tddft calculation for a water molecule with SMD solvation:

```python
from calcflow.geometry.static import Geometry
from calcflow.inputs.qchem import QchemInput

# 1. Define Molecular Geometry
atoms = [
    ("O", (0.000000, 0.000000, 0.117300)),
    ("H", (0.000000, 0.757200, -0.469200)),
    ("H", (0.000000, -0.757200, -0.469200)),
]
water_molecule = Geometry(atoms=atoms, comment="Water molecule for Q-Chem SP")
# or load from xyz file
water_molecule = Geometry.from_xyz_file(data_path / "1h2o.xyz")

# 2. Configure Q-Chem Calculation Input
base_job = QchemInput(
    charge=0, spin_multiplicity=1, task="energy", 
    level_of_theory="wB97X-D3", basis_set="def2-tzvp", n_cores=16,
    run_tddft=True, tddft_nroots=10, tddft_singlets=True, tddft_triplets=False
)

# Uses a fluent API for modifications.
qchem_job = base_job.set_solvation(model="smd", solvent="water") 

# 3. Export the Input File Content
with open("h2o_tddft.in", "w") as f:
    f.write(qchem_job.export_input_file(water_molecule))
```

Want to create another input file for a tddft with triplets or with state analysis? ezy.

```python
job2 = qchem_job.set_tddft(nroots=10, singlets=True, triplets=True, state_analysis=True)
with open("h2o_tddft.in", "w") as f:
    f.write(job2.export_input_file(water_molecule))
```

what about getting an O K-Edge XAS spectrum with an element-specific basis?

```python
job3 = (
    qchem_job.set_tddft(nroots=10, singlets=True, triplets=False, state_analysis=True)
    .set_basis({"H": "pc-2", "O": "pcX-2"})
    .set_reduced_excitation_space(solute_orbitals=[1])
)
with open("h2o_xas.in", "w") as f:
    f.write(job3.export_input_file(water_molecule))
```


## Parsing Q-Chem Output Files

Once your Q-Chem calculation is complete, use `CalcFlow` parsers. Example for a single point energy calculation:

```python
from calcflow.parsers.qchem import parse_qchem_sp_output # Specific parser for Q-Chem SP

# Path to your Q-Chem output file
output_file_path = "h2o_qchem_sp.out" # Replace with your actual file path

try:
    with open(output_file_path, "r") as f:
        output_content = f.read()

    # Parse the content
    parsed_data = parse_qchem_sp_output(output_content)

    # Access extracted information
    if parsed_data.success:
        print(f"Q-Chem calculation successful!")
        print(f"Final SCF Energy: {parsed_data.final_energy} Eh")
        if parsed_data.multipole_moments:
            print(f"Dipole Moment (Debye): {parsed_data.multipole_moments.dipole_moment_debye}")
        if parsed_data.smd: # If SMD solvation was used
            print(f"SMD Solvation Energy: {parsed_data.smd.smd_energy_solvent} Eh")
            print(f"Total Energy with SMD: {parsed_data.smd.total_energy_with_solvent} Eh")
    else:
        print(f"Q-Chem calculation failed or did not complete.")
        if parsed_data.errors: # Check for errors attribute
            print(f"Errors: {', '.join(parsed_data.errors)}")

except FileNotFoundError:
    print(f"Error: Output file '{output_file_path}' not found.")
except Exception as e:
    print(f"An error occurred during parsing: {e}")
```
*Note: The exact parser module (e.g., `parse_qchem_tddft_output`, `parse_qchem_mom_output`) and the attributes of `parsed_data` will vary based on the Q-Chem calculation type. Consult the `calcflow.parsers` documentation for details.*

## Contributing

Direct, effective contributions are welcome. Fork, modify, test, and pull request. Adhere to existing quality standards.

## License

MIT. Pure and simple.


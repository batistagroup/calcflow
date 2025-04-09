# CalcFlow

A Python package to simplify preparing and parsing quantum chemistry calculations, with no external dependencies. Provides a clean, Pythonic API for interacting with QChem, ORCA, and potentially other quantum chemistry software.

## Features

- **Program-Agnostic API**: Abstract away program-specific syntax with a consistent interface
- **Strongly Typed**: Comprehensive type hints and validation for safer code
- **Immutable Objects**: Dataclass-based with a fluent API for transparent, thread-safe state changes
- **Extensible Design**: Abstract base classes make it easy to add support for additional programs
- **No Dependencies**: Built entirely on Python standard library
- **Comprehensive Validation**: Prevent errors before they happen with rigorous validation of inputs

## Project Structure

```
calcflow/
├── __init__.py       # Package exports
├── core.py           # Base CalculationInput abstract class
├── results.py        # Result data structures
├── inputs/           # Input file generators
│   ├── __init__.py     # Public API exports
│   ├── qchem.py        # QChem input generation
│   ├── orca.py         # ORCA input generation
│   └── slurm.py        # SLURM job submission helpers
├── parsers/          # Output file parsers
│   ├── __init__.py     # Parser exports
│   ├── qchem.py        # QChem output parsing
│   └── orca.py         # ORCA output parsing
└── basis_sets/       # Basis set definitions
    ├── __init__.py     # Basis set exports
    └── qchem/          # QChem specific basis sets
        ├── __init__.py
        └── pcX.py      # pcX basis sets for QChem

tests/               # Test suite
├── __init__.py
├── test_core.py     # Tests for core functionality
├── test_qchem.py    # Tests for QChem implementation
├── test_orca.py     # Tests for ORCA implementation
├── inputs/
│   └── __init__.py
└── parsers/
    └── __init__.py

examples/            # Usage examples
└── simple_usage.py  # Simple usage demonstrations
```

## Usage Example

```python
from calcflow.inputs.qchem import QChemArgs

# Create a single point DFT calculation with implicit solvation
qchem_args = (QChemArgs(
    charge=0,
    spin_multiplicity=1,
    task="energy",          # Maps to job="sp" in QChem
    level_of_theory="B3LYP",
    basis_set="6-31G*",
)
.set_solvation("smd", "water")  # Add SMD implicit solvation
.enable_nbo())                  # Enable NBO analysis

# Generate the input file with a molecule geometry
geom = """
O  0.000000  0.000000  0.117790
H  0.000000  0.756950 -0.471160
H  0.000000 -0.756950 -0.471160
"""
input_file = qchem_args.export_input_file(geom)
```

Similar interfaces are available for ORCA:

```python
from calcflow.inputs.orca import OrcaDFTArgs

# Create a TD-DFT calculation with RI approximation
orca_args = (OrcaDFTArgs(
    charge=0,
    spin_multiplicity=1,
    task="energy",
    level_of_theory="PBE0",
    basis_set="def2-TZVP",
    n_procs=4,
)
.enable_ri("RIJCOSX", "def2/J")  # Add RI approximation
.enable_tddft(nroots=10)         # Calculate 10 excited states
.set_solvation("cpcm", "water"))  # Add implicit solvation
```

## Development

### Setup Development Environment

This project uses poetry for dependency management:

```bash
# Install dependencies
poetry install

# Run tests
pytest
```

### Adding a New Quantum Chemistry Program

1. Create a new file in `calcflow/inputs/` for your program
2. Create a class that inherits from `CalculationInput` 
3. Implement the required abstract methods and add program-specific features
4. Add comprehensive validation
5. Register your class in `calcflow/inputs/__init__.py`
6. Add corresponding tests in the `tests/` directory

## License

MIT License

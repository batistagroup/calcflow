# Project Structure


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


# Project Structure


```
src/calcflow/
├── __init__.py       # Package exports
├── core.py           # Base CalculationInput abstract class
├── results.py        # Result data structures
├── utils.py          # Utility functions
├── py.typed          # Typing marker file
├── typing/
│   └── __init__.py     # Typing exports
├── geometry/
│   ├── __init__.py
│   └── static.py       # Static geometry definitions
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
        ├── pcX.py      # pcX basis sets for QChem
        └── README.md   # Readme for QChem basis sets

tests/               # Test suite
├── __init__.py
├── test_core.py     # Tests for core functionality
├── geometries/       # Test geometries
├── inputs/
│   ├── __init__.py
│   └── test_slurm.py # Tests for SLURM input generation
└── parsers/
    └── __init__.py
```


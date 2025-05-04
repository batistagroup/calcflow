"""Initialize the basis_sets package and trigger registration."""

# Import modules containing basis set definitions and registrations.
# This ensures CustomBasisSet objects are registered when the package is imported.
from calcflow.basis_sets import qchem

# Declare qchem in __all__ to signal to linters that the import is intentional,
# primarily for the side-effect of running register_basis_set() within pcX.py.
__all__ = ["qchem"]

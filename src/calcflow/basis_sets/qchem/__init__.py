"""Initialize the QChem basis set sub-package and trigger registrations."""

# Import the module(s) containing the basis set definitions and registrations.
# This ensures the register_basis_set() calls within pcX.py are executed
# when this sub-package is imported.
from calcflow.basis_sets.qchem import pcX

# Declare pcX in __all__ to signal to linters that the import is intentional,
# primarily for the side-effect of running register_basis_set() within pcX.py.
__all__ = ["pcX"]

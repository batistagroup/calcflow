"""Central registry mapping basis set names to CustomBasisSet objects."""

from calcflow.basis_sets.core import CustomBasisSet
from calcflow.utils import logger

# Maps basis set name (lowercase string) to the CustomBasisSet object.
_BASIS_SET_REGISTRY: dict[str, CustomBasisSet] = {}


def register_basis_set(basis_set_object: CustomBasisSet) -> None:
    """Registers a custom basis set object.

    Args:
        basis_set_object: The CustomBasisSet instance to register.

    Raises:
        ValueError: If the name (case-insensitive) is already registered.
        TypeError: If the provided object is not a CustomBasisSet instance.
    """
    if not isinstance(basis_set_object, CustomBasisSet):
        raise TypeError(f"Object to register must be an instance of CustomBasisSet, got {type(basis_set_object)}")

    # Name is already lowercased in CustomBasisSet.__post_init__
    normalized_name = basis_set_object.name
    if normalized_name in _BASIS_SET_REGISTRY:
        raise ValueError(f"Basis set name '{normalized_name}' is already registered.")

    _BASIS_SET_REGISTRY[normalized_name] = basis_set_object
    logger.info(f"Registered CustomBasisSet: '{normalized_name}'")


def get_basis_set_object(name: str) -> CustomBasisSet | None:
    """Retrieves the CustomBasisSet object for a registered name.

    Performs case-insensitive lookup.

    Args:
        name: The name of the basis set.

    Returns:
        The CustomBasisSet object if found, otherwise None.
    """
    return _BASIS_SET_REGISTRY.get(name.lower())

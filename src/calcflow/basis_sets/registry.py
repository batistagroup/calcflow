"""Central registry mapping program names and basis set names to CustomBasisSet objects."""

from calcflow.basis_sets.custom_basis import CustomBasisSet
from calcflow.utils import logger

# Central storage: {program_name_lower: {basis_name_lower: CustomBasisSet}}
_PROGRAM_REGISTRIES: dict[str, dict[str, CustomBasisSet]] = {}


def register_basis_set(basis_set_object: CustomBasisSet) -> None:
    """Registers a custom basis set object in the central store.

    Organizes registration based on the 'program' attribute of the object.

    Args:
        basis_set_object: The CustomBasisSet instance to register.

    Raises:
        ValueError: If the name (case-insensitive) is already registered
                    *for the same program*.
        TypeError: If the provided object is not a CustomBasisSet instance.
        ValidationError: If the basis_set_object itself is invalid (e.g., empty program/name).
    """
    if not isinstance(basis_set_object, CustomBasisSet):
        raise TypeError(f"Object to register must be an instance of CustomBasisSet, got {type(basis_set_object)}.")

    # Names/programs are already lowercased in CustomBasisSet.__post_init__
    program_key = basis_set_object.program
    basis_key = basis_set_object.name

    # Ensure the program-specific dictionary exists
    if program_key not in _PROGRAM_REGISTRIES:
        _PROGRAM_REGISTRIES[program_key] = {}
        logger.debug(f"Created new basis set registry for program: '{program_key}'")

    program_registry = _PROGRAM_REGISTRIES[program_key]

    # Check for duplicates within this program's registry
    if basis_key in program_registry:
        raise ValueError(f"Basis set name '{basis_key}' is already registered for program '{program_key}'.")

    program_registry[basis_key] = basis_set_object
    logger.info(f"Registered CustomBasisSet: '{program_key}/{basis_key}'")


class ProgramBasisRegistry:
    """Provides access to basis sets registered for a specific program."""

    def __init__(self, program: str):
        """Initializes a registry accessor for a specific program.

        Args:
            program: The name of the program (case-insensitive).

        Raises:
            ValueError: If no basis sets have been registered for the specified program.
                      Use register_basis_set first.
        """
        self._program_key = program.lower()
        if self._program_key not in _PROGRAM_REGISTRIES:
            # We could lazily create it here, but requiring prior registration
            # makes intent clearer - registry must be populated first.
            raise KeyError(
                f"No basis sets registered for program '{self._program_key}'. Use register_basis_set() first."
            )
        # Store a direct reference to this program's dictionary
        self._registry: dict[str, CustomBasisSet] = _PROGRAM_REGISTRIES[self._program_key]
        logger.debug(f"ProgramBasisRegistry initialized for '{self._program_key}'")

    def __getitem__(self, name: str) -> CustomBasisSet:
        """Retrieves the CustomBasisSet object for a registered name within this program.

        Performs case-insensitive lookup for the basis set name.

        Args:
            name: The name of the basis set.

        Returns:
            The CustomBasisSet object if found, otherwise None.
        """
        if name not in self:
            raise KeyError(f"Basis set '{name}' not found for program '{self._program_key}'.")
        return self._registry[name.lower()]

    def __contains__(self, name: str) -> bool:
        """Checks if a basis set name (case-insensitive) is registered for this program.

        Allows using the `in` operator (e.g., `"pcseg-2" in registry`).
        Provides an O(1) lookup.
        """
        return name.lower() in self._registry

    @property
    def program(self) -> str:
        """Returns the normalized program name this registry instance is for."""
        return self._program_key

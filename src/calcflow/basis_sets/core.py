from dataclasses import dataclass, field

from calcflow.exceptions import ValidationError
from calcflow.utils import logger


@dataclass(frozen=True)
class CustomBasisSet:
    """Represents a custom, internally defined basis set for a specific program.

    Attributes:
        program: The computational chemistry program this definition is for
                 (e.g., "orca", "qchem"). Stored lowercase.
        name: The common name for this basis set (e.g., "pcX-2"). Should be
              stored and looked up case-insensitively (lowercase).
        definitions: A dictionary mapping uppercase element symbols (str) to their
                     full, multi-line basis set definition strings (str).
    """

    program: str
    name: str
    definitions: dict[str, str] = field(repr=False)  # Avoid printing huge dicts

    def __post_init__(self) -> None:
        """Validate and normalize the fields."""
        if not self.program:
            raise ValidationError("CustomBasisSet program cannot be empty.")
        if not self.name:
            raise ValidationError("CustomBasisSet name cannot be empty.")
        if not self.definitions:
            raise ValidationError(f"CustomBasisSet '{self.program}/{self.name}' must have definitions.")

        # Use object.__setattr__ because frozen=True
        object.__setattr__(self, "program", self.program.lower())
        object.__setattr__(self, "name", self.name.lower())

        logger.debug(
            f"Initialized CustomBasisSet '{self.program}/{self.name}' with elements: {list(self.definitions.keys())}"
        )

    def __getitem__(self, element: str) -> str:
        """Returns the basis set definition string for a specific element."""
        if element not in self:
            raise KeyError(f"Element '{element}' not found in basis set '{self.program}/{self.name}'.")
        return self.definitions[element.upper()]

    def supports_element(self, element: str) -> bool:
        """Checks if a definition exists for the given element."""
        return element in self

    def __contains__(self, element: str) -> bool:
        """Checks if a definition exists for the given element."""
        return element.upper() in self.definitions

    @property
    def supported_elements(self) -> set[str]:
        """Returns a set of uppercase element symbols defined in this basis set."""
        return set(self.definitions.keys())

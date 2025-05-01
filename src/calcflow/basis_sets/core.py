from dataclasses import dataclass, field

from calcflow.exceptions import ValidationError
from calcflow.utils import logger


@dataclass(frozen=True)
class CustomBasisSet:
    """Represents a custom, internally defined basis set.

    Attributes:
        name: The common name for this basis set (e.g., "pcX-2"). Should be
              stored and looked up case-insensitively.
        definitions: A dictionary mapping uppercase element symbols (str) to their
                     full, multi-line basis set definition strings (str).
    """

    name: str
    definitions: dict[str, str] = field(repr=False)  # Avoid printing huge dicts

    def __post_init__(self) -> None:
        """Validate and normalize the definitions."""
        if not self.name:
            raise ValidationError("CustomBasisSet name cannot be empty.")
        if not self.definitions:
            raise ValidationError(f"CustomBasisSet '{self.name}' must have definitions.")

        object.__setattr__(self, "name", self.name.lower())  # Store name uppercased

        logger.debug(f"Initialized CustomBasisSet '{self.name}' with elements: {list(self.definitions.keys())}")

    def get_definition_for_element(self, element: str) -> str:
        """Returns the basis set definition string for a specific element."""
        if element.upper() not in self.definitions:
            raise ValueError(f"Element '{element}' not found in basis set '{self.name}'.")
        return self.definitions[element.upper()]

    def supports_element(self, element: str) -> bool:
        """Checks if a definition exists for the given element."""
        return element.upper() in self.definitions

    @property
    def supported_elements(self) -> set[str]:
        """Returns a set of uppercase element symbols defined in this basis set."""
        return set(self.definitions.keys())

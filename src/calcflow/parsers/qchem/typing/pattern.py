"""Pattern definition types for version-aware QChem output parsing.

This module provides dataclasses for defining version-specific regex patterns
that can be used to parse QChem output files across different versions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from re import Match, Pattern
from typing import Any, TypeVar

from calcflow.exceptions import InternalCodeError

# Type variable for regex pattern
T = TypeVar("T", bound=str)


@dataclass
class VersionSpec:
    """Specification for a version string."""

    major: int
    minor: int
    patch: int

    @classmethod
    def from_str(cls, version_str: str) -> "VersionSpec":
        """Parse a version string into a VersionSpec."""
        parts = version_str.split(".")
        return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]) if len(parts) > 2 else 0)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            other = VersionSpec.from_str(other)
        elif not isinstance(other, VersionSpec):
            return False
        return self.major == other.major and self.minor == other.minor and self.patch == other.patch

    def __lt__(self, other: str | object) -> bool:
        if isinstance(other, str):
            other = VersionSpec.from_str(other)
        elif not isinstance(other, VersionSpec):
            return False
        return (
            self.major < other.major
            or (self.major == other.major and self.minor < other.minor)
            or (self.major == other.major and self.minor == other.minor and self.patch < other.patch)
        )

    def __le__(self, other: str | object) -> bool:
        if isinstance(other, str):
            other = VersionSpec.from_str(other)
        elif not isinstance(other, VersionSpec):
            return False
        return self < other or self == other

    def __gt__(self, other: str | object) -> bool:
        if isinstance(other, str):
            other = VersionSpec.from_str(other)
        elif not isinstance(other, VersionSpec):
            return False
        return not self <= other

    def __ge__(self, other: str | object) -> bool:
        if isinstance(other, str):
            other = VersionSpec.from_str(other)
        elif not isinstance(other, VersionSpec):
            return False
        return not self < other

    @property
    def version(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __str__(self) -> str:
        return self.version

    def __repr__(self) -> str:
        return f"VersionSpec(major={self.major}, minor={self.minor}, patch={self.patch})"


@dataclass
class VersionedPattern:
    """A pattern with an associated version."""

    pattern: Pattern[str]
    version: VersionSpec | None
    transform: Callable[[Match[str]], Any] = lambda m: m.group(1)  # Default transform


@dataclass
class PatternDefinition:
    """Defines a field to extract with version-specific regex patterns.

    This class allows defining multiple patterns for different QChem versions
    that all map to the same result field.

    Args:
        field_name: The name of the field to update in the results
        required: Whether this field is required for valid parsing
        block_type: The type of block this pattern belongs to (e.g., "smd_summary")
        description: Human-readable description of what this pattern extracts
        versioned_patterns: Optional list of (pattern, version, transform) tuples to add
    """

    field_name: str  # Result field to update
    patterns: list[VersionedPattern] = field(default_factory=list)
    required: bool = False  # Is this field required?
    block_type: str | None = None  # e.g., "scf_iteration", "smd_summary"
    description: str = ""  # Human-readable description

    def __init__(
        self,
        field_name: str,
        *,
        required: bool = False,
        block_type: str | None = None,
        description: str = "",
        versioned_patterns: list[tuple[Pattern[str], VersionSpec | str | None, Callable[[Match[str]], Any] | None]]
        | None = None,
    ) -> None:
        """Initialize a PatternDefinition with optional versioned patterns."""
        self.field_name = field_name
        self.required = required
        self.block_type = block_type
        self.description = description
        self.patterns = []

        # Add any provided patterns
        if versioned_patterns:
            for pattern, version, transform in versioned_patterns:
                self.add_pattern(pattern, version, transform or (lambda m: m.group(1)))

    def add_pattern(
        self,
        pattern: Pattern[str],
        version: str | VersionSpec | None,
        transform: Callable[[Match[str]], Any] = lambda m: m.group(1),
    ) -> None:
        """Add a new pattern with version to this definition."""
        resolved_version: VersionSpec | None
        if isinstance(version, VersionSpec):
            resolved_version = version
        elif isinstance(version, str):
            resolved_version = VersionSpec.from_str(version)
        else:  # version is None
            resolved_version = None
        self.patterns.append(VersionedPattern(pattern=pattern, version=resolved_version, transform=transform))
        # Sort patterns: None versions first, then by major, minor, patch ascending
        self.patterns.sort(
            key=lambda vp: (0, -1, -1, -1)
            if vp.version is None
            else (1, vp.version.major, vp.version.minor, vp.version.patch)
        )

    def get_matching_pattern(self, version: VersionSpec) -> VersionedPattern | None:
        """Get the best matching pattern for the given version.

        If only one pattern exists and it has no version, it's used for all versions.
        If multiple patterns exist, use the one from the latest version that's <= current version.
        If no patterns match the version criteria, return None.
        """
        if not self.patterns:
            raise InternalCodeError("No patterns defined for this definition")

        # If only one pattern with no version, use it for all versions
        if len(self.patterns) == 1:
            return self.patterns[0]

        # Find the latest version <= current version
        best_pattern = None
        for pattern in self.patterns:
            if pattern.version <= version:
                best_pattern = pattern
            else:
                break

        if best_pattern is None:
            raise InternalCodeError(f"No pattern found for version {version}")
        return best_pattern

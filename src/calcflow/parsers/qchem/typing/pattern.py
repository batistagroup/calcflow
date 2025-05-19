"""Pattern definition types for version-aware QChem output parsing.

This module provides dataclasses for defining version-specific regex patterns
that can be used to parse QChem output files across different versions.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from re import Pattern
from typing import Any, TypeAlias

# Type for version specification
VersionSpec: TypeAlias = str | None


def compare_versions(version1: str, version2: str) -> int:
    """Compare two version strings.

    Args:
        version1: First version string in format "X.Y" or "X.Y.Z"
        version2: Second version string in format "X.Y" or "X.Y.Z"

    Returns:
        -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2
    """
    if not version1 and not version2:
        return 0
    if not version1:
        return -1
    if not version2:
        return 1

    # Handle potential suffixes like "beta" by splitting on non-numeric chars
    v1_clean = re.split(r"[^0-9.]", version1)[0]
    v2_clean = re.split(r"[^0-9.]", version2)[0]

    v1_parts = [int(part) for part in v1_clean.split(".")]
    v2_parts = [int(part) for part in v2_clean.split(".")]

    # Pad shorter version with zeros
    while len(v1_parts) < len(v2_parts):
        v1_parts.append(0)
    while len(v2_parts) < len(v1_parts):
        v2_parts.append(0)

    # Compare version components
    for v1, v2 in zip(v1_parts, v2_parts, strict=False):
        if v1 < v2:
            return -1
        if v1 > v2:
            return 1
    return 0  # Versions are equal


@dataclass
class VersionedPattern:
    """A pattern with an associated version."""

    pattern: Pattern
    version: VersionSpec = None  # None means applicable to any version
    transform: Callable[[re.Match], Any] = lambda m: m.group(1)  # Default transform


@dataclass
class PatternDefinition:
    """Defines a field to extract with version-specific regex patterns."""

    field_name: str  # Result field to update
    patterns: list[VersionedPattern] = field(default_factory=list)
    required: bool = False  # Is this field required?
    block_type: str | None = None  # e.g., "scf_iteration", "smd_summary"
    description: str = ""  # Human-readable description

    def add_pattern(
        self, pattern: Pattern, version: VersionSpec = None, transform: Callable[[re.Match], Any] = lambda m: m.group(1)
    ) -> None:
        """Add a new pattern with version to this definition."""
        self.patterns.append(VersionedPattern(pattern=pattern, version=version, transform=transform))

    def get_matching_pattern(self, version: str) -> VersionedPattern | None:
        """Get the best matching pattern for the given version.

        If only one pattern exists and it has no version, it's used for all versions.
        If multiple patterns exist, use the one from the latest version that's <= current version.
        If no patterns match the version criteria, return None.
        """
        if not self.patterns:
            return None

        # If only one pattern with no version, use it for all versions
        if len(self.patterns) == 1 and self.patterns[0].version is None:
            return self.patterns[0]

        # Find the latest version <= current version
        best_pattern = None
        for pattern in self.patterns:
            # Patterns with no version are fallbacks
            if pattern.version is None:
                if best_pattern is None:
                    best_pattern = pattern
                continue

            # Skip patterns with versions > current version
            if compare_versions(pattern.version, version) > 0:
                continue

            # Update best pattern if this one is newer
            if (
                best_pattern is None
                or best_pattern.version is None
                or (pattern.version is not None and compare_versions(pattern.version, best_pattern.version) > 0)
            ):
                best_pattern = pattern

        return best_pattern

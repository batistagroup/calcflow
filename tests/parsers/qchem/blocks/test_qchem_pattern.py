import re
from collections.abc import Callable
from re import Match, Pattern
from typing import Any

import pytest

from calcflow.exceptions import InternalCodeError
from calcflow.parsers.qchem.typing.pattern import PatternDefinition, VersionSpec

# Tests for VersionSpec


def test_version_spec_from_str() -> None:
    """Test creating VersionSpec from a string."""
    vs = VersionSpec.from_str("5.4.2")
    assert vs.major == 5
    assert vs.minor == 4
    assert vs.patch == 2
    vs_no_patch = VersionSpec.from_str("6.1")
    assert vs_no_patch.major == 6
    assert vs_no_patch.minor == 1
    assert vs_no_patch.patch == 0


def test_version_spec_equality() -> None:
    """Test equality comparisons for VersionSpec."""
    assert VersionSpec(5, 4, 2) == VersionSpec(5, 4, 2)
    assert VersionSpec(5, 4, 2) == "5.4.2"
    assert VersionSpec(5, 4, 0) == "5.4"
    assert VersionSpec(5, 4, 2) != VersionSpec(5, 4, 1)
    assert VersionSpec(5, 4, 2) != "5.4.1"
    assert VersionSpec(5, 4, 2) != VersionSpec(6, 0, 0)


def test_version_spec_less_than() -> None:
    """Test less than comparisons for VersionSpec."""
    assert VersionSpec(5, 4, 1) < VersionSpec(5, 4, 2)
    assert VersionSpec(5, 3, 9) < VersionSpec(5, 4, 0)
    assert VersionSpec(4, 9, 9) < VersionSpec(5, 0, 0)
    assert VersionSpec(5, 4, 1) < "5.4.2"
    assert not (VersionSpec(5, 4, 2) < VersionSpec(5, 4, 2))
    assert not (VersionSpec(5, 4, 2) < VersionSpec(5, 4, 1))


def test_version_spec_less_than_or_equal() -> None:
    """Test less than or equal comparisons for VersionSpec."""
    assert VersionSpec(5, 4, 1) <= VersionSpec(5, 4, 2)
    assert VersionSpec(5, 4, 2) <= VersionSpec(5, 4, 2)
    assert VersionSpec(5, 4, 2) <= "5.4.2"
    assert not (VersionSpec(5, 4, 3) <= VersionSpec(5, 4, 2))


def test_version_spec_greater_than() -> None:
    """Test greater than comparisons for VersionSpec."""
    assert VersionSpec(5, 4, 2) > VersionSpec(5, 4, 1)
    assert VersionSpec(5, 4, 0) > VersionSpec(5, 3, 9)
    assert VersionSpec(5, 0, 0) > VersionSpec(4, 9, 9)
    assert VersionSpec(5, 4, 2) > "5.4.1"
    assert not (VersionSpec(5, 4, 2) > VersionSpec(5, 4, 2))
    assert not (VersionSpec(5, 4, 1) > VersionSpec(5, 4, 2))


def test_version_spec_greater_than_or_equal() -> None:
    """Test greater than or equal comparisons for VersionSpec."""
    assert VersionSpec(5, 4, 2) >= VersionSpec(5, 4, 1)
    assert VersionSpec(5, 4, 2) >= VersionSpec(5, 4, 2)
    assert VersionSpec(5, 4, 2) >= "5.4.2"
    assert not (VersionSpec(5, 4, 1) >= VersionSpec(5, 4, 2))


def test_version_spec_str_and_repr() -> None:
    """Test string representation of VersionSpec."""
    vs = VersionSpec(5, 4, 2)
    assert str(vs) == "5.4.2"
    assert vs.version == "5.4.2"
    assert repr(vs) == "VersionSpec(major=5, minor=4, patch=2)"
    vs_no_patch = VersionSpec(6, 1, 0)
    assert str(vs_no_patch) == "6.1.0"  # Default patch is 0, so it should be included
    assert vs_no_patch.version == "6.1.0"
    assert repr(vs_no_patch) == "VersionSpec(major=6, minor=1, patch=0)"


# Tests for PatternDefinition


def test_pattern_definition_initialization_empty() -> None:
    """Test basic initialization of PatternDefinition."""
    pd = PatternDefinition(field_name="test_field", required=True, block_type="test_block", description="A test field")
    assert pd.field_name == "test_field"
    assert pd.required is True
    assert pd.block_type == "test_block"
    assert pd.description == "A test field"
    assert pd.patterns == []


def test_pattern_definition_add_pattern() -> None:
    """Test adding patterns to PatternDefinition."""
    pd = PatternDefinition(field_name="energy")
    pattern1 = re.compile(r"energy_v1: (\\d+)")
    transform1: Callable[[Match[str]], int] = lambda m: int(m.group(1))  # noqa: E731

    pd.add_pattern(pattern1, "1.0.0", transform1)
    assert len(pd.patterns) == 1
    assert pd.patterns[0].pattern == pattern1
    assert pd.patterns[0].version == VersionSpec(1, 0, 0)
    assert pd.patterns[0].transform == transform1

    pattern2 = re.compile(r"energy_v2: (\\d+\\.\\d+)")
    transform2: Callable[[Match[str]], float] = lambda m: float(m.group(1))  # noqa: E731
    pd.add_pattern(pattern2, "2.0.0", transform2)
    assert len(pd.patterns) == 2
    assert pd.patterns[1].pattern == pattern2
    assert pd.patterns[1].version == VersionSpec(2, 0, 0)
    assert pd.patterns[1].transform == transform2

    # Test default transform
    pattern3 = re.compile(r"default_transform_pattern: (.*)")
    pd.add_pattern(pattern3, "3.0.0")
    assert len(pd.patterns) == 3
    match_obj = re.match(pattern3, "default_transform_pattern: some_value")
    assert match_obj is not None
    assert pd.patterns[2].transform(match_obj) == "some_value"


def test_pattern_definition_initialization_with_patterns() -> None:
    """Test initializing PatternDefinition with versioned_patterns."""
    pattern1: Pattern[str] = re.compile(r"val_v1: (.*)")
    version1 = VersionSpec(1, 0, 0)
    transform1: Callable[[Match[str]], str] = lambda m: m.group(1).upper()  # noqa: E731

    pattern2: Pattern[str] = re.compile(r"val_v2: (\\d+)")
    version2 = VersionSpec(2, 0, 0)
    transform2: Callable[[Match[str]], int] = lambda m: int(m.group(1))  # noqa: E731

    versioned_patterns_data: list[tuple[Pattern[str], VersionSpec, Callable[[Match[str]], Any] | None]] = [
        (pattern1, version1, transform1),
        (pattern2, version2, transform2),
    ]

    pd = PatternDefinition(field_name="my_field", versioned_patterns=versioned_patterns_data)

    assert len(pd.patterns) == 2
    assert pd.patterns[0].pattern == pattern1
    assert pd.patterns[0].version == version1
    assert pd.patterns[0].transform == transform1

    assert pd.patterns[1].pattern == pattern2
    assert pd.patterns[1].version == version2
    assert pd.patterns[1].transform == transform2


def test_get_matching_pattern_no_patterns() -> None:
    """Test get_matching_pattern when no patterns are defined."""
    pd = PatternDefinition(field_name="empty_field")
    with pytest.raises(InternalCodeError, match="No patterns defined for this definition"):
        pd.get_matching_pattern(VersionSpec(1, 0, 0))


def test_get_matching_pattern_single_no_version() -> None:
    """Test get_matching_pattern with a single pattern having no version."""
    pd = PatternDefinition(field_name="generic_field")
    pattern = re.compile(r"generic: (.*)")
    pd.add_pattern(pattern, None)  # No version specified

    retrieved_pattern = pd.get_matching_pattern(VersionSpec(1, 0, 0))
    assert retrieved_pattern is not None
    assert retrieved_pattern.pattern == pattern
    assert retrieved_pattern.version is None  # Original VersionedPattern has None

    retrieved_pattern_v2 = pd.get_matching_pattern(VersionSpec(5, 5, 5))
    assert retrieved_pattern_v2 is not None
    assert retrieved_pattern_v2.pattern == pattern
    assert retrieved_pattern_v2.version is None


def test_get_matching_pattern_selects_correct_version() -> None:
    """Test get_matching_pattern selects the latest pattern <= current version."""
    pd = PatternDefinition(field_name="versioned_data")

    p_1_0_0 = re.compile(r"v1.0: (.*)")
    t_1_0_0: Callable[[Match[str]], str] = lambda m: m.group(1) + "_v1"  # noqa: E731
    pd.add_pattern(p_1_0_0, "1.0.0", t_1_0_0)

    p_1_2_0 = re.compile(r"v1.2: (.*)")
    t_1_2_0: Callable[[Match[str]], str] = lambda m: m.group(1) + "_v1.2"  # noqa: E731
    pd.add_pattern(p_1_2_0, "1.2.0", t_1_2_0)

    p_2_0_0 = re.compile(r"v2.0: (.*)")
    t_2_0_0: Callable[[Match[str]], str] = lambda m: m.group(1) + "_v2.0"  # noqa: E731
    pd.add_pattern(p_2_0_0, "2.0.0", t_2_0_0)

    # Test exact match
    match_1_2_0 = pd.get_matching_pattern(VersionSpec(1, 2, 0))
    assert match_1_2_0 is not None
    assert match_1_2_0.pattern == p_1_2_0
    assert match_1_2_0.version == VersionSpec(1, 2, 0)

    # Test version between defined patterns (should get the lower one)
    match_1_5_0 = pd.get_matching_pattern(VersionSpec(1, 5, 0))
    assert match_1_5_0 is not None
    assert match_1_5_0.pattern == p_1_2_0  # Should be 1.2.0 as it's latest <= 1.5.0
    assert match_1_5_0.version == VersionSpec(1, 2, 0)

    # Test version higher than all defined (should get the latest one)
    match_3_0_0 = pd.get_matching_pattern(VersionSpec(3, 0, 0))
    assert match_3_0_0 is not None
    assert match_3_0_0.pattern == p_2_0_0
    assert match_3_0_0.version == VersionSpec(2, 0, 0)

    # Test exact match for the first version
    match_1_0_0 = pd.get_matching_pattern(VersionSpec(1, 0, 0))
    assert match_1_0_0 is not None
    assert match_1_0_0.pattern == p_1_0_0
    assert match_1_0_0.version == VersionSpec(1, 0, 0)


def test_get_matching_pattern_no_suitable_version_found() -> None:
    """Test get_matching_pattern when no pattern version is <= current version."""
    pd = PatternDefinition(field_name="future_field")

    p_5_0_0 = re.compile(r"v5.0: (.*)")
    pd.add_pattern(p_5_0_0, "5.0.0")

    p_6_0_0 = re.compile(r"v6.0: (.*)")
    pd.add_pattern(p_6_0_0, "6.0.0")

    with pytest.raises(InternalCodeError, match="No pattern found for version 4.0.0"):
        pd.get_matching_pattern(VersionSpec(4, 0, 0))

    with pytest.raises(InternalCodeError, match="No pattern found for version 1.0.0"):  # Also check much older
        pd.get_matching_pattern(VersionSpec(1, 0, 0))


def test_pattern_definition_transform_usage() -> None:
    """Test that the transform function is correctly applied."""
    pd = PatternDefinition(field_name="transformed_value")

    pattern = re.compile(r"value: (\d+)")
    transform_func: Callable[[Match[str]], int] = lambda m: int(m.group(1)) * 10  # noqa: E731
    pd.add_pattern(pattern, "1.0.0", transform_func)

    current_version = VersionSpec(1, 0, 0)
    versioned_pattern = pd.get_matching_pattern(current_version)
    assert versioned_pattern is not None

    text_to_match = "value: 5"
    match = versioned_pattern.pattern.match(text_to_match)
    assert match is not None

    transformed_result = versioned_pattern.transform(match)
    assert transformed_result == 50


def test_pattern_definition_default_transform_if_none_provided_in_init() -> None:
    """Test default transform when None is passed for transform in versioned_patterns list during init."""
    pattern1: Pattern[str] = re.compile(r"field_val: (.*)")
    version1 = VersionSpec(1, 0, 0)

    versioned_patterns_data: list[tuple[Pattern[str], VersionSpec, Callable[[Match[str]], Any] | None]] = [
        (pattern1, version1, None),  # Pass None as transform
    ]

    pd = PatternDefinition(field_name="my_field_default_transform", versioned_patterns=versioned_patterns_data)

    assert len(pd.patterns) == 1
    vp = pd.patterns[0]
    assert vp.pattern == pattern1
    assert vp.version == version1

    match_obj = re.match(vp.pattern, "field_val: test_data")
    assert match_obj is not None
    assert vp.transform(match_obj) == "test_data"  # Default transform should extract group(1)

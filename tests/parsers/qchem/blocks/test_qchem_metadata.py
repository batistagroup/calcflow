from collections.abc import Iterator

import pytest

from calcflow.parsers.qchem.blocks.metadata import MetadataParser
from calcflow.parsers.qchem.typing import _MutableCalculationData


@pytest.fixture
def initial_data() -> _MutableCalculationData:
    """Provides a fresh _MutableCalculationData instance for each test."""
    return _MutableCalculationData(raw_output="")


@pytest.fixture
def metadata_parser() -> MetadataParser:
    """Provides an instance of the MetadataParser."""
    return MetadataParser()


# === Test Data ===

QCHEM_VERSION_LINE = " Q-Chem 6.2.1, Q-Chem, Inc., Pleasanton, CA (2024)"
HOST_LINE = " Host: login30"
RUN_DATE_LINE = " Q-Chem begins on Sun May  4 14:52:42 2025  "
OTHER_LINE = "This line contains no metadata."


# Iterator fixture not strictly needed as parse() only consumes current_line,
# but included for potential future use or consistency.
@pytest.fixture
def empty_iterator() -> Iterator[str]:
    return iter([])


# === Tests for matches() ===


@pytest.mark.parametrize(
    "line, expected_match",
    [
        (QCHEM_VERSION_LINE, True),
        (HOST_LINE, True),
        (RUN_DATE_LINE, True),
        (OTHER_LINE, False),
    ],
)
def test_metadata_matches(
    metadata_parser: MetadataParser, initial_data: _MutableCalculationData, line: str, expected_match: bool
) -> None:
    """Verify matches() correctly identifies various metadata lines."""
    assert metadata_parser.matches(line, initial_data) is expected_match


def test_metadata_matches_returns_false_if_already_parsed(
    metadata_parser: MetadataParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() returns False if the specific metadata type is already parsed."""
    # Test for version
    initial_data.parsed_meta_version = True
    assert metadata_parser.matches(QCHEM_VERSION_LINE, initial_data) is False
    initial_data.parsed_meta_version = False  # Reset

    # Test for host
    initial_data.parsed_meta_host = True
    assert metadata_parser.matches(HOST_LINE, initial_data) is False
    initial_data.parsed_meta_host = False  # Reset

    # Test for run_date
    initial_data.parsed_meta_run_date = True
    assert metadata_parser.matches(RUN_DATE_LINE, initial_data) is False


# === Tests for parse() ===


def test_metadata_parse_qchem_version(
    metadata_parser: MetadataParser, initial_data: _MutableCalculationData, empty_iterator: Iterator[str]
) -> None:
    """Test parsing the Q-Chem version line."""
    metadata_parser.parse(empty_iterator, QCHEM_VERSION_LINE, initial_data)

    assert initial_data.qchem_version == "6.2.1"
    assert initial_data.parsed_meta_version is True
    assert initial_data.host is None
    assert initial_data.parsed_meta_host is False
    assert initial_data.run_date is None
    assert initial_data.parsed_meta_run_date is False


def test_metadata_parse_host(
    metadata_parser: MetadataParser, initial_data: _MutableCalculationData, empty_iterator: Iterator[str]
) -> None:
    """Test parsing the host line."""
    metadata_parser.parse(empty_iterator, HOST_LINE, initial_data)

    assert initial_data.qchem_version is None
    assert initial_data.parsed_meta_version is False
    assert initial_data.host == "login30"
    assert initial_data.parsed_meta_host is True
    assert initial_data.run_date is None
    assert initial_data.parsed_meta_run_date is False


def test_metadata_parse_run_date(
    metadata_parser: MetadataParser, initial_data: _MutableCalculationData, empty_iterator: Iterator[str]
) -> None:
    """Test parsing the run date line."""
    metadata_parser.parse(empty_iterator, RUN_DATE_LINE, initial_data)

    assert initial_data.qchem_version is None
    assert initial_data.parsed_meta_version is False
    assert initial_data.host is None
    assert initial_data.parsed_meta_host is False
    assert initial_data.run_date == "Sun May  4 14:52:42 2025"
    assert initial_data.parsed_meta_run_date is True


def test_metadata_parse_does_not_advance_iterator(
    metadata_parser: MetadataParser, initial_data: _MutableCalculationData
) -> None:
    """Verify parse() does not consume lines from the iterator."""
    lines = ["line1", "line2"]
    lines_iter = iter(lines)
    metadata_parser.parse(lines_iter, HOST_LINE, initial_data)  # Parse using HOST_LINE
    # Iterator should be untouched
    assert next(lines_iter) == "line1"
    assert next(lines_iter) == "line2"
    with pytest.raises(StopIteration):
        next(lines_iter)


def test_metadata_parse_skips_already_parsed(
    metadata_parser: MetadataParser, initial_data: _MutableCalculationData, empty_iterator: Iterator[str]
) -> None:
    """Test that parse() does not overwrite already parsed data."""
    initial_data.host = "old_host"
    initial_data.parsed_meta_host = True

    # Attempt to parse the host line again
    metadata_parser.parse(empty_iterator, HOST_LINE, initial_data)

    # Host should remain unchanged
    assert initial_data.host == "old_host"
    assert initial_data.parsed_meta_host is True

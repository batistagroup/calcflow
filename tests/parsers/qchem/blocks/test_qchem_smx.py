import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.blocks.smx import SmxBlockParser
from calcflow.parsers.qchem.typing import _MutableCalculationData


@pytest.fixture
def initial_data() -> _MutableCalculationData:
    """Provides a fresh _MutableCalculationData instance for each test."""
    return _MutableCalculationData(raw_output="")


@pytest.fixture
def smx_parser() -> SmxBlockParser:
    """Provides an instance of the SmxBlockParser."""
    return SmxBlockParser()


# === Test Data ===

SMX_BLOCK_SIMPLE = """$smx
   solvent    water
$end
Some other line""".splitlines()

SMX_BLOCK_CASE_INSENSITIVE = """$SmX
   SoLvEnT    MeThAnOl
$END
Another line""".splitlines()

SMX_BLOCK_DIFFERENT_SOLVENT = """$smx
   solvent    toluene
$end""".splitlines()

SMX_BLOCK_EXTRA_SPACING = """$smx
     solvent      dmso  
$end""".splitlines()

SMX_BLOCK_EMPTY = """$smx
$end""".splitlines()

SMX_BLOCK_NO_SOLVENT_KEYWORD = """$smx
   some_other_param  value
$end""".splitlines()

SMX_BLOCK_UNTERMINATED = """$smx
   solvent    water
   """.splitlines()

NON_SMX_LINE = "This is not an smx block start line"
SMX_START_LINE = "$smx"


# === Tests for matches() ===


def test_smx_matches_start_line(smx_parser: SmxBlockParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() identifies the start of the $smx block."""
    assert smx_parser.matches(SMX_START_LINE, initial_data) is True


def test_smx_matches_start_line_case_insensitive(
    smx_parser: SmxBlockParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() identifies the start of the $smx block ignoring case."""
    assert smx_parser.matches("$SmX", initial_data) is True
    assert smx_parser.matches("$SMX", initial_data) is True


def test_smx_does_not_match_other_line(smx_parser: SmxBlockParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() returns False for non-matching lines."""
    assert smx_parser.matches(NON_SMX_LINE, initial_data) is False


def test_smx_matches_returns_false_if_already_parsed(
    smx_parser: SmxBlockParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() returns False if solvent_name is already parsed."""
    initial_data.solvent_name = "water"
    assert smx_parser.matches(SMX_START_LINE, initial_data) is False


def test_smx_matches_returns_true_if_not_yet_parsed(
    smx_parser: SmxBlockParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() returns True if solvent_name is not yet parsed."""
    # initial_data.solvent_name is None by default from the fixture
    assert smx_parser.matches(SMX_START_LINE, initial_data) is True


# === Tests for parse() ===


@pytest.mark.parametrize(
    "block_lines, start_line_str, expected_solvent, next_expected_line",
    [
        (SMX_BLOCK_SIMPLE, SMX_BLOCK_SIMPLE[0], "water", "Some other line"),
        (SMX_BLOCK_CASE_INSENSITIVE, SMX_BLOCK_CASE_INSENSITIVE[0], "methanol", "Another line"),
        (SMX_BLOCK_DIFFERENT_SOLVENT, SMX_BLOCK_DIFFERENT_SOLVENT[0], "toluene", None),
        (SMX_BLOCK_EXTRA_SPACING, SMX_BLOCK_EXTRA_SPACING[0], "dmso", None),
    ],
)
def test_smx_parse_valid_blocks(
    smx_parser: SmxBlockParser,
    initial_data: _MutableCalculationData,
    block_lines: list[str],
    start_line_str: str,
    expected_solvent: str | None,
    next_expected_line: str | None,
) -> None:
    """Test parsing various valid $smx blocks."""
    lines_iter = iter(block_lines[1:])  # Start after the $smx line
    smx_parser.parse(lines_iter, start_line_str, initial_data)

    assert initial_data.solvent_name == expected_solvent
    assert not initial_data.parsing_warnings

    if next_expected_line:
        assert next(lines_iter) == next_expected_line
    else:
        with pytest.raises(StopIteration):
            next(lines_iter)  # Ensure iterator is exhausted if no next line expected


def test_smx_parse_empty_block(smx_parser: SmxBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing an empty $smx block results in a warning."""
    lines_iter = iter(SMX_BLOCK_EMPTY[1:])
    smx_parser.parse(lines_iter, SMX_BLOCK_EMPTY[0], initial_data)

    assert initial_data.solvent_name is None
    assert len(initial_data.parsing_warnings) == 1
    assert "Solvent name not found" in initial_data.parsing_warnings[0]


def test_smx_parse_no_solvent_keyword(smx_parser: SmxBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing an $smx block without the 'solvent' keyword results in a warning."""
    lines_iter = iter(SMX_BLOCK_NO_SOLVENT_KEYWORD[1:])
    smx_parser.parse(lines_iter, SMX_BLOCK_NO_SOLVENT_KEYWORD[0], initial_data)

    assert initial_data.solvent_name is None
    assert len(initial_data.parsing_warnings) == 1
    assert "Solvent name not found" in initial_data.parsing_warnings[0]


def test_smx_parse_unterminated_block(smx_parser: SmxBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing raises ParsingError for an unterminated block."""
    lines_iter = iter(SMX_BLOCK_UNTERMINATED[1:])
    with pytest.raises(ParsingError, match="Unexpected end of file in \$smx block"):
        smx_parser.parse(lines_iter, SMX_BLOCK_UNTERMINATED[0], initial_data)

    # Check state after exception (should be partially parsed if possible, though smx is simple)
    # In this case, solvent name might be parsed before the unexpected end.
    assert initial_data.solvent_name == "water"
    assert not initial_data.parsing_warnings  # Warnings only on successful parse completion

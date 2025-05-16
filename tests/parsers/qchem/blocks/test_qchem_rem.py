import pytest

from calcflow.exceptions import ParsingError
from calcflow.parsers.qchem.blocks.rem import RemBlockParser
from calcflow.parsers.qchem.typing import _MutableCalculationData


@pytest.fixture
def initial_data() -> _MutableCalculationData:
    """Provides a fresh _MutableCalculationData instance for each test."""
    # Provide an empty string for raw_output as it's not relevant for these tests
    return _MutableCalculationData(raw_output="")


@pytest.fixture
def rem_parser() -> RemBlockParser:
    """Provides an instance of the RemBlockParser."""
    return RemBlockParser()


# === Test Data ===

REM_BLOCK_SIMPLE = """$rem
   METHOD          wb97x-d3
   BASIS           sto-3g
   UNRESTRICTED    False
$end
Some other line""".splitlines()

REM_BLOCK_CASE_INSENSITIVE = """$ReM
   method          PBE0
   BaSiS           def2-svp
$END
Another line""".splitlines()

REM_BLOCK_MISSING_BASIS = """$rem
   METHOD          b3lyp
   SOME_OTHER_VAR  true
$end""".splitlines()

REM_BLOCK_MISSING_METHOD = """$rem
   BASIS           6-31g*
$end""".splitlines()

REM_BLOCK_EMPTY = """$rem
$end""".splitlines()

REM_BLOCK_UNTERMINATED = """$rem
   METHOD          hf
   BASIS           sto-3g
   """.splitlines()

REM_BLOCK_WITH_SOLVENT = """$rem
   METHOD          wb97x-d3
   BASIS           sto-3g
   SOLVENT_METHOD  smd
$end
Yet another line""".splitlines()

REM_BLOCK_SOLVENT_CASE_INSENSITIVE = """$rem
   method          PBE0
   basis           def2-svp
   SoLvEnT_MeThoD  CpCm
$end""".splitlines()

NON_REM_LINE = "This is not a rem block start line"
REM_START_LINE = "$rem"


# === Tests for matches() ===


def test_rem_matches_start_line(rem_parser: RemBlockParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() identifies the start of the $rem block."""
    assert rem_parser.matches(REM_START_LINE, initial_data) is True


def test_rem_matches_start_line_case_insensitive(
    rem_parser: RemBlockParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() identifies the start of the $rem block ignoring case."""
    assert rem_parser.matches("$ReM", initial_data) is True


def test_rem_does_not_match_other_line(rem_parser: RemBlockParser, initial_data: _MutableCalculationData) -> None:
    """Verify matches() returns False for non-matching lines."""
    assert rem_parser.matches(NON_REM_LINE, initial_data) is False


def test_rem_matches_returns_false_if_already_parsed(
    rem_parser: RemBlockParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() returns False if essential data is already parsed."""
    initial_data.parsed_rem_block = True
    initial_data.solvent_method = "smd"
    assert rem_parser.matches(REM_START_LINE, initial_data) is False


def test_rem_matches_returns_true_if_basis_missing(
    rem_parser: RemBlockParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() returns True if basis is not yet parsed."""
    initial_data.parsed_meta_method = True
    initial_data.parsed_meta_basis = False
    assert rem_parser.matches(REM_START_LINE, initial_data) is True


def test_rem_matches_returns_true_if_method_missing(
    rem_parser: RemBlockParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() returns True if method is not yet parsed."""
    initial_data.parsed_meta_method = False
    initial_data.parsed_meta_basis = True
    assert rem_parser.matches(REM_START_LINE, initial_data) is True


def test_rem_matches_returns_true_if_solvent_method_missing(
    rem_parser: RemBlockParser, initial_data: _MutableCalculationData
) -> None:
    """Verify matches() returns True if solvent_method is not yet parsed but others are."""
    initial_data.parsed_meta_method = True
    initial_data.parsed_meta_basis = True
    initial_data.solvent_method = None  # Explicitly None, though default
    assert rem_parser.matches(REM_START_LINE, initial_data) is True


# === Tests for parse() ===


def test_rem_parse_simple(rem_parser: RemBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing a standard $rem block."""
    lines_iter = iter(REM_BLOCK_SIMPLE[1:])  # Start after the $rem line
    rem_parser.parse(lines_iter, REM_BLOCK_SIMPLE[0], initial_data)

    assert initial_data.calculation_method == "wb97x-d3"
    assert initial_data.basis_set == "sto-3g"
    assert initial_data.parsed_meta_method is True
    assert initial_data.parsed_meta_basis is True
    assert not initial_data.parsing_warnings
    # Ensure iterator stops after $end
    assert next(lines_iter) == "Some other line"


def test_rem_parse_case_insensitive(rem_parser: RemBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing a $rem block with different casing."""
    lines_iter = iter(REM_BLOCK_CASE_INSENSITIVE[1:])
    rem_parser.parse(lines_iter, REM_BLOCK_CASE_INSENSITIVE[0], initial_data)

    assert initial_data.calculation_method == "PBE0"
    assert initial_data.basis_set == "def2-svp"
    assert initial_data.parsed_meta_method is True
    assert initial_data.parsed_meta_basis is True
    assert not initial_data.parsing_warnings


def test_rem_parse_missing_basis(rem_parser: RemBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing when BASIS is missing."""
    lines_iter = iter(REM_BLOCK_MISSING_BASIS[1:])
    rem_parser.parse(lines_iter, REM_BLOCK_MISSING_BASIS[0], initial_data)

    assert initial_data.calculation_method == "b3lyp"
    assert initial_data.basis_set is None
    assert initial_data.parsed_meta_method is True
    assert initial_data.parsed_meta_basis is False
    assert len(initial_data.parsing_warnings) == 1
    assert "BASIS not found" in initial_data.parsing_warnings[0]


def test_rem_parse_missing_method(rem_parser: RemBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing when METHOD is missing."""
    lines_iter = iter(REM_BLOCK_MISSING_METHOD[1:])
    rem_parser.parse(lines_iter, REM_BLOCK_MISSING_METHOD[0], initial_data)

    assert initial_data.calculation_method is None
    assert initial_data.basis_set == "6-31g*"
    assert initial_data.parsed_meta_method is False
    assert initial_data.parsed_meta_basis is True
    assert len(initial_data.parsing_warnings) == 1
    assert "METHOD not found" in initial_data.parsing_warnings[0]


def test_rem_parse_empty_block(rem_parser: RemBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing an empty $rem block."""
    lines_iter = iter(REM_BLOCK_EMPTY[1:])
    rem_parser.parse(lines_iter, REM_BLOCK_EMPTY[0], initial_data)

    assert initial_data.calculation_method is None
    assert initial_data.basis_set is None
    assert initial_data.parsed_meta_method is False
    assert initial_data.parsed_meta_basis is False
    assert len(initial_data.parsing_warnings) == 2
    assert "METHOD not found" in initial_data.parsing_warnings[0]
    assert "BASIS not found" in initial_data.parsing_warnings[1]
    # Add check for solvent_method if a specific warning is expected for it in empty blocks
    # For now, the parser doesn't add a specific warning for missing solvent_method from empty block
    assert initial_data.solvent_method is None


def test_rem_parse_unterminated_block(rem_parser: RemBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing raises ParsingError for an unterminated block."""
    lines_iter = iter(REM_BLOCK_UNTERMINATED[1:])
    with pytest.raises(ParsingError, match=r"Unexpected end of file in \$rem block"):
        rem_parser.parse(lines_iter, REM_BLOCK_UNTERMINATED[0], initial_data)

    # Check state after exception (should be partially parsed)
    assert initial_data.calculation_method == "hf"
    assert initial_data.basis_set == "sto-3g"
    assert initial_data.parsed_meta_method is True
    assert initial_data.parsed_meta_basis is True
    assert not initial_data.parsing_warnings  # Warnings are only added on successful completion


def test_rem_parse_with_solvent_method(rem_parser: RemBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing a $rem block that includes SOLVENT_METHOD."""
    lines_iter = iter(REM_BLOCK_WITH_SOLVENT[1:])
    rem_parser.parse(lines_iter, REM_BLOCK_WITH_SOLVENT[0], initial_data)

    assert initial_data.calculation_method == "wb97x-d3"
    assert initial_data.basis_set == "sto-3g"
    assert initial_data.solvent_method == "smd"
    assert initial_data.parsed_meta_method is True
    assert initial_data.parsed_meta_basis is True
    # Assuming a new flag parsed_meta_solvent_method would be added to _MutableCalculationData
    # For now, we check the value directly and that no warnings specific to solvent_method appear.
    assert not initial_data.parsing_warnings
    assert next(lines_iter) == "Yet another line"


def test_rem_parse_solvent_method_case_insensitive(
    rem_parser: RemBlockParser, initial_data: _MutableCalculationData
) -> None:
    """Test SOLVENT_METHOD parsing is case insensitive for the value and keyword."""
    lines_iter = iter(REM_BLOCK_SOLVENT_CASE_INSENSITIVE[1:])
    rem_parser.parse(lines_iter, REM_BLOCK_SOLVENT_CASE_INSENSITIVE[0], initial_data)

    assert initial_data.calculation_method == "PBE0"
    assert initial_data.basis_set == "def2-svp"
    assert initial_data.solvent_method == "cpcm"  # Should be lowercased
    assert not initial_data.parsing_warnings


def test_rem_parse_missing_solvent_method(rem_parser: RemBlockParser, initial_data: _MutableCalculationData) -> None:
    """Test parsing when SOLVENT_METHOD is missing from the block."""
    # Using REM_BLOCK_SIMPLE which doesn't have SOLVENT_METHOD
    lines_iter = iter(REM_BLOCK_SIMPLE[1:])
    rem_parser.parse(lines_iter, REM_BLOCK_SIMPLE[0], initial_data)

    assert initial_data.calculation_method == "wb97x-d3"
    assert initial_data.basis_set == "sto-3g"
    assert initial_data.solvent_method is None
    # No warning should be generated just because it's missing, unless other conditions trigger it.
    assert not any("SOLVENT_METHOD not found" in w for w in initial_data.parsing_warnings)

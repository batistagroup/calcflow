from collections.abc import Sequence  # Required for NTOContribution fields if checking type more strictly

import pytest

from calcflow.parsers.qchem.typing import (
    CalculationData,
    NTOStateAnalysis,
)


def find_nto_state_analysis(
    analyses: Sequence[NTOStateAnalysis] | None, state_number: int, multiplicity: str
) -> NTOStateAnalysis | None:
    """Helper to find a specific NTO state analysis."""
    if not analyses:
        return None
    for analysis in analyses:
        if analysis.state_number == state_number and analysis.multiplicity == multiplicity:
            return analysis
    return None


def test_sa_nto_decomposition_singlet_2(parsed_tddft_pc2_data: CalculationData):
    """Test parsing of SA-NTO decomposition for Singlet state 2."""
    assert parsed_tddft_pc2_data.tddft_data is not None, "TddftData should be present"
    assert parsed_tddft_pc2_data.tddft_data.nto_state_analyses is not None, "NTO state analyses should be parsed"

    s2_analysis = find_nto_state_analysis(
        parsed_tddft_pc2_data.tddft_data.nto_state_analyses,
        state_number=2,
        multiplicity="Singlet",
    )

    assert s2_analysis is not None, "Singlet 2 NTO analysis not found"
    assert s2_analysis.state_number == 2
    assert s2_analysis.multiplicity == "Singlet"
    assert s2_analysis.omega_percent == pytest.approx(100.0)
    assert len(s2_analysis.contributions) == 1, "Singlet 2 should have 1 NTO contribution"

    contrib1 = s2_analysis.contributions[0]
    assert contrib1.hole_reference == "H"
    assert contrib1.hole_offset == 0
    assert contrib1.electron_reference == "L"
    assert contrib1.electron_offset == 1
    assert contrib1.coefficient == pytest.approx(-0.7048)
    assert contrib1.weight_percent == pytest.approx(99.3)


def test_sa_nto_decomposition_singlet_8(parsed_tddft_pc2_data: CalculationData):
    """Test parsing of SA-NTO decomposition for Singlet state 8."""
    assert parsed_tddft_pc2_data.tddft_data is not None
    assert parsed_tddft_pc2_data.tddft_data.nto_state_analyses is not None

    s8_analysis = find_nto_state_analysis(
        parsed_tddft_pc2_data.tddft_data.nto_state_analyses,
        state_number=8,
        multiplicity="Singlet",
    )

    assert s8_analysis is not None, "Singlet 8 NTO analysis not found"
    assert s8_analysis.state_number == 8
    assert s8_analysis.multiplicity == "Singlet"
    assert s8_analysis.omega_percent == pytest.approx(100.1)
    assert len(s8_analysis.contributions) == 2, "Singlet 8 should have 2 NTO contributions"

    # Contribution 1: H- 2 -> L+ 1:  0.6480 ( 84.0%)
    contrib1 = s8_analysis.contributions[0]
    assert contrib1.hole_reference == "H"
    assert contrib1.hole_offset == -2
    assert contrib1.electron_reference == "L"
    assert contrib1.electron_offset == 1
    assert contrib1.coefficient == pytest.approx(0.6480)
    assert contrib1.weight_percent == pytest.approx(84.0)

    # Contribution 2: H- 0 -> L+ 5:  0.2694 ( 14.5%)
    contrib2 = s8_analysis.contributions[1]
    assert contrib2.hole_reference == "H"
    assert contrib2.hole_offset == 0
    assert contrib2.electron_reference == "L"
    assert contrib2.electron_offset == 5
    assert contrib2.coefficient == pytest.approx(0.2694)
    assert contrib2.weight_percent == pytest.approx(14.5)


def test_sa_nto_decomposition_singlet_10(parsed_tddft_pc2_data: CalculationData):
    """Test parsing of SA-NTO decomposition for Singlet state 10."""
    assert parsed_tddft_pc2_data.tddft_data is not None
    assert parsed_tddft_pc2_data.tddft_data.nto_state_analyses is not None

    s10_analysis = find_nto_state_analysis(
        parsed_tddft_pc2_data.tddft_data.nto_state_analyses,
        state_number=10,
        multiplicity="Singlet",
    )

    assert s10_analysis is not None, "Singlet 10 NTO analysis not found"
    assert s10_analysis.state_number == 10
    assert s10_analysis.multiplicity == "Singlet"
    assert s10_analysis.omega_percent == pytest.approx(100.2)
    assert len(s10_analysis.contributions) == 3, "Singlet 10 should have 3 NTO contributions"

    # Contribution 1: H- 1 -> L+ 2: -0.5072 ( 51.5%)
    contrib1 = s10_analysis.contributions[0]
    assert contrib1.hole_reference == "H"
    assert contrib1.hole_offset == -1
    assert contrib1.electron_reference == "L"
    assert contrib1.electron_offset == 2
    assert contrib1.coefficient == pytest.approx(-0.5072)
    assert contrib1.weight_percent == pytest.approx(51.5)

    # Contribution 2: H- 0 -> L+ 5:  0.4636 ( 43.0%)
    contrib2 = s10_analysis.contributions[1]
    assert contrib2.hole_reference == "H"
    assert contrib2.hole_offset == 0
    assert contrib2.electron_reference == "L"
    assert contrib2.electron_offset == 5
    assert contrib2.coefficient == pytest.approx(0.4636)
    assert contrib2.weight_percent == pytest.approx(43.0)

    # Contribution 3: H- 2 -> L+ 1: -0.1545 (  4.8%)
    contrib3 = s10_analysis.contributions[2]
    assert contrib3.hole_reference == "H"
    assert contrib3.hole_offset == -2
    assert contrib3.electron_reference == "L"
    assert contrib3.electron_offset == 1
    assert contrib3.coefficient == pytest.approx(-0.1545)
    assert contrib3.weight_percent == pytest.approx(4.8)

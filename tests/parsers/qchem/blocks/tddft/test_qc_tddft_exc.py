import pytest

from calcflow.parsers.qchem.typing import CalculationData, ExcitedStateProperties, OrbitalTransition


def test_tddft_uks_pc2_state_2_properties(parsed_tddft_uks_pc2_data: CalculationData) -> None:
    """Test properties of State 2 from UKS TDDFT output."""
    assert parsed_tddft_uks_pc2_data.tddft is not None
    assert parsed_tddft_uks_pc2_data.tddft.tddft_states is not None

    state_2: ExcitedStateProperties | None = None
    for state in parsed_tddft_uks_pc2_data.tddft.tddft_states:
        if state.state_number == 2:
            state_2 = state
            break

    assert state_2 is not None, "State 2 not found in parsed TDDFT states."

    assert state_2.state_number == 2
    assert state_2.excitation_energy_ev == pytest.approx(7.5725)
    assert state_2.total_energy_au == pytest.approx(-76.16297039)
    assert state_2.multiplicity is None  # UKS TDDFT doesn't print multiplicity here
    assert state_2.trans_moment_x == pytest.approx(-0.0870)
    assert state_2.trans_moment_y == pytest.approx(-0.5098)
    assert state_2.trans_moment_z == pytest.approx(0.1093)
    assert state_2.oscillator_strength == pytest.approx(0.0518397940)

    assert len(state_2.transitions) == 2

    expected_transitions = [
        OrbitalTransition(from_label="D", from_idx=5, to_label="V", to_idx=1, amplitude=0.7041, is_alpha_spin=True),
        OrbitalTransition(from_label="D", from_idx=5, to_label="V", to_idx=1, amplitude=0.7041, is_alpha_spin=False),
    ]

    # Convert to a comparable representation (e.g., tuples) if order doesn't matter, or sort if it does
    # For simplicity, assuming order is fixed or check membership
    for et in expected_transitions:
        found = False
        for pt in state_2.transitions:
            if (
                et.from_idx == pt.from_idx
                and et.to_idx == pt.to_idx
                and et.is_alpha_spin == pt.is_alpha_spin
                and abs(et.amplitude - pt.amplitude) < 1e-4
            ):  # Using approx for amplitude
                found = True
                break
        assert found, f"Expected transition {et} not found or amplitude mismatch."


def test_tddft_uks_pc2_state_4_properties(parsed_tddft_uks_pc2_data: CalculationData) -> None:
    """Test properties of State 4 from UKS TDDFT output."""
    assert parsed_tddft_uks_pc2_data.tddft is not None
    assert parsed_tddft_uks_pc2_data.tddft.tddft_states is not None

    state_4: ExcitedStateProperties | None = None
    for state in parsed_tddft_uks_pc2_data.tddft.tddft_states:
        if state.state_number == 4:
            state_4 = state
            break

    assert state_4 is not None, "State 4 not found in parsed TDDFT states."

    assert state_4.state_number == 4
    assert state_4.excitation_energy_ev == pytest.approx(9.1931)
    assert state_4.total_energy_au == pytest.approx(-76.10341438)
    assert state_4.multiplicity is None
    assert state_4.trans_moment_x == pytest.approx(0.0000, abs=1e-4)  # Allow for -0.0000
    assert state_4.trans_moment_y == pytest.approx(0.0000, abs=1e-4)
    assert state_4.trans_moment_z == pytest.approx(0.0000, abs=1e-4)
    assert state_4.oscillator_strength == pytest.approx(0.0000000000)

    assert len(state_4.transitions) == 2

    expected_transitions = [
        OrbitalTransition(from_label="D", from_idx=4, to_label="V", to_idx=1, amplitude=0.6963, is_alpha_spin=True),
        OrbitalTransition(from_label="D", from_idx=4, to_label="V", to_idx=1, amplitude=-0.6963, is_alpha_spin=False),
    ]

    for et in expected_transitions:
        found = False
        for pt in state_4.transitions:
            if (
                et.from_idx == pt.from_idx
                and et.to_idx == pt.to_idx
                and et.is_alpha_spin == pt.is_alpha_spin
                and abs(et.amplitude - pt.amplitude) < 1e-4
            ):
                found = True
                break
        assert found, f"Expected transition {et} not found or amplitude mismatch."

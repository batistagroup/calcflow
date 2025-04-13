import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.blocks.charges import LOEWDIN_CHARGES_START_PAT, MULLIKEN_CHARGES_START_PAT, ChargesParser
from calcflow.parsers.orca.blocks.dipole import DipoleParser
from calcflow.parsers.orca.blocks.dispersion import DispersionParser
from calcflow.parsers.orca.blocks.geometry import GeometryParser
from calcflow.parsers.orca.blocks.orbitals import OrbitalsParser
from calcflow.parsers.orca.blocks.scf import ScfParser
from calcflow.parsers.orca.typing import (
    Atom,
    AtomicCharges,
    DipoleMomentData,
    DispersionCorrectionData,
    OrbitalData,
    ScfData,
    SectionParser,
    _MutableCalculationData,  # Reuse or adapt parts of this
)
from calcflow.utils import logger

# --- Regex Patterns (Add OPT-specific ones later) --- #
NORMAL_TERM_PAT = re.compile(r"\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*")
ERROR_TERM_PAT = re.compile(r"(TERMINATING THE PROGRAM|ORCA finished with error)")
# Add CYCLE_START_PAT, FINAL_EVAL_START_PAT, OPT_CONVERGED_PAT in Phase 1
CYCLE_START_PAT = re.compile(r"\*+\s+GEOMETRY OPTIMIZATION CYCLE\s+(\d+)\s+\*+")
FINAL_EVAL_START_PAT = re.compile(r"\*+\s+FINAL ENERGY EVALUATION AT THE STATIONARY POINT\s+\*+")
OPT_CONVERGED_PAT = re.compile(r"\*\*\*\s+THE OPTIMIZATION HAS CONVERGED\s+\*\*\*")


# --- Data Structures (Phase 0) --- #


@dataclass(frozen=True)
class GradientData:
    """Holds Cartesian gradient information for a specific optimization cycle."""

    gradients: Mapping[int, tuple[float, float, float]]  # Atom index -> (Gx, Gy, Gz)
    norm: float
    rms: float
    max: float


@dataclass(frozen=True)
class RelaxationStepData:
    """Holds geometry relaxation step details and convergence criteria."""

    energy_change: float | None = None
    rms_gradient: float | None = None
    max_gradient: float | None = None
    rms_step: float | None = None
    max_step: float | None = None
    converged_items: Mapping[str, bool] = field(default_factory=dict)  # e.g., {"Energy": True, "Gradient": False}
    trust_radius: float | None = None


@dataclass
class OptimizationCycleData:
    """Stores parsed data for a single geometry optimization cycle."""

    cycle_number: int
    geometry: Sequence[Atom] | None = None  # Geometry *at the start* of this cycle's calculation
    energy_eh: float | None = None  # Usually the SCF energy for this cycle
    scf_data: ScfData | None = None
    dispersion: DispersionCorrectionData | None = None
    gradient: GradientData | None = None
    relaxation_step: RelaxationStepData | None = None
    # Add other per-cycle properties if needed (e.g., orbitals, charges for each step)


@dataclass
class _MutableOptData:
    """Internal mutable container for accumulating optimization results during parsing."""

    raw_output: str
    termination_status: Literal["CONVERGED", "NOT_CONVERGED", "ERROR", "UNKNOWN"] = "UNKNOWN"
    input_geometry: Sequence[Atom] | None = None
    cycles: list[OptimizationCycleData] = field(default_factory=list)
    final_geometry: Sequence[Atom] | None = None  # Converged geometry
    final_energy_eh: float | None = None
    final_scf: ScfData | None = None
    final_orbitals: OrbitalData | None = None
    final_charges: list[AtomicCharges] = field(default_factory=list)
    final_dipole: DipoleMomentData | None = None
    final_dispersion: DispersionCorrectionData | None = None
    n_cycles: int = 0  # Track number of cycles parsed

    # --- Flags to manage parsing state ---
    # These might need adjustment as we add cycle logic
    parsed_input_geometry: bool = False
    # parsed_final_geometry: bool = False # Replaced by in_final_evaluation state
    # We'll need flags/logic for tracking state *within* cycles vs final eval
    # Add flags to track if specific components were parsed in the final block
    parsed_final_scf: bool = False
    parsed_final_orbitals: bool = False
    parsed_final_charges: bool = False
    parsed_final_dipole: bool = False
    parsed_final_dispersion: bool = False
    # Keep track of whether normal termination pattern was seen
    normal_termination_found: bool = False


@dataclass(frozen=True)
class OptimizationData:
    """Final immutable container for parsed ORCA geometry optimization results."""

    raw_output: str = field(repr=False)
    termination_status: Literal["CONVERGED", "NOT_CONVERGED", "ERROR", "UNKNOWN"]
    input_geometry: Sequence[Atom] | None
    cycles: Sequence[OptimizationCycleData]
    final_geometry: Sequence[Atom] | None
    final_energy_eh: float | None
    final_scf: ScfData | None
    final_orbitals: OrbitalData | None
    final_charges: list[AtomicCharges]
    final_dipole: DipoleMomentData | None
    final_dispersion: DispersionCorrectionData | None
    n_cycles: int

    @classmethod
    def from_mutable(cls, mutable_data: _MutableOptData) -> "OptimizationData":
        """Creates an immutable OptimizationData from the mutable version."""
        return cls(
            raw_output=mutable_data.raw_output,
            termination_status=mutable_data.termination_status,
            input_geometry=mutable_data.input_geometry,
            cycles=list(mutable_data.cycles),  # Ensure list copy
            final_geometry=mutable_data.final_geometry,
            final_energy_eh=mutable_data.final_energy_eh,
            final_scf=mutable_data.final_scf,
            final_orbitals=mutable_data.final_orbitals,
            final_charges=list(mutable_data.final_charges),  # Ensure list copy
            final_dipole=mutable_data.final_dipole,
            final_dispersion=mutable_data.final_dispersion,
            n_cycles=mutable_data.n_cycles,
        )

    def __repr__(self) -> str:
        # Basic repr, can be enhanced later
        final_energy_str = f"{self.final_energy_eh:.8f} Eh" if self.final_energy_eh is not None else "None"
        return (
            f"{type(self).__name__}("
            f"status='{self.termination_status}', "
            f"n_cycles={self.n_cycles}, "
            f"final_energy={final_energy_str}"
            f")"
        )


# --- Parser Registry (Phase 0 - Reusing SP Parsers) --- #
# We will add OPT-specific parsers (Gradient, RelaxationStep) later.
# The order matters for how blocks are detected.
OPT_PARSER_REGISTRY: Sequence[SectionParser] = [
    GeometryParser(),  # Handles input geom and geometry in final eval block
    ScfParser(),  # Handles SCF in cycles and final eval
    OrbitalsParser(),  # Handles orbitals in final eval (usually not per cycle)
    ChargesParser("Mulliken", MULLIKEN_CHARGES_START_PAT),  # Final eval
    ChargesParser("Loewdin", LOEWDIN_CHARGES_START_PAT),  # Final eval
    DipoleParser(),  # Final eval
    DispersionParser(),  # Handles dispersion in cycles and final eval
    # Add GradientParser here in Phase 2
    # Add RelaxationStepParser here in Phase 3
]


# --- Main Parsing Function (Phase 0 Stub) --- #
def parse_orca_opt_output(output: str) -> OptimizationData:
    """
    Parses the text output of an ORCA geometry optimization calculation.

    Args:
        output: The string content of the ORCA output file.

    Returns:
        An OptimizationData object containing the parsed results.

    Raises:
        ParsingError: If essential components cannot be parsed or critical errors occur.
    """
    lines = output.splitlines()
    line_iterator = iter(lines)
    results = _MutableOptData(raw_output=output)

    current_line_num = 0
    in_optimization_cycle: bool = False
    in_final_evaluation: bool = False
    current_cycle_data: OptimizationCycleData | None = None  # Mutable version not strictly needed here

    try:
        while True:
            try:
                line = next(line_iterator)
                current_line_num += 1
            except StopIteration:
                logger.debug("Reached end of input.")
                break

            # --- Cycle Boundary and Context Detection --- #
            match_cycle_start = CYCLE_START_PAT.search(line)
            match_final_eval = FINAL_EVAL_START_PAT.search(line)
            match_opt_converged = OPT_CONVERGED_PAT.search(line)

            if match_cycle_start:
                if current_cycle_data is not None:
                    # Finalize and store the previous cycle's data
                    results.cycles.append(current_cycle_data)
                    logger.debug(f"Stored data for cycle {current_cycle_data.cycle_number}")

                cycle_number = int(match_cycle_start.group(1))
                logger.info(f"Entering Optimization Cycle {cycle_number}")
                current_cycle_data = OptimizationCycleData(cycle_number=cycle_number)
                results.n_cycles += 1
                in_optimization_cycle = True
                in_final_evaluation = False
                continue  # Move to next line after handling cycle start

            if match_final_eval:
                if current_cycle_data is not None:
                    # Store the last cycle's data before final eval
                    results.cycles.append(current_cycle_data)
                    logger.debug(f"Stored data for final cycle {current_cycle_data.cycle_number}")
                    current_cycle_data = None  # Clear current cycle

                logger.info("Entering Final Energy Evaluation block.")
                in_optimization_cycle = False
                in_final_evaluation = True
                continue  # Move to next line

            if match_opt_converged:
                logger.info("Optimization Converged message found.")
                # Set final status only if no error occurred
                if results.termination_status != "ERROR":
                    results.termination_status = "CONVERGED"
                continue  # Move to next line

            # --- Termination Status (Refined) ---
            if NORMAL_TERM_PAT.search(line):
                logger.debug("Found Normal Termination pattern.")
                results.normal_termination_found = True
                # Don't break, continue parsing
            elif ERROR_TERM_PAT.search(line):
                # Error overrides any other status, including CONVERGED
                logger.error("Found Error Termination pattern.")
                results.termination_status = "ERROR"
                in_optimization_cycle = False  # Stop parsing cycle/final data on error
                in_final_evaluation = False
                # Optionally break or continue parsing for final error messages

            # --- Parser Matching Logic (Phase 1 - Context Aware) ---
            # Only run parsers if not in an error state
            if results.termination_status != "ERROR":
                parser_found = False
                for parser in OPT_PARSER_REGISTRY:
                    try:
                        # Always create a fresh temp data object for each potential match
                        # to avoid state pollution between parser checks/runs.
                        temp_sp_data = _MutableCalculationData(raw_output=results.raw_output)

                        if parser.matches(line, temp_sp_data):
                            # Found a matching block, now parse and route the data
                            logger.debug(
                                f"Parser {type(parser).__name__} matched line {current_line_num}. Context: Cycle={in_optimization_cycle}, Final={in_final_evaluation}"
                            )
                            parser.parse(line_iterator, line, temp_sp_data)

                            # --- Route Parsed Data --- #
                            # Special handling for the very first geometry parsed
                            if isinstance(parser, GeometryParser) and temp_sp_data.input_geometry:
                                if not results.parsed_input_geometry:
                                    results.input_geometry = temp_sp_data.input_geometry
                                    results.parsed_input_geometry = True
                                    logger.debug(
                                        f"Captured input geometry (from cycle/block starting near line {current_line_num})."
                                    )
                                # Also store it in the current cycle if applicable
                                if in_optimization_cycle and current_cycle_data is not None:
                                    current_cycle_data.geometry = temp_sp_data.input_geometry

                            # Handle other parsers based on context
                            elif in_optimization_cycle and current_cycle_data is not None:
                                # Store in current cycle data
                                if isinstance(parser, ScfParser) and temp_sp_data.scf:
                                    current_cycle_data.scf_data = temp_sp_data.scf
                                    current_cycle_data.energy_eh = (
                                        temp_sp_data.final_energy_eh
                                    )  # Often SCF energy is the cycle energy
                                elif isinstance(parser, DispersionParser) and temp_sp_data.dispersion_correction:
                                    current_cycle_data.dispersion = temp_sp_data.dispersion_correction
                                # Add other relevant per-cycle parsers if needed (e.g., charges)

                            elif in_final_evaluation:
                                # Store in final results fields
                                if (
                                    isinstance(parser, GeometryParser)
                                    and temp_sp_data.input_geometry
                                    and not results.final_geometry
                                ):
                                    results.final_geometry = temp_sp_data.input_geometry
                                elif (
                                    isinstance(parser, ScfParser) and temp_sp_data.scf and not results.parsed_final_scf
                                ):
                                    results.final_scf = temp_sp_data.scf
                                    results.final_energy_eh = (
                                        temp_sp_data.final_energy_eh
                                    )  # Assume final SCF energy is THE final energy
                                    results.parsed_final_scf = True
                                elif (
                                    isinstance(parser, OrbitalsParser)
                                    and temp_sp_data.orbitals
                                    and not results.parsed_final_orbitals
                                ):
                                    results.final_orbitals = temp_sp_data.orbitals
                                    results.parsed_final_orbitals = True
                                elif (
                                    isinstance(parser, ChargesParser)
                                    and temp_sp_data.atomic_charges
                                    and not results.parsed_final_charges
                                ):
                                    # Assume we only care about the first set of charges found in final block
                                    results.final_charges = list(temp_sp_data.atomic_charges)
                                    results.parsed_final_charges = True  # Avoid appending multiple charge sets
                                elif (
                                    isinstance(parser, DipoleParser)
                                    and temp_sp_data.dipole_moment
                                    and not results.parsed_final_dipole
                                ):
                                    results.final_dipole = temp_sp_data.dipole_moment
                                    results.parsed_final_dipole = True
                                elif (
                                    isinstance(parser, DispersionParser)
                                    and temp_sp_data.dispersion_correction
                                    and not results.parsed_final_dispersion
                                ):
                                    results.final_dispersion = temp_sp_data.dispersion_correction
                                    results.parsed_final_dispersion = True

                            parser_found = True
                            break  # Important: Only one parser handles the start of a block

                    except ParsingError as e:
                        logger.error(f"Parser {type(parser).__name__} failed critically: {e}", exc_info=True)
                        raise
                    except Exception as e:
                        logger.error(f"Unexpected error in {type(parser).__name__}: {e}", exc_info=True)
                        # Continue parsing for robustness, but log error

                if parser_found:
                    continue  # Skip to next line if handled by a block parser

            # Add line-specific parsing here if needed (e.g., final converged energy outside blocks)

    except ParsingError:
        raise  # Re-raise critical parsing errors
    except Exception as e:
        logger.critical(f"Unexpected error in main parsing loop at line ~{current_line_num}: {e}", exc_info=True)
        results.termination_status = "ERROR"  # Mark as error on unexpected failure
        raise ParsingError(f"An unexpected error occurred during parsing: {e}") from e
    finally:
        # Ensure the last cycle's data is stored if parsing finishes mid-cycle
        if current_cycle_data is not None:
            results.cycles.append(current_cycle_data)
            logger.debug(f"Stored data for final partial cycle {current_cycle_data.cycle_number}")

    # --- Final Checks (Phase 1 - Improved Status) ---
    if results.input_geometry is None:
        logger.error("Input geometry block was not found or parsed.")
        raise ParsingError("Input geometry block was not found in the output file.")

    # Refine termination status based on convergence and normal termination
    if results.termination_status == "UNKNOWN":
        if results.normal_termination_found:
            # Finished normally but didn't hit the CONVERGED message
            results.termination_status = "NOT_CONVERGED"
            logger.warning("Optimization terminated normally but did not converge.")
        else:
            # Did not finish normally and status is still unknown -> Error
            logger.error("Termination status unknown and normal termination not found. Setting status to ERROR.")
            results.termination_status = "ERROR"
    elif results.termination_status == "CONVERGED" and not results.normal_termination_found:
        # This case might be unlikely but indicates convergence message seen but not normal termination
        logger.warning(
            "Convergence message found, but ORCA did not terminate normally. Status kept as CONVERGED, but review output."
        )

    logger.info(f"ORCA OPT parsing finished. Status: {results.termination_status}, Cycles: {results.n_cycles}")
    # Convert back to immutable dataclass
    return OptimizationData.from_mutable(results)

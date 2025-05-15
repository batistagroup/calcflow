import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from calcflow.exceptions import ParsingError
from calcflow.parsers.orca.blocks.charges import LOEWDIN_CHARGES_START_PAT, MULLIKEN_CHARGES_START_PAT, ChargesParser
from calcflow.parsers.orca.blocks.dipole import DipoleParser
from calcflow.parsers.orca.blocks.dispersion import DispersionParser
from calcflow.parsers.orca.blocks.geometry import GeometryParser
from calcflow.parsers.orca.blocks.gradient import GradientParser
from calcflow.parsers.orca.blocks.orbitals import OrbitalsParser
from calcflow.parsers.orca.blocks.relaxation import RelaxationStepParser
from calcflow.parsers.orca.blocks.scf import ScfParser
from calcflow.parsers.orca.typing import (
    Atom,
    AtomicCharges,
    DipoleMoment,
    DispersionCorrectionData,
    OptimizationCycleData,
    OrbitalsSet,
    ScfData,
    SectionParser,
    _MutableCalculationData,
    _MutableOptData,
)
from calcflow.utils import logger

NORMAL_TERM_PAT = re.compile(r"\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*")
ERROR_TERM_PAT = re.compile(r"(TERMINATING THE PROGRAM|ORCA finished with error)")
CYCLE_START_PAT = re.compile(r"\*+\s+GEOMETRY OPTIMIZATION CYCLE\s+(\d+)\s+\*+")
FINAL_EVAL_START_PAT = re.compile(r"\*+\s+FINAL ENERGY EVALUATION AT THE STATIONARY POINT\s+\*+")
OPT_CONVERGED_PAT = re.compile(r"\*\*\*\s+THE OPTIMIZATION HAS CONVERGED\s+\*\*\*")


@dataclass(frozen=True)
class OptimizationData:
    raw_output: str = field(repr=False)
    termination_status: Literal["CONVERGED", "NOT_CONVERGED", "ERROR", "UNKNOWN"]
    input_geometry: Sequence[Atom] | None
    cycles: Sequence[OptimizationCycleData]
    final_geometry: Sequence[Atom] | None
    final_energy_eh: float | None
    final_scf: ScfData | None
    final_orbitals: OrbitalsSet | None
    final_charges: list[AtomicCharges]
    final_dipole: DipoleMoment | None
    final_dispersion: DispersionCorrectionData | None
    n_cycles: int

    @classmethod
    def from_mutable(cls, mutable_data: _MutableOptData) -> "OptimizationData":
        return cls(
            raw_output=mutable_data.raw_output,
            termination_status=mutable_data.termination_status,
            input_geometry=mutable_data.input_geometry,
            cycles=list(mutable_data.cycles),
            final_geometry=mutable_data.final_geometry,
            final_energy_eh=mutable_data.final_energy_eh,
            final_scf=mutable_data.final_scf,
            final_orbitals=mutable_data.final_orbitals,
            final_charges=list(mutable_data.final_charges),
            final_dipole=mutable_data.final_dipole,
            final_dispersion=mutable_data.final_dispersion,
            n_cycles=mutable_data.n_cycles,
        )

    def __repr__(self) -> str:
        final_energy_str = f"{self.final_energy_eh:.8f} Eh" if self.final_energy_eh is not None else "None"
        return (
            f"{type(self).__name__}("
            f"status='{self.termination_status}', "
            f"n_cycles={self.n_cycles}, "
            f"final_energy={final_energy_str}"
            f")"
        )


OPT_PARSER_REGISTRY: Sequence[SectionParser | GradientParser | RelaxationStepParser] = [
    GeometryParser(),
    ScfParser(),
    GradientParser(),
    OrbitalsParser(),
    ChargesParser("Mulliken", MULLIKEN_CHARGES_START_PAT),
    ChargesParser("Loewdin", LOEWDIN_CHARGES_START_PAT),
    DipoleParser(),
    DispersionParser(),
    RelaxationStepParser(),
]


def parse_orca_opt_output(output: str) -> OptimizationData:
    lines = output.splitlines()
    line_iterator = iter(lines)
    results = _MutableOptData(raw_output=output)

    current_line_num = 0
    in_optimization_cycle: bool = False
    in_final_evaluation: bool = False
    current_cycle_data: OptimizationCycleData | None = None

    try:
        while True:
            try:
                line = next(line_iterator)
                current_line_num += 1
            except StopIteration:
                break

            match_cycle_start = CYCLE_START_PAT.search(line)
            match_final_eval = FINAL_EVAL_START_PAT.search(line)
            match_opt_converged = OPT_CONVERGED_PAT.search(line)

            if match_cycle_start:
                if current_cycle_data is not None:
                    results.cycles.append(current_cycle_data)

                cycle_number = int(match_cycle_start.group(1))
                logger.info(f"Entering Optimization Cycle {cycle_number}")
                current_cycle_data = OptimizationCycleData(cycle_number=cycle_number)
                results.n_cycles += 1
                in_optimization_cycle = True
                in_final_evaluation = False
                continue

            if match_final_eval:
                if current_cycle_data is not None:
                    results.cycles.append(current_cycle_data)
                    current_cycle_data = None

                logger.info("Entering Final Energy Evaluation block.")
                in_optimization_cycle = False
                in_final_evaluation = True
                continue

            if match_opt_converged:
                logger.info("Optimization Converged message found.")
                if results.termination_status != "ERROR":
                    results.termination_status = "CONVERGED"
                continue

            if NORMAL_TERM_PAT.search(line):
                results.normal_termination_found = True
            elif ERROR_TERM_PAT.search(line):
                logger.error("Found Error Termination pattern.")
                results.termination_status = "ERROR"
                in_optimization_cycle = False
                in_final_evaluation = False

            if results.termination_status != "ERROR":
                parser_found = False
                for parser in OPT_PARSER_REGISTRY:
                    try:
                        temp_sp_data = _MutableCalculationData(raw_output=results.raw_output)

                        if isinstance(parser, GradientParser | RelaxationStepParser):
                            if parser.matches(line, results):
                                parser.parse(line_iterator, line, results, current_cycle_data)
                                parser_found = True
                                break
                        elif parser.matches(line, temp_sp_data):
                            parser.parse(line_iterator, line, temp_sp_data)

                            if in_final_evaluation:
                                if isinstance(parser, GeometryParser) and not results.final_geometry:
                                    if temp_sp_data.input_geometry:
                                        results.final_geometry = temp_sp_data.input_geometry
                                elif (
                                    isinstance(parser, ScfParser) and temp_sp_data.scf and not results.parsed_final_scf
                                ):
                                    results.final_scf = temp_sp_data.scf
                                    results.final_energy_eh = temp_sp_data.scf.energy_eh
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
                                    results.final_charges = list(temp_sp_data.atomic_charges)
                                    results.parsed_final_charges = True
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

                            elif in_optimization_cycle and current_cycle_data is not None:
                                if isinstance(parser, GeometryParser) and temp_sp_data.input_geometry:
                                    current_cycle_data.geometry = temp_sp_data.input_geometry
                                    if not results.parsed_input_geometry:
                                        results.input_geometry = temp_sp_data.input_geometry
                                        results.parsed_input_geometry = True
                                elif isinstance(parser, ScfParser) and temp_sp_data.scf:
                                    current_cycle_data.scf_data = temp_sp_data.scf
                                    current_cycle_data.energy_eh = (
                                        temp_sp_data.final_energy_eh
                                        if temp_sp_data.final_energy_eh is not None
                                        else temp_sp_data.scf.energy_eh
                                    )
                                elif isinstance(parser, DispersionParser) and temp_sp_data.dispersion_correction:
                                    current_cycle_data.dispersion = temp_sp_data.dispersion_correction

                            elif (
                                isinstance(parser, GeometryParser)
                                and temp_sp_data.input_geometry
                                and not results.parsed_input_geometry
                            ):
                                results.input_geometry = temp_sp_data.input_geometry
                                results.parsed_input_geometry = True

                            parser_found = True
                            break

                    except ParsingError as e:
                        logger.error(f"Parser {type(parser).__name__} failed critically: {e}", exc_info=True)
                        raise
                    except Exception as e:
                        logger.error(f"Unexpected error in {type(parser).__name__}: {e}", exc_info=True)

                if parser_found:
                    continue

    except ParsingError:
        raise
    except Exception as e:
        logger.critical(f"Unexpected error in main parsing loop at line ~{current_line_num}: {e}", exc_info=True)
        results.termination_status = "ERROR"
        raise ParsingError(f"An unexpected error occurred during parsing: {e}") from e
    finally:
        if current_cycle_data is not None:
            results.cycles.append(current_cycle_data)

    if results.input_geometry is None:
        logger.error("Input geometry block was not found or parsed.")
        raise ParsingError("Input geometry block was not found in the output file.")

    if results.termination_status == "UNKNOWN":
        if results.normal_termination_found:
            results.termination_status = "NOT_CONVERGED"
            logger.warning("Optimization terminated normally but did not converge.")
        else:
            logger.error("Termination status unknown and normal termination not found. Setting status to ERROR.")
            results.termination_status = "ERROR"
    elif results.termination_status == "CONVERGED" and not results.normal_termination_found:
        logger.warning(
            "Convergence message found, but ORCA did not terminate normally. Status kept as CONVERGED, but review output."
        )

    logger.info(f"ORCA OPT parsing finished. Status: {results.termination_status}, Cycles: {results.n_cycles}")
    return OptimizationData.from_mutable(results)

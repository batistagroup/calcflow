"""SCF Parser module for Q-Chem output files.

This module provides a pattern-based approach to parsing SCF calculation blocks
in Q-Chem output files. It uses a version-aware registry of regex patterns to extract data
in a flexible manner that adapts to different Q-Chem versions.
"""

import re

from calcflow.parsers.qchem.typing import (
    LineIterator,
    ScfIteration,
    ScfResults,
    SectionParser,
    SmdResults,
    _MutableCalculationData,
)
from calcflow.parsers.qchem.typing.pattern import (
    PatternDefinition,
)
from calcflow.utils import logger

# --- Regex Patterns --- #

# Start marker for the SCF section
SCF_START_PAT = re.compile(r"^\s*General SCF calculation program by")

# SCF Iteration line
SCF_ITER_PAT = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+([\d\.eE+-]+)")
ROOTHAAN_STEP_PAT = re.compile(r"Roothaan Step")

# SCF Convergence marker
SCF_CONVERGENCE_PAT = re.compile(r"Convergence criterion met")

# SCF Iteration table header and end markers
SCF_ITER_TABLE_HEADER_PAT = re.compile(r"^\s*Cycle\s+Energy\s+DIIS error")
SCF_ITER_TABLE_END_PAT = re.compile(r"^\s*-{20,}")  # Dashed line after table

# Energy patterns
SCF_FINAL_ENERGY_PAT = re.compile(r"^\s*SCF\s+energy\s*=\s*(-?\d+\.\d+)")
TOTAL_ENERGY_PAT = re.compile(r"^\s*Total energy\s*=\s*(-?\d+\.\d+)")

# SMD Summary Patterns
SMD_SUMMARY_START_PAT = re.compile(r"^\s*Summary of SMD free energies:")
G_PCM_PAT = re.compile(r"^\s*G_PCM\s*=\s*(-?\d+\.\d+)\s*kcal/mol")
G_CDS_PAT = re.compile(r"^\s*G_CDS\s*=\s*(-?\d+\.\d+)\s*kcal/mol")
G_ENP_PAT = re.compile(r"^\s*G_ENP\s*=\s*(-?\d+\.\d+)\s*a\.u\.")
G_TOT_PAT = re.compile(r"^\s*G\(tot\)\s*=\s*(-?\d+\.\d+)\s*a\.u\.")

# MOM Specific Patterns
MOM_ACTIVE_PAT = re.compile(r"^\s*Maximum Overlap Method Active")
IMOM_METHOD_PAT = re.compile(r"^\s*IMOM method")
MOM_OVERLAP_PAT = re.compile(r"^\s*MOM overlap:\s+(-?\d+\.\d+)\s+/\s+(-?\d+\.?\d*)")

# End marker heuristics
SCF_BLOCK_END_HEURISTIC_PAT = re.compile(r"Mulliken Net Atomic Charges|Multipole Moments|^\s*-{60}")
NORMAL_TERM_PAT = re.compile(r"Thank you very much for using Q-Chem")

# Pattern to detect the start of orbital energies section
ORBITAL_HEADER_PAT = re.compile(r"^\s*Orbital Energies \(a\.u\.\)")

# Patterns to detect the start of TDDFT excitation blocks
TDA_HEADER_PAT = re.compile(r"^\s*TDDFT/TDA\s+Excitation\s+Energies\s*$")
TDDFT_HEADER_PAT = re.compile(r"^\s*TDDFT\s+Excitation\s+Energies\s*$")


# Using PatternDefinition from calcflow.parsers.qchem.typing.pattern


# Define a pattern for "Total energy in the final basis set" for QChem 5.4
TOTAL_ENERGY_FINAL_BASIS_PAT = re.compile(r"^\s*Total energy in the final basis set\s*=\s*(-?\d+\.\d+)")

# Registry of patterns for SCF data extraction
SCF_PATTERNS = [
    # SCF energy pattern - consistent across versions
    PatternDefinition(
        field_name="scf_energy",
        required=True,
        description="Final SCF energy value",
        versioned_patterns=[
            (SCF_FINAL_ENERGY_PAT, None, lambda m: float(m.group(1))),
        ]
    ),
    
    # Final energy pattern - version dependent
    PatternDefinition(
        field_name="final_energy",
        description="Total energy including corrections",
        versioned_patterns=[
            # QChem 6.0+ uses "Total energy"
            (TOTAL_ENERGY_PAT, "6.0", lambda m: float(m.group(1))),
            # QChem 5.x uses "Total energy in the final basis set"
            (TOTAL_ENERGY_FINAL_BASIS_PAT, "5.4", lambda m: float(m.group(1))),
        ]
    ),
    
    # SMD patterns
    PatternDefinition(
        field_name="smd_g_pcm_kcal_mol",
        block_type="smd_summary",
        description="SMD polarization energy component",
        versioned_patterns=[
            (G_PCM_PAT, None, lambda m: float(m.group(1))),
        ]
    ),
    
    PatternDefinition(
        field_name="smd_g_cds_kcal_mol",
        block_type="smd_summary",
        description="SMD non-electrostatic energy component",
        versioned_patterns=[
            (G_CDS_PAT, None, lambda m: float(m.group(1))),
        ]
    ),
    
    PatternDefinition(
        field_name="smd_g_enp_au",
        block_type="smd_summary",
        description="SCF energy in solvent (E_SCF + G_PCM)",
        versioned_patterns=[
            (G_ENP_PAT, None, lambda m: float(m.group(1))),
        ]
    ),
    
    PatternDefinition(
        field_name="smd_g_tot_au",
        block_type="smd_summary",
        description="Total free energy in solution (G_ENP + G_CDS)",
        versioned_patterns=[
            (G_TOT_PAT, None, lambda m: float(m.group(1))),
        ]
    ),
    
    # MOM patterns
    PatternDefinition(
        field_name="mom_active",
        block_type="mom",
        description="MOM is active in this calculation",
        versioned_patterns=[
            (MOM_ACTIVE_PAT, None, lambda _: True),
        ]
    ),
    
    PatternDefinition(
        field_name="mom_method_type",
        block_type="mom",
        description="IMOM method is being used",
        versioned_patterns=[
            (IMOM_METHOD_PAT, None, lambda _: "IMOM"),
        ]
    ),
    
    PatternDefinition(
        field_name="mom_overlap",
        block_type="mom",
        description="MOM overlap with current and target orbitals",
        versioned_patterns=[
            (MOM_OVERLAP_PAT, None, lambda m: (float(m.group(1)), float(m.group(2)))),
        ]
    ),
]


class ScfParser(SectionParser):
    """
    Parses the SCF calculation block using a pattern-based approach.

    This parser processes SCF iteration tables, final energies, and SMD summary blocks
    in a flexible manner that doesn't rely on specific line ordering.
    """

    def matches(self, line: str, current_data: _MutableCalculationData) -> bool:
        """Check if the line indicates the start of the SCF calculation header."""
        return (SCF_START_PAT.search(line) is not None) and not current_data.parsed_scf

    def parse(self, iterator: LineIterator, current_line: str, results: _MutableCalculationData) -> None:
        """
        Parse SCF iterations, final results, and SMD summary.

        This implementation uses a pattern registry approach to decouple pattern matching
        from parsing logic, making it more robust to variations in Q-Chem output format.
        """
        logger.debug("Starting parsing of SCF block.")

        # Initialize tracking variables
        all_iterations: list[ScfIteration] = []
        converged: bool = False
        scf_energy: float | None = None
        in_smd_summary_block: bool = False
        smd_summary_found: bool = False

        # MOM tracking variables
        pending_mom_active: bool = False
        pending_mom_method_type: str | None = None
        pending_mom_overlap_current: float | None = None
        pending_mom_overlap_target: float | None = None

        # Process the current line that matched the SCF start pattern
        line = current_line

        # Main parsing loop
        try:
            # First, find the SCF iteration table header
            while True:
                if SCF_ITER_TABLE_HEADER_PAT.search(line):
                    # Found the header, next lines will be iteration data
                    break
                line = next(iterator)

            # Process SCF iterations and other data
            while True:
                try:
                    line = next(iterator)
                except StopIteration:
                    break

                # Check for MOM context lines
                if MOM_ACTIVE_PAT.search(line):
                    pending_mom_active = True
                    logger.debug("MOM active signal received.")
                    continue

                if IMOM_METHOD_PAT.search(line):
                    pending_mom_method_type = "IMOM"
                    logger.debug("MOM method type 'IMOM' received.")
                    continue

                mom_overlap_match = MOM_OVERLAP_PAT.search(line)
                if mom_overlap_match:
                    try:
                        pending_mom_overlap_current = float(mom_overlap_match.group(1))
                        pending_mom_overlap_target = float(mom_overlap_match.group(2))
                        logger.debug(
                            f"MOM overlap {pending_mom_overlap_current}/{pending_mom_overlap_target} received."
                        )
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse MOM overlap from line: {line.strip()}")
                        pending_mom_overlap_current = None
                        pending_mom_overlap_target = None
                    continue

                # Process SCF iteration line
                iter_match = SCF_ITER_PAT.search(line)
                if iter_match:
                    # Create iteration with MOM data if available
                    iteration = self._parse_scf_iteration(
                        line,
                        iter_match,
                        mom_active=pending_mom_active if pending_mom_active else None,
                        mom_method_type=pending_mom_method_type,
                        mom_overlap_current=pending_mom_overlap_current,
                        mom_overlap_target=pending_mom_overlap_target,
                    )
                    all_iterations.append(iteration)

                    # Reset MOM tracking for next iteration
                    pending_mom_active = False
                    pending_mom_method_type = None
                    pending_mom_overlap_current = None
                    pending_mom_overlap_target = None

                    # Check for convergence on the same line
                    if SCF_CONVERGENCE_PAT.search(line):
                        converged = True

                    continue

                # Check for end of iteration table
                if SCF_ITER_TABLE_END_PAT.search(line):
                    # End of current iteration table, look for more data
                    continue

                # Check for SMD summary block start
                if SMD_SUMMARY_START_PAT.search(line):
                    in_smd_summary_block = True
                    smd_summary_found = True
                    continue

                # Process patterns based on current context
                self._process_patterns(line, results, in_smd_summary_block)

                # Check for SCF energy line
                scf_energy_match = SCF_FINAL_ENERGY_PAT.search(line)
                if scf_energy_match:
                    scf_energy = float(scf_energy_match.group(1))

                # Check for end of SMD summary block (heuristic)
                if in_smd_summary_block and (
                    SCF_FINAL_ENERGY_PAT.search(line)
                    or TOTAL_ENERGY_PAT.search(line)
                    or SCF_BLOCK_END_HEURISTIC_PAT.search(line)
                ):
                    in_smd_summary_block = False

                # Check for orbital energies section which follows SCF
                if ORBITAL_HEADER_PAT.search(line):
                    # We found the orbital header - this is the end of our section
                    # Buffer this line so the orbital parser can match it
                    results.buffered_line = line
                    break

                # Check for TDDFT excitation blocks which may follow SCF
                if TDA_HEADER_PAT.search(line) or TDDFT_HEADER_PAT.search(line):
                    # We found a TDDFT excitation header - this is the end of our section
                    # Buffer this line so the TDDFT parser can match it
                    results.buffered_line = line
                    break

                # Check for other end markers
                if SCF_BLOCK_END_HEURISTIC_PAT.search(line) or NORMAL_TERM_PAT.search(line):
                    break

        except StopIteration:
            logger.warning("File ended during SCF block parsing.")

        # Create SCF results object
        if not all_iterations:
            logger.warning("No SCF iterations found.")
            results.parsed_scf = True
            return

        # Determine final energy
        if scf_energy is None:
            # Use the energy from the last iteration if no explicit SCF energy line was found
            scf_energy = all_iterations[-1].energy
            logger.warning(
                "Using energy from last SCF iteration as final SCF energy; explicit 'SCF energy =' line not found or parsed."
            )
            logger.debug(f"Using last iteration energy as SCF energy: {scf_energy}")

        # Create and store SCF results
        results.scf = ScfResults(
            converged=converged,
            energy=scf_energy,
            n_iterations=len(all_iterations),
            iterations=all_iterations,
        )

        # Validate G_ENP vs SCF energy if both found
        if smd_summary_found and results.smd_g_enp_au is not None and scf_energy is not None:
            if abs(results.smd_g_enp_au - scf_energy) > 1e-6:
                logger.warning(
                    f"Mismatch between G_ENP from SMD summary ({results.smd_g_enp_au:.8f}) "
                    f"and explicit SCF energy ({scf_energy:.8f}). Using explicit SCF energy."
                )
                # Keep the explicit SCF energy value for the results
        elif smd_summary_found and (results.smd_g_enp_au is None or results.smd_g_tot_au is None):
            logger.warning("SMD summary block identified, but some energy components (G_ENP, G_TOT) were not parsed.")

        # Create SMD results if any SMD data was found
        if (
            results.smd_g_pcm_kcal_mol is not None
            or results.smd_g_cds_kcal_mol is not None
            or results.smd_g_enp_au is not None
            or results.smd_g_tot_au is not None
        ):
            results.smd = SmdResults(
                g_pcm_kcal_mol=results.smd_g_pcm_kcal_mol,
                g_cds_kcal_mol=results.smd_g_cds_kcal_mol,
                g_enp_au=results.smd_g_enp_au,
                g_tot_au=results.smd_g_tot_au,
            )

            # For SMD calculations, if we have G_TOT but no final_energy, use G_TOT as the final energy
            if results.smd_g_tot_au is not None and results.final_energy is None:
                results.final_energy = results.smd_g_tot_au
                logger.debug(f"Using SMD G(tot) as final energy: {results.final_energy:.8f}")

        results.parsed_scf = True
        logger.info(
            f"Parsed SCF data. Converged: {converged}, Energy: {scf_energy:.8f}, Iterations: {len(all_iterations)}."
        )

    def _parse_scf_iteration(
        self,
        line: str,
        match: re.Match,
        mom_active: bool = False,
        mom_method_type: str | None = None,
        mom_overlap_current: float | None = None,
        mom_overlap_target: float | None = None,
    ) -> ScfIteration:
        """Parse a single SCF iteration line and return an ScfIteration object."""
        iteration = int(match.group(1))
        energy = float(match.group(2))
        diis_error = float(match.group(3))

        # Check for Roothaan step
        step_type = "Roothaan" if ROOTHAAN_STEP_PAT.search(line) else None

        return ScfIteration(
            iteration=iteration,
            energy=energy,
            diis_error=diis_error,
            step_type=step_type,
            mom_active=mom_active,
            mom_method_type=mom_method_type,
            mom_overlap_current=mom_overlap_current,
            mom_overlap_target=mom_overlap_target,
        )

    def _process_patterns(self, line: str, results: _MutableCalculationData, in_smd_block: bool = False) -> None:
        """
        Process all patterns that might match the current line.

        Args:
            line: The current line to process
            results: The mutable calculation data to update
            in_smd_block: Whether we're currently in an SMD summary block
        """
        # Get QChem version from results
        qchem_version = getattr(results, "qchem_version", "")

        for pattern_def in SCF_PATTERNS:
            # Skip patterns that don't match the current context
            if in_smd_block and pattern_def.block_type != "smd_summary":
                continue
            if not in_smd_block and pattern_def.block_type == "smd_summary":
                continue

            # Get the appropriate pattern for this QChem version
            versioned_pattern = pattern_def.get_matching_pattern(qchem_version)
            if not versioned_pattern:
                continue

            match = versioned_pattern.pattern.search(line)
            if not match:
                continue

            # Process the match
            if pattern_def.field_name:
                # Special handling for MOM overlap which has two values
                if pattern_def.field_name == "mom_overlap":
                    current, target = versioned_pattern.transform(match)
                    # These go into the next SCF iteration
                    results.mom_overlap_current = current
                    results.mom_overlap_target = target
                else:
                    # Standard field update
                    value = versioned_pattern.transform(match)
                    setattr(results, pattern_def.field_name, value)

                    logger.debug(f"Found {pattern_def.description}: {value}")

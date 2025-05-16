from dataclasses import dataclass, field

from calcflow.parsers.qchem.typing.core import (
    CalculationData,  # Assuming .core is where CalculationData is defined within typing
)


@dataclass(frozen=True)
class MomCalculationResult:
    """
    Holds the results of a two-step MOM (Maximum Overlap Method) calculation.
    """

    initial_scf_job: CalculationData = field(default_factory=CalculationData)
    mom_scf_job: CalculationData = field(default_factory=CalculationData)
    raw_output: str = field(default_factory=str)

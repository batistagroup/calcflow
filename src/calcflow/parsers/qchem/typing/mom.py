from dataclasses import dataclass, field

from calcflow.parsers.qchem.typing.core import CalculationData


@dataclass(frozen=True)
class MomCalculationResult:
    """
    Holds the results of a two-step MOM (Maximum Overlap Method) calculation.
    """

    raw_output: str = field(default_factory=str)
    job1: CalculationData = field(default_factory=CalculationData)
    job2: CalculationData = field(default_factory=CalculationData)

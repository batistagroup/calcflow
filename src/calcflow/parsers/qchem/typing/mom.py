from dataclasses import dataclass

from calcflow.parsers.qchem.typing.core import CalculationData


@dataclass(frozen=True)
class MomCalculationResult:
    """
    Holds the results of a two-step MOM (Maximum Overlap Method) calculation.
    """

    job1: CalculationData
    job2: CalculationData
    raw_output: str

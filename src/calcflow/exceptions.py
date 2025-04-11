class CalcflowError(Exception):
    """Base class for exceptions in the calcflow package."""

    pass


class NotSupportedError(CalcflowError):
    """Exception raised for features that are not yet implemented."""

    pass


class ValidationError(CalcflowError):
    """Exception raised for errors during input validation."""

    pass


class InputGenerationError(CalcflowError):
    """Exception raised for errors during input file generation."""

    pass


class ConfigurationError(CalcflowError):
    """Exception raised for configuration-related errors."""

    pass

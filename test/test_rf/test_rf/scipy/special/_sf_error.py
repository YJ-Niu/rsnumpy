"""Warnings and Exceptions that can be raised by special functions."""
import warnings


class SpecialFunctionWarning(Warning):
    """Warning that can be emitted by special functions."""


warnings.simplefilter("always", category=SpecialFunctionWarning)


class SpecialFunctionError(Exception):
    """Exception that can be raised by special functions."""

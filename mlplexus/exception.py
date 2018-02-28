"""
Define all Exceptions for mlPlexus.

Author: Ankit N. Khambhati
Created: 2018/02/27
Updated: 2018/02/27
"""

__all__ = ['mlPlexusException', 'mlPlexusTypeError']


class mlPlexusException(Exception):
    """Base class for exceptions in mlPlexus."""


class mlPlexusNotImplemented(mlPlexusException):
    """Exception for class or function not yet implemented."""


class mlPlexusTypeError(mlPlexusException):
    """Exception for object of wrong type."""

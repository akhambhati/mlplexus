"""
Define helper functions for mlPlexus.

Author: Ankit N. Khambhati
Created: 2018/04/09
"""


def class_as_string(obj):
    """Return the full class of an object as string"""
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__
    return module + '.' + obj.__class__.__name__

def vprint(obj, verbose=True, **kwargs):
    if verbose:
        print(obj, **kwargs)

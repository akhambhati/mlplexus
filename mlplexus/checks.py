"""
Define all Checks for mlPlexus.

Author: Ankit N. Khambhati
Created: 2018/02/28
Updated: 2018/02/28
"""

import os

import numpy as np


def checkArrSqr(arr):
    """Check if arr has square shape"""
    if arr.shape[0] == arr.shape[1]:
        return True
    else:
        return False


def checkArrDims(arr, n_dim):
    """Check if arr has n_dim dimensions"""
    if arr.ndim == n_dim:
        return True
    else:
        return False


def checkArrLen(arr, length):
    """Check if arr has length of length"""
    if len(arr) == length:
        return True
    else:
        return False


def checkArrDTypeStr(arr):
    """Check if arr has dtype of type string"""
    if (arr.dtype.type == np.str_) or (arr.dtype.type == np.unicode_):
        return True
    else:
        return False


def checkNone(obj):
    """Check if obj is None"""
    if obj is None:
        return True
    else:
        return False


def checkType(obj, ref_type):
    """Check if obj is of ref_type"""
    if type(obj) is ref_type:
        return True
    else:
        return False


def checkPath(path):
    """Check if path exists"""
    return os.path.exists(path)

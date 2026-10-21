"""Type aliases and annotations for rsnumpy.

This module provides type hints used across rsnumpy and by downstream
projects such as rsSciPy.
"""

from __future__ import annotations

from typing import Any, Sequence, Union

import rsnumpy as np

# ArrayLike: anything that can be converted to an rsnumpy array
ArrayLike = Union[
    np.ndarray,
    Sequence[Any],
    int,
    float,
    complex,
    bool,
]

__all__ = ['ArrayLike']

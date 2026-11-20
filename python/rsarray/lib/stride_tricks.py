# SPDX-License-Identifier: MIT
"""Stride-based view helpers (a compatible subset of `lib.stride_tricks`).

`rsarray` arrays always own one contiguous row-major buffer, so a true
zero-copy *view* with arbitrary strides is not representable here the way it is
in .  `as_strided` therefore gathers into a fresh array.

The important consequence is about **negative strides that reach before the
start of the array you pass**.  reads that memory from the parent buffer,
because `parent[k:]` is a view sharing storage.  In rsarray basic slicing copies
and `.base` is `None`, so the earlier elements simply are not reachable; the
caller has the start of the new copy and nothing behind it.  Such accesses are
rejected with a `ValueError` naming the offending offset, rather than silently
wrapping around Python-style into the tail of the buffer.

Reads that stay inside the array -- including backwards traversal via a
negative stride, as long as the offsets do not go below zero -- behave exactly
as the reference implementation's.
"""

import rsarray as np

__all__ = ['as_strided']


def as_strided(x, shape=None, strides=None, subok=False, writeable=True,
               check_bounds=True):
    """Create a strided array from an existing one.

    Parameters
    ----------
    x : array_like
        Array whose underlying buffer supplies the values.
    shape : sequence of int, optional
        Shape of the result.  Defaults to `x.shape`.
    strides : sequence of int, optional
        Strides of the result **in bytes**, -style.  Defaults to the
        C-contiguous strides of `shape`.
    subok : bool, optional
        Accepted for compatibility; rsarray has no subclass hierarchy.
    writeable : bool, optional
        Accepted for compatibility.  The result owns its memory, so it is
        always writeable and writes never reach `x`.
    check_bounds : bool, optional
        Accepted for compatibility.  Bounds are always checked
        (out-of-range strides raise ValueError below).

    Returns
    -------
    out : rsarray.ndarray
        A new array with the requested shape, gathered from `x`.

    Raises
    ------
    ValueError
        If the strides reach outside `x`, which for negative strides means they
        reach before its start.  See the module docstring.
    """
    _ = check_bounds
    shape = tuple(int(s) for s in (x.shape if shape is None else shape))
    strides = tuple(int(s) for s in (x.strides if strides is None else strides))
    if len(shape) != len(strides):
        raise ValueError(
            f'strides and shape must have the same length, got '
            f'{len(strides)} and {len(shape)}')

    itemsize = int(x.itemsize)
    # Element-count steps; a stride that is not a whole number of elements
    # cannot be walked element-wise at all.  Strides on length-1 axes are
    # never walked (offset contribution is always 0), so 's "relaxed
    # strides" allow arbitrary values there — skip the check for them.
    for value, dim in zip(strides, shape):
        if dim > 1 and value % itemsize != 0:
            raise ValueError(
                f'stride {value} is not a multiple of the itemsize '
                f'{itemsize} of dtype {x.dtype}')
    steps = [value // itemsize for value in strides]

    flat = np.asarray(x).ravel().tolist()
    size = len(flat)
    total = 1
    for axis in shape:
        total *= axis

    values = []
    for index in range(total):
        offset = 0
        remainder = index
        for axis in reversed(range(len(shape))):
            remainder, along = divmod(remainder, shape[axis])
            offset += along * steps[axis]
        if offset < 0:
            raise ValueError(
                f'strided access reaches element {offset}, before the start of '
                f'the array. would serve that from the parent buffer, '
                f'but rsarray slicing copies, so there is nothing there to '
                f'read; build those values explicitly or pass an array whose '
                f'own storage begins further back.')
        if offset >= size:
            raise ValueError(
                f'strided access reaches element {offset}, past the end of the '
                f'array holding {size} elements.')
        values.append(flat[offset])

    # 保留输入 dtype（asarray 从 Python 值推断会得 float64，须显式回投到 x.dtype）
    return np.asarray(values).reshape(shape).astype(x.dtype)

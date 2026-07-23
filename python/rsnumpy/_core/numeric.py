import operator


def normalize_axis_index(axis, ndim, argname=None):
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        if argname:
            raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
        else:
            raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis


def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    if not isinstance(axis, (tuple, list)):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    axis = tuple(normalize_axis_index(ax, ndim, argname) for ax in axis)
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError(f'repeated axis in `{argname}` argument')
        else:
            raise ValueError('repeated axis')
    return axis

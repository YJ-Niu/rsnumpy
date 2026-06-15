"""数组操作模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnum._core as _core


def _nd():
    from .__init__ import ndarray as _n
    return _n


def _wrap(result):
    return _nd()(result)


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
    return _core.ndarray(a)


def reshape(a, newshape, order='C'):
    """改变数组形状而不改变数据。"""
    if isinstance(newshape, int):
        newshape = (newshape,)
    return _wrap(_core.reshape(_ensure_raw(a), newshape))


def ravel(a, order='C'):
    """展平数组。"""
    return _wrap(_core.flatten(_ensure_raw(a)))


def moveaxis(a, source, destination):
    """移动数组的轴（简化实现）。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    return arr


def rollaxis(a, axis, start=0):
    """滚动数组的轴到指定位置（简化实现）。"""
    return moveaxis(a, axis, start)


def broadcast_to(a, shape):
    """将数组广播到新形状。"""
    return _wrap(_core.broadcast_to(_ensure_raw(a), shape))


def transpose(a, axes=None):
    """转置数组。"""
    if axes is None:
        return _wrap(_core.transpose(_ensure_raw(a)))
    return _wrap(_core.transpose(_ensure_raw(a)))


def swapaxes(a, axis1, axis2):
    """交换数组的两个轴。"""
    return transpose(a)


def expand_dims(a, axis):
    """扩展数组的形状。"""
    return _wrap(_core.expand_dims(_ensure_raw(a), axis))


def squeeze(a, axis=None):
    """从数组形状中删除单维度条目。"""
    return _wrap(_core.squeeze(_ensure_raw(a)))


def concatenate(arrays, axis=0):
    """沿着指定轴连接数组。"""
    arr_list = [_ensure_raw(a) for a in arrays]
    return _wrap(_core.concatenate(arr_list, axis))


def stack(arrays, axis=0):
    """沿着新轴堆叠数组。"""
    arr_list = [_ensure_raw(a) for a in arrays]
    return _wrap(_core.stack(arr_list, axis))


def vstack(arrays):
    """垂直堆叠数组（沿行）。"""
    arr_list = [_ensure_raw(a) for a in arrays]
    return _wrap(_core.vstack(arr_list))


def hstack(arrays):
    """水平堆叠数组（沿列）。"""
    arr_list = [_ensure_raw(a) for a in arrays]
    return _wrap(_core.hstack(arr_list))


def dstack(arrays):
    """沿深度方向堆叠数组。"""
    arr_list = [_ensure_raw(a) for a in arrays]
    result_raw = _core.concatenate(arr_list, 2)
    return _wrap(result_raw)


def column_stack(arrays):
    """将 1-D 数组作为列堆叠成 2-D 数组。"""
    arr_list = [_ensure_raw(a) for a in arrays]
    return _wrap(_core.column_stack(arr_list))


def split(ary, indices_or_sections, axis=0):
    """将数组拆分为多个子数组（使用 Rust 实现）。"""
    raw = _ensure_raw(ary)
    sections = indices_or_sections
    result_raw = _core.split_rs(raw, sections, axis)
    return [_wrap(r) for r in result_raw]


def hsplit(ary, indices_or_sections):
    """水平拆分数组。"""
    return split(ary, indices_or_sections, axis=1)


def vsplit(ary, indices_or_sections):
    """垂直拆分数组。"""
    return split(ary, indices_or_sections, axis=0)


def dsplit(ary, indices_or_sections):
    """沿深度拆分数组。"""
    return split(ary, indices_or_sections, axis=2)


def tile(A, reps):
    """重复数组。"""
    if isinstance(reps, int):
        reps = [reps]
    return _wrap(_core.tile(_ensure_raw(A), reps))


def repeat(a, repeats, axis=None):
    """重复数组元素（使用 Python 层实现）。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    return arr.repeat(repeats)


def flip(m, axis=None):
    """沿指定轴翻转数组。"""
    arr = m if hasattr(m, '_array') else _wrap(m)
    return _wrap(_core.flip(_ensure_raw(arr), axis))


def fliplr(m):
    """左右翻转数组。"""
    return flip(m, axis=1)


def flipud(m):
    """上下翻转数组。"""
    return flip(m, axis=0)


def roll(a, shift, axis=None):
    """沿指定轴滚动数组。"""
    return _wrap(_core.roll(_ensure_raw(a), shift, axis))


def rot90(m, k=1, axes=(0, 1)):
    """将数组旋转 90 度。"""
    return _wrap(_core.rot90(_ensure_raw(m), k))

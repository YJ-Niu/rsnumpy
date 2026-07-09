"""数组操作模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnumpy._core as _core


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
    arr = a if hasattr(a, '_array') else _wrap(a)
    if order == 'F':
        flat = arr.ravel(order='F')
        return _wrap(_core.reshape(flat._array, newshape))
    return _wrap(_core.reshape(_ensure_raw(a), newshape))


def ravel(a, order='C'):
    """展平数组。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    return arr.ravel(order=order)


def moveaxis(a, source, destination):
    """移动数组的轴（简化实现）。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    return arr


def rollaxis(a, axis, start=0):
    """滚动数组的轴到指定位置（简化实现）。"""
    return moveaxis(a, axis, start)


def broadcast_to(a, shape):
    """将数组广播到新形状。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    result = _core.broadcast_to(arr._array, shape)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def transpose(a, axes=None):
    """转置数组。axes 为 None 时反转所有轴，否则按给定顺序置换。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    ndim = arr.ndim
    if axes is not None:
        axes = [ax % ndim for ax in axes]
        if axes != list(reversed(range(ndim))) and axes != list(range(ndim)):
            # 通过交换轴序列达到任意置换（result 轴 i 来自原轴 axes[i]）
            result = arr
            cur = list(range(ndim))
            for i in range(ndim):
                j = cur.index(axes[i])
                if j != i:
                    result = swapaxes(result, i, j)
                    cur[i], cur[j] = cur[j], cur[i]
            return result
        if axes == list(range(ndim)):
            return nd._wrap(arr._array, _dtype=dtype, _fields=fields, _raw_data=raw_data)
    result = _core.transpose(arr._array)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def swapaxes(a, axis1, axis2):
    """交换数组的两个轴。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    result = _core.swapaxes(arr._array, axis1, axis2)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def expand_dims(a, axis):
    """扩展数组的形状。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    result = _core.expand_dims(arr._array, axis)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def squeeze(a, axis=None):
    """从数组形状中删除单维度条目。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    result = _core.squeeze(arr._array)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def concatenate(arrays, axis=0):
    """沿着指定轴连接数组。"""
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("concatenate requires at least one array")
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    nd = _nd()
    dtype = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [a._array if hasattr(a, '_array') else _core.ndarray(a) for a in arr_list]
    result = _core.concatenate(raw_list, axis)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def stack(arrays, axis=0):
    """沿着新轴堆叠数组。"""
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("stack requires at least one array")
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    nd = _nd()
    dtype = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [a._array if hasattr(a, '_array') else _core.ndarray(a) for a in arr_list]
    result = _core.stack(raw_list, axis)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def vstack(arrays):
    """垂直堆叠数组（沿行）。"""
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("vstack requires at least one array")
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    nd = _nd()
    dtype = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [a._array if hasattr(a, '_array') else _core.ndarray(a) for a in arr_list]
    result = _core.vstack(raw_list)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def hstack(arrays):
    """水平堆叠数组（沿列）。"""
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("hstack requires at least one array")
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    nd = _nd()
    dtype = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [a._array if hasattr(a, '_array') else _core.ndarray(a) for a in arr_list]
    result = _core.hstack(raw_list)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def dstack(arrays):
    """沿深度方向堆叠数组。"""
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("dstack requires at least one array")
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    nd = _nd()
    dtype = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [a._array if hasattr(a, '_array') else _core.ndarray(a) for a in arr_list]
    result = _core.concatenate(raw_list, 2)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def column_stack(arrays):
    """将 1-D 数组作为列堆叠成 2-D 数组。"""
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("column_stack requires at least one array")
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    nd = _nd()
    dtype = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [a._array if hasattr(a, '_array') else _core.ndarray(a) for a in arr_list]
    result = _core.column_stack(raw_list)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def split(ary, indices_or_sections, axis=0):
    """将数组拆分为多个子数组（使用 Rust 实现）。"""
    arr = ary if hasattr(ary, '_array') else _wrap(ary)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    result_raw = _core.split_rs(arr._array, indices_or_sections, axis)
    return [nd._wrap(r, _dtype=dtype, _fields=fields, _raw_data=raw_data) for r in result_raw]


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
    """重复数组元素。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    return arr.repeat(repeats, axis)


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


def resize(a, new_shape):
    """返回指定形状的新数组，数据重复填充至新形状。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    
    result = _core.resize_rs(arr._array, list(new_shape))
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def append(arr, values, axis=None):
    """将值附加到数组末尾。"""
    arr = arr if hasattr(arr, '_array') else _wrap(arr)
    values_nd = values if hasattr(values, '_array') else _wrap(values)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    if axis is None:
        a_flat = arr.ravel()
        v_flat = values_nd.ravel()
        return concatenate((a_flat, v_flat))
    raw_list = [arr._array, values_nd._array]
    result = _core.concatenate(raw_list, axis)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def delete(arr, obj, axis=None):
    """删除指定位置的元素。"""
    arr = arr if hasattr(arr, '_array') else _wrap(arr)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    if isinstance(obj, slice):
        start, stop, step = obj.indices(arr.size if axis is None else arr.shape[axis])
        obj = list(range(start, stop, step))
    elif isinstance(obj, int):
        obj = [obj]
    obj = sorted(set(obj))
    
    result = _core.delete_rs(arr._array, obj, axis)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def insert(arr, obj, values, axis=None):
    """沿指定轴在指定下标前插入值。"""
    arr = arr if hasattr(arr, '_array') else _wrap(arr)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    if isinstance(obj, int):
        obj = [obj]
    
    if not hasattr(values, '_array'):
        if isinstance(values, (list, tuple)):
            values_nd = _nd()(values)
        else:
            values_nd = _nd()([values])
    else:
        values_nd = values
    
    values_flat = values_nd.ravel().tolist()
    result = _core.insert_rs(arr._array, obj, values_flat, axis)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def unique(a, return_index=False, return_inverse=False, return_counts=False):
    """查找数组内的唯一元素。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    results = _core.unique_full(arr._array, return_index, return_inverse, return_counts)
    wrapped = [nd._wrap(r, _dtype=dtype, _fields=fields, _raw_data=raw_data) for r in results]
    
    if not (return_index or return_inverse or return_counts):
        return wrapped[0]
    return tuple(wrapped)

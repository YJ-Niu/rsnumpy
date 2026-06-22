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
    """转置数组。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
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
    new_size = 1
    for s in new_shape:
        new_size *= s

    flat = arr.ravel()
    flat_data = flat._array.flatten().tolist()
    if not flat_data:
        result = _core.ndarray([]).reshape(list(new_shape))
        return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)

    # Cycle or truncate data to fit new_size
    cycled = []
    for i in range(new_size):
        cycled.append(flat_data[i % len(flat_data)])

    result = _core.ndarray(cycled).reshape(list(new_shape))
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

    if axis is None:
        flat = arr.ravel()
        raw = flat._array
        data = raw.flatten().tolist()
        keep = [v for i, v in enumerate(data) if i not in obj]
        result_data = _core.ndarray(keep)
        return nd._wrap(result_data, _dtype=dtype, _fields=fields, _raw_data=raw_data)

    parts = []
    start = 0
    shape = list(arr.shape)
    axis_size = shape[axis]
    for idx in obj:
        if idx > start:
            sl = tuple(
                slice(start, idx) if i == axis else slice(None)
                for i in range(len(shape))
            )
            parts.append(arr[sl])
        start = idx + 1 if idx + 1 <= axis_size else axis_size
    if start < axis_size:
        sl = tuple(
            slice(start, axis_size) if i == axis else slice(None)
            for i in range(len(shape))
        )
        parts.append(arr[sl])

    if not parts:
        empty_shape = list(arr.shape)
        empty_shape[axis] = 0
        result_data = _core.ndarray([]).reshape(empty_shape)
        return nd._wrap(result_data, _dtype=dtype, _fields=fields, _raw_data=raw_data)
    if len(parts) == 1:
        return parts[0]
    return concatenate(parts, axis=axis)


def insert(arr, obj, values, axis=None):
    """沿指定轴在指定下标前插入值。"""
    arr = arr if hasattr(arr, '_array') else _wrap(arr)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    if isinstance(obj, int):
        obj = [obj]

    flat_data = arr.ravel()._array.flatten().tolist()

    if axis is None:
        vals = []
        if hasattr(values, '_array') or isinstance(values, (list, tuple)):
            vals_arr = values if hasattr(values, '_array') else _nd()(values)
            vals = vals_arr.ravel()._array.flatten().tolist()
        else:
            vals = [float(values)]
        for i, pos in enumerate(sorted(obj)):
            pos = min(pos, len(flat_data))
            for v in reversed(vals):
                flat_data.insert(pos, v)
        result_data = _core.array(flat_data)
        return nd._wrap(result_data, _dtype=dtype, _fields=fields, _raw_data=raw_data)

    # Axis-based: operate on flat data grouped by pre/post
    shape = list(arr.shape)
    axis_size = shape[axis]
    pre = 1
    for i in range(axis):
        pre *= shape[i]
    post = 1
    for i in range(axis + 1, len(shape)):
        post *= shape[i]

    # Prepare insertion values as flat list (size = pre * post)
    vals = []
    if hasattr(values, '_array') or isinstance(values, (list, tuple)):
        vals_arr = values if hasattr(values, '_array') else _nd()(values)
        flat_v = vals_arr.ravel()._array.flatten().tolist()
        needed = pre * post
        for i in range(needed):
            vals.append(flat_v[i % len(flat_v)])
    else:
        needed = pre * post
        vals = [float(values)] * needed

    # Build result flat data by inserting values at each obj position
    result_flat = []
    new_axis_size = axis_size + len(obj)
    insert_positions = set(sorted(obj))

    for p in range(pre):
        src_axis_idx = 0
        for dest_axis_idx in range(new_axis_size):
            if dest_axis_idx in insert_positions:
                # Insert values for this position
                for k in range(post):
                    result_flat.append(vals[p * post + k])
            else:
                # Copy from source
                for k in range(post):
                    src_flat_idx = p * axis_size * post + src_axis_idx * post + k
                    result_flat.append(flat_data[src_flat_idx])
                src_axis_idx += 1

    new_shape = list(shape)
    new_shape[axis] = new_axis_size
    result_data = _core.array(result_flat).reshape(new_shape)
    return nd._wrap(result_data, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def unique(a, return_index=False, return_inverse=False, return_counts=False):
    """查找数组内的唯一元素。

    参数:
        a: 输入数组，如果不是一维数组则会展开。
        return_index: 如果为 True，返回唯一值在原始数组中的首次出现位置。
        return_inverse: 如果为 True，返回原始数组元素在唯一值数组中的下标。
        return_counts: 如果为 True，返回唯一值在原始数组中的出现次数。

    返回:
        唯一值数组，如果指定了多个返回选项则返回元组。
    """
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    # Flatten the array
    flat = arr.ravel()
    flat_data = flat._array.flatten().tolist()

    # Get unique values from Rust
    unique_raw = _core.unique(arr._array)
    unique_vals = unique_raw.flatten().tolist()

    result = nd._wrap(unique_raw, _dtype=dtype, _fields=fields, _raw_data=raw_data)

    if not (return_index or return_inverse or return_counts):
        return result

    # Build lookup: value → first index
    seen = {}
    first_idx = []
    for i, v in enumerate(flat_data):
        if v not in seen:
            seen[v] = i
    for v in unique_vals:
        first_idx.append(seen[v])

    # Build inverse
    val_to_unique_idx = {v: i for i, v in enumerate(unique_vals)}
    inverse = [val_to_unique_idx[v] for v in flat_data]

    # Build counts
    counts = [0] * len(unique_vals)
    for v in flat_data:
        counts[val_to_unique_idx[v]] += 1

    results = [result]
    if return_index:
        idx_arr = _core.array(first_idx)
        results.append(nd._wrap(idx_arr, _dtype=dtype, _fields=fields, _raw_data=raw_data))
    if return_inverse:
        inv_arr = _core.array(inverse)
        results.append(nd._wrap(inv_arr, _dtype=dtype, _fields=fields, _raw_data=raw_data))
    if return_counts:
        cnt_arr = _core.array(counts)
        results.append(nd._wrap(cnt_arr, _dtype=dtype, _fields=fields, _raw_data=raw_data))

    return tuple(results) if len(results) > 1 else results[0]

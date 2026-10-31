"""数组操作模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnumpy.num_core as _core


def _nd():
    from .__init__ import ndarray as _n
    return _n


def _wrap(result):
    return _nd()(result)


def _nested_to_list(x):
    """Recursively turn nested arrays / lists / tuples into pure nested lists."""
    if hasattr(x, '_array'):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_nested_to_list(i) for i in x]
    return x


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
    if isinstance(a, (list, tuple)):
        # Rust core cannot parse lists that contain array objects; promote
        # them to pure nested lists first (e.g. concatenate([[X[0]], [X[1]]])).
        return _core.ndarray(_nested_to_list(a))
    return _core.ndarray(a)


def _scalar_idx(x):
    """标量下标/形状参数：Python int/bool 或 rsnumpy 整数标量（非序列/数组）。
    rsnumpy 整型标量不再是 int 子类，需用 __index__ 协议识别。"""
    if isinstance(x, int):
        return True
    return (hasattr(x, '__index__') and not hasattr(x, '__iter__')
            and not hasattr(x, '_array'))


def reshape(a, shape=None, order='C', *, newshape=None, copy=None):
    """改变数组形状而不改变数据。

    shape 为新形状（整数或整数元组）。newshape 为已弃用的旧参数名，
    order 支持 'C'（行优先）与 'F'（列优先）。
    """
    _ = copy
    if shape is None:
        shape = newshape
    if shape is None:
        raise TypeError("reshape() missing 1 required positional argument: 'shape'")
    if isinstance(shape, int) or _scalar_idx(shape):
        shape = (int(shape),)
    else:
        shape = tuple(shape)
    arr = a if hasattr(a, '_array') else _wrap(a)
    if order == 'F':
        flat = arr.ravel(order='F')
        out = _wrap(_core.reshape(flat._array, shape))
    else:
        out = _wrap(_core.reshape(_ensure_raw(a), shape))
    try:
        out._rs_parent = arr
    except Exception:
        pass
    return out


def ravel(a, order='C'):
    """展平数组。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    return arr.ravel(order=order)


def moveaxis(a, source, destination):
    """将数组的轴从 source 移动到 destination。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    if isinstance(source, int) or _scalar_idx(source):
        source = [int(source)]
    if isinstance(destination, int) or _scalar_idx(destination):
        destination = [int(destination)]
    result = _core.moveaxis(arr._array, list(source), list(destination))
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def rollaxis(a, axis, start=0):
    """滚动数组的轴到指定位置（简化实现）。"""
    return moveaxis(a, axis, start)


def broadcast_to(a, shape, subok=None, device=None):
    """将数组广播到新形状。subok/device 仅为兼容 numpy/array-api 签名，忽略。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    # Handle broadcasting to scalar shape ()
    if shape == () or shape == []:
        result = arr._array  # 0-d array is already scalar-shaped
    else:
        _shp = ((int(shape),) if (isinstance(shape, int) or _scalar_idx(shape))
                else tuple(int(s) for s in shape))
        _tot = 1
        for _s in _shp:
            _tot *= _s
        if _tot > 2**40:  # 与 __init__._MAX_ALLOC_ELEMENTS 保持一致
            # 超大广播：零分配 Python 层视图（Rust 物化会 OOM / capacity overflow
            # abort 进程）。numpy 语义：broadcast_to 从不分配；物化时抛 MemoryError。
            res = nd._wrap(arr._array, _dtype=dtype, _fields=fields, _raw_data=raw_data)
            res._rs_shape_override = _shp
            return res
        result = _core.broadcast_to(arr._array, shape)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def transpose(a, axes=None):
    """转置数组。axes 为 None 时反转所有轴，否则按给定顺序置换。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    result = _core.transpose_axes(arr._array, axes)
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
    """从数组形状中删除单维度条目（axis 指定时只删除指定轴）。"""
    # list 输入（可能含 ndarray 元素）：先递归转纯嵌套列表再构造，
    # 使 np.squeeze([np.zeros((3, 1))]) 得到 (1, 3, 1) 而非 object 数组
    arr = a if hasattr(a, '_array') else _nd()(_nested_to_list(a))
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    if axis is None:
        result = _core.squeeze(arr._array)
        return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)
    # numpy 语义：只删除指定轴（该轴大小须为 1）
    shape = tuple(arr.shape)
    ndim = len(shape)
    if isinstance(axis, int):
        _axes = (axis,)
    else:
        _axes = tuple(axis)
    _axes = tuple(_x % ndim for _x in _axes)
    new_shape = []
    for _i, _s in enumerate(shape):
        if _i in _axes:
            if _s != 1:
                raise ValueError(
                    "cannot select an axis to squeeze out which has size "
                    "not equal to one")
        else:
            new_shape.append(_s)
    result = arr.reshape(tuple(new_shape))
    return nd._wrap(result._array, _dtype=dtype, _fields=fields,
                    _raw_data=raw_data)


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
    raw_list = [_ensure_raw(a) for a in arr_list]
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
    raw_list = [_ensure_raw(a) for a in arr_list]
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
    raw_list = [_ensure_raw(a) for a in arr_list]
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
    raw_list = [_ensure_raw(a) for a in arr_list]
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
    raw_list = [_ensure_raw(a) for a in arr_list]
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
    raw_list = [_ensure_raw(a) for a in arr_list]
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
    if isinstance(reps, int) or _scalar_idx(reps):
        reps = [int(reps)]
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
    arr = a if hasattr(a, '_array') else _wrap(a)
    # 空数组 roll 是恒等映射：Rust 侧 shift % 0 会 panic（除零）。
    if getattr(arr, 'size', 0) == 0:
        return arr.copy()
    return _wrap(_core.roll(_ensure_raw(a), shift, axis))


def rot90(m, k=1, axes=(0, 1)):
    """将数组旋转 90 度。"""
    return _wrap(_core.rot90(_ensure_raw(m), k, axes[0], axes[1]))


def resize(a, new_shape):
    """返回指定形状的新数组，数据重复填充至新形状。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    if isinstance(new_shape, int) or _scalar_idx(new_shape):
        new_shape = (int(new_shape),)
    
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
    elif isinstance(obj, int) or _scalar_idx(obj):
        obj = [int(obj)]
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

    # 检查数据中是否包含复数（即使 dtype 声明为 float64）
    def _contains_complex(data):
        """递归检查数据中是否包含复数"""
        if isinstance(data, complex):
            return True
        if isinstance(data, (list, tuple)):
            return any(_contains_complex(x) for x in data)
        return False

    # 处理复数数据：展平为 [real, imag, real, imag, ...]
    if dtype == 'complex128' or _contains_complex(values_flat):
        # 递归展平嵌套复数列表
        def _flatten_complex_nested(lst):
            result = []
            for x in lst:
                if isinstance(x, complex):
                    result.extend([x.real, x.imag])
                elif isinstance(x, (list, tuple)):
                    result.extend(_flatten_complex_nested(x))
                else:
                    result.extend([float(x), 0.0])
            return result

        # 列表推导 + 闭包，避免在循环中重复定义函数
        flat_real_imag = []
        for item in values_flat:
            if isinstance(item, complex):
                flat_real_imag.extend([item.real, item.imag])
            elif isinstance(item, (list, tuple)):
                flat_real_imag.extend(_flatten_complex_nested(item))
            else:
                flat_real_imag.extend([float(item), 0.0])
        values_flat = flat_real_imag

    result = _core.insert_rs(arr._array, obj, values_flat, axis)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def unique(a, return_index=False, return_inverse=False, return_counts=False,
           axis=None, *, equal_nan=True, sorted=True):
    """查找数组内的唯一元素。

    axis 为 None 时展平后去重；为整数时，将该轴移到最前、其余维度展平，
    把每个子数组视为一个元素按字典序去重（唯一“行”）。
    """
    _ = equal_nan
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    if axis is None:
        results = _core.unique_full(arr._array, return_index, return_inverse, return_counts)
        wrapped = [nd._wrap(r, _dtype=dtype, _fields=fields, _raw_data=raw_data) for r in results]
        if not (return_index or return_inverse or return_counts):
            return wrapped[0]
        return tuple(wrapped)

    results = _core.unique_axis(
        arr._array, axis, return_index, return_inverse, return_counts, sorted)
    unique_arr = nd._wrap(results[0], _dtype=dtype, _fields=fields, _raw_data=raw_data)
    if not (return_index or return_inverse or return_counts):
        return unique_arr
    outputs = [unique_arr] + [nd._wrap(r, _dtype="int64") for r in results[1:]]
    return tuple(outputs)

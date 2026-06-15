"""
数组操作模块

此模块包含数组变形、拼接、拆分、翻转等操作函数。
"""

import rsnum._core as _core


def _wrap_result(result):
    """包装结果为 ndarray 对象"""
    from .__init__ import ndarray
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return ndarray._wrap(result)
    return result


def _ensure(x):
    """转换 list/tuple 为 ndarray"""
    from .__init__ import ndarray
    if isinstance(x, (list, tuple)):
        return _core.ndarray(x)
    elif isinstance(x, ndarray):
        return x._array
    elif hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray':
        return x
    return x


def reshape(a, newshape, order='C'):
    """
    改变数组形状。

    参数:
        a: 输入数组。
        newshape: 新形状，可以是整数或整数元组。
        order: 索引顺序，'C'表示行优先，'F'表示列优先。

    返回:
        ndarray: 重新塑形后的数组。
    """
    return _wrap_result(_core.reshape(_ensure(a), newshape))


def ravel(a, order='C'):
    """
    将数组展平为一维。

    参数:
        a: 输入数组。
        order: 展平顺序。

    返回:
        ndarray: 展平后的数组。
    """
    return _wrap_result(_core.flatten(_ensure(a)))


def moveaxis(a, source, destination):
    """
    将轴从一个位置移动到另一个位置。

    参数:
        a: 输入数组。
        source: 要移动的轴。
        destination: 目标位置。

    返回:
        ndarray: 移动轴后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    axes = list(range(arr.ndim))
    source = [source] if isinstance(source, int) else source
    destination = [destination] if isinstance(destination, int) else destination
    
    for s, d in sorted(zip(source, destination)):
        axes.insert(d, axes.pop(s))
    
    return arr.transpose(*axes)


def rollaxis(a, axis, start=0):
    """
    滚动轴到指定位置。

    参数:
        a: 输入数组。
        axis: 要滚动的轴。
        start: 目标位置。

    返回:
        ndarray: 滚动后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    ndim = arr.ndim
    axis = axis % ndim
    start = start % ndim
    
    if axis < start:
        start -= 1
    
    axes = list(range(ndim))
    axes.insert(start, axes.pop(axis))
    
    return arr.transpose(*axes)


def broadcast_to(a, shape):
    """
    将数组广播到指定形状。

    参数:
        a: 输入数组。
        shape: 目标形状。

    返回:
        ndarray: 广播后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    result = arr._array.broadcast_to(shape)
    return _wrap_result(result)


def transpose(a, axes=None):
    """
    转置数组。

    参数:
        a: 输入数组。
        axes: 轴的顺序，默认为反转。

    返回:
        ndarray: 转置后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    if axes is None:
        return arr.T
    return arr.transpose(*axes)


def swapaxes(a, axis1, axis2):
    """
    交换两个轴。

    参数:
        a: 输入数组。
        axis1: 第一个轴。
        axis2: 第二个轴。

    返回:
        ndarray: 交换轴后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.swapaxes(axis1, axis2)


def expand_dims(a, axis):
    """
    在指定位置添加新轴。

    参数:
        a: 输入数组。
        axis: 添加轴的位置。

    返回:
        ndarray: 扩展后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    shape = list(arr.shape)
    shape.insert(axis, 1)
    return arr.reshape(shape)


def squeeze(a, axis=None):
    """
    移除长度为1的轴。

    参数:
        a: 输入数组。
        axis: 可选，指定要移除的轴。

    返回:
        ndarray: 压缩后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.squeeze()


def concatenate(arrays, axis=0):
    """
    沿着指定轴连接数组。

    参数:
        arrays: 数组序列。
        axis: 连接的轴。

    返回:
        ndarray: 连接后的数组。
    """
    from .__init__ import ndarray
    arr_list = [ndarray(a) for a in arrays]
    
    if len(arr_list) == 0:
        raise ValueError("至少需要一个数组")
    
    shape = list(arr_list[0].shape)
    for arr in arr_list[1:]:
        arr_shape = list(arr.shape)
        for i in range(len(shape)):
            if i != axis and arr_shape[i] != shape[i]:
                raise ValueError("除连接轴外，其他维度必须相同")
    
    all_data = []
    for arr in arr_list:
        all_data.extend(arr.tolist())
    
    return ndarray(all_data).reshape(tuple(shape[:axis] + [sum(a.shape[axis] for a in arr_list)] + shape[axis+1:]))


def stack(arrays, axis=0):
    """
    沿着新轴堆叠数组。

    参数:
        arrays: 数组序列。
        axis: 堆叠的轴。

    返回:
        ndarray: 堆叠后的数组。
    """
    from .__init__ import ndarray
    arr_list = [ndarray(a) for a in arrays]
    
    if len(arr_list) == 0:
        raise ValueError("至少需要一个数组")
    
    shape = list(arr_list[0].shape)
    for arr in arr_list[1:]:
        if list(arr.shape) != shape:
            raise ValueError("所有数组必须有相同的形状")
    
    new_shape = shape[:axis] + [len(arr_list)] + shape[axis:]
    total_size = 1
    for s in new_shape:
        total_size *= s
    
    flat_result = []
    for arr in arr_list:
        flat_result.extend(arr.ravel().tolist())
    
    result = ndarray(flat_result)
    return result.reshape(new_shape)


def vstack(tup):
    """
    垂直堆叠数组（行方向）。

    参数:
        tup: 数组元组。

    返回:
        ndarray: 垂直堆叠后的数组。
    """
    return concatenate(tup, axis=0)


def hstack(tup):
    """
    水平堆叠数组（列方向）。

    参数:
        tup: 数组元组。

    返回:
        ndarray: 水平堆叠后的数组。
    """
    return concatenate(tup, axis=1)


def dstack(tup):
    """
    深度堆叠数组（第三轴方向）。

    参数:
        tup: 数组元组。

    返回:
        ndarray: 深度堆叠后的数组。
    """
    return stack(tup, axis=2)


def column_stack(tup):
    """
    将一维数组作为列堆叠成二维数组。

    参数:
        tup: 数组元组。

    返回:
        ndarray: 堆叠后的数组。
    """
    from .__init__ import ndarray
    arrays = [ndarray(a).ravel() for a in tup]
    return hstack([arr.reshape(-1, 1) for arr in arrays])


def split(ary, indices_or_sections, axis=0):
    """
    将数组拆分为多个子数组。

    参数:
        ary: 输入数组。
        indices_or_sections: 拆分位置或数量。
        axis: 拆分的轴。

    返回:
        list: 子数组列表。
    """
    from .__init__ import ndarray
    arr = ndarray(ary)
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        size = arr.shape[axis] // sections
        indices = [i * size for i in range(1, sections)]
    else:
        indices = indices_or_sections
    
    result = []
    prev = 0
    for idx in indices:
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(prev, idx)
        result.append(arr[tuple(slc)])
        prev = idx
    
    slc = [slice(None)] * arr.ndim
    slc[axis] = slice(prev, None)
    result.append(arr[tuple(slc)])
    
    return result


def hsplit(ary, indices_or_sections):
    """
    水平拆分数组。

    参数:
        ary: 输入数组。
        indices_or_sections: 拆分位置或数量。

    返回:
        list: 子数组列表。
    """
    return split(ary, indices_or_sections, axis=1)


def vsplit(ary, indices_or_sections):
    """
    垂直拆分数组。

    参数:
        ary: 输入数组。
        indices_or_sections: 拆分位置或数量。

    返回:
        list: 子数组列表。
    """
    return split(ary, indices_or_sections, axis=0)


def dsplit(ary, indices_or_sections):
    """
    深度拆分数组。

    参数:
        ary: 输入数组。
        indices_or_sections: 拆分位置或数量。

    返回:
        list: 子数组列表。
    """
    return split(ary, indices_or_sections, axis=2)


def tile(A, reps):
    """
    重复数组。

    参数:
        A: 输入数组。
        reps: 重复次数。

    返回:
        ndarray: 重复后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(A)
    arr_data = arr.tolist()
    
    if not isinstance(reps, tuple):
        reps = (reps,)
    
    def tile_recursive(data, reps, dim=0):
        if dim >= len(reps):
            return data
        if dim >= len(arr.shape):
            return data * reps[dim]
        return [tile_recursive(row, reps, dim + 1) for _ in range(reps[dim]) for row in data]
    
    result_data = tile_recursive(arr_data, reps)
    return ndarray(result_data)


def repeat(a, repeats, axis=None):
    """
    重复数组元素。

    参数:
        a: 输入数组。
        repeats: 重复次数。
        axis: 重复的轴。

    返回:
        ndarray: 重复后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.repeat(repeats, axis)


def flip(m, axis=None):
    """
    沿指定轴翻转数组。

    参数:
        m: 输入数组。
        axis: 翻转的轴。

    返回:
        ndarray: 翻转后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(m)
    data = arr.tolist()
    
    def flip_recursive(data, axis, current_dim=0):
        if not isinstance(data, list):
            return data
        if axis is None:
            if not isinstance(data[0], list):
                return data[::-1]
            return [flip_recursive(row, axis, current_dim + 1) for row in data][::-1]
        if current_dim == axis:
            return data[::-1]
        return [flip_recursive(row, axis, current_dim + 1) for row in data]
    
    result_data = flip_recursive(data, axis)
    return ndarray(result_data)


def fliplr(m):
    """
    左右翻转数组。

    参数:
        m: 输入数组。

    返回:
        ndarray: 左右翻转后的数组。
    """
    return flip(m, axis=1)


def flipud(m):
    """
    上下翻转数组。

    参数:
        m: 输入数组。

    返回:
        ndarray: 上下翻转后的数组。
    """
    return flip(m, axis=0)


def roll(a, shift, axis=None):
    """
    沿指定轴滚动数组。

    参数:
        a: 输入数组。
        shift: 滚动的位数。
        axis: 滚动的轴。

    返回:
        ndarray: 滚动后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    data = arr.tolist()
    
    def roll_recursive(data, shift, axis, current_dim=0):
        if not isinstance(data, list):
            return data
        if axis is None:
            if not isinstance(data[0], list):
                shift = shift % len(data)
                return data[-shift:] + data[:-shift]
            flat = []
            for row in data:
                flat.extend(row)
            shift = shift % len(flat)
            flat = flat[-shift:] + flat[:-shift]
            result = []
            row_len = len(data[0])
            for i in range(len(data)):
                result.append(flat[i*row_len:(i+1)*row_len])
            return result
        if current_dim == axis:
            shift = shift % len(data)
            return data[-shift:] + data[:-shift]
        return [roll_recursive(row, shift, axis, current_dim + 1) for row in data]
    
    result_data = roll_recursive(data, shift, axis)
    return ndarray(result_data)


def rot90(m, k=1, axes=(0, 1)):
    """
    将数组旋转90度。

    参数:
        m: 输入数组。
        k: 旋转次数。
        axes: 旋转的轴。

    返回:
        ndarray: 旋转后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(m)
    k = k % 4
    
    for _ in range(k):
        arr = arr.swapaxes(*axes)[::-1]
    
    return arr

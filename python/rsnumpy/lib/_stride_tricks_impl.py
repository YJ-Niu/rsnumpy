# SPDX-License-Identifier: MIT
"""numpy.lib._stride_tricks_impl 兼容实现。

公共名从 rsnumpy 顶层 / rsnumpy.lib.stride_tricks 再导出。
sliding_window_view：Rust 后端暂无 strides 级 view，这里按 numpy 语义
生成值与形状正确的结果（当前为拷贝语义；strides 视图待 Rust 侧支持）。
"""

import builtins


def _rs():
    import rsnumpy
    return rsnumpy


def _broadcast_shape(*shapes):
    """计算广播结果形状（numpy._broadcast_shape 兼容）。"""
    rs = _rs()
    result = (0,)
    for shape in shapes:
        shape = tuple(shape)
        for i, dim in enumerate(reversed(shape)):
            if i >= len(result) or result[-(i + 1)] == 0:
                if i == len(result):
                    result = (0,) * i + result
                if result[-(i + 1)] not in (0, dim) and dim != 1:
                    raise ValueError(f"shapes {shapes} are not aligned")
                result = result[:-(i + 1)] + (dim,) + result[-i:]
    return result


def sliding_window_view(x, window_shape, axis=None, *, subok=False,
                        writeable=True):
    """滑动窗口视图（numpy.lib.stride_tricks.sliding_window_view 兼容）。

    axis 可为 None(滑所有轴，window_shape 须为长 ndim 的序列)、标量或序列；
    输出布局 = (各原轴序：滑轴取偏移量 d-w+1 / 非滑轴取原尺寸) + (各滑轴窗口尺寸)。
    writeable/subok 仅为兼容 numpy 签名接受（rsnumpy 视图恒可写、无子类视图）。
    当前为值正确的拷贝实现（Rust 后端无 strides view）。
    """
    rs = _rs()
    x = rs.asanyarray(x)
    _ = subok, writeable
    ndim = x.ndim
    is_seq = isinstance(window_shape, (list, tuple))
    ws = tuple(window_shape) if is_seq else (window_shape,)
    n = len(ws)
    if axis is None:
        if n != ndim:
            raise ValueError(
                f"Since axis is `None`, must provide window_shape for all "
                f"dimensions of `x`; got {n} window_shape elements and "
                f"`x.ndim` is {ndim}.")
        axes = tuple(builtins.range(ndim))
    elif isinstance(axis, int):
        if n != 1:
            raise ValueError(
                "Must provide matching length window_shape and axis.")
        axes = (axis,)
    else:
        axes = tuple(axis)
        if n != len(axes):
            raise ValueError(
                "Must provide matching length window_shape and axis.")
    if any(w <= 0 for w in ws):
        raise ValueError("`window_shape` cannot contain negative values")
    axes = tuple(a % ndim for a in axes)
    for w, a in zip(ws, axes):
        if w > x.shape[a]:
            raise ValueError(
                "window shape cannot be larger than input array shape")
    # 每个原轴（按原序）：滑轴 → (偏移量, True, 窗口)；非滑轴 → (尺寸, False, None)
    prefix = []
    win_of = {}
    for a, w in zip(axes, ws):
        win_of[a] = w
    for i in builtins.range(ndim):
        if i in win_of:
            prefix.append((x.shape[i] - win_of[i] + 1, True, win_of[i]))
        else:
            prefix.append((x.shape[i], False, None))

    def build(level, cur):
        if level == ndim:
            key = [None] * ndim
            for i, (size, isl, w) in builtins.enumerate(prefix):
                key[i] = (slice(cur[i], cur[i] + w) if isl else cur[i])
            return rs.asarray(x[tuple(key)]).tolist()
        return [build(level + 1, cur + [i])
                for i in builtins.range(prefix[level][0])]

    return rs.array(build(0, []), dtype=x.dtype)


def __getattr__(name):
    # 惰性从 rsnumpy 顶层 / lib.stride_tricks 取回（lib 在 rsnumpy
    # 初始化期间导入，模块级取属性会撞上部分初始化的命名空间）。
    if name == 'as_strided':
        from rsnumpy.lib.stride_tricks import as_strided
        return as_strided
    if name in ('broadcast_arrays', 'broadcast_shapes', 'broadcast_to'):
        import rsnumpy
        return getattr(rsnumpy, name)
    raise AttributeError(f"module 'rsnumpy.lib._stride_tricks_impl' has no "
                         f"attribute {name!r}")


__all__ = ['as_strided', 'broadcast_arrays', 'broadcast_shapes',
           'broadcast_to', 'sliding_window_view', '_broadcast_shape']

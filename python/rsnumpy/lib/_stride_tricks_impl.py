"""numpy.lib._stride_tricks_impl 兼容实现。

公共名从 rsnumpy 顶层 / rsnumpy.lib.stride_tricks 再导出。
sliding_window_view：Rust 后端暂无 strides 级 view，这里按 numpy 语义
生成值与形状正确的结果（当前为拷贝语义；strides 视图待 Rust 侧支持）。
"""


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


def sliding_window_view(x, window_shape):
    """滑动窗口视图（numpy.lib.stride_tricks.sliding_window_view 兼容）。

    窗口形状为标量或序列；返回形状 = x.shape[:-n] + (d-w+1, ...)。
    rsnumpy 当前实现为值正确的拷贝（Rust 后端无 strides view）。
    """
    rs = _rs()
    x = rs.asanyarray(x)
    if isinstance(window_shape, int):
        window_shape = (window_shape,)
    window_shape = tuple(window_shape)
    n = len(window_shape)
    if n > x.ndim:
        raise ValueError(
            f"shape of sliding view ({window_shape}) must have at most as many "
            f"dimensions as the original array ({x.shape})")
    if any(w <= 0 for w in window_shape):
        raise ValueError("window_shape must be positive")
    for w, d in zip(window_shape, x.shape[x.ndim - n:]):
        if w > d:
            raise ValueError(
                f"window size {w} exceeds dimension size {d}")
    from itertools import product
    base = x.shape[:x.ndim - n]
    out = []
    for lead in product(*[range(d) for d in base]):
        sub = []
        for off in product(*[range(d - w + 1)
                             for d, w in zip(x.shape[x.ndim - n:],
                                             window_shape)]):
            key = lead + tuple(
                slice(off, off + w) for off, w in zip(off, window_shape))
            sub.append(x[key].tolist())
        out.append(sub)
    return rs.array(out, dtype=x.dtype)


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

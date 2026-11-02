"""numpy.lib._nanfunctions_impl 兼容实现（私有辅助）。

nanmean/nanmedian 等的核心辅助：NaN 掩码、NaN 替换、按计数除零安全除法。
"""


def _rs():
    import rsnumpy
    return rsnumpy


def _nan_mask(a, out=None):
    """返回 a 中非有限值（NaN/±inf）的布尔掩码（numpy._nan_mask 兼容）。

    非浮点 dtype：无 NaN 概念，返回全 False（out 给定则写入 out）。
    """
    rs = _rs()
    a = rs.asanyarray(a)
    if a.dtype.kind not in ('f', 'c'):
        mask = rs.zeros(a.shape, dtype='bool')
    else:
        mask = ~rs.isfinite(a)
    if out is not None:
        rs.copyto(out, mask)
        return out
    return mask


def _replace_nan(a, val):
    """把 a 的 NaN 换成 val，返回 (result, mask)（numpy._replace_nan 兼容）。

    无 NaN 时 result 是 a 本身（不拷贝）；非浮点 dtype 的 mask 为 None。
    """
    rs = _rs()
    a = rs.asanyarray(a)
    if a.dtype.kind == 'f':
        if val == 0 and a.dtype.kind == 'f':
            mask = a == 0
            if bool(mask.any()):
                a = a.copy()
                a[mask] = 1
        else:
            mask = ~rs.isfinite(a)
            if bool(mask.any()):
                a = a.copy()
                a[mask] = val
    else:
        mask = None
    return a, mask


def _divide_by_count(a, count, out=None):
    """a / count，count==0 处置 0（numpy._divide_by_count 兼容）。

    count==1 处直接取 a（避免浮点除法精度损失）。
    """
    rs = _rs()
    a = rs.asanyarray(a)
    count = rs.asanyarray(count)
    if out is None:
        out = rs.array(1.0, dtype=a.dtype)
    nzc = count != 0
    try:
        a_b = rs.broadcast_to(a, count.shape)
    except Exception:
        a_b = a
    if not bool(nzc.any()):
        out[...] = 0
        return out
    with rs.errstate(invalid='ignore', divide='ignore'):
        rs.divide(a_b, count, out=out, where=nzc)
    ones = nzc & (count == 1)
    if bool(ones.any()):
        out[ones] = a_b[ones]
    zero = ~nzc
    if bool(zero.any()):
        out[zero] = 0
    return out


__all__ = ['_nan_mask', '_replace_nan', '_divide_by_count']

"""numpy._core._multiarray_umath 兼容实现（Python 版）。

MAXDIMS 与 rsnumpy 嵌套序列发现路径的实际维度上限一致（64）。
"""

MAXDIMS = 64

__all__ = ['MAXDIMS', '_discover_array_parameters']


def _rs():
    import rsnumpy
    return rsnumpy


def _discover_shape(sequence):
    """按 numpy 公共前缀规则推断嵌套序列形状。"""
    rs = _rs()
    if isinstance(sequence, rs.ndarray):
        return tuple(sequence.shape)
    if not isinstance(sequence, (list, tuple)):
        return ()
    if len(sequence) == 0:
        return (0,)
    subs = []
    for el in sequence:
        if isinstance(el, rs.ndarray):
            subs.append(tuple(el.shape))
        elif isinstance(el, (list, tuple)):
            subs.append(_discover_shape(el))
        else:
            return (len(sequence),)
    prefix = ()
    first = subs[0]
    for i in range(len(first)):
        if all(len(s) > i and s[i] == first[i] for s in subs[1:]):
            prefix += (first[i],)
        else:
            break
    return (len(sequence),) + prefix


def _type_matches(dt, hint):
    try:
        return dt.type is hint
    except Exception:
        return False


def _discover_array_parameters(sequence, dtype=None, *args, **kwargs):
    """推断 (dtype, shape)（numpy C-helper 兼容）。

    空序列时 dtype 由 hint 类型决定：
    - bytes_/unicode_ 按 numpy 惯例映射为 S1/U1；
    - datetime64/timedelta64 类型类优先匹配 M8/m8 spec（rsnumpy 的
      M/m dtype 类型类为 unicode_，与 U 同形，无法直接反解析）；
    - 其余类型先直接反解析，失败后按 .type 逐一匹配候选 spec。
    """
    rs = _rs()
    if dtype is None:
        dt = rs.dtype('float64')
    elif isinstance(dtype, rs.dtype):
        dt = dtype
    else:
        dt = None
        if dtype is rs.bytes_:
            dt = rs.dtype('S1')
        elif dtype is rs.unicode_:
            dt = rs.dtype('U1')
        elif dtype is rs.void:
            dt = rs.dtype('V8')
        if dt is None:
            # datetime64/timedelta64 类型类特判（.type 与 U 同形）
            for spec in ('M8', 'm8'):
                try:
                    cand = rs.dtype(spec)
                except Exception:
                    continue
                if _type_matches(cand, dtype):
                    dt = cand
                    break
        if dt is None and isinstance(dtype, type):
            try:
                if issubclass(dtype, rs.dtype):
                    # dtype 类本身（rsnumpy 扁平 dtype 类，NumPy 2 中为各具体
                    # DType 类）：空序列时按 numpy 传统默认 float64
                    dt = rs.dtype('float64')
            except TypeError:
                pass
        if dt is None:
            try:
                dt = rs.dtype(dtype)
            except Exception:
                for spec in ('M', 'm', 'V8', 'M8', 'm8', 'O'):
                    try:
                        cand = rs.dtype(spec)
                    except Exception:
                        continue
                    if _type_matches(cand, dtype):
                        dt = cand
                        break
                if dt is None:
                    raise
    return dt, _discover_shape(sequence)

"""rsnumpy.char - 向量化字符串操作模块（Rust 层实现，本模块为薄包装）。

所有元素级字符串运算由 Rust 端 ``num_core.char_str_op/char_int_op/
char_bool_op/char_add/char_multiply`` 一次跨边界批量完成。
"""

import builtins

import rsnumpy.num_core as _core


def _np_mod():
    import rsnumpy
    return rsnumpy


def _get_strings(a):
    """从 ndarray 或列表中提取字符串列表（bytes 解码为 utf-8 str）。"""
    if isinstance(a, (str, bytes)):
        return [a]
    rd = getattr(a, '_raw_data', None)
    if rd is None and hasattr(a, '_inst'):
        rd = getattr(a._inst, '_raw_data', None)
    if rd is not None:
        return builtins.list(rd)
    if hasattr(a, 'tolist'):
        return _flatten(a.tolist(), [])
    if isinstance(a, (builtins.list, builtins.tuple)):
        return _flatten(a, [])
    return _flatten([a], [])


def _flatten(x, out):
    for v in x:
        if isinstance(v, (builtins.list, builtins.tuple)):
            _flatten(v, out)
        else:
            out.append(v)
    return out


def _decode(vals):
    """bytes 元素解码（surrogateescape 往返无损）。"""
    return [v.decode('utf-8', 'surrogateescape') if isinstance(v, bytes) else v
            for v in vals]


def _is_bytes(vals):
    return builtins.any(isinstance(v, bytes) for v in vals)


def _a_was_bytes(a):
    """数组输入：以 dtype.kind 为准（tolist/_raw_data 元素类型不可信）。"""
    try:
        k = a.dtype.kind
        if k in 'SU':
            return k == 'S'
    except Exception:
        pass
    return _is_bytes(_get_strings(a))


def _shape_of(a):
    try:
        return builtins.tuple(a.shape)
    except Exception:
        return ()


def _nest(flat, dims):
    """flat → 嵌套列表（最外层长度 = dims[0]，1 维即 flat 本身）。"""
    nested = builtins.list(flat)
    for dim in builtins.reversed(dims[1:]):
        nested = [nested[i:i + dim]
                  for i in range(0, builtins.len(nested), dim)]
    return nested


def _bc_shapes(shapes):
    """numpy 式形状广播（尾对齐，1 可扩，短形状前导补 1）。"""
    ss = [builtins.tuple(reversed(s)) for s in shapes if s]
    out = []
    nd = builtins.max((builtins.len(s) for s in ss), default=0)
    for i in builtins.range(nd):
        ds = [s[i] for s in ss if i < builtins.len(s) and s[i] != 1]
        if builtins.len(set(ds)) > 1:
            raise ValueError('could not broadcast input shapes')
        out.append(ds[0] if ds else 1)
    return builtins.tuple(reversed(out))


def _bc_linear(shape, idx):
    """广播形状 → 扁平索引（尾对齐，dim=1 → 0）。"""
    if not shape:
        return 0
    pad = builtins.max(0, builtins.len(idx) - builtins.len(shape))
    padded = (1,) * pad + builtins.tuple(shape)
    off = 0
    for d, i in builtins.zip(padded, idx):
        if d != 1:
            off = off * d + i
    return off


def _bc_get(flat, shape, idx):
    if builtins.len(flat) <= 1 or not shape:
        return flat[0]
    return flat[_bc_linear(shape, idx)]


def _iter_indices(shape):
    if not shape:
        yield ()
        return
    total = 1
    for d in shape:
        total *= d
    for n in builtins.range(total):
        idx = []
        m = n
        for d in reversed(shape):
            idx.append(m % d)
            m //= d
        yield builtins.tuple(reversed(idx))


def _operand(x):
    """把 values 输入转成 (flat 列表, shape)；str 标量 → 单元素。

    numpy 语义：数值标量/数组元素提升为 rsnumpy 标量（repr 为
    'np.int64(1)' 等），保证 '%r' 格式化与 numpy 一致。"""
    if x is None:
        return None, ()
    if isinstance(x, (str, bytes)):
        return [_decode([x])[0]], ()
    if isinstance(x, bool):
        return [builtins.int(x)], ()
    if isinstance(x, (int, float)):
        from .__init__ import int64 as _i64, float64 as _f64
        return [(_f64(x) if isinstance(x, float) else _i64(x))], ()
    flat = _decode(_get_strings(x))
    if isinstance(x, (builtins.list, builtins.tuple)):
        shape = _nested_shape(x)
    else:
        shape = _shape_of(x)
    # 数值输入（标量已处理）：逐元素包装 rsnumpy 标量类（numpy '%r' 语义）。
    if flat and all(isinstance(v, (int, float)) for v in flat):
        from .__init__ import _SCALAR_CLASS_BY_DTYPE, int64 as _i64
        from .__init__ import float64 as _f64
        if isinstance(x, (builtins.list, builtins.tuple)):
            _cls = _f64 if isinstance(flat[0], float) else _i64
        else:
            _cls = _SCALAR_CLASS_BY_DTYPE.get(
                str(getattr(x, '_dtype', '')))
            if _cls is None:
                _cls = _f64 if isinstance(flat[0], float) else _i64
        flat = [_cls(v) for v in flat]
    return flat, shape


def _wrap_str(vals, shape, was_bytes=False):
    """把扁平结果列表按 shape 包装为字符串数组。

    裸 'S'/'U' dtype 按内容推断最小宽度（保留 bytes/str 类别：S 输入→
    S 结果，U 输入→U 结果），宽度随内容增长（如 replace 变宽）。
    """
    from .__init__ import array as _array
    kind = 'S' if was_bytes else 'U'
    if not shape:
        return _array([vals[0] if vals else ''], dtype=kind)
    nested = vals
    for dim in builtins.reversed(shape[1:]):
        nested = [nested[i:i + dim] for i in range(0, builtins.len(nested), dim)]
    return _array(nested, dtype=kind)


def _str_op(op, a, arg1=None, arg2=None, arg3=None):
    """字符串算子薄包装：解码 → Rust → 编码 → 包装。"""
    vals = _decode(_get_strings(a))
    out = _core.char_str_op(op, vals,
                            None if arg1 is None else _as_str_arg(arg1),
                            None if arg2 is None else _as_str_arg(arg2),
                            None if arg3 is None else _as_str_arg(arg3))
    if _a_was_bytes(a):
        out = [v.encode('utf-8', 'surrogateescape') for v in out]
    shape = _shape_of(a)
    if isinstance(a, (str, bytes)):
        return out[0] if out else ''
    if shape == ():
        return out[0] if out else ''
    return _wrap_str(out, shape)


def _as_str_arg(x):
    """算子参数转字符串（宽度/填充符等）。"""
    if isinstance(x, (str, bytes)):
        return x.decode('utf-8', 'surrogateescape') if isinstance(x, bytes) else x
    return builtins.str(x)


def _int_op(op, a, arg1=None, arg2=None, arg3=None):
    vals = _decode(_get_strings(a))
    out = _core.char_int_op(op, vals,
                            None if arg1 is None else _as_str_arg(arg1),
                            None if arg2 is None else _as_str_arg(arg2),
                            None if arg3 is None else _as_str_arg(arg3))
    shape = _shape_of(a)
    from .__init__ import ndarray
    if isinstance(a, (str, bytes)) or shape == ():
        # numpy 语义：标量输入 → 0-d int 数组。
        res = ndarray([builtins.float(out[0]) if out else -1], _dtype='int64')
        return res.reshape(())
    res = ndarray([builtins.float(v) for v in out], _dtype='int64')
    return res.reshape(shape)


def _bool_op(op, a, arg1=None, other=None):
    vals = _decode(_get_strings(a))
    other_vals = None
    if other is not None:
        other_vals = _decode(_get_strings(other))
    out = _core.char_bool_op(op, vals,
                             None if arg1 is None else _as_str_arg(arg1),
                             other_vals)
    shape = _shape_of(a)
    from .__init__ import ndarray
    if isinstance(a, (str, bytes)) or shape == ():
        # numpy 语义：标量比较返回 0-d bool 数组（isinstance ndarray）。
        val = builtins.bool(out[0]) if out else False
        res = ndarray([1.0 if val else 0.0])
        res._dtype = 'bool'
        return res.reshape(())
    res = ndarray([1.0 if v else 0.0 for v in out])
    res._dtype = 'bool'
    return res.reshape(shape)


# ==================== 模块级函数（np.char.*） ====================

def add(x1, x2):
    """逐元素字符串连接（np.char.add）。"""
    v1 = _decode(_get_strings(x1))
    v2 = _decode(_get_strings(x2))
    out = _core.char_add(v1, v2)
    if _a_was_bytes(x1):
        out = [v.encode('utf-8', 'surrogateescape') for v in out]
    shape = _shape_of(x1)
    if shape == () and _shape_of(x2) == ():
        return out[0] if out else ''
    if shape == ():
        shape = _shape_of(x2)
    res = _wrap_str(out, shape)
    if isinstance(x1, chararray) or isinstance(x2, chararray):
        return _as_chararray(res)
    return res


def multiply(a, i):
    """逐元素重复（np.char.multiply）。"""
    vals = _decode(_get_strings(a))
    if hasattr(i, 'tolist'):
        fl = []
        _flatten(i.tolist(), fl)
        ns = [builtins.int(v) for v in fl]
    elif isinstance(i, (builtins.list, builtins.tuple)):
        fl = []
        _flatten(builtins.list(i), fl)
        ns = [builtins.int(v) for v in fl]
    else:
        if not isinstance(i, (int,)) or isinstance(i, bool):
            raise ValueError('Can only multiply by integers')
        ns = [builtins.int(i)]
    out = _core.char_multiply(vals, ns)
    if _is_bytes(_get_strings(a)):
        out = [v.encode('utf-8', 'surrogateescape') for v in out]
    shape = _shape_of(a)
    if shape == ():
        return out[0] if out else ''
    return _wrap_str(out, shape)


def capitalize(a):
    """首字母大写，其余小写。"""
    return _str_op('capitalize', a)


def title(a):
    """numpy title 语义（非字母后首字母大写）。"""
    return _str_op('title', a)


def lower(a):
    """全部小写。"""
    return _str_op('lower', a)


def upper(a):
    """全部大写。"""
    return _str_op('upper', a)


def swapcase(a):
    """大小写互换。"""
    return _str_op('swapcase', a)


def strip(a, chars=None):
    """两侧去字符。"""
    return _str_op('strip', a, chars)


def lstrip(a, chars=None):
    return _str_op('lstrip', a, chars)


def rstrip(a, chars=None):
    return _str_op('rstrip', a, chars)


def _pad_op(op, a, width, fillchar=None):
    """center/ljust/rjust/zfill：width 支持标量/数组（numpy 广播）。"""
    vals = _decode(_get_strings(a))
    a_shape = _shape_of(a)
    was_bytes = _a_was_bytes(a)
    w_flat, w_shape = _operand(width)
    bshape = _bc_shapes([a_shape, w_shape]) if w_shape else a_shape
    fc = _as_str_arg(fillchar) if fillchar is not None else ' '
    out = []
    for idx in _iter_indices(bshape):
        v = _bc_get(vals, a_shape, idx)
        w = _bc_get(w_flat, w_shape, idx)
        if op == 'zfill':
            out.append(v.zfill(w))
        else:
            out.append(getattr(v, op)(w, fc))
    res = _wrap_str(out, bshape, was_bytes=was_bytes)
    if isinstance(a, chararray):
        return _as_chararray(res)
    return res


def center(a, width, fillchar=' '):
    """居中填充（width 支持数组广播）。"""
    return _pad_op('center', a, width, fillchar)


def ljust(a, width, fillchar=' '):
    return _pad_op('ljust', a, width, fillchar)


def rjust(a, width, fillchar=' '):
    return _pad_op('rjust', a, width, fillchar)


def zfill(a, width):
    return _pad_op('zfill', a, width)


def replace(a, old, new, count=None):
    """逐元素替换（old/new/count 支持数组广播，numpy 语义）。"""
    vals = _decode(_get_strings(a))
    a_shape = _shape_of(a)
    was_bytes = _a_was_bytes(a)
    o_flat, o_shape = _operand(old)
    n_flat, n_shape = _operand(new)
    c_flat, c_shape = _operand(count)
    bshape = a_shape
    for sh in (o_shape, n_shape, c_shape):
        bshape = _bc_shapes([bshape, sh]) if sh else bshape
    out = []
    for idx in _iter_indices(bshape):
        v = _bc_get(vals, a_shape, idx)
        o = _bc_get(o_flat, o_shape, idx)
        n = _bc_get(n_flat, n_shape, idx)
        c = -1 if c_flat is None else _bc_get(c_flat, c_shape, idx)
        out.append(v.replace(o, n, c))
    res = _wrap_str(out, bshape, was_bytes=was_bytes)
    if isinstance(a, chararray):
        return _as_chararray(res)
    return res


def expandtabs(a, tabsize=8):
    return _str_op('expandtabs', a, tabsize)


def join(sep, a):
    """以 sep 连接序列元素（sep 支持数组逐元素，numpy 广播）。"""
    s = _decode(_get_strings(a))
    a_shape = _shape_of(a)
    if isinstance(sep, (str, bytes)):
        seps = [_decode([sep])[0]]
        sep_shape = ()
    else:
        seps = _decode(_get_strings(sep))
        sep_shape = (_nested_shape(sep)
                     if isinstance(sep, (builtins.list, builtins.tuple))
                     else _shape_of(sep))
    bshape = _bc_shapes([a_shape, sep_shape]) if sep_shape else a_shape
    out = []
    for idx in _iter_indices(bshape):
        v = _bc_get(s, a_shape, idx)
        sp = _bc_get(seps, sep_shape, idx)
        out.append(sp.join(v))
    if isinstance(a, (str, bytes)) and not sep_shape:
        return out[0] if out else ''
    res = _wrap_str(out, bshape, was_bytes=False)
    if isinstance(a, chararray):
        return _as_chararray(res)
    return res


def find(a, sub, start=0, end=None):
    """逐元素 str.find（未找到返回 -1）。"""
    return _int_op('find', a, sub, start, end)


def rfind(a, sub, start=0, end=None):
    return _int_op('rfind', a, sub, start, end)


def index(a, sub, start=0, end=None):
    return _int_op('index', a, sub, start, end)


def rindex(a, sub, start=0, end=None):
    return _int_op('rindex', a, sub, start, end)


def count(a, sub, start=0, end=None):
    return _int_op('count', a, sub)


def startswith(a, prefix, start=0, end=None):
    return _bool_op('startswith', a, prefix)


def endswith(a, suffix, start=0, end=None):
    return _bool_op('endswith', a, suffix)


def equal(x1, x2):
    """逐元素 ==。"""
    return _bool_op('equal', x1, other=x2)


def not_equal(x1, x2):
    return _bool_op('not_equal', x1, other=x2)


def greater(x1, x2):
    return _bool_op('greater', x1, other=x2)


def greater_equal(x1, x2):
    return _bool_op('greater_equal', x1, other=x2)


def less(x1, x2):
    return _bool_op('less', x1, other=x2)


def less_equal(x1, x2):
    return _bool_op('less_equal', x1, other=x2)


def _is_method(a, op):
    return _bool_op(op, a)


def isalnum(a):
    return _is_method(a, 'isalnum')


def isalpha(a):
    return _is_method(a, 'isalpha')


def isdecimal(a):
    return _is_method(a, 'isdecimal')


def isdigit(a):
    return _is_method(a, 'isdigit')


def islower(a):
    return _is_method(a, 'islower')


def isnumeric(a):
    return _is_method(a, 'isnumeric')


def isspace(a):
    return _is_method(a, 'isspace')


def istitle(a):
    return _is_method(a, 'istitle')


def isupper(a):
    return _is_method(a, 'isupper')


def split(a, sep=None, maxsplit=None):
    """逐元素分割（Rust 层暂不支持，回退 Python）。

    numpy 的 split 返回 ragged object 数组，无法映射为规则字符串算子。
    """
    s = _get_strings(a)
    result = [v.split(sep, maxsplit) if maxsplit is not None
              else v.split(sep) for v in s]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def rsplit(a, sep=None, maxsplit=None):
    s = _get_strings(a)
    result = [v.rsplit(sep, maxsplit) if maxsplit is not None
              else v.rsplit(sep) for v in s]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def splitlines(a, keepends=False):
    s = _get_strings(a)
    result = [v.splitlines(keepends) for v in s]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def decode(a, encoding='utf-8', errors='strict'):
    """数组元素依次调用 bytes.decode（返回 U 数组）。"""
    s = _get_strings(a)
    result = [x.decode(encoding, errors) if isinstance(x, bytes) else x
              for x in s]
    if isinstance(a, (str, bytes)):
        return result[0] if result else ''
    shape = _shape_of(a)
    if not shape:
        return result[0] if result else ''
    from .__init__ import array as _array
    if builtins.len(result) == 0:
        res = _array([], dtype='U')
        if builtins.len(shape) > 1:
            res = res.reshape(shape)
    else:
        res = _array(_nest(result, shape), dtype='U')
    if isinstance(a, chararray):
        return _as_chararray(res)
    return res


def encode(a, encoding='utf-8', errors='strict'):
    """数组元素依次调用 str.encode（返回 S 数组）。"""
    s = _get_strings(a)
    result = [x.encode(encoding, errors) if isinstance(x, str) else x
              for x in s]
    if isinstance(a, (str, bytes)):
        return result[0] if result else b''
    shape = _shape_of(a)
    if not shape:
        return result[0] if result else b''
    from .__init__ import array as _array
    if builtins.len(result) == 0:
        res = _array([], dtype='S')
        if builtins.len(shape) > 1:
            res = res.reshape(shape)
    else:
        res = _array(_nest(result, shape), dtype='S')
    if isinstance(a, chararray):
        return _as_chararray(res)
    return res


def partition(a, sep):
    """逐元素 partition（结果 shape+(3,)；标量输入返回 3 元列表）。"""
    s = _decode(_get_strings(a))
    a_shape = _shape_of(a)
    if isinstance(a, (str, bytes)):
        seps = _decode(_get_strings(sep))
        r = s[0].partition(seps[0] if seps else '')
        return [r[0], r[1], r[2]]
    seps = _decode(_get_strings(sep))
    sep_shape = () if isinstance(sep, (str, bytes)) else (
        _nested_shape(sep) if isinstance(sep, (builtins.list, builtins.tuple))
        else _shape_of(sep))
    was_bytes = _a_was_bytes(a)
    bshape = _bc_shapes([a_shape, sep_shape]) if sep_shape else a_shape
    out = []
    for idx in _iter_indices(bshape):
        v = _bc_get(s, a_shape, idx)
        sp = _bc_get(seps, sep_shape, idx)
        out.extend(builtins.list(v.partition(sp)))
    shape = bshape + (3,)
    from .__init__ import array as _array
    res = _array(_nest(out, shape), dtype='S' if was_bytes else 'U')
    if isinstance(a, chararray):
        return _as_chararray(res)
    return res


def rpartition(a, sep):
    """逐元素 rpartition（结果 shape+(3,)；标量输入返回 3 元列表）。"""
    s = _decode(_get_strings(a))
    a_shape = _shape_of(a)
    if isinstance(a, (str, bytes)):
        seps = _decode(_get_strings(sep))
        r = s[0].rpartition(seps[0] if seps else '')
        return [r[0], r[1], r[2]]
    seps = _decode(_get_strings(sep))
    sep_shape = () if isinstance(sep, (str, bytes)) else (
        _nested_shape(sep) if isinstance(sep, (builtins.list, builtins.tuple))
        else _shape_of(sep))
    was_bytes = _a_was_bytes(a)
    bshape = _bc_shapes([a_shape, sep_shape]) if sep_shape else a_shape
    out = []
    for idx in _iter_indices(bshape):
        v = _bc_get(s, a_shape, idx)
        sp = _bc_get(seps, sep_shape, idx)
        out.extend(builtins.list(v.rpartition(sp)))
    shape = bshape + (3,)
    from .__init__ import array as _array
    res = _array(_nest(out, shape), dtype='S' if was_bytes else 'U')
    if isinstance(a, chararray):
        return _as_chararray(res)
    return res


def mod(a, values):
    """逐元素 % 格式化（numpy char.mod，支持广播）。"""
    s = _get_strings(a)
    a_shape = _shape_of(a)
    v_flat, v_shape = _operand(values)
    was_bytes = _a_was_bytes(a)
    bshape = _bc_shapes([a_shape, v_shape]) if v_shape else a_shape
    out = []
    for idx in _iter_indices(bshape):
        v = _bc_get(s, a_shape, idx)
        x = _bc_get(v_flat, v_shape, idx)
        if isinstance(v, bytes) and isinstance(x, str):
            x = x.encode('utf-8', 'surrogateescape')
        out.append(v % x)
    if isinstance(a, (str, bytes)) and not v_shape:
        return out[0]
    res = _wrap_str(out, bshape, was_bytes=was_bytes)
    if isinstance(a, chararray):
        return _as_chararray(res)
    return res


def _make_result(strings):
    """将字符串列表包装为 ndarray（ragged 场景 object 数组）。"""
    from .__init__ import ndarray
    return ndarray._wrap_raw(strings)


def str_len(a, axis=None):
    """逐元素字符串长度（np.char.str_len）→ 整数数组。"""
    s = _decode(_get_strings(a))
    if isinstance(a, (str, bytes)):
        return builtins.len(s[0])
    flat = [builtins.len(v) for v in s]
    shape = _shape_of(a)
    from .__init__ import array as _array
    if not shape:
        return flat[0] if flat else 0
    nested = flat
    for dim in builtins.reversed(shape[1:]):
        nested = [nested[i:i + dim]
                  for i in range(0, builtins.len(nested), dim)]
    return _array(nested, dtype='int64')


# ==================== chararray（组合代理，方法委托 Rust） ====================

_RUST_STR_OPS = {
    'capitalize': ('capitalize',), 'lower': ('lower',), 'upper': ('upper',),
    'title': ('title',), 'swapcase': ('swapcase',), 'strip': ('strip',),
    'lstrip': ('lstrip',), 'rstrip': ('rstrip',), 'expandtabs': ('expandtabs',),
}
_RUST_ARG_OPS = {
    'center': ('center', 2), 'ljust': ('ljust', 2), 'rjust': ('rjust', 2),
    'zfill': ('zfill', 1), 'replace': ('replace', None),
}
_RUST_INT_OPS = {'find': 1, 'rfind': 1, 'index': 1, 'rindex': 1, 'count': 1}
_RUST_BOOL_OPS = {'startswith': 1, 'endswith': 1, 'isalnum': 0, 'isalpha': 0,
                  'isdecimal': 0, 'isdigit': 0, 'islower': 0, 'isnumeric': 0,
                  'isspace': 0, 'istitle': 0, 'isupper': 0}


def _nested_shape(data):
    shape = []
    while isinstance(data, (builtins.list, builtins.tuple)):
        shape.append(builtins.len(data))
        data = data[0] if data else None
    return builtins.tuple(shape) or (0,)


def _rstrip_tree(x):
    """递归 rstrip str/bytes 元素（numpy char.array 构造语义）。"""
    if isinstance(x, (str, bytes)):
        return x.rstrip()
    if isinstance(x, (builtins.list, builtins.tuple)):
        return [_rstrip_tree(v) for v in x]
    return x


class chararray:
    """字符串/字节 ndarray（numpy chararray 语义，方法委托 Rust 算子）。

    组合代理：内部持有真实 ndarray（_inst），ndarray 协议经 __getattr__
    代理，元素级字符串方法经 _core.char_* 算子在 Rust 层完成。
    """

    def __new__(cls, obj, itemsize=None, copy=True, unicode=None, order=None):
        from .__init__ import ndarray, array as _array
        _ = order
        if isinstance(obj, chararray):
            if not copy:
                return obj
            inst = ndarray(obj._inst._array.copy())
            inst._dtype = getattr(obj._inst, '_dtype', 'string_')
            inst._raw_data = builtins.list(
                getattr(obj._inst, '_raw_data', None) or [])
            inst._str_dtype = getattr(obj._inst, '_str_dtype', None)
            return _as_chararray(inst)
        if hasattr(obj, '_array'):
            if unicode is False:
                # numpy 语义：U 数组 → S 需 ascii 编码，非 ascii 抛错。
                _rd = list(getattr(obj, '_raw_data', None) or [])
                for _v in _rd:
                    if isinstance(_v, str):
                        _v.encode('ascii')
                shape = _shape_of(obj)
                inst = _array(list(_rd), dtype='S')
                if builtins.len(shape) > 1:
                    inst = inst.reshape(shape)
            elif unicode is True:
                _rd = list(getattr(obj, '_raw_data', None) or [])
                if getattr(getattr(obj, 'dtype', None), 'kind', 'U') == 'S':
                    _dec = [(_v.decode('utf-8', 'surrogateescape')
                             if isinstance(_v, (bytes, bytearray))
                             else str(_v)) for _v in _rd]
                    shape = _shape_of(obj)
                    inst = _array(list(_dec), dtype='U')
                    if builtins.len(shape) > 1:
                        inst = inst.reshape(shape)
                else:
                    inst = obj if not copy else obj.copy()
            else:
                inst = obj if not copy else obj.copy()
        else:
            # numpy char.array：Python 输入构造期元素 rstrip
            # （全空白元素 → 空串）。ndarray 输入保持原样。
            obj = _rstrip_tree(obj)
            if isinstance(obj, (str, bytes)):
                # numpy 语义：单个 str/bytes 是 1 维单元素数组（|S3 / <U1），
                # 元素本身为完整字符串（len(A)==1, len(A[0])==len(输入)）。
                inst = _array([obj], dtype='S' if isinstance(obj, bytes) else 'U')
            else:
                flat0 = None
                if isinstance(obj, (builtins.list, builtins.tuple)) and obj:
                    fl = []
                    _flatten(builtins.list(obj), fl)
                    flat0 = fl[0] if fl else None
                if unicode is True or (unicode is None and
                                       isinstance(flat0, str)):
                    inst = _array(obj, dtype='U')
                elif unicode is False:
                    # numpy 语义：强制 ascii 编码，非 ascii 抛 UnicodeEncodeError。
                    for _v in _flatten(builtins.list(obj), []):
                        if isinstance(_v, str):
                            _v.encode('ascii')
                    inst = _array(obj, dtype='S')
                else:
                    inst = _array(obj)
        return _as_chararray(inst)

    def __getattr__(self, name):
        # ndarray 协议代理：shape/dtype/tolist/ndim/size/reshape/astype/...
        return getattr(self._inst, name)

    def astype(self, dtype, *a, **k):
        # numpy 语义：S/U 结果保持 chararray，其余返回普通 ndarray。
        res = self._inst.astype(dtype, *a, **k)
        from .__init__ import ndarray as _ndarray
        if isinstance(res, _ndarray) and res.dtype.kind in 'SU':
            return _as_chararray(res)
        return res

    def _vals(self):
        rd = getattr(self._inst, '_raw_data', None)
        if rd is not None:
            return _flatten(builtins.list(rd), [])
        return _flatten(self._inst.tolist(), [])

    def _apply(self, op, args):
        """元素级方法委托 Rust 算子。"""
        vals = _decode(self._vals())
        # S dtype 数组 tolist() 返回 str（字节边界丢失），以 dtype kind 为准。
        try:
            was_bytes = self._inst.dtype.kind == 'S'
        except Exception:
            was_bytes = _is_bytes(vals)
        shape = _shape_of(self._inst)
        # None 参数保持 None（stringify 成 'None' 会改变 strip 等语义）；
        # 数组参数传 None（Rust 端按"未指定"处理）。
        str_args = []
        for x in args:
            if x is None:
                str_args.append(None)
            elif (isinstance(x, (builtins.list, builtins.tuple))
                  or hasattr(x, 'tolist')):
                str_args.append(None)
            else:
                str_args.append(_as_str_arg(x))
        if op in _RUST_STR_OPS or op in _RUST_ARG_OPS:
            a1 = str_args[0] if builtins.len(str_args) > 0 else None
            a2 = str_args[1] if builtins.len(str_args) > 1 else None
            a3 = str_args[2] if builtins.len(str_args) > 2 else None
            if op == 'replace' and builtins.len(args) < 3:
                a3 = None
            out = _core.char_str_op(op, vals, a1, a2, a3)
        elif op in _RUST_INT_OPS:
            out = _core.char_int_op(op, vals,
                                    str_args[0] if str_args else None)
            return self._pack(out, shape, as_int=True)
        elif op in _RUST_BOOL_OPS:
            out = _core.char_bool_op(op, vals,
                                     str_args[0] if str_args else None)
            return self._pack(out, shape, as_bool=True)
        else:
            raise AttributeError(op)
        return self._pack(out, shape, was_bytes=was_bytes)

    def _pack(self, vals, shape, as_int=False, as_bool=False,
              was_bytes=False):
        from .__init__ import ndarray
        if as_int:
            res = ndarray([builtins.float(v) for v in vals], _dtype='int64')
            return res.reshape(shape) if shape else res
        if as_bool:
            res = ndarray([1.0 if v else 0.0 for v in vals])
            res._dtype = 'bool'
            return res.reshape(shape) if shape else res
        if was_bytes:
            vals = [v.encode('utf-8', 'surrogateescape')
                    if isinstance(v, str) else v for v in vals]
        return _as_chararray(_wrap_str(vals, shape, was_bytes=was_bytes))

    def capitalize(self):
        return self._apply('capitalize', [])

    def lower(self):
        return self._apply('lower', [])

    def upper(self):
        return self._apply('upper', [])

    def title(self):
        return self._apply('title', [])

    def swapcase(self):
        return self._apply('swapcase', [])

    def strip(self, chars=None):
        return self._cstrip('strip', chars)

    def lstrip(self, chars=None):
        return self._cstrip('lstrip', chars)

    def rstrip(self, chars=None):
        return self._cstrip('rstrip', chars)

    def _cstrip(self, op, chars):
        try:
            kind = self._inst.dtype.kind
        except Exception:
            kind = 'U'
        if (chars is not None
                and (isinstance(chars, (builtins.list, builtins.tuple))
                     or hasattr(chars, 'tolist'))):
            # chars 数组：逐元素（numpy 广播 i % len）。
            # strip：chars strip 后 C 截断（含 \0）；lstrip：chars lstrip 后
            # 去尾部空白；rstrip：仅 chars rstrip（numpy C 实现语义）。
            vals = _decode(self._vals())
            cs = _decode(_get_strings(chars))

            def _c_trunc(s):
                parts = s.split('\x00', 1)
                return parts[0] + '\x00' if builtins.len(parts) > 1 else parts[0]

            if op == 'strip':
                out = [_c_trunc(getattr(v, op)(cs[i % builtins.len(cs)]))
                       for i, v in builtins.enumerate(vals)]
            elif op == 'lstrip':
                out = [getattr(v, op)(cs[i % builtins.len(cs)]).rstrip()
                       for i, v in builtins.enumerate(vals)]
            else:
                out = [_c_trunc(getattr(v, op)(cs[i % builtins.len(cs)]))
                       for i, v in builtins.enumerate(vals)]
            return self._pack(out, _shape_of(self._inst),
                              was_bytes=kind == 'S')
        if kind in 'SU':
            if op == 'lstrip':
                # lstrip 无参：仅空白 lstrip（不 \0 截断，numpy 语义）。
                return self._apply(op, [chars])
            # strip/rstrip 无参：C 字符串截断（除 \0）+ 空白 op。
            vals = [v.split('\x00', 1)[0] for v in _decode(self._vals())]
            sub = _as_chararray(_wrap_str(
                vals, _shape_of(self._inst), was_bytes=kind == 'S'))
            return sub._apply(op, [chars])
        return self._apply(op, [chars])

    def center(self, width, fillchar=' '):
        return _pad_op('center', self, width, fillchar)

    def ljust(self, width, fillchar=' '):
        return _pad_op('ljust', self, width, fillchar)

    def rjust(self, width, fillchar=' '):
        return _pad_op('rjust', self, width, fillchar)

    def zfill(self, width):
        return _pad_op('zfill', self, width)

    def expandtabs(self, tabsize=8):
        # Python 层实现（Rust 对含 \0 元素的处理与 numpy 不一致）。
        vals = _decode(self._vals())
        out = [v.expandtabs(tabsize) for v in vals]
        try:
            wb = self._inst.dtype.kind == 'S'
        except Exception:
            wb = False
        return self._pack(out, _shape_of(self._inst), was_bytes=wb)

    def replace(self, old, new, count=None):
        return replace(self, old, new, count)

    def _find_all(self, op, sub, start, end):
        # 统一 Python 层：支持标量/数组 sub 与 start/end 边界。
        vals = _decode(self._vals())
        if isinstance(sub, (str, bytes)):
            s_ = (sub.decode('utf-8', 'surrogateescape')
                  if isinstance(sub, bytes) else sub)
            subs = [s_]
        else:
            subs = _decode(_get_strings(sub))
        out = [getattr(v, op)(subs[i % builtins.len(subs)], start, end)
               for i, v in builtins.enumerate(vals)]
        return self._pack(out, _shape_of(self._inst), as_int=True)

    def find(self, sub, start=0, end=None):
        return self._find_all('find', sub, start, end)

    def rfind(self, sub, start=0, end=None):
        return self._find_all('rfind', sub, start, end)

    def index(self, sub, start=0, end=None):
        r = self.find(sub, start, end)
        flat = _flatten(builtins.list(r.tolist()), [])
        if builtins.any(v == -1 for v in flat):
            raise ValueError('substring not found')
        return r

    def rindex(self, sub, start=0, end=None):
        r = self.rfind(sub, start, end)
        flat = _flatten(builtins.list(r.tolist()), [])
        if builtins.any(v == -1 for v in flat):
            raise ValueError('substring not found')
        return r

    def count(self, sub, start=0, end=None):
        return self._find_all('count', sub, start, end)

    def _prefix_py(self, op, prefix, start, end):
        for s_ in (start, end):
            if s_ is not None and not isinstance(s_, (int,)):
                raise TypeError(
                    op + ': start/end must be integers or None')
        if start is None:
            start = 0
        if start != 0 or end is not None:
            vals = _decode(self._vals())
            if isinstance(prefix, (str, bytes)):
                p = (prefix.decode('utf-8', 'surrogateescape')
                     if isinstance(prefix, bytes) else prefix)
                out = [getattr(v[start:end], op)(p) for v in vals]
            else:
                ps = _decode(_get_strings(prefix))
                out = [getattr(v[start:end], op)(ps[i % builtins.len(ps)])
                       for i, v in builtins.enumerate(vals)]
            return self._pack(out, _shape_of(self._inst), as_bool=True)
        return self._apply(op, [prefix])

    def startswith(self, prefix, start=None, end=None):
        return self._prefix_py('startswith', prefix, start, end)

    def endswith(self, suffix, start=None, end=None):
        return self._prefix_py('endswith', suffix, start, end)

    def isalnum(self):
        return self._apply('isalnum', [])

    def isalpha(self):
        return self._apply('isalpha', [])

    def isdecimal(self):
        try:
            if self._inst.dtype.kind == 'S':
                raise TypeError(
                    'isdecimal() not supported for byte-string arrays')
        except AttributeError:
            pass
        return self._apply('isdecimal', [])

    def isdigit(self):
        return self._apply('isdigit', [])

    def islower(self):
        return self._apply('islower', [])

    def isnumeric(self):
        try:
            if self._inst.dtype.kind == 'S':
                raise TypeError(
                    'isnumeric() not supported for byte-string arrays')
        except AttributeError:
            pass
        return self._apply('isnumeric', [])

    def isspace(self):
        return self._apply('isspace', [])

    def istitle(self):
        return self._apply('istitle', [])

    def isupper(self):
        return self._apply('isupper', [])

    def join(self, seq):
        return join(self, seq)

    def mod(self, values):
        return mod(self, values)

    def multiply(self, i):
        return multiply(self, i)

    def split(self, sep=None, maxsplit=None):
        return split(self, sep, maxsplit)

    def rsplit(self, sep=None, maxsplit=None):
        return rsplit(self, sep, maxsplit)

    def splitlines(self, keepends=False):
        return splitlines(self, keepends)

    def decode(self, encoding='utf-8', errors='strict'):
        return decode(self, encoding, errors)

    def encode(self, encoding='utf-8', errors='strict'):
        return encode(self, encoding, errors)

    def partition(self, sep):
        return partition(self, sep)

    def rpartition(self, sep):
        return rpartition(self, sep)

    def __len__(self):
        return builtins.len(self._inst)

    def __iter__(self):
        return iter(self._inst)

    def __getitem__(self, key):
        inst = self._inst
        shape = _shape_of(inst)
        ndim = builtins.len(shape)

        def _trim_tuple(v):
            # numpy S/U 元组索引 getitem：C 字符串截断（含 \x00）+
            # 尾部空白去除（在 \x00 处停止）。
            if isinstance(v, str):
                parts = v.split('\x00', 1)
                s = parts[0] + '\x00' if builtins.len(parts) > 1 else parts[0]
                return s.rstrip()
            if isinstance(v, bytes):
                parts = v.split(b'\x00', 1)
                s = parts[0] + b'\x00' if builtins.len(parts) > 1 else parts[0]
                return s.rstrip()
            return v

        if isinstance(key, (int,)) and not isinstance(key, bool):
            if ndim > 1:
                # numpy：n 维(n>1)整数索引 → (n-1) 维子数组。
                # rsnumpy 的 ndarray [int] 返回标量——此处重建行。
                flat = self._vals()
                stride = 1
                for d in shape[1:]:
                    stride *= d
                k = key % shape[0]
                row = flat[k * stride:(k + 1) * stride]
                kind = getattr(getattr(inst, 'dtype', None), 'kind', 'U')
                if kind not in ('S', 'U'):
                    kind = 'U'
                from .__init__ import array as _array
                sub = _array(_nest(row, shape[1:]), dtype=kind)
                return _as_chararray(sub)
            return inst[key]
        if (isinstance(key, tuple)
                and builtins.all(
                    isinstance(k, (int,)) and not isinstance(k, bool)
                    for k in key)):
            # S/U 元组整数 getitem 可能返回占位 0-d 数组——从 _vals 取标量。
            flat = self._vals()
            if builtins.len(key) == ndim:
                idx = 0
                for k, d in builtins.zip(key, shape):
                    idx = idx * d + (k % d)
                if idx < builtins.len(flat):
                    return _trim_tuple(flat[idx])
            return _trim_tuple(inst[key])
        # S/U 一维切片：从 _vals 重建（Rust 切片路径可能返回占位值）。
        _k1 = getattr(getattr(inst, 'dtype', None), 'kind', '')
        if isinstance(key, slice) and ndim == 1 and _k1 in ('S', 'U'):
            flat = self._vals()
            sel = flat[key]
            from .__init__ import array as _array
            sub = _array(sel, dtype=_k1)
            sub.base = self
            return sub
        # 全切片：从 _vals 重建 S/U 原始数据（Rust 切片路径返回占位值）。
        is_full = False
        if isinstance(key, slice):
            # 任意维度的 arr[:]（省略维度视为整维）都是全切片。
            is_full = (key.start is None and key.stop is None
                       and key.step in (None, 1))
        elif isinstance(key, tuple):
            is_full = builtins.all(
                isinstance(k, slice) and k.start is None and k.stop is None
                and k.step in (None, 1) for k in key)
        if is_full:
            kind = getattr(getattr(inst, 'dtype', None), 'kind', 'U')
            if kind in ('S', 'U'):
                flat = self._vals()
                from .__init__ import array as _array
                sub = _array(_nest(flat, shape), dtype=kind)
                sub.base = self
                return sub
        v = inst[key]
        if hasattr(v, 'base'):
            try:
                v.base = self
            except Exception:
                pass
        return v

    def __setitem__(self, key, value):
        try:
            kind = self._inst.dtype.kind
        except Exception:
            kind = 'U'
        if kind in 'SU':
            shape = _shape_of(self._inst)
            ndim = builtins.len(shape)
            idxs = None
            if isinstance(key, (int,)) and not isinstance(key, bool):
                if ndim == 1:
                    idxs = [key % shape[0]]
                elif ndim > 1:
                    stride = 1
                    for d in shape[1:]:
                        stride *= d
                    k = key % shape[0]
                    idxs = list(builtins.range(k * stride, (k + 1) * stride))
            elif (isinstance(key, tuple)
                    and builtins.len(key) == ndim
                    and builtins.all(
                        isinstance(k, (int,)) and not isinstance(k, bool)
                        for k in key)):
                idx = 0
                for k, d in builtins.zip(key, shape):
                    idx = idx * d + (k % d)
                idxs = [idx]
            if idxs is not None:
                flat = self._vals()
                for i in idxs:
                    if i < builtins.len(flat):
                        if kind == 'S':
                            flat[i] = (value if isinstance(value, bytes)
                                       else str(value).encode(
                                           'utf-8', 'surrogateescape'))
                        else:
                            flat[i] = (value if isinstance(value, str)
                                       else (value.decode(
                                           'utf-8', 'surrogateescape')
                                             if isinstance(
                                                 value, (bytes, bytearray))
                                             else str(value)))
                self._inst._raw_data = list(flat)
                sd = getattr(self._inst, '_str_dtype', None)
                w = builtins.max((builtins.len(v) for v in flat), default=0)
                if sd is not None:
                    self._inst._str_dtype = (
                        sd[0], builtins.max(sd[1], w))
                return
        self._inst[key] = value

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(self, other)

    def __mod__(self, other):
        return mod(self, other)

    def _cmp(self, other, op):
        # numpy chararray 语义：比较前对两侧值 strip 尾部空格
        # （'123' == '123  ' 为 True，'123' < '123  ' 为 False）
        a_vals = _decode(self._vals())
        if hasattr(other, '_vals'):
            b_vals = _decode(other._vals())
        elif hasattr(other, 'tolist'):
            b_vals = _decode(_flatten(other.tolist(), []))
        else:
            b_vals = [_as_str_arg(other)] * builtins.len(a_vals)
        ops = {'equal': lambda x, y: x == y,
               'not_equal': lambda x, y: x != y,
               'greater': lambda x, y: x > y,
               'less': lambda x, y: x < y,
               'greater_equal': lambda x, y: x >= y,
               'less_equal': lambda x, y: x <= y}
        _fn = ops.get(op)
        if _fn is None:
            return _bool_op(op, self, other=other)
        shape = _shape_of(self._inst)
        a_t = [v.rstrip(' ') if isinstance(v, str) else v for v in a_vals]
        b_t = [v.rstrip(' ') if isinstance(v, str) else v for v in b_vals]
        n = builtins.len(a_t)
        if builtins.len(b_t) == 1 and n != 1:
            b_t = b_t * n
        res = [_fn(x, y) for x, y in builtins.zip(a_t, b_t)]
        np = _np_mod()
        out = np.array(res, dtype='bool')
        return out.reshape(shape) if shape else out

    def __eq__(self, other):
        return self._cmp(other, 'equal')

    def __ne__(self, other):
        return self._cmp(other, 'not_equal')

    def __gt__(self, other):
        return self._cmp(other, 'greater')

    def __lt__(self, other):
        return self._cmp(other, 'less')

    def __ge__(self, other):
        return self._cmp(other, 'greater_equal')

    def __le__(self, other):
        return self._cmp(other, 'less_equal')

    def __bool__(self):
        v = self._vals()
        return builtins.bool(v[0]) if builtins.len(v) == 1 else builtins.bool(self._inst.size)


def _as_chararray(inst):
    """把 ndarray 实例包装为 chararray（组合代理）。"""
    wrapper = object.__new__(chararray)
    wrapper._inst = inst
    return wrapper


def array(obj, itemsize=None, copy=True, unicode=None, order=None):
    """创建 chararray（np.char.array）。"""
    return chararray(obj, itemsize=itemsize, copy=copy, unicode=unicode,
                     order=order)


def asarray(obj, itemsize=None, unicode=None):
    """创建 chararray（np.char.asarray，视图语义——共享数据）。"""
    return chararray(obj, itemsize=itemsize, copy=False, unicode=unicode)

"""numpy._core.multiarray 兼容实现（C-helper 的 Python 版）。

提供 ``_vec_string``（字符串方法向量化）、``scalar``（pickle 协议
构造标量）以及结构化数组二进制 pack/unpack 辅助。
"""

import struct

__all__ = ['_vec_string', 'scalar', '_get_madvise_hugepage']


def _rs():
    import rsnumpy
    return rsnumpy


def _flat_values(arr):
    vals = []

    def rec(x):
        if isinstance(x, (list, tuple)):
            for e in x:
                rec(e)
        else:
            vals.append(x)

    rec(arr.tolist())
    return vals


def _is_string_array(arr):
    rs = _rs()
    return isinstance(arr, rs.ndarray) and arr.dtype.kind in ('S', 'U')


def _vec_string(a, descr, func, args=()):
    """对字符串数组逐元素应用 str/bytes 方法（numpy ``_vec_string`` 兼容）。

    校验顺序与 numpy 一致：args 类型 → 输入类型 → 广播 →
    结果 dtype → 方法存在 → 逐元素应用。
    """
    rs = _rs()
    if args is None:
        args = ()
    if not isinstance(args, tuple):
        raise TypeError("The fourth argument must be a tuple")
    arr = a if isinstance(a, rs.ndarray) else rs.asarray(a)
    if not _is_string_array(arr):
        raise TypeError(f"input {a!r} is not a string array")
    # 广播检查：数组型参数必须能广播到 arr.shape
    arraylike = [x for x in args if isinstance(x, (list, tuple, rs.ndarray))]
    if arraylike:
        try:
            rs.broadcast(arr, *[rs.asarray(x) for x in arraylike])
        except Exception:
            raise ValueError("could not broadcast input array") from None
    try:
        dt = rs.dtype(descr)
    except Exception:
        raise TypeError(f"invalid type description {descr!r}") from None
    if dt.kind not in ('S', 'U'):
        raise TypeError(f"invalid result type {descr!r}")
    sample = '' if dt.kind == 'U' else b''
    if not hasattr(sample, func):
        raise AttributeError(f"string has no function {func!r}")
    flat_in = _flat_values(arr)
    n = len(flat_in)
    flat_args = []
    for x in args:
        if isinstance(x, (list, tuple, rs.ndarray)):
            ax = rs.asarray(x)
            if ax.ndim == 0:
                flat_args.append([ax.item()] * n)
            else:
                flat_args.append(
                    _flat_values(rs.broadcast_to(ax, arr.shape)))
        else:
            flat_args.append([x] * n)
    out = []
    for i, v in enumerate(flat_in):
        call_args = [fa[i] for fa in flat_args]
        out.append(getattr(v, func)(*call_args))
    if dt.kind == 'U':
        out = [v if isinstance(v, str) else str(v) for v in out]
        res = rs.array(out, dtype='U')
    else:
        out = [v if isinstance(v, (bytes, bytearray))
               else str(v).encode('utf-8') for v in out]
        res = rs.array(out, dtype='S')
    if arr.shape:
        res = res.reshape(arr.shape)
    return res


def scalar(dtype, obj):
    """从 (dtype, value) 构造标量（numpy ``core.multiarray.scalar`` 兼容）。

    - int 值按 numpy C 语义视为内存地址：rsnumpy 无地址协议，恒抛
      RuntimeError（pickle 兼容性测试依赖此行为）。
    - 结构化 dtype 接受字段值元组，返回记录标量（_Record）。
    - 纯 object dtype 不接受元组/列表（numpy 抛 TypeError）。
    """
    rs = _rs()
    dt = rs.dtype(dtype)
    if isinstance(obj, (int, rs.integer)) and not isinstance(obj, bool):
        raise RuntimeError(f"invalid address {obj} for scalar of dtype {dt}")
    names = getattr(dt, 'names', None)
    if names is not None:
        if not isinstance(obj, tuple) or len(obj) != len(names):
            raise RuntimeError(
                f"invalid value {obj!r} for structured dtype {dt}")
        return rs._Record(list(obj), dt)
    if dt.kind == 'O':
        if isinstance(obj, (tuple, list)):
            raise TypeError(
                f"cannot create object scalar from {type(obj).__name__}")
        return rs.object_(obj)
    try:
        return dt.type(obj)
    except Exception:
        raise RuntimeError(
            f"cannot create scalar of dtype {dt} from {obj!r}") from None


def _get_madvise_hugepage(*args, **kwargs):
    """rsnumpy 无 hugepage 支持。"""
    return False


# ---------------------------------------------------------------------------
# 结构化数组二进制 IO（ndarray.tofile / rec.fromfile 共用）
# ---------------------------------------------------------------------------

_STRUCT_CODES = (
    ('f', 8, 'd'), ('f', 4, 'f'),
    ('i', 8, 'q'), ('i', 4, 'i'), ('i', 2, 'h'), ('i', 1, 'b'),
    ('u', 8, 'Q'), ('u', 4, 'I'), ('u', 2, 'H'), ('u', 1, 'B'),
    ('c', 16, 'dd'), ('c', 8, 'ff'),
)
_INT_CODES = ('q', 'i', 'h', 'b', 'Q', 'I', 'H', 'B')


def _field_dtype(dt, name):
    if hasattr(dt, '_fields'):
        return dt._fields[name][0]
    return dt.fields[name][0]


def _field_code(fdt):
    for kind, size, code in _STRUCT_CODES:
        if fdt.kind == kind and fdt.itemsize == size:
            return code
    if fdt.kind == 'S':
        return f"{fdt.itemsize}s"
    if fdt.kind == 'U':
        return f"{fdt.itemsize * 4}s"
    raise TypeError(f"unsupported dtype for binary IO: {fdt}")


def _pack_record(values, dt, prefix):
    out = bytearray()
    for name, v in zip(dt.names, values):
        fdt = _field_dtype(dt, name)
        if getattr(fdt, 'names', None) is not None:
            out += _pack_record(v, fdt, prefix)
            continue
        code = _field_code(fdt)
        if code.endswith('s'):
            n = int(code[:-1])
            if isinstance(v, str):
                b = v.encode('utf-8')
            elif isinstance(v, (bytes, bytearray)):
                b = bytes(v)
            else:
                b = str(v).encode('utf-8')
            out += b[:n].ljust(n, b'\x00')
        elif code in ('dd', 'ff'):
            out += struct.pack(prefix + code,
                               v.real if isinstance(v, complex) else float(v),
                               v.imag if isinstance(v, complex) else 0.0)
        elif code in ('d', 'f'):
            out += struct.pack(prefix + code, float(v))
        elif code in _INT_CODES:
            out += struct.pack(prefix + code, int(v))
    return bytes(out)


def _unpack_record(chunk, dt, prefix, off):
    vals = []
    for name in dt.names:
        fdt = _field_dtype(dt, name)
        if getattr(fdt, 'names', None) is not None:
            v, off = _unpack_record(chunk, fdt, prefix, off)
            vals.append(v)
            continue
        code = _field_code(fdt)
        if code.endswith('s'):
            n = int(code[:-1])
            raw = bytes(chunk[off:off + n])
            off += n
            if fdt.kind == 'U':
                vals.append(raw.decode('utf-8', 'replace'))
            else:
                vals.append(raw)
        elif code in ('dd', 'ff'):
            a, b = struct.unpack_from(prefix + code, chunk, off)
            off += fdt.itemsize
            vals.append(complex(a, b))
        elif code in ('d', 'f'):
            (v,) = struct.unpack_from(prefix + code, chunk, off)
            off += fdt.itemsize
            vals.append(v)
        else:
            (v,) = struct.unpack_from(prefix + code, chunk, off)
            off += fdt.itemsize
            vals.append(v)
    return tuple(vals), off


def _byteorder_prefix(byteorder):
    if byteorder is None:
        return '<'
    if byteorder in ('big', '>', 'BE'):
        return '>'
    if byteorder in ('little', '<', 'LE'):
        return '<'
    if byteorder in ('=', '@', 'native'):
        return '='
    return byteorder


def _extract_records(arr):
    """逐记录取值：剥掉外层 ndim 层嵌套，返回记录字段值列表（不拍平字段）。"""
    data = arr.tolist()
    if arr.ndim == 0:
        return [data]
    out = []

    def rec(x, d):
        if d > 0 and isinstance(x, (list, tuple)):
            for e in x:
                rec(e, d - 1)
        else:
            out.append(x)

    rec(data, arr.ndim)
    return out


def pack_array_bytes(arr, byteorder=None):
    """数组 → 原始字节（结构化逐记录 pack；简单数值走 tobytes）。"""
    dt = arr.dtype
    prefix = _byteorder_prefix(byteorder)
    if getattr(dt, 'names', None) is not None:
        return b''.join(_pack_record(r, dt, prefix)
                        for r in _extract_records(arr))
    if dt.kind in 'fiuc' and prefix in ('<', '='):
        try:
            return arr.tobytes()
        except Exception:
            pass
    flat = _flat_values(arr)
    code = _field_code(dt)
    parts = []
    for v in flat:
        if code in ('dd', 'ff'):
            parts.append(struct.pack(prefix + code,
                                     v.real if isinstance(v, complex)
                                     else float(v),
                                     v.imag if isinstance(v, complex)
                                     else 0.0))
        elif code in ('d', 'f'):
            parts.append(struct.pack(prefix + code, float(v)))
        elif code in _INT_CODES:
            parts.append(struct.pack(prefix + code, int(v)))
        else:
            raise TypeError(f"unsupported dtype for binary IO: {dt}")
    return b''.join(parts)


def unpack_array_bytes(data, dt, n, byteorder=None):
    """原始字节 → 记录元组列表（与 pack_array_bytes 互逆）。"""
    prefix = _byteorder_prefix(byteorder)
    if getattr(dt, 'names', None) is None:
        code = _field_code(dt)
        vals = []
        for k in range(n):
            chunk = data[k * dt.itemsize:(k + 1) * dt.itemsize]
            if code in ('dd', 'ff'):
                a, b = struct.unpack_from(prefix + code, chunk, 0)
                vals.append(complex(a, b))
            elif code in ('d', 'f'):
                (v,) = struct.unpack_from(prefix + code, chunk, 0)
                vals.append(v)
            else:
                (v,) = struct.unpack_from(prefix + code, chunk, 0)
                vals.append(v)
        return vals
    recs = []
    for k in range(n):
        chunk = data[k * dt.itemsize:(k + 1) * dt.itemsize]
        rec, _ = _unpack_record(chunk, dt, prefix, 0)
        recs.append(rec)
    return recs


def dot(a, b):
    """ndarray 点积（numpy._core.multiarray.dot 兼容）。"""
    import rsnumpy
    return rsnumpy.dot(a, b)


def _get_ndarray_c_version(*args, **kwargs):
    """C 层 ndarray 版本号桩：rsnumpy 无对应 C 实现（相关测试经 conftest skip）。"""
    raise NotImplementedError(
        "rsnumpy has no C _get_ndarray_c_version; test is expected to be skipped")

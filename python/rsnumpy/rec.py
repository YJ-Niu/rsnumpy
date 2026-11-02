"""记录数组（numpy.rec 兼容层）。

NumPy 中 ``numpy.rec`` 是 ``numpy._core.records`` 的 star-import；
rsnumpy 在本模块直接提供同名集合：array / fromrecords / fromarrays /
fromstring / fromfile / find_duplicate / recarray。
"""

from collections import Counter

import rsnumpy as np


def __getattr__(name):
    if name == 'recarray':
        return np.recarray
    if name == 'record':
        return np.record
    raise AttributeError("module 'rsnumpy.rec' has no attribute %r" % name)


def _resolve_rec_dtype(dtype, formats, names):
    """formats/names → dtype 规格（numpy rec 函数共用）。"""
    if dtype is not None:
        return dtype
    if formats is None:
        return None
    if isinstance(formats, str):
        return formats
    if names is not None:
        if isinstance(names, str):
            names = [n.strip() for n in names.split(',')]
        return list(zip(names, formats))
    return formats


def _read_bytes(file):
    if hasattr(file, 'read'):
        return file.read()
    with open(file, 'rb') as f:
        return f.read()


def _infer_scalar_dtype(v):
    """从标量值推断类型（numpy rec.array 推断语义）。"""
    if isinstance(v, bool):
        return '?'
    if isinstance(v, int):
        return 'i8'
    if isinstance(v, float):
        return 'f8'
    if isinstance(v, complex):
        return 'c16'
    if isinstance(v, str):
        return 'U' + str(max(1, len(v)))
    if isinstance(v, (bytes, bytearray)):
        return 'S' + str(max(1, len(v)))
    return 'f8'


def _infer_struct_dtype(obj, names):
    """从数据与字段名推断结构化 dtype（numpy rec.array 语义：仅给 names、
    无 formats 时按首条记录推断各字段类型）。"""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(',')]
    try:
        first = obj[0]
    except Exception:
        first = obj
    if isinstance(first, (list, tuple)):
        vals = list(first)
    else:
        vals = [first] * len(names)
    fields = []
    for i, n in enumerate(names):
        v = vals[i] if i < len(vals) else 0
        fields.append((n, _infer_scalar_dtype(v)))
    return fields


def array(obj, dtype=None, shape=None, formats=None, names=None, **kwargs):
    """从数据构造一个 recarray（支持属性方式访问字段）。

    对象参数为文件对象或原始 bytes（配合 formats/shape/byteorder）时
    走二进制读取路径（numpy rec.array 兼容）。
    """
    dt = _resolve_rec_dtype(dtype, formats, names)
    if hasattr(obj, 'read'):
        return _as_recarray(fromfile(obj, dtypes=dt, shape=shape,
                                     **kwargs))
    if isinstance(obj, (bytes, bytearray)) and dt is not None:
        return _as_recarray(fromstring(bytes(obj), dtypes=dt, shape=shape,
                                       **kwargs))
    # numpy 语义：仅给 names（无 dtype/formats）→ 从数据推断各字段类型构造
    # 结构化 dtype（否则 np.array 会推断成浮点数组、丢失结构）
    if dt is None and names is not None:
        dt = _infer_struct_dtype(obj, names)
    a = np.array(obj, dtype=dt)
    return np._as_recarray(a)


def fromrecords(recList, dtype=None, shape=None, formats=None, names=None,
                **kwargs):
    """从记录列表构造 recarray。"""
    return array(recList, dtype=dtype, shape=shape, formats=formats,
                 names=names, **kwargs)


def fromarrays(itemsets, dtypes=None, names=None, shape=None, **kwargs):
    """从字段数组列表构造 recarray（数据拷贝，numpy rec.fromarrays 兼容）。"""
    rs = np
    # dtype 关键字（numpy 兼容）：单个结构化 dtype，取其字段名与各字段类型
    dt_kw = kwargs.get('dtype', None)
    if names is not None and isinstance(names, str):
        names = [n.strip() for n in names.split(',')]
    if dt_kw is not None:
        _d = rs.dtype(dt_kw)
        names = list(_d.names) if _d.names \
            else ['f%d' % i for i in range(len(itemsets))]
        dtypes = [_d.fields[n][0] for n in names]
        dt = _d
    elif dtypes is None:
        if names is None:
            names = ['f%d' % i for i in range(len(itemsets))]
        dtypes = []
        for a in itemsets:
            arr = a if isinstance(a, rs.ndarray) else rs.asarray(a)
            dtypes.append(arr.dtype)
        dt = rs.dtype(list(zip(names, dtypes)))
    else:
        dtypes = list(dtypes)
        if names is None:
            names = ['f%d' % i for i in range(len(dtypes))]
        dt = rs.dtype(list(zip(names, dtypes)))
    n = len(itemsets[0])
    recs = []
    for i in range(n):
        recs.append(tuple(itemsets[k][i] for k in range(len(itemsets))))
    a = rs.array(recs, dtype=dt)
    return rs._as_recarray(a)


def fromstring(string, dtypes=None, shape=None, offset=0, sep='',
               formats=None, names=None, byteorder=None, **kwargs):
    """从 bytes 字符串读取数组（numpy rec.fromstring 兼容）。

    object dtype（引用）不支持，抛 ValueError。
    """
    from rsnumpy._core import multiarray
    rs = np
    dt = _resolve_rec_dtype(dtypes, formats, names)
    if dt is None:
        dt = 'f8'
    dt = rs.dtype(dt)
    if dt.kind == 'O':
        raise ValueError("Arrays containing references are not supported")
    data = bytes(string)
    if sep:
        text = data.decode('ascii', 'replace')
        vals = [float(v) for v in text.replace('\n', sep).split(sep)
                if v.strip()]
        a = rs.array(vals, dtype=dt)
        return a.reshape(shape) if shape else a
    data = data[offset:]
    n = shape if shape is not None else len(data) // dt.itemsize
    vals = multiarray.unpack_array_bytes(data, dt, n, byteorder)
    return rs.array(vals, dtype=dt)


def fromfile(file, dtypes=None, shape=None, offset=0, sep='',
             formats=None, names=None, byteorder=None, **kwargs):
    """从文件读取数组（numpy rec.fromfile 兼容）。

    object dtype（引用）不支持，抛 ValueError。
    """
    from rsnumpy._core import multiarray
    rs = np
    dt = _resolve_rec_dtype(dtypes, formats, names)
    if dt is None:
        dt = 'f8'
    dt = rs.dtype(dt)
    if dt.kind == 'O':
        raise ValueError("Arrays containing references are not supported")
    if sep:
        data = _read_bytes(file)
        text = data.decode('ascii', 'replace')
        vals = [float(v) for v in text.replace('\n', sep).split(sep)
                if v.strip()]
        a = rs.array(vals, dtype=dt)
        return a.reshape(shape) if shape else a
    data = _read_bytes(file)
    data = data[offset:]
    n = shape if shape is not None else len(data) // dt.itemsize
    vals = multiarray.unpack_array_bytes(data, dt, n, byteorder)
    return rs.array(vals, dtype=dt)


def _as_recarray(a):
    return np._as_recarray(a)


def find_duplicate(lst):
    """Find duplication in a list, return a list of duplicated elements."""
    return [item for item, counts in Counter(lst).items() if counts > 1]

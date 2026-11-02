"""掩码记录数组（numpy.ma.mrecords 兼容实现）。

MaskedRecords：基于结构化（记录）数据的掩码数组；逐记录访问返回
带字段接口的记录对象。
"""

import rsnumpy as np
from .core import MaskedArray, asarray, getmaskarray, nomask


class _MaskedRecord(tuple):
    """带字段访问的记录（numpy.ma.mrecords 记录语义）。"""

    def __new__(cls, values, dtype_obj, mask=None, parent=None, index=0):
        obj = tuple.__new__(cls, tuple(values))
        obj._values = list(values)
        obj._dtype_obj = dtype_obj
        obj._mask = list(mask) if mask is not None else [False] * len(values)
        obj._parent = parent
        obj._index = index
        return obj

    def __getattr__(self, name):
        dt = self.__dict__.get('_dtype_obj')
        names = getattr(dt, '_names', None) if dt is not None else None
        if names and name in names:
            i = names.index(name)
            if self.__dict__.get('_mask', [False] * len(names))[i]:
                return np.ma.masked
            return self._values[i]
        raise AttributeError(name)

    @property
    def dtype(self):
        return self._dtype_obj


class MaskedRecords(MaskedArray):
    """记录型掩码数组（numpy.ma.mrecords.MaskedRecords 兼容子集）。"""

    def __init__(self, data=None, mask=False, dtype=None, fill_value=None,
                 **kwargs):
        if data is None:
            data = np.array([], dtype=dtype) if dtype is not None else np.array([])
        elif not hasattr(data, '_array'):
            data = np.array(data, dtype=dtype)
        super().__init__(data, mask=mask, dtype=dtype,
                         fill_value=fill_value, **kwargs)

    def __getitem__(self, key):
        dt = self._data.dtype
        if dt.names and isinstance(key, int) and self._data.ndim == 1:
            names = dt.names
            i = key % self._data.shape[0]
            row = self._data[i]
            values = [row[n] for n in names]
            m = self._mask
            rowmask = m[i * len(names):(i + 1) * len(names)]
            return _MaskedRecord(values, dt, mask=rowmask, parent=self,
                                 index=i)
        return super().__getitem__(key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def mrecarray(data=None, dtype=None, shape=None, **kwargs):
    """构造 MaskedRecords（numpy.ma.mrecords.mrecarray 兼容）。"""
    return MaskedRecords(data, mask=kwargs.pop('mask', False),
                         dtype=dtype, fill_value=kwargs.pop('fill_value', None),
                         **kwargs)


def fromarrays(*args, **kwargs):
    """从列数组构造 MaskedRecords（numpy.ma.mrecords.fromarrays 兼容）。

    单 list/tuple 参数按列解包（numpy 语义）；MaskedArray 列取其底层
    数据（rec 层不理解 ma；掩码位置数据保留原值，与 numpy 一致）。
    """
    mask = kwargs.pop('mask', None)
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        _first = args[0][0] if args[0] else None
        if _first is not None and (
                isinstance(_first, MaskedArray)
                or hasattr(_first, '_array') or hasattr(_first, 'dtype')):
            args = tuple(args[0])
    dtype = kwargs.pop('dtype', None)
    cols = []
    for a in args:
        if isinstance(a, MaskedArray):
            cols.append(a.data)
        else:
            cols.append(a)
    if dtype is None:
        # 由列推断字段名/类型（names= 支持）
        names = kwargs.pop('names', None)
        if not cols or not names:
            raise ValueError("fromarrays requires names= when dtype is None")
        dtype = [(n, np.asarray(c).dtype) for n, c in zip(names, cols)]
    recs = np.rec.fromarrays(cols, dtype=dtype, **kwargs)
    return MaskedRecords(recs, mask=mask)


def fromrecords(recs, **kwargs):
    """从记录序列构造 MaskedRecords（numpy.ma.mrecords.fromrecords 兼容）。"""
    arr = np.rec.fromrecords(recs, **kwargs)
    return MaskedRecords(arr)


def fromtextfile(name, **kwargs):
    raise NotImplementedError(
        "rsnumpy.ma.mrecords.fromtextfile is not implemented")


def addfield(a, data, name, dtype=None, newbytes=0):
    """给记录数组追加字段（numpy.ma.mrecords.addfield 兼容）。"""
    a = asarray(a)
    if not np.dtype(a.data.dtype).names:
        raise TypeError("addfield: a must be a record array")
    rec = a.data
    if name in rec.dtype.names:
        raise ValueError(f"field {name!r} already exists")
    vals = np.asarray(data, dtype=dtype)
    newdt = list(rec.dtype.names) + [name]
    # 逐记录重组
    oldnames = rec.dtype.names
    rows = []
    for i in range(rec.shape[0]):
        row = tuple(rec[i][n] for n in oldnames) + (vals[i] if i < vals.size else None,)
        rows.append(row)
    newarr = np.array(rows, dtype=rec.dtype.names and
                       [(n, rec.dtype.fields[n][0]) for n in oldnames] +
                       [(name, vals.dtype)])
    m = list(a._mask)
    extra = [False] * len(oldnames) if data is not None else []
    a._data = newarr
    a._mask = m + [False] * rec.shape[0]
    return a


__all__ = ['MaskedRecords', 'mrecarray', 'fromarrays', 'fromrecords',
           'fromtextfile', 'addfield']

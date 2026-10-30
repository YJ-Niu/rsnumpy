"""掩码数组（numpy.ma 的兼容子集）。"""

import rsnumpy as np


def _flatten(seq):
    out = []
    for x in seq:
        if isinstance(x, (list, tuple)):
            out.extend(_flatten(x))
        else:
            out.append(x)
    return out


def _record_count(data):
    dt = getattr(data, '_dtype_obj', None)
    if dt is not None:
        return len(np._flatten_records(data._raw_data))
    return data.size


class MaskedArray:
    """带掩码的数组：包装底层 ndarray 与逐元素布尔掩码。"""

    def __init__(self, data=None, mask=False, dtype=None, fill_value=None,
                 copy=False, **kwargs):
        _ = copy, kwargs
        if isinstance(data, MaskedArray):
            base_mask = list(data._mask)
            data = data._data
        else:
            base_mask = None
            if not hasattr(data, '_array'):
                data = np.array(data, dtype=dtype) if dtype is not None else np.array(data)
            elif dtype is not None:
                data = data.view(dtype)
        self._data = data
        n = _record_count(data)
        if mask is None or mask is False:
            self._mask = base_mask if base_mask is not None else [False] * n
        elif mask is True:
            self._mask = [True] * n
        else:
            m = mask.tolist() if hasattr(mask, 'tolist') else list(mask)
            self._mask = [bool(x) for x in _flatten(m)]
        if len(self._mask) != n:
            self._mask = (self._mask + [False] * n)[:n]
        self.fill_value = fill_value

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def data(self):
        return self._data

    @property
    def mask(self):
        return np.array(self._mask, dtype='bool').reshape(self.shape)

    @mask.setter
    def mask(self, value):
        if value is None:
            self._mask = [False] * _record_count(self._data)
            return
        mask = np.asarray(value, dtype='bool')
        if mask.ndim == 0:
            self._mask = [bool(mask)] * _record_count(self._data)
            return
        if mask.shape != self.shape:
            raise ValueError("mask and data must have the same shape")
        self._mask = [bool(item) for item in mask.reshape((-1,)).tolist()]

    def view(self, dtype=None, type=None):
        _ = type
        if dtype is None:
            return MaskedArray(self._data, list(self._mask))
        newdata = self._data.view(dtype)
        newn = _record_count(newdata)
        newmask = list(self._mask) if len(self._mask) == newn else [False] * newn
        return MaskedArray(newdata, newmask)

    def filled(self, fill_value=None):
        value = self.fill_value if fill_value is None else fill_value
        if value is None:
            value = 0
        result = self._data.copy()
        result[self.mask] = value
        return result

    def compressed(self):
        return self._data[~self.mask].copy()

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return "masked_array(data=%s,\n             mask=%s)" % (self._data, self._mask)

    def __str__(self):
        return self.__repr__()


def array(data=None, mask=False, dtype=None, fill_value=None, copy=False, **kwargs):
    """构造一个掩码数组。"""
    return MaskedArray(data, mask=mask, dtype=dtype, fill_value=fill_value,
                       copy=copy, **kwargs)


masked_array = MaskedArray


def getdata(a):
    """返回掩码数组的底层数据。"""
    return a._data if isinstance(a, MaskedArray) else a


def getmask(a):
    """numpy.ma.getmask 兼容：非掩码数组返回 nomask。"""
    if isinstance(a, MaskedArray):
        return a._mask
    return nomask


def getmaskarray(a):
    """返回掩码数组的布尔掩码。"""
    if isinstance(a, MaskedArray):
        return a.mask
    return np.zeros(getattr(a, 'shape', (0,)), dtype='bool')


def asarray(a, dtype=None):
    if isinstance(a, MaskedArray):
        if dtype is not None and a.dtype != np.dtype(dtype):
            return MaskedArray(a.data.astype(dtype), a.mask, fill_value=a.fill_value)
        return a
    return MaskedArray(np.asarray(a, dtype=dtype))


def isMaskedArray(a):
    return isinstance(a, MaskedArray)


def masked_values(data, value, rtol=1e-5, atol=1e-8, copy=True):
    data = np.array(data, copy=copy)
    mask = np.abs(data - value) <= atol + rtol * np.abs(value)
    return MaskedArray(data, mask=mask, fill_value=value)


def masked_invalid(data, copy=True):
    data = np.array(data, copy=copy)
    return MaskedArray(data, mask=~np.isfinite(data))


class _MaskedConstant:
    def __repr__(self):
        return 'masked'
    def __str__(self):
        return 'masked'
    def __bool__(self):
        return False


masked = _MaskedConstant()
nomask = False


def zeros(shape, dtype=float, **kwargs):
    """全零掩码数组（numpy.ma.zeros 兼容）。"""
    return MaskedArray(np.zeros(shape, dtype=dtype))


def ones(shape, dtype=float, **kwargs):
    """全一掩码数组（numpy.ma.ones 兼容）。"""
    return MaskedArray(np.ones(shape, dtype=dtype))


def empty(shape, dtype=float, **kwargs):
    """未初始化掩码数组（numpy.ma.empty 兼容）。"""
    return MaskedArray(np.empty(shape, dtype=dtype))


def full(shape, fill_value, dtype=None, **kwargs):
    """定值掩码数组（numpy.ma.full 兼容）。"""
    return MaskedArray(np.full(shape, fill_value, dtype=dtype))


def masked_where(condition, a, **kwargs):
    """按条件生成掩码数组（numpy.ma.masked_where 兼容）。"""
    cond = np.array(condition, dtype=bool) if not hasattr(condition, '_array') else condition
    return MaskedArray(a, mask=cond)

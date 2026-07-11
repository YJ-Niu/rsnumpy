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
    def data(self):
        return self._data

    @property
    def mask(self):
        return np.array(self._mask, dtype='bool')

    def view(self, dtype=None, type=None):
        _ = type
        if dtype is None:
            return MaskedArray(self._data, list(self._mask))
        newdata = self._data.view(dtype)
        newn = _record_count(newdata)
        newmask = list(self._mask) if len(self._mask) == newn else [False] * newn
        return MaskedArray(newdata, newmask)

    def filled(self, fill_value=None):
        _ = fill_value
        return self._data

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


def getmaskarray(a):
    """返回掩码数组的布尔掩码。"""
    if isinstance(a, MaskedArray):
        return a.mask
    return np.zeros(getattr(a, 'shape', (0,)), dtype='bool')

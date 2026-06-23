"""I/O 模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnumpy._core as _core


def _get_ndarray():
    from .__init__ import ndarray as _n
    return _n


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
    return _core.ndarray(a)


def _wrap(result):
    return _get_ndarray()(result)


def save(file, arr):
    """将数组保存为二进制文件 (.npy)。"""
    _core.save_npy(file, _ensure_raw(arr))


class NpzFile:
    """用于表示加载的 .npz 文件的类。"""
    def __init__(self, data):
        self._data = data
    
    @property
    def files(self):
        """返回文件中所有数组的名称列表。"""
        return list(self._data.keys())
    
    def __getitem__(self, key):
        """通过键名获取数组。"""
        return self._data[key]


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    """从二进制文件加载数组 (.npy 或 .npz)。"""
    if file.endswith('.npz'):
        return NpzFile(load_npz(file))
    return _wrap(_core.load_npy(file))


def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
    """将数组保存为文本文件。"""
    _core.save_text(fname, _ensure_raw(X), fmt, delimiter)


def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes',
            max_rows=None, *, like=None):
    """从文本文件加载数据。"""
    if delimiter is None:
        delimiter = ""
    result = _wrap(_core.load_text(fname, delimiter, skiprows))
    if dtype == int:
        result = result.astype("int")
    elif dtype == float:
        pass
    elif isinstance(dtype, type):
        result = result.astype(dtype.__name__)
    else:
        result = result.astype(dtype)
    if len(result) == 1:
        return result[0]
    else:
        return result


def savez(file, *args, **kwds):
    """将多个数组保存为未压缩的 .npz 文件。"""
    arrays = []
    names = []
    for name in sorted(kwds.keys()):
        arrays.append(_ensure_raw(kwds[name]))
        names.append(name)
    for i, arg in enumerate(args):
        arrays.append(_ensure_raw(arg))
        names.append('arr_%d' % i)
    _core.savez_npz(file, arrays, names)


def _is_integer_array(arr):
    """检查数组是否应该是整数类型。"""
    raw_list = arr.tolist()
    
    def check_nested(data):
        if isinstance(data, list):
            return all(check_nested(x) for x in data)
        return abs(data - round(data)) < 1e-10
    
    return check_nested(raw_list)


def load_npz(file):
    """从 .npz 文件加载数组（返回 dict）。"""
    pairs = _core.load_npz(file)
    result = {}
    ndarray_cls = _get_ndarray()
    for key, raw in pairs:
        if _is_integer_array(raw):
            result[key] = ndarray_cls._wrap(raw, _dtype='int64')
        else:
            result[key] = ndarray_cls._wrap(raw, _dtype='float64')
    return result

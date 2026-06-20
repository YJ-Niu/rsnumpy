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


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    """从二进制文件加载数组 (.npy)。"""
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
    return _wrap(_core.load_text(fname, delimiter, skiprows))


def savez(file, *args, **kwds):
    """将多个数组保存为未压缩的 .npz 文件。"""
    arrays = []
    names = []
    for i, arg in enumerate(args):
        arrays.append(_ensure_raw(arg))
        names.append('arr_%d' % i)
    for name, arr in kwds.items():
        arrays.append(_ensure_raw(arr))
        names.append(name)
    _core.savez_npz(file, arrays, names)


def load_npz(file):
    """从 .npz 文件加载数组（返回 dict）。"""
    pairs = _core.load_npz(file)
    result = {}
    for key, raw in pairs:
        result[key] = _wrap(raw)
    return result

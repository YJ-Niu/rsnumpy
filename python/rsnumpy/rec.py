"""记录数组（numpy.rec 的兼容子集）。"""

import rsnumpy as np


def __getattr__(name):
    if name == 'recarray':
        return np.recarray
    raise AttributeError("module 'rsnumpy.rec' has no attribute %r" % name)


def array(obj, dtype=None, shape=None, formats=None, names=None, **kwargs):
    """从数据构造一个 recarray（支持属性方式访问字段）。"""
    _ = shape, kwargs
    if dtype is None and formats is not None:
        if names is not None:
            dtype = list(zip(names, formats))
        else:
            dtype = ','.join(str(f) for f in formats)
    a = np.array(obj, dtype=dtype)
    return np._as_recarray(a)


def fromrecords(recList, dtype=None, shape=None, formats=None, names=None, **kwargs):
    """从记录列表构造 recarray。"""
    return array(recList, dtype=dtype, shape=shape, formats=formats,
                 names=names, **kwargs)

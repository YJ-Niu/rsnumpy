# SPDX-License-Identifier: MIT
"""rsnumpy._core — NumPy 2.x ``numpy._core`` 兼容层。

Rust 核心（rsnumpy.num_core）提供计算；本包提供命名空间与 C-helper
兼容 API：

- ``numerictypes``：标量类型表（sctypes/sctypeDict 等）
- ``function_base``：add_newdoc 文档工具
- ``multiarray``：_vec_string / scalar（pickle 协议）/ 二进制 IO
- ``_multiarray_umath``：MAXDIMS / _discover_array_parameters
- ``_multiarray_tests``：array_indexing 测试辅助
- ``_rational_tests``：测试用 rational 标量
- ``shape_base``：_block_* 内部辅助（skip 语义）
- ``umath``：ufunc 集合门面
- ``flatiter`` / ``ufunc``：类型占位（docstring 契约）
- ``records`` → ``rsnumpy.rec``
"""

from .numeric import normalize_axis_index, normalize_axis_tuple

__all__ = ['normalize_axis_index', 'normalize_axis_tuple',
           'flatiter', 'ufunc', 'sctypes', 'sctypeDict', 'records']

_FLATITER_INDEX_DOC = (
    "Current flat index into the array.\n"
    "\n"
    "Examples\n"
    "--------\n"
    ">>> import numpy as np\n"
    ">>> x = np.arange(6).reshape(2, 3)\n"
    ">>> fl = x.flat\n"
    ">>> fl.index\n"
    "0\n"
    ">>> next(fl)\n"
    "0\n"
    ">>> fl.index\n"
    "1\n"
)


class _DocDescriptor:
    """模拟 C 层 getset 描述符：``__doc__`` 只读（add_newdoc 静默失败）。"""

    def __init__(self, doc, fget):
        object.__setattr__(self, '_doc', doc)
        object.__setattr__(self, '_fget', fget)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self._fget(obj)

    @property
    def __doc__(self):
        return object.__getattribute__(self, '_doc')

    def __setattr__(self, name, value):
        if name == '__doc__':
            raise TypeError("cannot set '__doc__' on C-level descriptor")
        object.__setattr__(self, name, value)


def _flat_values_python(arr):
    vals = []

    def rec(x):
        if isinstance(x, (list, tuple)):
            for e in x:
                rec(e)
        else:
            vals.append(x)

    rec(arr.tolist())
    return vals


def _flatiter_index_get(self):
    return self._index


class flatiter:
    """数组的扁平迭代器（numpy._core.flatiter 兼容）。

    支持索引、切片、迭代与 ``index`` 属性（docstring 契约与 NumPy
    一致，且不可被 add_newdoc 改写——模拟 C 层 getset 行为）。
    """

    def __init__(self, array):
        self._array = array
        self._values = _flat_values_python(array)
        self._index = 0

    index = _DocDescriptor(_FLATITER_INDEX_DOC, _flatiter_index_get)

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._values):
            raise StopIteration
        v = self._values[self._index]
        self._index += 1
        return v

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._values[key]
        n = len(self._values)
        i = key if key >= 0 else n + key
        if i < 0 or i >= n:
            raise IndexError(f"index {key} is out of bounds for flat array "
                             f"of size {n}")
        return self._values[i]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            raise TypeError("flatiter slice assignment is not supported")
        n = len(self._values)
        i = key if key >= 0 else n + key
        if i < 0 or i >= n:
            raise IndexError(f"index {key} is out of bounds for flat array "
                             f"of size {n}")
        self._values[i] = value
        try:
            self._array.flat[key] = value
        except Exception:
            pass


class _UFuncMeta(type):
    """ufunc 元类：isinstance(x, ufunc) 识别 rsnumpy 标记过的 ufunc 函数。

    rsnumpy 的 ufunc 是 Python 层函数对象（带 ``_rs_ufunc`` 标记与
    ``.types`` 循环签名），NumPy 中则是 C 层 ufunc 类的实例。
    """

    def __instancecheck__(cls, inst):
        # 真正的类实例（含子类，如 scicore 兼容层的 _gufunc(np.ufunc) 桩）
        # 按标准 MRO 判定；rsnumpy 标记函数（_rs_ufunc）再兜底识别。
        if type.__instancecheck__(cls, inst):
            return True
        return bool(getattr(inst, '_rs_ufunc', False))


class ufunc(metaclass=_UFuncMeta):
    """ufunc 类型兼容占位（numpy._core.ufunc）。

    rsnumpy 的 ufunc 是 Python 层可调用对象；本类提供与 NumPy 兼容的
    类型占位（identity 常量与文档契约）。
    """

    class identity:
        """ufunc 恒等元（identity）说明。

        NumPy ufunc 通过 ``identity`` 属性暴露其代数恒等元 e（满足
        ``f(e, x) == f(x, e) == x``）：

        - ``identity == -1``（NO_IDENTITY）：ufunc 没有恒等元；此时对
          空序列做 reduce 会抛错，除非显式给出 ``initial``/``start``
          值；``accumulate`` 要求数组非空。
        - ``identity == 0``（ADD）：恒等元为 0，如 ``numpy.add``——
          空 reduce 得到 0。
        - ``identity == 1``（MULTIPLY）：恒等元为 1，如
          ``numpy.multiply``——空 reduce 得到 1。
        - ``identity == 2``（REDUCE）：恒等元仅对 reduce 有意义
          （如 ``numpy.maximum`` 使用可表示的最小值）。

        rsnumpy 的 Python 层 ufunc 镜像这些常量，使按
        ``numpy._core.ufunc.identity`` 编写的代码可解析。
        """

        NO_IDENTITY = -1
        ADD = 0
        MULTIPLY = 1
        REDUCE = 2

    nargs = 2
    nin = 1
    nout = 1


def __getattr__(name):
    if name in ('sctypes', 'sctypeDict'):
        from . import numerictypes
        return getattr(numerictypes, name)
    if name == 'records':
        import rsnumpy
        return rsnumpy.rec
    # 顶层常用名的 _core 级暴露（numpy._core 同样导出这些公共名，
    # 移植测试按 numpy 路径从 _core 导入）
    if name in ('arange', 'array', 'asarray', 'atleast_1d', 'atleast_2d',
                'atleast_3d', 'take', 'all', 'array2string', 'errstate',
                'inf', 'isnan', 'max', 'object_', 'number', 'result_type',
                'float32', 'intp', 'isnat', 'empty', 'array_repr',
                'block', 'concatenate', 'hstack', 'newaxis', 'stack',
                'vstack', 'finfo', 'iinfo', 'swapaxes',
                'isscalar', 'ndarray', 'signbit'):
        # testing/_private/utils.py 按 numpy 路径从 _core 导入这三个公共名
        import rsnumpy
        return getattr(rsnumpy, name)
    # 子模块：惰性导入（`import rsnumpy._core.X` 由导入机制直接解析，
    # 属性访问走这里）
    if name in ('numerictypes', 'function_base', 'multiarray',
                'shape_base', '_multiarray_umath', '_multiarray_tests',
                '_rational_tests', 'umath', 'tests', 'numeric',
                '_exceptions', 'arrayprint', 'fromnumeric', 'records',
                'strings'):
        from importlib import import_module
        return import_module(f'.{name}', __name__)
    raise AttributeError(f"module 'rsnumpy._core' has no attribute {name!r}")

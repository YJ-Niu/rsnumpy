"""rsnumpy - A Rust-powered rsnumpy-compatible array library.

rsnumpy provides a rsnumpy-compatible API implemented in Rust for better
performance and memory efficiency while maintaining full API compatibility.

Examples:
    >>> import rsnumpy as np
    >>> a = np.ndarray([1, 2, 3])
    >>> np.sum(a)
    6
    >>> np.mean(a)
    2.0
"""
# ========== 子模块导入和函数挂载 ==========
# 这些导入放在 ndarray 类和 _ensure/_wrap_result/_scalar 函数定义之后以避免循环导入
import builtins
from . import array_methods
from . import math_functions as _math_functions_module
from . import statistics as _statistics_module
from . import array_ops as _array_ops_module

from .io import save, load, loadtxt, savetxt, savez, load_npz, genfromtxt
from .polynomial import Poly, polyval, polyfit, polyder, polyint, polyroots
from .linalg import linalg_module as _linalg_module
from .random import random_module as _random_module
from . import char as _char_module
from . import matlib as _matlib_module
# ========== rec / ma 子模块（在顶层完全初始化后导入以避免循环依赖） ==========
from . import ma
from . import rec
from . import lib
# 子模块
from . import _core
from . import fft  # noqa: E402  FFT 子模块（numpy 风格 np.fft.*）
from . import typing  # noqa: E402  类型注解子模块（numpy 风格 np.typing.*）
from . import exceptions
import datetime as _datetime
import sys as _sys
from . import _extra as _extra_module
from ._extra import chunked_apply, mmap_array, LazyArray
# ---------- dtype 提升与浮点判定 ----------
import rsnumpy.num_core as np
from rsnumpy.num_core import ndarray_iter as NdArrayIter
from rsnumpy.num_core import (
    set_num_threads,
    get_num_threads,
    get_num_cpus,
    get_parallel_thresholds,
    set_parallel_thresholds,
    parallel_context,
)
from ._dtypes import (
    DType, dtype, _make_flexible, _make_subarray,
    _build_struct, _scalar_typestr_short,
)
_current_module = _sys.modules[__name__]
_sys.modules['rsnumpy'] = _current_module
_sys.modules['rsnumpy.__init__'] = _current_module
_current_module.__name__ = 'rsnumpy'

__version__ = "1.2.9"

# 捕获内建函数别名：_extra 挂载会向本模块 globals 注入同名的 rsnumpy 函数
# （all/any/round），会遮蔽内建函数。以下别名保证本文件内部逻辑始终使用内建实现。
_py_round = round
_py_all = all
_py_any = any
_py_max = max
_py_min = min


def _flatten_vals(x):
    """将嵌套列表/元组扁平化为单层列表（标量原样返回单元素列表）。"""
    out = []
    if isinstance(x, (list, tuple)):
        for v in x:
            out.extend(_flatten_vals(v))
    else:
        out.append(x)
    return out


class ArrayFlags:
    """数组内存布局信息，与 rsnumpy 的 np.ndarray.flags 兼容。"""

    def __init__(self, arr):
        self._arr = arr

    @property
    def C_CONTIGUOUS(self):
        return not getattr(self._arr, '_rs_strided', False)

    @property
    def F_CONTIGUOUS(self):
        # Compute from the real shape/byte-strides so a C-order array is
        # (correctly) reported as NOT F-contiguous. The previous stub returned
        # the same value as C_CONTIGUOUS, which made scipy's sparse ``toarray``
        # take the Fortran scatter path on C-order buffers and scramble output.
        shape = getattr(self._arr, 'shape', ())
        strides = getattr(self._arr, 'strides', None)
        itemsize = getattr(self._arr, 'itemsize', 8)
        ndim = len(shape)
        if ndim == 0:
            return True
        if strides is None:
            # No stride info: rsnumpy arrays are C-order, so F-contiguous only
            # when at most one dimension is non-trivial (0-d / 1-d / 1xN / Nx1).
            return sum(1 for d in shape if d > 1) <= 1
        expected = 1
        for i in range(ndim):
            if shape[i] != 1 and strides[i] != expected * itemsize:
                return False
            expected *= shape[i]
        return True

    @property
    def OWNDATA(self):
        return True

    @property
    def WRITEABLE(self):
        return getattr(self._arr, '_rs_writeable', True)

    @property
    def ALIGNED(self):
        return not getattr(self._arr, '_rs_unaligned', False)

    @property
    def WRITEBACKIFCOPY(self):
        return False

    @property
    def UPDATEIFCOPY(self):
        return False

    def __eq__(self, other):
        if not isinstance(other, ArrayFlags):
            return NotImplemented
        return (self.C_CONTIGUOUS == other.C_CONTIGUOUS and
                self.F_CONTIGUOUS == other.F_CONTIGUOUS and
                self.OWNDATA == other.OWNDATA and
                self.WRITEABLE == other.WRITEABLE and
                self.ALIGNED == other.ALIGNED and
                self.WRITEBACKIFCOPY == other.WRITEBACKIFCOPY and
                self.UPDATEIFCOPY == other.UPDATEIFCOPY)

    # numpy lowercase aliases
    c_contiguous = property(lambda self: self.C_CONTIGUOUS)
    f_contiguous = property(lambda self: self.F_CONTIGUOUS)
    aligned = property(lambda self: self.ALIGNED)
    own_data = property(lambda self: self.OWNDATA)

    @property
    def writeable(self):
        return self.WRITEABLE

    @writeable.setter
    def writeable(self, v):
        try:
            self._arr._rs_writeable = builtins.bool(v)
        except Exception:
            pass

    def __getitem__(self, key):
        """numpy 兼容：flags['C'] / flags['W'] / flags['writeable'] 等。"""
        mapping = {
            'C': self.C_CONTIGUOUS, 'F': self.F_CONTIGUOUS, 'O': self.OWNDATA,
            'W': self.WRITEABLE, 'A': self.ALIGNED,
            'c': self.C_CONTIGUOUS, 'f': self.F_CONTIGUOUS, 'w': self.WRITEABLE,
            'a': self.ALIGNED, 'o': self.OWNDATA,
            'c_contiguous': self.C_CONTIGUOUS, 'f_contiguous': self.F_CONTIGUOUS,
            'writeable': self.WRITEABLE, 'aligned': self.ALIGNED,
            'owndata': self.OWNDATA,
            'C_CONTIGUOUS': self.C_CONTIGUOUS, 'F_CONTIGUOUS': self.F_CONTIGUOUS,
            'OWNDATA': self.OWNDATA, 'WRITEABLE': self.WRITEABLE,
            'ALIGNED': self.ALIGNED, 'WRITEBACKIFCOPY': self.WRITEBACKIFCOPY,
            'UPDATEIFCOPY': self.UPDATEIFCOPY,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        lines = [
            f"  C_CONTIGUOUS : {self.C_CONTIGUOUS}",
            f"  F_CONTIGUOUS : {self.F_CONTIGUOUS}",
            f"  OWNDATA : {self.OWNDATA}",
            f"  WRITEABLE : {self.WRITEABLE}",
            f"  ALIGNED : {self.ALIGNED}",
            f"  WRITEBACKIFCOPY : {self.WRITEBACKIFCOPY}",
            f"  UPDATEIFCOPY : {self.UPDATEIFCOPY}",
        ]
        return "\n".join(lines)


def _format_float_scalar_str(v):
    """0 维浮点标量的字符串，与 rsnumpy 标量 str 对齐（如 75.0、85850000000.0）。"""
    if v != v:
        return 'nan'
    if v == float('inf'):
        return 'inf'
    if v == float('-inf'):
        return '-inf'
    return str(v)


class ndarray:
    """
    rsnumpy.ndarray - 多维数组对象。

    这是一个 Rust 后端支持的数组类型，提供与 rsnumpy 兼容的功能。

    参数:
        data (array_like): 数组的初始数据，可以是列表、元组或可迭代对象。

    属性:
        shape: 数组维度的元组。
        ndim: 维度数量。
        size: 元素总数。
        dtype: 元素数据类型。
        itemsize: 每个元素的字节大小。
        nbytes: 总字节数。
        T: 数组的转置。
        real: 数组的实部。
        imag: 数组的虚部。
    """

    def __init__(self, data=None, _dtype="float64", _fields=None, _raw_data=None,
                 shape=None, dtype=None, buffer=None, offset=0, strides=None,
                 order=None):
        _ = buffer, offset, strides, order
        # NumPy 的底层构造器 `np.ndarray(shape=(0, 0), dtype=...)`：按形状与
        # 类型造一个数组。rsnumpy 原本只接受数据，这类调用会直接 TypeError。
        # NumPy 此处返回的是未初始化内存；rsnumpy 没有该语义，用零填充，
        # 结果可预测得多。
        if shape is not None:
            import rsnumpy as _np
            built = _np.zeros(shape, dtype=dtype if dtype is not None else _dtype)
            self._array = built._array
            self._dtype = built._dtype
            self._fields = None
            self._raw_data = getattr(built, '_raw_data', None)
            self._complex_data = getattr(built, '_complex_data', None)
            return
        # pickle 反序列化时先 __new__（不传参），再 __setstate__；此时 data=None，跳过初始化
        if data is None:
            return
        if _is_ndarray(data):
            self._array = data._array
            self._dtype = data._dtype
            self._fields = getattr(data, '_fields', None)
            self._raw_data = getattr(data, '_raw_data', None)
            self._complex_data = getattr(data, '_complex_data', None)
        elif hasattr(data, '__class__') and data.__class__.__name__ == 'ndarray':
            self._array = data
            if getattr(data, 'is_complex', False):
                self._dtype = "complex128"
            else:
                self._dtype = _dtype
            self._fields = _fields
            self._raw_data = _raw_data
        else:
            # rsnumpy 整型/bool 标量不再是 int 子类：先转 Python 原生值，
            # Rust 构造只认原生数值/列表。object dtype 保留标量对象身份。
            if (isinstance(data, generic) and hasattr(data, 'tolist')
                    and str(_dtype) not in ('object', 'O')):
                data = data.tolist()
            self._raw_data = _raw_data
            if _raw_data is not None:
                # 原始数据存储：用一个虚拟 Rust 数组占位
                self._array = np.zeros((len(data),) if isinstance(data, (list, tuple)) else (1,))
            elif not isinstance(data, (list, tuple)) and hasattr(data, '__array_interface__'):
                # 实现数组接口协议的对象（如 PIL 图像）→ 由 Rust 底层解码缓冲区
                self._array, _dtype = _from_array_interface(data)
            else:
                if isinstance(data, tuple):
                    data = list(data)
                # numpy 语义：[arr, [arr, arr]]（同层裸数组 + 内含同形状数组的列表，
                # 内层长度 == arr.shape[0]）→ 矩形化尝试 → 广播错误
                if isinstance(data, (list, tuple)) and builtins.any(
                        _is_arr_like_el(e) for e in data):
                    _ae0 = [e for e in data if _is_arr_like_el(e)]
                    _s0 = getattr(_ae0[0], 'shape', None)
                    if _s0 is not None:
                        for e in data:
                            if (isinstance(e, (list, tuple))
                                    and len(e) > 1 and len(e) == _s0[0]
                                    and builtins.all(
                                        _is_arr_like_el(x) for x in e)):
                                if builtins.all(
                                        getattr(x, 'shape', None) == _s0
                                        for x in e):
                                    raise ValueError(
                                        "could not broadcast input "
                                        f"array from shape {_s0} into "
                                        f"shape {(len(e),) + _s0}")
                if isinstance(data, list) and not _is_rectangular(data):
                    # 不规则列表（子列表长度不同），存储为原始数据
                    self._raw_data = data
                    self._array = np.zeros((len(data),))
                else:
                    # 单次遍历同时获取扁平数据与复数/字符串标记
                    flat, has_c, has_s = _flatten_check(data)
                    if _dtype in ('object', 'O') and not isinstance(data, (list, tuple)):
                        # 0-d object 标量（str/complex/Fraction 等）：Rust 构造不接受任意
                        # Python 对象，原值存入 _raw_data，形状 ()。
                        self._array = np.zeros(())
                        self._raw_data = [data]
                    elif has_s or str(_dtype) in ('bytes_', 'string_', 'S', 'U'):
                        # 字符串数据或请求字符串 dtype → 存储为原始 Python 数据
                        # （标量 → 0-d）。非字符串元素按 numpy 语义 str() 转换。
                        _vals = list(flat) if flat else (
                            list(data) if isinstance(data, (list, tuple)) else [data])
                        _scalar_in = not isinstance(data, (list, tuple))
                        if str(_dtype) in ('bytes_', 'S'):
                            _vals = [
                                (v if isinstance(v, (bytes, bytearray))
                                 else str(v).encode('utf-8'))
                                for v in _vals]
                            self._str_dtype = ('S', _py_max(
                                (len(v) for v in _vals), default=0))
                        else:
                            _vals = [v if isinstance(v, str) else str(v)
                                     for v in _vals]
                            # 记录 (kind, width)，供 dtype/itemsize 与整数索引取标量使用：
                            # Python str 按 rsnumpy 惯例归为 unicode 'U'，bytes 归为 'S'。
                            if flat and _py_all(
                                    isinstance(v, (bytes, bytearray)) for v in flat):
                                self._str_dtype = ('S', _py_max(
                                    (len(bytes(v)) for v in flat), default=0))
                            else:
                                self._str_dtype = ('U', _py_max(
                                    (len(v) for v in _vals), default=0))
                        self._raw_data = _vals
                        self._array = (np.zeros(()) if _scalar_in
                                       else np.zeros((len(self._raw_data),)))
                        _dtype = "string_"
                    elif has_c:
                        # 包含复数 → 交由 Rust 原生复数解析（保留形状与虚部）
                        self._array = np.ndarray(data)
                        _dtype = "complex128"
                    else:
                        try:
                            raw = np.ndarray(data)
                        except (TypeError, ValueError):
                            # numpy 语义：object 数组元素含多个同形状数组 → 广播错误
                            if isinstance(data, (list, tuple)):
                                _ae = [e for e in data
                                       if _is_ndarray(e) or hasattr(e, '__array__')]
                                if _ae:
                                    _s0 = getattr(_ae[0], 'shape', None)
                                    if _s0 is not None:
                                        for e in data:
                                            if (isinstance(e, (list, tuple))
                                                    and len(e) > 1 and builtins.all(
                                                        _is_ndarray(x) or hasattr(x, '__array__')
                                                        for x in e)):
                                                if builtins.all(
                                                        getattr(x, 'shape', None) == _s0
                                                        for x in e):
                                                    raise ValueError(
                                                        "could not broadcast input "
                                                        f"array from shape {_s0} into "
                                                        f"shape {(len(e),) + _s0}")
                            # 元素为任意 Python 对象（Fraction 等）：Rust 构造失败 →
                            # 存为 object 原始数据，保留嵌套形状。
                            self._array = np.zeros(_nested_shape(data))
                            self._raw_data = list(data) if isinstance(data, (list, tuple)) else [data]
                            _dtype = 'object'
                        else:
                            # 实数输入但显式指定复数 dtype → 提升为原生复数（零虚部）
                            if _dtype in ('complex128', 'complex64'):
                                raw = _maybe_native_complex(raw, _dtype)
                            self._array = raw
            self._dtype = _dtype
            self._fields = _fields

    @staticmethod
    def _wrap(raw_array, _dtype="float64", _fields=None, _raw_data=None):
        """包装原始 Rust ndarray 到 Python 类。"""
        obj = ndarray.__new__(ndarray)
        obj._array = raw_array
        obj._dtype = _dtype
        obj._fields = _fields
        obj._raw_data = _raw_data
        return obj

    def _wrap_raw(raw_data):
        """包装原始 Python 数据（如字符串列表）到 ndarray。"""
        obj = ndarray.__new__(ndarray)
        obj._array = np.zeros((len(raw_data),))
        obj._dtype = "string_"
        obj._raw_data = list(raw_data)
        return obj

    def __repr__(self):
        fields = getattr(self, '_fields', None)
        if fields:
            return _format_structured_repr(self)
        if getattr(self, '_dtype', "float64") == "string_":
            raw_data = getattr(self, '_raw_data', None)
            if raw_data is not None:
                val_strs = [repr(x) for x in raw_data]
                return "array([" + " ".join(val_strs) + "])"
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            return _format_ragged_repr(self)
        cpx = getattr(self, '_complex_data', None)
        if cpx is not None:
            inner = _format_complex_repr_1d(cpx)
            return f"array({inner})"
        if getattr(self._array, 'is_complex', False):
            if self.ndim == 0:
                return _format_complex_scalar(self.tolist())
            inner = _format_complex_nested(self.tolist(), ", ")
            return f"array({inner})"
        dt = getattr(self, '_dtype', "float64")
        if dt == "float64" and self.ndim == 1:
            values = self._array.tolist()
            inner = _format_float_repr_1d(values)
            return f"array({inner})"
        return f"array({self.__str__()})"

    def __str__(self):
        # 空数组会触发 Rust 格式化器的越界 panic（src/formatting/mod.rs 里对
        # 空串做 `&s[1..]`）；NumPy 对空数组的 str 就是 `[]`。
        try:
            if self.size == 0:
                return "[]"
        except Exception:
            pass
        fields = getattr(self, '_fields', None)
        if fields:
            return _format_structured_str(self)
        if getattr(self, '_dtype', "float64") == "string_":
            raw_data = getattr(self, '_raw_data', None)
            if raw_data is not None:
                sd = getattr(self, '_str_dtype', None)
                if sd is not None and sd[0] == 'V':
                    _, width = sd
                    parts = []
                    for v in raw_data:
                        bs = bytes(v) if isinstance(v, (bytes, bytearray)) else str(v).encode('latin-1')
                        if width:
                            bs = bs[:width].ljust(width, b'\x00')
                        parts.append(_void_repr(bs))
                    return "[" + " ".join(parts) + "]"
                nested = isinstance(raw_data, list) and len(raw_data) > 0
                if nested and isinstance(raw_data[0], list):
                    lines = []
                    for row in raw_data:
                        inner = " ".join(repr(x) for x in row)
                        lines.append("[" + inner + "]")
                    return "[" + "\n ".join(lines) + "]"
                val_strs = [repr(x) for x in raw_data]
                return "[" + " ".join(val_strs) + "]"
            return str(raw_data)
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            return _format_ragged_str(self)
        cpx = getattr(self, '_complex_data', None)
        if cpx is not None:
            inner = _format_complex_repr_1d(cpx)
            return inner
        if getattr(self._array, 'is_complex', False):
            if self.ndim == 0:
                return _format_complex_scalar(self.tolist())
            return _format_complex_nested(self.tolist(), " ")
        if getattr(self, '_dtype', "float64") in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
            if getattr(self, '_is_empty', False):
                return np._format_int_str(self._array)
            return np._format_int_val_str(self._array)
        if getattr(self, '_dtype', "float64") == "bool":
            def format_bool_list(data):
                if isinstance(data, list):
                    inner = " ".join(format_bool_list(v) for v in data)
                    return "[" + inner + "]"
                else:
                    return "True" if data else "False"
            raw = self._array.tolist()
            if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
                lines = []
                for row in raw:
                    inner = " ".join(format_bool_list(v) for v in row)
                    lines.append("[" + inner + "]")
                return "[" + "\n ".join(lines) + "]"
            return format_bool_list(raw)
        if self.ndim == 0:
            # 0 维浮点数组的 str 与 rsnumpy 标量一致（保留末尾 .0，如 75.0）。
            return _format_float_scalar_str(float(self._array.tolist()))
        return np._format_float_str(self._array)

    def __format__(self, fmt):
        if self.ndim == 0:
            val = self.tolist()
            if isinstance(val, complex):
                return format(val, fmt)
            return format(val, fmt)
        return format(str(self), fmt)

    def __len__(self):
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            return len(raw)
        if self.ndim == 0:
            return 1
        return len(self._array)

    def tolist(self):
        """转换为 Python 列表，元素类型与 _dtype 一致。"""
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            return raw
        cpx = getattr(self, '_complex_data', None)
        if cpx is not None:
            return list(cpx)
        dt = getattr(self, '_dtype', 'float64')
        raw_list = self._array.tolist()
        if dt == 'int64':
            return _convert_nested(raw_list, int)
        if dt == 'bool':
            return _convert_nested(raw_list, builtins.bool)
        if dt in ('uint8', 'uint16', 'uint32', 'uint64'):
            # 底层为 f64 存储：无符号值按位宽补码还原（负存储 = 高位值）
            _umask = {8: 0xFF, 16: 0xFFFF, 32: 0xFFFFFFFF,
                      64: (1 << 64) - 1}[int(dt[4:])]

            def _uint_conv(v):
                try:
                    return int(v) & _umask
                except (ValueError, OverflowError):
                    return 0  # NaN 等非法值（与 array() 路径一致）
            return _convert_nested(raw_list, _uint_conv)
        if dt == 'string_':
            raw = getattr(self, '_raw_data', None)
            if raw is not None:
                return list(raw)
        return raw_list

    def tobytes(self, order='C'):
        """返回数组原始字节（与 rsnumpy.ndarray.tobytes 兼容）。

        字符串数组（S/U/V dtype）按 itemsize 补零对齐；数值数组按 dtype
        字节布局编码（小端）。
        """
        raw = getattr(self, '_raw_data', None)
        sd = getattr(self, '_str_dtype', None)
        if raw is not None and sd is not None:
            return _string_array_tobytes(_flatten_data(raw), sd)
        return _numeric_tobytes(self, order)

    def __iter__(self):
        # 一维（及标量）按元素迭代产生 Python 标量；高维按首轴迭代产生子数组，
        # 与 rsnumpy 行为一致（如 a, b = np.random.randn(2, n) 得到两个数组）。
        if self.ndim <= 1:
            vals = self.tolist()
            if isinstance(vals, list):
                return iter(vals)
            return iter([vals])
        return (self[i] for i in range(self.shape[0]))

    @property
    def __array_struct__(self):
        # numpy 兼容句柄（此处返回自身，array() 会解包）
        return self

    @property
    def __array_interface__(self):
        """NumPy 数组接口协议：让真实 rsnumpy/matplotlib 能直接消费 rsnumpy 数组。

        底层数据恒为 f64，按追踪的 _dtype 编码为对应类型字节（在 Rust 层完成）。
        字符串/复数/不规则数组无法用简单缓冲区表示，抛 AttributeError 交回默认处理。
        """
        if getattr(self, '_raw_data', None) is not None:
            raise AttributeError('__array_interface__')
        if getattr(self, '_complex_data', None) is not None:
            raise AttributeError('__array_interface__')
        if getattr(self._array, 'is_complex', False):
            # 原生复数无法用单一 f64 缓冲表示，交回默认处理。
            raise AttributeError('__array_interface__')
        typestr = _DTYPE_TO_TYPESTR.get(getattr(self, '_dtype', 'float64'))
        if typestr is None:
            raise AttributeError('__array_interface__')
        return np.array_interface(self._array, typestr)

    def __buffer__(self, flags):
        """PEP 3118 缓冲协议（PEP 688 / Python 3.12+）：转发到底层 Rust `_array`，
        让 memoryview(a) 与下游（如 rsplotlib 的 PyBuffer::get）零拷贝读取底层连续
        f64 内存，免去 __array_interface__ 的 bytes 副本开销。

        仅对 float64 暴露缓冲：底层存储恒为 f64，其字节布局与 float64 dtype 完全一致，
        缓冲 format 'd' 语义正确。int/bool 等 dtype 的底层仍是 f64，若也暴露缓冲，
        rsnumpy 会优先按缓冲（f8）而非 __array_interface__（如 <i8）解读，导致 dtype 失真；
        故这些 dtype 不暴露缓冲，消费方自动回退到 dtype 精确的 __array_interface__ bytes。
        绘图坐标以 float64 为主，快路径覆盖热点场景。

        注：__buffer__ 由 CPython 在 3.12+ 才识别；更早版本上 memoryview(a) 会失败，
        消费方自动回退到 __array_interface__ bytes，行为安全。

        非 float64 dtype 同样暴露底层 f64 缓冲（值正确，format 为 'd'）：
        numpy 的测试/消费方需要 memoryview 协议可用；dtype 敏感的协议消费方
        仍会经 __array_interface__ 路径获得精确 dtype。
        """
        # _raw_data（字符串/object/不规则）数组也暴露底层 f64 占位缓冲：
        # numpy 的测试/消费方需要 memoryview 协议可用。
        if getattr(self, '_complex_data', None) is not None or \
                str(getattr(self, '_dtype', '')).startswith('complex'):
            raise BufferError("complex array does not support the buffer protocol")
        return memoryview(self._array)

    def __release_buffer__(self, view):
        view.release()

    def __bool__(self):
        if self.size == 1:
            return builtins.bool(self.item())
        raise ValueError("The truth value of an array with more than one element is ambiguous.")

    def __invert__(self):
        """逐元素取反（~ 运算符）。"""
        return bitwise_not(self)

    def __getitem__(self, key):
        # Empty tuple key: return a copy of the array (numpy semantics: no
        # indexing applied). 0-d 数组：arr[()] 返回标量（numpy 语义）。
        if isinstance(key, tuple) and len(key) == 0:
            if self.ndim == 0:
                _dt_0 = str(getattr(self, '_dtype', 'float64'))
                _raw_0 = getattr(self, '_raw_data', None)
                if _raw_0 is not None:
                    _v_0 = _raw_0[0] if _raw_0 else None
                    _sd_0 = getattr(self, '_str_dtype', None)
                    if _sd_0 is not None:
                        return _str_scalar(_v_0, _sd_0)
                    return _v_0
                _cpx_0 = getattr(self, '_complex_data', None)
                if _cpx_0 is not None and _dt_0.startswith('complex'):
                    _v_0 = _cpx_0[0] if _cpx_0 else 0j
                else:
                    _v_0 = self.item()
                _cls_0 = _SCALAR_CLASS_BY_DTYPE.get(_dt_0)
                if _cls_0 is None:
                    return _v_0
                # 无符号高位值以补码负值存储：还原无符号值
                _u_bits_0 = {'uint8': 8, 'uint16': 16, 'uint32': 32,
                             'uint64': 64}.get(_dt_0)
                if _u_bits_0 and isinstance(_v_0, (int, float)) and _v_0 < 0:
                    _v_0 = int(_v_0) & ((1 << _u_bits_0) - 1)
                try:
                    return _cls_0(_v_0)
                except Exception:
                    return _v_0
            return self.copy()
        # numpy 语义：Ellipsis 展开为全切片（Rust 不支持 ... 索引）。
        # 注意必须用身份检查（`is`）：`in` 会触发 ndarray.__eq__ 崩溃。
        if key is Ellipsis:
            key = (slice(None),) * self.ndim
        elif isinstance(key, tuple):
            # 必须用身份检查（`is`）扫描：`in`/`==` 会触发 ndarray.__eq__；
            # 且不能用 any(...)——模块级 `any` 被 rsnumpy ufunc 遮蔽，
            # 对生成器恒返回 True（曾把 (0,1) 错误展开成 4 元 key 致 Rust panic）。
            _ell_c = 0
            for _k in key:
                if _k is Ellipsis:
                    _ell_c += 1
            if _ell_c > 1:
                # numpy 语义：多个省略号非法
                raise IndexError(
                    "an index can only have a single ellipsis ('...')")
            _e_i = _py_index(key, lambda k: k is Ellipsis)
            if _e_i >= 0:
                _e_n = self.ndim - (len(key) - 1)
                key = key[:_e_i] + (slice(None),) * _e_n + key[_e_i + 1:]
        # 0-d 数组的布尔标量 / 0-d 布尔数组索引（numpy 语义）：
        # True → 1-d [value]，False → 空数组。不走掩码/花式路径（Rust 0-d panic）。
        if self.ndim == 0:
            _bflag = None
            if builtins.isinstance(key, builtins.bool):
                _bflag = key
            elif builtins.isinstance(key, bool_) and not _is_ndarray(key):
                _bflag = builtins.bool(key)
            elif _is_ndarray(key) and str(getattr(key, '_dtype', '')) == 'bool' \
                    and getattr(key, 'ndim', 0) == 0:
                _bflag = builtins.bool(key.item())
            if _bflag is not None:
                _v0 = self.item()
                return ndarray([_v0] if _bflag else [], _dtype=self._dtype)
        # 0-d 布尔数组索引（任意维度数组）：numpy 语义等价 a[None] 的选择
        if _is_ndarray(key) and getattr(key, 'ndim', 0) == 0 and \
                str(getattr(key, '_dtype', '')) == 'bool':
            _v0b = builtins.bool(key.item())
            _base_b = self[None]
            return _base_b[:1] if _v0b else _base_b[:0]
        # 标量布尔值作为索引：转换为 0-d rsnumpy 数组
        if isinstance(key, bool):
            key = ndarray([key], _dtype='bool')
        # 结构化数组索引：字段 / 多字段 / 记录标量 / 子结构切片
        if getattr(self, '_dtype_obj', None) is not None:
            handled, res = _struct_getitem(self, key)
            if handled:
                return res
        # object 数组整数索引：返回存储的 Python 引用（numpy 保持对象身份）
        if str(self._dtype) == 'object' and getattr(self, '_raw_data', None) is not None:
            keyseq = key if isinstance(key, tuple) else (key,)
            if builtins.all(isinstance(k, int) and not isinstance(k, bool)
                            for k in keyseq):
                try:
                    cur = self._raw_data
                    for k in keyseq:
                        cur = cur[int(k)]
                    return cur
                except Exception:
                    pass
        # 字符串/字节/void（S/U/V）dtype 数组的整数索引：返回标量。
        # S→去尾零 bytes，U→去尾零 str，V→void。
        sd = getattr(self, '_str_dtype', None)
        raw = getattr(self, '_raw_data', None)
        if sd is not None and raw is not None and isinstance(key, int) and not isinstance(key, bool):
            flat = _flatten_data(raw)
            return _str_scalar(flat[key], sd)
        # 复数数组的布尔索引
        cpx = getattr(self, '_complex_data', None)
        if _is_ndarray(key) and cpx is not None:
            mask = key.tolist()
            filtered = [v for v, m in zip(cpx, mask) if m > 0.5]
            return ndarray(filtered)
        # 布尔掩码索引：a[bool_mask]。掩码形状需与 self 的前若干维一致，
        # 返回被掩码选中的元素（按 C 序展开），剩余维度保留。
        if (_is_ndarray(key) and getattr(key, '_dtype', None) == 'bool' and key.ndim >= 1):
            mask_shape = tuple(key.shape)
            self_shape = tuple(self.shape)
            k = len(mask_shape)
            if mask_shape != self_shape[:k] and k <= len(self_shape):
                # numpy 语义：布尔掩码维度必须与数组前若干轴严格一致
                for _bi in range(builtins.min(k, len(self_shape))):
                    if mask_shape[_bi] != self_shape[_bi]:
                        raise IndexError(
                            f"boolean index did not match indexed array along "
                            f"axis {_bi}; size of axis is {self_shape[_bi]} "
                            f"but size of corresponding boolean axis is "
                            f"{mask_shape[_bi]}")
            if mask_shape == self_shape[:k]:
                rest = self_shape[k:]
                block = 1
                for s in rest:
                    block *= s
                raw = np.masked_select(self._array, key._array, block)
                result = ndarray._wrap(raw, _dtype=self._dtype)
                if rest:
                    return result.reshape((len(result) // block,) + rest)
                return result
        # np.newaxis (None) 支持：a[np.newaxis, :] / a[:, np.newaxis]
        # 先用去掉 None 的键做常规索引，再在结果的相应输出位置插入 size-1 维度。
        _key_seq = key if isinstance(key, tuple) else (key,)
        if _py_any(k is None for k in _key_seq):
            newaxis_positions = []
            base_key = []
            out_pos = 0
            for k in _key_seq:
                if k is None:
                    newaxis_positions.append(out_pos)
                    out_pos += 1
                elif isinstance(k, bool):
                    base_key.append(k)
                    out_pos += 1
                elif isinstance(k, int):
                    base_key.append(k)  # 整数索引消耗输入轴但不产生输出维度
                else:
                    base_key.append(k)
                    out_pos += 1
            base = self[tuple(base_key)]
            new_shape = list(base.shape)
            for p in newaxis_positions:
                new_shape.insert(p, 1)
            return base.reshape(tuple(new_shape))
        # 展开省略号（...），补充完整切片以匹配数组维度
        if isinstance(key, tuple):
            new_key = []
            ellipsis_count = np.isum([1 for k in key if k is Ellipsis])
            if ellipsis_count > 0:
                non_ellipsis = [k for k in key if k is not Ellipsis]
                fill = self.ndim - len(non_ellipsis)
                for k in key:
                    if k is Ellipsis:
                        for _ in range(fill):
                            new_key.append(slice(None, None, None))
                    else:
                        new_key.append(k)
                key = tuple(new_key)
        # 将 Python ndarray 索引展平为 Python list，确保 Rust 端可正确解析；
        # range 对象同样转为 list（Rust 端不支持 range 索引）。
        if isinstance(key, tuple):
            key = tuple(
                list(k) if isinstance(k, range)
                else _ndarray_to_index_list(k)
                if hasattr(k, '_array') else k
                for k in key
            )
        else:
            key = (list(key) if isinstance(key, range)
                   else _ndarray_to_index_list(key)
                   if hasattr(key, '_array') else key,)
        # 越界整数索引：numpy 语义抛 IndexError
        if isinstance(key, tuple):
            for _ax_i, _vi in enumerate(key):
                if _ax_i >= len(self.shape):
                    break
                _ns_i = self.shape[_ax_i]
                if isinstance(_vi, int) and not isinstance(_vi, bool):
                    if _vi < -_ns_i or _vi >= _ns_i:
                        raise IndexError(f"index {_vi} is out of bounds for axis {_ax_i} with size {_ns_i}")
                elif isinstance(_vi, list):
                    for _vj in _vi:
                        if isinstance(_vj, int) and not isinstance(_vj, bool) and (_vj < -_ns_i or _vj >= _ns_i):
                            raise IndexError(f"index {_vj} is out of bounds for axis {_ax_i} with size {_ns_i}")
        # 索引维度过多：numpy 语义 IndexError（Rust 花式路径对 len(key) > ndim 会 panic）
        if isinstance(key, tuple) and len(key) > self.ndim:
            raise IndexError(
                f"too many indices for array: array is {self.ndim}-dimensional, "
                f"but {len(key)} were indexed")
        # 浮点数索引校验（numpy 语义）：
        # 1) float 作为整数索引 → IndexError（优先于切片边界检查）
        # 2) float 作为切片边界 → TypeError
        _k_seq = key if isinstance(key, tuple) else (key,)
        for _k in _k_seq:
            if isinstance(_k, float) and not isinstance(_k, builtins.bool):
                raise IndexError(
                    "only integers, slices (`:`), ellipsis (`...`) and "
                    "integer or boolean arrays are valid indices")
            if _is_ndarray(_k) and str(getattr(_k, '_dtype', '')) \
                    .startswith('float'):
                # numpy 语义：浮点数组不可作索引
                raise IndexError(
                    "arrays used as indices must be of integer (or boolean) "
                    "type")
        for _k in _k_seq:
            if isinstance(_k, slice):
                for _sb in (_k.start, _k.stop, _k.step):
                    if isinstance(_sb, float) and \
                            not isinstance(_sb, builtins.bool):
                        raise TypeError(
                            "slice indices must be integers or None or have "
                            "an __index__ method")
        # 高级索引校验（numpy 语义）：
        # 1) 列表索引含非法元素（切片/None/字符串等）→ IndexError
        # 2) 多个高级索引形状不可广播 → IndexError
        # 3) 高级索引（列表/ndarray/布尔标量）个数 > 32（numpy NPY_MAXARGS）→ IndexError
        if isinstance(key, tuple):
            _adv_n = 0
            for _k in key:
                if isinstance(_k, list) or _is_ndarray(_k) or \
                        isinstance(_k, builtins.bool):
                    _adv_n += 1
            if _adv_n > 32:
                raise IndexError("too many indices for array")
            _adv_shapes = []
            for _k in key:
                if isinstance(_k, list):
                    _fl = _NdFlatIter._flat_list(_k)
                    for _v in _fl:
                        if isinstance(_v, (slice, str, bytes)) or \
                                _v is None or _v is Ellipsis:
                            raise IndexError(
                                "only integers, slices (`:`), ellipsis (`...`) "
                                "and integer or boolean arrays are valid "
                                "indices")
                        if isinstance(_v, float) and \
                                not isinstance(_v, builtins.bool):
                            # numpy 语义：浮点列表元素不可作索引
                            raise IndexError(
                                "arrays used as indices must be of integer "
                                "(or boolean) type")
                    if _fl:
                        _adv_shapes.append(_NdFlatIter._list_shape(_k))
                elif isinstance(_k, tuple):
                    # numpy 语义：元组键内不允许嵌套元组
                    raise IndexError(
                        "too many indices for array: arrays used as indices "
                        "must be of integer (or boolean) type")
                elif _is_ndarray(_k):
                    if str(getattr(_k, '_dtype', '')) != 'bool':
                        _adv_shapes.append(tuple(_k.shape))
            if len(_adv_shapes) >= 2:
                _bshape = _adv_shapes[0]
                for _s in _adv_shapes[1:]:
                    for _a, _b in zip(reversed(_bshape), reversed(_s)):
                        if _a != _b and _a != 1 and _b != 1:
                            raise IndexError(
                                "shape mismatch: indexing arrays could not "
                                f"be broadcast together with shapes {_bshape} "
                                f"{_s}")
            # 元组内布尔掩码：形状必须与对应轴严格一致（numpy 语义）。
            # 注意 bool ndarray 此时可能已被转为（嵌套）bool 列表，需同查。
            _ax_pos = 0
            for _k in key:
                if _k is None:
                    continue
                _mask_shape = None
                if _is_ndarray(_k) and \
                        str(getattr(_k, '_dtype', '')) == 'bool' and \
                        getattr(_k, 'ndim', 0) >= 1:
                    _mask_shape = tuple(_k.shape)
                elif isinstance(_k, list) and _k:
                    _fl_m = _NdFlatIter._flat_list(_k)
                    if _fl_m and _py_all(isinstance(x, builtins.bool)
                                         for x in _fl_m):
                        _mask_shape = _NdFlatIter._list_shape(_k)
                if _mask_shape is not None:
                    _ms = _mask_shape
                    _ss = tuple(self.shape)
                    if _ax_pos + len(_ms) <= len(_ss) and \
                            _ms != _ss[_ax_pos:_ax_pos + len(_ms)]:
                        for _bi in range(len(_ms)):
                            if _ms[_bi] != _ss[_ax_pos + _bi]:
                                raise IndexError(
                                    "boolean index did not match indexed "
                                    f"array along axis {_ax_pos + _bi}; size "
                                    f"of axis is {_ss[_ax_pos + _bi]} but "
                                    f"size of corresponding boolean axis is "
                                    f"{_ms[_bi]}")
                    _ax_pos += len(_ms)
                elif isinstance(_k, slice) or (
                        isinstance(_k, int) and
                        not isinstance(_k, builtins.bool)):
                    _ax_pos += 1
        # 委托给 Rust getitem_multi 处理所有索引逻辑
        result = np.getitem_multi(self._array, key, list(self.shape))
        result = _wrap_result(result, self._dtype)
        # Track strided slices so flags.c_contiguous reports False like numpy.
        _ks_s = key if isinstance(key, tuple) else (key,)
        if _is_ndarray(result) and _py_any(
                isinstance(k, slice) and k.step not in (None, 1) for k in _ks_s):
            result._rs_strided = True
        # numpy 语义：简单索引返回视图（继承 writeable）；
        # 花式索引（ndarray 索引/布尔掩码/list 索引）返回拷贝（恒可写）
        if _is_ndarray(result) and not _py_any(
                _is_ndarray(k) or isinstance(k, list) for k in _ks_s):
            result._rs_writeable = getattr(self, '_rs_writeable', True)
        return result

    def __setitem__(self, key, value):
        # numpy 语义：Ellipsis 展开为全切片（Rust 不支持 ... 索引）。
        # 注意必须用身份检查（`is`）：`in` 会触发 ndarray.__eq__ 崩溃；
        # 且不能用 any(...)——模块级 `any` 被 rsnumpy ufunc 遮蔽，对生成器恒 True。
        if key is Ellipsis:
            key = (slice(None),) * self.ndim
        elif isinstance(key, tuple):
            _ell_c = 0
            for _k in key:
                if _k is Ellipsis:
                    _ell_c += 1
            if _ell_c > 1:
                # numpy 语义：多个省略号非法
                raise IndexError(
                    "an index can only have a single ellipsis ('...')")
            _adv_n = 0
            for _k in key:
                if isinstance(_k, list) or _is_ndarray(_k) or \
                        isinstance(_k, builtins.bool):
                    _adv_n += 1
            if _adv_n > 32:
                # numpy NPY_MAXARGS 限制
                raise IndexError("too many indices for array")
            # 多个高级索引形状不可广播 → IndexError（numpy 语义，避免
            # 形状不匹配直达 Rust setitem_multi 触发 panic abort）
            _adv_shapes = []
            for _k in key:
                if isinstance(_k, list):
                    _fl = _NdFlatIter._flat_list(_k)
                    for _v in _fl:
                        if isinstance(_v, (slice, str, bytes)) or \
                                _v is None or _v is Ellipsis:
                            raise IndexError(
                                "only integers, slices (`:`), ellipsis (`...`) "
                                "and integer or boolean arrays are valid "
                                "indices")
                    if _fl:
                        _adv_shapes.append(_NdFlatIter._list_shape(_k))
                elif _is_ndarray(_k):
                    if str(getattr(_k, '_dtype', '')) != 'bool':
                        _adv_shapes.append(tuple(_k.shape))
            if len(_adv_shapes) >= 2:
                _bshape = _adv_shapes[0]
                for _s in _adv_shapes[1:]:
                    for _a, _b in zip(reversed(_bshape), reversed(_s)):
                        if _a != _b and _a != 1 and _b != 1:
                            raise IndexError(
                                "shape mismatch: indexing arrays could not "
                                f"be broadcast together with shapes {_bshape} "
                                f"{_s}")
            _e_i = _py_index(key, lambda k: k is Ellipsis)
            if _e_i >= 0:
                _e_n = self.ndim - (len(key) - 1)
                key = key[:_e_i] + (slice(None),) * _e_n + key[_e_i + 1:]
        # 结构化数组赋值：字段 / 多字段 / 记录 / 切片
        if getattr(self, '_dtype_obj', None) is not None:
            if _struct_setitem(self, key, value):
                return
        # object 数组赋值：直接写入 _raw_data 引用（Rust 无 object 存储）
        _raw_o = getattr(self, '_raw_data', None)
        if str(self._dtype) == 'object' and _raw_o is None:
            # _raw_data 缺失（empty/fromiter 等路径）：先初始化
            try:
                _t_o = self.tolist()
                _init_o = list(_t_o) if isinstance(_t_o, list) else [_t_o]
            except Exception:
                _init_o = [None] * max(int(getattr(self, 'size', 1) or 1), 1)
            if _init_o and builtins.all(v in (0.0, 0) for v in _init_o):
                _init_o = [None] * len(_init_o)
            self._raw_data = _init_o
            _raw_o = _init_o
        if str(self._dtype) == 'object' and _raw_o is not None:
            keyseq = key if isinstance(key, tuple) else (key,)
            if builtins.all(isinstance(k, int) and not isinstance(k, bool)
                            for k in keyseq):
                try:
                    cur = _raw_o
                    for k in keyseq[:-1]:
                        cur = cur[int(k)]
                    if keyseq:
                        cur[int(keyseq[-1])] = value
                    elif isinstance(_raw_o, list) and len(_raw_o) == 1:
                        _raw_o[0] = value
                    return
                except Exception:
                    pass
        # 布尔掩码空选择：numpy 语义下为 no-op，避免 Rust 端空索引 panic
        if _empty_bool_mask_key(key):
            return
        # 整数溢出语义：向整数数组赋越界/NaN 值抛异常（numpy 语义）
        _dt_si = getattr(self, '_dtype', None)
        if _dt_si in _INT_DTYPES:
            _vv = value
            if _is_ndarray(_vv) and getattr(_vv, 'ndim', 1) == 0:
                try:
                    _vv = _vv.item()
                except Exception:
                    _vv = value
            if not _is_ndarray(_vv) and not isinstance(_vv, (list, tuple)):
                if isinstance(_vv, float) and _vv != _vv:
                    raise ValueError("cannot convert float NaN to integer")
                if isinstance(_vv, (int, float, generic)) and not isinstance(_vv, builtins.bool):
                    _lo_i, _hi_i = _INT_RANGES[_dt_si]
                    if int(_vv) < _lo_i or int(_vv) > _hi_i:
                        raise OverflowError(f"Python int too large to convert to {_dt_si}")
        # 展开省略号（...），补充完整切片以匹配数组维度
        if key is Ellipsis:
            key = tuple(slice(None, None, None) for _ in range(self.ndim))
        elif isinstance(key, tuple):
            new_key = []
            ellipsis_count = np.isum([1 for k in key if k is Ellipsis])
            if ellipsis_count > 0:
                non_ellipsis = [k for k in key if k is not Ellipsis]
                fill = self.ndim - len(non_ellipsis)
                for k in key:
                    if k is Ellipsis:
                        for _ in range(fill):
                            new_key.append(slice(None, None, None))
                    else:
                        new_key.append(k)
                key = tuple(new_key)
        # 索引维度过多：numpy 语义 IndexError（Rust 花式路径对 len(key) > ndim 会 panic）
        if isinstance(key, tuple) and len(key) > self.ndim:
            raise IndexError(
                f"too many indices for array: array is {self.ndim}-dimensional, "
                f"but {len(key)} were indexed")
        # 越界整数索引：numpy 语义 IndexError（与 getitem 一致）
        if isinstance(key, tuple):
            for _ax_i, _vi in builtins.enumerate(key):
                if _ax_i >= len(self.shape):
                    break
                _ns_i = self.shape[_ax_i]
                if isinstance(_vi, int) and not isinstance(_vi, builtins.bool):
                    if _vi < -_ns_i or _vi >= _ns_i:
                        raise IndexError(
                            f"index {_vi} is out of bounds for axis {_ax_i} "
                            f"with size {_ns_i}")
                elif isinstance(_vi, list):
                    for _vj in _vi:
                        if isinstance(_vj, int) and \
                                not isinstance(_vj, builtins.bool) and \
                                (_vj < -_ns_i or _vj >= _ns_i):
                            raise IndexError(
                                f"index {_vj} is out of bounds for axis "
                                f"{_ax_i} with size {_ns_i}")
        elif isinstance(key, list) and self.ndim >= 1:
            _ns_0 = self.shape[0]
            for _vj in key:
                if isinstance(_vj, int) and not isinstance(_vj, builtins.bool) \
                        and (_vj < -_ns_0 or _vj >= _ns_0):
                    raise IndexError(
                        f"index {_vj} is out of bounds for axis 0 with size {_ns_0}")
        elif _is_ndarray(key) and self.ndim >= 1 and \
                str(getattr(key, '_dtype', '')) != 'bool':
            _ns_0 = self.shape[0]
            _kl = key.flatten().tolist()
            for _vj in _kl:
                if isinstance(_vj, int) and not isinstance(_vj, builtins.bool) \
                        and (_vj < -_ns_0 or _vj >= _ns_0):
                    raise IndexError(
                        f"index {_vj} is out of bounds for axis 0 with size {_ns_0}")
        if isinstance(key, tuple):
            # 将 Python ndarray 索引展平为 list。一维布尔掩码（选择 axis-0）
            # 转成 bool 列表，由 Rust 的 PyList-bool 分支映射为 True 位置；
            # 与整体形状相同的掩码保留数组形式，走 Rust FullBoolMask 路径。
            # range 索引同样转为 list。
            key = tuple(
                list(k) if isinstance(k, range)
                else _ndarray_to_index_list(k)
                if hasattr(k, '_array') and (k.dtype != 'bool' or k.shape != self.shape)
                else k
                for k in key
            )
        elif not isinstance(key, str):
            key = (list(key) if isinstance(key, range)
                   else _ndarray_to_index_list(key)
                   if hasattr(key, '_array') and (key.dtype != 'bool' or key.shape != self.shape)
                   else key,)
        if isinstance(key, tuple):
            # 计算目标形状：将切片/整数索引应用到 self.shape 得到赋值目标的形状
            target_shape = []
            key_idx = 0
            for dim_size in self.shape:
                if key_idx < len(key):
                    k = key[key_idx]
                    key_idx += 1
                    if isinstance(k, slice):
                        start = k.start if k.start is not None else 0
                        stop = k.stop if k.stop is not None else dim_size
                        step = k.step if k.step is not None else 1
                        if hasattr(start, '__index__'):
                            start = start.__index__()
                        if hasattr(stop, '__index__'):
                            stop = stop.__index__()
                        target_shape.append(_py_max(0, (stop - start + step - 1) // step))
                    elif isinstance(k, int):
                        continue
                    elif isinstance(k, list):
                        # Bool lists are masks: Rust maps them to True
                        # positions, so the selection size is the number of
                        # True entries, not the list length.
                        if k and _py_all(isinstance(x, builtins.bool) for x in k):
                            target_shape.append(len([x for x in k if x]))
                        else:
                            target_shape.append(len(k))
                    elif _is_ndarray(k):
                        if k.dtype == 'bool':
                            flat_k = k.flatten()
                            true_count = 0
                            for x in flat_k:
                                if x:
                                    true_count += 1
                            if k.shape == self.shape:
                                target_shape = [true_count]
                                break
                            else:
                                target_shape.append(true_count)
                        else:
                            target_shape.append(dim_size)
                    else:
                        target_shape.append(dim_size)
                else:
                    target_shape.append(dim_size)
            # 广播支持：value 形状必须能广播到目标形状（numpy 语义），
            # 否则报 "could not broadcast input array from shape ... into shape ..."
            if _is_ndarray(value):
                try:
                    bcast_shape = _broadcast_shape(tuple(target_shape), value.shape)
                except ValueError:
                    bcast_shape = None
                if bcast_shape != tuple(target_shape):
                    if not target_shape and value.size == 1:
                        pass  # size-1 数组 → 标量目标：允许
                    else:
                        # 消息用实际选择形状（相邻 fancy 索引有广播语义，
                        # target_shape 的逐维近似会多出一个维度）
                        try:
                            _sel_t = tuple(getattr(self[key], 'shape',
                                                   tuple(target_shape)))
                        except Exception:
                            _sel_t = tuple(target_shape)
                        def _ns(_s):
                            return "(" + ",".join(map(str, _s)) + ")"
                        raise ValueError(
                            "could not broadcast input array from shape "
                            f"{_ns(value.shape)} into shape "
                            f"{_ns(_sel_t)}")
                elif bcast_shape != value.shape:
                    value = broadcast_to(value, bcast_shape)
            val = _cast_setitem_value(_setitem_value(value), self._dtype)
            np.setitem_multi(self._array, key, list(self.shape), val)
        else:
            self._array[key] = value

    def __add__(self, other):
        dt = _promote_dtype(self._dtype, other)
        if self.size == 0:
            return self.copy()
        if _is_ndarray(other):
            return _wrap_result(self._array + other._array, dt)
        if isinstance(other, (tuple, list)):
            other_arr = ndarray(other)
            return _wrap_result(self._array + other_arr._array, dt)
        return _wrap_result(self._array + other, dt)

    def __radd__(self, other):
        dt = _promote_dtype(self._dtype, other)
        if self.size == 0:
            return self.copy()
        return _wrap_result(other + self._array, dt)

    def __sub__(self, other):
        dt = _promote_dtype(self._dtype, other)
        if self.size == 0:
            return self.copy()
        if _is_ndarray(other):
            return _wrap_result(self._array - other._array, dt)
        if isinstance(other, (tuple, list)):
            other_arr = ndarray(other)
            return _wrap_result(self._array - other_arr._array, dt)
        return _wrap_result(self._array - other, dt)

    def __rsub__(self, other):
        dt = _promote_dtype(self._dtype, other)
        if self.size == 0:
            return self.copy()
        return _wrap_result(other - self._array, dt)

    def __mul__(self, other):
        dt = _promote_dtype(self._dtype, other)
        if self.size == 0:
            return self.copy()
        if _is_ndarray(other):
            # Rust core handles NumPy broadcasting (e.g. (4,4)*(4,)).
            return _wrap_result(self._array * other._array, dt)
        if isinstance(other, (tuple, list)):
            other_arr = ndarray(other)
            return _wrap_result(self._array * other_arr._array, dt)
        # Python scalar
        try:
            scalar = float(other)
        except (TypeError, ValueError):
            try:
                scalar = complex(other)
            except (TypeError, ValueError):
                return NotImplemented
        a_flat = self.ravel().tolist()
        if not isinstance(a_flat, list):
            a_flat = [a_flat]
        result = [x * scalar for x in a_flat]
        out = _wrap_result(result, dt)
        if self.ndim > 1:
            out = out.reshape(self.shape)
        return out

    def __rmul__(self, other):
        dt = _promote_dtype(self._dtype, other)
        if self.size == 0:
            return self.copy()
        return self.__mul__(other)

    def __truediv__(self, other):
        dt = _truediv_dtype(self._dtype)
        if self.size == 0:
            return self.copy()
        if _is_ndarray(other):
            return _wrap_result(self._array / other._array, dt)
        if isinstance(other, (tuple, list)):
            other_arr = ndarray(other)
            return _wrap_result(self._array / other_arr._array, dt)
        return _wrap_result(self._array / other, dt)

    def __rtruediv__(self, other):
        dt = _truediv_dtype(self._dtype)
        if self.size == 0:
            return self.copy()
        return _wrap_result(other / self._array, dt)

    def __floordiv__(self, other):
        """逐元素地板除（// 运算符）。"""
        dt = _truediv_dtype(self._dtype)
        if self.size == 0:
            return self.copy()
        if _is_ndarray(other):
            a_vals = self.ravel().tolist()
            b_vals = other.ravel().tolist()
            if not isinstance(a_vals, list):
                a_vals = [a_vals]
            if not isinstance(b_vals, list):
                b_vals = [b_vals]
            result = [x // y for x, y in zip(a_vals, b_vals)]
        else:
            a_vals = self.ravel().tolist()
            if not isinstance(a_vals, list):
                a_vals = [a_vals]
            result = [x // other for x in a_vals]
        out = _wrap_result(result, dt)
        if self.ndim > 1:
            out = out.reshape(self.shape)
        return out

    def __ifloordiv__(self, other):
        """原地地板除（//= 运算符）。"""
        res = self.__floordiv__(other)
        # numpy 语义：写回时按 same-kind 规则检查提升结果能否写回原 dtype
        # （如 int64 //= uint64 → float64 结果无法写回 int64）
        if self._dtype in _INT_DTYPES:
            if _is_ndarray(other):
                _odt = str(other._dtype)
            elif isinstance(other, bool):
                _odt = 'bool'
            elif isinstance(other, int):
                _odt = 'int64'
            elif isinstance(other, float):
                _odt = 'float64'
            elif isinstance(other, complex):
                _odt = 'complex128'
            else:
                _odt = str(self._dtype)
            _raise_fd = 'float' in _odt or 'complex' in _odt
            if (not _raise_fd and _odt in _INT_DTYPES
                    and self._dtype.startswith('int')
                    and _odt.startswith('uint')):
                # numpy：等宽/更宽无符号 → float64 结果
                _raise_fd = (_INT_ITEMSIZES[_odt] >= _INT_ITEMSIZES[self._dtype])
            if _raise_fd:
                raise TypeError(
                    f"Cannot cast ufunc 'floor_divide' output from "
                    f"float64 to {self._dtype} with casting rule 'same_kind'")
        if res.shape == self.shape:
            self._array = res._array
        return self

    def __rfloordiv__(self, other):
        """逐元素地板除（右操作数，// 运算符）。"""
        dt = _truediv_dtype(self._dtype)
        if self.size == 0:
            return self.copy()
        b_vals = other.ravel().tolist() if _is_ndarray(other) else [other]
        if not isinstance(b_vals, list):
            b_vals = [b_vals]
        a_vals = self.ravel().tolist()
        if not isinstance(a_vals, list):
            a_vals = [a_vals]
        result = [x // y for x, y in zip(b_vals, a_vals)]
        out = _wrap_result(result, dt)
        if self.ndim > 1:
            out = out.reshape(self.shape)
        return out

    def __mod__(self, other):
        """逐元素取模（% 运算符）。"""
        return mod(self, other)

    def __rmod__(self, other):
        """逐元素取模（右操作数，% 运算符）。"""
        return mod(other, self)

    def __matmul__(self, other):
        if _is_ndarray(other):
            try:
                return _wrap_result(np.linalg.matmul(self._array, other._array), self._dtype)
            except ValueError:
                return _matmul_fallback(self, other)
        # Non-array operand (e.g. a sparse matrix): try to convert it; if
        # that fails, return NotImplemented so other.__rmatmul__ can kick in.
        try:
            return _wrap_result(np.linalg.matmul(self._array, np.asarray(other)._array), self._dtype)
        except (TypeError, ValueError, AttributeError):
            return NotImplemented

    def __pow__(self, other):
        _dt_p = str(getattr(self, '_dtype', ''))
        if _dt_p.startswith('complex') or getattr(self, '_complex_data', None) is not None:
            # 复数 pow：Rust core 不支持复数幂 → Python 级复数运算
            try:
                def _fc_p(x):
                    if isinstance(x, list):
                        out = []
                        for v in x:
                            out.extend(_fc_p(v))
                        return out
                    return [x]
                _t_p = self.tolist()
                _flat_p = _fc_p(_t_p) if isinstance(_t_p, list) else [_t_p]
                if _is_ndarray(other):
                    _to_p = other.tolist()
                    _oflat_p = _fc_p(_to_p) if isinstance(_to_p, list) else [_to_p]
                    res = [complex(a) ** complex(b) for a, b in zip(_flat_p, _oflat_p)]
                else:
                    res = [complex(a) ** complex(other) for a in _flat_p]
                _r_p = ndarray(res)
                if self.ndim == 0:
                    _r_p = _r_p.reshape(())
                elif _r_p.ndim < self.ndim and self.shape:
                    _r_p = _r_p.reshape(self.shape)
                return _r_p
            except Exception:
                pass
        if _is_ndarray(other):
            # Rust core broadcasts elementwise power (e.g. (1,)**(4,1)).
            return _wrap_result(self._array ** other._array, self._dtype)
        if isinstance(other, (tuple, list)):
            other_arr = ndarray(other)
            return _wrap_result(self._array ** other_arr._array, self._dtype)
        return _wrap_result(np.power(self._array, np.ndarray([other])), self._dtype)

    def __rpow__(self, other):
        _dt_p = str(getattr(self, '_dtype', ''))
        if _dt_p.startswith('complex') or getattr(self, '_complex_data', None) is not None:
            # 复数 rpow：Python 级复数运算
            try:
                def _fc_p(x):
                    if isinstance(x, list):
                        out = []
                        for v in x:
                            out.extend(_fc_p(v))
                        return out
                    return [x]
                _t_p = self.tolist()
                _flat_p = _fc_p(_t_p) if isinstance(_t_p, list) else [_t_p]
                if _is_ndarray(other):
                    _to_p = other.tolist()
                    _oflat_p = _fc_p(_to_p) if isinstance(_to_p, list) else [_to_p]
                    res = [complex(b) ** complex(a) for a, b in zip(_flat_p, _oflat_p)]
                else:
                    res = [complex(other) ** complex(a) for a in _flat_p]
                _r_p = ndarray(res)
                if self.ndim == 0:
                    _r_p = _r_p.reshape(())
                elif _r_p.ndim < self.ndim and self.shape:
                    _r_p = _r_p.reshape(self.shape)
                return _r_p
            except Exception:
                pass
        if _is_ndarray(other):
            return _wrap_result(other._array ** self._array, self._dtype)
        if isinstance(other, (tuple, list)):
            other_arr = ndarray(other)
            return _wrap_result(other_arr._array ** self._array, self._dtype)
        return _wrap_result(np.power(np.ndarray([other]), self._array), self._dtype)

    def __neg__(self):
        """逐元素取负（- 运算符）。"""
        return _wrap_result(self._array * -1, self._dtype)

    def __pos__(self):
        """一元正号（+ 运算符），返回副本。"""
        return _wrap_result(self._array.copy(), self._dtype)

    def __abs__(self):
        """逐元素绝对值（内建 abs() 函数）。"""
        result = _wrap_result(np.abs(self._array), self._dtype)
        if result.ndim == 0:
            return result.item()
        return result

    def __eq__(self, other):
        if getattr(self, '_dtype_obj', None) is not None and \
                _is_ndarray(other) and getattr(other, '_dtype_obj', None) is not None:
            return _struct_eq(self, other)
        # 字符串/字节数组：按原始 Python 值逐元素比较（底层是 f64 占位）
        if getattr(self, '_raw_data', None) is not None and \
                _is_ndarray(other) and getattr(other, '_raw_data', None) is not None:
            a_flat = _flatten_vals(self._raw_data)
            b_flat = _flatten_vals(other._raw_data)
            if self.size == other.size and self.shape == other.shape:
                res = [x == y for x, y in zip(a_flat, b_flat)]
                out = ndarray(res)
                out._dtype = 'bool'
                return out.reshape(self.shape)
        # 0-d 数值 vs 字符串/字节 → False 标量（numpy 语义）
        if _incomparable_0d(self, other):
            return bool_(False)
        # 非结构化 void：按原始值逐元素比较（numpy 按字节语义）
        if _void_pair(self, other):
            a_flat = _flatten_vals(
                self._raw_data if self._raw_data is not None else self.tolist())
            b_flat = _flatten_vals(
                other._raw_data if other._raw_data is not None else other.tolist())
            if self.size == other.size and self.shape == other.shape:
                res = [x == y for x, y in zip(a_flat, b_flat)]
                out = ndarray(res)
                out._dtype = 'bool'
                return out.reshape(self.shape)
        # NEP50：低精度浮点数组 vs Python float 标量 → 标量先降到数组精度
        if self._dtype in ('float32', 'float16') and \
                isinstance(other, float) and not isinstance(other, bool):
            other = _round_scalar_to_f(self._dtype, other)
        if isinstance(other, str):
            flat_list = self.flatten().tolist()
            flat_result = [x == other for x in flat_list]
            result = ndarray(flat_result)
            result._dtype = "bool"
            return result.reshape(self.shape)
        if _is_ndarray(other):
            return _wrap_result(self._array.__eq__(other._array), "bool")
        if isinstance(other, complex):
            # Rust 比较不接受 complex（Python/rncomplex 标量）：先转为 0-d
            # 复数数组，再按 self 形状广播后比较。
            o = ndarray(other)
            if self.shape and o.shape != self.shape:
                o = ndarray._wrap(o._array.reshape(self.shape), _dtype=o._dtype)
            return _wrap_result(self._array.__eq__(o._array), "bool")
        return _wrap_result(self._array.__eq__(other), "bool")

    def __ne__(self, other):
        if _incomparable_0d(self, other):
            return bool_(True)
        if _void_pair(self, other):
            a_flat = _flatten_vals(
                self._raw_data if self._raw_data is not None else self.tolist())
            b_flat = _flatten_vals(
                other._raw_data if other._raw_data is not None else other.tolist())
            if self.size == other.size and self.shape == other.shape:
                res = [x != y for x, y in zip(a_flat, b_flat)]
                out = ndarray(res)
                out._dtype = 'bool'
                return out.reshape(self.shape)
        if self._dtype in ('float32', 'float16') and \
                isinstance(other, float) and not isinstance(other, bool):
            other = _round_scalar_to_f(self._dtype, other)
        if getattr(self, '_raw_data', None) is not None and \
                _is_ndarray(other) and getattr(other, '_raw_data', None) is not None:
            a_flat = _flatten_vals(self._raw_data)
            b_flat = _flatten_vals(other._raw_data)
            if self.size == other.size and self.shape == other.shape:
                res = [x != y for x, y in zip(a_flat, b_flat)]
                out = ndarray(res)
                out._dtype = 'bool'
                return out.reshape(self.shape)
        if _is_ndarray(other):
            return _wrap_result(self._array.__ne__(other._array), "bool")
        if isinstance(other, complex):
            o = ndarray(other)
            if self.shape and o.shape != self.shape:
                o = ndarray._wrap(o._array.reshape(self.shape), _dtype=o._dtype)
            return _wrap_result(self._array.__ne__(o._array), "bool")
        return _wrap_result(self._array.__ne__(other), "bool")

    def __lt__(self, other):
        if self._dtype in ('float32', 'float16') and \
                isinstance(other, float) and not isinstance(other, bool):
            other = _round_scalar_to_f(self._dtype, other)
        if _is_ndarray(other):
            return _wrap_result(self._array.__lt__(other._array), "bool")
        return _wrap_result(self._array.__lt__(other), "bool")

    def __le__(self, other):
        if self._dtype in ('float32', 'float16') and \
                isinstance(other, float) and not isinstance(other, bool):
            other = _round_scalar_to_f(self._dtype, other)
        if _is_ndarray(other):
            return _wrap_result(self._array.__le__(other._array), "bool")
        return _wrap_result(self._array.__le__(other), "bool")

    def __gt__(self, other):
        if self._dtype in ('float32', 'float16') and \
                isinstance(other, float) and not isinstance(other, bool):
            other = _round_scalar_to_f(self._dtype, other)
        if _is_ndarray(other):
            return _wrap_result(self._array.__gt__(other._array), "bool")
        return _wrap_result(self._array.__gt__(other), "bool")

    def __ge__(self, other):
        if self._dtype in ('float32', 'float16') and \
                isinstance(other, float) and not isinstance(other, bool):
            other = _round_scalar_to_f(self._dtype, other)
        if _is_ndarray(other):
            return _wrap_result(self._array.__ge__(other._array), "bool")
        return _wrap_result(self._array.__ge__(other), "bool")

    def __round__(self, ndigits=None):
        if ndigits is None:
            return _wrap_result(np.round(self._array, 0), self._dtype)
        return _wrap_result(np.round(self._array, ndigits), self._dtype)

    def __and__(self, other):
        """按位与（& 运算符）。"""
        return bitwise_and(self, other)

    def __or__(self, other):
        """按位或（| 运算符）。"""
        return bitwise_or(self, other)

    def __xor__(self, other):
        """按位异或（^ 运算符）。"""
        return bitwise_xor(self, other)

    def __rand__(self, other):
        """按位与（& 运算符）反向：标量 & ndarray。"""
        return bitwise_and(other, self)

    def __ror__(self, other):
        """按位或（| 运算符）反向：标量 | ndarray。"""
        return bitwise_or(other, self)

    def __rxor__(self, other):
        """按位异或（^ 运算符）反向：标量 ^ ndarray。"""
        return bitwise_xor(other, self)

    def __lshift__(self, other):
        """左移（<< 运算符）。"""
        return left_shift(self, other)

    def __rshift__(self, other):
        """右移（>> 运算符）。"""
        return right_shift(self, other)

    # ========== 属性 ==========

    @property
    def shape(self):
        """返回数组维度的元组。"""
        _ovr = getattr(self, '_rs_shape_override', None)
        if _ovr is not None:
            return tuple(_ovr)
        raw = getattr(self, '_raw_data', None)
        arr = getattr(self, '_array', None)
        if raw is not None and arr is not None:
            rust_shape = self._array.shape
            return tuple(int(s) if hasattr(s, '__index__') else s for s in rust_shape)
        if raw is not None:
            # 从嵌套列表递归计算形状
            shape = []
            obj = raw
            while isinstance(obj, list):
                shape.append(len(obj))
                if obj:
                    obj = obj[0]
                else:
                    break
            return tuple(shape)
        return self._array.shape

    @property
    def device(self):
        """返回数组所在的设备（rsnumpy 仅支持 CPU）。"""
        return 'cpu'

    @shape.setter
    def shape(self, new_shape):
        """设置数组形状，相当于原地 reshape。"""
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            total = len(raw)
            new_flat = list(raw)
            if isinstance(new_shape, int):
                new_shape = (new_shape,)
            new_size = 1
            for s in new_shape:
                new_size *= s
            if new_size != total:
                raise ValueError(f"cannot reshape array of size {total} into shape {new_shape}")
            # 扁平化 raw_data 并按新形状展平
            if len(new_shape) == 1:
                self._raw_data = new_flat
            else:
                self._raw_data = new_flat
        else:
            total = self._array.size
            if isinstance(new_shape, int):
                new_shape = (new_shape,)
            new_size = 1
            for s in new_shape:
                new_size *= s
            if new_size != total:
                raise ValueError(f"cannot reshape array of size {total} into shape {new_shape}")
            self._array = self._array.reshape(new_shape)

    @property
    def flags(self):
        """返回数组的内存布局信息，与 rsnumpy 的 ndarray.flags 兼容。"""
        return ArrayFlags(self)

    def setflags(self, write=None, align=None, warn=True):
        """numpy 兼容：设置可写标志（rsnumpy 仅记录标志位）。"""
        if write is not None:
            self._rs_writeable = bool(write)

    @property
    def base(self):
        """若数组是另一数组的视图则返回其基数组，否则返回 None。"""
        return getattr(self, '_base', None)

    @property
    def ndim(self):
        """返回数组维度数量。"""
        _ovr = getattr(self, '_rs_shape_override', None)
        if _ovr is not None:
            return len(_ovr)
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            return len(self.shape)
        return self._array.ndim

    @property
    def size(self):
        """返回元素总数。"""
        _ovr = getattr(self, '_rs_shape_override', None)
        if _ovr is not None:
            n = 1
            for v in _ovr:
                n *= int(v)
            return n
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            s = self.shape
            n = 1
            for v in s:
                n *= v
            return n
        return self._array.size

    @property
    def dtype(self):
        """返回元素数据类型。"""
        dobj = getattr(self, '_dtype_obj', None)
        if dobj is not None:
            return dobj
        sd = getattr(self, '_str_dtype', None)
        if sd is not None:
            kind, width = sd
            return _make_flexible(kind, width, None)
        return DType(getattr(self, '_dtype', "float64"))

    def __array_namespace__(self, api_version=None):
        """Array API 标准命名空间方法，返回 rsnumpy 模块自身。"""
        import rsnumpy as np
        return np

    @property
    def itemsize(self):
        """每个元素的字节大小。"""
        try:
            return self.dtype.itemsize
        except Exception:
            return 8

    @property
    def strides(self):
        """C 连续布局下各维度的字节步长。"""
        itemsize = self.itemsize
        shape = self.shape
        strides = []
        acc = itemsize
        for s in reversed(shape):
            strides.append(acc)
            acc *= s
        return tuple(reversed(strides))

    @property
    def nbytes(self):
        """总字节数。"""
        return self.size * self.itemsize

    @property
    def T(self):
        """数组的转置。"""
        return _ndarray_methods().transpose(self)

    @property
    def flat(self):
        """返回数组元素的扁平迭代器，逐个访问每个元素。"""
        return _NdFlatIter(self)

    @flat.setter
    def flat(self, values):
        """设置数组元素的扁平值。"""
        size = self.size
        flat_values = list(values)
        indices = np.ndarray(list(range(size)))
        vals = np.ndarray([float(v) for v in flat_values])
        self._array.put(indices, vals)

    @property
    def real(self):
        """数组的实部。"""
        result = self._array.real
        return _wrap_result(result, "float64")

    @property
    def imag(self):
        """数组的虚部（实数数组为全零）。"""
        return _wrap_result(self._array.imag, "float64")

    def conj(self):
        """返回数组的复共轭。"""
        if self._dtype == 'complex128':
            return _wrap_result(self._array.conj(), "complex128")
        return self.copy()

    # ========== 对象方法 ==========

    def astype(self, dtype, copy=True):
        """转换数组为指定类型。"""
        return _ndarray_methods().astype(self, dtype, copy=copy)

    def reshape(self, *shape, order='C', **kwargs):
        """改变数组形状。"""
        return _ndarray_methods().reshape(self, *shape)

    def resize(self, new_shape):
        """改变数组大小。"""
        _ndarray_methods().resize(self, new_shape)

    def ravel(self, order='C'):
        """展平数组。"""
        return _ndarray_methods().ravel(self, order)

    def flatten(self, order='C'):
        """返回数组的一维副本。"""
        return _ndarray_methods().flatten(self, order)

    def copy(self, order='K', thread_safe=False):
        """返回数组的副本。

        Args:
            order: 内存布局顺序（'C', 'F', 'A', 'K'）。
            thread_safe: 是否创建线程安全的副本（用于多线程环境）。

        Returns:
            ndarray: 数组副本。若 thread_safe=True，返回的副本可安全在多线程间共享。

        【线程安全提示】
        - thread_safe=True 会创建独立的数据副本，避免多线程竞态条件
        - 对于只读共享场景，建议使用此选项
        - 对于读写共享场景，使用 np.threadsafe_view() 代替

        【使用示例】
        >>> import rsnumpy as np
        >>> import threading
        >>> data = np.array([1, 2, 3])
        >>> # 线程安全副本
        >>> safe_copy = data.copy(thread_safe=True)
        """
        if getattr(self, '_rs_shape_override', None) is not None:
            _check_alloc_size(self.shape)  # 广播视图 copy = 物化 → 超限 MemoryError
        result = _wrap_result(self._array.copy(), self._dtype)
        # 携带字符串/object/结构化/复数元数据，保证 copy 后语义不变
        result._raw_data = getattr(self, '_raw_data', None)
        result._str_dtype = getattr(self, '_str_dtype', None)
        result._fields = getattr(self, '_fields', None)
        result._complex_data = getattr(self, '_complex_data', None)
        result._dtype_obj = getattr(self, '_dtype_obj', None)
        if thread_safe:
            # 标记为线程安全副本（文档性标记）
            result._thread_safe = True
        return result

    def transpose(self, *axes):
        """转置数组。"""
        return _ndarray_methods().transpose(self, *axes)

    def swapaxes(self, axis1, axis2):
        """交换两个轴。"""
        return _ndarray_methods().swapaxes(self, axis1, axis2)

    def squeeze(self, axis=None):
        """移除长度为1的轴。"""
        return _ndarray_methods().squeeze(self, axis)

    def max(self, axis=None, out=None, keepdims=False):
        """返回最大值。"""
        return _ndarray_methods().max(self, axis, out, keepdims)

    def min(self, axis=None, out=None, keepdims=False):
        """返回最小值。"""
        return _ndarray_methods().min(self, axis, out, keepdims)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """计算平均值。"""
        return _ndarray_methods().mean(self, axis, dtype, out, keepdims)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, **kwargs):
        """计算标准差。"""
        kwargs.pop('mean', None)
        kwargs.pop('where', None)
        kwargs.pop('correction', None)
        return _ndarray_methods().std(self, axis, dtype, out, ddof, keepdims)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, **kwargs):
        """计算方差。"""
        kwargs.pop('mean', None)
        kwargs.pop('where', None)
        return _ndarray_methods().var(self, axis, dtype, out, ddof, keepdims)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """计算和。"""
        return _ndarray_methods().sum(self, axis, dtype, out, keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        """计算乘积。"""
        return _ndarray_methods().prod(self, axis, dtype, out, keepdims)

    def cumsum(self, axis=None):
        """计算累积和。"""
        return _ndarray_methods().cumsum(self, axis)

    def cumprod(self, axis=None):
        """计算累积乘积。"""
        return _ndarray_methods().cumprod(self, axis)

    def argmax(self, axis=None):
        """返回最大值的索引。"""
        return _ndarray_methods().argmax(self, axis)

    def argmin(self, axis=None):
        """返回最小值的索引。"""
        return _ndarray_methods().argmin(self, axis)

    def argsort(self, axis=-1):
        """返回排序后的索引。"""
        return _ndarray_methods().argsort(self, axis)

    def sort(self, axis=-1):
        """原地排序。"""
        _ndarray_methods().sort(self, axis)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """返回对角线元素。"""
        return _ndarray_methods().diagonal(self, offset, axis1, axis2)

    def trace(self, offset=0, axis1=0, axis2=1):
        """计算迹。"""
        return _ndarray_methods().trace(self, offset, axis1, axis2)

    def dot(self, b):
        """矩阵/向量点积。"""
        return linalg.dot(self, b)

    def fill(self, value):
        """用值填充数组。"""
        _ndarray_methods().fill(self, value)

    def item(self, *args):
        """获取单个元素。"""
        return _ndarray_methods().item(self, *args)

    def __index__(self):
        """支持将 0 维数组用作整数索引或切片参数。"""
        if self.ndim == 0:
            val = self.item()
            if isinstance(val, float):
                if val != val:  # NaN check
                    return 0
                return int(val)
            if isinstance(val, builtins.bool):
                # bool is not a valid __index__ result (deprecated); coerce.
                return int(val)
            return val
        # Non-0-d arrays are NOT integer indices.  Returning a value here
        # would let pyo3 `extract::<isize>` hijack boolean-mask / fancy
        # index keys in setitem paths.
        raise TypeError(
            "only integer scalar arrays can be converted to a scalar index")

    def __float__(self):
        """支持将 0 维数组转换为 Python float。"""
        if self.ndim == 0:
            val = self.item()
            return float(val)
        # For non-0-d arrays, return the float of the first element
        val = self.ravel().tolist()[0]
        return float(val)

    def __int__(self):
        """支持将 0 维数组转换为 Python int。"""
        if self.ndim == 0:
            val = self.item()
            return int(val)
        # For non-0-d arrays, return the int of the first element
        val = self.ravel().tolist()[0]
        return int(val)

    def __complex__(self):
        """支持将 0 维数组转换为 Python complex。"""
        if self.ndim == 0:
            val = self.item()
            return complex(val)
        raise TypeError(f"only 0-dimensional arrays can be converted to scalars, got {self.ndim}D")

    def take(self, indices, axis=None):
        """根据索引取元素。"""
        return _ndarray_methods().take(self, indices, axis)

    def put(self, indices, values):
        """设置指定位置的元素。"""
        _ndarray_methods().put(self, indices, values)

    def repeat(self, repeats, axis=None):
        """重复元素。"""
        return _ndarray_methods().repeat(self, repeats, axis)

    def nonzero(self):
        """返回非零元素索引。"""
        return nonzero(self)

    # ========== 数学方法 ==========

    def sin(self):
        """计算正弦。"""
        return _math_funcs().sin(self)

    def cos(self):
        """计算余弦。"""
        return _math_funcs().cos(self)

    def tan(self):
        """计算正切。"""
        return _math_funcs().tan(self)

    def sqrt(self):
        """计算平方根。"""
        return _math_funcs().sqrt(self)

    def abs(self):
        """计算绝对值。"""
        return _math_funcs().abs(self)

    def exp(self):
        """计算指数。"""
        return _math_funcs().exp(self)

    def log(self):
        """计算自然对数。"""
        return _math_funcs().log(self)

    def log10(self):
        """计算以10为底的对数。"""
        return _math_funcs().log10(self)

    def log2(self):
        """计算以2为底的对数。"""
        return _math_funcs().log2(self)

    def clip(self, a_min=None, a_max=None, out=None, **kwargs):
        """限制值范围。"""
        # numpy 兼容别名：clip(min=..., max=...)
        if a_min is None and 'min' in kwargs:
            a_min = kwargs.pop('min')
        if a_max is None and 'max' in kwargs:
            a_max = kwargs.pop('max')
        kwargs.pop('like', None)
        casting = kwargs.get('casting')
        if casting is not None and casting not in ('no', 'same_kind', 'unsafe'):
            raise ValueError("casting must be one of no, same_kind, unsafe")
        if a_min is None and a_max is None:
            # numpy 2.x 语义：无界 = 不裁剪（模块级 clip 统一处理）
            res = self.copy()
            if out is not None:
                try:
                    out[...] = res
                except Exception:
                    pass
                return out
            return res
        _sdt = str(self._dtype)
        _complex_arr = _sdt.startswith('complex')
        # 复数界：实数数组先提升为 complex128（numpy 语义：结果按复数处理）
        if not _complex_arr and \
                builtins.any(_clip_bound_is_complex(b) for b in (a_min, a_max)):
            self = self + builtins.complex(0, 0)
            _complex_arr = True
        # numpy 语义：nan 界经 maximum/minimum 传播 → 结果全 nan
        if _clip_bound_is_nan(a_min) or _clip_bound_is_nan(a_max):
            _n = float('nan')
            if _complex_arr:
                res = ndarray([complex(_n, 0.0)] * self.size, _dtype='complex128')
            else:
                res = ndarray([_n] * self.size, _dtype='float64')
            if self.ndim > 1:
                res = res.reshape(self.shape)
            if out is not None:
                try:
                    out[...] = res
                except Exception:
                    pass
                return out
            return res
        # 复数：numpy 语义为 (实部, 虚部) 词典序比较，越界元素被界值整体
        # 替换（不保留原虚部）。Rust __lt__/__gt__ 不支持复数，故用模块级
        # less/greater ufunc（与测试参考实现同一语义）。
        if _complex_arr:
            _amin = self.min() if a_min is None else a_min
            _amax = self.max() if a_max is None else a_max
            _sel = less(self, _amin) + 2 * greater(self, _amax)
            res = _sel.choose([self, _amin, _amax])
            if out is not None:
                try:
                    out[...] = res
                except Exception:
                    pass
                return out
            return res
        # numpy 语义：casting=None/'same_kind' 时，结果 dtype 必须可安全写回 out
        if out is not None and casting in (None, 'same_kind'):
            _res_dt = _sdt
            _FW = {'float16': 2, 'float32': 4, 'float64': 8,
                   'complex64': 8, 'complex128': 16}
            for _b in (a_min, a_max):
                _bdt = _operand_dtype_name(_b)
                if _bdt is None:
                    continue
                # 浮点/复数界按实际位宽决定结果 dtype（int×float 混合不做
                # _promote_dtype 的 int 位宽提升）
                if (_FW.get(_bdt, 0) > _FW.get(_res_dt, 0) or
                        'complex' in _bdt):
                    _res_dt = _bdt
                elif _bdt in _INT_DTYPES and _res_dt in _INT_DTYPES:
                    if (_INT_ITEMSIZES.get(_bdt, 0) >
                            _INT_ITEMSIZES.get(_res_dt, 0)):
                        _res_dt = _bdt
            _odt = str(getattr(out, '_dtype', _sdt))
            if _res_dt != _odt and not _same_kind_cast_ok(_res_dt, _odt):
                raise TypeError(
                    f"Cannot cast ufunc 'clip' output from {_res_dt} to "
                    f"{_odt} with casting rule 'same_kind'")
        # 界为 None 时由模块级 clip 归一化（用数组自身 min/max）
        res = clip(self, a_min, a_max)
        if out is not None:
            try:
                out[...] = res
            except Exception:
                pass
            return out
        return res

    def choose(self, choices, out=None, mode='raise'):
        """按 self 的元素值从 choices 中逐元素选取（numpy ndarray.choose 兼容）。"""
        c_list = list(choices)
        try:
            idx = [int(v) for v in self._array.flatten().tolist()]
        except Exception:
            idx = [int(v) for v in _flatten_vals(self.tolist())]
        out_vals = []
        for i in range(self.size):
            k = idx[i] if i < len(idx) else 0
            if mode == 'raise':
                if k < 0 or k >= len(c_list):
                    raise IndexError(f"invalid choice {k} for choose")
            elif mode == 'clip':
                k = builtins.max(0, builtins.min(k, len(c_list) - 1))
            else:
                k %= len(c_list)
            c = c_list[k]
            if hasattr(c, '_array'):
                cf = _flatten_vals(c.tolist())
                out_vals.append(cf[i] if i < len(cf) else (cf[0] if cf else 0))
            else:
                out_vals.append(c)
        if self.size == 1 and not self.shape:
            res = array(out_vals[0])
        else:
            res = array(out_vals).reshape(self.shape)
        if out is not None:
            try:
                out[...] = res
            except Exception:
                pass
        return res

    def conjugate(self, out=None):
        """共轭（实数数组返回副本）。"""
        if 'complex' not in str(self._dtype):
            res = self.copy()
        else:
            flat = _flatten_vals(self.tolist())
            out_vals = [builtins.complex(v).conjugate() for v in flat]
            res = array(out_vals).reshape(self.shape) if self.shape else array(out_vals[0])
        if out is not None:
            try:
                out[...] = res
            except Exception:
                pass
        return res

    def round(self, decimals=0):
        """四舍五入到指定小数位。"""
        if decimals == 0:
            return _wrap_result(self._array.__round__(None), self._dtype)
        return _wrap_result(self._array.__round__(decimals), self._dtype)

    def floor(self):
        """向下取整。"""
        return _math_funcs().floor(self)

    def ceil(self):
        """向上取整。"""
        return _math_funcs().ceil(self)

    def all(self, axis=None, out=None, keepdims=False):
        """测试是否所有元素为真。"""
        return _extra_module.all(self, axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        """测试是否有任何元素为真。"""
        return _extra_module.any(self, axis=axis, out=out, keepdims=keepdims)

    def byteswap(self, inplace=False):
        """交换数组中每个元素的字节顺序（大小端转换）。"""
        data = self._array.flatten().tolist()
        dtype = getattr(self, '_dtype', 'float64')
        
        swapped = []
        for v in data:
            if isinstance(v, float):
                v = int(v)
            if dtype in ('int16', 'uint16'):
                swapped_val = ((v & 0xFF) << 8) | ((v >> 8) & 0xFF)
            elif dtype in ('int32', 'uint32'):
                swapped_val = ((v & 0xFF) << 24) | ((v >> 8 & 0xFF) << 16) | ((v >> 16 & 0xFF) << 8) | ((v >> 24) & 0xFF)
            elif dtype in ('int64', 'uint64'):
                swapped_val = 0
                for i in range(8):
                    swapped_val |= ((v >> (i * 8)) & 0xFF) << ((7 - i) * 8)
            else:
                swapped_val = v
            swapped.append(swapped_val)
        
        if inplace:
            arr = np.ndarray(swapped)
            arr = arr.reshape(self._array.shape)
            self._array = arr
            return self
        else:
            return ndarray(swapped, _dtype=dtype)

    def view(self, dtype=None, type=None):
        """创建一个新的数组对象，共享相同的数据但拥有不同的视图。"""
        # 注意：参数名 type 遮蔽了内建 type，判断必须用 builtins.type
        if dtype is not None:
            if isinstance(dtype, builtins.type) and issubclass(dtype, ndarray):
                # numpy 的 view(子类)：返回共享数据的该子类实例
                return dtype._wrap(self._array, _dtype=self._dtype)
            return _view_dtype(self, dtype)
        if type is not None:
            if isinstance(type, builtins.type) and issubclass(type, ndarray):
                return type._wrap(self._array, _dtype=self._dtype)
            return _view_dtype(self, type)
        return ndarray(self._array, _dtype=getattr(self, '_dtype', 'float64'))

    def __sizeof__(self):
        """返回数组占用的内存字节数（含 Python 对象头）。

        供 sys.getsizeof() 使用，便于内存使用分析和泄漏检测。
        """
        # 基础对象开销 + 底层 Rust 数组数据 + 原始数据（如果有）
        base = object.__sizeof__(self)
        data_bytes = self.nbytes
        raw_bytes = 0
        _raw = getattr(self, '_raw_data', None)
        if _raw is not None:
            import sys
            # 粗略估计原始数据列表的内存占用
            raw_bytes = sum(sys.getsizeof(item) for item in _raw[:100])
            if len(_raw) > 100:
                raw_bytes = raw_bytes * (len(_raw) / 100)
        return int(base + data_bytes + raw_bytes)

    def __reduce__(self):
        """支持 pickle 序列化。"""
        return (
            self.__class__,
            (),
            self.__getstate__(),
        )

    def __getstate__(self):
        """返回可序列化的状态元组。"""
        return (
            self.tolist(),
            getattr(self, '_dtype', 'float64'),
            getattr(self, '_fields', None),
            getattr(self, '_raw_data', None),
        )

    # pickle 反序列化时允许的合法 dtype 名称
    _PICKLE_VALID_DTYPES = frozenset({
        'float64', 'float32', 'float16',
        'int8', 'int16', 'int32', 'int64',
        'uint8', 'uint16', 'uint32', 'uint64',
        'bool', 'complex64', 'complex128',
        'object', 'datetime64', 'timedelta64',
        'string_', 'void', 'bytes_',
    })

    def __setstate__(self, state):
        """从 pickle 状态恢复对象，带安全校验。

        防止恶意 pickle 数据导致类型混淆或内存耗尽。
        """
        if not isinstance(state, tuple) or len(state) != 4:
            raise ValueError(
                f"pickle 状态无效：期望 4 元素元组，得到 {type(state).__name__}"
                f"（长度 {len(state) if hasattr(state, '__len__') else '?'}）"
            )
        data, _dtype, _fields, _raw_data = state

        # 校验 data：必须是列表/元组，且元素数不超过上限
        if not isinstance(data, (list, tuple)):
            raise ValueError(
                f"pickle 数据无效：data 必须是 list/tuple，得到 {type(data).__name__}"
            )
        _MAX_PICKLE_ELEMENTS = 2**31  # 21 亿元素上限
        if len(data) > _MAX_PICKLE_ELEMENTS:
            raise ValueError(
                f"pickle 数据过大：{len(data)} 个元素，超过上限 {_MAX_PICKLE_ELEMENTS}"
            )

        # 校验 _dtype：必须是已知 dtype 字符串
        if not isinstance(_dtype, str) or _dtype not in self._PICKLE_VALID_DTYPES:
            raise ValueError(
                f"pickle dtype 无效：{_dtype!r} 不在合法 dtype 集合中"
            )

        # 校验 _fields：None 或列表/元组
        if _fields is not None and not isinstance(_fields, (list, tuple)):
            raise ValueError(
                f"pickle _fields 无效：必须是 None 或 list/tuple，得到 {type(_fields).__name__}"
            )

        # 校验 _raw_data：None 或列表/元组
        if _raw_data is not None and not isinstance(_raw_data, (list, tuple)):
            raise ValueError(
                f"pickle _raw_data 无效：必须是 None 或 list/tuple，得到 {type(_raw_data).__name__}"
            )

        # 通过校验后，调用正常构造逻辑
        self.__init__(data, _dtype, _fields, _raw_data)


class recarray(ndarray):
    """结构化数组，支持以属性方式访问字段（numpy.recarray 兼容子集）。"""

    def __getattr__(self, name):
        dt = self.__dict__.get('_dtype_obj')
        if dt is not None and dt._names and name in dt._names:
            return self[name]
        raise AttributeError(
            "'recarray' object has no attribute %r" % name)

    def __getitem__(self, key):
        res = ndarray.__getitem__(self, key)
        if (isinstance(res, ndarray) and res.__class__ is ndarray and getattr(res, '_dtype_obj', None) is not None):
            res.__class__ = recarray
        return res


def _as_recarray(arr):
    """将结构化 ndarray 就地提升为 recarray。"""
    if isinstance(arr, ndarray) and getattr(arr, '_dtype_obj', None) is not None:
        arr.__class__ = recarray
    return arr


# ---------- 内部数组判定与结果包装 ----------

def _ensure(x):
    """将列表/元组转换为 ndarray。"""
    if isinstance(x, (list, tuple)):
        return np.ndarray(x)
    elif _is_ndarray(x):
        return x._array
    elif hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray':
        return x
    return x


def _py_index(seq, pred):
    """返回 seq 中第一个满足 pred 的元素下标（身份/谓词检查，不触发 __eq__）。"""
    for i, x in enumerate(seq):
        if pred(x):
            return i
    return -1


def _is_arr_like_el(e):
    """列表元素是否为数组类对象（ndarray/__array__/buffer/数组接口）。"""
    return (_is_ndarray(e) or hasattr(e, '__array__')
            or isinstance(e, memoryview)
            or (hasattr(e, '__array_interface__')
                and not isinstance(e, (list, tuple, str, bytes))))


def _is_ndarray(obj):
    """检查对象是否为 rsnumpy ndarray（用 hasattr 避免类身份不一致问题）。"""
    return hasattr(obj, '_array')


def _is_raw_ndarray(obj):
    """检查对象是否为底层 Rust 绑定 ndarray（builtins.ndarray，无 _array 属性）。"""
    return (
        hasattr(obj, 'shape')
        and hasattr(obj, 'dtype')
        and hasattr(obj, 'ndim')
        and hasattr(obj, 'tolist')
        and not hasattr(obj, '_array')
        # rsnumpy 标量也带 shape/dtype/ndim/tolist 属性，需排除（generic 为其公共基类）。
        and not isinstance(obj, generic)
    )


def _is_any_ndarray(obj):
    """检查对象是否为 rsnumpy ndarray 或底层 Rust 绑定 ndarray。"""
    return _is_ndarray(obj) or _is_raw_ndarray(obj)


def _wrap_result(result, dtype="float64"):
    """将原始 ndarray 结果包装到 ndarray 类中。"""
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        if getattr(result, 'is_complex', False):
            dtype = "complex128"
        inner_array = getattr(result, '_array', result)
        return ndarray._wrap(inner_array, _dtype=dtype)
    if isinstance(result, (list, tuple)):
        return ndarray(result, _dtype=dtype)
    # Handle scalar results from Rust (int, float, etc.)
    if isinstance(result, (int, float, complex)) and not isinstance(result, bool):
        return ndarray(result, _dtype=dtype)
    if hasattr(result, 'tolist'):
        return ndarray._wrap(result, _dtype=dtype)
    return result


def _scalar(x):
    """转换为标量。"""
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x


_FLOAT_DTYPES = ("float16", "float32", "float64")


def _is_float_dtype(dt):
    return dt in _FLOAT_DTYPES


def _operand_is_float(other):
    """判断算术运算的另一操作数是否为浮点（bool 视为整数，与 rsnumpy 一致）。"""
    if _is_ndarray(other):
        return _is_float_dtype(getattr(other, '_dtype', 'float64'))
    if isinstance(other, bool):
        return False
    return isinstance(other, float)


def _operand_is_complex(other):
    """判断算术运算的另一操作数是否为复数。"""
    if _is_ndarray(other):
        return getattr(other, '_dtype', 'float64') == 'complex128'
    return isinstance(other, complex)


def _promote_dtype(self_dtype, other):
    """按 rsnumpy 规则推导加/减/乘结果 dtype：整数遇到浮点操作数提升为 float64，
    实数遇到复数操作数提升为 complex128；bool 参与算术提升为 int64（numpy 语义）。"""
    if self_dtype == 'bool':
        if _operand_is_complex(other):
            return 'complex128'
        if _operand_is_float(other):
            return 'float64'
        return 'int64'
    if self_dtype == 'complex128':
        return 'complex128'
    if _operand_is_complex(other):
        return 'complex128'
    if _is_float_dtype(self_dtype):
        return self_dtype
    if _operand_is_float(other):
        return 'float64'
    return self_dtype


def _matmul_fallback(a, b):
    """Fallback for batch matrix multiplication when Rust matmul doesn't support the shapes."""
    a_shape = a.shape
    b_shape = b.shape
    
    if len(a_shape) == 3 and len(b_shape) == 3:
        batch_size = a_shape[0]
        result = empty((batch_size, a_shape[1], b_shape[2]), dtype=a._dtype)
        for i in range(batch_size):
            result[i] = a[i] @ b[i]
        return result
    
    if len(a_shape) == 3 and len(b_shape) == 2:
        batch_size = a_shape[0]
        a_rows, a_cols = a_shape[1], a_shape[2]
        b_rows, b_cols = b_shape[0], b_shape[1]
        
        if a_cols == b_rows:
            result = empty((batch_size, a_rows, b_cols), dtype=a._dtype)
            for i in range(batch_size):
                result[i] = a[i] @ b
            return result
        
        if batch_size == b_rows and a_cols == b_cols:
            result = empty((batch_size, a_rows, 1), dtype=a._dtype)
            for i in range(batch_size):
                ai = a[i]
                bi = b[i:i+1].T
                result[i] = ai @ bi
            return result
        
        if b_rows == 1 and a_cols == b_cols:
            result = empty((batch_size, a_rows, 1), dtype=a._dtype)
            for i in range(batch_size):
                for r in range(a_rows):
                    s = 0.0
                    for c in range(a_cols):
                        s += a[i, r, c] * b[0, c]
                    result[i, r, 0] = s
            return result
    
    if len(a_shape) == 2 and len(b_shape) == 3:
        batch_size = b_shape[0]
        result = empty((batch_size, a_shape[0], b_shape[2]), dtype=a._dtype)
        for i in range(batch_size):
            result[i] = a @ b[i]
        return result
    
    if len(a_shape) == 2 and len(b_shape) == 2:
        return np.linalg.matmul(a._array, b._array)
    
    raise ValueError(f"Unsupported shapes for matmul: {a_shape} and {b_shape}")


def _truediv_dtype(self_dtype):
    """真除法结果恒为浮点：整数提升为 float64，浮点保持自身精度。"""
    return self_dtype if _is_float_dtype(self_dtype) else 'float64'


# ---------- 嵌套数据与索引转换 ----------

def _convert_nested(data, converter):
    """递归转换嵌套列表中的每个元素。"""
    if isinstance(data, list):
        return [_convert_nested(x, converter) for x in data]
    return converter(data)


def _ndarray_to_index_list(k):
    """将 ndarray 索引转为 Rust 可解析的 Python list，保留多维嵌套结构。"""
    raw = k._array.tolist()  # preserve shape: [[1],[5],[7],[2]] or [1,0,1,0,1]
    dtype_name = getattr(k, '_dtype', 'float64')

    def _convert(v):
        if isinstance(v, (list, tuple)):
            return [_convert(x) for x in v]
        if dtype_name == 'bool':
            return builtins.bool(v)
        return int(v)

    return _convert(raw)


def _empty_bool_mask_key(key):
    """检测布尔掩码索引是否选中零个元素；numpy 语义下对空选择赋值为 no-op。"""
    items = key if isinstance(key, tuple) else (key,)

    def _has_true(v):
        if isinstance(v, (list, tuple)):
            return _py_any(_has_true(x) for x in v)
        return builtins.bool(v)

    for k in items:
        if hasattr(k, '_array') and getattr(k, '_dtype', None) == 'bool':
            if not _has_true(k._array.tolist()):
                return True
    return False


def _nested_zeros(shape):
    """生成给定形状的嵌套零列表（标量形状返回 0）。"""
    if not shape:
        return 0
    return [_nested_zeros(shape[1:]) for _ in range(shape[0])]


# ---------- 结构化字段规格与视图 ----------

def _normalize_field(item):
    """归一化结构化字段规格：子数组字段保留形状为 (name, type, subshape)。"""
    if len(item) >= 3 and item[2]:
        sub = item[2] if isinstance(item[2], (list, tuple)) else (item[2],)
        return (item[0], item[1], tuple(sub))
    return (item[0], item[1])


def _field_subshape(fspec):
    """返回字段的子数组形状（标量字段为空元组）。"""
    if len(fspec) >= 3 and fspec[2]:
        return tuple(fspec[2])
    return ()


def _field_dtype_name(ftype):
    """将字段类型规格解析为本库 dtype 名称（字符串字段归一为 string_）。"""
    name = _resolve_type_name(ftype)
    if isinstance(name, str) and name and name[0] in ('S', 'a', 'U'):
        return 'string_'
    return name


def _structured_field_kind(code):
    """判断结构化字段类型：'int' / 'float' / 'str'。"""
    if not isinstance(code, str):
        code = _resolve_type_name(code) if code else 'f8'
    if not isinstance(code, str):
        return 'float'
    c = code.lstrip('<>=|')
    first = c[:1]
    if first in ('S', 'U', 'a'):
        return 'str'
    low = c.lower()
    if low.startswith('bytes') or low.startswith('str'):
        return 'str'
    is_int = _py_any((
        low.startswith('int'),
        low.startswith('uint'),
        low.startswith('bool'),
        first == 'i',
        first == 'u' and c[1:2].isdigit(),
    ))
    if is_int:
        return 'int'
    return 'float'


def _extract_field(data, fi, depth):
    """从嵌套结构化 raw_data 中递归抽取第 fi 个字段。depth 为外层维度数。"""
    if depth <= 0:
        return data[fi]
    return [_extract_field(row, fi, depth - 1) for row in data]


def _select_fields(data, idxs, depth):
    """从嵌套结构化 raw_data 中递归保留 idxs 指定的若干字段。"""
    if depth <= 0:
        return tuple(data[i] for i in idxs)
    return [_select_fields(row, idxs, depth - 1) for row in data]


def _structured_field_view(arr, name):
    """结构化数组的单字段访问：返回与 arr 同形（子数组字段追加子形状）的字段数组。"""
    fields = arr._fields
    names = [f[0] for f in fields]
    fi = names.index(name)
    fspec = fields[fi]
    dt = _field_dtype_name(fspec[1])
    raw = getattr(arr, '_raw_data', None)
    if raw is None:
        # 单字段数组：数据已展平进底层 _array
        if dt == 'string_':
            return ndarray._wrap(arr._array, _dtype=dt, _raw_data=getattr(arr, '_raw_data', None))
        return ndarray._wrap(arr._array, _dtype=dt)
    extracted = _extract_field(raw, fi, arr.ndim)
    if dt == 'string_':
        return ndarray(extracted)
    return ndarray(extracted, _dtype=dt)


def _structured_multifield_view(arr, keys):
    """结构化数组的多字段访问 x[['f1','f2']]：返回仅含所选字段的新结构化数组。"""
    fields = arr._fields
    names = [f[0] for f in fields]
    idxs = [names.index(k) for k in keys]
    new_fields = [fields[i] for i in idxs]
    raw = getattr(arr, '_raw_data', None)
    if raw is None:
        return ndarray._wrap(arr._array, _dtype='void', _fields=new_fields)
    new_raw = _select_fields(raw, idxs, arr.ndim)
    return ndarray._wrap(arr._array, _dtype='void', _fields=new_fields, _raw_data=new_raw)


# ---------- 结构化数组格式化 ----------

def _fmt_structured_str_val(v):
    if isinstance(v, bytes):
        return "b'" + v.decode("utf-8", errors="replace") + "'"
    return "b'" + str(v) + "'"


def _trim_positional_float(v):
    """numpy dragon4 positional (unique, trim='.') 的最短浮点表示。"""
    s = repr(float(v))
    if 'e' in s or 'E' in s:
        return s
    if '.' not in s:
        return s + '.'
    intp, frac = s.split('.', 1)
    return intp + '.' + frac.rstrip('0')


def _format_float_field_column(col):
    """按 rsnumpy 浮点列规则将一列浮点值格式化为等宽对齐的字符串。"""
    n = len(col)
    finite = [None] * n
    special = [None] * n
    for i, v in enumerate(col):
        fv = float(v)
        if fv != fv:
            special[i] = 'nan'
        elif fv == float('inf'):
            special[i] = 'inf'
        elif fv == float('-inf'):
            special[i] = '-inf'
        else:
            finite[i] = _trim_positional_float(fv)
    out = [None] * n
    pad_left = 0
    pad_right = 0
    for s in finite:
        if s is not None and 'e' not in s:
            intp, frac = s.split('.', 1)
            pad_left = _py_max(pad_left, len(intp))
            pad_right = _py_max(pad_right, len(frac))
    for i, s in enumerate(finite):
        if s is None or 'e' in s:
            continue
        intp, frac = s.split('.', 1)
        out[i] = intp.rjust(pad_left) + '.' + frac.ljust(pad_right)
    width = 0
    for i in range(n):
        s = out[i] if out[i] is not None else (finite[i] if finite[i] is not None else special[i])
        if s is not None:
            width = _py_max(width, len(s))
    for i in range(n):
        if out[i] is None:
            out[i] = (finite[i] if finite[i] is not None else special[i])
        out[i] = out[i].rjust(width)
    return out


def _format_int_field_column(col):
    strs = [str(v) if isinstance(v, str) else str(int(_py_round(float(v)))) for v in col]
    width = _py_max((len(s) for s in strs), default=0)
    return [s.rjust(width) for s in strs]


def _format_structured_field_column(col, code):
    kind = _structured_field_kind(code)
    if kind == 'str':
        return [_fmt_structured_str_val(v) for v in col]
    if kind == 'int':
        return _format_int_field_column(col)
    return _format_float_field_column(col)


def _format_structured_str(arr):
    """__str__ 用于结构化数组：基于富 DType 递归格式化，支持嵌套/子数组。"""
    dt = getattr(arr, '_dtype_obj', None)
    if dt is None:
        return _format_structured_str_legacy(arr)

    def build(o):
        if isinstance(o, tuple):
            return _fmt_record(o, dt)
        return "[" + " ".join(build(x) for x in o) + "]"

    raw = arr._raw_data
    if arr.ndim == 0:
        return _fmt_record(raw, dt)
    return build(raw)


def _format_structured_str_legacy(arr):
    """旧版结构化数组格式化（基于 _fields codes），保留作回退。"""
    fields = getattr(arr, '_fields', None)
    data = arr.tolist()
    ndim = arr.ndim
    nfields = len(fields)
    codes = [f[1] for f in fields]

    leaves = []
    if ndim <= 1:
        leaves = list(data)
    else:
        def _collect(d, depth):
            if depth == ndim - 1:
                leaves.extend(d)
            else:
                for sub in d:
                    _collect(sub, depth + 1)
        _collect(data, 0)
    norm = [t if isinstance(t, (list, tuple)) else (t,) for t in leaves]

    col_strs = []
    for j in range(nfields):
        col_strs.append(_format_structured_field_column([r[j] for r in norm], codes[j]))

    tuple_strs = []
    for i in range(len(norm)):
        parts = [col_strs[j][i] for j in range(nfields)]
        if nfields == 1:
            tuple_strs.append("(" + parts[0] + ",)")
        else:
            tuple_strs.append("(" + ", ".join(parts) + ")")

    if ndim <= 1:
        return "[" + " ".join(tuple_strs) + "]"

    shape = arr.shape
    pos = [0]

    def _build(dim):
        if dim == ndim - 1:
            chunk = tuple_strs[pos[0]:pos[0] + shape[dim]]
            pos[0] += shape[dim]
            return "[" + " ".join(chunk) + "]"
        return "[" + "\n ".join(_build(dim + 1) for _ in range(shape[dim])) + "]"

    return _build(0)


def _format_structured_repr(arr):
    """__repr__ 用于结构化数组。"""
    inner = _format_structured_str(arr)
    return f"array({inner})"


# ---------- 不规则/嵌套数组格式化 ----------

def _format_ragged_str(arr):
    """__str__ 用于不规则数组（包含列表/元组元素）。"""
    data = arr.tolist()
    return _format_nested_iterable(data)


def _format_ragged_repr(arr):
    """__repr__ 用于不规则数组。"""
    inner = _format_ragged_str(arr)
    return f"array({inner})"


def _format_nested_iterable(data):
    """递归格式化嵌套可迭代对象为数组字符串。"""
    if isinstance(data, (list, tuple)):
        if not data:
            return "[]"
        items = list(data)
        has_nested = _py_any(isinstance(x, (list, tuple)) for x in items)
        if not has_nested:
            return "[" + " ".join(str(x) for x in items) + "]"
        parts = [_format_nested_iterable(x) for x in items]
        return "[" + "\n ".join(parts) + "]"
    return str(data)


# ---------- 浮点/复数标量格式化 ----------

class _float64:
    """float64 标量，显示为 np.float64(value)。"""
    def __init__(self, value):
        self._value = float(value)

    def __repr__(self):
        return f"np.float64({self._value})"

    def __str__(self):
        return repr(self)

    def __float__(self):
        return self._value


def _format_float_repr_1d(values):
    """格式化 1D float64 数组的 repr，带逗号和对齐（匹配 rsnumpy）。"""
    fmt = [format_float_scalar(v) for v in values]
    max_w = np.imax([len(f) for f in fmt]) if fmt else 0
    parts = [f.rjust(max_w) for f in fmt]
    return "[" + ", ".join(parts) + "]"


def _format_complex_repr_1d(values):
    """格式化 1D complex128 数组的 repr。"""
    parts = [_format_complex_scalar(v) for v in values]
    return "[" + ", ".join(parts) + "]"


def _format_complex_nested(data, sep):
    """递归格式化原生复数数组（nested complex）为 rsnumpy 风格字符串。"""
    if isinstance(data, list):
        inner = sep.join(_format_complex_nested(x, sep) for x in data)
        return "[" + inner + "]"
    return _format_complex_scalar(data)


def _format_complex_scalar(val):
    """格式化单个复数为字符串（如 1.+0.j, 2.+6.j）。"""
    real = val.real
    imag = val.imag
    if imag >= 0:
        return f"({real}{imag:+}j)"
    return f"({real}{imag}j)"


def format_float_scalar(val):
    """Python 版的浮点数格式化（四舍五入，保留有效小数）。"""
    if val != val:
        return "nan"
    if val == float("inf"):
        return "inf"
    if val == float("-inf"):
        return "-inf"
    if val == 0:
        return "-0." if str(val)[0] == '-' else "0."
    val_rounded = _py_round(val, 8)
    if val_rounded == int(val_rounded) and abs(val_rounded) < 1e16:
        v = int(val_rounded)
        if float(v) == val_rounded:
            return f"{v}."
    if abs(val_rounded) >= 1e10 or (abs(val_rounded) < 1e-10 and val_rounded != 0):
        return f"{val_rounded:.10e}"
    s = f"{val_rounded:.8f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


# ---------- 子模块延迟访问 ----------

def _ndarray_methods():
    """延迟导入 array_methods 模块。"""
    return array_methods.NdArrayMethods


def _math_funcs():
    """延迟导入 math_functions 模块。"""
    from . import math_functions as mf
    return mf


def _stat_funcs():
    """延迟导入 statistics 模块。"""
    from . import statistics as st
    return st


def _array_ops():
    """延迟导入 array_ops 模块。"""
    from . import array_ops as ao
    return ao


# ========== 类型代码映射 ==========

_RS_DTYPE_CODES = {
    # 有符号整数
    'i1': 'int8', 'int8': 'int8',
    'i2': 'int16', 'int16': 'int16',
    'i4': 'int32', 'int32': 'int32',
    'i8': 'int64', 'int64': 'int64',
    # 无符号整数
    'u1': 'uint8', 'uint8': 'uint8',
    'u2': 'uint16', 'uint16': 'uint16',
    'u4': 'uint32', 'uint32': 'uint32',
    'u8': 'uint64', 'uint64': 'uint64',
    # 浮点
    'f2': 'float16', 'float16': 'float16',
    'f4': 'float32', 'float32': 'float32',
    'f8': 'float64', 'float64': 'float64', 'float': 'float64',
    # 复数
    'c8': 'complex64', 'c64': 'complex64', 'complex64': 'complex64',
    'c16': 'complex128', 'c128': 'complex128', 'complex128': 'complex128', 'complex': 'complex128',
    # 其他
    'bool': 'bool', 'b1': 'bool',
    'object': 'object', 'O': 'object',
}

# NumPy's single-character type codes, as produced by `dtype.char` and accepted
# by NumPy APIs such as `finfo('d')`.  Deliberately kept out of
# `_RS_DTYPE_CODES`, whose lookups are case-insensitive: these codes are not,
# since 'd' means float64 while 'D' means complex128.
_DTYPE_CHAR_CODES = {
    '?': 'bool',
    'b': 'int8', 'B': 'uint8',
    'h': 'int16', 'H': 'uint16',
    'i': 'int32', 'I': 'uint32',
    # NumPy sizes 'l'/'L' by platform long width; 64-bit everywhere we target.
    'l': 'int64', 'L': 'uint64', 'q': 'int64', 'Q': 'uint64',
    'e': 'float16', 'f': 'float32', 'd': 'float64',
    # 'g' is longdouble, which rsnumpy has no storage for; float64 is the
    # documented NumPy fallback width on this platform.
    'g': 'float64',
    'F': 'complex64', 'D': 'complex128', 'G': 'complex128',
    'O': 'object',
}

_PY_TYPE_TO_RS = {
    int: 'int64',
    float: 'float64',
    complex: 'complex128',
    bool: 'bool',
    bytes: 'bytes',
    str: 'str',
}

# 运行时收集已知的 rsnumpy 类型名称
_numpy_type_names = {}


def _init_numpy_types():
    """尝试导入 rsnumpy 并注册其标量类型名称。"""
    try:
        import rsnnumpy as np
    except ImportError:
        return
    for name in dir(np):
        if name.startswith('int') or name.startswith('uint') or name.startswith('float') \
                or name.startswith('complex') or name == 'bool_' or name == 'bool':
            _numpy_type_names[name] = name


def _resolve_type_name(tp):
    """将各种 dtype 输入解析为类型名字符串。"""
    if isinstance(tp, str):
        # Single-character codes must be resolved before `lower()` sees them:
        # 'd' is float64 but 'D' is complex128.
        if len(tp) == 1 and tp in _DTYPE_CHAR_CODES:
            return _DTYPE_CHAR_CODES[tp]
        # 直接映射
        lower = tp.lower()
        if lower in _RS_DTYPE_CODES:
            return _RS_DTYPE_CODES[lower]
        return tp  # 未知字符串，原样返回
    if isinstance(tp, DType):
        return tp.name
    if type(tp).__name__ == 'dtype' and hasattr(tp, 'name'):
        return _RS_DTYPE_CODES.get(tp.name, tp.name)
    # Python 内置类型
    if tp in _PY_TYPE_TO_RS:
        return _PY_TYPE_TO_RS[tp]
    # 检查类名
    name = getattr(tp, '__name__', None) or getattr(tp, 'name', str(tp))
    # 先查已知的 rsnumpy 类型名
    if name in _numpy_type_names:
        return _numpy_type_names[name]
    # 再查代码表
    if name in _RS_DTYPE_CODES:
        return _RS_DTYPE_CODES[name]
    # int_ / float_ 等 rsnumpy 特殊名称
    name_clean = name.rstrip('_')
    if name_clean in _RS_DTYPE_CODES:
        return _RS_DTYPE_CODES[name_clean]
    return name


# 类型名 → 类型码 逆向映射（用于结构化 dtype）
_RS_NAME_TO_CODE = {}
for _code, _name in _RS_DTYPE_CODES.items():
    # 只保留短码映射，忽略 'int32': 'int32' 这样的自映射
    if len(_code) <= 4 and _code not in ('float', 'complex', 'bool') and _code != _name:
        _RS_NAME_TO_CODE[_name] = _code


# 立即初始化 rsnumpy 类型名称
_init_numpy_types()


def _legacy_fields_from_dtype(dt):
    """从富 DType 结构化类型构建旧版 _fields 列表：(name, code[, subshape])。

    仅用于与依赖旧格式的模块（io/statistics/array_ops）保持兼容；
    结构化格式化与索引优先使用 _dtype_obj。嵌套结构字段存 DType 本身。
    """
    if dt is None or dt._names is None:
        return None
    out = []
    for n in dt._names:
        fdt = dt._fields[n][0]
        if fdt._subdtype is not None:
            base, shape = fdt._subdtype
            out.append((n, _scalar_typestr_short(base), tuple(shape)))
        elif fdt._names is not None:
            out.append((n, fdt))
        else:
            out.append((n, _scalar_typestr_short(fdt)))
    return out


def _struct_dt_of(arr):
    """返回数组的结构化富 DType：优先 _dtype_obj，否则从旧版 _fields 重建。"""
    dt = getattr(arr, '_dtype_obj', None)
    if dt is not None:
        return dt
    fields = getattr(arr, '_fields', None)
    if not fields:
        return None
    names, formats = [], []
    for f in fields:
        names.append(f[0])
        typ = f[1]
        fdt = typ if isinstance(typ, DType) else dtype(typ)
        if len(f) > 2 and f[2]:
            sub = f[2] if isinstance(f[2], (list, tuple)) else (f[2],)
            fdt = _make_subarray(fdt, tuple(sub))
        formats.append(fdt)
    return _build_struct(names, formats, None, None, None, False)


# ========== 结构化数组统一数据模型 ==========
# 结构化数组统一存储：_dtype_obj（富 DType，语义真源）、_fields（旧版列表，供
# 既有格式化/视图代码）、_raw_data（Python 记录嵌套列表）、_array（占位取形状）。

def _struct_shape(raw):
    """从结构化 raw_data 的嵌套列表推断形状（记录为 tuple，视作叶子）。"""
    shape = []
    obj = raw
    while isinstance(obj, list):
        shape.append(len(obj))
        obj = obj[0] if obj else None
    return tuple(shape)


def _zero_scalar_for(fdt):
    k = fdt._kind
    if k in ('i', 'u', 'b'):
        return 0
    if k == 'f':
        return 0.0
    if k == 'c':
        return 0j
    if k == 'S':
        return b''
    if k == 'U':
        return ''
    if k == 'V':
        return b'\x00' * fdt._itemsize
    return 0


def _one_scalar_for(fdt):
    k = fdt._kind
    if k in ('i', 'u', 'b'):
        return 1
    if k == 'f':
        return 1.0
    if k == 'c':
        return 1 + 0j
    if k == 'S':
        return b'1'
    if k == 'U':
        return '1'
    return _zero_scalar_for(fdt)


def _coerce_scalar_by_dt(v, fdt):
    k = fdt._kind
    if k in ('i', 'u'):
        return int(v)
    if k == 'b':
        return builtins.bool(v)
    if k == 'f':
        return float(v)
    if k == 'c':
        return complex(v)
    if k == 'S':
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        return str(v).encode('latin-1')
    if k == 'U':
        if isinstance(v, (bytes, bytearray)):
            return bytes(v).decode('latin-1')
        return str(v)
    return v


def _coerce_field_value(val, fdt):
    if fdt._names is not None:
        seq = tuple(val) if isinstance(val, (list, tuple)) else (val,)
        return tuple(_coerce_field_value(seq[i], fdt._fields[n][0])
                     for i, n in enumerate(fdt._names))
    if fdt._subdtype is not None:
        base, shape = fdt._subdtype
        return _coerce_subarray(val, base, shape)
    return _coerce_scalar_by_dt(val, fdt)


def _coerce_subarray(val, base, shape):
    if not shape:
        return _coerce_field_value(val, base)
    if not isinstance(val, (list, tuple)):
        return [_coerce_subarray(val, base, shape[1:]) for _ in range(shape[0])]
    return [_coerce_subarray(val[i], base, shape[1:]) for i in range(shape[0])]


def _coerce_record(rec, dt):
    seq = tuple(rec) if isinstance(rec, (list, tuple)) else (rec,)
    return tuple(_coerce_field_value(seq[i], dt._fields[n][0])
                 for i, n in enumerate(dt._names))


def _coerce_struct_data(data, dt):
    if isinstance(data, tuple):
        return _coerce_record(data, dt)
    if isinstance(data, list):
        return [_coerce_struct_data(x, dt) for x in data]
    # 标量广播到单字段记录
    return _coerce_record((data,), dt)


def _default_field(fdt, filler):
    if fdt._names is not None:
        return tuple(_default_field(fdt._fields[n][0], filler) for n in fdt._names)
    if fdt._subdtype is not None:
        base, shape = fdt._subdtype
        return _default_subarray(base, shape, filler)
    return filler(fdt)


def _default_subarray(base, shape, filler):
    if not shape:
        return _default_field(base, filler)
    return [_default_subarray(base, shape[1:], filler) for _ in range(shape[0])]


def _default_record(dt, filler):
    return tuple(_default_field(dt._fields[n][0], filler) for n in dt._names)


def _make_struct_filled(shape, dt, filler):
    rec = _default_record(dt, filler)
    if isinstance(shape, int):
        shape = (shape,)
    total = 1
    for s in shape:
        total *= s
    flat = [rec for _ in range(total)]
    if len(shape) <= 1:
        return flat
    result = flat
    for dim in reversed(shape[1:]):
        result = [result[i:i + dim] for i in range(0, len(result), dim)]
    return result


def _wrap_structured(dt, raw_data):
    """用富 DType 与已归一的 raw_data 构造结构化 ndarray。"""
    raw_data = list(raw_data)
    shape = _struct_shape(raw_data)
    obj = ndarray.__new__(ndarray)
    obj._array = np.zeros(shape if shape else (0,))
    obj._dtype = 'void'
    obj._dtype_obj = dt
    obj._fields = _legacy_fields_from_dtype(dt)
    obj._raw_data = raw_data
    return obj


def _make_structured_array(data, dt):
    """从任意输入数据 + 富结构化 DType 构造 ndarray。"""
    raw = _coerce_struct_data(list(data) if isinstance(data, (list, tuple)) else data, dt)
    if not isinstance(raw, list):
        raw = [raw]
    return _wrap_structured(dt, raw)


def _as_struct_dtype(dt):
    """若 dt 可解析为结构化 DType 则返回富 DType，否则 None。"""
    if dt is None:
        return None
    try:
        d = dt if isinstance(dt, DType) else dtype(dt)
    except Exception:
        return None
    if isinstance(d, DType) and d._names is not None:
        return d
    return None


# ========== 结构化数组操作（索引/赋值/视图/比较/格式化） ==========

def _scalar_name_of(fdt):
    """标量 DType → 内部 dtype 名称（int32/float64/...）。"""
    if fdt._typename is not None:
        return fdt._typename
    return fdt.name


def _flatten_records(raw):
    """将结构化 raw_data（嵌套列表，记录为 tuple 叶子）展平为记录列表（行主序）。"""
    out = []

    def rec(o):
        if isinstance(o, tuple):
            out.append(o)
        elif isinstance(o, list):
            for x in o:
                rec(x)
        else:
            out.append(o)

    rec(raw)
    return out


def _reshape_flat(flat, shape):
    """将扁平列表按 shape 重塑为嵌套列表（叶子视为原子）。"""
    flat = list(flat)
    if not shape or len(shape) <= 1:
        return flat
    result = flat
    for dim in reversed(shape[1:]):
        result = [result[i:i + dim] for i in range(0, len(result), dim)]
    return result


def _flat_to_multi(flat, shape):
    idx = []
    for s in reversed(shape):
        idx.append(flat % s)
        flat //= s
    return tuple(reversed(idx))


def _get_flat_record(arr, flat):
    idx = _flat_to_multi(flat, arr.shape)
    o = arr._raw_data
    for i in idx:
        o = o[i]
    return o


def _set_flat_record(arr, flat, new_rec):
    shape = arr.shape
    if len(shape) <= 1:
        arr._raw_data[flat] = new_rec
        return
    idx = _flat_to_multi(flat, shape)
    o = arr._raw_data
    for i in idx[:-1]:
        o = o[i]
    o[idx[-1]] = new_rec


def _build_plain_array(nested, base_dt, shape):
    """从嵌套列表 + 标量基础 DType 构造非结构化 ndarray。"""
    k = base_dt._kind
    if k in ('S', 'U', 'V'):
        flat = _flatten_data(nested) if isinstance(nested, list) else [nested]
        arr = ndarray._wrap(np.zeros((len(flat),) if flat else (0,)),
                            _dtype='string_', _raw_data=list(flat))
        width = base_dt._itemsize // 4 if k == 'U' else base_dt._itemsize
        arr._str_dtype = (k, width)
        return arr
    name = _scalar_name_of(base_dt)
    if k == 'c':
        flat = _flatten_data(nested) if isinstance(nested, list) else [nested]
        cflat = [complex(v) for v in flat]
        if not cflat:
            return ndarray._wrap(np.zeros(tuple(shape) if shape else (0,)),
                                 _dtype='complex128')
        nested_c = _reshape_flat(cflat, shape) if shape else cflat
        return ndarray._wrap(np.ndarray(nested_c), _dtype='complex128')
    return ndarray(nested, _dtype=name)


def _field_values_to_array(values, fdt, outer):
    """把每条记录的某字段值列表按字段 DType 组装成 ndarray。"""
    if fdt._names is not None:
        return _wrap_structured(fdt, _reshape_flat(values, outer))
    if fdt._subdtype is not None:
        base, sub = fdt._subdtype
        nested = _reshape_flat(values, outer)
        return _build_plain_array(nested, base, tuple(outer) + tuple(sub))
    nested = _reshape_flat(values, outer)
    return _build_plain_array(nested, fdt, tuple(outer))


def _get_field(arr, name):
    """结构化数组的单字段访问 a['f']。"""
    dt = _struct_dt_of(arr)
    fi = dt._names.index(name)
    fdt = dt._fields[name][0]
    recs = _flatten_records(arr._raw_data)
    values = [r[fi] for r in recs]
    res = _field_values_to_array(values, fdt, arr.shape)
    # numpy 语义：字段字节偏移未按字段宽度对齐时 flags.aligned 为 False
    try:
        off = dt._fields[name][1]
        fsz = getattr(fdt, '_itemsize', None) or getattr(fdt, 'itemsize', 1)
        if fsz and off % int(fsz) != 0:
            res._rs_unaligned = True
    except Exception:
        pass
    return res


def _get_multifield(arr, keys):
    """多字段访问 a[['f1','f2']]：返回保留原偏移/itemsize 的结构化视图。"""
    dt = _struct_dt_of(arr)
    newdt = dt[list(keys)]
    idxs = [dt._names.index(k) for k in keys]
    recs = _flatten_records(arr._raw_data)
    new_recs = [tuple(r[i] for i in idxs) for r in recs]
    return _wrap_structured(newdt, _reshape_flat(new_recs, arr.shape))


def _field_leaf_to_scalar(v, fdt):
    """将记录中单个字段值转换为对外标量/数组。"""
    if fdt._names is not None:
        return _make_record(v, fdt, None, None)
    if fdt._subdtype is not None:
        base, sub = fdt._subdtype
        return _build_plain_array(v, base, tuple(sub))
    k = fdt._kind
    if k == 'S':
        return _str_scalar(v, ('S', fdt._itemsize))
    if k == 'U':
        return _str_scalar(v, ('U', fdt._itemsize // 4))
    if k == 'V':
        return _str_scalar(v, ('V', fdt._itemsize))
    return v


def _make_record(values, dt, parent, flat_index):
    return _Record(list(values), dt, parent, flat_index)


def _get_record(arr, key):
    dt = _struct_dt_of(arr)
    recs = _flatten_records(arr._raw_data)
    n = len(recs)
    k = key + n if key < 0 else key
    return _make_record(recs[k], dt, arr, k)


def _struct_getitem(arr, key):
    """结构化数组索引分发，返回 (handled, result)。"""
    dt = _struct_dt_of(arr)
    if dt is None:
        return (False, None)
    if isinstance(key, str):
        if dt._names and key in dt._names:
            return (True, _get_field(arr, key))
        return (False, None)
    if isinstance(key, list) and key and _py_all(isinstance(k, str) for k in key):
        return (True, _get_multifield(arr, key))
    if isinstance(key, int) and not isinstance(key, bool):
        if arr.ndim <= 1:
            return (True, _get_record(arr, key))
        sub = arr._raw_data[key]
        return (True, _wrap_structured(dt, sub if isinstance(sub, list) else [sub]))
    if isinstance(key, slice):
        return (True, _wrap_structured(dt, arr._raw_data[key]))
    return (False, None)


def _broadcast_field_values(value, n):
    if _is_ndarray(value):
        value = value.tolist()
    if isinstance(value, list) and len(value) == n:
        return value
    if isinstance(value, tuple) and len(value) == n:
        return list(value)
    return [value] * n


def _set_field(arr, name, value):
    dt = _struct_dt_of(arr)
    fi = dt._names.index(name)
    fdt = dt._fields[name][0]
    recs = _flatten_records(arr._raw_data)
    vals = _broadcast_field_values(value, len(recs))
    new = []
    for r, v in zip(recs, vals):
        lst = list(r)
        lst[fi] = _coerce_field_value(v, fdt)
        new.append(tuple(lst))
    arr._raw_data = _reshape_flat(new, arr.shape)


def _set_multifield(arr, keys, value):
    dt = _struct_dt_of(arr)
    idxs = [dt._names.index(k) for k in keys]
    recs = _flatten_records(arr._raw_data)
    if _is_ndarray(value):
        value = value.tolist()
    new = []
    for r in recs:
        lst = list(r)
        for j, k in enumerate(keys):
            fv = value[j] if isinstance(value, (list, tuple)) else value
            lst[idxs[j]] = _coerce_field_value(fv, dt._fields[k][0])
        new.append(tuple(lst))
    arr._raw_data = _reshape_flat(new, arr.shape)


def _set_record(arr, key, value):
    dt = _struct_dt_of(arr)
    recs = _flatten_records(arr._raw_data)
    n = len(recs)
    k = key + n if key < 0 else key
    if _is_ndarray(value):
        value = value.tolist()
    seq = value if isinstance(value, (list, tuple)) else (value,)
    _set_flat_record(arr, k, _coerce_record(seq, dt))


def _set_slice(arr, key, value):
    dt = _struct_dt_of(arr)
    recs = _flatten_records(arr._raw_data)
    n = len(recs)
    indices = list(range(*key.indices(n))) if isinstance(key, slice) else list(range(n))
    names = dt._names
    nf = len(names)
    if _is_ndarray(value) and getattr(value, '_dtype_obj', None) is not None:
        vrecs = _flatten_records(value._raw_data)
        for pos, i in enumerate(indices):
            src = vrecs[pos % len(vrecs)]
            recs[i] = tuple(_coerce_field_value(src[j], dt._fields[names[j]][0])
                            for j in range(nf))
    elif _is_ndarray(value):
        vals = value.tolist()
        for pos, i in enumerate(indices):
            sv = vals[pos % len(vals)]
            if isinstance(sv, (list, tuple)):
                # 2-D ndarray：每行作为一条记录，sv[j] 对应第 j 个字段
                recs[i] = tuple(_coerce_field_value(sv[j], dt._fields[names[j]][0])
                                for j in range(nf))
            else:
                # 1-D ndarray：标量广播到所有字段
                recs[i] = tuple(_coerce_field_value(sv, dt._fields[names[j]][0])
                                for j in range(nf))
    elif isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        # 记录列表：每个元素是一条记录，按位置赋值（支持广播）
        for pos, i in enumerate(indices):
            recs[i] = _coerce_record(value[pos % len(value)], dt)
    else:
        # 单条记录或标量广播到所有索引
        if isinstance(value, (list, tuple)):
            # tuple/list 视为单条记录（符合 NumPy 约定）
            rec = _coerce_record(value, dt)
        else:
            # 标量：广播到所有字段
            rec = tuple(_coerce_field_value(value, dt._fields[names[j]][0])
                        for j in range(nf))
        for i in indices:
            recs[i] = rec
    arr._raw_data = _reshape_flat(recs, arr.shape)


def _struct_setitem(arr, key, value):
    dt = _struct_dt_of(arr)
    if dt is None:
        return False
    if isinstance(key, str):
        if dt._names and key in dt._names:
            _set_field(arr, key, value)
            return True
        return False
    if isinstance(key, list) and key and _py_all(isinstance(k, str) for k in key):
        _set_multifield(arr, key, value)
        return True
    if isinstance(key, int) and not isinstance(key, bool):
        _set_record(arr, key, value)
        return True
    if isinstance(key, slice):
        _set_slice(arr, key, value)
        return True
    return False


# ---------- 字节打包/解包（供 view 重解释使用） ----------

def _pack_scalar(buf, off, val, fdt):
    import struct
    k = fdt._kind
    sz = fdt._itemsize
    if k in ('i', 'u', 'b'):
        iv = int(val)
        if k == 'i' and iv < 0:
            iv &= (1 << (sz * 8)) - 1
        buf[off:off + sz] = (iv & ((1 << (sz * 8)) - 1)).to_bytes(sz, 'little')
    elif k == 'f':
        fmt = {2: 'e', 4: 'f', 8: 'd'}[sz]
        buf[off:off + sz] = struct.pack('<' + fmt, float(val))
    elif k == 'c':
        half = sz // 2
        fmt = {4: 'f', 8: 'd'}[half]
        c = complex(val)
        buf[off:off + half] = struct.pack('<' + fmt, c.real)
        buf[off + half:off + sz] = struct.pack('<' + fmt, c.imag)
    elif k in ('S', 'V'):
        bs = bytes(val) if isinstance(val, (bytes, bytearray)) else (
            val.tobytes() if isinstance(val, void) else str(val).encode('latin-1'))
        buf[off:off + sz] = bs[:sz].ljust(sz, b'\x00')
    elif k == 'U':
        s = val if isinstance(val, str) else (
            bytes(val).decode('latin-1') if isinstance(val, (bytes, bytearray)) else str(val))
        buf[off:off + sz] = s.encode('utf-32-le')[:sz].ljust(sz, b'\x00')
    else:
        buf[off:off + sz] = b'\x00' * sz


def _pack_field(buf, off, val, fdt):
    if fdt._names is not None:
        for i, n in enumerate(fdt._names):
            _pack_field(buf, off + fdt._fields[n][1], val[i], fdt._fields[n][0])
    elif fdt._subdtype is not None:
        base, shape = fdt._subdtype
        flat = _flatten_data(val)
        esize = base._itemsize
        for j, e in enumerate(flat):
            _pack_field(buf, off + j * esize, e, base)
    else:
        _pack_scalar(buf, off, val, fdt)


def _pack_record(rec, dt):
    buf = bytearray(dt._itemsize)
    for i, n in enumerate(dt._names):
        _pack_field(buf, dt._fields[n][1], rec[i], dt._fields[n][0])
    return bytes(buf)


def _unpack_scalar(buf, off, fdt):
    import struct
    k = fdt._kind
    sz = fdt._itemsize
    seg = bytes(buf[off:off + sz])
    if k in ('i', 'u', 'b'):
        v = int.from_bytes(seg, 'little', signed=(k == 'i'))
        return builtins.bool(v) if k == 'b' else v
    if k == 'f':
        return struct.unpack('<' + {2: 'e', 4: 'f', 8: 'd'}[sz], seg)[0]
    if k == 'c':
        half = sz // 2
        fmt = {4: 'f', 8: 'd'}[half]
        return complex(struct.unpack('<' + fmt, seg[:half])[0],
                       struct.unpack('<' + fmt, seg[half:])[0])
    if k == 'S':
        return seg.rstrip(b'\x00')
    if k == 'V':
        return void(seg)
    if k == 'U':
        return seg.decode('utf-32-le').rstrip('\x00')
    return 0


def _unpack_field(buf, off, fdt):
    if fdt._names is not None:
        return tuple(_unpack_field(buf, off + fdt._fields[n][1], fdt._fields[n][0])
                     for n in fdt._names)
    if fdt._subdtype is not None:
        base, shape = fdt._subdtype
        total = 1
        for s in shape:
            total *= s
        esize = base._itemsize
        flat = [_unpack_field(buf, off + j * esize, base) for j in range(total)]
        return _reshape_flat(flat, shape)
    return _unpack_scalar(buf, off, fdt)


def _unpack_record(buf, off, dt):
    return tuple(_unpack_field(buf, off + dt._fields[n][1], dt._fields[n][0])
                 for n in dt._names)


def _struct_to_bytes(arr):
    dt = _struct_dt_of(arr)
    recs = _flatten_records(arr._raw_data)
    return b''.join(_pack_record(r, dt) for r in recs), len(recs)


def _view_dtype(arr, target):
    """按目标 dtype 重解释数组字节。"""
    tdt = target if isinstance(target, DType) else dtype(target)
    src_dt = getattr(arr, '_dtype_obj', None)
    if src_dt is not None:
        raw, _ = _struct_to_bytes(arr)
        ssize = src_dt._itemsize
    else:
        raw = arr.tobytes()
        ssize = dtype(getattr(arr, '_dtype', 'float64'))._itemsize
    tsize = tdt._itemsize
    if tsize == 0:
        raise ValueError("cannot view with zero-width dtype")
    count = len(raw) // tsize
    # rsnumpy 语义：view 仅改变最后一个轴的长度，其余轴保持不变。
    src_shape = tuple(arr.shape)
    if not src_shape:
        new_shape = (count,)
    else:
        last_bytes = src_shape[-1] * ssize
        if last_bytes % tsize != 0:
            raise ValueError(
                "When changing to a larger dtype, its size must be a divisor "
                "of the total size in bytes of the last axis of the array.")
        new_shape = src_shape[:-1] + (last_bytes // tsize,)
    if tdt._names is not None:
        records = [_unpack_record(raw, i * tsize, tdt) for i in range(count)]
        return _wrap_structured(tdt, records)
    vals = [_unpack_field(raw, i * tsize, tdt) for i in range(count)]
    return _build_plain_array(vals, tdt, new_shape)


# ---------- 比较 ----------

def _leaf_eq(x, y):
    if isinstance(x, void) or isinstance(y, void):
        return bytes(x.tobytes() if isinstance(x, void) else x) == \
            bytes(y.tobytes() if isinstance(y, void) else y)
    return x == y


def _struct_eq(a, b):
    ra = _flatten_records(a._raw_data)
    rb = _flatten_records(b._raw_data)
    out = []
    for x, y in zip(ra, rb):
        out.append(_py_all(_leaf_eq(x[i], y[i]) for i in range(_py_min(len(x), len(y)))))
    res = _reshape_flat(out, a.shape)
    return ndarray(res, _dtype='bool')


# ---------- 格式化 ----------

def _fmt_scalar_value(v, kind):
    if kind == 'b':
        return 'True' if v else 'False'
    if kind in ('i', 'u'):
        return str(int(v))
    if kind == 'f':
        return _trim_positional_float(float(v))
    if kind == 'c':
        return _format_complex_scalar(complex(v))
    if kind == 'S':
        b = bytes(v) if isinstance(v, (bytes, bytearray)) else str(v).encode('latin-1')
        return repr(b)
    if kind == 'U':
        return repr(v if isinstance(v, str) else str(v))
    if kind == 'V':
        return _void_repr(v.tobytes() if isinstance(v, void) else bytes(v))
    return repr(v)


def _fmt_sublist(v, base):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_sublist(x, base) for x in v) + "]"
    return _fmt_scalar_value(v, base._kind)


def _fmt_field(v, fdt):
    if fdt._names is not None:
        return _fmt_record(v, fdt)
    if fdt._subdtype is not None:
        base, shape = fdt._subdtype
        return _fmt_sublist(v, base)
    return _fmt_scalar_value(v, fdt._kind)


def _fmt_record(rec, dt):
    parts = [_fmt_field(rec[i], dt._fields[n][0]) for i, n in enumerate(dt._names)]
    if len(parts) == 1:
        return "(" + parts[0] + ",)"
    return "(" + ", ".join(parts) + ")"


class _Record:
    """结构化数组的记录标量（对应 rsnumpy.void 记录），支持字段名/整数索引读写。"""

    def __init__(self, values, dt, parent=None, flat_index=None):
        self._values = list(values)
        self._dtype_obj = dt
        self._parent = parent
        self._flat_index = flat_index

    @property
    def dtype(self):
        return self._dtype_obj

    def __getitem__(self, key):
        dt = self._dtype_obj
        if isinstance(key, str):
            i = dt._names.index(key)
            return _field_leaf_to_scalar(self._values[i], dt._fields[key][0])
        if isinstance(key, int):
            n = dt._names[key]
            return _field_leaf_to_scalar(self._values[key], dt._fields[n][0])
        raise KeyError(key)

    def __setitem__(self, key, value):
        dt = self._dtype_obj
        if isinstance(key, str):
            i = dt._names.index(key)
            fdt = dt._fields[key][0]
        elif isinstance(key, int):
            i = key
            fdt = dt._fields[dt._names[key]][0]
        else:
            raise KeyError(key)
        self._values[i] = _coerce_field_value(value, fdt)
        if self._parent is not None and self._flat_index is not None:
            _set_flat_record(self._parent, self._flat_index, tuple(self._values))

    def __getattr__(self, name):
        try:
            dt = self.__dict__['_dtype_obj']
        except KeyError:
            raise AttributeError(name)
        if dt is not None and dt._names and name in dt._names:
            i = dt._names.index(name)
            return _field_leaf_to_scalar(self.__dict__['_values'][i], dt._fields[name][0])
        raise AttributeError(name)

    def item(self):
        return _record_to_pytuple(tuple(self._values), self._dtype_obj)

    def tolist(self):
        return self.item()

    def __len__(self):
        return len(self._dtype_obj._names)

    def __eq__(self, other):
        if isinstance(other, _Record):
            return tuple(self._values) == tuple(other._values)
        return NotImplemented

    def __repr__(self):
        return _fmt_record(tuple(self._values), self._dtype_obj)

    def __str__(self):
        return _fmt_record(tuple(self._values), self._dtype_obj)


def _record_to_pytuple(rec, dt):
    out = []
    for i, n in enumerate(dt._names):
        fdt = dt._fields[n][0]
        v = rec[i]
        if fdt._names is not None:
            out.append(_record_to_pytuple(v, fdt))
        else:
            out.append(v)
    return tuple(out)


# ========== 标量类型别名 ==========
# 与 rsnumpy 一致的标量类型层次结构，使 issubdtype / isinstance 检查按预期工作。

class generic:
    """所有 rsnumpy 标量类型的抽象基类。"""


class number(generic):
    """数值标量的抽象基类。"""


class integer(number):
    """整数标量的抽象基类。"""


class signedinteger(integer):
    """有符号整数标量的抽象基类。"""


class unsignedinteger(integer):
    """无符号整数标量的抽象基类。"""


class inexact(number):
    """浮点/复数标量的抽象基类。"""


class floating(inexact):
    """浮点标量的抽象基类。"""

    def __new__(cls, val=0.0):
        return float(val)

    def __init__(self, val=0.0):
        pass


class complexfloating(inexact):
    """复数标量的抽象基类。"""


class flexible(generic):
    """可变长度标量（字符串/void）的抽象基类。"""


class character(flexible):
    """字符标量的抽象基类。"""


import operator as _operator
import warnings as _warnings


def _scalarize(r):
    """0-d ndarray → 同 dtype 的 rsnumpy 标量；非 0-d 或非数值 dtype 原样返回。"""
    if not _is_ndarray(r):
        return r
    if getattr(r, "ndim", 0) != 0:
        return r
    cls = _SCALAR_CLASS_BY_DTYPE.get(str(r.dtype))
    if cls is None:
        return r
    try:
        v = r.item()
        if isinstance(v, (list, tuple)):
            v = v[0]
        return cls(v)
    except Exception:
        return r


def _coerce0d(r, dt):
    """0-d 数组 dunder 可能返回裸值（如 __abs__ 对 0-d 用 item() 解包，且 Rust
    侧可能把 int 提升为 float64）：重包为 0-d 数组，统一交给守卫/提取路径
    处理 dtype 与溢出。"""
    if _is_ndarray(r):
        return r
    try:
        return ndarray(r, _dtype=str(dt))
    except Exception:
        return r


def _scalar_dunder_ns(pybase, is_int=False):
    """标量类的算术/比较 dunder：委托 0-d 数组路径，使结果 dtype/数值语义
    与 ndarray 运算完全一致（对齐 numpy 标量行为）。"""

    def _as0d(self):
        return ndarray(self, _dtype=str(self.dtype))

    def _fix0d(r, a):
        # __mul__ 等 Python 列表路径会把 0-d 结果变形成 (1,)：恢复 0-d，
        # 与数组路径的形状语义保持一致（也让 0-d 溢出守卫生效）。
        if (a.ndim == 0 and _is_ndarray(r)
                and getattr(r, 'ndim', 0) == 1 and getattr(r, 'size', None) == 1):
            try:
                return ndarray(r.item(), _dtype=str(r.dtype))
            except Exception:
                return r
        return r

    def _binop(self, other, op):
        a = _as0d(self)
        return _scalarize(_overflow_guard(_fix0d(_coerce0d(op(a, other), self.dtype), a), op.__name__))

    def _rbinop(self, other, op):
        a = _as0d(self)
        return _scalarize(_overflow_guard(_fix0d(_coerce0d(op(other, a), self.dtype), a), op.__name__))

    def _unary(self, op):
        return _scalarize(_overflow_guard(_coerce0d(op(_as0d(self)), self.dtype), op.__name__))

    def _idiv(self, other, op, name, r=False):
        # 整数除法/取模除零：numpy 语义为警告并返回 0。
        if str(self.dtype) in _INT_DTYPES and not _is_ndarray(other):
            try:
                if other == 0:
                    _warnings.warn(f"divide by zero encountered in {name}",
                                   RuntimeWarning, stacklevel=3)
                    return _SCALAR_CLASS_BY_DTYPE[str(self.dtype)](0)
            except Exception:
                pass
        if r:
            return _scalarize(_overflow_guard(_coerce0d(op(other, _as0d(self)), self.dtype), name))
        return _scalarize(_overflow_guard(_coerce0d(op(_as0d(self), other), self.dtype), name))

    def _cmp(self, other, op):
        r = op(_as0d(self), other)
        if _is_ndarray(r) and getattr(r, "ndim", 0) == 0:
            return _scalarize(r)
        return r

    def _safe_eq(self, other):
        # 优先 Python 数值语义：inf==inf 为 True、nan!=nan 为 False、complex/大整数
        # 精确。Rust eq 对 inf==inf 返回 False（IEEE 违规），故不能直接走数组路径。
        try:
            b = other.item() if _is_ndarray(other) else other
            return bool_(pybase(self) == b)
        except Exception:
            try:
                r = _operator.eq(_as0d(self), other)
            except TypeError:
                return bool_(False)
            if _is_ndarray(r) and getattr(r, "ndim", 0) == 0:
                return _scalarize(r)
            return r

    def _safe_ne(self, other):
        try:
            b = other.item() if _is_ndarray(other) else other
            return bool_(not (pybase(self) == b))
        except Exception:
            try:
                r = _operator.ne(_as0d(self), other)
            except TypeError:
                return bool_(True)
            if _is_ndarray(r) and getattr(r, "ndim", 0) == 0:
                return _scalarize(r)
            return r

    # numpy 语义：round(np.float64(x)) 返回同类型标量（而非 Python float）
    if pybase is float:
        _round_impl = lambda self, ndigits=None: type(self)(round(float(self), ndigits))
    elif pybase is int:
        _round_impl = lambda self, ndigits=None: type(self)(round(int(self), ndigits))
    elif pybase is complex:
        _round_impl = lambda self, ndigits=None: round(complex(self), ndigits)
    else:
        _round_impl = lambda self, ndigits=None: round(float(self), ndigits)
    d = {
        "__round__": _round_impl,
        "__add__": lambda self, other: _binop(self, other, _operator.add),
        "__radd__": lambda self, other: _rbinop(self, other, _operator.add),
        "__sub__": lambda self, other: _binop(self, other, _operator.sub),
        "__rsub__": lambda self, other: _rbinop(self, other, _operator.sub),
        "__mul__": lambda self, other: _binop(self, other, _operator.mul),
        "__rmul__": lambda self, other: _rbinop(self, other, _operator.mul),
        "__truediv__": lambda self, other: _binop(self, other, _operator.truediv),
        "__rtruediv__": lambda self, other: _rbinop(self, other, _operator.truediv),
        "__floordiv__": lambda self, other: _idiv(self, other, _operator.floordiv, "floor_divide", False),
        "__rfloordiv__": lambda self, other: _idiv(self, other, _operator.floordiv, "floor_divide", True),
        "__mod__": lambda self, other: _idiv(self, other, _operator.mod, "remainder", False),
        "__rmod__": lambda self, other: _idiv(self, other, _operator.mod, "remainder", True),
        "__pow__": lambda self, other: _binop(self, other, _operator.pow),
        "__rpow__": lambda self, other: _rbinop(self, other, _operator.pow),
        "__neg__": lambda self: _unary(self, _operator.neg),
        "__pos__": lambda self: _unary(self, _operator.pos),
        "__abs__": lambda self: _unary(self, _operator.abs),
        "__lt__": lambda self, other: _cmp(self, other, _operator.lt),
        "__le__": lambda self, other: _cmp(self, other, _operator.le),
        "__eq__": _safe_eq,
        "__ne__": _safe_ne,
        "__gt__": lambda self, other: _cmp(self, other, _operator.gt),
        "__ge__": lambda self, other: _cmp(self, other, _operator.ge),
        "__hash__": lambda self: builtins.hash(pybase(self)),
    }
    if is_int:
        # 整数标量：位运算/移位/取反（对齐 numpy 整数标量）
        d.update({
            "__and__": lambda self, other: _binop(self, other, _operator.and_),
            "__rand__": lambda self, other: _rbinop(self, other, _operator.and_),
            "__or__": lambda self, other: _binop(self, other, _operator.or_),
            "__ror__": lambda self, other: _rbinop(self, other, _operator.or_),
            "__xor__": lambda self, other: _binop(self, other, _operator.xor),
            "__rxor__": lambda self, other: _rbinop(self, other, _operator.xor),
            "__lshift__": lambda self, other: _binop(self, other, _operator.lshift),
            "__rshift__": lambda self, other: _binop(self, other, _operator.rshift),
            "__invert__": lambda self: _unary(self, _operator.invert),
        })
    return d


def _make_scalar_type(name, base, pybase, itemsize, zero):
    """Create a callable scalar type mirroring numpy's scalar classes.

    ``int64(5)`` returns a scalar instance; ``int64(array)`` with size > 1
    returns the array cast to this dtype.

    整型标量（pybase is int）**不**再继承 Python ``int``——对齐真实 numpy 的
    MRO。若继承 int，CPython 内置算子（``float.__add__``/``complex.__add__``
    等，对 int 子类返回 NotImplemented 前就会成功）会直接吃掉标量并返回裸
    Python 数值，rsnumpy 的 ``__radd__`` 等反射方法永远轮不到，结果丢失
    dtype（``test_scalar_matches_array_op_with_pyscalar[True-...]`` 簇的根因）。
    值改存 ``_value`` 属性，``__int__``/``__float__``/``__index__`` 等协议
    dunders 显式提供（CPython/PyO3 的抽取均会走这些协议）。
    浮点/复数标量保持继承 float/complex（其失败方向已被反射优先规则覆盖）。
    """
    is_int_scalar = pybase is int

    def _new(cls, val=zero, *args):
        if isinstance(val, ndarray):
            if val.size == 1:
                if is_int_scalar:
                    inst = base.__new__(cls)
                    inst._value = pybase(val.ravel().tolist()[0])
                    return inst
                return pybase.__new__(cls, pybase(val.ravel().tolist()[0]))
            return val.astype(name)
        if isinstance(val, (list, tuple)):
            return ndarray(val).astype(name)
        if is_int_scalar:
            inst = base.__new__(cls)
            inst._value = pybase(val)
            return inst
        return pybase.__new__(cls, pybase(val))

    ns = {
        '__new__': staticmethod(_new),
        '__init__': lambda self, *a: None,
        'itemsize': itemsize,
        'nbytes': property(lambda self: itemsize),
        'dtype': property(lambda self: DType(name)),
        'ndim': 0,
        'shape': (),
        'size': 1,
    }
    if is_int_scalar:
        ns.update({
            'item': lambda self: self._value,
            'tolist': lambda self: self._value,
            '__int__': lambda self: builtins.int(self._value),
            '__float__': lambda self: builtins.float(self._value),
            '__complex__': lambda self: builtins.complex(self._value),
            '__bool__': lambda self: builtins.bool(self._value),
            '__index__': lambda self: builtins.int(self._value),
            '__repr__': lambda self: builtins.repr(self._value),
            '__str__': lambda self: builtins.str(self._value),
            '__format__': lambda self, fmt: format(self._value, fmt),
        })
    else:
        ns.update({
            'item': lambda self: pybase(self),
            'tolist': lambda self: pybase(self),
        })
    ns.update(_scalar_dunder_ns(pybase, pybase is int))
    return type(name, (base,) if is_int_scalar else (base, pybase), ns)


int8 = _make_scalar_type('int8', signedinteger, int, 1, 0)
int16 = _make_scalar_type('int16', signedinteger, int, 2, 0)
int32 = _make_scalar_type('int32', signedinteger, int, 4, 0)
int64 = _make_scalar_type('int64', signedinteger, int, 8, 0)
uint8 = _make_scalar_type('uint8', unsignedinteger, int, 1, 0)
uint16 = _make_scalar_type('uint16', unsignedinteger, int, 2, 0)
uint32 = _make_scalar_type('uint32', unsignedinteger, int, 4, 0)
uint64 = _make_scalar_type('uint64', unsignedinteger, int, 8, 0)
float16 = _make_scalar_type('float16', floating, float, 2, 0.0)
float32 = _make_scalar_type('float32', floating, float, 4, 0.0)
float64 = _make_scalar_type('float64', floating, float, 8, 0.0)
float_ = float64  # rsnumpy 别名
complex64 = _make_scalar_type('complex64', complexfloating, complex, 8, 0j)
complex128 = _make_scalar_type('complex128', complexfloating, complex, 16, 0j)


def _bool_new(cls, val=False, *args):
    if isinstance(val, ndarray):
        if val.size == 1:
            inst = generic.__new__(cls)
            inst._value = 1 if builtins.bool(val.ravel().tolist()[0]) else 0
            return inst
        return val.astype('bool')
    if isinstance(val, (list, tuple)):
        return ndarray(val).astype('bool')
    inst = generic.__new__(cls)
    inst._value = 1 if builtins.bool(val) else 0
    return inst


string_ = type('string_', (character,), {})
unicode_ = type('unicode_', (character,), {})
bytes_ = string_  # rsnumpy 中 bytes_ 为字节字符串类型
str_ = unicode_   # rsnumpy 中 str_ 为 unicode 字符串类型
_bool_ns = {
    '__new__': staticmethod(_bool_new),
    '__init__': lambda self, *a: None,
    'itemsize': 1,
    'nbytes': property(lambda self: 1),
    'dtype': property(lambda self: DType('bool')),
    'ndim': 0,
    'shape': (),
    'size': 1,
    '__int__': lambda self: self._value,
    '__float__': lambda self: builtins.float(self._value),
    '__complex__': lambda self: builtins.complex(self._value),
    '__bool__': lambda self: builtins.bool(self._value),
    '__index__': lambda self: self._value,
    '__format__': lambda self, fmt: format(self._value, fmt),
    'item': lambda self: builtins.bool(int(self)),
    'tolist': lambda self: builtins.bool(int(self)),
    '__repr__': lambda self: builtins.repr(builtins.bool(int(self))),
    '__str__': lambda self: builtins.str(builtins.bool(int(self))),
}
_bool_ns.update(_scalar_dunder_ns(int))
# 与整型标量同理：不继承 Python int（否则 CPython 内置算子会先吃掉标量）。
bool_ = type('bool', (generic,), _bool_ns)
bool = bool_
# numpy 语义：bool_ 标量按值单例化（np.bool_(False) is np.bool_(False)）
_bool_cache = {}
_bool_new_orig = bool_.__new__

def _bool_new_cached(cls, *args, **kwargs):
    # 注意：模块级 bool 已被 bool_ 遮蔽，必须用 builtins.bool
    key = builtins.bool(args[0]) if args else False
    inst = _bool_cache.get(key)
    if inst is None:
        inst = _bool_new_orig(cls, *args, **kwargs)
        _bool_cache[key] = inst
    return inst


bool_.__new__ = _bool_new_cached
object_ = type('object_', (generic,), {})

# dtype 名 → 标量类（_scalarize 将 0-d 数组路径结果提取为标量）
_SCALAR_CLASS_BY_DTYPE = {
    'bool': bool_,
    'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64,
    'uint8': uint8, 'uint16': uint16, 'uint32': uint32, 'uint64': uint64,
    'float16': float16, 'float32': float32, 'float64': float64,
    'complex64': complex64, 'complex128': complex128,
}

# 单次分配允许的最大元素总数：Rust 侧 OOM / capacity overflow 是不可恢复的
# abort（直接杀死进程），必须在 Python 侧预检查并抛 MemoryError
# （numpy 语义：物化超大广播视图时抛 MemoryError）。
_MAX_ALLOC_ELEMENTS = 2**40


def _check_alloc_size(shape):
    """总元素数预检查；超过 _MAX_ALLOC_ELEMENTS 时抛 MemoryError。"""
    _tot = 1
    for _s in shape:
        _tot *= int(_s)
        if _tot > _MAX_ALLOC_ELEMENTS:
            raise MemoryError(
                f"rsnumpy: cannot allocate array of shape {tuple(shape)} "
                f"(exceeds {_MAX_ALLOC_ELEMENTS}-element limit)")
    return _tot


_INT_DTYPES = frozenset(('int8', 'int16', 'int32', 'int64',
                         'uint8', 'uint16', 'uint32', 'uint64'))
_INT_RANGES = {
    'int8': (-1 << 7, (1 << 7) - 1), 'int16': (-1 << 15, (1 << 15) - 1),
    'int32': (-1 << 31, (1 << 31) - 1), 'int64': (-1 << 63, (1 << 63) - 1),
    'uint8': (0, (1 << 8) - 1), 'uint16': (0, (1 << 16) - 1),
    'uint32': (0, (1 << 32) - 1), 'uint64': (0, (1 << 64) - 1),
}
_INT_ITEMSIZES = {
    'int8': 1, 'uint8': 1, 'int16': 2, 'uint16': 2,
    'int32': 4, 'uint32': 4, 'int64': 8, 'uint64': 8,
}


def _round_scalar_to_f(dtype, v):
    """Python float 标量按目标浮点精度舍入（NEP50 比较语义）。"""
    import struct as _st_n
    fmt = 'f' if dtype == 'float32' else 'e'
    try:
        return _st_n.unpack(fmt, _st_n.pack(fmt, v))[0]
    except Exception:
        return v


def _incomparable_0d(self, other):
    """0-d 数值 vs 字符串/字节：numpy 语义 → == 为 False 标量、!= 为 True 标量。"""
    if not _is_ndarray(other) or self.ndim != 0 or other.ndim != 0:
        return False
    a_raw = getattr(self, '_raw_data', None) is not None
    b_raw = getattr(other, '_raw_data', None) is not None
    if a_raw == b_raw:
        return False
    num_side = other if a_raw else self
    nd = str(getattr(num_side, '_dtype', ''))
    if nd.startswith('void') or nd.startswith('V'):
        return False
    return True


def _void_pair(self, other):
    """非结构化 void 比较对（两侧均无结构化 dtype）。"""
    if not _is_ndarray(other):
        return False
    if getattr(self, '_dtype_obj', None) is not None or \
            getattr(other, '_dtype_obj', None) is not None:
        return False
    a = str(getattr(self, '_dtype', ''))
    b = str(getattr(other, '_dtype', ''))
    return (a.startswith('void') or a.startswith('V') or
            b.startswith('void') or b.startswith('V'))


def _operand_dtype_name(x):
    """操作数的强 dtype（Python 裸标量为 weak → None，NEP50 语义）。"""
    if x is None:
        return None
    if _is_ndarray(x):
        return str(x._dtype)
    dt = getattr(x, 'dtype', None)
    if dt is not None:
        return str(dt)
    return None


def _same_kind_cast_ok(src, dst):
    """same-kind 规则：src 能否写回 dst（numpy 语义的子集）。"""
    if src == dst:
        return True
    if src in _INT_DTYPES and dst in _INT_DTYPES:
        return _INT_ITEMSIZES[dst] >= _INT_ITEMSIZES[src]
    if src in ('float16', 'float32', 'float64') and \
            dst in ('float16', 'float32', 'float64'):
        return {'float16': 2, 'float32': 4, 'float64': 8}[dst] >= \
            {'float16': 2, 'float32': 4, 'float64': 8}[src]
    if (src in _INT_DTYPES or src == 'bool') and \
            dst in ('float16', 'float32', 'float64', 'complex64', 'complex128'):
        return True
    return False


def _clip_bound_is_nan(b):
    """clip 界（标量/0-d 数组）是否为 NaN。numpy 语义：nan 界经
    maximum/minimum 传播，clip 结果全为 nan。"""
    try:
        if _is_ndarray(b):
            b = b.item()
        if isinstance(b, complex):
            return b.real != b.real or b.imag != b.imag
        return builtins.float(b) != builtins.float(b)
    except Exception:
        return False


def _clip_bound_is_complex(b):
    """clip 界是否为复数（Python complex / 复数标量 / 复数数组）。"""
    try:
        if _is_ndarray(b):
            return str(getattr(b, '_dtype', '')).startswith('complex')
        return isinstance(b, builtins.complex)
    except Exception:
        return False


def _overflow_guard(r, opname):
    """整数 0-d 运算结果溢出检测：按位宽回绕并告警（numpy 语义：
    RuntimeWarning "overflow encountered in ..."）。"""
    try:
        if not _is_ndarray(r) or getattr(r, 'ndim', 0) != 0:
            return r
        dt = str(r.dtype)
        if dt not in _INT_DTYPES:
            return r
        lo, hi = _INT_RANGES[dt]
        # 直接读底层 f64 存储（item()/tolist 的 uint64 补码掩码会先于守卫
        # 把越界值折回区间内，导致漏报）
        try:
            import struct as _st_ovf
            v = float(_st_ovf.unpack('<d',
                                     memoryview(r._array).tobytes()[:8])[0])
        except Exception:
            v = r.item()
        if isinstance(v, bool):
            return r
        if isinstance(v, float):
            if not v.is_integer():
                return r
            v = int(v)
        elif not isinstance(v, int):
            return r
        if lo <= v <= hi:
            return r
        width = 1 << (8 * _INT_ITEMSIZES[dt])
        w = (v - lo) % width + lo
        _warnings.warn(f"overflow encountered in {opname}",
                       RuntimeWarning, stacklevel=3)
        return _SCALAR_CLASS_BY_DTYPE[dt](w)
    except Exception:
        return r


def _dtype_type_hook(dt):
    try:
        return _SCALAR_CLASS_BY_DTYPE.get(str(dt))
    except Exception:
        return None


from ._dtypes import _set_type_hook as _set_dtype_type_hook
_set_dtype_type_hook(_dtype_type_hook)


def _void_repr(bs):
    """void 标量的显示：每字节 \\xHH（大写十六进制）。"""
    return "b'" + ''.join('\\x%02X' % b for b in bs) + "'"


class void(flexible):
    """void 标量：存储原始字节，支持按字节相等比较（兼容 rsnumpy.void）。"""

    def __init__(self, data):
        if isinstance(data, (bytes, bytearray)):
            self._bytes = bytes(data)
        elif isinstance(data, int):
            self._bytes = b'\x00' * data
        else:
            self._bytes = bytes(data)

    def tobytes(self):
        return self._bytes

    @property
    def dtype(self):
        return dtype('V%d' % len(self._bytes))

    def __eq__(self, other):
        if isinstance(other, void):
            return self._bytes == other._bytes
        if isinstance(other, (bytes, bytearray)):
            return self._bytes == bytes(other)
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        return hash(self._bytes)

    def __repr__(self):
        return _void_repr(self._bytes)

    def __str__(self):
        return _void_repr(self._bytes)


# C 语言宽度别名（与 rsnumpy 在 64 位平台上的取值一致）
byte = int8
short = int16
intc = int32
int_ = int64
long = int64
longlong = int64
intp = int64
ubyte = uint8
ushort = uint16
uintc = uint32
uint = uint64
ulong = uint64
ulonglong = uint64
uintp = uint64
half = float16
single = float32
double = float64
longdouble = float64
csingle = complex64
cdouble = complex128
clongdouble = complex128

# 布尔常量与其他特殊常量
True_ = True
False_ = False
little_endian = (_sys.byteorder == 'little')

# 标量类型名 → 类型对象 映射（供 issubdtype / result_type 等使用）
_SCTYPE_BY_NAME = {
    'bool': bool_,
    'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64,
    'uint8': uint8, 'uint16': uint16, 'uint32': uint32, 'uint64': uint64,
    'float16': float16, 'float32': float32, 'float64': float64,
    'complex64': complex64, 'complex128': complex128,
    'string_': string_, 'unicode_': unicode_, 'bytes': bytes_,
    'str': str_, 'object': object_, 'void': void,
}

# rsnumpy 兼容的 sctypeDict / ScalarType / typecodes
ScalarType = (int, float, complex, bool, bytes, str,
              int8, int16, int32, int64, uint8, uint16, uint32, uint64,
              float16, float32, float64, complex64, complex128)

sctypeDict = {
    'bool': bool_, 'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64,
    'uint8': uint8, 'uint16': uint16, 'uint32': uint32, 'uint64': uint64,
    'float16': float16, 'float32': float32, 'float64': float64,
    'complex64': complex64, 'complex128': complex128,
    'i1': int8, 'i2': int16, 'i4': int32, 'i8': int64,
    'u1': uint8, 'u2': uint16, 'u4': uint32, 'u8': uint64,
    'f2': float16, 'f4': float32, 'f8': float64,
    'c8': complex64, 'c16': complex128, '?': bool_,
}

typecodes = {
    'Character': 'c',
    'Integer': 'bhilqp',
    'UnsignedInteger': 'BHILQP',
    'Float': 'efdg',
    'Complex': 'FDG',
    'AllInteger': 'bBhHiIlLqQpP',
    'AllFloat': 'efdgFDG',
    'Datetime': 'Mm',
    'All': '?bhilqpBHILQPefdgFDGSUVOMm',
}


def _to_scalar_type(x):
    """将 dtype 类字符串 / 类型 / 数组解析为标量类型对象。"""
    if isinstance(x, type) and issubclass(x, generic):
        return x
    if hasattr(x, '_dtype'):
        return _SCTYPE_BY_NAME.get(x._dtype, generic)
    name = _resolve_type_name(x)
    return _SCTYPE_BY_NAME.get(name, generic)


def issubdtype(arg1, arg2):
    """判断第一个类型是否为第二个类型（含抽象类别）的子类型。"""
    c1 = _to_scalar_type(arg1)
    c2 = arg2 if (isinstance(arg2, type) and issubclass(arg2, generic)) else _to_scalar_type(arg2)
    return issubclass(c1, c2)


class finfo:
    """浮点类型的机器精度信息（兼容 rsnumpy.finfo）。"""

    _DATA = {
        'float16': dict(bits=16, eps=0.0009765625, epsneg=0.00048828125,
                        max=65504.0, min=-65504.0, tiny=6.103515625e-05,
                        resolution=0.001, nmant=10, nexp=5, precision=3),
        'float32': dict(bits=32, eps=1.1920929e-07, epsneg=5.9604645e-08,
                        max=3.4028235e+38, min=-3.4028235e+38, tiny=1.1754944e-38,
                        resolution=1e-06, nmant=23, nexp=8, precision=6),
        'float64': dict(bits=64, eps=2.220446049250313e-16, epsneg=1.1102230246251565e-16,
                        max=1.7976931348623157e+308, min=-1.7976931348623157e+308,
                        tiny=2.2250738585072014e-308, resolution=1e-15,
                        nmant=52, nexp=11, precision=15),
    }

    def __init__(self, dtype):
        name = _resolve_type_name(dtype)
        if name in ('complex64',):
            name = 'float32'
        elif name in ('complex128',):
            name = 'float64'
        if name not in self._DATA:
            raise ValueError(f"data type {dtype!r} not inexact")
        self._name = name
        d = self._DATA[name]
        self.bits = d['bits']
        self.eps = d['eps']
        self.epsneg = d['epsneg']
        self.max = d['max']
        self.min = d['min']
        self.tiny = d['tiny']
        self.smallest_normal = d['tiny']
        self.resolution = d['resolution']
        self.nmant = d['nmant']
        self.nexp = d['nexp']
        self.maxexp = 2 ** (self.nexp - 1)
        self.minexp = 2 - self.maxexp
        self.precision = d['precision']

    @property
    def dtype(self):
        return dtype(self._name)

    def __repr__(self):
        return (f"finfo(resolution={self.resolution}, min={self.min}, "
                f"max={self.max}, dtype={self._name})")


class iinfo:
    """整数类型的取值范围信息（兼容 rsnumpy.iinfo）。"""

    _BITS = {
        'int8': (8, True), 'int16': (16, True), 'int32': (32, True), 'int64': (64, True),
        'uint8': (8, False), 'uint16': (16, False), 'uint32': (32, False), 'uint64': (64, False),
    }

    def __init__(self, int_type):
        name = _resolve_type_name(int_type)
        if name not in self._BITS:
            # 兼容单字符码别名（long 平台差异 / 测试用 p/P 等）
            name = {'l': 'int64', 'p': 'int64', 'q': 'int64',
                    'L': 'uint64', 'P': 'uint64', 'Q': 'uint64',
                    'b': 'int8', 'h': 'int16', 'i': 'int32',
                    'B': 'uint8', 'H': 'uint16', 'I': 'uint32'}.get(name, name)
        if name not in self._BITS:
            raise ValueError(f"Invalid integer data type {int_type!r}.")
        self._name = name
        self.bits, signed = self._BITS[name]
        self.kind = 'i' if signed else 'u'
        if signed:
            self.min = -(1 << (self.bits - 1))
            self.max = (1 << (self.bits - 1)) - 1
        else:
            self.min = 0
            self.max = (1 << self.bits) - 1

    @property
    def dtype(self):
        return dtype(self._name)

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self._name})"

    def __str__(self):
        dashes = '-' * 63
        return (
            f"Machine parameters for {self._name}\n"
            f"{dashes}\n"
            f"min = {self.min}\n"
            f"max = {self.max}\n"
            f"{dashes}\n"
        )


class ndindex:
    """按 C 顺序遍历给定形状的多维索引（兼容 rsnumpy.ndindex）。"""

    def __init__(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        self._shape = tuple(int(s) for s in shape)

    def __iter__(self):
        import itertools as _it
        if not self._shape:
            yield ()
            return
        yield from _it.product(*[range(s) for s in self._shape])


class ndenumerate:
    """同时给出多维索引与元素值的迭代器（兼容 rsnumpy.ndenumerate）。"""

    def __init__(self, arr):
        self._arr = asarray(arr)

    def __iter__(self):
        shape = self._arr.shape
        flat = _flatten_to_list(self._arr.tolist())
        import itertools as _it
        coords = _it.product(*[range(s) for s in shape]) if shape else [()]
        for c, v in zip(coords, flat):
            yield c, v


def _flatten_to_list(data):
    out = []
    stack = [data]
    while stack:
        v = stack.pop()
        if isinstance(v, list):
            stack.extend(reversed(v))
        else:
            out.append(v)
    return out


class index_exp_class:
    """构造索引元组：index_exp[1:3, 2] -> (slice(1,3), 2)。"""

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return item
        return (item,)


index_exp = index_exp_class()


# ========== 日期时间类型 ==========
# 内部统一以“自 1970-01-01 (UTC) 起的天数”存储，与 matplotlib/rsplotlib 的日期约定
# 一致，从而可直接参与绘图并被 ConciseDateFormatter 正确格式化为刻度标签。

_DT_EPOCH = _datetime.datetime(1970, 1, 1)

# 时间单位 → 天数换算（Y/M 因日历长度可变，取近似值）
_TD_UNIT_DAYS = {
    'W': 7.0,
    'D': 1.0,
    'h': 1.0 / 24.0,
    'm': 1.0 / 1440.0,
    's': 1.0 / 86400.0,
    'ms': 1.0 / 86400.0e3,
    'us': 1.0 / 86400.0e6,
    'ns': 1.0 / 86400.0e9,
    'Y': 365.0,
    'M': 30.0,
}


def _parse_datetime_string(s, unit=None):
    """解析 ISO 日期字符串为 (自纪元起天数, 推断出的单位)。"""
    s = s.strip()
    date_part, time_part = s, None
    if 'T' in s:
        date_part, time_part = s.split('T', 1)
    elif ':' in s and ' ' in s:
        date_part, time_part = s.split(' ', 1)
    ymd = date_part.split('-')
    year = int(ymd[0])
    month = int(ymd[1]) if len(ymd) > 1 else 1
    day = int(ymd[2]) if len(ymd) > 2 else 1
    resolved = 'D' if len(ymd) >= 3 else ('M' if len(ymd) == 2 else 'Y')
    hour = minute = second = micro = 0
    if time_part:
        tparts = time_part.split(':')
        hour = int(tparts[0])
        resolved = 'h'
        if len(tparts) > 1:
            minute = int(tparts[1])
            resolved = 'm'
        if len(tparts) > 2:
            sec = tparts[2]
            if '.' in sec:
                sec_i, frac = sec.split('.', 1)
                second = int(sec_i)
                micro = int(_py_round(float('0.' + frac) * 1e6))
                resolved = 'us'
            else:
                second = int(sec)
                resolved = 's'
    dt = _datetime.datetime(year, month, day, hour, minute, second, micro)
    days = (dt - _DT_EPOCH).total_seconds() / 86400.0
    return days, (unit or resolved)


def _parse_datetime_to_days(value, unit=None):
    """将 datetime64 的各种输入统一解析为 (自纪元起天数, 单位)。"""
    if isinstance(value, datetime64):
        return value._days, (unit or value._unit)
    if isinstance(value, _datetime.datetime):
        return (value - _DT_EPOCH).total_seconds() / 86400.0, (unit or 'us')
    if isinstance(value, _datetime.date):
        dt = _datetime.datetime(value.year, value.month, value.day)
        return (dt - _DT_EPOCH).total_seconds() / 86400.0, (unit or 'D')
    if isinstance(value, str):
        return _parse_datetime_string(value, unit)
    # 数值 + 单位：表示纪元之后的偏移量
    u = unit or 'us'
    return float(value) * _TD_UNIT_DAYS.get(u, _TD_UNIT_DAYS['us']), u


class timedelta64:
    """时间间隔标量，内部以天数存储，兼容 rsnumpy.timedelta64 的常用构造与运算。"""

    def __init__(self, value=0, unit=None):
        if isinstance(value, timedelta64):
            self._days = value._days
            self._unit = unit or value._unit
            return
        u = unit or 'us'
        self._unit = u
        self._days = float(value) * _TD_UNIT_DAYS.get(u, _TD_UNIT_DAYS['us'])

    @classmethod
    def _from_days(cls, days, unit='us'):
        obj = cls.__new__(cls)
        obj._days = float(days)
        obj._unit = unit
        return obj

    def _value_in_unit(self):
        return self._days / _TD_UNIT_DAYS.get(self._unit, 1.0)

    def __float__(self):
        return self._days

    def __int__(self):
        return int(self._value_in_unit())

    def __add__(self, other):
        if isinstance(other, datetime64):
            return datetime64._from_days(self._days + other._days, other._unit)
        if isinstance(other, timedelta64):
            return timedelta64._from_days(self._days + other._days, self._unit)
        return timedelta64._from_days(self._days + float(other), self._unit)
    __radd__ = __add__

    def __sub__(self, other):
        d = other._days if isinstance(other, timedelta64) else float(other)
        return timedelta64._from_days(self._days - d, self._unit)

    def __mul__(self, other):
        return timedelta64._from_days(self._days * float(other), self._unit)
    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, timedelta64):
            return self._days / other._days
        return timedelta64._from_days(self._days / float(other), self._unit)

    def __eq__(self, other):
        if isinstance(other, timedelta64):
            return self._days == other._days
        if isinstance(other, (int, float)):
            return self._days == float(other)
        return NotImplemented

    def __hash__(self):
        return hash(self._days)

    def __lt__(self, other):
        if isinstance(other, timedelta64):
            return self._days < other._days
        if isinstance(other, (int, float)):
            return self._days < float(other)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, timedelta64):
            return self._days <= other._days
        if isinstance(other, (int, float)):
            return self._days <= float(other)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, timedelta64):
            return self._days > other._days
        if isinstance(other, (int, float)):
            return self._days > float(other)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, timedelta64):
            return self._days >= other._days
        if isinstance(other, (int, float)):
            return self._days >= float(other)
        return NotImplemented

    def __repr__(self):
        v = self._value_in_unit()
        vi = int(_py_round(v))
        vout = vi if abs(v - vi) < 1e-9 else v
        return "numpy.timedelta64(%r,'%s')" % (vout, self._unit)


class datetime64:
    """日期时间标量，内部以“自 1970-01-01 起的天数”存储，兼容 rsnumpy.datetime64 的常用用法。"""

    def __init__(self, value=None, unit=None):
        if value is None:
            self._days = 0.0
            self._unit = unit or 'D'
            return
        self._days, self._unit = _parse_datetime_to_days(value, unit)

    @classmethod
    def _from_days(cls, days, unit='D'):
        obj = cls.__new__(cls)
        obj._days = float(days)
        obj._unit = unit
        return obj

    def to_datetime(self):
        return _DT_EPOCH + _datetime.timedelta(days=self._days)

    def __float__(self):
        return self._days

    def __int__(self):
        return int(self._days)

    def __add__(self, other):
        if isinstance(other, timedelta64):
            return datetime64._from_days(self._days + other._days, self._unit)
        return datetime64._from_days(self._days + float(other), self._unit)
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, datetime64):
            return timedelta64._from_days(self._days - other._days, self._unit)
        if isinstance(other, timedelta64):
            return datetime64._from_days(self._days - other._days, self._unit)
        return datetime64._from_days(self._days - float(other), self._unit)

    def __eq__(self, other):
        if isinstance(other, datetime64):
            return self._days == other._days
        return NotImplemented

    def __hash__(self):
        return hash(self._days)

    def __lt__(self, other):
        if isinstance(other, datetime64):
            return self._days < other._days
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, datetime64):
            return self._days <= other._days
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, datetime64):
            return self._days > other._days
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, datetime64):
            return self._days >= other._days
        return NotImplemented

    def __repr__(self):
        dt = self.to_datetime()
        u = self._unit
        if u == 'Y':
            s = dt.strftime('%Y')
        elif u == 'M':
            s = dt.strftime('%Y-%m')
        elif u == 'D':
            s = dt.strftime('%Y-%m-%d')
        elif u in ('h', 'm'):
            s = dt.strftime('%Y-%m-%dT%H:%M')
        elif u == 's':
            s = dt.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            s = dt.isoformat()
        return "numpy.datetime64('%s')" % s


# ========== 构造/工厂函数 ==========

# np.build_array 返回的 dtype 编码 → rsnumpy dtype 名称（与 _infer_int_dtype 一致）
_BUILD_ARRAY_DTYPES = ('float64', 'int64', 'bool')


def _flatten_check(data):
    """单次遍历：展平数据并检查是否包含复数/字符串。"""
    flat = []
    has_complex = False
    has_string = False
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            flat.append(item)
            if isinstance(item, complex):
                has_complex = True
            elif isinstance(item, str):
                has_string = True
    return flat, has_complex, has_string


def _flatten_data(data):
    """展平嵌套的可迭代对象为扁平列表。"""
    flat, _, _ = _flatten_check(data)
    return flat


def _contains_ndarray(data):
    """递归检查嵌套列表/元组中是否包含 rsnumpy ndarray 或底层 Rust 绑定 ndarray。"""
    if _is_any_ndarray(data):
        return True
    if isinstance(data, (list, tuple)):
        for x in data:
            if _contains_ndarray(x):
                return True
    return False


def _convert_ndarray_deep(data):
    """递归将嵌套列表/元组中的 ndarray（含底层 Rust 绑定）转为 Python 原生数据（tolist）。

    保持原有嵌套结构，仅替换 ndarray 节点为其 tolist() 结果。
    """
    if _is_any_ndarray(data):
        return data.tolist()
    if isinstance(data, list):
        return [_convert_ndarray_deep(x) for x in data]
    if isinstance(data, tuple):
        return tuple(_convert_ndarray_deep(x) for x in data)
    return data


def _setitem_value(value):
    """规范化赋值右值供 Rust setitem_multi 使用：
    标量原样返回；ndarray 或嵌套列表展平为 C 序浮点列表（逐元素赋值）。
    复数右值原样透传，交由 Rust coerce_value_to_nd 保留虚部。"""
    if (isinstance(value, generic) and not _is_ndarray(value)
            and hasattr(value, 'tolist')):
        # rsnumpy 整型/bool 标量（非 int 子类）→ Python 原生值
        value = value.tolist()
    if _is_ndarray(value):
        rust = value._array
        if getattr(rust, 'is_complex', False):
            return rust
        return [float(v) for v in _flatten_data(rust.tolist())]
    if value.__class__.__name__ == 'ndarray' and hasattr(value, 'tolist'):
        if getattr(value, 'is_complex', False):
            return value
        return [float(v) for v in _flatten_data(value.tolist())]
    if isinstance(value, complex):
        return value
    if isinstance(value, (list, tuple)):
        flat = _flatten_data(value)
        # 检查是否含复数：Python complex，或 0 维复数 ndarray 元素（如 arr[i,j,j]）
        # 复数右值原样透传，交由 Rust coerce_value_to_nd -> parse_py_complex 保留虚部
        has_complex = _py_any(
            isinstance(v, complex)
            or (hasattr(v, '_array') and getattr(v._array, 'is_complex', False))
            for v in flat
        )
        if has_complex:
            return value
        return [float(v) for v in flat]
    return value


_INTEGER_DTYPES = frozenset((
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
))


def _cast_setitem_value(value, dtype):
    """按目标数组 dtype 转换赋值右值：整数类型向零截断，bool 类型布尔化，
    与 rsnumpy 赋值时的隐式类型转换一致（底层仍存 f64）。"""
    if dtype in _INTEGER_DTYPES:
        if isinstance(value, list):
            return [float(int(v)) for v in value]
        return float(int(value))
    if dtype == "bool":
        if isinstance(value, list):
            return [1.0 if v else 0.0 for v in value]
        return 1.0 if value else 0.0
    return value


# 数组接口协议 typestr 的 (kind, itemsize) → rsnumpy dtype 名称
_ARRAY_INTERFACE_DTYPE = {
    ('b', 1): 'bool',
    ('u', 1): 'uint8', ('i', 1): 'int8',
    ('u', 2): 'uint16', ('i', 2): 'int16',
    ('u', 4): 'uint32', ('i', 4): 'int32',
    ('u', 8): 'uint64', ('i', 8): 'int64',
    ('f', 4): 'float32', ('f', 8): 'float64',
}

# rsnumpy dtype 名称 → 数组接口 typestr（供 __array_interface__ 暴露给真实 rsnumpy）
_DTYPE_TO_TYPESTR = {
    'bool': '|b1',
    'uint8': '|u1', 'int8': '|i1',
    'uint16': '<u2', 'int16': '<i2',
    'uint32': '<u4', 'int32': '<i4',
    'uint64': '<u8', 'int64': '<i8',
    'float32': '<f4', 'float64': '<f8',
}


def _from_array_interface(obj):
    """将实现数组接口协议的对象（如 PIL 图像）解码为 (Rust 数组, dtype)。

    缓冲区的字节解码由 Rust 底层 `np.from_buffer_typed` 完成。
    """
    ai = obj.__array_interface__
    shape = tuple(ai['shape'])
    typestr = ai['typestr']
    data = ai['data']
    if isinstance(data, tuple):
        # (指针, 只读标志) 形式，无法直接读取 → 退回到 tobytes()
        data = obj.tobytes()
    elif not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    kind, itemsize = typestr[1], int(typestr[2:])
    dtype = _ARRAY_INTERFACE_DTYPE.get((kind, itemsize))
    if dtype is None:
        raise TypeError(f"Unsupported array interface type: {typestr}")
    arr = np.from_buffer_typed(bytes(data), typestr, list(shape))
    return arr, dtype


def _nested_shape(data):
    """嵌套列表的形状（沿首元素下行）；非列表返回 ()。"""
    shape = []
    while isinstance(data, (list, tuple)):
        shape.append(len(data))
        data = data[0] if data else None
    return tuple(shape) or (0,)


def _is_rectangular(data):
    """检查嵌套列表是否为矩形（所有子列表长度相同）。"""
    if not isinstance(data, (list, tuple)):
        return True
    if not data:
        return True
    first_len = None
    for item in data:
        if isinstance(item, (list, tuple)):
            if first_len is None:
                first_len = len(item)
            elif len(item) != first_len:
                return False
        else:
            if first_len is not None:
                return False
    return True


def array(data, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    """创建数组。

    【性能提示】
    ⚠️ 避免 dtype=object：会完全失去向量化性能优势！
    - object 类型数组存储 Python 对象引用，无法使用 SIMD 加速
    - 循环性能比 Python 列表还慢
    - 内存占用高，无法释放 GIL

    【推荐】
    - 字符串：让 rsnumpy 自动推断为固定长度字符串
    - 大整数（<2^53）：使用 int64
    - 混合类型：使用结构化数组或 Python 列表
    """
    if dtype is None:
        # ndarray 输入快速路径：直接沿用其 dtype，避免走回退路径产生 object dtype 警告。
        if _is_ndarray(data):
            # 广播 override 视图：物化（或按 numpy 语义即使 copy=False 也需物化）→ 超限 MemoryError
            if getattr(data, '_rs_shape_override', None) is not None:
                _check_alloc_size(data.shape)
            arr = data.copy() if copy else data
            if ndmin > arr.ndim:
                new_shape = (1,) * (ndmin - arr.ndim) + arr.shape
                arr = ndarray._wrap(arr._array.reshape(new_shape), _dtype=arr._dtype)
            return arr
        # numpy 兼容：带 __array_struct__ 的包装对象先解包
        if not _is_ndarray(data) and hasattr(data, '__array_struct__'):
            data = data.__array_struct__
        # numpy 语义：带 __getitem__/__len__ 的序列对象先 list()（错误向上传播）
        if (not _is_ndarray(data)
                and not isinstance(data, (list, tuple, str, bytes, dict, set,
                                          frozenset, int, float, complex, bool,
                                          generic))
                and not hasattr(data, '__array_interface__')
                and not hasattr(data, '__array__')
                and hasattr(data, '__getitem__') and hasattr(data, '__len__')
                and type(data).__module__ != 'builtins'):
            data = list(data)
        # 数值 list/tuple 快速路径：Rust 单次完成展平 + dtype 推断 + 构造。
        if isinstance(data, (list, tuple)):
            # MemoryError 预检查：列表含超大广播视图时 Rust 物化会 abort
            for _e in data:
                if _is_ndarray(_e) and getattr(_e, '_rs_shape_override', None) is not None:
                    _check_alloc_size(_e.shape)
            _scalar_dt = _uniform_scalar_dtype(data)
            if _scalar_dt is not None:
                # 全 rsnumpy 标量列表：保留标量自身 dtype（值推断无法区分 int8/int64 等）。
                # Rust 构造不接受标量叶子元素：叶子先转 Python 原生值。
                return array(_scalar_leaves_to_py(data), dtype=_scalar_dt)
            if _has_rscalar_leaf(data):
                # 混合列表含 rsnumpy 标量叶子：同样先转 Python 原生值
                data = _scalar_leaves_to_py(data)
            try:
                raw, code = np.build_array(data)
            except (ValueError, TypeError):
                raw = None
            if raw is not None:
                _dtype = _BUILD_ARRAY_DTYPES[code]
                arr = ndarray._wrap(raw, _dtype=_dtype)
                if ndmin > arr.ndim:
                    new_shape = (1,) * (ndmin - arr.ndim) + arr.shape
                    arr = ndarray._wrap(arr._array.reshape(new_shape), _dtype=_dtype)
                return arr
        # build_array 不支持 ndarray 元素：递归展平嵌套列表中的 ndarray 再重试
        if _contains_ndarray(data):
            converted = _convert_ndarray_deep(data)
            if _has_rscalar_leaf(converted):
                converted = _scalar_leaves_to_py(converted)
            try:
                raw, code = np.build_array(converted)
            except (ValueError, TypeError):
                raw = None
            if raw is not None:
                _dtype = _BUILD_ARRAY_DTYPES[code]
                arr = ndarray._wrap(raw, _dtype=_dtype)
                if ndmin > arr.ndim:
                    new_shape = (1,) * (ndmin - arr.ndim) + arr.shape
                    arr = ndarray._wrap(arr._array.reshape(new_shape), _dtype=_dtype)
                return arr
            data = converted
        # 回退慢路径：复数/字符串/不规则/标量
        if isinstance(data, tuple):
            data = list(data)
        _is_rscalar = (not _is_ndarray(data)) and getattr(data, 'dtype', None) is not None
        if _is_rscalar or (isinstance(data, builtins.complex) and not isinstance(data, bool)):
            # rsnumpy 标量 / Python complex：保留标量自身 dtype（对齐 numpy 语义）。
            _dtype = _infer_scalar_dtype(data)
            if (isinstance(data, builtins.complex)
                    or str(_dtype).startswith('complex')) and (
                    dtype is None or str(_resolve_dtype(dtype)).startswith('complex')):
                # 复数标量：Rust 抽取只取实部 → 用复数列表构造并恢复 0-d
                _c = complex(data)
                _arr = ndarray([_c]).reshape(())
                if dtype is not None:
                    _arr = _arr.astype(_resolve_dtype(dtype))
                if ndmin > _arr.ndim:
                    _arr = _arr.reshape((1,) * (ndmin - _arr.ndim) + _arr.shape)
                return _arr
        else:
            flat, has_c, has_s = _flatten_check(data)
            if has_c:
                _dtype = "complex128"
            elif has_s:
                # 全字符串数据：自动推断固定长度字符串 dtype（比 object 性能更好）
                str_lens = [len(str(x)) for x in flat]
                max_len = max(str_lens) if str_lens else 1
                _dtype = f"U{max_len}"
            else:
                # 智能类型推断：避免不必要的 object dtype
                _dtype = _infer_smart_dtype(flat)
    else:
        _dtype = _resolve_dtype(dtype)
        # numpy 语义：float NaN/inf → 整数 dtype：0-d 标量允许 cast（警告并给 0），
        # 与 astype 路径值一致
        if _dtype in _INT_DTYPES and isinstance(data, float) and \
                not isinstance(data, bool):
            if data != data or data in (float('inf'), float('-inf')):
                import warnings as _w_intc
                _w_intc.warn("invalid value encountered in cast",
                             RuntimeWarning, stacklevel=3)
                data = 0
        # numpy 语义：列表输入 + 显式整数 dtype 时，NaN 抛 ValueError、越界抛
        # OverflowError（0-d 标量例外，允许 cast）
        if isinstance(data, (list, tuple)):
            _check_int_coerce(data, _dtype)
        elif _dtype in _INT_DTYPES and (
                (isinstance(data, int) and not isinstance(data, bool))
                or (isinstance(data, generic)
                    and str(getattr(data, 'dtype', '')) in _INT_DTYPES)):
            if isinstance(data, generic):
                # rsnumpy 标量：same-kind cast 语义——按目标位宽补码回绕，不报错
                try:
                    _v = int(ndarray(data, _dtype=str(data.dtype)).item())
                except Exception:
                    _v = int(data)
                _bits_s = 8 * _INT_ITEMSIZES[_dtype]
                _v = _v & ((1 << _bits_s) - 1)
                if not _dtype.startswith('uint') and _v >= 1 << (_bits_s - 1):
                    _v -= 1 << _bits_s
                data = _v
            else:
                # 标量 Python int → 窄整数 dtype：越界报 OverflowError
                # （负值 → 无符号目标按补码回绕，与 numpy 一致）
                lo, hi = _INT_RANGES[_dtype]
                v = data
                if _dtype.startswith('uint') and v < 0:
                    v += 1 << (8 * _INT_ITEMSIZES[_dtype])
                if v < lo or v > hi:
                    raise OverflowError(
                        f"Python int too large to convert to C integer")
    # 字符串/字节/void dtype（S/U/V）：存为原始 Python 数据，避免数值化。
    if dtype is not None:
        sd = _parse_str_dtype(dtype)
        if sd is None and isinstance(dtype, type) and issubclass(dtype, DType):
            sd = ('S', 0)
        if sd is not None:
            _s_shape = None
            _s_scalar = False
            if _is_ndarray(data):
                flat = _pyflat(data.tolist())
            elif isinstance(data, (list, tuple)):
                # numpy 只解包一层 array-like：ndarray/__array__ 元素展开为其元素
                _exp = []
                for _it in data:
                    if _is_ndarray(_it):
                        _t = _it.tolist()
                        _exp.append(_pyflat(_t) if isinstance(_t, list) else [_t])
                    elif (not isinstance(_it, (list, tuple, str, bytes))
                          and hasattr(_it, '__array__')):
                        try:
                            _inner = _it.__array__()
                        except Exception:
                            _inner = _it
                        if _is_ndarray(_inner):
                            _t = _inner.tolist()
                            _exp.append(_pyflat(_t) if isinstance(_t, list) else [_t])
                        elif isinstance(_inner, (list, tuple)):
                            _exp.extend(_inner)
                        else:
                            _exp.append(_inner)
                    elif (isinstance(_it, memoryview)
                          or (not isinstance(_it, (list, tuple, str, bytes,
                                                   int, float, complex, bool))
                              and (hasattr(_it, '__array_interface__')
                                   or hasattr(_it, '__array_struct__')
                                   or hasattr(_it, '__buffer__')))):
                        # buffer/数组接口对象：解包一层（numpy 语义）
                        try:
                            _inner = np.array(_it)
                            _iv = _inner.tolist()
                            _mv = _pyflat(_iv) if isinstance(_iv, list) else [_iv]
                        except Exception:
                            _mv = [_it]
                        if _mv and builtins.all(
                                not isinstance(v, (list, tuple)) for v in _mv):
                            _exp.append(_mv)
                        else:
                            _exp.extend(_mv)
                    else:
                        _exp.append(_it)
                if _exp and builtins.all(isinstance(r, list) for r in _exp) and _exp[0]:
                    _w0 = len(_exp[0])
                    if builtins.all(len(r) == _w0 for r in _exp):
                        _s_shape = (len(_exp), _w0)
                        flat = [v for r in _exp for v in r]
                    else:
                        flat = _flatten_data(_exp)
                else:
                    flat = _flatten_data(_exp)
            else:
                flat = [data]
                _s_scalar = True
            kind, width = sd
            if width == 0:
                # numpy 的裸 'S'/'U'：按数据推断最小宽度
                if kind == 'S':
                    width = _py_max(
                        (len(v) if isinstance(v, (bytes, bytearray)) else len(str(v)) for v in flat),
                        default=0)
                else:
                    width = _py_max((len(str(v)) for v in flat), default=0)
                sd = (kind, width)
            arr = ndarray._wrap(np.zeros((len(flat),)), _dtype='string_', _raw_data=list(flat))
            if _s_shape is not None and _s_shape != (len(flat),):
                arr = arr.reshape(_s_shape)
            if _s_scalar:
                arr = arr.reshape(())
            # reshape 返回新对象：_str_dtype 必须在最终对象上设置
            arr._str_dtype = sd
            return arr
        # 结构化 dtype：统一走富 DType 模型
        _dt_obj = _as_struct_dtype(dtype)
        if _dt_obj is not None:
            return _make_structured_array(data, _dt_obj)
    arr = _coerce_dtype(ndarray(data, _dtype=_dtype), _dtype)
    # 显式复数 dtype：即使输入已是实数 ndarray（__init__ 会沿用其 dtype），
    # 也需提升为原生复数（零虚部）。
    if _dtype in ('complex128', 'complex64') and not getattr(arr._array, 'is_complex', False):
        arr = ndarray._wrap(_maybe_native_complex(arr._array, _dtype), _dtype=_dtype)
    if ndmin > arr.ndim:
        new_shape = (1,) * (ndmin - arr.ndim) + arr.shape
        arr = ndarray._wrap(arr._array.reshape(new_shape), _dtype=_dtype)
    return arr


def _infer_scalar_dtype(a):
    """推断标量（非 list/tuple/ndarray）输入的 dtype。

    rsnumpy 标量保留自身 dtype（asarray(np.int64(2)) 不应退化为 float64）；
    Python complex 推断为 complex128（不能丢虚部）；其余 Python 标量保持
    原有 float64 行为。
    """
    dt = getattr(a, "dtype", None)
    if dt is not None and not _is_ndarray(a):
        try:
            name = str(dt)
            if name:
                return name
        except Exception:
            pass
    if isinstance(a, builtins.complex):
        return "complex128"
    return "float64"


def _uniform_scalar_dtype(f):
    """嵌套 list/tuple 中所有元素都是同一 dtype 的 rsnumpy 标量时返回该 dtype 名，
    否则返回 None（回退值推断）。混合 Python 数值时不启用（numpy 会按值提升）。"""
    dts = []

    def rec(x):
        if _is_ndarray(x):
            return False
        if isinstance(x, (list, tuple)):
            return builtins.all(rec(v) for v in x)
        if ((isinstance(x, (int, float, complex)) or isinstance(x, generic))
                and not isinstance(x, builtins.bool)
                and getattr(x, 'dtype', None) is not None):
            try:
                dts.append(str(x.dtype))
                return True
            except Exception:
                return False
        return False

    if not rec(f):
        return None
    if not dts or builtins.any(d != dts[0] for d in dts):
        return None
    return dts[0]


def _scalar_leaves_to_py(x):
    """嵌套结构中 rsnumpy 标量叶子 → Python 原生值（Rust 构造不接受标量叶子元素）。"""
    if isinstance(x, (list, tuple)):
        return [_scalar_leaves_to_py(v) for v in x]
    if isinstance(x, bool_):
        return builtins.bool(x)
    if isinstance(x, complex):
        return builtins.complex(x)
    if isinstance(x, float):
        return float(x)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, generic) and hasattr(x, 'tolist'):
        return x.tolist()
    return x


def _has_rscalar_leaf(x):
    """嵌套 list/tuple 中是否含 rsnumpy 标量叶子（非 ndarray）。"""
    if isinstance(x, (list, tuple)):
        return builtins.any(_has_rscalar_leaf(v) for v in x)
    return (isinstance(x, generic) and not _is_ndarray(x)
            and hasattr(x, 'tolist'))


def asarray(a, dtype=None, order=None):
    """转换输入为数组。"""
    if _is_ndarray(a):
        if dtype is None:
            return a
        # An explicit dtype must cast even when the input is already an
        # ndarray (matching np.asarray semantics).
        return array(a, dtype=dtype, copy=False)
    if dtype is not None:
        # 显式 dtype：委托给 array()。array() 里统一处理字符串 (S/U/V) 与结构化
        # dtype；若在此处自行 _resolve_dtype + _coerce_dtype，字符串数据会被
        # 错误地强制成 float64 数组（MATLAB 字符数组转换失败的根因）。
        return array(a, dtype=dtype, copy=False)
    else:
        # 数值 list/tuple 快速路径：Rust 单次完成展平 + dtype 推断 + 构造。
        if isinstance(a, (list, tuple)):
            _scalar_dt = _uniform_scalar_dtype(a)
            if _scalar_dt is not None:
                return asarray(_scalar_leaves_to_py(a), dtype=_scalar_dt)
            try:
                raw, code = np.build_array(a)
            except (ValueError, TypeError):
                raw = None
            if raw is not None:
                return ndarray._wrap(raw, _dtype=_BUILD_ARRAY_DTYPES[code])
            flat, _, _ = _flatten_check(a)
            _dtype = _infer_int_dtype(flat)
        else:
            _dtype = _infer_scalar_dtype(a)
    return _coerce_dtype(ndarray(a, _dtype=_dtype), _dtype)


def _check_int_coerce(data, dt):
    """numpy 语义：列表输入 + 显式整数 dtype 时，NaN 抛 ValueError、越界抛 OverflowError。"""
    if dt not in _INT_DTYPES:
        return
    try:
        if _contains_ndarray(data):
            vals = _pyflat(_convert_ndarray_deep(data))
        else:
            vals = _pyflat(data)
    except Exception:
        return
    lo, hi = _INT_RANGES[dt]
    for v in vals:
        if isinstance(v, float) and v != v:
            raise ValueError("cannot convert float NaN to integer")
        if isinstance(v, (int, float, generic)) and not isinstance(v, builtins.bool):
            if int(v) < lo or int(v) > hi:
                raise OverflowError(f"Python int too large to convert to {dt}")


def _coerce_dtype(arr, _dtype):
    """ndarray.__init__ 会沿用输入 dtype，这里把结果强制为请求的类型。"""
    if _dtype is None:
        return arr
    # float→int 转换的 NaN→0 统一处理（numpy 未定义行为，保证
    # np.array(x, dtype=int) 与 np.array(x).astype(int) 两条路径值一致）
    if _dtype in _INT_DTYPES:
        _storage = str(getattr(getattr(arr, '_array', None), 'dtype', '') or '')
        if _storage in ('float16', 'float32', 'float64'):
            try:
                flat = _pyflat(arr.tolist())
                if builtins.any(isinstance(v, float) and v != v for v in flat):
                    newflat = [0 if (isinstance(v, float) and v != v) else v for v in flat]
                    if getattr(arr, 'ndim', 1) == 0:
                        return ndarray(newflat[0], _dtype=_dtype)
                    new = ndarray(newflat, _dtype=_dtype)
                    if new.ndim < arr.ndim and arr.shape:
                        new = new.reshape(arr.shape)
                    return new
            except Exception:
                pass
    if getattr(arr, '_dtype', None) == _dtype:
        # 标签已是目标类型，但底层存储是不同宽度的整数时仍需真实转换
        # （如 uint64 值装入 int8：numpy 语义为回绕）
        _storage = str(getattr(getattr(arr, '_array', None), 'dtype', '') or '')
        if _storage in ('int8', 'int16', 'int32', 'int64',
                        'uint8', 'uint16', 'uint32', 'uint64') and _storage != _dtype:
            try:
                return arr.astype(_dtype)
            except Exception:
                return arr
        # f64 存储 + f32/f16 标签：真正舍入到目标精度（NEP50 语义）
        if _dtype in ('float32', 'float16') and _storage in ('float64', 'float32'):
            try:
                return arr.astype(_dtype)
            except Exception:
                pass
        return arr
    if _dtype.startswith(('complex', 'U', 'S')):
        return arr
    # 复数 → 非复数：Rust 路径会丢虚部 → 值存入 _complex_data 保留
    if str(getattr(arr, '_dtype', '')).startswith('complex') and \
            not str(_dtype).startswith('complex'):
        try:
            _t_co = arr.tolist()
            _fl_co = _pyflat(_t_co) if isinstance(_t_co, list) else [_t_co]
            arr._complex_data = [complex(v) for v in _fl_co]
            arr._dtype = _dtype
            return arr
        except Exception:
            pass
    try:
        return arr.astype(_dtype)
    except (TypeError, ValueError):
        return arr


def asanyarray(a, dtype=None, order=None):
    """转换输入为 ndarray。"""
    return asarray(a, dtype, order)


class matrix(ndarray):
    """NumPy 的 ``np.matrix`` 等价物：二维 ndarray 子类。

    保留 0-d/1-d 输入自动升为 2-d 的 NumPy 语义；``isinstance``/``issubclass``
    检查因此可以正常工作。算术运算沿用基类（逐元素）语义。
    """

    def __init__(self, data=None, dtype=None, copy=True):
        if isinstance(data, str):
            rows = data.split(';')
            data = [[float(e.strip()) for e in row.split(',')] for row in rows]
        arr = asarray(data, dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            raise ValueError("matrix must be 2-dimensional")
        super().__init__(arr)


def asmatrix(data, dtype=None):
    """将输入转换为矩阵对象。"""
    if isinstance(data, matrix) and (dtype is None or data.dtype == dtype):
        return data
    return matrix(data, dtype=dtype)


def bmat(obj, ldict=None, gdict=None):
    """从字符串、嵌套序列或数组构建矩阵。"""
    if isinstance(obj, str):
        obj = obj.strip()
        if obj.startswith('[') and obj.endswith(']'):
            obj = obj[1:-1].strip()
        rows = obj.split(';')
        data = []
        for row in rows:
            elements = row.split(',')
            row_data = []
            for e in elements:
                e = e.strip()
                if e.startswith('[') and e.endswith(']'):
                    inner = e[1:-1].split()
                    row_data.extend([float(x) for x in inner])
                else:
                    row_data.append(float(e))
            data.append(row_data)
        return asarray(data)
    if isinstance(obj, (list, tuple)):
        if isinstance(obj[0], (list, tuple)):
            if isinstance(obj[0][0], (list, tuple)):
                rows = []
                for row_block in obj:
                    row_len = sum(len(block[0]) for block in row_block)
                    # current_row = [0.0] * row_len
                    col_start = 0
                    for block in row_block:
                        b_arr = _to_ndarray(block)
                        b_rows, b_cols = b_arr.shape
                        for i in range(b_rows):
                            if i >= len(rows):
                                rows.append([0.0] * row_len)
                            for j in range(b_cols):
                                rows[i][col_start + j] = b_arr[i, j]
                        col_start += b_cols
                return asarray(rows)
            else:
                return asarray(obj)
        else:
            return asarray(obj)
    return _to_ndarray(obj)


def repmat(a, m, n):
    """将数组或矩阵重复 m 行 n 列。"""
    a = _to_ndarray(a)
    if a.ndim == 1:
        a = a.reshape(1, a.size)
    return tile(a, (m, n))


def _to_ndarray(obj):
    """将任意数组类对象转换为 rsnumpy ndarray（不依赖第三方库）。"""
    if _is_ndarray(obj):
        return obj
    if hasattr(obj, 'tolist'):
        return ndarray(obj.tolist())
    return ndarray(obj)


def _pyflat(x):
    """递归展平 tolist() 结果（0-d 标量 / 多维嵌套列表 / 结构化元组）。"""
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_pyflat(v))
        return out
    return [x]


def array_equal(a, b, equal_nan=False):
    """判断两个数组是否形状和元素完全相同（equal_nan=True 时 NaN==NaN 视为相等）。"""
    a = _to_ndarray(a)
    b = _to_ndarray(b)
    if a.shape != b.shape:
        return False
    la = _pyflat(a.tolist())
    lb = _pyflat(b.tolist())
    if len(la) != len(lb):
        return False
    for x, y in zip(la, lb):
        try:
            same = bool(x == y)
        except Exception:
            same = False
        if same:
            continue
        if equal_nan and x != x and y != y:
            continue
        return False
    return True


def copy(a, order='K', thread_safe=False):
    """返回数组的副本。

    Args:
        a: 输入数组。
        order: 内存布局顺序（'C', 'F', 'A', 'K'）。
        thread_safe: 是否创建线程安全的副本（用于多线程环境）。

    Returns:
        ndarray: 数组副本。若 thread_safe=True，返回的副本可安全在多线程间共享。

    【线程安全示例】
    >>> import rsnumpy as np
    >>> import threading
    >>> data = np.array([1, 2, 3])
    >>> # 创建线程安全副本
    >>> safe_data = np.copy(data, thread_safe=True)
    >>> def worker(arr):
    ...     return arr.sum()
    >>> threads = [threading.Thread(target=worker, args=(safe_data,)) for _ in range(4)]
    >>> for t in threads:
    ...     t.start()
    >>> for t in threads:
    ...     t.join()
    """
    return ndarray(a).copy(order=order, thread_safe=thread_safe)


def _parse_str_dtype(dtype):
    """解析字符串类 dtype，返回 (kind, width)；非字符串 dtype 返回 None。

    kind: 'S'（字节串，1 字节/元素）、'U'（Unicode，4 字节/元素）、
    'V'（原始字节）。width 为元素长度（字符数），未指定为 0。
    """
    if dtype is bytes:
        return ('S', 0)
    if dtype is str:
        return ('U', 0)
    if not isinstance(dtype, str):
        return None
    s = dtype.lstrip('<>=|')
    if not s:
        return None
    k = s[0]
    if k in ('S', 'U', 'V', 'a'):
        kind = 'S' if k == 'a' else k
        rest = s[1:]
        width = int(rest) if rest.isdigit() else 0
        return (kind, width)
    return None


def _string_array_tobytes(items, str_dtype):
    """将字符串/字节元素列表按 (kind, width) 编码为定宽补零字节串。"""
    kind, width = str_dtype
    out = bytearray()
    for v in items:
        if kind == 'U':
            if isinstance(v, (bytes, bytearray)):
                v = bytes(v).decode('latin-1')
            enc = str(v).encode('utf-32-le')
            size = width * 4 if width else len(enc)
        else:
            if isinstance(v, (bytes, bytearray)):
                enc = bytes(v)
            else:
                enc = str(v).encode('latin-1')
            size = width if width else len(enc)
        enc = enc[:size].ljust(size, b'\x00')
        out += enc
    return bytes(out)


def _str_scalar(v, str_dtype):
    """按 (kind, width) 将单个原始元素转换为标量：
    S→截断到 width 并去尾零的 bytes；U→去尾零的 str；V→定宽补零的 void。
    """
    kind, width = str_dtype
    if kind == 'U':
        s = bytes(v).decode('latin-1') if isinstance(v, (bytes, bytearray)) else str(v)
        if width:
            s = s[:width]
        return s.rstrip('\x00')
    bs = bytes(v) if isinstance(v, (bytes, bytearray)) else str(v).encode('latin-1')
    if kind == 'V':
        if width:
            bs = bs[:width].ljust(width, b'\x00')
        return void(bs)
    # kind == 'S'
    if width:
        bs = bs[:width]
    return bs.rstrip(b'\x00')


_TOBYTES_STRUCT = {
    'float64': 'd', 'float32': 'f',
    'int64': 'q', 'int32': 'i', 'int16': 'h', 'int8': 'b',
    'uint64': 'Q', 'uint32': 'I', 'uint16': 'H', 'uint8': 'B',
    'bool': '?',
}


def _numeric_tobytes(arr, order='C'):
    """将数值数组按 dtype 小端布局编码为字节串。"""
    import struct
    dt = getattr(arr, '_dtype', 'float64')
    # 快路径：float64 数组的字节布局与底层存储一致，直接从缓冲协议 memcpy，
    # 避免逐元素 tolist()→float()→struct.pack 的 Python 开销（百万点级差异达百毫秒）。
    if (dt == 'float64' and getattr(arr, '_raw_data', None) is None and getattr(arr, '_complex_data', None) is None):
        return memoryview(arr._array).tobytes(order)
    fmt = _TOBYTES_STRUCT.get(dt, 'd')
    flat = _flatten_data(arr.tolist())
    if fmt == '?':
        return bytes(1 if v else 0 for v in flat)
    if fmt in ('d', 'f'):
        vals = [float(v) for v in flat]
    else:
        vals = [int(v) for v in flat]
    return struct.pack('<%d%s' % (len(vals), fmt), *vals)


def _resolve_dtype(dtype):
    """解析 dtype 字符串/Types/DType 为内部表示。"""
    if dtype is None:
        return "float64"
    if isinstance(dtype, DType):
        return dtype.name
    # DType 类本身（type(np.dtype(...))）：按裸 S 处理（宽度由数据推断）
    if isinstance(dtype, type) and issubclass(dtype, DType):
        return "bytes_"
    if isinstance(dtype, (list, tuple)):
        return "void"
    if isinstance(dtype, str):
        dt_str = dtype
    else:
        _nm = getattr(dtype, 'name', None)
        if not isinstance(_nm, str):
            _nm = getattr(dtype, '__name__', None)
        dt_str = _nm if isinstance(_nm, str) else str(dtype)
    if dt_str in ("complex", "complex128", "complex64", "cfloat", "cdouble"):
        return "complex128"
    if dt_str in ("int", "int_", "intp", "intc"):
        return "int64"
    if dt_str in ("float", "double", "float64", "floating", "number"):
        return "float64"
    if dt_str in ("float32", "single"):
        return "float32"
    if dt_str in ("float16", "half"):
        return "float16"
    if dt_str in ("int8", "int16", "int32", "int64"):
        return dt_str
    if dt_str in ("uint", "uint8", "uint16", "uint32", "uint64"):
        return dt_str
    if dt_str in ("bool", "bool_"):
        return "bool"
    # 单字符类型码（numpy typecode）
    _CHARCODES = {'b': 'int8', 'h': 'int16', 'i': 'int32', 'l': 'int64',
                  'q': 'int64', 'p': 'int64', 'B': 'uint8', 'H': 'uint16',
                  'I': 'uint32', 'L': 'uint64', 'Q': 'uint64',
                  'e': 'float16', 'f': 'float32', 'd': 'float64',
                  'F': 'complex64', 'D': 'complex128', '?': 'bool'}
    if dt_str in _CHARCODES:
        return _CHARCODES[dt_str]
    if dt_str in _PREFIX_DTYPE_NAMES:
        return _PREFIX_DTYPE_NAMES[dt_str]
    # 非结构化 void：'V0'/'V3'/'>V3' 等保留原样作标签
    _core_dt = dt_str[1:] if dt_str and dt_str[0] in '<>|=' else dt_str
    if len(_core_dt) >= 2 and _core_dt[0] in 'vV' and _core_dt[1:].isdigit():
        return _core_dt
    if dt_str in ("object", "object_", "O"):
        import warnings
        warnings.warn(
            "使用 object dtype 会完全失去向量化性能优势！\n"
            "推荐替代方案：\n"
            "  1. 变长字符串 → 使用固定长度字符串 dtype（如 'U10'）\n"
            "  2. 大整数（<2^53）→ 使用 int64\n"
            "  3. 混合类型 → 使用结构化数组或 Python 列表",
            UserWarning,
            stacklevel=2,
        )
        return "object"
    # 裸 'S'/'U'（无宽度）：保留 kind 标记，宽度由数据推断
    if dt_str in ("S", "bytes"):
        return "bytes_"
    if dt_str in ("U", "str", "string"):
        return "string_"
    return "float64"


def _infer_int_dtype(args):
    """推断参数中的类型：全 bool→bool，全 int→int64，否则 float64。"""
    has_bool = False
    has_int = False
    for a in args:
        if isinstance(a, bool):
            has_bool = True
        elif isinstance(a, int):
            has_int = True
        else:
            return "float64"
    if has_bool and not has_int:
        return "bool"
    return "int64"


def _infer_smart_dtype(flat):
    """智能类型推断：避免不必要的 object dtype。

    策略：
    1. 全 bool → bool
    2. 全 int（在 int64 范围内）→ int64
    3. 全 int（超出范围）→ float64（精度警告）
    4. 全 float → float64
    5. int + float → float64
    6. 其他混合类型 → object（给出警告）
    """
    if not flat:
        return "float64"

    # NOTE: the module-level names ``bool``/``sum`` shadow the builtins, so
    # canonicalise element types against ``builtins`` explicitly.  Scalar
    # types created by ``_make_scalar_type`` subclass int/float/complex and
    # must fold into the corresponding builtin category.
    def _canon_type(x):
        if isinstance(x, (builtins.bool, bool_)):
            return builtins.bool
        if isinstance(x, builtins.int):
            return builtins.int
        if isinstance(x, builtins.float):
            return builtins.float
        if isinstance(x, builtins.str):
            return builtins.str
        return type(x)

    types = set(_canon_type(x) for x in flat)

    # 全 bool
    if types == {builtins.bool}:
        return "bool"

    # 全 int 或 int+bool
    if types <= {builtins.int, builtins.bool}:
        int_vals = [x for x in flat
                    if isinstance(x, builtins.int)
                    and not isinstance(x, (builtins.bool, bool_))]
        if int_vals:
            min_val = min(int_vals)
            max_val = max(int_vals)
            if min_val >= -(2**53) and max_val < 2**53:
                return "int64"
            else:
                import warnings
                warnings.warn(
                    f"整数值范围 [{min_val}, {max_val}] 超出 ±2^53，"
                    f"将使用 float64 存储（可能丢失精度）。",
                    UserWarning,
                    stacklevel=3,
                )
                return "float64"
        return "int64"

    # 全 float 或 float+int/bool
    if types <= {builtins.float, builtins.int, builtins.bool}:
        return "float64"

    # 全字符串
    if types == {builtins.str}:
        str_lens = [len(x) for x in flat]
        max_len = max(str_lens) if str_lens else 1
        return f"U{max_len}"

    # 混合类型 → object（但尝试给出更好的建议）
    import warnings
    suggested = _suggest_better_dtype(flat)
    if suggested and suggested != "object":
        warnings.warn(
            f"数据包含多种类型 {types}，将使用 object dtype（性能低）。\n"
            f"建议：可以转换为 '{suggested}' 以获得更好性能。\n"
            f"使用方式：np.array(data, dtype='{suggested}')",
            UserWarning,
            stacklevel=3,
        )
    else:
        warnings.warn(
            f"数据包含多种类型 {types}，将使用 object dtype（性能低）。\n"
            f"推荐：使用结构化数组或 Python 列表替代。",
            UserWarning,
            stacklevel=3,
        )
    return "object"


def _suggest_better_dtype(flat):
    """为混合类型数据推荐更好的 dtype。"""
    types = set(type(x) for x in flat)

    # 尝试全部转为字符串
    if types <= {builtins.str, builtins.int, builtins.float, builtins.bool}:
        str_lens = [len(str(x)) for x in flat]
        max_len = max(str_lens) if str_lens else 1
        return f"U{max_len}"

    # 尝试全部转为 float
    try:
        for x in flat:
            float(x)
        return "float64"
    except (ValueError, TypeError):
        pass

    return None


def _make_structured_zeros(shape, fields):
    """为结构化 dtype 生成零填充的原始数据（子数组字段填充嵌套零）。"""
    elem = tuple(_nested_zeros(_field_subshape(f)) for f in fields)
    if isinstance(shape, int):
        total = shape
    else:
        total = 1
        for s in shape:
            total *= s
    flat = [elem] * total
    if isinstance(shape, int):
        shape = (shape,)
    if len(shape) <= 1:
        return flat
    # 重塑为多维
    result = flat
    for dim in reversed(shape[1:]):
        result = [result[i:i+dim] for i in range(0, len(result), dim)]
    return result


def _maybe_native_complex(raw, _dtype):
    """dtype 为复数时，把实数 Rust 数组提升为原生复数（零虚部）。"""
    if _dtype in ('complex128', 'complex64'):
        return raw + 0j
    return raw


def zeros(shape, dtype=None, order='C', buffer=None, *, device=None, like=None):
    """返回指定形状的零数组。"""
    dt_obj = _as_struct_dtype(dtype)
    if dt_obj is not None:
        return _wrap_structured(dt_obj, _make_struct_filled(shape, dt_obj, _zero_scalar_for))
    _dtype = _resolve_dtype(dtype)
    return ndarray(_maybe_native_complex(np.zeros(shape), _dtype), _dtype=_dtype)


def ones(shape, dtype=None, order='C', buffer=None, *, device=None, like=None):
    """返回指定形状的1数组。"""
    dt_obj = _as_struct_dtype(dtype)
    if dt_obj is not None:
        return _wrap_structured(dt_obj, _make_struct_filled(shape, dt_obj, _one_scalar_for))
    _dtype = _resolve_dtype(dtype)
    return ndarray(_maybe_native_complex(np.ones(shape), _dtype), _dtype=_dtype)


def empty(shape, dtype=None, order='C', buffer=None, *, device=None, like=None):
    """返回指定形状的空数组。"""
    dt_obj = _as_struct_dtype(dtype)
    if dt_obj is not None:
        return _wrap_structured(dt_obj, _make_struct_filled(shape, dt_obj, _zero_scalar_for))
    _dtype = _resolve_dtype(dtype)
    arr = ndarray(_maybe_native_complex(np.empty(shape), _dtype), _dtype=_dtype)
    arr._is_empty = True
    return arr


def full(shape, fill_value, dtype=None, order='C', *, device=None, like=None):
    """返回指定形状的填充数组。"""
    _dtype = _resolve_dtype(dtype)
    if isinstance(fill_value, complex) or _dtype in ('complex128', 'complex64'):
        c = complex(fill_value)
        raw = np.full(shape, c.real) + complex(0.0, c.imag)
        return ndarray(raw, _dtype='complex128' if _dtype not in ('complex128', 'complex64') else _dtype)
    return ndarray(np.full(shape, fill_value), _dtype=_dtype)


def _like_str_width(dtype):
    """like 函数中的 str/bytes 类或宽度 0 的 S/U dtype：numpy 语义取最小宽度 1。"""
    if dtype is str:
        return ('U', 1)
    if dtype is bytes:
        return ('S', 1)
    if isinstance(dtype, str):
        sd = _parse_str_dtype(dtype)
        if sd is not None and sd[1] == 0:
            return (sd[0], 1)
    return None


def zeros_like(a, dtype=None, order='K', subok=True, shape=None, device=None):
    """返回与输入形状相同的零数组。"""
    arr = ndarray(a)
    _ssd = _like_str_width(dtype)
    if _ssd is not None:
        obj = ndarray._wrap(np.zeros_like(arr._array), _dtype='string_')
        obj._str_dtype = _ssd
        return obj
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return ndarray(np.zeros_like(arr._array), _dtype=_dtype)


def ones_like(a, dtype=None, order='K', subok=True, shape=None, device=None):
    """返回与输入形状相同的1数组。"""
    arr = ndarray(a)
    _ssd = _like_str_width(dtype)
    if _ssd is not None:
        obj = ndarray._wrap(np.zeros_like(arr._array), _dtype='string_')
        obj._str_dtype = _ssd
        return obj
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return ndarray(np.ones_like(arr._array), _dtype=_dtype)


def empty_like(a, dtype=None, order='K', subok=True, shape=None, device=None):
    """返回与输入形状相同的空数组。"""
    arr = ndarray(a)
    _ssd = _like_str_width(dtype)
    if _ssd is not None:
        obj = ndarray._wrap(np.empty_like(arr._array), _dtype='string_')
        obj._str_dtype = _ssd
        return obj
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return ndarray(np.empty_like(arr._array), _dtype=_dtype)


def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None, device=None):
    """返回与输入形状相同的填充数组。"""
    arr = ndarray(a)
    _ssd = _like_str_width(dtype)
    if _ssd is not None:
        obj = ndarray._wrap(np.zeros_like(arr._array), _dtype='string_')
        obj._str_dtype = _ssd
        return obj
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return ndarray(np.full_like(arr._array, float(fill_value)), _dtype=_dtype)


def eye(N, M=None, k=0, dtype=None, order='C'):
    """返回对角线为1的二维数组。"""
    _dtype = _resolve_dtype(dtype)
    return ndarray(np.eye(N, M, k), _dtype=_dtype)


def identity(n, dtype=None):
    """返回单位矩阵。"""
    return eye(n, dtype=dtype)


def arange(start=0, stop=None, step=1, dtype=None, device=None):
    """返回给定间隔内均匀间隔的值。"""
    if stop is None:
        stop = start
        start = 0
    if isinstance(start, datetime64) or isinstance(stop, datetime64) \
            or isinstance(step, timedelta64):
        return _datetime_arange(start, stop, step)
    _dtype = _resolve_dtype(dtype) if dtype is not None else _infer_int_dtype((start, stop, step))
    return ndarray(np.arange(start, stop, step), _dtype=_dtype)


def searchsorted(a, v, side='left', sorter=None):
    """查找值在有序数组中应插入的位置。"""
    arr = ndarray(a)
    v_arr = ndarray(v)
    # 纯 Python 二分查找实现，不依赖 numpy
    import bisect
    a_flat = arr.ravel().tolist()
    if not isinstance(a_flat, list):
        a_flat = [a_flat]
    v_flat = v_arr.ravel().tolist()
    if not isinstance(v_flat, list):
        v_flat = [v_flat]
    result = []
    for val in v_flat:
        if side == 'left':
            result.append(bisect.bisect_left(a_flat, val))
        else:
            result.append(bisect.bisect_right(a_flat, val))
    return ndarray(result, _dtype='int64')


def _datetime_arange(start, stop, step):
    """datetime64/timedelta64 版本的 arange：以天数（自纪元起）生成浮点数组。

    产出的数组值遵循 matplotlib/rsplotlib 的日期数值约定，可直接绘图并被
    ConciseDateFormatter 正确格式化。
    """
    if isinstance(step, timedelta64):
        unit = step._unit
    elif isinstance(start, datetime64):
        unit = start._unit
    elif isinstance(stop, datetime64):
        unit = stop._unit
    else:
        unit = 'D'
    raw = np.arange(float(start), float(stop), float(step))
    return ndarray(raw, _dtype='datetime64[%s]' % unit)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, device=None):
    """返回指定间隔内均匀间隔的数字。"""
    start_val = _scalar(_ensure(start))
    stop_val = _scalar(_ensure(stop))
    _dtype = _resolve_dtype(dtype)
    result = ndarray(np.linspace(start_val, stop_val, num, endpoint=endpoint), _dtype=_dtype)
    if retstep:
        step = (stop_val - start_val) / (num - 1) if endpoint else (stop_val - start_val) / num
        return result, _float64(step)
    return result


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0, device=None):
    """返回对数刻度上均匀间隔的数字。"""
    _dtype = _resolve_dtype(dtype) if dtype else "float64"
    return ndarray(np.logspace(float(start), float(stop), num, float(base)), _dtype=_dtype)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0, device=None):
    """返回几何级数上均匀间隔的数字。"""
    if start <= 0 or stop <= 0:
        raise ValueError("geomspace requires positive start and stop values")
    _dtype = _resolve_dtype(dtype) if dtype else "float64"
    return ndarray(np.geomspace(float(start), float(stop), num), _dtype=_dtype)


# ========== 索引函数 ==========

def ix_(*args):
    """从多个序列构造开放网格，用于笛卡尔积花式索引。

    返回 ndarray 元组，每个数组沿对应维度广播。

    示例:
        >>> x = np.arange(32).reshape((8, 4))
        >>> x[np.ix_([1,5,7,2], [0,3,1,2])]
    """
    # Rust 层的 ix_rs 只解析 Python list/tuple，故先把 ndarray 等参数
    # 归一化为一维下标列表；布尔序列按 rsnumpy 语义取 True 位置的下标。
    prepared = []
    for a in args:
        dtype = getattr(a, '_dtype', None)
        seq = a.tolist() if hasattr(a, 'tolist') else list(a)
        if dtype == 'bool' or (len(seq) > 0 and _py_all(isinstance(v, bool) for v in seq)):
            seq = [i for i, v in enumerate(seq) if v]
        else:
            seq = [int(v) for v in seq]
        prepared.append(seq)
    raw = np.ix_rs(tuple(prepared))
    return tuple(ndarray._wrap(r) for r in raw)


def where(condition, x=None, y=None):
    """根据条件返回元素或索引。"""
    if x is not None and y is not None:
        dtype = getattr(x, '_dtype', 'float64')
        fields = getattr(x, '_fields', None)
        raw_data = getattr(x, '_raw_data', None)
        result = np.where(_ensure(condition), _ensure(x), _ensure(y))
        return ndarray._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)
    return nonzero(condition)


def take(a, indices, axis=None, out=None, mode='raise'):
    """沿指定轴取元素。"""
    arr = ndarray(a)
    return arr.take(indices, axis)


def put(a, indices, values, mode='raise'):
    """设置指定位置的元素。"""
    arr = ndarray(a)
    arr.put(indices, values)


def select(condlist, choicelist, default=0):
    """根据条件列表选择值（Rust 单次遍历实现）。"""
    cond_arrays = [_ensure(c) for c in condlist]
    choice_arrays = [_ensure(c) for c in choicelist]
    result = np.select_rs(cond_arrays, choice_arrays, float(default))
    return ndarray(result)


def nonzero(a):
    """返回非零元素的索引，返回元组形式的数组（兼容NumPy）。"""
    # numpy 语义：先试鸭子类型 .nonzero()，失败（TypeError）时回退 __array__
    # （回退异常在 except 子句内抛出，保留 __context__ 链）
    if not _is_ndarray(a):
        if hasattr(a, 'nonzero') and callable(a.nonzero):
            try:
                return a.nonzero()
            except TypeError:
                a = a.__array__()
        elif hasattr(a, '__array__'):
            a = a.__array__()
    a = asarray(a)
    if a.ndim == 0:
        raise ValueError("Calling nonzero on 0d arrays is not allowed")
    if a.size == 0:
        _empty_raw = np.zeros((0,))
        return tuple(ndarray._wrap(_empty_raw, _dtype="int64") for _ in range(a.ndim))
    try:
        raw = np.nonzero_arrs(a._array)
    except TypeError:
        _empty_raw = np.zeros((0,))
        return tuple(ndarray._wrap(_empty_raw, _dtype="int64") for _ in range(a.ndim))
    return tuple(ndarray._wrap(arr, _dtype="int64") for arr in raw)


def argwhere(a):
    """查找非零元素的索引。"""
    return ndarray(np.argwhere(_ensure(a)))


def flatnonzero(a):
    """返回扁平化数组中非零元素的索引（numpy 语义：整数索引数组）。"""
    return ndarray._wrap(np.flatnonzero(_ensure(a)), _dtype="int64")


# ========== 缺失构造函数 ==========

def fromfunction(function, shape, *, dtype=None, **kwargs):
    """根据函数和形状创建数组。"""
    indices = np._arange_arrays(shape)
    grid = np.meshgrid(*indices, indexing='ij')
    return function(*grid, **kwargs)


# dtype → struct (格式符, 字节宽)。rsnumpy 内部为 f64，缓冲区按 dtype 解成
# float/complex 数值序列（整数也解为 float 值，与 numpy 数值比较语义一致）。
_STRUCT_FOR_DTYPE = {
    'bool': ('?', 1),
    'int8': ('b', 1), 'int16': ('h', 2), 'int32': ('i', 4), 'int64': ('q', 8),
    'uint8': ('B', 1), 'uint16': ('H', 2), 'uint32': ('I', 4), 'uint64': ('Q', 8),
    'float16': ('e', 2), 'float32': ('f', 4), 'float64': ('d', 8),
    'complex64': ('f', 4), 'complex128': ('d', 8),
}
# 带字节序前缀的 dtype 串（<i2、>f4、|u1、=i4 …）→ 归一化名
_PREFIX_DTYPE_NAMES = {
    'b1': 'int8', 'i1': 'int8', 'u1': 'uint8',
    'b2': 'int16', 'i2': 'int16', 'u2': 'uint16',
    'b4': 'int32', 'i4': 'int32', 'u4': 'uint32',
    'b8': 'int64', 'i8': 'int64', 'u8': 'uint64',
    'f2': 'float16', 'f4': 'float32', 'f8': 'float64',
    'c8': 'complex64', 'c16': 'complex128',
}


def _decode_buffer_bytes(buffer, dtype, count):
    """按 dtype 把字节缓冲解码为数值序列（支持小端/大端）。"""
    import struct as _struct
    dt = dtype
    bo = '<'  # 文件缓冲（WAV/MAT 等）默认小端；目标平台均为小端
    if isinstance(dt, str) and dt and dt[0] in '<>=':
        bo = '<' if dt[0] == '<' else ('>' if dt[0] == '>' else '=')
        name = _PREFIX_DTYPE_NAMES.get(dt[1:])
    else:
        name = _resolve_dtype(dt)
    info = _STRUCT_FOR_DTYPE.get(name) if name else None
    if info is None:
        return list(np.bytes_to_floats(buffer, count).tolist())
    char, size = info
    n = len(buffer) // size
    if count >= 0:
        n = _py_min(n, count)
    if n == 0:
        return []
    if char == '?':
        return [b != 0 for b in buffer[:n]]
    if name.startswith('complex'):
        flat = list(_struct.unpack(bo + char * (2 * n), buffer[:n * 2 * size]))
        return [flat[i] + flat[i + 1] * 1j for i in range(0, len(flat), 2)]
    return list(_struct.unpack(bo + char * n, buffer[:n * size]))


def frombuffer(buffer, dtype=None, count=-1, offset=0, *, like=None):
    """从缓冲区创建一维数组（按 dtype 解码字节）。"""
    if isinstance(buffer, (bytes, bytearray, memoryview)):
        buffer = bytes(buffer)
        if offset > 0:
            buffer = buffer[offset:]
        if isinstance(dtype, str) and len(dtype) > 1 and dtype[0] == 'S' and dtype[1:].isdigit():
            # Byte string type (e.g. S1, S2, ...)
            elem_size = int(dtype[1:])
            total = len(buffer) // elem_size
            if count >= 0:
                total = _py_min(total, count)
            raw = [buffer[i * elem_size:(i + 1) * elem_size] for i in range(total)]
            return ndarray(np.zeros((len(raw),)), _dtype=dtype, _raw_data=raw)
        values = _decode_buffer_bytes(buffer, dtype, count)
        return ndarray(values)
    return ndarray(list(buffer))


def _dtype_itemsize(dtype):
    """返回 dtype 的元素字节大小（complex 为对应 float 的两倍）。"""
    dt = dtype
    if isinstance(dt, str) and dt and dt[0] in '<>=':
        name = _PREFIX_DTYPE_NAMES.get(dt[1:])
    else:
        name = _resolve_dtype(dt)
    info = _STRUCT_FOR_DTYPE.get(name) if name else None
    if info is None:
        return 8  # 默认按 float64
    _char, size = info
    if name and name.startswith('complex'):
        return size * 2
    return size


def fromfile(file, dtype=None, count=-1, offset=0, *, like=None):
    """从文件名/文件对象读取一维数组（按 dtype 解码，offset/count 以元素计）。

    与 numpy.fromfile 语义一致：offset 为跳过的元素数，count 为读取的元素数
    （-1 表示读到结尾）。
    """
    import os as _os
    elem_size = _dtype_itemsize(dtype)
    skip = offset * elem_size
    nbytes = count * elem_size if count >= 0 else None
    if isinstance(file, (str, bytes, _os.PathLike)):
        with open(file, 'rb') as f:
            if skip:
                f.seek(skip)
            raw = f.read(nbytes)
    elif hasattr(file, 'read'):
        if skip:
            file.read(skip)
        raw = file.read(nbytes)
    else:
        raise TypeError("file must be a filename or a file-like object")
    return frombuffer(raw, dtype=dtype)


def fromiter(iterable, dtype=None, count=-1, *, like=None):
    """从可迭代对象创建一维数组。"""
    # numpy 语义：裸 'S'/'U'/'V'（无宽度）或零宽度必须显式指定长度
    # （必须在 _resolve_dtype 之前检查，避免 'S' 被归一成 float64）
    if isinstance(dtype, str):
        _sd_ft = _parse_str_dtype(dtype)
        if _sd_ft is not None and _sd_ft[1] == 0:
            raise ValueError(
                f"Must specify length for dtype '{dtype}'")
    # numpy 语义：fromiter 不支持结构化/子数组 dtype
    try:
        _d_ft = dtype(dtype)
        if isinstance(_d_ft, DType) and (
                _d_ft._subdtype is not None or _d_ft._names is not None):
            raise ValueError(
                "fromiter does not support structured or subarray dtypes")
    except ValueError:
        raise
    except Exception:
        pass
    _dtype = _resolve_dtype(dtype)
    if count >= 0:
        data = []
        for _ in range(count):
            try:
                data.append(next(iterable))
            except StopIteration:
                raise ValueError(
                    f"iterator too short: Expected {count} but iterator "
                    f"had only {len(data)} items.")
    else:
        data = list(iterable)
    # numpy 语义：None 无法写入整型/布尔元素
    if _dtype in _INT_DTYPES or _dtype == 'bool':
        for _v in data:
            if _v is None:
                raise TypeError(
                    f"cannot convert None to {_dtype} in fromiter")
    return ndarray(data, _dtype=_dtype)


class _RClass:
    """行连接辅助类，模拟 np.r_。"""
    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        arrays = []
        for it in item:
            arrays.append(ndarray(it))
        if len(arrays) == 0:
            return ndarray([])
        return concatenate(arrays)


r_ = _RClass()


class _CClass:
    """列连接辅助类，模拟 np.c_。"""
    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        arrays = [ndarray(it) for it in item]
        if _py_all(a.ndim <= 1 for a in arrays):
            return column_stack(arrays)
        return concatenate(arrays, axis=1)


c_ = _CClass()


class _SClass:
    """切片辅助类，模拟 np.s_。"""
    def __getitem__(self, item):
        return item


s_ = _SClass()


def mgrid(*ranges):
    """密集网格索引，返回密集网格数组。"""
    arrays = []
    for r in ranges:
        if isinstance(r, slice):
            arrays.append(ndarray(arange(r.start or 0, r.stop, r.step or 1)))
        elif isinstance(r, (int, float)):
            arrays.append(ndarray([float(r)]))
        else:
            arrays.append(ndarray(r))
    grids = np.meshgrid(*arrays, indexing='ij')
    if isinstance(grids, tuple) and len(grids) > 1:
        return stack(grids, axis=0)
    return grids


def ogrid(*ranges):
    """开放网格索引，返回开放网格数组。"""
    arrays = []
    for r in ranges:
        if isinstance(r, slice):
            arrays.append(ndarray(arange(r.start or 0, r.stop, r.step or 1)))
        elif isinstance(r, (int, float)):
            arrays.append(ndarray([float(r)]))
        else:
            arrays.append(ndarray(r))
    return np.meshgrid(*arrays, indexing='ij')


def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    """从坐标向量返回坐标矩阵。

    默认使用 'xy' 索引（与 rsnumpy 一致），返回一维输入数组两两组合的网格。
    """
    arrays = []
    for x in xi:
        if hasattr(x, 'tolist'):
            arrays.append([int(v) for v in x.tolist()])
        elif isinstance(x, (list, tuple)):
            arrays.append([int(v) for v in x])
        else:
            arrays.append([int(x)])
    
    n = len(arrays)
    if n == 0:
        return []
    
    shapes = [len(a) for a in arrays]
    
    if sparse:
        result = []
        for i in range(n):
            if i == 0:
                grid = ndarray([[arrays[i][j]] for j in range(shapes[i])])
            else:
                grid = ndarray([arrays[i]])
            result.append(grid)
        return result
    
    result = []
    
    for i in range(n):
        if indexing == 'xy' and n == 2:
            if i == 0:
                grid = ndarray([[arrays[0][k] for k in range(shapes[0])] for _ in range(shapes[1])])
            else:
                grid = ndarray([[arrays[1][j] for _ in range(shapes[0])] for j in range(shapes[1])])
        elif indexing == 'ij':
            if n == 2:
                if i == 0:
                    grid = ndarray([[arrays[0][j] for _ in range(shapes[1])] for j in range(shapes[0])])
                else:
                    grid = ndarray([[arrays[1][k] for k in range(shapes[1])] for _ in range(shapes[0])])
            elif n == 3:
                grid = ndarray([[[arrays[i][j] if i == 0 else (arrays[i][k] if i == 1 else arrays[i][l_])
                                 for l_ in range(shapes[2])]
                                for k in range(shapes[1])]
                               for j in range(shapes[0])])
            else:
                grid = ndarray([[arrays[i][j] for _ in range(shapes[1])] for j in range(shapes[0])])
        else:
            grid = ndarray([[arrays[i][j] for _ in range(shapes[1])] for j in range(shapes[0])])
        result.append(grid)
    
    return result


# ========== FFT ==========
# FFT 相关函数均由子模块 `rsnumpy.fft` 提供（与 rsnumpy 一致，
# 通过 `np.fft.fft(...)` 或 `from rsnumpy.fft import fft` 使用）。
# 子模块的绑定见文件末尾的“子模块”一节。


# ========== 常量 ==========
pi = 3.141592653589793
e = 2.718281828459045
euler_gamma = 0.5772156649015328606
inf = float('inf')
nan = float('nan')
newaxis = None


# 字符串操作模块
char = _char_module


# ========== 判断函数 ==========

class _NdFlatIter:
    """ndarray.flat 的扁平迭代器，逐元素访问数组。"""
    def __init__(self, arr):
        self._arr = arr
        self._flat = arr._array.flatten()
        self._size = self._flat.size
        self._idx = 0
        self._is_int = getattr(arr, '_dtype', 'float64') == 'int64'

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= self._size:
            raise StopIteration
        val = self._flat[self._idx]
        self._idx += 1
        return int(val) if self._is_int else float(val)

    def _val(self, v):
        return int(v) if self._is_int else float(v)

    def __len__(self):
        """numpy flatiter 兼容：len() 返回扁平元素数。"""
        return self._size

    def copy(self, /):
        """numpy 兼容：返回底层数组的副本（numpy 中已弃用）。"""
        import warnings as _w_flat
        _w_flat.warn("flatiter.copy() is deprecated", DeprecationWarning,
                     stacklevel=2)
        return self._arr.copy()

    def __array__(self, /, dtype=None, copy=None):
        """numpy 兼容：返回底层数组（可按 dtype 转换）。"""
        _ = copy
        if dtype is None:
            return self._arr
        return self._arr.astype(dtype)

    # ---------- flatiter 索引辅助 ----------
    def _norm_flat(self, i):
        """归一化扁平索引并做越界检查（numpy 消息格式）。"""
        i = builtins.int(i)
        if i < 0:
            i += self._size
        if i < 0 or i >= self._size:
            raise IndexError(
                f"index {i} is out of bounds for flat iterator with size "
                f"{self._size}")
        return i

    def _sel_vals(self, idxs):
        vals = [self._val(self._flat[self._norm_flat(i)]) for i in idxs]
        return ndarray(vals).astype('int64') if self._is_int else ndarray(vals)

    def _take_all(self):
        return self._sel_vals(range(self._size))

    @staticmethod
    def _flat_list(l):
        out = []
        for v in l:
            if isinstance(v, list):
                out.extend(_NdFlatIter._flat_list(v))
            else:
                out.append(v)
        return out

    @staticmethod
    def _list_shape(l):
        s = []
        cur = l
        while isinstance(cur, list) and cur:
            s.append(builtins.len(cur))
            cur = cur[0]
        return builtins.tuple(s)

    @staticmethod
    def _raise_invalid_idx():
        raise IndexError(
            "only integers, slices (`:`), ellipsis (`...`) and "
            "integer or boolean arrays are valid indices")

    def __getitem__(self, idx):
        """numpy flatiter 兼容：整数/切片/布尔掩码/花式索引。"""
        import warnings as _w
        if idx is None:
            self._raise_invalid_idx()
        if idx is Ellipsis:
            return self._take_all()
        if isinstance(idx, tuple):
            if builtins.len(idx) == 0:
                return self._take_all()
            if builtins.len(idx) > 1:
                raise IndexError(
                    f"too many indices for flat iterator: flat iterator "
                    f"is 1-dimensional, but {builtins.len(idx)} were indexed")
            return self.__getitem__(idx[0])
        if isinstance(idx, builtins.bool):
            _w.warn("0-dimensional boolean index", DeprecationWarning,
                    stacklevel=2)
            if idx:
                return self._val(self._flat[0])
            return ndarray([]).astype('int64') if self._is_int else ndarray([])
        if isinstance(idx, slice):
            return self._sel_vals(range(*idx.indices(self._size)))
        if _is_ndarray(idx):
            _t = idx.tolist()
            _fi = _NdFlatIter._flat_list(_t) if isinstance(_t, list) else [_t]
            if getattr(idx, '_dtype', '') == 'bool':
                if builtins.len(_fi) != self._size:
                    raise IndexError(
                        "boolean index did not match indexed flat iterator")
                return self._sel_vals(
                    [i for i, m in builtins.enumerate(_fi) if m])
            _res = self._sel_vals(_fi)
            _shp = getattr(idx, 'shape', None)
            if _shp is not None and _res.ndim == 1 and builtins.len(_shp) > 1:
                _res = _res.reshape(_shp)
            return _res
        if isinstance(idx, _NdFlatIter):
            if idx._size > 0 and \
                    getattr(idx._arr, '_str_dtype', None) is not None:
                self._raise_invalid_idx()
            items = [idx[i] for i in range(idx._size)]
            for it in items:
                if isinstance(it, (str, bytes)):
                    self._raise_invalid_idx()
            return self._sel_vals(items)
        if isinstance(idx, list):
            if builtins.len(idx) == 0:
                return ndarray([]).astype('int64') if self._is_int \
                    else ndarray([])
            _fi = _NdFlatIter._flat_list(idx)
            if builtins.all(isinstance(v, builtins.bool) for v in _fi):
                raise IndexError(
                    "boolean indices for iterators are not supported")
            for v in _fi:
                if not isinstance(v, (builtins.int, builtins.float,
                                      builtins.bool)):
                    self._raise_invalid_idx()
            if builtins.any(isinstance(v, builtins.float) and
                            not isinstance(v, builtins.bool) for v in _fi):
                _w.warn("Invalid non-array indices for iterator objects are "
                        "deprecated", DeprecationWarning, stacklevel=2)
            res = self._sel_vals(
                [1 if isinstance(v, builtins.bool) else builtins.int(v)
                 for v in _fi])
            shape = _NdFlatIter._list_shape(idx)
            if builtins.len(shape) > 1:
                res = res.reshape(shape)
            return res
        return self._val(self._flat[self._norm_flat(idx)])

    def __setitem__(self, idx, val):
        """numpy flatiter 兼容：整数/切片/布尔掩码/花式赋值。"""
        import warnings as _w
        if idx is None:
            self._raise_invalid_idx()
        if idx is Ellipsis:
            for i in range(self._size):
                self._set(i, val)
            return
        if isinstance(idx, tuple):
            if builtins.len(idx) == 0:
                raise IndexError(
                    "Assigning to a flat iterator with a 0-D index")
            if builtins.len(idx) > 1:
                raise IndexError(
                    f"too many indices for flat iterator: flat iterator "
                    f"is 1-dimensional, but {builtins.len(idx)} were indexed")
            self.__setitem__(idx[0], val)
            return
        if isinstance(idx, builtins.bool):
            _w.warn("0-dimensional boolean index", DeprecationWarning,
                    stacklevel=2)
            if idx:
                self._set(0, val)
            return
        if isinstance(idx, slice):
            idxs = list(range(*idx.indices(self._size)))
            if isinstance(val, (list, tuple)):
                vals = list(val)
            elif hasattr(val, 'tolist'):
                vals = _flatten_vals(val.tolist())
            else:
                vals = [val] * builtins.len(idxs)
            for i, v in zip(idxs, vals):
                self._set(i, v)
            return
        if _is_ndarray(idx):
            _t = idx.tolist()
            _fi = _NdFlatIter._flat_list(_t) if isinstance(_t, list) else [_t]
            if getattr(idx, '_dtype', '') == 'bool':
                if builtins.len(_fi) != self._size:
                    raise IndexError(
                        "boolean index did not match indexed flat iterator")
                for i, m in builtins.enumerate(_fi):
                    if m:
                        self._set(i, val)
                return
            for i in _fi:
                self._set(self._norm_flat(i), val)
            return
        if isinstance(idx, _NdFlatIter):
            if idx._size > 0 and \
                    getattr(idx._arr, '_str_dtype', None) is not None:
                self._raise_invalid_idx()
            items = [idx[i] for i in range(idx._size)]
            for it in items:
                if isinstance(it, (str, bytes)):
                    self._raise_invalid_idx()
            for i in items:
                self._set(self._norm_flat(i), val)
            return
        if isinstance(idx, list):
            if builtins.len(idx) == 0:
                return
            _fi = _NdFlatIter._flat_list(idx)
            if builtins.all(isinstance(v, builtins.bool) for v in _fi):
                raise IndexError(
                    "boolean indices for iterators are not supported")
            for v in _fi:
                if not isinstance(v, (builtins.int, builtins.float,
                                      builtins.bool)):
                    self._raise_invalid_idx()
            if builtins.any(isinstance(v, builtins.float) and
                            not isinstance(v, builtins.bool) for v in _fi):
                _w.warn("Invalid non-array indices for iterator objects are "
                        "deprecated", DeprecationWarning, stacklevel=2)
            positions = [
                self._norm_flat(1 if isinstance(v, builtins.bool)
                                else builtins.int(v))
                for v in _fi]
            if isinstance(val, (list, tuple)):
                vals = list(val)
            elif hasattr(val, 'tolist'):
                vals = _flatten_vals(val.tolist())
            else:
                vals = [val] * builtins.len(positions)
            if builtins.len(vals) != builtins.len(positions):
                vals = [val] * builtins.len(positions)
            for i, v in zip(positions, vals):
                self._set(i, v)
            return
        self._set(self._norm_flat(idx), val)

    def _set(self, idx, val):
        try:
            val = int(val) if self._is_int else float(val)
        except (TypeError, ValueError):
            pass
        shape = self._arr.shape
        try:
            if not shape:
                self._arr[...] = val
            else:
                idxs = []
                r = idx
                for d in range(len(shape) - 1, -1, -1):
                    idxs.append(r % shape[d])
                    r //= shape[d]
                idxs.reverse()
                self._arr[tuple(idxs)] = val
        except Exception:
            pass
        try:
            self._flat[idx] = val
        except Exception:
            pass


# numpy 兼容：np.flatiter 是 ndarray.flat 返回的迭代器类型
flatiter = _NdFlatIter


class _DTypeNamespace:
    """numpy 2.x 风格 dtypes 命名空间（兼容子集）。"""
    VoidDType = type('VoidDType', (), {})
    BoolDType = type('BoolDType', (), {})
    Int8DType = type('Int8DType', (), {})
    Int16DType = type('Int16DType', (), {})
    Int32DType = type('Int32DType', (), {})
    Int64DType = type('Int64DType', (), {})
    UInt8DType = type('UInt8DType', (), {})
    UInt16DType = type('UInt16DType', (), {})
    UInt32DType = type('UInt32DType', (), {})
    UInt64DType = type('UInt64DType', (), {})
    Float16DType = type('Float16DType', (), {})
    Float32DType = type('Float32DType', (), {})
    Float64DType = type('Float64DType', (), {})
    Complex64DType = type('Complex64DType', (), {})
    Complex128DType = type('Complex128DType', (), {})
    StrDType = type('StrDType', (), {})
    BytesDType = type('BytesDType', (), {})
    ObjectDType = type('ObjectDType', (), {})
    DatetimeDType = type('DatetimeDType', (), {})
    TimedeltaDType = type('TimedeltaDType', (), {})


dtypes = _DTypeNamespace


# numpy 兼容：np.dtype 既是类型（isinstance 可用）又可调用（np.dtype('int64')）
dtype = DType


def top_k(a, k, axis=-1, **kwargs):
    """返回沿最内轴的 top-k 值与下标（numpy 2.x np.top_k 兼容子集）。"""
    arr = a if hasattr(a, '_array') else array(a)
    shp = arr.shape
    rows = arr.tolist()
    if not (isinstance(rows, list) and rows and isinstance(rows[0], (list, tuple))):
        rows = [rows]
    vals_out, idx_out = [], []
    for row in rows:
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))[:k]
        vals_out.append([row[i] for i in order])
        idx_out.append([i for i in order])
    tail = tuple(shp[:-1]) + (k,)
    if shp[:-1]:
        v = array(vals_out).reshape(tail)
        i = array(idx_out).reshape(tail)
    else:
        v = array([x for sub in vals_out for x in sub])
        i = array([x for sub in idx_out for x in sub])
    return (v, i)


class _NdIterElement:
    """nditer 的可写元素包装，支持 op_flags=['readwrite'] 时修改原数组。"""
    __slots__ = ('_arr', '_idx', '_val')

    def __init__(self, arr, idx, val):
        self._arr = arr
        self._idx = idx
        self._val = val

    def __repr__(self):
        return repr(self._val)

    def __float__(self):
        return float(self._val)

    def __int__(self):
        return int(self._val)

    def __bool__(self):
        return builtins.bool(self._val)

    def __eq__(self, other):
        return self._val == other

    def __ne__(self, other):
        return self._val != other

    def __lt__(self, other):
        return self._val < other

    def __le__(self, other):
        return self._val <= other

    def __gt__(self, other):
        return self._val > other

    def __ge__(self, other):
        return self._val >= other

    def __add__(self, other):
        return self._val + other

    def __radd__(self, other):
        return other + self._val

    def __sub__(self, other):
        return self._val - other

    def __rsub__(self, other):
        return other - self._val

    def __mul__(self, other):
        return self._val * other

    def __rmul__(self, other):
        return other * self._val

    def __truediv__(self, other):
        return self._val / other

    def __rtruediv__(self, other):
        return other / self._val

    def __neg__(self):
        return -self._val

    def __abs__(self):
        return abs(self._val)

    def __getitem__(self, key):
        return self._val

    def __setitem__(self, key, value):
        self._val = float(value)
        self._arr.put([self._idx], [self._val])


def nditer(a, order='C', op_flags=None, flags=None):
    """创建数组元素的迭代器，逐个访问数组元素。

    参数:
        a: 输入数组或数组列表（广播迭代）
        order: 遍历顺序，'C'（行序优先）或 'F'（列序优先）
        op_flags: 操作标志列表，如 ['readwrite'] 表示可读写
        flags: 迭代器标志，如 ['external_loop'] 表示返回一维数组块而非标量

    >>> a = np.array([[0, 1, 2], [3, 4, 5]])
    >>> for x in np.nditer(a):
    ...     print(x, end=", ")
    0, 1, 2, 3, 4, 5,
    """
    # 多数组广播迭代
    if isinstance(a, (list, tuple)):
        arrays = [ndarray(x) for x in a]
        # 计算广播形状：每个维度取最大值
        all_shapes = [x.shape for x in arrays]
        max_ndim = np.imax([len(s) for s in all_shapes])
        padded = []
        for s in all_shapes:
            pad = [1] * (max_ndim - len(s)) + list(s)
            padded.append(pad)
        bcast_shape = tuple(np.imax([p[i] for p in padded]) for i in range(max_ndim))
        # 广播每个数组到统一形状
        bcast_arrays = []
        for arr in arrays:
            if arr.shape == bcast_shape:
                bcast_arrays.append(arr)
            else:
                bcast_arrays.append(ndarray(np.broadcast_to(arr._array, bcast_shape)))
        # 展平并迭代
        flats = [x._array.flatten().tolist() for x in bcast_arrays]
        is_ints = [getattr(x, '_dtype', 'float64') == 'int64' for x in bcast_arrays]
        for i in range(len(flats[0])):
            yield tuple(int(f[i]) if is_ints[j] else f[i] for j, f in enumerate(flats))
        return

    arr = ndarray(a)
    ext_loop = flags and 'external_loop' in flags

    # external_loop: 按 chunk 返回一维数组
    if ext_loop:
        if arr.ndim <= 1:
            yield arr
            return
        if order == 'F':
            # F-order: 每次返回一列（最后一维）
            ncols = arr.shape[-1]
            for col in range(ncols):
                key = tuple([slice(None)] * (arr.ndim - 1) + [col])
                col_arr = arr[key]
                yield ndarray(col_arr)
        else:
            # C-order: 每次返回一行（第一维）
            for i in range(arr.shape[0]):
                yield ndarray(arr[i])
        return

    # 原有逐元素逻辑
    flat_arr = arr.copy().flatten()
    size = flat_arr.size
    is_int = getattr(arr, '_dtype', 'float64') == 'int64'
    readwrite = op_flags and 'readwrite' in op_flags

    def _to_val(v, ci):
        if readwrite:
            return _NdIterElement(arr, ci, int(v) if is_int else v)
        return int(v) if is_int else v

    if order == 'F' and arr.ndim > 1:
        # Fortran-order: 按列序优先（首维最快）生成多维索引，转为 C-flat 索引
        shape = arr.shape
        ndim = len(shape)
        # C-order strides: stride[i] = product(shape[i+1:])
        c_strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            c_strides[i] = c_strides[i + 1] * shape[i + 1]

        # F-order 等同于最后一维最慢，首维最快
        # 递归遍历 dim = ndim-1, ndim-2, ..., 0
        def _iter_f_order(dim, coords):
            if dim < 0:
                # coords 是反向顺序 [last_dim, ..., first_dim]，需反转
                idx = np.isum([c * s for c, s in zip(reversed(coords), c_strides)])
                if 0 <= idx < size:
                    val = flat_arr[idx].item()
                    yield _to_val(val, idx)
            else:
                for v in range(shape[dim]):
                    yield from _iter_f_order(dim - 1, coords + [v])

        yield from _iter_f_order(ndim - 1, [])
    else:
        # C-order: 行序优先（默认）
        for i in range(size):
            val = flat_arr[i].item()
            yield _to_val(val, i)

def isnan(x):
    """逐元素检测是否为 NaN。"""
    arr = ndarray(x)
    return ndarray._wrap(np.isnan(arr._array), _dtype="bool")


def binary_repr(num, width=None):
    """返回整数的二进制表示字符串。"""
    return np.binary_repr(num, width)


def isinf(x):
    """逐元素检测是否为无穷大。"""
    arr = ndarray(x)
    return ndarray._wrap(np.isinf(arr._array), _dtype="bool")


def isfinite(x):
    """逐元素检测是否为有限值。"""
    arr = ndarray(x)
    return ndarray._wrap(np.isfinite(arr._array), _dtype="bool")


def iscomplex(x):
    """逐元素检测元素的虚部是否非零。"""
    arr = ndarray(x)
    cpx = getattr(arr, '_complex_data', None)
    if cpx is not None:
        return ndarray(np.iscomplex_cpx(cpx))
    # 不是复数数组 → 全零
    shape = arr.shape
    return ndarray(np.zeros(shape))


def _broadcast_shape(*shapes):
    """计算多个数组的广播形状。"""
    max_ndim = np.imax([len(s) for s in shapes])
    result = []
    for i in range(max_ndim):
        dims = set()
        for s in shapes:
            if i < max_ndim - len(s):
                dims.add(1)
            else:
                dims.add(s[i - (max_ndim - len(s))])
        non_one = [d for d in dims if d != 1]
        if len(non_one) > 1:
            raise ValueError("operands could not be broadcast together")
        result.append(non_one[0] if non_one else 1)
    return tuple(result)


def broadcast(*arrays):
    """产生模仿广播的对象，将多个数组广播到相同形状。"""
    if not arrays:
        raise ValueError("broadcast requires at least one argument")
    _shapes = []
    for _a in arrays:
        _s = getattr(_a, 'shape', None)
        if _s is None:
            _s = getattr(np.array(_a), 'shape', ())
        _shapes.append(tuple(_s))
    try:
        _bshape = _broadcast_shape(*_shapes)
    except ValueError:
        # numpy 风格错误消息：指出第一个互不兼容的参数对
        for _i in range(len(_shapes)):
            for _j in range(_i + 1, len(_shapes)):
                _si, _sj = _shapes[_i], _shapes[_j]
                _k = builtins.max(len(_si), len(_sj))
                _p = (1,) * (_k - len(_si)) + _si
                _q = (1,) * (_k - len(_sj)) + _sj
                if builtins.any(x != y and x != 1 and y != 1
                       for x, y in zip(_p, _q)):
                    raise ValueError(
                        f"arg {_i} with shape {_si} and arg {_j} "
                        f"with shape {_sj} cannot be broadcast "
                        f"to a single shape")
        raise
    return _Broadcast(*arrays)


def _broadcast_reset(self, /):
    """numpy broadcast 对象方法占位（no-op）。"""
    return None


broadcast.reset = _broadcast_reset


class _Broadcast:
    """广播对象，将多个数组广播到相同形状。"""

    def __init__(self, *args):
        self._arrays = [ndarray(a) for a in args]
        # 计算广播形状
        shapes = [a.shape for a in self._arrays]
        self._shape = _broadcast_shape(*shapes)
        self._size = 1
        for s in self._shape:
            self._size *= s

    @property
    def shape(self):
        """广播后的形状。"""
        return self._shape

    @property
    def ndim(self):
        """广播后的维数。"""
        return len(self._shape)

    nd = ndim  # numpy broadcast 对象的旧别名

    @property
    def numiter(self):
        """广播后的总元素数。"""
        return self._size

    @property
    def iters(self):
        """返回各数组的迭代器元组。"""
        return tuple(_BroadcastIter(a, self._shape) for a in self._arrays)

    def __iter__(self):
        """迭代广播后的值元组。"""
        iters = self.iters
        for _ in range(self._size):
            yield tuple(next(it) for it in iters)


class _BroadcastIter:
    """广播迭代器，按广播形状遍历单个数组。"""

    def __init__(self, arr, shape):
        self._arr = arr
        self.base = arr  # numpy 兼容：迭代器引用的源数组
        self._flat = arr._array.flatten().tolist()
        self._size = len(self._flat)
        self._arr_shape = arr.shape
        self._broadcast_shape = shape
        # Left-pad array shape for broadcasting
        ndim_pad = len(shape) - len(arr.shape)
        self._padded = (1,) * ndim_pad + arr.shape
        self._indices = [0] * len(shape)
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        # Map broadcast index to flat array index
        arr_idx = 0
        stride = 1
        for d in range(len(self._arr_shape) - 1, -1, -1):
            dim = self._indices[len(self._broadcast_shape) - len(self._arr_shape) + d]
            dim = dim % self._arr_shape[d]
            arr_idx += dim * stride
            stride *= self._arr_shape[d]
        val = self._flat[arr_idx]

        # Advance to next C-order position
        for i in range(len(self._indices) - 1, -1, -1):
            self._indices[i] += 1
            if self._indices[i] < self._broadcast_shape[i]:
                break
            self._indices[i] = 0
            if i == 0:
                self._done = True

        dtype = getattr(self._arr, '_dtype', 'float64')
        if dtype == 'int64':
            return int(val)
        return float(val)


# 数组操作函数
reshape = _array_ops_module.reshape
ravel = _array_ops_module.ravel
moveaxis = _array_ops_module.moveaxis
rollaxis = _array_ops_module.rollaxis
broadcast_to = _array_ops_module.broadcast_to
transpose = _array_ops_module.transpose
swapaxes = _array_ops_module.swapaxes
expand_dims = _array_ops_module.expand_dims
squeeze = _array_ops_module.squeeze
concatenate = _array_ops_module.concatenate
stack = _array_ops_module.stack
vstack = _array_ops_module.vstack
hstack = _array_ops_module.hstack
dstack = _array_ops_module.dstack
column_stack = _array_ops_module.column_stack
split = _array_ops_module.split
hsplit = _array_ops_module.hsplit
vsplit = _array_ops_module.vsplit
dsplit = _array_ops_module.dsplit
tile = _array_ops_module.tile
repeat = _array_ops_module.repeat
flip = _array_ops_module.flip
fliplr = _array_ops_module.fliplr
flipud = _array_ops_module.flipud
roll = _array_ops_module.roll
rot90 = _array_ops_module.rot90
append = _array_ops_module.append
insert = _array_ops_module.insert
delete = _array_ops_module.delete
unique = _array_ops_module.unique
resize = _array_ops_module.resize

# 位运算
def _to_raw(a):
    """将输入转为 Rust ndarray，返回 (ndarray, dtype) 元组。"""
    if hasattr(a, '_array'):
        return a._array, getattr(a, '_dtype', 'float64')
    raw = np.ndarray(a)
    if isinstance(a, int) and not isinstance(a, bool):
        return raw, 'int64'
    return raw, 'float64'

def bitwise_and(x1, x2):
    """按位与"""
    r1, dt = _to_raw(x1)
    r2, _ = _to_raw(x2)
    return _wrap_result(np.bitwise_and(r1, r2), dt)

def bitwise_or(x1, x2):
    """按位或"""
    r1, dt = _to_raw(x1)
    r2, _ = _to_raw(x2)
    return _wrap_result(np.bitwise_or(r1, r2), dt)

def bitwise_xor(x1, x2):
    """按位异或"""
    r1, dt = _to_raw(x1)
    r2, _ = _to_raw(x2)
    return _wrap_result(np.bitwise_xor(r1, r2), dt)

def bitwise_not(x):
    """按位取反（等效于 ~ 运算符，对布尔数组使用逻辑取反）"""
    r, dt = _to_raw(x)
    # dtype 可能是字符串 'bool'，也可能是类对象（zeros_like(dtype=xp.bool)
    # 产生的 rsnumpy.bool），两者都要识别为布尔并走逻辑取反。
    if dt == 'bool' or 'bool' in getattr(dt, '__name__', str(dt)).lower():
        # 布尔数组：逻辑取反（numpy 语义）。np.invert 是补码位取反，会把
        # 0/1 表示的 bool 变成 -1/-2；对 0/1 值异或 1 才是逻辑非。
        r = np.bitwise_xor(r, np.ndarray([1]))
    elif dt == 'uint8':
        raw = np.bitwise_not(r)
        r = np.bitwise_and(raw, np.ndarray([255]))
        dt = 'uint8'
    else:
        r = np.bitwise_not(r)
    return _wrap_result(r, dt)


# invert 与 bitwise_not 行为完全一致，直接作为别名，避免重复实现。
invert = bitwise_not

def _int_width(dt):
    return {'bool': 8,
            'int8': 8, 'uint8': 8, 'int16': 16, 'uint16': 16,
            'int32': 32, 'uint32': 32, 'int64': 64, 'uint64': 64}.get(str(dt))


def _python_shift(x1, x2, is_rshift, dt):
    """按 dtype 位宽用 Python 补码语义重算移位：Rust 移位对 >= 位宽的移位量
    不生效（返回原值），这里统一重算保证 numpy 语义（超宽右移 → 0/-1）。"""
    a = x1 if _is_ndarray(x1) else ndarray(x1)
    b = x2 if _is_ndarray(x2) else ndarray(x2)
    w = _int_width(a.dtype) or 64
    signed = str(a.dtype).startswith('int')
    la = _pyflat(a.tolist())
    lb = _pyflat(b.tolist())
    if len(lb) == 1 and len(la) != 1:
        lb = lb * len(la)
    m = (1 << w) - 1
    out = []
    for v, s in zip(la, lb):
        iv, sh = int(v), int(s)
        vv = iv & m
        if sh >= w:
            r = -1 if (is_rshift and signed and iv < 0) else 0
        elif is_rshift:
            r = vv >> sh
            if iv < 0:
                r |= m << (w - sh)
        else:
            r = (vv << sh) & m
        if signed and r >= (1 << (w - 1)):
            r -= (1 << w)
        out.append(r)
    if a.ndim == 0:
        return ndarray(out[0], _dtype=dt)
    arr = ndarray(out, _dtype=dt)
    if a.ndim > 1:
        arr = arr.reshape(a.shape)
    return arr


def left_shift(x1, x2):
    """左移"""
    r1, dt = _to_raw(x1)
    r2, _ = _to_raw(x2)
    if _int_width(dt) is not None:
        return _python_shift(x1, x2, False, dt)
    return _wrap_result(np.left_shift(r1, r2), dt)

def right_shift(x1, x2):
    """右移"""
    r1, dt = _to_raw(x1)
    r2, _ = _to_raw(x2)
    if _int_width(dt) is not None:
        return _python_shift(x1, x2, True, dt)
    return _wrap_result(np.right_shift(r1, r2), dt)


# 数学函数
sin = _math_functions_module.sin
cos = _math_functions_module.cos
tan = _math_functions_module.tan
arcsin = _math_functions_module.arcsin
arccos = _math_functions_module.arccos
arctan = _math_functions_module.arctan
arctan2 = _math_functions_module.arctan2
deg2rad = _math_functions_module.deg2rad
rad2deg = _math_functions_module.rad2deg
degrees = rad2deg
sinh = _math_functions_module.sinh
cosh = _math_functions_module.cosh
tanh = _math_functions_module.tanh
asinh = _math_functions_module.asinh
acosh = _math_functions_module.acosh
atanh = _math_functions_module.atanh
exp = _math_functions_module.exp
expm1 = _math_functions_module.expm1
log = _math_functions_module.log
log10 = _math_functions_module.log10
log2 = _math_functions_module.log2
log1p = _math_functions_module.log1p
around = _math_functions_module.around
floor = _math_functions_module.floor
ceil = _math_functions_module.ceil
trunc = _math_functions_module.trunc
fix = _math_functions_module.fix
sqrt = _math_functions_module.sqrt
square = _math_functions_module.square
cbrt = _math_functions_module.cbrt
abs = _math_functions_module.abs
sign = _math_functions_module.sign
reciprocal = _math_functions_module.reciprocal
_clip_core = _math_functions_module.clip

_CLIP_NOVALUE = object()


def clip(a, a_min=_CLIP_NOVALUE, a_max=_CLIP_NOVALUE, out=None,
         min=_CLIP_NOVALUE, max=_CLIP_NOVALUE):
    """限制值范围。

    支持 numpy 的 min=/max= 关键字别名；None 界表示该侧不裁剪。
    显式 None（位置或别名）与"未提供"用哨兵区分。
    """
    # numpy 语义：min/max 别名与 a_min/a_max 不可混用
    if (min is not _CLIP_NOVALUE or max is not _CLIP_NOVALUE) and \
            (a_min is not _CLIP_NOVALUE or a_max is not _CLIP_NOVALUE):
        raise ValueError(
            "Passing `min` or `max` keyword argument when `a_min` and "
            "`a_max` are provided is forbidden.")
    if min is not _CLIP_NOVALUE:
        a_min = min
    if max is not _CLIP_NOVALUE:
        a_max = max
    a_arr = asarray(a)
    if a_min is _CLIP_NOVALUE and a_max is _CLIP_NOVALUE:
        # 无界：不裁剪（numpy 2.x 语义，等价于返回副本）
        result = a_arr.copy()
        if out is not None:
            for i in range(len(result)):
                out[i] = result[i]
            return out
        return result
    if a_min is _CLIP_NOVALUE:
        raise TypeError("clip() missing 1 required positional argument: 'a_min'")
    if a_max is _CLIP_NOVALUE:
        raise TypeError("clip() missing 1 required positional argument: 'a_max'")
    if a_min is None:
        a_min = a_arr.min()
    if a_max is None:
        a_max = a_arr.max()
    result = _clip_core(a, a_min, a_max)
    # 保持输入 dtype（numpy 的 clip 不改类型）
    if _is_ndarray(result) and _is_ndarray(a_arr):
        _dt_in = str(a_arr.dtype)
        if _dt_in != str(result.dtype) and _dt_in in _INT_DTYPES:
            try:
                result = result.astype(_dt_in)
            except (TypeError, ValueError):
                pass
    if out is not None:
        for i in range(len(result)):
            out[i] = result[i]
        return out
    return result

    
sinc = _math_functions_module.sinc
heaviside = _math_functions_module.heaviside
add = _math_functions_module.add
subtract = _math_functions_module.subtract
multiply = _math_functions_module.multiply
divide = _math_functions_module.divide
power = _math_functions_module.power
mod = _math_functions_module.mod
remainder = _math_functions_module.remainder
greater = _math_functions_module.greater
less = _math_functions_module.less
equal = _math_functions_module.equal
not_equal = _math_functions_module.not_equal
greater_equal = _math_functions_module.greater_equal
less_equal = _math_functions_module.less_equal
logical_and = _math_functions_module.logical_and
logical_or = _math_functions_module.logical_or
logical_xor = _math_functions_module.logical_xor
isclose = _math_functions_module.isclose
allclose = _math_functions_module.allclose

# 统计函数
sum = _statistics_module.sum
mean = _statistics_module.mean
std = _statistics_module.std
var = _statistics_module.var
min = _statistics_module.min
max = _statistics_module.max
amin = _statistics_module.amin
amax = _statistics_module.amax
ptp = _statistics_module.ptp
median = _statistics_module.median
average = _statistics_module.average
percentile = _statistics_module.percentile
quantile = _statistics_module.quantile
nanmedian = _statistics_module.nanmedian
nanpercentile = _statistics_module.nanpercentile
argmax = _statistics_module.argmax
argmin = _statistics_module.argmin
argsort = _statistics_module.argsort
sort = _statistics_module.sort
msort = _statistics_module.msort
sort_complex = _statistics_module.sort_complex
lexsort = _statistics_module.lexsort
partition = _statistics_module.partition
argpartition = _statistics_module.argpartition
# searchsorted = _statistics_module.searchsorted  # Use local searchsorted instead
extract = _statistics_module.extract
cov = _statistics_module.cov
corrcoef = _statistics_module.corrcoef
histogram = _statistics_module.histogram
histogram2d = _statistics_module.histogram2d
histogramdd = _statistics_module.histogramdd
digitize = _statistics_module.digitize


def cumsum(a, axis=None):
    """计算数组元素的累积和。"""
    return asarray(a).cumsum(axis)


def cumprod(a, axis=None):
    """计算数组元素的累积乘积。"""
    return asarray(a).cumprod(axis)


linalg = _linalg_module()
from . import random  # np.random 指向 random 模块本身（numpy 中 np.random 是模块，from rsnumpy.random import rand 必须可用）
matlib = _matlib_module

dot = linalg.dot
vdot = linalg.vdot
inner = linalg.inner
matmul = linalg.matmul


# ========== 导出列表 ==========
__all__ = [
    'ndarray', 'NdArrayIter',
    'array', 'asarray', 'asanyarray', 'copy',
    'matrix', 'asmatrix', 'bmat', 'repmat',
    'dot', 'vdot', 'inner', 'matmul',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'full', 'full_like',
    'empty', 'empty_like', 'eye', 'identity',
    'arange', 'linspace', 'logspace', 'geomspace',
    'fromfunction', 'frombuffer', 'r_', 'c_', 's_', 'mgrid', 'ogrid', 'meshgrid',
    'reshape', 'ravel', 'moveaxis', 'rollaxis', 'broadcast_to',
    'transpose', 'swapaxes', 'expand_dims', 'squeeze',
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',
    'split', 'hsplit', 'vsplit', 'dsplit',
    'tile', 'repeat', 'LazyArray',
    'flip', 'fliplr', 'flipud', 'roll', 'rot90',
    'append', 'insert', 'delete', 'unique', 'resize',
    'where', 'take', 'put', 'select', 'nonzero', 'argwhere', 'flatnonzero',
    'binary_repr',
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
    'deg2rad', 'rad2deg', 'degrees',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
    'invert', 'left_shift', 'right_shift',
    'string_', 'unicode_', 'char',
    'datetime64', 'timedelta64',
    'exp', 'expm1', 'log', 'log10', 'log2', 'log1p',
    'around', 'floor', 'ceil', 'trunc', 'fix',
    'sqrt', 'square', 'cbrt', 'abs', 'sign', 'reciprocal', 'clip', 'sinc', 'heaviside',
    'add', 'subtract', 'multiply', 'divide', 'power', 'mod', 'remainder',
    'greater', 'less', 'equal', 'logical_and', 'logical_or', 'isclose', 'allclose',
    'sum', 'cumsum', 'cumprod', 'mean', 'std', 'var', 'min', 'max', 'amin', 'amax', 'ptp',
    'median', 'average', 'percentile', 'quantile', 'nanmedian', 'nanpercentile',
    'argmax', 'argmin', 'argsort', 'sort', 'searchsorted', 'extract',
    'cov', 'corrcoef',
    'histogram', 'histogram2d', 'histogramdd', 'digitize',
    'fft', 'typing',
    'pi', 'e', 'euler_gamma', 'inf', 'nan', 'newaxis',
    'nditer', '_core', 'exceptions',
    'isnan', 'isinf', 'isfinite',
    'save', 'load', 'loadtxt', 'savetxt', 'savez', 'genfromtxt',
    'Poly', 'polyval', 'polyfit', 'polyder', 'polyint', 'polyroots',
    'linalg', 'random', 'matlib', 'load_npz', 'lib',
    # 标量类型层次
    'generic', 'number', 'integer', 'signedinteger', 'unsignedinteger',
    'inexact', 'floating', 'complexfloating', 'flexible', 'character',
    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
    'float16', 'float32', 'float64', 'float_', 'complex64', 'complex128',
    'bytes_', 'str_', 'bool_', 'object_', 'void',
    'byte', 'short', 'intc', 'int_', 'long', 'longlong', 'intp',
    'ubyte', 'ushort', 'uintc', 'uint', 'ulong', 'ulonglong', 'uintp',
    'half', 'single', 'double', 'longdouble', 'csingle', 'cdouble', 'clongdouble',
    'True_', 'False_', 'little_endian', 'ScalarType', 'sctypeDict', 'typecodes',
    'issubdtype', 'finfo', 'iinfo', 'ndindex', 'ndenumerate', 'index_exp',
    'dtype', 'DType', 'rec', 'ma', 'recarray', 'memory_usage',
    'chunked_apply', 'mmap_array',
    # 线程控制接口
    'set_num_threads', 'get_num_threads', 'get_num_cpus',
    'get_parallel_thresholds', 'set_parallel_thresholds', 'parallel_context',
]


# ========== 补充 API（_extra）挂载 ==========
# _extra 中的函数均由现有 Rust 原语组合实现；已存在的原生实现（如 iscomplex）优先保留。

for _extra_name in _extra_module.__all__:
    if _extra_name not in globals():
        globals()[_extra_name] = getattr(_extra_module, _extra_name)
        __all__.append(_extra_name)
del _extra_name


# ========== ufunc 归约方法（reduce/accumulate） ==========
def _make_ufunc_reduce(reducer):
    def reduce(a, axis=0, dtype=None, out=None, keepdims=False):
        _ = keepdims
        res = reducer(a, axis)
        if not hasattr(res, '_array'):
            res = array(res)
        if dtype is not None:
            res = res.astype(dtype)
        if out is not None:
            out[:] = res
            return out
        return res
    return reduce


def _make_ufunc_accumulate(accumulator):
    def accumulate(a, axis=0, dtype=None, out=None):
        _ = dtype, out
        return accumulator(a, axis)
    return accumulate


add.reduce = _make_ufunc_reduce(lambda a, ax: sum(a, ax))
multiply.reduce = _make_ufunc_reduce(lambda a, ax: _extra_module.prod(a, ax))
add.accumulate = _make_ufunc_accumulate(lambda a, ax: cumsum(a, ax))
multiply.accumulate = _make_ufunc_accumulate(lambda a, ax: cumprod(a, ax))

class _NamespaceInfo:
    """Array-API ``__array_namespace_info__`` payload (CPU-only)."""

    version = '2024.12'

    def capabilities(self):
        # array_api_compat convention: space-separated keys (scipy's vendored
        # _external code indexes the dict with e.g. "boolean indexing").
        return {
            'boolean indexing': True,
            'data-dependent shapes': True,
            'eager execution': True,
            'finite only': False,
            'linalg': True,
            'integer pow': True,
            'searchsorted': True,
            'shape': True,
            'var': True,
            'std': True,
            'product': True,
            'minmax': True,
            'where': True,
            'indexing': 'basic',
        }

    def default_device(self):
        return 'cpu'

    def devices(self):
        return ('cpu',)

    def extensions(self):
        return []

    def _all_dtypes(self):
        return {
            'bool': bool_,
            'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64,
            'uint8': uint8, 'uint16': uint16, 'uint32': uint32,
            'uint64': uint64,
            'float32': float32, 'float64': float64,
            'complex64': complex64, 'complex128': complex128,
        }

    def default_dtypes(self, *, device=None, kind=None):
        defaults = {
            'real floating': float64,
            'complex floating': complex128,
            'integral': int64,
            'indexing': int64,
            'bool': bool_,
        }
        if kind is None:
            return dict(defaults)
        return defaults[kind]

    def dtypes(self, *, device=None, kind=None):
        all_dtypes = self._all_dtypes()
        if kind is None:
            return dict(all_dtypes)
        kinds = kind if isinstance(kind, tuple) else (kind,)
        out = {}
        for k in kinds:
            if k == 'bool':
                out['bool'] = bool_
            elif k == 'real floating':
                out.update({'float32': float32, 'float64': float64})
            elif k == 'complex floating':
                out.update({'complex64': complex64, 'complex128': complex128})
            elif k == 'signed integer':
                out.update({n: all_dtypes[n] for n in
                            ('int8', 'int16', 'int32', 'int64')})
            elif k == 'unsigned integer':
                out.update({n: all_dtypes[n] for n in
                            ('uint8', 'uint16', 'uint32', 'uint64')})
            elif k == 'integral':
                out.update({n: all_dtypes[n] for n in
                            ('int8', 'int16', 'int32', 'int64',
                             'uint8', 'uint16', 'uint32', 'uint64')})
                out['indexing'] = int64
            elif k == 'numeric':
                out.update(all_dtypes)
                out.pop('bool')
            else:
                raise ValueError(f"unrecognized data type kind: {k!r}")
        return out


def __array_namespace_info__():
    return _NamespaceInfo()


_sys.modules['rsnumpy'] = _sys.modules[__name__]
_sys.modules['rsnumpy.__init__'] = _sys.modules[__name__]

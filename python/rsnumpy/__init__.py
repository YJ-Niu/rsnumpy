"""rsnumpy - A Rust-powered NumPy-compatible array library.

rsnumpy provides a NumPy-compatible API implemented in Rust for better
performance and memory efficiency while maintaining full API compatibility.

Examples:
    >>> import rsnumpy as np
    >>> a = np.ndarray([1, 2, 3])
    >>> np.sum(a)
    6
    >>> np.mean(a)
    2.0
"""

import datetime as _datetime
import sys as _sys
import math as _math
from . import _extra as _extra_module

import rsnumpy._core as _core

_current_module = _sys.modules[__name__]
_sys.modules['rsnumpy'] = _current_module
_sys.modules['rsnumpy.__init__'] = _current_module
_current_module.__name__ = 'rsnumpy'
from rsnumpy._core import ndarray_iter as NdArrayIter
# ========== 子模块导入和函数挂载 ==========
# 这些导入放在 ndarray 类定义之后以避免循环导入
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
from ._dtypes import (
    DType, dtype, _make_flexible, _make_subarray,
    _build_struct, _scalar_typestr_short,
)
# ========== rec / ma 子模块（在顶层完全初始化后导入以避免循环依赖） ==========
from . import ma
from . import rec

__version__ = "1.1.8"

# 捕获内建函数别名：_extra 挂载会向本模块 globals 注入同名的 numpy 函数
# （all/any/round），会遮蔽内建函数。以下别名保证本文件内部逻辑始终使用内建实现。
_py_round = round
_py_all = all
_py_any = any
_py_max = max
_py_min = min


class ArrayFlags:
    """数组内存布局信息，与 NumPy 的 np.ndarray.flags 兼容。"""

    def __init__(self, arr):
        self._arr = arr

    @property
    def C_CONTIGUOUS(self):
        return True

    @property
    def F_CONTIGUOUS(self):
        return True

    @property
    def OWNDATA(self):
        return True

    @property
    def WRITEABLE(self):
        return True

    @property
    def ALIGNED(self):
        return True

    @property
    def WRITEBACKIFCOPY(self):
        return False

    @property
    def UPDATEIFCOPY(self):
        return False

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
    """0 维浮点标量的字符串，与 numpy 标量 str 对齐（如 75.0、85850000000.0）。"""
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

    这是一个 Rust 后端支持的数组类型，提供与 NumPy 兼容的功能。

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

    def __init__(self, data, _dtype="float64", _fields=None, _raw_data=None):
        if _is_ndarray(data):
            self._array = data._array
            self._dtype = data._dtype
            self._fields = getattr(data, '_fields', None)
            self._raw_data = getattr(data, '_raw_data', None)
            self._complex_data = getattr(data, '_complex_data', None)
        elif hasattr(data, '__class__') and data.__class__.__name__ == 'ndarray':
            self._array = data
            self._dtype = _dtype
            self._fields = _fields
            self._raw_data = _raw_data
        else:
            self._raw_data = _raw_data
            if _raw_data is not None:
                # 原始数据存储：用一个虚拟 Rust 数组占位
                self._array = _core.zeros((len(data),) if isinstance(data, (list, tuple)) else (1,))
            elif not isinstance(data, (list, tuple)) and hasattr(data, '__array_interface__'):
                # 实现数组接口协议的对象（如 PIL 图像）→ 由 Rust 底层解码缓冲区
                self._array, _dtype = _from_array_interface(data)
            else:
                if isinstance(data, tuple):
                    data = list(data)
                if isinstance(data, list) and not _is_rectangular(data):
                    # 不规则列表（子列表长度不同），存储为原始数据
                    self._raw_data = data
                    self._array = _core.zeros((len(data),))
                else:
                    # 单次遍历同时获取扁平数据与复数/字符串标记
                    flat, has_c, has_s = _flatten_check(data)
                    if has_s:
                        # 字符串数据 → 存储为原始 Python 数据
                        self._raw_data = list(data) if isinstance(data, (list, tuple)) else [data]
                        self._array = _core.zeros((len(self._raw_data),))
                        _dtype = "string_"
                        # 记录 (kind, width)，供 dtype/itemsize 与整数索引取标量使用：
                        # Python str 按 numpy 惯例归为 unicode 'U'，bytes 归为 'S'，宽度取最长元素。
                        if flat and _py_all(isinstance(v, (bytes, bytearray)) for v in flat):
                            self._str_dtype = ('S', _py_max((len(bytes(v)) for v in flat), default=0))
                        else:
                            self._str_dtype = ('U', _py_max((len(str(v)) for v in flat), default=0))
                    elif has_c:
                        # 包含复数 → 交由 Rust 原生复数解析（保留形状与虚部）
                        self._array = _core.ndarray(data)
                        _dtype = "complex128"
                    else:
                        raw = _core.ndarray(data)
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
        obj._array = _core.zeros((len(raw_data),))
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
                return _core._format_int_str(self._array)
            return _core._format_int_val_str(self._array)
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
            # 0 维浮点数组的 str 与 numpy 标量一致（保留末尾 .0，如 75.0）。
            return _format_float_scalar_str(float(self._array.tolist()))
        return _core._format_float_str(self._array)

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
            return _convert_nested(raw_list, bool)
        if dt in ('uint8', 'uint16', 'uint32', 'uint64'):
            return _convert_nested(raw_list, int)
        if dt == 'string_':
            raw = getattr(self, '_raw_data', None)
            if raw is not None:
                return list(raw)
        return raw_list

    def tobytes(self, order='C'):
        """返回数组原始字节（与 numpy.ndarray.tobytes 兼容）。

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
        # 与 NumPy 行为一致（如 a, b = np.random.randn(2, n) 得到两个数组）。
        if self.ndim <= 1:
            return iter(self.tolist())
        return (self[i] for i in range(self.shape[0]))

    @property
    def __array_interface__(self):
        """NumPy 数组接口协议：让真实 numpy/matplotlib 能直接消费 rsnumpy 数组。

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
        return _core.array_interface(self._array, typestr)

    def __buffer__(self, flags):
        """PEP 3118 缓冲协议（PEP 688 / Python 3.12+）：转发到底层 Rust `_array`，
        让 memoryview(a) 与下游（如 rsplotlib 的 PyBuffer::get）零拷贝读取底层连续
        f64 内存，免去 __array_interface__ 的 bytes 副本开销。

        仅对 float64 暴露缓冲：底层存储恒为 f64，其字节布局与 float64 dtype 完全一致，
        缓冲 format 'd' 语义正确。int/bool 等 dtype 的底层仍是 f64，若也暴露缓冲，
        numpy 会优先按缓冲（f8）而非 __array_interface__（如 <i8）解读，导致 dtype 失真；
        故这些 dtype 不暴露缓冲，消费方自动回退到 dtype 精确的 __array_interface__ bytes。
        绘图坐标以 float64 为主，快路径覆盖热点场景。

        注：__buffer__ 由 CPython 在 3.12+ 才识别；更早版本上 memoryview(a) 会失败，
        消费方自动回退到 __array_interface__ bytes，行为安全。
        """
        if getattr(self, '_dtype', 'float64') != 'float64':
            raise BufferError("only float64 arrays expose a zero-copy buffer")
        if getattr(self, '_raw_data', None) is not None:
            raise BufferError("string/ragged array does not support the buffer protocol")
        if getattr(self, '_complex_data', None) is not None:
            raise BufferError("complex array does not support the buffer protocol")
        return memoryview(self._array)

    def __release_buffer__(self, view):
        view.release()

    def __bool__(self):
        if self.size == 1:
            return bool(self.item())
        raise ValueError("The truth value of an array with more than one element is ambiguous.")

    def __invert__(self):
        """逐元素取反（~ 运算符）。"""
        return bitwise_not(self)

    def __getitem__(self, key):
        # 结构化数组索引：字段 / 多字段 / 记录标量 / 子结构切片
        if getattr(self, '_dtype_obj', None) is not None:
            handled, res = _struct_getitem(self, key)
            if handled:
                return res
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
            if mask_shape == self_shape[:k]:
                rest = self_shape[k:]
                block = 1
                for s in rest:
                    block *= s
                raw = _core.masked_select(self._array, key._array, block)
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
            ellipsis_count = _core.isum([1 for k in key if k is Ellipsis])
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
        # 将 Python ndarray 索引展平为 Python list，确保 Rust 端可正确解析
        if isinstance(key, tuple):
            key = tuple(
                _ndarray_to_index_list(k)
                if hasattr(k, '_array') else k
                for k in key
            )
        else:
            key = (_ndarray_to_index_list(key)
                   if hasattr(key, '_array') else key,)
        # 委托给 Rust getitem_multi 处理所有索引逻辑
        result = _core.getitem_multi(self._array, key, list(self.shape))
        return _wrap_result(result, self._dtype)

    def __setitem__(self, key, value):
        # 结构化数组赋值：字段 / 多字段 / 记录 / 切片
        if getattr(self, '_dtype_obj', None) is not None:
            if _struct_setitem(self, key, value):
                return
        # 布尔掩码空选择：numpy 语义下为 no-op，避免 Rust 端空索引 panic
        if _empty_bool_mask_key(key):
            return
        # 展开省略号（...），补充完整切片以匹配数组维度
        if key is Ellipsis:
            key = tuple(slice(None, None, None) for _ in range(self.ndim))
        elif isinstance(key, tuple):
            new_key = []
            ellipsis_count = _core.isum([1 for k in key if k is Ellipsis])
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
        if isinstance(key, tuple):
            # 将 Python ndarray 索引展平为 list（布尔类型保留给 Rust 处理）
            key = tuple(
                _ndarray_to_index_list(k)
                if hasattr(k, '_array') and k.dtype != 'bool' else k
                for k in key
            )
        elif not isinstance(key, str):
            key = (_ndarray_to_index_list(key)
                   if hasattr(key, '_array') and key.dtype != 'bool' else key,)
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
            # 广播支持：若 value 是数组且形状可广播到目标形状，则先广播
            if _is_ndarray(value):
                try:
                    bcast_shape = _broadcast_shape(tuple(target_shape), value.shape)
                    if bcast_shape != value.shape:
                        value = broadcast_to(value, bcast_shape)
                except ValueError:
                    pass
            val = _cast_setitem_value(_setitem_value(value), self._dtype)
            _core.setitem_multi(self._array, key, list(self.shape), val)
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
            return _wrap_result(self._array * other._array, dt)
        return _wrap_result(self._array * other, dt)

    def __rmul__(self, other):
        dt = _promote_dtype(self._dtype, other)
        if self.size == 0:
            return self.copy()
        return _wrap_result(other * self._array, dt)

    def __truediv__(self, other):
        dt = _truediv_dtype(self._dtype)
        if self.size == 0:
            return self.copy()
        if _is_ndarray(other):
            return _wrap_result(self._array / other._array, dt)
        return _wrap_result(self._array / other, dt)

    def __rtruediv__(self, other):
        dt = _truediv_dtype(self._dtype)
        if self.size == 0:
            return self.copy()
        return _wrap_result(other / self._array, dt)

    def __mod__(self, other):
        """逐元素取模（% 运算符）。"""
        return mod(self, other)

    def __rmod__(self, other):
        """逐元素取模（右操作数，% 运算符）。"""
        return mod(other, self)

    def __matmul__(self, other):
        if _is_ndarray(other):
            try:
                return _wrap_result(_core.linalg.matmul(self._array, other._array), self._dtype)
            except ValueError:
                return _matmul_fallback(self, other)
        return _wrap_result(_core.linalg.matmul(self._array, other), self._dtype)

    def __pow__(self, other):
        if _is_ndarray(other):
            return _wrap_result(_core.power(self._array, other._array), self._dtype)
        return _wrap_result(_core.power(self._array, _core.ndarray([other])), self._dtype)

    def __rpow__(self, other):
        if _is_ndarray(other):
            return _wrap_result(_core.power(other._array, self._array), self._dtype)
        return _wrap_result(_core.power(_core.ndarray([other]), self._array), self._dtype)

    def __neg__(self):
        """逐元素取负（- 运算符）。"""
        return _wrap_result(self._array * -1, self._dtype)

    def __pos__(self):
        """一元正号（+ 运算符），返回副本。"""
        return _wrap_result(self._array.copy(), self._dtype)

    def __abs__(self):
        """逐元素绝对值（内建 abs() 函数）。"""
        return _wrap_result(_core.abs(self._array), self._dtype)

    def __eq__(self, other):
        if getattr(self, '_dtype_obj', None) is not None and \
                _is_ndarray(other) and getattr(other, '_dtype_obj', None) is not None:
            return _struct_eq(self, other)
        if isinstance(other, str):
            flat_list = self.flatten().tolist()
            flat_result = [x == other for x in flat_list]
            result = ndarray(flat_result)
            result._dtype = "bool"
            return result.reshape(self.shape)
        if _is_ndarray(other):
            return _wrap_result(self._array.__eq__(other._array), "bool")
        return _wrap_result(self._array.__eq__(other), "bool")

    def __ne__(self, other):
        if _is_ndarray(other):
            return _wrap_result(self._array.__ne__(other._array), "bool")
        return _wrap_result(self._array.__ne__(other), "bool")

    def __lt__(self, other):
        if _is_ndarray(other):
            return _wrap_result(self._array.__lt__(other._array), "bool")
        return _wrap_result(self._array.__lt__(other), "bool")

    def __le__(self, other):
        if _is_ndarray(other):
            return _wrap_result(self._array.__le__(other._array), "bool")
        return _wrap_result(self._array.__le__(other), "bool")

    def __gt__(self, other):
        if _is_ndarray(other):
            return _wrap_result(self._array.__gt__(other._array), "bool")
        return _wrap_result(self._array.__gt__(other), "bool")

    def __ge__(self, other):
        if _is_ndarray(other):
            return _wrap_result(self._array.__ge__(other._array), "bool")
        return _wrap_result(self._array.__ge__(other), "bool")

    def __round__(self, ndigits=None):
        if ndigits is None:
            return _wrap_result(_core.round(self._array, 0), self._dtype)
        return _wrap_result(_core.round(self._array, ndigits), self._dtype)

    def __and__(self, other):
        """按位与（& 运算符）。"""
        return bitwise_and(self, other)

    def __or__(self, other):
        """按位或（| 运算符）。"""
        return bitwise_or(self, other)

    def __xor__(self, other):
        """按位异或（^ 运算符）。"""
        return bitwise_xor(self, other)

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
        """返回数组的内存布局信息，与 NumPy 的 ndarray.flags 兼容。"""
        return ArrayFlags(self)

    @property
    def base(self):
        """若数组是另一数组的视图则返回其基数组，否则返回 None。"""
        return getattr(self, '_base', None)

    @property
    def ndim(self):
        """返回数组维度数量。"""
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            return len(self.shape)
        return self._array.ndim

    @property
    def size(self):
        """返回元素总数。"""
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
        indices = _core.ndarray(list(range(size)))
        vals = _core.ndarray([float(v) for v in flat_values])
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

    def astype(self, dtype):
        """转换数组为指定类型。"""
        return _ndarray_methods().astype(self, dtype)

    def reshape(self, *shape):
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

    def copy(self, order='K'):
        """返回数组的副本。"""
        return _wrap_result(self._array.copy(), self._dtype)

    def transpose(self, *axes):
        """转置数组。"""
        return _ndarray_methods().transpose(self, *axes)

    def swapaxes(self, axis1, axis2):
        """交换两个轴。"""
        return _ndarray_methods().swapaxes(self, axis1, axis2)

    def squeeze(self, axis=None):
        """移除长度为1的轴。"""
        return _ndarray_methods().squeeze(self, axis)

    def max(self, axis=None):
        """返回最大值。"""
        return _ndarray_methods().max(self, axis)

    def min(self, axis=None):
        """返回最小值。"""
        return _ndarray_methods().min(self, axis)

    def mean(self, axis=None):
        """计算平均值。"""
        return _ndarray_methods().mean(self, axis)

    def std(self, axis=None):
        """计算标准差。"""
        return _ndarray_methods().std(self, axis)

    def var(self, axis=None):
        """计算方差。"""
        return _ndarray_methods().var(self, axis)

    def sum(self, axis=None):
        """计算和。"""
        return _ndarray_methods().sum(self, axis)

    def prod(self, axis=None):
        """计算乘积。"""
        return _ndarray_methods().prod(self, axis)

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
                return int(val)
            return val
        raise TypeError(f"only 0-dimensional arrays can be converted to integers, got {self.ndim}D")

    def __float__(self):
        """支持将 0 维数组转换为 Python float。"""
        if self.ndim == 0:
            val = self.item()
            return float(val)
        raise TypeError(f"only 0-dimensional arrays can be converted to scalars, got {self.ndim}D")

    def __int__(self):
        """支持将 0 维数组转换为 Python int。"""
        if self.ndim == 0:
            val = self.item()
            return int(val)
        raise TypeError(f"only 0-dimensional arrays can be converted to scalars, got {self.ndim}D")

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
        return _ndarray_methods().nonzero(self)

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

    def clip(self, a_min, a_max):
        """限制值范围。"""
        return _math_funcs().clip(self, a_min, a_max)

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

    def all(self):
        """测试是否所有元素为真。"""
        return _core.all(self._array)

    def any(self):
        """测试是否有任何元素为真。"""
        return _core.any(self._array)

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
            arr = _core.ndarray(swapped)
            arr = arr.reshape(self._array.shape)
            self._array = arr
            return self
        else:
            return ndarray(swapped, _dtype=dtype)

    def view(self, dtype=None, type=None):
        """创建一个新的数组对象，共享相同的数据但拥有不同的视图。"""
        _ = type
        if dtype is not None:
            return _view_dtype(self, dtype)
        return ndarray(self._array, _dtype=getattr(self, '_dtype', 'float64'))

    def __reduce__(self):
        """支持 pickle 序列化。"""
        return (
            self.__class__,
            (
                self.tolist(),
                getattr(self, '_dtype', 'float64'),
                getattr(self, '_fields', None),
                getattr(self, '_raw_data', None),
            ),
        )


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
        return _core.ndarray(x)
    elif _is_ndarray(x):
        return x._array
    elif hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray':
        return x
    return x


def _is_ndarray(obj):
    """检查对象是否为 rsnumpy ndarray（用 hasattr 避免类身份不一致问题）。"""
    return hasattr(obj, '_array')


def _wrap_result(result, dtype="float64"):
    """将原始 ndarray 结果包装到 ndarray 类中。"""
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        if getattr(result, 'is_complex', False):
            dtype = "complex128"
        inner_array = getattr(result, '_array', result)
        return ndarray._wrap(inner_array, _dtype=dtype)
    if isinstance(result, (list, tuple)):
        return ndarray(result, _dtype=dtype)
    if isinstance(result, float) and dtype == "int64":
        return int(result)
    if hasattr(result, 'tolist'):
        return ndarray._wrap(result, _dtype=dtype)
    return result


def _scalar(x):
    """转换为标量。"""
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x


# ---------- dtype 提升与浮点判定 ----------

_FLOAT_DTYPES = ("float16", "float32", "float64")


def _is_float_dtype(dt):
    return dt in _FLOAT_DTYPES


def _operand_is_float(other):
    """判断算术运算的另一操作数是否为浮点（bool 视为整数，与 numpy 一致）。"""
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
    """按 numpy 规则推导加/减/乘结果 dtype：整数遇到浮点操作数提升为 float64，
    实数遇到复数操作数提升为 complex128。"""
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
        return _core.linalg.matmul(a._array, b._array)
    
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
            return bool(v)
        return int(v)

    return _convert(raw)


def _empty_bool_mask_key(key):
    """检测布尔掩码索引是否选中零个元素；numpy 语义下对空选择赋值为 no-op。"""
    items = key if isinstance(key, tuple) else (key,)

    def _has_true(v):
        if isinstance(v, (list, tuple)):
            return _py_any(_has_true(x) for x in v)
        return bool(v)

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
    """按 numpy 浮点列规则将一列浮点值格式化为等宽对齐的字符串。"""
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
    """格式化 1D float64 数组的 repr，带逗号和对齐（匹配 NumPy）。"""
    fmt = [format_float_scalar(v) for v in values]
    max_w = _core.imax([len(f) for f in fmt]) if fmt else 0
    parts = [f.rjust(max_w) for f in fmt]
    return "[" + ", ".join(parts) + "]"


def _format_complex_repr_1d(values):
    """格式化 1D complex128 数组的 repr。"""
    parts = [_format_complex_scalar(v) for v in values]
    return "[" + ", ".join(parts) + "]"


def _format_complex_nested(data, sep):
    """递归格式化原生复数数组（nested complex）为 numpy 风格字符串。"""
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

_PY_TYPE_TO_RS = {
    int: 'int64',
    float: 'float64',
    complex: 'complex128',
    bool: 'bool',
    bytes: 'bytes',
    str: 'str',
}

# 运行时收集已知的 numpy 类型名称
_numpy_type_names = {}


def _init_numpy_types():
    """尝试导入 numpy 并注册其标量类型名称。"""
    try:
        import numpy
    except ImportError:
        return
    for name in dir(numpy):
        if name.startswith('int') or name.startswith('uint') or name.startswith('float') \
                or name.startswith('complex') or name == 'bool_' or name == 'bool':
            _numpy_type_names[name] = name


def _resolve_type_name(tp):
    """将各种 dtype 输入解析为类型名字符串。"""
    if isinstance(tp, str):
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
    # 先查已知的 numpy 类型名
    if name in _numpy_type_names:
        return _numpy_type_names[name]
    # 再查代码表
    if name in _RS_DTYPE_CODES:
        return _RS_DTYPE_CODES[name]
    # int_ / float_ 等 numpy 特殊名称
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


# 立即初始化 numpy 类型名称
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
        return bool(v)
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
    obj._array = _core.zeros(shape if shape else (0,))
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
        arr = ndarray._wrap(_core.zeros((len(flat),) if flat else (0,)),
                            _dtype='string_', _raw_data=list(flat))
        width = base_dt._itemsize // 4 if k == 'U' else base_dt._itemsize
        arr._str_dtype = (k, width)
        return arr
    name = _scalar_name_of(base_dt)
    if k == 'c':
        flat = _flatten_data(nested) if isinstance(nested, list) else [nested]
        cflat = [complex(v) for v in flat]
        if not cflat:
            return ndarray._wrap(_core.zeros(tuple(shape) if shape else (0,)),
                                 _dtype='complex128')
        nested_c = _reshape_flat(cflat, shape) if shape else cflat
        return ndarray._wrap(_core.ndarray(nested_c), _dtype='complex128')
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
    return _field_values_to_array(values, fdt, arr.shape)


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
    if _is_ndarray(value) and getattr(value, '_dtype_obj', None) is not None:
        vrecs = _flatten_records(value._raw_data)
        for pos, i in enumerate(indices):
            src = vrecs[pos % len(vrecs)]
            recs[i] = tuple(_coerce_field_value(src[j], dt._fields[names[j]][0])
                            for j in range(len(names)))
    elif _is_ndarray(value):
        vals = value.tolist()
        for pos, i in enumerate(indices):
            sv = vals[pos % len(vals)]
            recs[i] = tuple(_coerce_field_value(sv, dt._fields[nm][0]) for nm in names)
    elif isinstance(value, (list, tuple)) and len(value) == len(names):
        rec = _coerce_record(value, dt)
        for i in indices:
            recs[i] = rec
    else:
        for i in indices:
            recs[i] = tuple(_coerce_field_value(value, dt._fields[nm][0]) for nm in names)
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
        return bool(v) if k == 'b' else v
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
    # numpy 语义：view 仅改变最后一个轴的长度，其余轴保持不变。
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
    """结构化数组的记录标量（对应 numpy.void 记录），支持字段名/整数索引读写。"""

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
# 与 NumPy 一致的标量类型层次结构，使 issubdtype / isinstance 检查按预期工作。

class generic:
    """所有 NumPy 标量类型的抽象基类。"""


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


class complexfloating(inexact):
    """复数标量的抽象基类。"""


class flexible(generic):
    """可变长度标量（字符串/void）的抽象基类。"""


class character(flexible):
    """字符标量的抽象基类。"""


int8 = type('int8', (signedinteger,), {})
int16 = type('int16', (signedinteger,), {})
int32 = type('int32', (signedinteger,), {})
int64 = type('int64', (signedinteger,), {})
uint8 = type('uint8', (unsignedinteger,), {})
uint16 = type('uint16', (unsignedinteger,), {})
uint32 = type('uint32', (unsignedinteger,), {})
uint64 = type('uint64', (unsignedinteger,), {})
float16 = type('float16', (floating,), {})
float32 = type('float32', (floating,), {})
float64 = type('float64', (floating,), {})
float_ = float64  # NumPy 别名
complex64 = type('complex64', (complexfloating,), {})
complex128 = type('complex128', (complexfloating,), {})
string_ = type('string_', (character,), {})
unicode_ = type('unicode_', (character,), {})
bytes_ = string_  # NumPy 中 bytes_ 为字节字符串类型
str_ = unicode_   # NumPy 中 str_ 为 unicode 字符串类型
bool_ = type('bool', (generic,), {})
object_ = type('object_', (generic,), {})


def _void_repr(bs):
    """void 标量的显示：每字节 \\xHH（大写十六进制）。"""
    return "b'" + ''.join('\\x%02X' % b for b in bs) + "'"


class void(flexible):
    """void 标量：存储原始字节，支持按字节相等比较（兼容 numpy.void）。"""

    def __init__(self, data):
        if isinstance(data, (bytes, bytearray)):
            self._bytes = bytes(data)
        elif isinstance(data, int):
            self._bytes = b'\x00' * data
        else:
            self._bytes = bytes(data)

    def tobytes(self):
        return self._bytes

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


# C 语言宽度别名（与 NumPy 在 64 位平台上的取值一致）
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

# NumPy 兼容的 sctypeDict / ScalarType / typecodes
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
    """浮点类型的机器精度信息（兼容 numpy.finfo）。"""

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
        self.precision = d['precision']

    @property
    def dtype(self):
        return dtype(self._name)

    def __repr__(self):
        return (f"finfo(resolution={self.resolution}, min={self.min}, "
                f"max={self.max}, dtype={self._name})")


class iinfo:
    """整数类型的取值范围信息（兼容 numpy.iinfo）。"""

    _BITS = {
        'int8': (8, True), 'int16': (16, True), 'int32': (32, True), 'int64': (64, True),
        'uint8': (8, False), 'uint16': (16, False), 'uint32': (32, False), 'uint64': (64, False),
    }

    def __init__(self, int_type):
        name = _resolve_type_name(int_type)
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
    """按 C 顺序遍历给定形状的多维索引（兼容 numpy.ndindex）。"""

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
    """同时给出多维索引与元素值的迭代器（兼容 numpy.ndenumerate）。"""

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
    """时间间隔标量，内部以天数存储，兼容 numpy.timedelta64 的常用构造与运算。"""

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
        return NotImplemented

    def __hash__(self):
        return hash(self._days)

    def __lt__(self, other):
        return self._days < float(other)

    def __le__(self, other):
        return self._days <= float(other)

    def __gt__(self, other):
        return self._days > float(other)

    def __ge__(self, other):
        return self._days >= float(other)

    def __repr__(self):
        v = self._value_in_unit()
        vi = int(_py_round(v))
        vout = vi if abs(v - vi) < 1e-9 else v
        return "numpy.timedelta64(%r,'%s')" % (vout, self._unit)


class datetime64:
    """日期时间标量，内部以“自 1970-01-01 起的天数”存储，兼容 numpy.datetime64 的常用用法。"""

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
        return self._days < float(other)

    def __le__(self, other):
        return self._days <= float(other)

    def __gt__(self, other):
        return self._days > float(other)

    def __ge__(self, other):
        return self._days >= float(other)

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

# _core.build_array 返回的 dtype 编码 → rsnumpy dtype 名称（与 _infer_int_dtype 一致）
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


def _setitem_value(value):
    """规范化赋值右值供 Rust setitem_multi 使用：
    标量原样返回；ndarray 或嵌套列表展平为 C 序浮点列表（逐元素赋值）。
    复数右值原样透传，交由 Rust coerce_value_to_nd 保留虚部。"""
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
        if _py_any(isinstance(v, complex) for v in flat):
            return value
        return [float(v) for v in flat]
    return value


_INTEGER_DTYPES = frozenset((
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
))


def _cast_setitem_value(value, dtype):
    """按目标数组 dtype 转换赋值右值：整数类型向零截断，bool 类型布尔化，
    与 numpy 赋值时的隐式类型转换一致（底层仍存 f64）。"""
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

# rsnumpy dtype 名称 → 数组接口 typestr（供 __array_interface__ 暴露给真实 numpy）
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

    缓冲区的字节解码由 Rust 底层 `_core.from_buffer_typed` 完成。
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
    arr = _core.from_buffer_typed(bytes(data), typestr, list(shape))
    return arr, dtype


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
    """创建数组。"""
    if dtype is None:
        # 数值 list/tuple 快速路径：Rust 单次完成展平 + dtype 推断 + 构造。
        if isinstance(data, (list, tuple)):
            try:
                raw, code = _core.build_array(data)
            except (ValueError, TypeError):
                raw = None
            if raw is not None:
                _dtype = _BUILD_ARRAY_DTYPES[code]
                arr = ndarray._wrap(raw, _dtype=_dtype)
                if ndmin > arr.ndim:
                    new_shape = (1,) * (ndmin - arr.ndim) + arr.shape
                    arr = ndarray._wrap(arr._array.reshape(new_shape), _dtype=_dtype)
                return arr
        # 回退慢路径：复数/字符串/不规则/标量
        if isinstance(data, tuple):
            data = list(data)
        flat, has_c, has_s = _flatten_check(data)
        if has_c:
            _dtype = "complex128"
        else:
            _dtype = _infer_int_dtype(flat)
    else:
        _dtype = _resolve_dtype(dtype)
    # 字符串/字节/void dtype（S/U/V）：存为原始 Python 数据，避免数值化。
    if dtype is not None:
        sd = _parse_str_dtype(dtype)
        if sd is not None:
            flat = _flatten_data(data) if isinstance(data, (list, tuple)) else [data]
            arr = ndarray._wrap(_core.zeros((len(flat),)), _dtype='string_', _raw_data=list(flat))
            arr._str_dtype = sd
            return arr
        # 结构化 dtype：统一走富 DType 模型
        _dt_obj = _as_struct_dtype(dtype)
        if _dt_obj is not None:
            return _make_structured_array(data, _dt_obj)
    arr = ndarray(data, _dtype=_dtype)
    # 显式复数 dtype：即使输入已是实数 ndarray（__init__ 会沿用其 dtype），
    # 也需提升为原生复数（零虚部）。
    if _dtype in ('complex128', 'complex64') and not getattr(arr._array, 'is_complex', False):
        arr = ndarray._wrap(_maybe_native_complex(arr._array, _dtype), _dtype=_dtype)
    if ndmin > arr.ndim:
        new_shape = (1,) * (ndmin - arr.ndim) + arr.shape
        arr = ndarray._wrap(arr._array.reshape(new_shape), _dtype=_dtype)
    return arr


def asarray(a, dtype=None, order=None):
    """转换输入为数组。"""
    if _is_ndarray(a):
        return a
    if dtype is not None:
        if isinstance(dtype, str):
            dt_str = dtype
        elif hasattr(dtype, '__name__'):
            dt_str = dtype.__name__
        elif hasattr(dtype, 'name'):
            dt_str = dtype.name
        else:
            dt_str = str(dtype)
        if dt_str in ("complex", "complex128", "complex64", "cfloat", "cdouble"):
            _dtype = "complex128"
        else:
            _dtype = _resolve_dtype(dtype)
    else:
        # 数值 list/tuple 快速路径：Rust 单次完成展平 + dtype 推断 + 构造。
        if isinstance(a, (list, tuple)):
            try:
                raw, code = _core.build_array(a)
            except (ValueError, TypeError):
                raw = None
            if raw is not None:
                return ndarray._wrap(raw, _dtype=_BUILD_ARRAY_DTYPES[code])
            flat, _, _ = _flatten_check(a)
            _dtype = _infer_int_dtype(flat)
        else:
            _dtype = "float64"
    return ndarray(a, _dtype=_dtype)


def asanyarray(a, dtype=None, order=None):
    """转换输入为 ndarray。"""
    return asarray(a, dtype, order)


def matrix(data, dtype=None, copy=True):
    """从数组或字符串创建矩阵对象。"""
    if isinstance(data, str):
        rows = data.split(';')
        data = []
        for row in rows:
            elements = row.split(',')
            data.append([float(e.strip()) for e in elements])
    return asarray(data, dtype=dtype)


def asmatrix(data, dtype=None):
    """将输入转换为矩阵对象。"""
    return asarray(data, dtype=dtype)


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


def array_equal(a, b):
    """判断两个数组是否形状和元素完全相同。"""
    a = _to_ndarray(a)
    b = _to_ndarray(b)
    if a.shape != b.shape:
        return False
    return bool(_core.array_equal(a._array, b._array))


def copy(a, order='K'):
    """返回数组的副本。"""
    return ndarray(a).copy(order=order)


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
    if isinstance(dtype, (list, tuple)):
        return "void"
    dt_str = dtype if isinstance(dtype, str) else dtype.__name__
    if dt_str in ("complex", "complex128", "complex64", "cfloat", "cdouble"):
        return "complex128"
    if dt_str in ("int", "int_", "intp", "intc"):
        return "int64"
    if dt_str in ("int8", "int16", "int32", "int64"):
        return dt_str
    if dt_str in ("uint", "uint8", "uint16", "uint32", "uint64"):
        return dt_str
    if dt_str in ("bool", "bool_"):
        return "bool"
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


def zeros(shape, dtype=None, order='C'):
    """返回指定形状的零数组。"""
    dt_obj = _as_struct_dtype(dtype)
    if dt_obj is not None:
        return _wrap_structured(dt_obj, _make_struct_filled(shape, dt_obj, _zero_scalar_for))
    _dtype = _resolve_dtype(dtype)
    return ndarray(_maybe_native_complex(_core.zeros(shape), _dtype), _dtype=_dtype)


def ones(shape, dtype=None, order='C'):
    """返回指定形状的1数组。"""
    dt_obj = _as_struct_dtype(dtype)
    if dt_obj is not None:
        return _wrap_structured(dt_obj, _make_struct_filled(shape, dt_obj, _one_scalar_for))
    _dtype = _resolve_dtype(dtype)
    return ndarray(_maybe_native_complex(_core.ones(shape), _dtype), _dtype=_dtype)


def empty(shape, dtype=None, order='C'):
    """返回指定形状的空数组。"""
    dt_obj = _as_struct_dtype(dtype)
    if dt_obj is not None:
        return _wrap_structured(dt_obj, _make_struct_filled(shape, dt_obj, _zero_scalar_for))
    _dtype = _resolve_dtype(dtype)
    arr = ndarray(_maybe_native_complex(_core.empty(shape), _dtype), _dtype=_dtype)
    arr._is_empty = True
    return arr


def full(shape, fill_value, dtype=None, order='C'):
    """返回指定形状的填充数组。"""
    _dtype = _resolve_dtype(dtype)
    if isinstance(fill_value, complex) or _dtype in ('complex128', 'complex64'):
        c = complex(fill_value)
        raw = _core.full(shape, c.real) + complex(0.0, c.imag)
        return ndarray(raw, _dtype='complex128' if _dtype not in ('complex128', 'complex64') else _dtype)
    return ndarray(_core.full(shape, fill_value), _dtype=_dtype)


def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的零数组。"""
    arr = ndarray(a)
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return ndarray(_core.zeros_like(arr._array), _dtype=_dtype)


def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的1数组。"""
    arr = ndarray(a)
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return ndarray(_core.ones_like(arr._array), _dtype=_dtype)


def empty_like(a, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的空数组。"""
    arr = ndarray(a)
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return ndarray(_core.empty_like(arr._array), _dtype=_dtype)


def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的填充数组。"""
    arr = ndarray(a)
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return ndarray(_core.full_like(arr._array, float(fill_value)), _dtype=_dtype)


def eye(N, M=None, k=0, dtype=None, order='C'):
    """返回对角线为1的二维数组。"""
    _dtype = _resolve_dtype(dtype)
    return ndarray(_core.eye(N, M, k), _dtype=_dtype)


def identity(n, dtype=None):
    """返回单位矩阵。"""
    return eye(n, dtype=dtype)


def arange(start=0, stop=None, step=1, dtype=None):
    """返回给定间隔内均匀间隔的值。"""
    if stop is None:
        stop = start
        start = 0
    if isinstance(start, datetime64) or isinstance(stop, datetime64) \
            or isinstance(step, timedelta64):
        return _datetime_arange(start, stop, step)
    _dtype = _resolve_dtype(dtype) if dtype is not None else _infer_int_dtype((start, stop, step))
    return ndarray(_core.arange(start, stop, step), _dtype=_dtype)


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
    raw = _core.arange(float(start), float(stop), float(step))
    return ndarray(raw, _dtype='datetime64[%s]' % unit)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """返回指定间隔内均匀间隔的数字。"""
    start_val = _scalar(_ensure(start))
    stop_val = _scalar(_ensure(stop))
    _dtype = _resolve_dtype(dtype)
    result = ndarray(_core.linspace(start_val, stop_val, num, endpoint=endpoint), _dtype=_dtype)
    if retstep:
        step = (stop_val - start_val) / (num - 1) if endpoint else (stop_val - start_val) / num
        return result, _float64(step)
    return result


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """返回对数刻度上均匀间隔的数字。"""
    _dtype = _resolve_dtype(dtype) if dtype else "float64"
    return ndarray(_core.logspace(float(start), float(stop), num, float(base)), _dtype=_dtype)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """返回几何级数上均匀间隔的数字。"""
    if start <= 0 or stop <= 0:
        raise ValueError("geomspace requires positive start and stop values")
    _dtype = _resolve_dtype(dtype) if dtype else "float64"
    return ndarray(_core.geomspace(float(start), float(stop), num), _dtype=_dtype)


# ========== 索引函数 ==========

def ix_(*args):
    """从多个序列构造开放网格，用于笛卡尔积花式索引。

    返回 ndarray 元组，每个数组沿对应维度广播。

    示例:
        >>> x = np.arange(32).reshape((8, 4))
        >>> x[np.ix_([1,5,7,2], [0,3,1,2])]
    """
    # Rust 层的 ix_rs 只解析 Python list/tuple，故先把 ndarray 等参数
    # 归一化为一维下标列表；布尔序列按 numpy 语义取 True 位置的下标。
    prepared = []
    for a in args:
        dtype = getattr(a, '_dtype', None)
        seq = a.tolist() if hasattr(a, 'tolist') else list(a)
        if dtype == 'bool' or (len(seq) > 0 and _py_all(isinstance(v, bool) for v in seq)):
            seq = [i for i, v in enumerate(seq) if v]
        else:
            seq = [int(v) for v in seq]
        prepared.append(seq)
    raw = _core.ix_rs(tuple(prepared))
    return tuple(ndarray._wrap(r) for r in raw)


def where(condition, x=None, y=None):
    """根据条件返回元素或索引。"""
    if x is not None and y is not None:
        dtype = getattr(x, '_dtype', 'float64')
        fields = getattr(x, '_fields', None)
        raw_data = getattr(x, '_raw_data', None)
        result = _core.where(_ensure(condition), _ensure(x), _ensure(y))
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
    result = _core.select_rs(cond_arrays, choice_arrays, float(default))
    return ndarray(result)


def nonzero(a):
    """返回非零元素的索引，返回元组形式的数组（兼容NumPy）。"""
    raw = _core.nonzero_arrs(_ensure(a))
    return tuple(ndarray._wrap(arr, _dtype="int64") for arr in raw)


def argwhere(a):
    """查找非零元素的索引。"""
    return ndarray(_core.argwhere(_ensure(a)))


def flatnonzero(a):
    """返回扁平化数组中非零元素的索引。"""
    return ndarray(_core.flatnonzero(_ensure(a)))


# ========== 缺失构造函数 ==========

def fromfunction(function, shape, *, dtype=None, **kwargs):
    """根据函数和形状创建数组。"""
    indices = _core._arange_arrays(shape)
    grid = _core.meshgrid(*indices, indexing='ij')
    return function(*grid, **kwargs)


def frombuffer(buffer, dtype=None, count=-1, offset=0, *, like=None):
    """从缓冲区创建一维数组（使用 Rust 层实现）。"""
    if isinstance(buffer, bytes):
        if offset > 0:
            buffer = buffer[offset:]
        if isinstance(dtype, str) and len(dtype) > 1 and dtype[0] == 'S' and dtype[1:].isdigit():
            # Byte string type (e.g. S1, S2, ...)
            elem_size = int(dtype[1:])
            total = len(buffer) // elem_size
            if count >= 0:
                total = min(total, count)
            raw = [buffer[i * elem_size:(i + 1) * elem_size] for i in range(total)]
            return ndarray(_core.zeros((len(raw),)), _dtype=dtype, _raw_data=raw)
        return ndarray(_core.bytes_to_floats(buffer, count))
    return ndarray(list(buffer))


def fromiter(iterable, dtype=None, count=-1, *, like=None):
    """从可迭代对象创建一维数组。"""
    _dtype = _resolve_dtype(dtype)
    if count >= 0:
        data = [next(iterable) for _ in range(count)]
    else:
        data = list(iterable)
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
    grids = _core.meshgrid(*arrays, indexing='ij')
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
    return _core.meshgrid(*arrays, indexing='ij')


def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    """从坐标向量返回坐标矩阵。

    默认使用 'xy' 索引（与 NumPy 一致），返回一维输入数组两两组合的网格。
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
# FFT 相关函数均由子模块 `rsnumpy.fft` 提供（与 numpy 一致，
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
        return bool(self._val)

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
        max_ndim = _core.imax([len(s) for s in all_shapes])
        padded = []
        for s in all_shapes:
            pad = [1] * (max_ndim - len(s)) + list(s)
            padded.append(pad)
        bcast_shape = tuple(_core.imax([p[i] for p in padded]) for i in range(max_ndim))
        # 广播每个数组到统一形状
        bcast_arrays = []
        for arr in arrays:
            if arr.shape == bcast_shape:
                bcast_arrays.append(arr)
            else:
                bcast_arrays.append(ndarray(_core.broadcast_to(arr._array, bcast_shape)))
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
                idx = _core.isum([c * s for c, s in zip(reversed(coords), c_strides)])
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
    return ndarray._wrap(_core.isnan(arr._array), _dtype="bool")


def binary_repr(num, width=None):
    """返回整数的二进制表示字符串。"""
    return _core.binary_repr(num, width)


def isinf(x):
    """逐元素检测是否为无穷大。"""
    arr = ndarray(x)
    return ndarray._wrap(_core.isinf(arr._array), _dtype="bool")


def isfinite(x):
    """逐元素检测是否为有限值。"""
    arr = ndarray(x)
    return ndarray._wrap(_core.isfinite(arr._array), _dtype="bool")


def iscomplex(x):
    """逐元素检测元素的虚部是否非零。"""
    arr = ndarray(x)
    cpx = getattr(arr, '_complex_data', None)
    if cpx is not None:
        return ndarray(_core.iscomplex_cpx(cpx))
    # 不是复数数组 → 全零
    shape = arr.shape
    return ndarray(_core.zeros(shape))


def _broadcast_shape(*shapes):
    """计算多个数组的广播形状。"""
    max_ndim = _core.imax([len(s) for s in shapes])
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


def broadcast(*args):
    """产生模仿广播的对象，将多个数组广播到相同形状。"""
    if not args:
        raise ValueError("broadcast requires at least one argument")
    return _Broadcast(*args)


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
    raw = _core.ndarray(a)
    if isinstance(a, int) and not isinstance(a, bool):
        return raw, 'int64'
    return raw, 'float64'

def bitwise_and(x1, x2):
    """按位与"""
    r1, dt = _to_raw(x1)
    r2, _ = _to_raw(x2)
    return _wrap_result(_core.bitwise_and(r1, r2), dt)

def bitwise_or(x1, x2):
    """按位或"""
    r1, dt = _to_raw(x1)
    r2, _ = _to_raw(x2)
    return _wrap_result(_core.bitwise_or(r1, r2), dt)

def bitwise_xor(x1, x2):
    """按位异或"""
    r1, dt = _to_raw(x1)
    r2, _ = _to_raw(x2)
    return _wrap_result(_core.bitwise_xor(r1, r2), dt)

def bitwise_not(x):
    """按位取反（等效于 ~ 运算符，对布尔数组使用逻辑取反）"""
    r, dt = _to_raw(x)
    if dt == 'bool':
        r = _core.invert(r)
    elif dt == 'uint8':
        raw = _core.bitwise_not(r)
        r = _core.bitwise_and(raw, _core.ndarray([255]))
        dt = 'uint8'
    else:
        r = _core.bitwise_not(r)
    return _wrap_result(r, dt)


# invert 与 bitwise_not 行为完全一致，直接作为别名，避免重复实现。
invert = bitwise_not

def left_shift(x1, x2):
    """左移"""
    r1, dt = _to_raw(x1)
    r2, _ = _to_raw(x2)
    return _wrap_result(_core.left_shift(r1, r2), dt)

def right_shift(x1, x2):
    """右移"""
    r1, dt = _to_raw(x1)
    r2, _ = _to_raw(x2)
    return _wrap_result(_core.right_shift(r1, r2), dt)


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

def clip(a, a_min, a_max, out=None):
    """限制值范围。"""
    a_arr = asarray(a)
    if a_min is None:
        a_min = a_arr.min()
    if a_max is None:
        a_max = a_arr.max()
    result = _clip_core(a, a_min, a_max)
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
searchsorted = _statistics_module.searchsorted
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


# 子模块

from . import fft  # noqa: E402  FFT 子模块（numpy 风格 np.fft.*）
from . import typing  # noqa: E402  类型注解子模块（numpy 风格 np.typing.*）

linalg = _linalg_module()
random = _random_module()
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
    'tile', 'repeat',
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
    'nditer',
    'isnan', 'isinf', 'isfinite',
    'save', 'load', 'loadtxt', 'savetxt', 'savez', 'genfromtxt',
    'Poly', 'polyval', 'polyfit', 'polyder', 'polyint', 'polyroots',
    'linalg', 'random', 'matlib', 'load_npz',
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
    'dtype', 'DType', 'rec', 'ma', 'recarray'
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

_sys.modules['rsnumpy'] = _sys.modules[__name__]
_sys.modules['rsnumpy.__init__'] = _sys.modules[__name__]
_sys.modules[__name__].__name__ = 'rsnumpy'

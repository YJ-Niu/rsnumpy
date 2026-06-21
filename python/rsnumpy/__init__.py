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

import builtins
import rsnumpy._core as _core
from rsnumpy._core import ndarray_iter as NdArrayIter
# ========== 子模块导入和函数挂载 ==========
# 这些导入放在 ndarray 类定义之后以避免循环导入
from . import array_methods
from . import math_functions as _math_functions_module
from . import statistics as _statistics_module
from . import array_ops as _array_ops_module

from .io import save, load, loadtxt, savetxt, savez, load_npz
from .polynomial import Poly, polyval, polyfit, polyder, polyint, polyroots
from .linalg import linalg_module as _linalg_module
from .random import random_module as _random_module


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
        if isinstance(data, ndarray):
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
            else:
                if isinstance(data, tuple):
                    data = list(data)
                if isinstance(data, list) and not _is_rectangular(data):
                    # 不规则列表（子列表长度不同），存储为原始数据
                    self._raw_data = data
                    self._array = _core.zeros((len(data),))
                elif _has_complex(data):
                    # 包含复数 → 存储为 _complex_data
                    flat = _flatten_data(data)
                    self._complex_data = [complex(v) for v in flat]
                    _dtype = "complex128"
                    self._array = _core.zeros((len(flat),))
                else:
                    self._array = _core.ndarray(data)
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

    def __repr__(self):
        fields = getattr(self, '_fields', None)
        if fields:
            return _format_structured_repr(self)
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            return _format_ragged_repr(self)
        cpx = getattr(self, '_complex_data', None)
        if cpx is not None:
            inner = _format_complex_repr_1d(cpx)
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
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            return _format_ragged_str(self)
        cpx = getattr(self, '_complex_data', None)
        if cpx is not None:
            inner = _format_complex_repr_1d(cpx)
            return inner
        if getattr(self, '_dtype', "float64") == "int64":
            if getattr(self, '_is_empty', False):
                return _core._format_int_str(self._array)
            return _core._format_int_val_str(self._array)
        return _core._format_float_str(self._array)

    def __len__(self):
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            return len(raw)
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return len(self._array)

    def tolist(self):
        """转换为 Python 列表。"""
        raw = getattr(self, '_raw_data', None)
        if raw is not None:
            return raw
        cpx = getattr(self, '_complex_data', None)
        if cpx is not None:
            return list(cpx)
        return self._array.tolist()

    def __bool__(self):
        if self.ndim == 0:
            return bool(self.tolist())
        raise ValueError("The truth value of an array with more than one element is ambiguous.")

    def __invert__(self):
        """逐元素取反（~ 运算符），用于布尔数组。"""
        return _wrap_result(_core.invert(self._array), self._dtype)

    def __getitem__(self, key):
        # 结构化数组的字段访问：a['age']
        fields = getattr(self, '_fields', None)
        if fields and isinstance(key, str):
            field_names = [f[0] for f in fields]
            if key in field_names:
                return ndarray._wrap(self._array, self._dtype, _fields=None)
        # 复数数组的布尔索引
        cpx = getattr(self, '_complex_data', None)
        if isinstance(key, ndarray) and cpx is not None:
            mask = key.tolist()
            filtered = [v for v, m in zip(cpx, mask) if m > 0.5]
            return ndarray(filtered)
        # 展开省略号（...），补充完整切片以匹配数组维度
        if isinstance(key, tuple):
            new_key = []
            ellipsis_count = builtins.sum(1 for k in key if k is Ellipsis)
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
        # 将 Python ndarray 索引替换为底层 _array（Rust 对象）
        if isinstance(key, tuple):
            key = tuple(k._array if isinstance(k, ndarray) else k for k in key)
        else:
            key = (key._array if isinstance(key, ndarray) else key,)
        # 委托给 Rust getitem_multi 处理所有索引逻辑
        result = _core.getitem_multi(self._array, key, list(self.shape))
        return _wrap_result(result, self._dtype)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            # 将 Python ndarray 索引替换为底层 _array
            key = tuple(k._array if isinstance(k, ndarray) else k for k in key)
        elif not isinstance(key, str):
            key = (key._array if isinstance(key, ndarray) else key,)
        if isinstance(key, tuple):
            _core.setitem_multi(self._array, key, list(self.shape), value)
        else:
            self._array[key] = value

    def __add__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array + other._array, self._dtype)
        return _wrap_result(self._array + other, self._dtype)

    def __radd__(self, other):
        return _wrap_result(other + self._array, self._dtype)

    def __sub__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array - other._array, self._dtype)
        return _wrap_result(self._array - other, self._dtype)

    def __rsub__(self, other):
        return _wrap_result(other - self._array, self._dtype)

    def __mul__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array * other._array, self._dtype)
        return _wrap_result(self._array * other, self._dtype)

    def __rmul__(self, other):
        return _wrap_result(other * self._array, self._dtype)

    def __truediv__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array / other._array, self._dtype)
        return _wrap_result(self._array / other, self._dtype)

    def __rtruediv__(self, other):
        return _wrap_result(other / self._array, self._dtype)

    def __matmul__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(_core.linalg.matmul(self._array, other._array), self._dtype)
        return _wrap_result(_core.linalg.matmul(self._array, other), self._dtype)

    def __pow__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(_core.power(self._array, other._array), self._dtype)
        return _wrap_result(_core.power(self._array, _core.ndarray([other])), self._dtype)

    def __eq__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__eq__(other._array), self._dtype)
        return _wrap_result(self._array.__eq__(other), self._dtype)

    def __ne__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__ne__(other._array), self._dtype)
        return _wrap_result(self._array.__ne__(other), self._dtype)

    def __lt__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__lt__(other._array), self._dtype)
        return _wrap_result(self._array.__lt__(other), self._dtype)

    def __le__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__le__(other._array), self._dtype)
        return _wrap_result(self._array.__le__(other), self._dtype)

    def __gt__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__gt__(other._array), self._dtype)
        return _wrap_result(self._array.__gt__(other), self._dtype)

    def __ge__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__ge__(other._array), self._dtype)
        return _wrap_result(self._array.__ge__(other), self._dtype)

    def __round__(self, ndigits=None):
        if ndigits is None:
            return _wrap_result(_core.round(self._array, 0), self._dtype)
        return _wrap_result(_core.round(self._array, ndigits), self._dtype)

    # ========== 属性 ==========

    @property
    def shape(self):
        """返回数组维度的元组。"""
        raw = getattr(self, '_raw_data', None)
        arr = getattr(self, '_array', None)
        if raw is not None and arr is not None:
            return self._array.shape
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
        return DType(getattr(self, '_dtype', "float64"))

    @property
    def itemsize(self):
        """每个元素的字节大小。"""
        return 8  # f64

    @property
    def nbytes(self):
        """总字节数。"""
        return self.size * self.itemsize

    @property
    def T(self):
        """数组的转置。"""
        return _ndarray_methods().transpose(self)

    @property
    def real(self):
        """数组的实部。"""
        return self.copy()

    @property
    def imag(self):
        """数组的虚部（全零）。"""
        return ndarray([0.0] * self.size).reshape(self.shape)

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

    def copy(self):
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

    def fill(self, value):
        """用值填充数组。"""
        _ndarray_methods().fill(self, value)

    def item(self, *args):
        """获取单个元素。"""
        return _ndarray_methods().item(self, *args)

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


def _ensure(x):
    """将列表/元组转换为 ndarray。"""
    if isinstance(x, (list, tuple)):
        return _core.ndarray(x)
    elif hasattr(x, '_array'):
        return x._array
    elif hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray':
        return x
    return x


def _wrap_result(result, dtype="float64"):
    """将原始 ndarray 结果包装到 ndarray 类中。"""
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return ndarray._wrap(result, _dtype=dtype)
    if isinstance(result, float) and dtype == "int64":
        return int(result)
    return result


def _wrap_with_source(result, source):
    """用源数组的 dtype 包装结果。"""
    dtype = getattr(source, '_dtype', "float64")
    return _wrap_result(result, dtype)


def _scalar(x):
    """转换为标量。"""
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x


def _format_structured_flat(arr):
    """将结构化 ndarray 展平为 Python 列表。"""
    raw = getattr(arr, '_raw_data', None)
    if raw is not None:
        return raw
    return arr.tolist()


def _format_structured_val(val, fields=None):
    """格式化结构化数组中的单个元素值。"""
    def fmt(v, field_type=None):
        if isinstance(v, str):
            return "b'" + v + "'"
        if isinstance(v, bytes):
            return "b'" + v.decode("utf-8", errors="replace") + "'"
        if isinstance(v, float):
            if v == int(v) and abs(v) < 1e16:
                return str(int(v)) + "."
            return str(v)
        # int 值且字段类型为浮点 → 显示小数点
        if isinstance(v, int) and field_type and field_type.startswith('f'):
            return str(v) + "."
        return str(v)
    if isinstance(val, (list, tuple)):
        parts = []
        for i, v in enumerate(val):
            ft = fields[i][1] if fields and i < len(fields) else None
            if not isinstance(ft, str):
                ft = _resolve_type_name(ft) if ft else None
            parts.append(fmt(v, ft))
        if len(parts) == 1:
            return "(" + ", ".join(parts) + ",)"
        return "(" + ", ".join(parts) + ")"
    return "(" + fmt(val) + ",)"


def _format_structured_str(arr):
    """__str__ 用于结构化数组。"""
    fields = getattr(arr, '_fields', None)
    flat = arr.tolist()
    if arr.ndim == 1:
        parts = [_format_structured_val(v, fields) for v in flat]
        return "[" + " ".join(parts) + "]"
    # 高维：递归格式化
    return _format_structured_recursive(flat, arr.ndim, fields)


def _format_structured_recursive(data, ndim, fields=None):
    if ndim == 1:
        parts = [_format_structured_val(v, fields) for v in data]
        return "[" + " ".join(parts) + "]"
    parts = []
    for row in data:
        parts.append(_format_structured_recursive(row, ndim - 1, fields))
    return "[" + "\n ".join(parts) + "]"


def _format_structured_repr(arr):
    """__repr__ 用于结构化数组。"""
    inner = _format_structured_str(arr)
    return f"array({inner})"


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
        has_nested = any(isinstance(x, (list, tuple)) for x in items)
        if not has_nested:
            return "[" + " ".join(str(x) for x in items) + "]"
        parts = [_format_nested_iterable(x) for x in items]
        return "[" + "\n ".join(parts) + "]"
    return str(data)


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
    max_w = builtins.max(len(f) for f in fmt) if fmt else 0
    parts = [f.rjust(max_w) for f in fmt]
    return "[" + ", ".join(parts) + "]"


def _format_complex_repr_1d(values):
    """格式化 1D complex128 数组的 repr。"""
    parts = [_format_complex_scalar(v) for v in values]
    return "[" + ", ".join(parts) + "]"


def _format_complex_scalar(val):
    """格式化单个复数为字符串（如 1.+0.j, 2.+6.j）。"""
    real = val.real
    imag = val.imag
    if real == int(real) and abs(real) < 1e16:
        real_s = f"{int(real)}."
    else:
        real_s = f"{real}"
    if imag == int(imag) and abs(imag) < 1e16:
        imag_s = f"{int(imag)}."
    else:
        imag_s = f"{imag}"
    if imag >= 0:
        return f"{real_s}+{imag_s}j"
    return f"{real_s}{imag_s}j"


def format_float_scalar(val):
    """Python 版的浮点数格式化（与 Rust 一致）。"""
    if val != val:
        return "nan"
    if val == float("inf"):
        return "inf"
    if val == float("-inf"):
        return "-inf"
    if val == int(val) and abs(val) < 1e16:
        v = int(val)
        if float(v) == val:
            return f"{v}."
    return str(val)


def _format_fields_for_dtype(fields):
    """将字段列表格式化为 dtype 字符串。"""
    parts = []
    for name, tp in fields:
        # tp 可能是类对象（如 np.int8），需要先解析为字符串
        if not isinstance(tp, str):
            tp = _resolve_type_name(tp)
        code = _RS_NAME_TO_CODE.get(tp, tp)
        parts.append(f"('{name}', '{code}')")
    return "[" + ", ".join(parts) + "]"


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
    global _numpy_type_names
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


class DType:
    """表示元素数据类型。"""

    def __init__(self, name, fields=None):
        self._name = name
        self._fields = fields  # [(field_name, type_code), ...] 或 None

    @property
    def name(self):
        return self._name

    def _field_to_code(self, tp):
        """将字段类型转换为短类型码。"""
        if isinstance(tp, str):
            return _RS_NAME_TO_CODE.get(_resolve_type_name(tp), tp)
        resolved = _resolve_type_name(tp)
        return _RS_NAME_TO_CODE.get(resolved, resolved)

    def __str__(self):
        if self._fields:
            parts = [f"('{f[0]}', '{self._field_to_code(f[1])}')" for f in self._fields]
            return "[" + ", ".join(parts) + "]"
        return self._name

    def __repr__(self):
        if self._fields:
            parts = [f"('{f[0]}', '{self._field_to_code(f[1])}')" for f in self._fields]
            return "dtype([" + ", ".join(parts) + "])"
        return f"dtype('{self._name}')"

    def __eq__(self, other):
        if isinstance(other, DType):
            if self._fields is not None or other._fields is not None:
                return self._fields == other._fields
            return self._name == other._name
        if isinstance(other, str):
            return self._name == other
        # 对外部类型（如 numpy dtype）通过字符串比较
        try:
            return self._name == str(other)
        except Exception:
            return NotImplemented


# 立即初始化 numpy 类型名称
_init_numpy_types()


def dtype(obj):
    """创建 dtype 对象。

    >>> dt = dtype('i4')
    >>> print(dt)
    int32

    >>> dt = dtype([('age', np.int8)])
    >>> print(dt)
    [('age', 'i1')]
    """
    if isinstance(obj, (list, tuple)):
        # 结构化 dtype
        fields = []
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                fields.append((item[0], item[1]))
        if fields:
            return DType("void", fields=fields)
    name = _resolve_type_name(obj)
    return DType(name)


# ========== 标量类型别名 ==========

int8 = type('int8', (), {})
int16 = type('int16', (), {})
int32 = type('int32', (), {})
int64 = type('int64', (), {})
uint8 = type('uint8', (), {})
uint16 = type('uint16', (), {})
uint32 = type('uint32', (), {})
uint64 = type('uint64', (), {})
float16 = type('float16', (), {})
float32 = type('float32', (), {})
float64 = type('float64', (), {})
complex64 = type('complex64', (), {})
complex128 = type('complex128', (), {})


# ========== 构造/工厂函数 ==========

def _has_complex(data):
    """检查数据中是否包含复数。"""
    for item in _flatten_data(data):
        if isinstance(item, complex):
            return True
    return False


def _flatten_data(data):
    """展平嵌套的可迭代对象为扁平列表。"""
    flat = []
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            flat.append(item)
    return flat


def _shape_of_nested(data):
    """计算嵌套列表的形状。"""
    shape = []
    obj = data
    while isinstance(obj, (list, tuple)):
        shape.append(len(obj))
        if obj:
            obj = obj[0]
        else:
            break
    return tuple(shape)


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
        # 从数据推断 dtype
        flat = _flatten_data(data)
        if _has_complex(data):
            _dtype = "complex128"
        else:
            _dtype = _infer_int_dtype(*flat)
    else:
        _dtype = _resolve_dtype(dtype)
    _fields = None
    _raw_data = None
    if isinstance(dtype, DType) and dtype._fields:
        _fields = dtype._fields
        # 多字段结构化数据 → 存储原始 Python 数据
        if len(_fields) > 1 or any(isinstance(row, (list, tuple)) and len(row) > 1 for row in data):
            _raw_data = list(data)
            data = [tuple(v for v in row) if isinstance(row, (list, tuple)) else row for row in data]
        else:
            # 单字段：展平数值
            flat_data = [row[0] if isinstance(row, (list, tuple)) and len(row) == 1 else row for row in data]
            data = flat_data
    arr = ndarray(data, _dtype=_dtype, _fields=_fields, _raw_data=_raw_data)
    if ndmin > arr.ndim:
        new_shape = (1,) * (ndmin - arr.ndim) + arr.shape
        arr = ndarray._wrap(arr._array.reshape(new_shape), _dtype=_dtype, _fields=_fields, _raw_data=_raw_data)
    return arr


def asarray(a, dtype=None, order=None):
    """转换输入为数组。"""
    if isinstance(a, ndarray):
        return a
    if dtype is not None:
        dt_str = dtype if isinstance(dtype, str) else dtype.__name__
        if dt_str in ("complex", "complex128", "complex64", "cfloat", "cdouble"):
            _dtype = "complex128"
        else:
            _dtype = _resolve_dtype(dtype)
    else:
        # 从数据推断 dtype
        if isinstance(a, (list, tuple)):
            flat = _flatten_data(a)
            _dtype = _infer_int_dtype(*flat)
        else:
            _dtype = "float64"
    return ndarray(a, _dtype=_dtype)


def asanyarray(a, dtype=None, order=None):
    """转换输入为 ndarray。"""
    return asarray(a, dtype, order)


def _to_ndarray(obj):
    """将任意数组类对象转换为 rsnumpy ndarray（不依赖第三方库）。"""
    if isinstance(obj, ndarray):
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
    """返回数组副本。"""
    return ndarray(a).copy()


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
    if dt_str in ("int", "int8", "int16", "int32", "int64", "intp", "int_", "intc", "uint", "uint8", "uint16", "uint32", "uint64"):
        return "int64"
    return "float64"


def _infer_int_dtype(*args):
    """推断参数中是否全部为整数，决定使用 int64 还是 float64。"""
    for a in args:
        if not isinstance(a, (int, bool)):
            return "float64"
    return "int64"


def _make_structured_zeros(shape, fields):
    """为结构化 dtype 生成零填充的原始数据。"""
    elem = tuple(0 for _ in fields)
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


def zeros(shape, dtype=None, order='C'):
    """返回指定形状的零数组。"""
    _fields = None
    _raw_data = None
    if isinstance(dtype, (list, tuple)):
        _fields = [(item[0], item[1]) for item in dtype]
        _raw_data = _make_structured_zeros(shape, _fields)
        _dtype = "void"
    elif isinstance(dtype, DType) and dtype._fields:
        _fields = dtype._fields
        _raw_data = _make_structured_zeros(shape, _fields)
        _dtype = "void"
    else:
        _dtype = _resolve_dtype(dtype)
    return ndarray(_core.zeros(shape), _dtype=_dtype, _fields=_fields, _raw_data=_raw_data)


def ones(shape, dtype=None, order='C'):
    """返回指定形状的1数组。"""
    _dtype = _resolve_dtype(dtype)
    return ndarray(_core.ones(shape), _dtype=_dtype)


def empty(shape, dtype=None, order='C'):
    """返回指定形状的空数组。"""
    _dtype = _resolve_dtype(dtype)
    arr = ndarray(_core.empty(shape), _dtype=_dtype)
    arr._is_empty = True
    return arr


def full(shape, fill_value, dtype=None, order='C'):
    """返回指定形状的填充数组。"""
    _dtype = _resolve_dtype(dtype)
    return ndarray(_core.full(shape, fill_value), _dtype=_dtype)


def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的零数组。"""
    arr = ndarray(a)
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return zeros(arr.shape, dtype=_dtype)


def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的1数组。"""
    arr = ndarray(a)
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return ones(arr.shape, dtype=_dtype)


def empty_like(a, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的空数组。"""
    arr = ndarray(a)
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return empty(arr.shape, dtype=_dtype)


def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
    """返回与输入形状相同的填充数组。"""
    arr = ndarray(a)
    _dtype = dtype if dtype is not None else getattr(arr, '_dtype', 'float64')
    return full(arr.shape, fill_value, dtype=_dtype)


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
    _dtype = _resolve_dtype(dtype) if dtype is not None else _infer_int_dtype(start, stop, step)
    return ndarray(_core.arange(start, stop, step), _dtype=_dtype)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """返回指定间隔内均匀间隔的数字。"""
    start_val = _scalar(_ensure(start))
    stop_val = _scalar(_ensure(stop))
    _dtype = _resolve_dtype(dtype)
    if num == 1:
        result = ndarray([start_val], _dtype=_dtype)
        step = 0.0
    else:
        result = ndarray(_core.linspace(start_val, stop_val, num, endpoint=endpoint))
        if _dtype != "float64":
            result._dtype = _dtype
        step = (stop_val - start_val) / (num - 1) if endpoint else (stop_val - start_val) / num
    if retstep:
        return result, _float64(step)
    return result


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """返回对数刻度上均匀间隔的数字。"""
    return power(ndarray([float(base)]), linspace(start, stop, num, endpoint))


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """返回几何级数上均匀间隔的数字。"""
    import math
    if start <= 0 or stop <= 0:
        raise ValueError("geomspace requires positive start and stop values")
    log_start = math.log(start)
    log_stop = math.log(stop)
    return exp(linspace(log_start, log_stop, num, endpoint))


# ========== 索引函数 ==========

def ix_(*args):
    """从多个序列构造开放网格，用于笛卡尔积花式索引。

    返回 ndarray 元组，每个数组沿对应维度广播。

    示例:
        >>> x = np.arange(32).reshape((8, 4))
        >>> x[np.ix_([1,5,7,2], [0,3,1,2])]
    """
    ndim = len(args)
    result = []
    for i, arr in enumerate(args):
        a = ndarray(arr)
        shape = [1] * ndim
        shape[i] = a.size
        reshaped = a.reshape(tuple(shape))
        result.append(reshaped)
    return tuple(result)


def where(condition, x=None, y=None):
    """根据条件返回元素或索引。"""
    if x is not None and y is not None:
        return ndarray(_core.where(_ensure(condition), _ensure(x), _ensure(y)))
    return ndarray(_core.nonzero(_ensure(condition)))


def take(a, indices, axis=None, out=None, mode='raise'):
    """沿指定轴取元素。"""
    arr = ndarray(a)
    return arr.take(indices, axis)


def put(a, indices, values, mode='raise'):
    """设置指定位置的元素。"""
    arr = ndarray(a)
    arr.put(indices, values)


def select(condlist, choicelist, default=0):
    """根据条件列表选择值。"""
    result = ndarray([default] * ndarray(condlist[0]).size)
    for cond, choice in zip(condlist, choicelist):
        mask = ndarray(cond)
        result = where(mask, choice, result)
    return result


def nonzero(a):
    """返回非零元素的索引，返回元组形式的数组（兼容NumPy）。"""
    raw = _core.nonzero(_ensure(a))
    # raw is Vec<Vec<usize>>, convert to tuple of 1D arrays
    result = []
    for indices in raw:
        arr = _core.ndarray(list(indices))
        result.append(ndarray(arr))
    return tuple(result)


def argwhere(a):
    """查找非零元素的索引。"""
    return ndarray(_core.argwhere(_ensure(a)))


def flatnonzero(a):
    """返回扁平化数组中非零元素的索引。"""
    return ndarray(_core.flatnonzero(_ensure(a)))


# ========== 缺失构造函数 ==========

def fromfunction(function, shape, *, dtype=None, **kwargs):
    """根据函数和形状创建数组。"""
    indices = [ndarray(list(range(d))) for d in shape]
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
        return concatenate([a.ravel() for a in arrays])


r_ = _RClass()


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


# ========== FFT 函数 ==========

def fft(a, n=None, axis=-1):
    """计算一维离散傅里叶变换。"""
    if isinstance(a, ndarray):
        a = a.tolist()
    elif isinstance(a, (list, tuple)):
        pass
    else:
        a = [float(a)]
    return _core.py_fft(a)


def ifft(a, n=None, axis=-1):
    """计算一维逆离散傅里叶变换。"""
    return _core.py_ifft(a)


def rfft(a, n=None, axis=-1):
    """计算实输入的一维离散傅里叶变换。"""
    if isinstance(a, ndarray):
        a = a.tolist()
    elif isinstance(a, (list, tuple)):
        pass
    else:
        a = [float(a)]
    return _core.py_rfft(a)


def irfft(a, n=None, axis=-1):
    """计算 rfft 的逆变换。"""
    return _core.py_irfft(a, n)


# ========== 常量 ==========
pi = 3.141592653589793
e = 2.718281828459045
euler_gamma = 0.5772156649015328606
inf = float('inf')
nan = float('nan')
newaxis = None


# ========== 判断函数 ==========

def isnan(x):
    """逐元素检测是否为 NaN。"""
    arr = ndarray(x)
    return ndarray(_core.isnan(arr._array))


def isinf(x):
    """逐元素检测是否为无穷大。"""
    arr = ndarray(x)
    return ndarray(_core.isinf(arr._array))


def isfinite(x):
    """逐元素检测是否为有限值。"""
    arr = ndarray(x)
    return ndarray(_core.isfinite(arr._array))


def iscomplex(x):
    """逐元素检测元素的虚部是否非零。"""
    arr = ndarray(x)
    cpx = getattr(arr, '_complex_data', None)
    if cpx is not None:
        return ndarray(_core.iscomplex_cpx(cpx))
    # 不是复数数组 → 全零
    shape = arr.shape
    return ndarray(_core.zeros(shape))


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
clip = _math_functions_module.clip
sinc = _math_functions_module.sinc
heaviside = _math_functions_module.heaviside
add = _math_functions_module.add
subtract = _math_functions_module.subtract
multiply = _math_functions_module.multiply
divide = _math_functions_module.divide
power = _math_functions_module.power
mod = _math_functions_module.mod
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
ptp = _statistics_module.ptp
median = _statistics_module.median
percentile = _statistics_module.percentile
quantile = _statistics_module.quantile
nanmedian = _statistics_module.nanmedian
nanpercentile = _statistics_module.nanpercentile
argmax = _statistics_module.argmax
argmin = _statistics_module.argmin
argsort = _statistics_module.argsort
sort = _statistics_module.sort
searchsorted = _statistics_module.searchsorted
cov = _statistics_module.cov
corrcoef = _statistics_module.corrcoef
histogram = _statistics_module.histogram
histogram2d = _statistics_module.histogram2d
histogramdd = _statistics_module.histogramdd
digitize = _statistics_module.digitize

# 子模块

linalg = _linalg_module()
random = _random_module()


# ========== 导出列表 ==========
__all__ = [
    'ndarray', 'NdArrayIter',
    'array', 'asarray', 'asanyarray', 'copy',
    'zeros', 'ones', 'empty', 'full',
    'zeros_like', 'ones_like', 'empty_like', 'full_like',
    'eye', 'identity',
    'arange', 'linspace', 'logspace', 'geomspace',
    'fromfunction', 'frombuffer', 'r_', 'mgrid', 'ogrid',
    'reshape', 'ravel', 'moveaxis', 'rollaxis', 'broadcast_to',
    'transpose', 'swapaxes', 'expand_dims', 'squeeze',
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',
    'split', 'hsplit', 'vsplit', 'dsplit',
    'tile', 'repeat',
    'flip', 'fliplr', 'flipud', 'roll', 'rot90',
    'where', 'take', 'put', 'select', 'nonzero', 'argwhere', 'flatnonzero',
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
    'deg2rad', 'rad2deg',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'exp', 'expm1', 'log', 'log10', 'log2', 'log1p',
    'around', 'floor', 'ceil', 'trunc', 'fix',
    'sqrt', 'square', 'cbrt', 'abs', 'sign', 'clip', 'sinc', 'heaviside',
    'add', 'subtract', 'multiply', 'divide', 'power', 'mod',
    'greater', 'less', 'equal', 'logical_and', 'logical_or', 'isclose', 'allclose',
    'sum', 'mean', 'std', 'var', 'min', 'max', 'ptp',
    'median', 'percentile', 'quantile', 'nanmedian', 'nanpercentile',
    'argmax', 'argmin', 'argsort', 'sort', 'searchsorted',
    'cov', 'corrcoef',
    'histogram', 'histogram2d', 'histogramdd', 'digitize',
    'fft', 'ifft', 'rfft', 'irfft',
    'pi', 'e', 'euler_gamma', 'inf', 'nan', 'newaxis',
    'isnan', 'isinf', 'isfinite',
    'save', 'load', 'loadtxt', 'savetxt', 'savez',
    'Poly', 'polyval', 'polyfit', 'polyder', 'polyint', 'polyroots',
    'linalg', 'random', 'load_npz'
]

"""rsnum - A Rust-powered NumPy-compatible array library.

rsnum provides a NumPy-compatible API implemented in Rust for better
performance and memory efficiency while maintaining full API compatibility.

Examples:
    >>> import rsnum as np
    >>> a = np.ndarray([1, 2, 3])
    >>> np.sum(a)
    6
    >>> np.mean(a)
    2.0
"""

import rsnum._core as _core
from rsnum._core import ndarray_iter as NdArrayIter

from . import array_methods
from . import array_ops
from . import math_functions
from . import statistics


def _ensure(x):
    """Convert list/tuple to ndarray if needed."""
    if isinstance(x, (list, tuple)):
        return _core.ndarray(x)
    elif isinstance(x, ndarray):
        return x._array
    elif hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray':
        return x
    return x


def _wrap_result(result):
    """Wrap a raw ndarray result in our ndarray class."""
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return ndarray._wrap(result)
    return result


class ndarray:
    """
    rsnum.ndarray - A multi-dimensional array object.

    rsnum.ndarray is a Rust-backed array type that provides NumPy-compatible
    functionality with improved performance and memory efficiency.

    Parameters:
        data (array_like): Initial data for the array. Can be a list, tuple,
            or any iterable.

    Attributes:
        shape: Tuple of array dimensions.
        ndim: Number of dimensions.
        size: Total number of elements.
        dtype: Data type of the array elements.
        itemsize: Size of each element in bytes.
        nbytes: Total number of bytes.
        T: Transpose of the array.
        real: Real part of the array.
        imag: Imaginary part of the array.

    Examples:
        >>> import rsnum as np
        >>> a = np.ndarray([1, 2, 3])
        >>> a.shape
        (3,)
        >>> a.ndim
        1
        >>> a.size
        3
    """
    
    def __init__(self, data):
        if isinstance(data, ndarray):
            self._array = data._array
        elif hasattr(data, '__class__') and data.__class__.__name__ == 'ndarray':
            self._array = data
        else:
            self._array = _core.ndarray(data)
    
    @staticmethod
    def _wrap(raw_array):
        """Wrap a raw Rust ndarray in our Python class."""
        obj = ndarray.__new__(ndarray)
        obj._array = raw_array
        return obj
    
    def __repr__(self):
        return repr(self._array)
    
    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return len(self._array)

    def __bool__(self):
        if self.ndim == 0:
            return bool(self.tolist())
        raise ValueError("The truth value of an array with more than one element is ambiguous.")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            import builtins
            from .__init__ import ndarray
            strides = [1]
            for s in reversed(self.shape[1:]):
                strides.insert(0, strides[0] * s)
            
            start_indices = []
            end_indices = []
            step_indices = []
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    start_indices.append(k.start or 0)
                    end_indices.append(k.stop or self.shape[i])
                    step_indices.append(k.step or 1)
                else:
                    start_indices.append(k)
                    end_indices.append(k + 1)
                    step_indices.append(1)
            
            new_shape = []
            for i in range(len(key)):
                size = (end_indices[i] - start_indices[i] + step_indices[i] - 1) // step_indices[i]
                new_shape.append(size)
            
            flat_data = self.ravel().tolist()
            result_data = []
            
            def extract(idx, dim):
                if dim == len(key):
                    flat_idx = builtins.sum(i * s for i, s in zip(idx, strides))
                    result_data.append(flat_data[flat_idx])
                else:
                    for i in range(start_indices[dim], end_indices[dim], step_indices[dim]):
                        extract(idx + [i], dim + 1)
            
            extract([], 0)
            
            return ndarray(result_data).reshape(tuple(new_shape))
        else:
            result = self._array[key]
            return _wrap_result(result)
    
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            import builtins
            from .__init__ import ndarray
            val = ndarray(value)
            flat_val = val.ravel().tolist()
            
            strides = [1]
            for s in reversed(self.shape[1:]):
                strides.insert(0, strides[0] * s)
            
            flat_key = []
            indices = []
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    indices.append(range(k.start or 0, k.stop or self.shape[i], k.step or 1))
                else:
                    indices.append([k])
            
            def assign(idx, dim):
                if dim == len(key):
                    flat_idx = builtins.sum(i * s for i, s in zip(idx, strides))
                    flat_key.append(flat_idx)
                else:
                    for i in indices[dim]:
                        assign(idx + [i], dim + 1)
            
            assign([], 0)
            
            flat_data = self.ravel().tolist()
            for i, idx in enumerate(flat_key):
                flat_data[idx] = flat_val[i % len(flat_val)]
            
            new_arr = ndarray(flat_data).reshape(self.shape)
            self._array = new_arr._array
        else:
            self._array[key] = value
    
    def __add__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array + other._array)
        return _wrap_result(self._array + other)
    
    def __radd__(self, other):
        return _wrap_result(other + self._array)
    
    def __sub__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array - other._array)
        return _wrap_result(self._array - other)
    
    def __rsub__(self, other):
        return _wrap_result(other - self._array)
    
    def __mul__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array * other._array)
        return _wrap_result(self._array * other)
    
    def __rmul__(self, other):
        return _wrap_result(other * self._array)
    
    def __truediv__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array / other._array)
        return _wrap_result(self._array / other)
    
    def __rtruediv__(self, other):
        return _wrap_result(other / self._array)
    
    def __matmul__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(_core.matmul(self._array, other._array))
        return _wrap_result(_core.matmul(self._array, other))
    
    def __pow__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(_core.pow(self._array, other._array))
        return _wrap_result(_core.pow(self._array, other))
    
    def __eq__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__eq__(other._array))
        return _wrap_result(self._array.__eq__(other))
    
    def __ne__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__ne__(other._array))
        return _wrap_result(self._array.__ne__(other))
    
    def __lt__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__lt__(other._array))
        return _wrap_result(self._array.__lt__(other))
    
    def __le__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__le__(other._array))
        return _wrap_result(self._array.__le__(other))
    
    def __gt__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__gt__(other._array))
        return _wrap_result(self._array.__gt__(other))
    
    def __ge__(self, other):
        if isinstance(other, ndarray):
            return _wrap_result(self._array.__ge__(other._array))
        return _wrap_result(self._array.__ge__(other))
    
    def __round__(self, ndigits=None):
        if ndigits is None:
            return _wrap_result(_core.round(self._array, 0))
        return _wrap_result(_core.round(self._array, ndigits))
    
    # ========== 属性 ==========
    
    @property
    def shape(self):
        """Tuple of array dimensions."""
        return self._array.shape
    
    @property
    def ndim(self):
        """Number of array dimensions."""
        return self._array.ndim
    
    @property
    def size(self):
        """Total number of elements."""
        return self._array.size
    
    @property
    def dtype(self):
        """Data type of the array elements."""
        return self._array.dtype
    
    @property
    def itemsize(self):
        """Size of each element in bytes."""
        return 8  # f64
    
    @property
    def nbytes(self):
        """Total number of bytes."""
        return self.size * self.itemsize
    
    @property
    def T(self):
        """Transpose of the array."""
        return self.transpose()
    
    @property
    def real(self):
        """Real part of the array."""
        return self.copy()
    
    @property
    def imag(self):
        """Imaginary part of the array."""
        return ndarray([0.0] * self.size).reshape(self.shape)
    
    # ========== 对象方法 ==========
    
    def astype(self, dtype):
        """Convert the array to a specified type."""
        return array_methods.NdArrayMethods.astype(self, dtype)
    
    def reshape(self, *shape):
        """Reshape the array."""
        return array_methods.NdArrayMethods.reshape(self, *shape)
    
    def resize(self, new_shape):
        """Resize the array."""
        array_methods.NdArrayMethods.resize(self, new_shape)
    
    def ravel(self, order='C'):
        """Return a flattened array."""
        return array_methods.NdArrayMethods.ravel(self, order)
    
    def flatten(self, order='C'):
        """Return a copy of the array collapsed into one dimension."""
        return array_methods.NdArrayMethods.flatten(self, order)
    
    def copy(self):
        """Return a copy of the array."""
        return _wrap_result(self._array.copy())
    
    def transpose(self, *axes):
        """Transpose the array."""
        return array_methods.NdArrayMethods.transpose(self, *axes)
    
    def swapaxes(self, axis1, axis2):
        """Swap two axes."""
        return array_methods.NdArrayMethods.swapaxes(self, axis1, axis2)
    
    def squeeze(self, axis=None):
        """Remove axes of length one."""
        return array_methods.NdArrayMethods.squeeze(self, axis)
    
    def max(self, axis=None):
        """Return the maximum value."""
        return array_methods.NdArrayMethods.max(self, axis)
    
    def min(self, axis=None):
        """Return the minimum value."""
        return array_methods.NdArrayMethods.min(self, axis)
    
    def mean(self, axis=None):
        """Compute the mean."""
        return array_methods.NdArrayMethods.mean(self, axis)
    
    def std(self, axis=None):
        """Compute the standard deviation."""
        return array_methods.NdArrayMethods.std(self, axis)
    
    def var(self, axis=None):
        """Compute the variance."""
        return array_methods.NdArrayMethods.var(self, axis)
    
    def sum(self, axis=None):
        """Compute the sum."""
        return array_methods.NdArrayMethods.sum(self, axis)
    
    def prod(self, axis=None):
        """Compute the product."""
        return array_methods.NdArrayMethods.prod(self, axis)
    
    def cumsum(self, axis=None):
        """Compute the cumulative sum."""
        return array_methods.NdArrayMethods.cumsum(self, axis)
    
    def cumprod(self, axis=None):
        """Compute the cumulative product."""
        return array_methods.NdArrayMethods.cumprod(self, axis)
    
    def argmax(self, axis=None):
        """Return the indices of the maximum values."""
        return array_methods.NdArrayMethods.argmax(self, axis)
    
    def argmin(self, axis=None):
        """Return the indices of the minimum values."""
        return array_methods.NdArrayMethods.argmin(self, axis)
    
    def argsort(self, axis=-1):
        """Return the indices that would sort the array."""
        return array_methods.NdArrayMethods.argsort(self, axis)
    
    def sort(self, axis=-1):
        """Sort the array in place."""
        array_methods.NdArrayMethods.sort(self, axis)
    
    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Return specified diagonals."""
        return array_methods.NdArrayMethods.diagonal(self, offset, axis1, axis2)
    
    def trace(self, offset=0, axis1=0, axis2=1):
        """Compute the trace."""
        return array_methods.NdArrayMethods.trace(self, offset, axis1, axis2)
    
    def fill(self, value):
        """Fill the array with a scalar value."""
        array_methods.NdArrayMethods.fill(self, value)
    
    def item(self, *args):
        """Get a single element."""
        return array_methods.NdArrayMethods.item(self, *args)
    
    def tolist(self):
        """Convert to a Python list."""
        return array_methods.NdArrayMethods.tolist(self)
    
    def take(self, indices, axis=None):
        """Take elements from an array along an axis."""
        return array_methods.NdArrayMethods.take(self, indices, axis)
    
    def put(self, indices, values):
        """Set elements in an array using indices."""
        array_methods.NdArrayMethods.put(self, indices, values)
    
    def repeat(self, repeats, axis=None):
        """Repeat elements of an array."""
        return array_methods.NdArrayMethods.repeat(self, repeats, axis)
    
    def nonzero(self):
        """Return the indices of non-zero elements."""
        return array_methods.NdArrayMethods.nonzero(self)
    
    # ========== 数学方法 ==========
    
    def sin(self):
        """Compute sine."""
        return math_functions.sin(self)
    
    def cos(self):
        """Compute cosine."""
        return math_functions.cos(self)
    
    def tan(self):
        """Compute tangent."""
        return math_functions.tan(self)
    
    def sqrt(self):
        """Compute square root."""
        return math_functions.sqrt(self)
    
    def abs(self):
        """Compute absolute value."""
        return math_functions.abs(self)
    
    def exp(self):
        """Compute exponential."""
        return math_functions.exp(self)
    
    def log(self):
        """Compute natural logarithm."""
        return math_functions.log(self)
    
    def log10(self):
        """Compute base-10 logarithm."""
        return math_functions.log10(self)
    
    def log2(self):
        """Compute base-2 logarithm."""
        return math_functions.log2(self)
    
    def clip(self, a_min, a_max):
        """Clip values."""
        return math_functions.clip(self, a_min, a_max)
    
    def round(self, decimals=0):
        """Round to specified decimals."""
        return math_functions.around(self, decimals)
    
    def floor(self):
        """Floor values."""
        return math_functions.floor(self)
    
    def ceil(self):
        """Ceil values."""
        return math_functions.ceil(self)
    
    def all(self):
        """Test if all elements are true."""
        return _core.all(self._array)
    
    def any(self):
        """Test if any element is true."""
        return _core.any(self._array)


# ========== 构造/工厂函数 ==========

def array(data, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    """Create an array."""
    return ndarray(data)


def asarray(a, dtype=None, order=None):
    """Convert the input to an array."""
    if isinstance(a, ndarray):
        return a
    return ndarray(a)


def asanyarray(a, dtype=None, order=None):
    """Convert the input to an ndarray."""
    return asarray(a, dtype, order)


def copy(a, order='K'):
    """Return an array copy of the given object."""
    return ndarray(a).copy()


def zeros(shape, dtype=None, order='C'):
    """Return a new array of given shape and type, filled with zeros."""
    return ndarray(_core.zeros(shape))


def ones(shape, dtype=None, order='C'):
    """Return a new array of given shape and type, filled with ones."""
    return ndarray(_core.ones(shape))


def empty(shape, dtype=None, order='C'):
    """Return a new array of given shape and type, without initializing entries."""
    return ndarray(_core.empty(shape))


def full(shape, fill_value, dtype=None, order='C'):
    """Return a new array of given shape and type, filled with fill_value."""
    return ndarray(_core.full(shape, fill_value))


def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    """Return an array of zeros with the same shape and type as a given array."""
    arr = ndarray(a)
    return zeros(arr.shape)


def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    """Return an array of ones with the same shape and type as a given array."""
    arr = ndarray(a)
    return ones(arr.shape)


def empty_like(a, dtype=None, order='K', subok=True, shape=None):
    """Return an empty array with the same shape and type as a given array."""
    arr = ndarray(a)
    return empty(arr.shape)


def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
    """Return a full array with the same shape and type as a given array."""
    arr = ndarray(a)
    return full(arr.shape, fill_value)


def eye(N, M=None, k=0, dtype=None, order='C'):
    """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
    return ndarray(_core.eye(N, M, k))


def identity(n, dtype=None):
    """Return the identity array."""
    return eye(n)


def arange(start=0, stop=None, step=1, dtype=None):
    """Return evenly spaced values within a given interval."""
    if stop is None:
        stop = start
        start = 0
    return ndarray(_core.arange(start, stop, step))


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """Return evenly spaced numbers over a specified interval."""
    start_val = _scalar(_ensure(start))
    stop_val = _scalar(_ensure(stop))
    result = ndarray(_core.linspace(start_val, stop_val, num))
    if retstep:
        step = (stop_val - start_val) / (num - 1) if num > 1 else 0
        return result, step
    return result


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """Return numbers spaced evenly on a log scale."""
    return power(base, linspace(start, stop, num, endpoint))


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """Return numbers spaced evenly on a log scale (a geometric progression)."""
    if start <= 0 or stop <= 0:
        raise ValueError("geomspace requires positive start and stop values")
    return exp(linspace(math.log(start), math.log(stop), num, endpoint))


def _scalar(x):
    """Convert to scalar."""
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x


# ========== 数组操作函数 ==========
reshape = array_ops.reshape
ravel = array_ops.ravel
moveaxis = array_ops.moveaxis
rollaxis = array_ops.rollaxis
broadcast_to = array_ops.broadcast_to
transpose = array_ops.transpose
swapaxes = array_ops.swapaxes
expand_dims = array_ops.expand_dims
squeeze = array_ops.squeeze
concatenate = array_ops.concatenate
stack = array_ops.stack
vstack = array_ops.vstack
hstack = array_ops.hstack
dstack = array_ops.dstack
column_stack = array_ops.column_stack
split = array_ops.split
hsplit = array_ops.hsplit
vsplit = array_ops.vsplit
dsplit = array_ops.dsplit
tile = array_ops.tile
repeat = array_ops.repeat
flip = array_ops.flip
fliplr = array_ops.fliplr
flipud = array_ops.flipud
roll = array_ops.roll
rot90 = array_ops.rot90


# ========== 索引函数 ==========
def where(condition, x=None, y=None):
    """Return elements chosen from x or y depending on condition."""
    if x is None and y is None:
        return ndarray(_core.nonzero(_ensure(condition)))
    return ndarray(_core.where(_ensure(condition), _ensure(x), _ensure(y)))


def take(a, indices, axis=None, out=None, mode='raise'):
    """Take elements from an array along an axis."""
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.take(indices, axis)


def put(a, indices, values, mode='raise'):
    """Set elements in an array using indices."""
    from .__init__ import ndarray
    arr = ndarray(a)
    arr.put(indices, values)


def select(condlist, choicelist, default=0):
    """Return an array drawn from elements in choicelist, depending on conditions."""
    from .__init__ import ndarray
    result = ndarray([default] * len(ndarray(condlist[0])))
    for cond, choice in zip(condlist, choicelist):
        mask = ndarray(cond)
        result = where(mask, choice, result)
    return result


def nonzero(a):
    """Return the indices of the non-zero elements."""
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.nonzero()


def argwhere(a):
    """Find the indices of array elements that are non-zero."""
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.nonzero()


def flatnonzero(a):
    """Return indices that are non-zero in the flattened version of a."""
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.nonzero()


# ========== 数学函数 ==========
sin = math_functions.sin
cos = math_functions.cos
tan = math_functions.tan
arcsin = math_functions.arcsin
arccos = math_functions.arccos
arctan = math_functions.arctan
arctan2 = math_functions.arctan2
deg2rad = math_functions.deg2rad
rad2deg = math_functions.rad2deg
sinh = math_functions.sinh
cosh = math_functions.cosh
tanh = math_functions.tanh
asinh = math_functions.asinh
acosh = math_functions.acosh
atanh = math_functions.atanh
exp = math_functions.exp
expm1 = math_functions.expm1
log = math_functions.log
log10 = math_functions.log10
log2 = math_functions.log2
log1p = math_functions.log1p
around = math_functions.around
floor = math_functions.floor
ceil = math_functions.ceil
trunc = math_functions.trunc
fix = math_functions.fix
sqrt = math_functions.sqrt
square = math_functions.square
cbrt = math_functions.cbrt
abs = math_functions.abs
sign = math_functions.sign
clip = math_functions.clip
sinc = math_functions.sinc
heaviside = math_functions.heaviside
add = math_functions.add
subtract = math_functions.subtract
multiply = math_functions.multiply
divide = math_functions.divide
power = math_functions.power
mod = math_functions.mod
greater = math_functions.greater
less = math_functions.less
equal = math_functions.equal
logical_and = math_functions.logical_and
logical_or = math_functions.logical_or
isclose = math_functions.isclose
allclose = math_functions.allclose


# ========== 统计函数 ==========
sum = statistics.sum
mean = statistics.mean
std = statistics.std
var = statistics.var
min = statistics.min
max = statistics.max
ptp = statistics.ptp
median = statistics.median
percentile = statistics.percentile
quantile = statistics.quantile
nanmedian = statistics.nanmedian
nanpercentile = statistics.nanpercentile
argmax = statistics.argmax
argmin = statistics.argmin
argsort = statistics.argsort
sort = statistics.sort
searchsorted = statistics.searchsorted
cov = statistics.cov
corrcoef = statistics.corrcoef
histogram = statistics.histogram
histogram2d = statistics.histogram2d
histogramdd = statistics.histogramdd
digitize = statistics.digitize


# ========== FFT 函数 ==========

def fft(a, n=None, axis=-1):
    """Compute the one-dimensional discrete Fourier Transform."""
    if isinstance(a, ndarray):
        a = a.tolist()
    elif isinstance(a, (list, tuple)):
        pass
    else:
        a = [float(a)]
    return _core.py_fft(a)


def ifft(a, n=None, axis=-1):
    """Compute the one-dimensional inverse discrete Fourier Transform."""
    return _core.py_ifft(a)


def rfft(a, n=None, axis=-1):
    """Compute the one-dimensional discrete Fourier Transform for real input."""
    if isinstance(a, ndarray):
        a = a.tolist()
    elif isinstance(a, (list, tuple)):
        pass
    else:
        a = [float(a)]
    return _core.py_rfft(a)


def irfft(a, n=None, axis=-1):
    """Compute the inverse of rfft."""
    return _core.py_irfft(a, n)


# ========== 常量 ==========
import math
pi = math.pi
e = math.e
euler_gamma = 0.5772156649015328606
inf = float('inf')
nan = float('nan')
newaxis = None


# ========== 判断函数 ==========

def isnan(x):
    """Test element-wise for NaN and return result as a boolean array."""
    from .__init__ import ndarray
    arr = ndarray(x)
    return ndarray(_core.isnan(arr._array))


def isinf(x):
    """Test element-wise for infinity and return result as a boolean array."""
    from .__init__ import ndarray
    arr = ndarray(x)
    return ndarray(_core.isinf(arr._array))


def isfinite(x):
    """Test element-wise for finiteness and return result as a boolean array."""
    from .__init__ import ndarray
    arr = ndarray(x)
    return ndarray(_core.isfinite(arr._array))


# ========== 子模块 ==========
from .linalg import linalg_module as _linalg_module
from .random import random_module as _random_module

linalg = _linalg_module()
random = _random_module()


# ========== 导出列表 ==========
__all__ = [
    'ndarray', 'NdArrayIter',
    # 构造函数
    'array', 'asarray', 'asanyarray', 'copy',
    'zeros', 'ones', 'empty', 'full',
    'zeros_like', 'ones_like', 'empty_like', 'full_like',
    'eye', 'identity',
    'arange', 'linspace', 'logspace', 'geomspace',
    # 数组操作
    'reshape', 'ravel', 'moveaxis', 'rollaxis', 'broadcast_to',
    'transpose', 'swapaxes', 'expand_dims', 'squeeze',
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',
    'split', 'hsplit', 'vsplit', 'dsplit',
    'tile', 'repeat',
    'flip', 'fliplr', 'flipud', 'roll', 'rot90',
    # 索引函数
    'where', 'take', 'put', 'select', 'nonzero', 'argwhere', 'flatnonzero',
    # 数学函数
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
    'deg2rad', 'rad2deg',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'exp', 'expm1', 'log', 'log10', 'log2', 'log1p',
    'around', 'floor', 'ceil', 'trunc', 'fix',
    'sqrt', 'square', 'cbrt', 'abs', 'sign', 'clip', 'sinc', 'heaviside',
    'add', 'subtract', 'multiply', 'divide', 'power', 'mod',
    'greater', 'less', 'equal', 'logical_and', 'logical_or', 'isclose', 'allclose',
    # 统计函数
    'sum', 'mean', 'std', 'var', 'min', 'max', 'ptp',
    'median', 'percentile', 'quantile', 'nanmedian', 'nanpercentile',
    'argmax', 'argmin', 'argsort', 'sort', 'searchsorted',
    'cov', 'corrcoef',
    'histogram', 'histogram2d', 'histogramdd', 'digitize',
    # FFT
    'fft', 'ifft', 'rfft', 'irfft',
    # 常量和判断函数
    'pi', 'e', 'euler_gamma', 'inf', 'nan', 'newaxis',
    'isnan', 'isinf', 'isfinite',
    # 子模块
    'linalg', 'random'
]

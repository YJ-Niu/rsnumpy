"""rsnum - A Rust-powered NumPy-compatible array library."""

from rsnum._core import ndarray as ndarray
from rsnum._core import ndarray_iter as NdArrayIter
import rsnum._core as _core

def _ensure(x):
    if isinstance(x, (list, tuple)):
        return _core.array(x)
    return x

def _scalar(x):
    if hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray':
        return x[0]
    return x

from rsnum._core import (
    zeros,
    ones,
    eye,
    arange,
    full,
    empty,
    argmin,
    argmax,
    concatenate,
    stack,
    vstack,
    hstack,
    nonzero,
    argwhere,
    count_nonzero,
    where,
    meshgrid,
    histogram,
)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    start = _scalar(_ensure(start))
    stop = _scalar(_ensure(stop))
    return _core.linspace(start, stop, num)

def sum(x, axis=None):
    result = _core.sum(_ensure(x), axis)
    if axis is None and hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return result[0]
    return result

def mean(x, axis=None):
    result = _core.mean(_ensure(x), axis)
    if axis is None and hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return result[0]
    return result

def min(x, axis=None):
    result = _core.min(_ensure(x), axis)
    if axis is None and hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return result[0]
    return result

def max(x, axis=None):
    result = _core.max(_ensure(x), axis)
    if axis is None and hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return result[0]
    return result

def std(x, axis=None):
    result = _core.std(_ensure(x), axis)
    if axis is None and hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return result[0]
    return result

def var(x, axis=None):
    result = _core.var(_ensure(x), axis)
    if axis is None and hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return result[0]
    return result

def sin(x):
    return _core.sin(_ensure(x))

def cos(x):
    return _core.cos(_ensure(x))

def tan(x):
    return _core.tan(_ensure(x))

def sqrt(x):
    return _core.sqrt(_ensure(x))

def exp(x):
    return _core.exp(_ensure(x))

def log(x):
    return _core.log(_ensure(x))

def log10(x):
    return _core.log10(_ensure(x))

def log2(x):
    return _core.log2(_ensure(x))

def log1p(x):
    return _core.log1p(_ensure(x))

def abs(x):
    return _core.abs(_ensure(x))

def cosh(x):
    return _core.cosh(_ensure(x))

def sinh(x):
    return _core.sinh(_ensure(x))

def tanh(x):
    return _core.tanh(_ensure(x))

def acosh(x):
    return _core.acosh(_ensure(x))

def asinh(x):
    return _core.asinh(_ensure(x))

def atanh(x):
    return _core.atanh(_ensure(x))

def asin(x):
    return _core.asin(_ensure(x))

def acos(x):
    return _core.acos(_ensure(x))

def atan(x):
    return _core.atan(_ensure(x))

def expm1(x):
    return _core.expm1(_ensure(x))

def floor(x):
    return _core.floor(_ensure(x))

def ceil(x):
    return _core.ceil(_ensure(x))

def round(x):
    return _core.round(_ensure(x))

def isnan(x):
    return _core.isnan(_ensure(x))

def isinf(x):
    return _core.isinf(_ensure(x))

def isfinite(x):
    return _core.isfinite(_ensure(x))

def all(x):
    return _core.all(_ensure(x))

def any(x):
    return _core.any(_ensure(x))

def sort(x, axis=-1):
    return _core.sort(_ensure(x), axis)

def argsort(x, axis=-1):
    return _core.argsort(_ensure(x), axis)

def median(x):
    return _core.median(_ensure(x))

def percentile(x, q):
    return _core.percentile(_ensure(x), q)

def clip(x, a_min, a_max):
    return _core.clip(_ensure(x), a_min, a_max)

def unique(x):
    return _core.unique(_ensure(x))

def transpose(x):
    return _core.transpose(_ensure(x))

def gradient(x):
    return _core.gradient(_ensure(x))

def diff(x, n=1, axis=-1):
    return _core.diff(_ensure(x), n, axis)

def trapz(y, dx=1.0):
    return _core.trapz(_ensure(y), dx)

def cross(a, b):
    return _core.cross(_ensure(a), _ensure(b))

def allclose(a, b, rtol=1e-5, atol=1e-8):
    return _core.allclose(_ensure(a), _ensure(b), rtol, atol)

def maximum(x1, x2):
    return _core.maximum(_ensure(x1), _ensure(x2))

def minimum(x1, x2):
    return _core.minimum(_ensure(x1), _ensure(x2))

def tile(a, reps):
    return _core.tile(_ensure(a), reps)

def squeeze(a):
    return _core.squeeze(_ensure(a))

def flatten(a):
    return _core.flatten(_ensure(a))

def reshape(a, shape):
    return _core.reshape(_ensure(a), shape)

class linalg_module:
    def _ensure(self, x):
        if isinstance(x, (list, tuple)):
            return _core.array(x)
        return x
    
    def dot(self, a, b):
        return _core.linalg.dot(self._ensure(a), self._ensure(b))
    
    def matmul(self, a, b):
        return _core.linalg.matmul(self._ensure(a), self._ensure(b))
    
    def inv(self, a):
        return _core.linalg.inv(self._ensure(a))
    
    def det(self, a):
        return _core.linalg.det(self._ensure(a))
    
    def norm(self, x, ord=None, axis=None):
        return _core.linalg.norm(self._ensure(x), ord, axis)
    
    def solve(self, a, b):
        return _core.linalg.solve(self._ensure(a), self._ensure(b))
    
    def eig(self, a):
        return _core.linalg.eig(self._ensure(a))
    
    def eigvals(self, a):
        return _core.linalg.eigvals(self._ensure(a))
    
    def svd(self, a):
        return _core.linalg.svd(self._ensure(a))
    
    def qr(self, a):
        return _core.linalg.qr(self._ensure(a))
    
    def cholesky(self, a):
        return _core.linalg.cholesky(self._ensure(a))
    
    def matrix_power(self, a, n):
        return _core.linalg.matrix_power(self._ensure(a), n)
    
    def pinv(self, a):
        return _core.linalg.pinv(self._ensure(a))
    
    def trace(self, a):
        return _core.linalg.trace(self._ensure(a))
    
    def diagonal(self, a):
        return _core.linalg.diagonal(self._ensure(a))
    
    def svdvals(self, a):
        return _core.linalg.svdvals(self._ensure(a))
    
    def eigh(self, a):
        return _core.linalg.eigh(self._ensure(a))
    
    def solve_banded(self, lower, upper, ab, b):
        return _core.linalg.solve_banded(lower, upper, self._ensure(ab), self._ensure(b))

class random_module:
    seed = staticmethod(_core.random.seed)
    rand = staticmethod(_core.random.rand)
    randn = staticmethod(_core.random.randn)
    randint = staticmethod(_core.random.randint)
    uniform = staticmethod(_core.random.uniform)

linalg = linalg_module()
random = random_module()
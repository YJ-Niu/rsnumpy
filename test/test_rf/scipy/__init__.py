"""
SciPy: A scientific computing package for Python
================================================

Documentation is available in the docstrings and
online at https://docs.scipy.org/doc/scipy/

Subpackages
-----------
::

 cluster                      --- Vector Quantization / Kmeans
 constants                    --- Physical and mathematical constants and units
 datasets                     --- Dataset methods
 differentiate                --- Finite difference differentiation tools
 fft                          --- Discrete Fourier transforms
 fftpack                      --- Legacy discrete Fourier transforms
 integrate                    --- Integration routines
 interpolate                  --- Interpolation Tools
 io                           --- Data input and output
 linalg                       --- Linear algebra routines
 ndimage                      --- N-D image package
 odr                          --- Orthogonal Distance Regression
 optimize                     --- Optimization Tools
 signal                       --- Signal Processing Tools
 sparse                       --- Sparse Matrices
 spatial                      --- Spatial data structures and algorithms
 special                      --- Special functions
 stats                        --- Statistical Functions

Public API in the main SciPy namespace
--------------------------------------
::

 __version__       --- SciPy version string
 LowLevelCallable  --- Low-level callback function
 show_config       --- Show scipy build configuration
 test              --- Run scipy unittests

"""

import types
import sys
import math
import os
from scipy._lib._testutils import PytestTester
from scipy._external.packaging_version.version import Version, parse
import importlib as _importlib

from rsnumpy import __version__ as __numpy_version__


try:
    from scipy.__config__ import show as show_config
except ImportError as e:
    msg = """Error importing SciPy: you cannot import SciPy while
    being in scipy source directory; please exit the SciPy source
    tree first and relaunch your Python interpreter."""
    raise ImportError(msg) from e


from scipy.version import version as __version__


# Allow distributors to run custom init code
from . import _distributor_init
del _distributor_init


# In maintenance branch, change to np_maxversion N+3 if rsnumpy is at N
np_minversion = '1.1.6'
np_maxversion = '2.8.0'
if (parse(__numpy_version__) < Version(np_minversion) or
        parse(__numpy_version__) >= Version(np_maxversion)):
    import warnings
    warnings.warn(f"A rsnumpy version >={np_minversion} and <{np_maxversion}"
                  f" is required for this version of SciPy (detected "
                  f"version {__numpy_version__})",
                  UserWarning, stacklevel=2)
del Version, parse


# This is the first import of an extension module within SciPy. If there's
# a general issue with the install, such that extension modules are missing
# or cannot be imported, this is where we'll get a failure - so give an
# informative error message.
try:
    from scipy._lib._ccallback import LowLevelCallable
except ImportError as e:
    msg = "The `scipy` install you are using seems to be broken, " + \
          "(extension modules cannot be imported), " + \
          "please try reinstalling."
    raise ImportError(msg) from e


test = PytestTester(__name__)
del PytestTester


submodules = [
    'cluster',
    'constants',
    'datasets',
    'differentiate',
    'fft',
    'fftpack',
    'integrate',
    'interpolate',
    'io',
    'linalg',
    'ndimage',
    'odr',
    'optimize',
    'signal',
    'sparse',
    'spatial',
    'special',
    'stats'
]

# Handle `_without-fortran` build option
if not os.path.exists('odr'):
    submodules.remove('odr')
del os


def _ellipk_scalar(m):
    m = float(m)
    a, b = 1.0, math.sqrt(1.0 - m)
    for _ in range(100):
        a_next = 0.5 * (a + b)
        b = math.sqrt(a * b)
        if abs(a_next - a) <= 1e-16 * abs(a_next):
            a = a_next
            break
        a = a_next
    return math.pi / (2.0 * a)


def _make_scipy_special():
    special = types.ModuleType("scipy.special")

    def ellipk(m):
        if hasattr(m, "tolist"):
            m = m.tolist()

        def _rec(v):
            if isinstance(v, list):
                return [_rec(x) for x in v]
            return _ellipk_scalar(v)

        result = _rec(m)
        if isinstance(result, list):
            import rsnumpy as _np
            return _np.array(result)
        return result

    special.ellipk = ellipk
    return special


def _make_scipy_constants():
    constants = types.ModuleType("scipy.constants")
    constants.c = constants.speed_of_light = 299792458.0
    constants.mu_0 = 1.25663706127e-06
    constants.epsilon_0 = 8.8541878188e-12
    constants.inch = 0.0254
    constants.mil = constants.inch / 1000
    return constants


sys.modules["scipy.special"] = _make_scipy_special()
sys.modules["scipy.constants"] = _make_scipy_constants()


def _make_scipy_interpolate():
    import rsnumpy as _np

    interpolate = types.ModuleType("scipy.interpolate")

    def _do_extrapolate(fill_value):
        return isinstance(fill_value, str) and fill_value == 'extrapolate'

    # kind 字符串到阶数的映射
    _KIND_ORDER = {
        'linear': 1, 'slinear': 1,
        'quadratic': 2,
        'cubic': 3,
        'zero': 0, 'nearest': 0, 'previous': 0, 'next': 0,
    }

    class interp1d:
        def __init__(self, x, y, kind='linear', axis=-1, copy=True,
                     bounds_error=None, fill_value=_np.nan, assume_sorted=False):
            self.x = x if hasattr(x, 'tolist') else _np.array(x)
            self.y = y if hasattr(y, 'tolist') else _np.array(y)
            self.kind = kind
            self.bounds_error = bounds_error
            self.fill_value = fill_value
            self._extrapolate = _do_extrapolate(fill_value)

            self.axis = axis % self.y.ndim if self.y.ndim > 0 else 0
            self._y = self.y
            if self.y.ndim > 1 and self.axis != 0:
                axes = list(range(self.y.ndim))
                axes[0], axes[self.axis] = axes[self.axis], axes[0]
                self._y = self.y.transpose(axes)

            if not assume_sorted:
                ind = _np.argsort(self.x)
                self.x = self.x[ind]
                # Use indexing instead of _np.take to preserve complex dtype
                self._y = self._y[ind]
                # Rebuild self.y from sorted _y
                if self.y.ndim > 1 and self.axis != 0:
                    axes = list(range(self._y.ndim))
                    axes[0], axes[self.axis] = axes[self.axis], axes[0]
                    self.y = self._y.transpose(axes)
                else:
                    self.y = self._y

            # 计算插值阶数
            if isinstance(kind, int):
                self._order = kind
            elif isinstance(kind, str):
                self._order = _KIND_ORDER.get(kind, 1)
            else:
                self._order = 1

        def _check_bounds(self, x_new):
            below_bounds = x_new < self.x[0]
            above_bounds = x_new > self.x[-1]
            return below_bounds, above_bounds

        def _call_linear(self, x_new):
            x_new_indices = _np.searchsorted(self.x, x_new)
            x_new_indices = x_new_indices.clip(1, len(self.x)-1).astype(int)

            lo = x_new_indices - 1
            hi = x_new_indices

            x_lo = self.x[lo]
            x_hi = self.x[hi]
            y_lo = self._y[lo]
            y_hi = self._y[hi]

            t = (x_new - x_lo) / (x_hi - x_lo)
            if self._y.ndim > 1:
                t = t.reshape([len(t)] + [1] * (self._y.ndim - 1))
            y_new = (1 - t) * y_lo + t * y_hi
            return y_new

        def _call_nearest(self, x_new):
            """最近邻插值"""
            x_new_indices = _np.searchsorted(self.x, x_new)
            x_new_indices = x_new_indices.clip(0, len(self.x)-1).astype(int)
            # 比较左右距离，取较近的
            idx_lo = (x_new_indices - 1).clip(0, len(self.x)-1)
            idx_hi = x_new_indices
            d_lo = _np.abs(x_new - self.x[idx_lo])
            d_hi = _np.abs(x_new - self.x[idx_hi])
            use_lo = d_lo < d_hi
            indices = _np.where(use_lo, idx_lo, idx_hi)
            return self._y[indices]

        def _call_poly(self, x_new, order):
            """局部多项式插值（Lagrange）。

            对每个查询点，选取最近的 order+1 个点做多项式拟合。
            由于 rsnumpy 的 polyfit 返回 Poly 对象（仅支持一维系数），
            多维数组需要展平后逐元素拟合，再恢复形状。

            注意：rsnumpy 的 reshape 可能返回副本而非视图，
            因此不能依赖 reshape 后的数组赋值回写到原数组。
            这里使用 np.unravel_index 计算多维索引后直接赋值。
            """
            x_arr = self.x
            y_arr = self._y
            n = len(x_arr)

            x_new_flat = _np.asarray(x_new).ravel()
            nq = len(x_new_flat)

            # 预分配结果
            if y_arr.ndim > 1:
                out_shape = list(y_arr.shape)
                out_shape[0] = nq
                result = _np.zeros(tuple(out_shape), dtype=y_arr.dtype)
                n_elem = int(_np.prod(y_arr.shape[1:]))
                # 预计算每个扁平索引对应的多维索引
                idx_tuples = [tuple(_np.unravel_index(i, y_arr.shape[1:])) for i in range(n_elem)]
            else:
                result = _np.zeros(nq, dtype=y_arr.dtype)
                n_elem = 1
                idx_tuples = [(0,)]

            for qi in range(nq):
                xq = float(x_new_flat[qi])
                if xq <= x_arr[0]:
                    start = 0
                    end = min(order + 1, n)
                elif xq >= x_arr[-1]:
                    start = max(0, n - order - 1)
                    end = n
                else:
                    idx = int(_np.searchsorted(x_arr, xq))
                    idx = max(1, min(idx, n - 1))
                    half = order // 2
                    start = max(0, idx - 1 - half)
                    end = min(n, start + order + 1)
                    start = max(0, end - order - 1)

                xs = x_arr[start:end]
                ys = y_arr[start:end]
                ns = len(xs)

                if ns <= 1:
                    if y_arr.ndim > 1:
                        y0 = ys[0].ravel()
                        for ei in range(n_elem):
                            result[(qi,) + idx_tuples[ei]] = float(y0[ei])
                    else:
                        result[qi] = float(ys[0])
                    continue

                deg = min(ns - 1, order)
                # 展平 ys 为 (ns, n_elem)
                if y_arr.ndim > 1:
                    ys_flat = ys.reshape(ns, n_elem)
                else:
                    ys_flat = ys.reshape(ns, 1)

                for ei in range(n_elem):
                    y_col = ys_flat[:, ei]
                    # 统一按复数处理：分别拟合实部和虚部，避免纯实数/复数混合检测问题
                    try:
                        yr = [complex(v).real for v in y_col]
                        yi = [complex(v).imag for v in y_col]
                    except (TypeError, ValueError):
                        yr = [float(v) for v in y_col]
                        yi = None
                    coeffs_r = _np.polyfit(xs, yr, deg)
                    val_r = float(_np.polyval(coeffs_r, xq))
                    if yi is not None and any(abs(im) > 1e-18 for im in yi):
                        coeffs_i = _np.polyfit(xs, yi, deg)
                        val_i = float(_np.polyval(coeffs_i, xq))
                        result[(qi,) + idx_tuples[ei]] = val_r + 1j * val_i
                    else:
                        result[(qi,) + idx_tuples[ei]] = val_r

            # 恢复形状
            orig_shape = _np.asarray(x_new).shape
            if y_arr.ndim > 1:
                if len(orig_shape) == 0:
                    result = result.reshape(y_arr.shape[1:])
                else:
                    result = result.reshape((nq,) + tuple(y_arr.shape[1:]))
            else:
                result = result.reshape(orig_shape)
            return result

        def _poly_elem(self, xs, ys, xq):
            """对多维 ys (ns, ...) 逐元素做多项式外推到 xq，保持原形状。"""
            ns = len(xs)
            if ns <= 1:
                return ys[0]
            deg = min(ns - 1, self._order)
            shape = ys.shape[1:]
            n_elem = int(_np.prod(shape)) if len(shape) > 0 else 1
            ys_flat = ys.reshape(ns, n_elem)
            out = _np.zeros(n_elem, dtype=ys.dtype)
            for ei in range(n_elem):
                y_col = ys_flat[:, ei]
                try:
                    yr = [complex(v).real for v in y_col]
                    yi = [complex(v).imag for v in y_col]
                except (TypeError, ValueError):
                    yr = [float(v) for v in y_col]
                    yi = None
                coeffs_r = _np.polyfit(xs, yr, deg)
                val_r = float(_np.polyval(coeffs_r, xq))
                if yi is not None and any(abs(im) > 1e-18 for im in yi):
                    coeffs_i = _np.polyfit(xs, yi, deg)
                    val_i = float(_np.polyval(coeffs_i, xq))
                    out[ei] = val_r + 1j * val_i
                else:
                    out[ei] = val_r
            return out.reshape(shape)

        def _evaluate(self, x_new):
            x_new = x_new if hasattr(x_new, 'tolist') else _np.array(x_new)

            if x_new.ndim == 0:
                # 标量输入
                x_new_val = float(x_new.item())
                if x_new_val < self.x[0]:
                    if self._extrapolate:
                        order = self._order
                        n = len(self.x)
                        end = min(order + 1, n)
                        xs = self.x[:end]
                        ys = self._y[:end]
                        if len(xs) <= 1:
                            return self.y[0]
                        return self._poly_elem(xs, ys, x_new_val)
                    return self.fill_value
                if x_new_val > self.x[-1]:
                    if self._extrapolate:
                        order = self._order
                        n = len(self.x)
                        start = max(0, n - order - 1)
                        xs = self.x[start:]
                        ys = self._y[start:]
                        if len(xs) <= 1:
                            return self.y[-1]
                        return self._poly_elem(xs, ys, x_new_val)
                    return self.fill_value

                # 内部点
                if self._order == 0 and self.kind in ('nearest', 'zero', 'previous', 'next'):
                    idx = int(_np.searchsorted(self.x, x_new_val))
                    idx = max(0, min(idx, len(self.x)-1))
                    y_new = self.y[idx]
                    if self.y.ndim > 1:
                        return y_new.squeeze()
                    return y_new
                elif self._order <= 1:
                    idx = int(_np.searchsorted(self.x, x_new_val))
                    idx = max(1, min(idx, len(self.x)-1))
                    lo, hi = idx - 1, idx
                    t = (x_new_val - self.x[lo]) / (self.x[hi] - self.x[lo])
                    y_new = (1 - t) * self.y[lo] + t * self.y[hi]
                    if self.y.ndim > 1:
                        return y_new.squeeze()
                    return y_new
                else:
                    order = self._order
                    n = len(self.x)
                    idx = int(_np.searchsorted(self.x, x_new_val))
                    idx = max(1, min(idx, n - 1))
                    half = order // 2
                    start = max(0, idx - 1 - half)
                    end = min(n, start + order + 1)
                    start = max(0, end - order - 1)
                    xs = self.x[start:end]
                    ys = self._y[start:end]
                    return self._poly_elem(xs, ys, x_new_val)

            # 数组输入
            if self._order == 0 and self.kind in ('nearest', 'zero', 'previous', 'next'):
                y_new = self._call_nearest(x_new)
            elif self._order <= 1:
                y_new = self._call_linear(x_new)
            else:
                y_new = self._call_poly(x_new, self._order)

            if not self._extrapolate:
                below_bounds, above_bounds = self._check_bounds(x_new)
                if y_new.size > 0:
                    y_new[below_bounds] = self.fill_value
                    y_new[above_bounds] = self.fill_value

            if self.y.ndim > 1 and self.axis != 0:
                axes = list(range(y_new.ndim))
                axes[0], axes[self.axis] = axes[self.axis], axes[0]
                y_new = y_new.transpose(axes)

            return y_new

        def __call__(self, x_new):
            return self._evaluate(x_new)

    interpolate.interp1d = interp1d
    return interpolate


sys.modules["scipy.interpolate"] = _make_scipy_interpolate()

__all__ = submodules + [
    'LowLevelCallable',
    'test',
    'show_config',
    '__version__',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'scipy.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'scipy' has no attribute '{name}'"
            )

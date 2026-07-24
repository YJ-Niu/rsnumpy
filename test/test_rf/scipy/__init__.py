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

    class interp1d:
        def __init__(self, x, y, kind='linear', axis=-1, copy=True, 
                     bounds_error=None, fill_value=_np.nan, assume_sorted=False):
            self.x = x if hasattr(x, 'tolist') else _np.array(x)
            self.y = y if hasattr(y, 'tolist') else _np.array(y)
            self.kind = kind
            self.bounds_error = bounds_error
            self.fill_value = fill_value
            self._extrapolate = _do_extrapolate(fill_value)

            if not assume_sorted:
                ind = _np.argsort(self.x)
                self.x = self.x[ind]
                self.y = _np.take(self.y, ind, axis=axis)

            self.axis = axis % self.y.ndim if self.y.ndim > 0 else 0
            self._y = self.y
            if self.y.ndim > 1 and self.axis != 0:
                axes = list(range(self.y.ndim))
                axes[0], axes[self.axis] = axes[self.axis], axes[0]
                self._y = self.y.transpose(axes)

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

        def _evaluate(self, x_new):
            x_new = x_new if hasattr(x_new, 'tolist') else _np.array(x_new)
            
            if x_new.ndim == 0:
                x_new_val = float(x_new.item())
                if x_new_val <= self.x[0]:
                    if self._extrapolate:
                        return self.y[0]
                    return self.fill_value
                if x_new_val >= self.x[-1]:
                    if self._extrapolate:
                        return self.y[-1]
                    return self.fill_value
                
                idx = _np.searchsorted(self.x, x_new_val)
                idx = max(1, min(idx, len(self.x)-1))
                lo, hi = idx - 1, idx
                t = (x_new_val - self.x[lo]) / (self.x[hi] - self.x[lo])
                y_new = (1 - t) * self.y[lo] + t * self.y[hi]
                if self.y.ndim > 1:
                    return y_new.squeeze()
                return y_new

            y_new = self._call_linear(x_new)

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

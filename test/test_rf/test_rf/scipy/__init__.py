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

    class interp1d:
        def __init__(self, x, y, kind='linear', **kwargs):
            self.x = x if hasattr(x, 'tolist') else _np.array(x)
            self.y = y if hasattr(y, 'tolist') else _np.array(y)
            self.kind = kind

        def __call__(self, x_new):
            x_new = x_new if hasattr(x_new, 'tolist') else _np.array(x_new)

            x = self.x
            y = self.y

            if x_new.ndim == 0:
                x_new_val = float(x_new.item())
                if x_new_val <= x[0]:
                    return y[0]
                if x_new_val >= x[-1]:
                    return y[-1]

                for i in range(len(x) - 1):
                    if x[i] <= x_new_val <= x[i+1]:
                        t = (x_new_val - x[i]) / (x[i+1] - x[i])
                        return y[i] * (1 - t) + y[i+1] * t

                return y[-1]
            else:
                if y.ndim > 1:
                    result_shape = list(x_new.shape) + list(y.shape[1:])
                    result = _np.empty(result_shape, dtype=y.dtype)
                else:
                    result = _np.empty_like(x_new)

                for i in range(len(x_new)):
                    xi = float(x_new[i].item())
                    if xi <= x[0]:
                        result[i] = y[0]
                    elif xi >= x[-1]:
                        result[i] = y[-1]
                    else:
                        for j in range(len(x) - 1):
                            if x[j] <= xi <= x[j+1]:
                                t = (xi - x[j]) / (x[j+1] - x[j])
                                result[i] = y[j] * (1 - t) + y[j+1] * t
                                break
                return result

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

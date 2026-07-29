"""
=============================================
Integration and ODEs (:mod:`scipy.integrate`)
=============================================

.. currentmodule:: scipy.integrate

Integrating functions, given function object
============================================

.. autosummary::
   :toctree: generated/

   quad          -- General purpose integration
   quad_vec      -- General purpose integration of vector-valued functions
   cubature      -- General purpose multi-dimensional integration of array-valued functions
   dblquad       -- General purpose double integration
   tplquad       -- General purpose triple integration
   nquad         -- General purpose N-D integration
   tanhsinh      -- General purpose elementwise integration
   fixed_quad    -- Integrate func(x) using Gaussian quadrature of order n
   newton_cotes  -- Weights and error coefficient for Newton-Cotes integration
   lebedev_rule
   qmc_quad      -- N-D integration using Quasi-Monte Carlo quadrature
   IntegrationWarning -- Warning on issues during integration


Integrating functions, given fixed samples
==========================================

.. autosummary::
   :toctree: generated/

   trapezoid            -- Use trapezoidal rule to compute integral.
   cumulative_trapezoid -- Use trapezoidal rule to cumulatively compute integral.
   simpson              -- Use Simpson's rule to compute integral from samples.
   cumulative_simpson   -- Use Simpson's rule to cumulatively compute integral from samples.
   romb                 -- Use Romberg Integration to compute integral from
                        -- (2**k + 1) evenly-spaced samples.

.. seealso::

   :mod:`scipy.special` for orthogonal polynomials (special) for Gaussian
   quadrature roots and weights for other weighting factors and regions.

Summation
=========

.. autosummary::
   :toctree: generated/

   nsum

Solving initial value problems for ODE systems
==============================================

The solvers are implemented as individual classes, which can be used directly
(low-level usage) or through a convenience function.

.. autosummary::
   :toctree: generated/

   solve_ivp     -- Convenient function for ODE integration.
   RK23          -- Explicit Runge-Kutta solver of order 3(2).
   RK45          -- Explicit Runge-Kutta solver of order 5(4).
   DOP853        -- Explicit Runge-Kutta solver of order 8.
   Radau         -- Implicit Runge-Kutta solver of order 5.
   BDF           -- Implicit multi-step variable order (1 to 5) solver.
   LSODA         -- LSODA solver from ODEPACK Fortran package.
   OdeSolver     -- Base class for ODE solvers.
   DenseOutput   -- Local interpolant for computing a dense output.
   OdeSolution   -- Class which represents a continuous ODE solution.


Old API
-------

These are the routines developed earlier for SciPy. They wrap older solvers
implemented in Fortran (mostly ODEPACK). While the interface to them is not
particularly convenient and certain features are missing compared to the new
API, the solvers themselves are of good quality and work fast as compiled
Fortran code. In some cases, it might be worth using this old API.

.. autosummary::
   :toctree: generated/

   odeint        -- General integration of ordinary differential equations.
   ode           -- Integrate ODE using VODE and ZVODE routines.
   complex_ode   -- Convert a complex-valued ODE to real-valued and integrate.
   ODEintWarning -- Warning raised during the execution of `odeint`.


Solving boundary value problems for ODE systems
===============================================

.. autosummary::
   :toctree: generated/

   solve_bvp     -- Solve a boundary value problem for a system of ODEs.
"""  # noqa: E501

import rsnumpy as np


def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """使用复合梯形法则累积积分 y(x)。

    参数
    ----------
    y : array_like
        要积分的值。
    x : array_like, optional
        沿其积分的坐标。如果为 None（默认），使用 `y` 中连续元素之间的间距 `dx`。
    dx : float, optional
        `y` 元素之间的间距。仅在 `x` 为 None 时使用。
    axis : int, optional
        指定累积的轴。默认为 -1（最后一个轴）。
    initial : scalar, optional
        如果给定，在返回结果的开头插入此值。仅接受 0 或 None。

    返回
    -------
    res : ndarray
        `y` 沿 `axis` 累积积分的结果。
    """
    y = np.asarray(y)

    if y.shape[axis] == 0:
        raise ValueError("At least one point is required along `axis`.")

    # 将负 axis 转为正 axis，避免后续操作出错
    nd = len(y.shape)
    if axis < 0:
        axis += nd

    if x is None:
        d = dx
    else:
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # 重塑为正确的形状
            shape = [1] * nd
            shape[axis] = -1
            d = d.reshape(tuple(shape))
        elif len(x.shape) != nd:
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        else:
            d = np.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    slice1 = tuple(slice(None) if i != axis else slice(1, None) for i in range(nd))
    slice2 = tuple(slice(None) if i != axis else slice(None, -1) for i in range(nd))
    res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        if initial != 0:
            raise ValueError("`initial` must be `None` or `0`.")

        shape = list(res.shape)
        shape[axis] = 1
        # 在 axis 前面插入 initial=0 的切片
        zeros_shape = tuple(shape)
        zeros_arr = np.full(zeros_shape, initial, dtype=res.dtype)
        res = np.concatenate((zeros_arr, res), axis=axis)

    return res


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """使用复合梯形法则沿给定轴积分。"""
    y = np.asarray(y)

    if y.shape[axis] == 0:
        return 0.0

    if x is None:
        d = dx
    else:
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(tuple(shape))
        else:
            d = np.diff(x, axis=axis)

    nd = len(y.shape)
    slice1 = tuple(slice(None) if i != axis else slice(1, None) for i in range(nd))
    slice2 = tuple(slice(None) if i != axis else slice(None, -1) for i in range(nd))
    return np.sum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)


# Deprecated namespaces, to be removed in v2.0.0

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester

"""多项式模块 - NumPy 兼容的 polynomial 子模块 stub。

提供与 numpy.polynomial.polynomial 兼容的接口，实现位于 Rust。
这里仅提供最小的兼容层。
"""

from __future__ import annotations

from ..polynomial import Poly, polyval, polyfit, polyder, polyint, polyroots

# Alias Polynomial to Poly for NumPy compatibility
Polynomial = Poly


def polyval(x, c):
    """Evaluate a polynomial at points x."""
    if isinstance(c, Poly):
        return c(x)
    return Poly(c)(x)


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    """Least-squares polynomial fit."""
    return polyfit(x, y, deg)


__all__ = ['Polynomial', 'polyval', 'polyfit', 'polyder', 'polyint', 'polyroots']

"""多项式模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnum._core as _core


def _nd():
    from ..__init__ import ndarray as _n
    return _n


def _wrap(result):
    return _nd()(result)


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
    return _core.ndarray(a)


class Poly:
    """多项式类（系数为 NumPy 兼容的高到低排列）。"""

    def __init__(self, coef):
        nd = _nd()
        arr = nd(coef)
        self.coef = arr.ravel().tolist()
        self._coef_raw = _ensure_raw(arr)

    def __call__(self, x):
        """计算多项式在 x 处的值。"""
        nd = _nd()
        x_arr = nd(x) if not hasattr(x, '_array') else x
        return _wrap(_core.polyval_rs(self._coef_raw, x_arr._array))

    def __repr__(self):
        terms = []
        deg = len(self.coef) - 1
        for i, c in enumerate(self.coef):
            if c == 0:
                continue
            p = deg - i
            if p == 0:
                terms.append(str(c))
            elif p == 1:
                terms.append(str(c) + "x")
            else:
                terms.append(str(c) + "x^" + str(p))
        if not terms:
            return "Poly([0.])"
        return "Poly([" + ", ".join(terms) + "])"

    def __add__(self, other):
        if isinstance(other, Poly):
            return Poly(_wrap(_core.polyadd(self._coef_raw, other._coef_raw)))
        return Poly(_wrap(_core.polyadd(self._coef_raw, _core.ndarray([float(other)]))))

    def __sub__(self, other):
        if isinstance(other, Poly):
            return Poly(_wrap(_core.polysub(self._coef_raw, other._coef_raw)))
        return Poly(_wrap(_core.polysub(self._coef_raw, _core.ndarray([float(other)]))))

    def __mul__(self, other):
        if isinstance(other, Poly):
            return Poly(_wrap(_core.polymul(self._coef_raw, other._coef_raw)))
        return Poly(_wrap(_core.polymul(self._coef_raw, _core.ndarray([float(other)]))))

    @property
    def degree(self):
        """多项式的次数。"""
        return len(self.coef) - 1

    def deriv(self, m=1):
        """计算多项式的 m 阶导数。"""
        return Poly(_wrap(_core.polyder_rs(self._coef_raw, m)).ravel().tolist())

    def integ(self, m=1, k=0):
        """计算多项式的不定积分。"""
        return Poly(_wrap(_core.polyint_rs(self._coef_raw, m, k)).ravel().tolist())

    def roots(self):
        """计算多项式的根。"""
        return _wrap(_core.polyroots_rs(self._coef_raw))


def polyval(p, x):
    """计算多项式在 x 处的值。"""
    if isinstance(p, Poly):
        return p(x)
    return Poly(p)(x)


def polyfit(x, y, deg):
    """多项式拟合（最小二乘法）。"""
    nd = _nd()
    x_arr = nd(x) if not hasattr(x, '_array') else x
    y_arr = nd(y) if not hasattr(y, '_array') else y
    coef = _wrap(_core.polyfit_rs(x_arr._array, y_arr._array, deg))
    return Poly(coef)


def polyder(p, m=1):
    """计算多项式的导数。"""
    if isinstance(p, Poly):
        return p.deriv(m)
    return Poly(p).deriv(m)


def polyint(p, m=1, k=0):
    """计算多项式的不定积分。"""
    if isinstance(p, Poly):
        return p.integ(m, k)
    return Poly(p).integ(m, k)


def polyroots(p):
    """计算多项式的根。"""
    if isinstance(p, Poly):
        return p.roots()
    return Poly(p).roots()

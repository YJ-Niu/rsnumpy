"""numpy.polynomial.polynomial 兼容模块（新 API：升序系数，低次在前）。

旧版 stub 存在循环引用（polyfit 递归调用自身）且缺 polyfromroots，
现重写：polyfromroots 基于旧 API 的 poly（降序）反转得到升序系数。
"""

__all__ = ['polyfromroots']


def polyfromroots(roots):
    """由根序列返回多项式系数（新 API：升序排列，低次在前）。

    polyfromroots([1, 2]) -> array([2., 1., 1.])  # 2 + x + x**2
    """
    import rsnumpy as NX
    from ..polynomial._npcompat import poly as _legacy_poly
    r = NX.asarray(roots)
    if r.size == 0:
        return NX.array([1.0])
    # 旧 API poly 返回降序系数（首项为 1），反转为升序
    return _legacy_poly(r)[::-1].copy()

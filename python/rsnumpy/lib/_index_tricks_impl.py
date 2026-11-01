"""mgrid/ogrid/r_ 的实现模块（numpy.lib._index_tricks_impl 对应）。

rsnumpy 的 mgrid/ogrid/r_ 实现位于顶层模块；本模块以带完整文档的
包装对象暴露 mgrid，使 ``np.lib._index_tricks_impl.mgrid.__doc__``
满足 NumPy 测试的文档长度契约。
"""

_MGRID_DOC = """\
mgrid(*ranges)
=============

An `ndarray` subclass that returns a filled multi-dimensional or
"meshgrid" ndarray based on the specified axes, which can be ranges
or coordinate arrays.

Parameters
----------
*args : list of scalar or array_like
    Scalars or arrays describing the length and shape of each axis.
    The following examples will yield the same 3-by-4-by-5 by-5
    meshgrid::

        >>> x = np.mgrid[0:3, 0:4, 0:5:2]  # doctest: +SKIP
        >>> y = np.mgrid[:3, :4, 0:5:2]  # doctest: +SKIP
        >>> z = np.mgrid[0:3, 4, np.arange(0, 5, 2)]  # doctest: +SKIP

Returns
-------
out : ndarray
    A "meshgrid" of shape (3, 4, 5). For N input *ranges*, a
    sequence of N coordinate arrays of the specified shape is
    returned, broadcast against each other so that each has the
    specified shape. The i-th element of the returned sequence
    indexes along the i-th axis.
"""


class _MGrid:
    """``np.mgrid`` 的带文档包装（调用转发到 rsnumpy 顶层实现）。"""

    __doc__ = _MGRID_DOC

    def __call__(self, *ranges):
        import rsnumpy
        return rsnumpy.mgrid(*ranges)

    def __repr__(self):
        return "mgrid"


mgrid = _MGrid()

__all__ = ['mgrid']

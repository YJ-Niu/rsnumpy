# SPDX-License-Identifier: MIT
"""numpy._core._rational_tests 兼容实现。

NumPy 在该模块定义测试专用 ``rational`` 标量 dtype（C 层类型：真实
dtype，但不在 sctypeDict 中）。rsnumpy 以纯 Python 类近似：

- ``dtype`` 属性返回 object dtype——使标量构造路径（``np.array`` /
  ``astype`` / setitem）可用；
- 类型本身不在 sctypeDict 中，且类级 ``rational.dtype`` 是 property
  对象（非 dtype 实例）——使 ``sctype2char`` 得到与 NumPy 一致的
  KeyError。

rational dtype 的专用存储与算术未实现（依赖它的 casting 测试仍失败）。
"""

__all__ = ['rational', 'rational2']


class rational:
    """测试专用有理数标量（分子/分母对）。"""

    def __init__(self, real, imag=1):
        self.real = real
        self.imag = imag

    @property
    def dtype(self):
        import rsnumpy
        return rsnumpy.dtype('O')

    def __repr__(self):
        return f"rational({self.real}, {self.imag})"

    def __eq__(self, other):
        return (isinstance(other, rational)
                and (self.real, self.imag) == (other.real, other.imag))

    def __hash__(self):
        return hash((self.real, self.imag))


class rational2(rational):
    """第二个测试用有理数类型（独立的类身份，共享 object dtype 近似）。"""

    def __repr__(self):
        return f"rational2({self.real}, {self.imag})"


# rsnumpy 不支持用户自定义 C 标量 dtype；通过 __rsnumpy_dtype__ 协议让
# np.dtype(rational) 解析到 object dtype（_dtypes._resolve_scalar_obj 识别）。
rational.__rsnumpy_dtype__ = 'O'
rational2.__rsnumpy_dtype__ = 'O'

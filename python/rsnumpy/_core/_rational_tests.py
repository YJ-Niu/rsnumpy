"""numpy._core._rational_tests 兼容实现。

NumPy 在该模块定义测试专用 ``rational`` 标量 dtype（C 层类型）。
rsnumpy 以纯 Python 类近似：提供 ``rational``/``rational2`` 名称与
非 dtype 的 ``.dtype`` 哨兵，使 ``sctype2char``/``obj2sctype`` 等检查
得到与 NumPy 一致的 KeyError 语义。rational dtype 的存储与算术
未实现（依赖它的 scalar coercion 测试仍会失败）。
"""

__all__ = ['rational', 'rational2']


class _RationalDType:
    """dtype 哨兵（非 rsnumpy dtype，仅用于类型检查兼容）。"""

    kind = 'V'
    itemsize = 16

    def __init__(self, type_):
        self.type = type_

    def __repr__(self):
        return f"rational_dtype({self.type.__name__})"

    def __eq__(self, other):
        return (isinstance(other, _RationalDType)
                and other.type is self.type)

    def __hash__(self):
        return id(self)


class rational:
    """测试专用有理数标量（分子/分母对）。"""

    def __init__(self, real, imag=1):
        self.real = real
        self.imag = imag

    @property
    def dtype(self):
        return _RATIONAL_DT

    def __repr__(self):
        return f"rational({self.real}, {self.imag})"

    def __eq__(self, other):
        return (isinstance(other, rational)
                and (self.real, self.imag) == (other.real, other.imag))

    def __hash__(self):
        return hash((self.real, self.imag))


_RATIONAL_DT = _RationalDType(rational)


class rational2(rational):
    """第二个测试用有理数类型（独立 dtype 哨兵）。"""

    @property
    def dtype(self):
        return _RATIONAL2_DT

    def __repr__(self):
        return f"rational2({self.real}, {self.imag})"


_RATIONAL2_DT = _RationalDType(rational2)

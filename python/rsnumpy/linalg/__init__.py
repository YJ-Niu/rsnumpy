"""rsnumpy.linalg - 线性代数模块 - 薄包装，所有实现位于 Rust。"""

import rsnumpy._core as _core


def _ensure(x):
    if isinstance(x, (list, tuple)):
        return _core.ndarray(x)
    elif hasattr(x, '_array'):
        return x._array
    return x


def _wrap(x):
    from ..__init__ import ndarray
    if isinstance(x, ndarray):
        return x
    return ndarray(x)


class linalg_module:
    """线性代数模块 - 所有方法都直接调用 Rust 实现。"""

    @staticmethod
    def dot(a, b):
        """计算两个数组的点积。"""
        return _wrap(_core.linalg.dot(_ensure(a), _ensure(b)))

    @staticmethod
    def matmul(a, b):
        """计算两个数组的矩阵乘积。"""
        return _wrap(_core.linalg.matmul(_ensure(a), _ensure(b)))

    @staticmethod
    def inv(a):
        """计算矩阵的逆。"""
        return _wrap(_core.linalg.inv(_ensure(a)))

    @staticmethod
    def det(a):
        """计算矩阵的行列式。"""
        return _core.linalg.det(_ensure(a))

    @staticmethod
    def norm(x, ord=None, axis=None):
        """计算矩阵或向量的范数。"""
        ord_val = ord if ord is not None else 2.0
        if ord == float('inf'):
            ord_val = None
        return _wrap(_core.linalg.norm(_ensure(x), ord_val, axis))

    @staticmethod
    def solve(a, b):
        """求解线性方程组。"""
        return _wrap(_core.linalg.solve(_ensure(a), _ensure(b)))

    @staticmethod
    def eig(a):
        """计算矩阵的特征值和特征向量。"""
        return _core.linalg.eig(_ensure(a))

    @staticmethod
    def eigvals(a):
        """计算矩阵的特征值。"""
        return _wrap(_core.linalg.eigvals(_ensure(a)))

    @staticmethod
    def svd(a):
        """计算奇异值分解。"""
        return _core.linalg.svd(_ensure(a))

    @staticmethod
    def qr(a):
        """计算 QR 分解。"""
        return _core.linalg.qr(_ensure(a))

    @staticmethod
    def cholesky(a):
        """计算 Cholesky 分解。"""
        return _wrap(_core.linalg.cholesky(_ensure(a)))

    @staticmethod
    def matrix_power(a, n):
        """计算矩阵的幂。"""
        return _wrap(_core.linalg.matrix_power(_ensure(a), n))

    @staticmethod
    def pinv(a):
        """计算矩阵的伪逆。"""
        return _wrap(_core.linalg.pinv(_ensure(a)))

    @staticmethod
    def trace(a):
        """计算矩阵的迹。"""
        return _core.linalg.trace(_ensure(a))

    @staticmethod
    def diagonal(a):
        """返回矩阵的对角线元素。"""
        return _wrap(_core.linalg.diagonal(_ensure(a)))

    @staticmethod
    def svdvals(a):
        """计算奇异值。"""
        return _wrap(_core.linalg.svdvals(_ensure(a)))

    @staticmethod
    def eigh(a):
        """计算 Hermitian 或对称矩阵的特征值和特征向量。"""
        return _core.linalg.eigh(_ensure(a))

    @staticmethod
    def solve_banded(lower, upper, ab, b):
        """求解带状线性方程组。"""
        return _wrap(_core.linalg.solve_banded(lower, upper, _ensure(ab), _ensure(b)))

"""rsnumpy.linalg - 线性代数模块 - 薄包装，所有实现位于 Rust。"""

import sys as _sys

import rsnumpy._core as _core


def _ensure(x):
    if isinstance(x, (list, tuple)):
        return _core.ndarray(x)
    elif hasattr(x, '_array'):
        return x._array
    return x


def _round_float(v):
    """对浮点数进行四舍五入，避免出现 .999999999999999 等精度问题。"""
    if abs(v - round(v)) < 1e-10:
        return round(v)
    return v


def _round_array(arr):
    """对数组中的每个元素进行四舍五入处理。"""
    if hasattr(arr, '__class__') and arr.__class__.__name__ == 'ndarray':
        raw = arr.tolist()
        
        def round_nested(data):
            if isinstance(data, list):
                return [round_nested(x) for x in data]
            return _round_float(data)
        
        rounded = round_nested(raw)
        return _core.ndarray(rounded)
    return arr


def _wrap(x):
    """包装/转换数组，避免类身份不一致问题。"""
    if getattr(x, '__class__', None) and x.__class__.__name__ == 'ndarray':
        x = _round_array(x)
        return x
    from ..__init__ import ndarray
    if hasattr(x, 'tolist'):
        x = _round_array(x)
    return ndarray(x)


def _to_np(x):
    """rsnumpy 数组/列表 → numpy 数组（复数经 tolist 保真）。"""
    import rsnumpy as _np
    if isinstance(x, (list, tuple)):
        return _np.array(x)
    if hasattr(x, 'tolist'):
        return _np.array(x.tolist())
    return _np.array(x)


def _from_np(arr):
    """numpy 数组 → rsnumpy 数组；0 维返回标量。"""
    if getattr(arr, 'ndim', None) == 0:
        return arr.item()
    from ..__init__ import ndarray
    return ndarray(arr.tolist())


def _use_numpy(*arrays):
    """复数或批量（>2 维）输入交由 numpy 委托，其余走 Rust 原生实现。"""
    for a in arrays:
        rust = getattr(a, '_array', None)
        if rust is not None and getattr(rust, 'is_complex', False):
            return True
        shp = getattr(a, 'shape', None)
        if shp is not None and len(shp) > 2:
            return True
    return False


class linalg_module:
    """线性代数模块 - 所有方法都直接调用 Rust 实现。"""

    @staticmethod
    def dot(a, b):
        """计算两个数组的点积。"""
        from ..__init__ import ndarray
        if _use_numpy(a, b):
            import rsnumpy as _np
            return _from_np(_np.dot(_to_np(a), _to_np(b)))
        result = _core.linalg.dot(_ensure(a), _ensure(b))
        a_dtype = getattr(a, '_dtype', 'float64')
        b_dtype = getattr(b, '_dtype', 'float64')
        if a_dtype in ('float64', 'float32', 'float16') or b_dtype in ('float64', 'float32', 'float16'):
            return ndarray._wrap(result, _dtype='float64')
        return ndarray._wrap(result, _dtype='int64')

    @staticmethod
    def vdot(a, b):
        """计算两个向量的点积（展开为一维）。"""
        result = _core.linalg.vdot(_ensure(a), _ensure(b))
        return _round_float(result)

    @staticmethod
    def inner(a, b):
        """计算两个数组的内积。"""
        from ..__init__ import ndarray
        result = _core.linalg.inner(_ensure(a), _ensure(b))
        a_dtype = getattr(a, '_dtype', 'float64')
        b_dtype = getattr(b, '_dtype', 'float64')
        if a_dtype in ('float64', 'float32', 'float16') or b_dtype in ('float64', 'float32', 'float16'):
            return ndarray._wrap(result, _dtype='float64')
        return ndarray._wrap(result, _dtype='int64')

    @staticmethod
    def matmul(a, b):
        """计算两个数组的矩阵乘积。"""
        from ..__init__ import ndarray
        if _use_numpy(a, b):
            import rsnumpy as _np
            return _from_np(_np.matmul(_to_np(a), _to_np(b)))
        result = _core.linalg.matmul(_ensure(a), _ensure(b))
        a_dtype = getattr(a, '_dtype', 'float64')
        b_dtype = getattr(b, '_dtype', 'float64')
        if a_dtype in ('float64', 'float32', 'float16') or b_dtype in ('float64', 'float32', 'float16'):
            return ndarray._wrap(result, _dtype='float64')
        return ndarray._wrap(result, _dtype='int64')

    @staticmethod
    def inv(a):
        """计算矩阵的逆。"""
        if _use_numpy(a):
            import rsnumpy as _np
            return _from_np(_np.linalg.inv(_to_np(a)))
        return _wrap(_core.linalg.inv(_ensure(a)))

    @staticmethod
    def det(a):
        """计算矩阵的行列式。"""
        result = _core.linalg.det(_ensure(a))
        return _round_float(result)

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
        if _use_numpy(a, b):
            import rsnumpy as _np
            return _from_np(_np.linalg.solve(_to_np(a), _to_np(b)))
        return _wrap(_core.linalg.solve(_ensure(a), _ensure(b)))

    @staticmethod
    def lstsq(a, b, rcond=None):
        """最小二乘解，委托 numpy（Rust 无原生实现）。"""
        import rsnumpy as _np
        x, res, rank, s = _np.linalg.lstsq(_to_np(a), _to_np(b), rcond=rcond)
        return (_from_np(x), _from_np(res), int(rank), _from_np(s))

    @staticmethod
    def matrix_rank(a, tol=None, hermitian=False):
        """矩阵秩，委托 numpy。"""
        import rsnumpy as _np
        return int(_np.linalg.matrix_rank(_to_np(a), tol=tol, hermitian=hermitian))

    @staticmethod
    def eig(a):
        """计算矩阵的特征值和特征向量。"""
        evals, evecs = _core.linalg.eig(_ensure(a))
        evals = _round_array(evals)
        evecs = _round_array(evecs)
        from ..__init__ import ndarray
        return (ndarray._wrap(evals), ndarray._wrap(evecs))

    @staticmethod
    def eigvals(a):
        """计算矩阵的特征值。"""
        return _wrap(_core.linalg.eigvals(_ensure(a)))

    @staticmethod
    def svd(a):
        """计算奇异值分解。"""
        u, s, vt = _core.linalg.svd(_ensure(a))
        u = _round_array(u)
        s = _round_array(s)
        vt = _round_array(vt)
        from ..__init__ import ndarray
        return (ndarray._wrap(u), ndarray._wrap(s), ndarray._wrap(vt))

    @staticmethod
    def qr(a):
        """计算 QR 分解。"""
        q, r = _core.linalg.qr(_ensure(a))
        q = _round_array(q)
        r = _round_array(r)
        from ..__init__ import ndarray
        return (ndarray._wrap(q), ndarray._wrap(r))

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
        result = _core.linalg.trace(_ensure(a))
        return _round_float(result)

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
        evals, evecs = _core.linalg.eigh(_ensure(a))
        evals = _round_array(evals)
        evecs = _round_array(evecs)
        from ..__init__ import ndarray
        return (ndarray._wrap(evals), ndarray._wrap(evecs))

    @staticmethod
    def solve_banded(lower, upper, ab, b):
        """求解带状线性方程组。"""
        return _wrap(_core.linalg.solve_banded(lower, upper, _ensure(ab), _ensure(b)))


# 将 linalg_module 的公有方法同时暴露为模块级函数，
# 兼容 numpy 风格的 `from rsnumpy.linalg import det, inv, norm, solve, ...`。
_this_module = _sys.modules[__name__]
for _name in dir(linalg_module):
    if not _name.startswith('_'):
        setattr(_this_module, _name, getattr(linalg_module, _name))
del _this_module, _name

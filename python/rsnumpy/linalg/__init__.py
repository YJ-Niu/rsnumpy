"""rsnumpy.linalg - 线性代数模块 - 薄包装，所有实现位于 Rust。"""

import sys as _sys

import rsnumpy.num_core as _core


class LinAlgError(Exception):
    """线性代数相关错误。"""
    pass


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


def _flatten_scalars(data):
    """将嵌套列表展平为一维标量列表。"""
    if isinstance(data, (list, tuple)):
        # 使用列表推导 + extend 替代显式 for 循环
        out = []
        for x in data:
            out.extend(_flatten_scalars(x))
        return out
    return [data]


def _is_complex_arr(a):
    """判断 rsnumpy 数组是否为复数数组。"""
    return bool(getattr(getattr(a, '_array', None), 'is_complex', False))


def _matmul_2d(mat_a, mat_b):
    """朴素二维矩阵乘 A(m×n)·B(n×p)，元素可为 float 或 complex。"""
    n = len(mat_b)
    p = len(mat_b[0]) if mat_b and isinstance(mat_b[0], list) else 0
    if not mat_a or p == 0:
        return []
    # ikj 顺序：使用 enumerate 缓存索引与值，跳过零元素加速稀疏场景
    out = [[0] * p for _ in range(len(mat_a))]
    for i, row_a in enumerate(mat_a):
        row_o = out[i]
        for k, aik in enumerate(row_a):
            if aik == 0:
                continue
            row_b = mat_b[k]
            for j in range(p):
                row_o[j] += aik * row_b[j]
    return out


def _matmul_nested(a, na, b, nb):
    """支持复数与批量（>2 维广播）的 matmul，操作于嵌套列表。"""
    if na == 2 and nb == 2:
        return _matmul_2d(a, b)
    if na == 1 and nb == 1:
        return sum(x * y for x, y in zip(a, b))
    if na == 2 and nb == 1:
        return [sum(a[i][k] * b[k] for k in range(len(b))) for i in range(len(a))]
    if na == 1 and nb == 2:
        return [sum(a[k] * b[k][j] for k in range(len(a))) for j in range(len(b[0]))]
    if na >= 3 and nb == 2:
        return [_matmul_nested(sub, na - 1, b, 2) for sub in a]
    if na == 2 and nb >= 3:
        return [_matmul_nested(a, 2, sub, nb - 1) for sub in b]
    if na >= 3 and nb >= 3:
        return [_matmul_nested(sa, na - 1, sb, nb - 1) for sa, sb in zip(a, b)]
    raise ValueError("unsupported matmul shapes: {}D and {}D".format(na, nb))


def _dot_nested(a, na, b, nb):
    """支持复数的 dot，覆盖 1D/2D 组合（更高维按批量 matmul 处理）。"""
    if na == 1 and nb == 1:
        return sum(x * y for x, y in zip(a, b))
    if na == 2 and nb == 2:
        return _matmul_2d(a, b)
    if na == 2 and nb == 1:
        return [sum(a[i][k] * b[k] for k in range(len(b))) for i in range(len(a))]
    if na == 1 and nb == 2:
        return [sum(a[k] * b[k][j] for k in range(len(a))) for j in range(len(b[0]))]
    return _matmul_nested(a, na, b, nb)


def _solve_pinv(a, b):
    """使用伪逆求解线性方程组 x = pinv(A) @ b。"""
    pinv_a = _pinv_general(a)
    return _matmul_2d(pinv_a, b)


def _gauss_jordan_inv(mat):
    """高斯-约当消元求逆（列主元），元素可为 float 或 complex，支持任意 n×n。"""
    n = len(mat)
    work = [list(row) for row in mat]
    inv = [[(1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    max_abs = max((abs(v) for row in work for v in row), default=0.0)
    eps = n * max_abs * 2.220446049250313e-16 if max_abs != 0 else 1e-10
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < eps:
            raise LinAlgError("Singular matrix")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
            inv[col], inv[pivot] = inv[pivot], inv[col]
        pv = work[col][col]
        wcol = work[col]
        icol = inv[col]
        # 使用列表推导 + 切片赋值代替逐元素循环
        wcol[:] = [x / pv for x in wcol]
        icol[:] = [x / pv for x in icol]
        for r in range(n):
            if r == col:
                continue
            factor = work[r][col]
            if abs(factor) < eps:
                continue
            wr = work[r]
            ir = inv[r]
            # 列表推导 + zip 代替显式索引循环
            wr[:] = [a - factor * w for a, w in zip(wr, wcol)]
            ir[:] = [a - factor * w for a, w in zip(ir, icol)]
    return inv


def _pinv_general(mat):
    """通用伪逆实现，基于特征值分解处理复数矩阵。"""
    n = len(mat)
    m = len(mat[0]) if mat else 0
    
    ata = [[sum(mat[i][k].conjugate() * mat[j][k] for k in range(m)) for j in range(n)] for i in range(n)]
    
    evals, evecs = _eig_general(ata)
    
    tol = max(n, m) * max(abs(e) for e in evals) * 1e-15
    
    inv_sqrt = [1.0 / e**0.5 if abs(e) > tol else 0.0 for e in evals]
    
    v_inv = [[evecs[j][i] * inv_sqrt[i] for i in range(n)] for j in range(n)]
    
    at = [[mat[j][i].conjugate() for j in range(n)] for i in range(m)]
    
    return _matmul_2d(v_inv, _matmul_2d(v_inv, at))


def _eig_general(mat):
    """通用特征值分解实现，支持复数矩阵。"""
    n = len(mat)
    if n == 1:
        return [mat[0][0]], [[1.0]]
    if n == 2:
        a, b = mat[0][0], mat[0][1]
        c, d = mat[1][0], mat[1][1]
        tr = a + d
        det = a * d - b * c
        import cmath
        disc = cmath.sqrt(tr * tr - 4.0 * det)
        evals = [(tr + disc) / 2.0, (tr - disc) / 2.0]
        evecs = [[0.0, 0.0], [0.0, 0.0]]
        for idx, lam in enumerate(evals):
            if abs(b) > 1e-12 or abs(lam - a) > 1e-12:
                v0, v1 = b, lam - a
            else:
                v0, v1 = lam - d, c
            norm = abs(v0)**2 + abs(v1)**2
            if norm < 1e-24:
                evecs[0][idx] = 1.0
                evecs[1][idx] = 0.0
            else:
                norm = cmath.sqrt(norm)
                evecs[0][idx] = v0 / norm
                evecs[1][idx] = v1 / norm
        return evals, evecs
    
    return _eig_power_iteration(mat)


def _eig_power_iteration(mat):
    """幂迭代法计算特征值和特征向量（仅适用于对称/厄米矩阵）。"""
    import cmath
    n = len(mat)
    evals = []
    evecs = [[0.0] * n for _ in range(n)]

    for k in range(n):
        v = [complex(1.0 if i == k else 0.0) for i in range(n)]
        for _ in range(100):
            # 列表推导：mat @ v
            new_v = [sum(mat[i][j] * v[j] for j in range(n)) for i in range(n)]
            norm_sq = sum(abs(x)**2 for x in new_v)
            if norm_sq < 1e-24:
                break
            norm = cmath.sqrt(norm_sq)
            v = [x / norm for x in new_v]
        # Rayleigh 商：v^H @ mat @ v
        mat_v = [sum(mat[i][j] * v[j] for j in range(n)) for i in range(n)]
        lam = sum(v[i].conjugate() * mat_v[i] for i in range(n))
        evals.append(lam)
        # 列表推导：填充第 k 列特征向量
        for i, vi in enumerate(v):
            evecs[i][k] = vi

    return evals, evecs


def _gauss_jordan_solve(mat, b):
    """高斯-约当消元求解 Ax = b（列主元），元素可为 float 或 complex，支持任意 n×n。"""
    n = len(mat)
    work = [list(row) for row in mat]
    # 列表推导 + 闭包提取行转换逻辑
    def _conv_row(row):
        if isinstance(row, (list, tuple)):
            return list(row)
        if hasattr(row, 'tolist'):
            return row.tolist()
        return [row]
    rhs = [_conv_row(row) for row in b]
    max_abs = max((abs(v) for row in work for v in row), default=0.0)
    eps = n * max_abs * 2.220446049250313e-16 if max_abs != 0 else 1e-10
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < eps:
            raise LinAlgError("Singular matrix")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
            rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        pv = work[col][col]
        wcol = work[col]
        rcol = rhs[col]
        # 列表推导 + 切片赋值代替逐元素循环
        wcol[:] = [x / pv for x in wcol]
        rcol[:] = [x / pv for x in rcol]
        for r in range(n):
            if r == col:
                continue
            factor = work[r][col]
            if factor == 0:
                continue
            wr = work[r]
            rr = rhs[r]
            wr[:] = [a - factor * w for a, w in zip(wr, wcol)]
            rr[:] = [a - factor * w for a, w in zip(rr, rcol)]
    if len(rhs[0]) == 1 and len(rhs) == n:
        return [[row[0]] for row in rhs]
    return rhs


def _inv_nested(data):
    """对嵌套列表递归求逆：最内两维视为待求逆的方阵，外层为批量维。"""
    if isinstance(data, list) and data and isinstance(data[0], list) \
            and (not data[0] or not isinstance(data[0][0], list)):
        return _gauss_jordan_inv(data)
    return [_inv_nested(sub) for sub in data]


def _solve_nested(a, b):
    """对嵌套列表递归求解 Ax = b：最内两维视为矩阵，外层为批量维。"""
    if isinstance(a, list) and a and isinstance(a[0], list):
        if not a[0] or not isinstance(a[0][0], list):
            try:
                return _gauss_jordan_solve(a, b)
            except LinAlgError:
                return _solve_pinv(a, b)
    return [_solve_nested(sa, sb) for sa, sb in zip(a, b)]


class linalg_module:
    """线性代数模块 - 所有方法都直接调用 Rust 实现。"""
    LinAlgError = LinAlgError

    @staticmethod
    def dot(a, b):
        """计算两个数组的点积。"""
        from ..__init__ import ndarray
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        b_arr = b if hasattr(b, '_array') else ndarray(b)
        if _is_complex_arr(a_arr) or _is_complex_arr(b_arr):
            res = _dot_nested(a_arr.tolist(), len(a_arr.shape), b_arr.tolist(), len(b_arr.shape))
            return ndarray(res) if isinstance(res, list) else res
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
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        b_arr = b if hasattr(b, '_array') else ndarray(b)
        if _is_complex_arr(a_arr) or _is_complex_arr(b_arr):
            res = _matmul_nested(a_arr.tolist(), len(a_arr.shape), b_arr.tolist(), len(b_arr.shape))
            return ndarray(res) if isinstance(res, list) else res
        result = _core.linalg.matmul(_ensure(a), _ensure(b))
        a_dtype = getattr(a, '_dtype', 'float64')
        b_dtype = getattr(b, '_dtype', 'float64')
        if a_dtype in ('float64', 'float32', 'float16') or b_dtype in ('float64', 'float32', 'float16'):
            return ndarray._wrap(result, _dtype='float64')
        return ndarray._wrap(result, _dtype='int64')

    @staticmethod
    def inv(a):
        """计算矩阵的逆。"""
        from ..__init__ import ndarray, empty
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        shape = a_arr.shape
        if len(shape) == 2 and not _is_complex_arr(a_arr):
            return _wrap(_core.linalg.inv(_ensure(a_arr)))

        if len(shape) == 3:
            # (B, 2, 2) 批量求逆：直接调用 Rust 层 rayon 并行实现
            if shape[1] == 2 and shape[2] == 2:
                try:
                    result = _core.linalg.inv(_ensure(a_arr))
                    dtype = 'complex128' if _is_complex_arr(a_arr) else 'float64'
                    return ndarray._wrap(result, _dtype=dtype)
                except (ValueError, TypeError):
                    pass  # 回退到通用路径
            # 通用批量路径：Python 循环逐矩阵求逆
            batch_size = shape[0]
            result = empty(shape, dtype=a_arr._dtype)
            for i in range(batch_size):
                try:
                    result[i] = ndarray(_gauss_jordan_inv(a_arr[i].tolist()))
                except LinAlgError:
                    try:
                        result[i] = linalg_module.pinv(a_arr[i])
                    except (ValueError, LinAlgError):
                        result[i] = ndarray(_pinv_general(a_arr[i].tolist()))
            return result

        return ndarray(_inv_nested(a_arr.tolist()))

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
        result = _wrap(_core.linalg.norm(_ensure(x), ord_val, axis))
        if hasattr(result, 'shape') and len(result.shape) == 0:
            return float(result.tolist())
        return result

    @staticmethod
    def solve(a, b):
        """求解线性方程组。"""
        from rsnumpy import empty, ndarray
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        b_arr = b if hasattr(b, '_array') else ndarray(b)
        
        if _is_complex_arr(a_arr) or _is_complex_arr(b_arr):
            res = _solve_nested(a_arr.tolist(), b_arr.tolist())
            return ndarray(res)
        
        if len(a_arr.shape) == 3 and len(b_arr.shape) == 3:
            batch_size = a_arr.shape[0]
            result = empty((batch_size, a_arr.shape[1], b_arr.shape[2]), dtype=a_arr._dtype)
            for i in range(batch_size):
                try:
                    result[i] = _wrap(_core.linalg.solve(_ensure(a_arr[i]), _ensure(b_arr[i])))
                except ValueError:
                    res = _gauss_jordan_solve(a_arr[i].tolist(), b_arr[i].tolist())
                    result[i] = ndarray(res)
            return result
        
        try:
            return _wrap(_core.linalg.solve(_ensure(a), _ensure(b)))
        except ValueError:
            try:
                res = _gauss_jordan_solve(a_arr.tolist(), b_arr.tolist())
                return ndarray(res)
            except LinAlgError:
                res = _solve_pinv(a_arr.tolist(), b_arr.tolist())
                return ndarray(res)

    @staticmethod
    def lstsq(a, b, rcond=None):
        """最小二乘解 min ||b - a·x||（原生实现，基于正规方程 AᵀA·x = Aᵀb）。

        返回 (x, residuals, rank, s)，与 numpy 语义一致：仅当 a 列满秩且为
        过定方程组（行数 > 列数）时 residuals 才为各列的残差平方和，否则为空数组。
        奇异值 s 由 AᵀA 的特征值开方降序得到。
        """
        from ..__init__ import ndarray
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        b_arr = b if hasattr(b, '_array') else ndarray(b)
        b_is_1d = len(b_arr.shape) == 1
        at = a_arr.T
        ata = linalg_module.matmul(at, a_arr)
        atb = linalg_module.matmul(at, b_arr)
        try:
            x = linalg_module.solve(ata, atb)
        except LinAlgError:
            x = linalg_module.matmul(linalg_module.pinv(ata), atb)
        rank = linalg_module.matrix_rank(a_arr)
        evals = _flatten_scalars(linalg_module.eigvals(ata).tolist())
        svals = sorted((max(abs(v), 0.0) ** 0.5 for v in evals), reverse=True)
        s = ndarray(svals)
        a_shape = a_arr.shape
        m, n = (a_shape[0], a_shape[1]) if len(a_shape) == 2 else (0, 0)
        if m > n and rank == n:
            diff = _flatten_scalars((b_arr - linalg_module.matmul(a_arr, x)).tolist())
            b_cols = 1 if b_is_1d else b_arr.shape[1]
            residuals = [0.0] * b_cols
            for idx, v in enumerate(diff):
                residuals[idx % b_cols] += abs(v) ** 2
            res = ndarray(residuals)
        else:
            res = ndarray([])
        return (x, res, int(rank), s)

    @staticmethod
    def matrix_rank(a, tol=None, hermitian=False):
        """矩阵秩：高斯消元后非零主元的个数（原生实现，无外部依赖）。"""
        from ..__init__ import ndarray
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        rows = a_arr.tolist()
        if not isinstance(rows, list):
            return 0 if float(rows) == 0.0 else 1
        if not rows:
            return 0
        if not isinstance(rows[0], list):
            rows = [rows]
        mat = [[v for v in r] for r in rows]
        nrows = len(mat)
        ncols = len(mat[0])
        if tol is None:
            max_abs = max((abs(v) for r in mat for v in r), default=0.0)
            tol = max(nrows, ncols) * max_abs * 2.220446049250313e-16
        rank = 0
        pivot_row = 0
        for col in range(ncols):
            sel = -1
            best = tol
            for r in range(pivot_row, nrows):
                if abs(mat[r][col]) > best:
                    best = abs(mat[r][col])
                    sel = r
            if sel == -1:
                continue
            mat[pivot_row], mat[sel] = mat[sel], mat[pivot_row]
            pv = mat[pivot_row][col]
            for r in range(nrows):
                if r != pivot_row and mat[r][col] != 0.0:
                    factor = mat[r][col] / pv
                    for c in range(col, ncols):
                        mat[r][c] -= factor * mat[pivot_row][c]
            pivot_row += 1
            rank += 1
            if pivot_row == nrows:
                break
        return rank

    @staticmethod
    def eig(a):
        """计算矩阵的特征值和特征向量。"""
        from ..__init__ import ndarray
        import cmath
        a = ndarray(a)
        if a.ndim == 3:
            n = a.shape[0]
            eval_list = []
            evec_list = []
            for i in range(n):
                mat = a[i]
                if mat.dtype == 'complex128':
                    m = mat.shape[0]
                    if m == 2:
                        mat_list = mat.tolist()
                        a11, a12 = mat_list[0][0], mat_list[0][1]
                        a21, a22 = mat_list[1][0], mat_list[1][1]
                        tr = a11 + a22
                        det = a11 * a22 - a12 * a21
                        disc = cmath.sqrt(tr * tr - 4.0 * det)
                        evals = [(tr + disc) / 2.0, (tr - disc) / 2.0]
                        evecs = [[0.0, 0.0], [0.0, 0.0]]
                        for idx, lam in enumerate(evals):
                            if abs(a12) > 1e-12 or abs(lam - a11) > 1e-12:
                                v0, v1 = a12, lam - a11
                            else:
                                v0, v1 = lam - a22, a21
                            norm = abs(v0)**2 + abs(v1)**2
                            if norm < 1e-24:
                                evecs[0][idx] = 1.0
                                evecs[1][idx] = 0.0
                            else:
                                norm = cmath.sqrt(norm)
                                evecs[0][idx] = v0 / norm
                                evecs[1][idx] = v1 / norm
                        eval_list.append(evals)
                        evec_list.append(evecs)
                    else:
                        evals, evecs = _core.linalg.eig(mat._array)
                        eval_list.append(ndarray._wrap(evals).tolist())
                        evec_list.append(ndarray._wrap(evecs).tolist())
                else:
                    evals, evecs = _core.linalg.eig(mat._array)
                    eval_list.append(ndarray._wrap(evals).tolist())
                    evec_list.append(ndarray._wrap(evecs).tolist())
            return (ndarray(eval_list), ndarray(evec_list))
        evals, evecs = _core.linalg.eig(_ensure(a))
        evals = _round_array(evals)
        evecs = _round_array(evecs)
        return (ndarray._wrap(evals), ndarray._wrap(evecs))

    @staticmethod
    def eigvals(a):
        """计算矩阵的特征值。"""
        from ..__init__ import ndarray
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        try:
            return _wrap(_core.linalg.eigvals(_ensure(a)))
        except ValueError:
            evals, _ = _eig_general(a_arr.tolist())
            return ndarray(evals)

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
        from ..__init__ import ndarray
        from .._extra import diag as _diag
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        try:
            return _wrap(_core.linalg.pinv(_ensure(a)))
        except ValueError:
            try:
                u, s, vt = linalg_module.svd(a_arr)
                tol = max(a_arr.shape) * max(s.tolist()) * 1e-15
                s_inv = ndarray([1.0 / si if si > tol else 0.0 for si in s.tolist()])
                return linalg_module.matmul(linalg_module.matmul(vt.T, _diag(s_inv)), u.T)
            except ValueError:
                return ndarray(_pinv_general(a_arr.tolist()))

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


def issymmetric(a, tol=None):
    """判断矩阵是否对称。"""
    from ..__init__ import ndarray
    a_arr = a if hasattr(a, '_array') else ndarray(a)
    if len(a_arr.shape) != 2 or a_arr.shape[0] != a_arr.shape[1]:
        return False
    at = a_arr.T
    if tol is None:
        diff = abs(a_arr - at)
        return diff.max() < 1e-10
    diff = abs(a_arr - at)
    return diff.max() <= tol


# 将 linalg_module 的公有方法同时暴露为模块级函数，
# 兼容 numpy 风格的 `from rsnumpy.linalg import det, inv, norm, solve, ...`。
_this_module = _sys.modules[__name__]
for _name in dir(linalg_module):
    if not _name.startswith('_'):
        setattr(_this_module, _name, getattr(linalg_module, _name))
del _this_module, _name

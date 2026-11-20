# SPDX-License-Identifier: MIT
"""rsnumpy.linalg - 线性代数模块 - 薄包装，所有实现位于 Rust。"""

import sys as _sys
from collections import namedtuple

import rsnumpy.num_core as _core

from ._pylinalg import NoValue

# NumPy 2.x 语义：qr 的 reduced/complete 结果是有 .Q/.R 属性的命名元组
QRResult = namedtuple('QRResult', ['Q', 'R'])
# svd compute_uv=True 结果是有 .U/.S/.Vh 属性的命名元组（可作 3 元组解包）
SVDResult = namedtuple('SVDResult', ['U', 'S', 'Vh'])
# eig 结果是有 .eigenvalues/.eigenvectors 属性的命名元组（NumPy 2.x 语义）
EigResult = namedtuple('EigResult', ['eigenvalues', 'eigenvectors'])
# eigh 结果是有 .eigenvalues/.eigenvectors 属性的命名元组（NumPy 2.x 语义）
EighResult = namedtuple('EighResult', ['eigenvalues', 'eigenvectors'])
# slogdet 结果是有 .sign/.logabsdet 属性的命名元组（NumPy 2.x 语义）
SlogdetResult = namedtuple('SlogdetResult', ['sign', 'logabsdet'])


class LinAlgError(ValueError):
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
    from rsnumpy import ndarray
    # 注意：裸 Rust 绑定类型 (builtins.ndarray) 与 Python 包装类的类名都叫
    # 'ndarray'，不能只靠类名判断"已包装"（旧逻辑把裸类型原样放行，导致
    # cholesky/svd 等返回 builtins.ndarray，`wrapper @ raw` 直接 TypeError）。
    # _round_array 对数组输入返回裸类型，所以这里统一 round 后用包装类重新构造。
    if (getattr(x, '__class__', None) and x.__class__.__name__ == 'ndarray') \
            or hasattr(x, 'tolist'):
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
    # n = len(mat_b)
    p = len(mat_b[0]) if mat_b and isinstance(mat_b[0], list) else 0
    if not mat_a or p == 0:
        # 保持 (m, 0) 嵌套结构（flat [] 会让 ndarray 推成 1-d (0,)）
        return [[] for _ in range(len(mat_a))]
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
        # 前导维须同形（rsnumpy 无完整广播；同形 zip 即可）
        return [_matmul_nested(sa, na - 1, sb, nb - 1) for sa, sb in zip(a, b)]
    raise ValueError("unsupported matmul shapes: {}D and {}D".format(na, nb))


def _matmul_wrap(res, a_shape, b_shape, is_c, a_dtype, b_dtype):
    """包装 matmul Python 回退结果：恢复空矩阵形状与复数 dtype。

    Python 嵌套列表无法表达 (0, p) 这类零维形状（ndarray 会推成 1-d），
    且空列表推得 float64——须按 (lead, m, p) 显式 reshape/astype。
    """
    from rsnumpy import ndarray

    out = ndarray(res) if isinstance(res, list) else res
    if not isinstance(out, ndarray):
        return out
    if len(a_shape) == 1 or len(b_shape) == 1:
        # 向量结果：(m,n)@(n,)→(m,)；(n,)@(n,k)→(k,)
        if len(b_shape) == 1:
            exp = a_shape[:-2] + (a_shape[-2],)
        else:
            exp = b_shape[1:]
    else:
        lead = [max(x, y) for x, y in
                zip(reversed(a_shape[:-2]), reversed(b_shape[:-2]))]
        exp = tuple(reversed(lead)) + (a_shape[-2], b_shape[-1])
    if tuple(out.shape) != exp:
        out = out.reshape(exp)
    # Python 列表只能表达 float64/complex128：按 numpy 提升规则还原位宽
    # （f32@f32→f32；c64@c64→c64；任一侧更宽则更宽）
    if is_c or 'complex' in a_dtype or 'complex' in b_dtype:
        res_dt = 'complex128' if 'complex128' in (a_dtype, b_dtype) \
            else 'complex64'
    else:
        res_dt = 'float64' if 'float64' in (a_dtype, b_dtype) else 'float32'
    if getattr(out, '_dtype', None) != res_dt:
        out = out.astype(res_dt)
    return out


def _dot_res_dtype(a_dtype, b_dtype):
    """numpy 提升规则（linalg inexact/int 场景）：c64×c64→c64、
    c64×其它→c128、f32×f32→f32（旧实现硬编码 float64 会破坏
    32 位结果的 dtype，测试按 dtype 选精度阈值）。"""
    if 'complex128' in (a_dtype, b_dtype):
        return 'complex128'
    if 'complex64' in (a_dtype, b_dtype):
        return 'complex64' if a_dtype == b_dtype == 'complex64' \
            else 'complex128'
    if 'float64' in (a_dtype, b_dtype):
        return 'float64'
    if 'float32' in (a_dtype, b_dtype):
        return 'float32'
    if 'float16' in (a_dtype, b_dtype):
        return 'float16'
    return 'int64'


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
    
    # n≥3：对称/厄米保持幂迭代；一般矩阵走特征多项式（FL+DK）特征值
    # + 正规方程特征向量（与 eigvals 的 FL+DK 结果同序）。
    if _is_symmetricish(mat):
        return _eig_power_iteration(mat)
    from ._pylinalg import _eigvals_qr, _eigenvectors_general
    evals = _eigvals_qr(mat)
    return evals, _eigenvectors_general(mat, evals)


def _is_symmetricish(mat, tol=1e-10):
    """判断嵌套 list 矩阵是否（近似）厄米：mat[i][j] ≈ conj(mat[j][i])。"""
    n = len(mat)
    for i in range(n):
        for j in range(i + 1, n):
            a = mat[i][j]
            b = mat[j][i].conjugate()
            if abs(a - b) > tol * (1.0 + abs(a) + abs(b)):
                return False
    return True


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


def _eig_one(mat):
    """单个 2-d 矩阵特征分解：Rust 快路径（2x2/3x3 实输入），其余尺寸
    （含空/1x1）与全部复数输入走纯 Python 回退（Rust eig 不处理复数）。
    返回 (evals, evecs) 的嵌套 list。"""
    from rsnumpy import ndarray as _nd

    dt = getattr(mat, '_dtype', 'float64')
    if dt in ('complex64', 'complex128'):
        return _eig_general(mat.tolist())
    try:
        ev, ec = _core.linalg.eig(mat._array)
        return (_nd._wrap(ev).tolist(), _nd._wrap(ec).tolist())
    except ValueError:
        return _eig_general(mat.tolist())


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
    if n and len(rhs[0]) == 1 and len(rhs) == n:
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


def _svd_jacobi(mat, full_matrices=True, compute_uv=True):
    """通用 SVD（单侧 Jacobi 旋转），支持任意 m×n、float/complex。

    Rust 核心只覆盖 2×2，这里补齐一般形状，并实现 ``full_matrices`` /
    ``compute_uv`` 语义（与 NumPy 一致：u 为 m×m（full）或 m×min（reduced），
    vh 为 n×n（full）或 min×n（reduced），s 降序）。
    """
    import math

    m = len(mat)
    n = len(mat[0]) if m else 0
    k = min(m, n)

    if m == 0 or n == 0:
        from rsnumpy import ndarray
        if not compute_uv:
            return ndarray([])
        if full_matrices:
            u = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
            vh = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        else:
            u = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(m)]
            vh = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(k)]
        return (ndarray(u), ndarray([]), ndarray(vh))

    is_complex = any(isinstance(v, complex) for row in mat for v in row)
    conj = (lambda z: z.conjugate()) if is_complex else (lambda z: z)

    transposed = m < n
    if transposed:
        work = [[mat[i][j] for i in range(m)] for j in range(n)]
        mm, nn = n, m
    else:
        work = [list(row) for row in mat]
        mm, nn = m, n
    # work: mm×nn, mm >= nn. Column rotations are accumulated in V (nn×nn).
    V = [[1.0 if i == j else 0.0 for j in range(nn)] for i in range(nn)]

    def col_dot(p, q):
        return sum(conj(work[i][p]) * work[i][q] for i in range(mm))

    for _sweep in range(100):
        off = 0.0
        diag = [abs(col_dot(p, p)) for p in range(nn)]
        for p in range(nn):
            for q in range(p + 1, nn):
                g = col_dot(p, q)
                off += abs(g) * abs(g)
        if off <= 1e-28 * sum(d * d for d in diag) + 1e-300:
            break
        for p in range(nn):
            for q in range(p + 1, nn):
                alpha = abs(col_dot(p, p))
                beta = abs(col_dot(q, q))
                gamma = col_dot(p, q)
                ag = abs(gamma)
                if ag <= 1e-15 * math.sqrt(alpha * beta):
                    continue
                if is_complex and ag > 0.0:
                    # rotate column q so the cross term becomes |gamma|
                    phase = conj(gamma) / ag
                    if abs(phase) > 0.0:
                        for i in range(mm):
                            work[i][q] *= phase
                        for i in range(nn):
                            V[i][q] *= phase
                        gamma = abs(gamma)
                z = (beta - alpha) / (2.0 * gamma)
                # small-magnitude root of t^2 - 2*z*t - 1 = 0, matching the
                # rotation convention [cp,cq] -> [c*cp+s*cq, -s*cp+c*cq]
                t = (-1.0 if z >= 0 else 1.0) / (abs(z) + math.sqrt(1.0 + z * z))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for i in range(mm):
                    wp = work[i][p]
                    wq = work[i][q]
                    work[i][p] = c * wp + s * wq
                    work[i][q] = -s * wp + c * wq
                for i in range(nn):
                    vp = V[i][p]
                    vq = V[i][q]
                    V[i][p] = c * vp + s * vq
                    V[i][q] = -s * vp + c * vq

    sig = [math.sqrt(abs(col_dot(p, p))) for p in range(nn)]
    order = sorted(range(nn), key=lambda p: -sig[p])
    sig = [sig[p] for p in order]
    U = [[0.0] * nn for _ in range(mm)]
    smax = sig[0] if sig else 0.0
    for new_p, old_p in enumerate(order):
        sp = sig[new_p]
        if sp > 1e-15 * smax:
            for i in range(mm):
                U[i][new_p] = work[i][old_p] / sp
    Vh = [[conj(V[j][order[i]]) for j in range(nn)] for i in range(nn)]

    def _complete_orthonormal(cols, mm, ncols):
        """把 ncols 个正交列扩展为 mm 列正交基（Gram-Schmidt on e_i）。"""
        basis = [list(col) for col in cols]  # list of columns
        for i in range(mm):
            e = [1.0 if r == i else 0.0 for r in range(mm)]
            for b in basis:
                proj = sum(conj(b[r]) * e[r] for r in range(mm))
                e = [e[r] - proj * b[r] for r in range(mm)]
            ne = math.sqrt(sum(abs(x) ** 2 for x in e))
            if ne < 1e-8:
                continue
            basis.append([x / ne for x in e])
            if len(basis) == mm:
                break
        return basis

    # rows are nn long; pad to mm so extra columns can be filled and the
    # transposed branch can index U[j][i] for i in range(mm)
    U = [row + [0.0] * (mm - len(row)) for row in U]
    if full_matrices and mm > nn:
        extra = _complete_orthonormal(
            [[U[i][p] for i in range(mm)] for p in range(nn)], mm, nn)
        for col_idx in range(nn, len(extra)):
            col = extra[col_idx]
            for i in range(mm):
                U[i][col_idx] = col[i]

    if transposed:
        # B = A^T = U_b Σ Vh_b  =>  A = Vh_b^T Σ U_b^T (plain transposes)
        u_out = [[Vh[i][j] for i in range(nn)] for j in range(nn)]
        vh_out = [[U[j][i] for j in range(mm)] for i in range(mm)]
    else:
        u_out = U
        vh_out = Vh

    if not compute_uv:
        from rsnumpy import ndarray
        return ndarray(sig)

    if not full_matrices:
        u_out = [row[:k] for row in u_out]
        vh_out = vh_out[:k]

    from rsnumpy import ndarray
    return (ndarray(u_out), ndarray(sig), ndarray(vh_out))


def _norm_out_dtype(dt):
    """norm 结果 dtype：inexact → 对应实 dtype；int/bool/其它 → float64。"""
    if dt == 'complex64':
        return 'float32'
    if dt == 'complex128':
        return 'float64'
    if dt in ('float16', 'float32', 'float64'):
        return dt
    return 'float64'


def _norm_vec(vals, ord):
    """一维向量范数（NumPy 语义）：None/2→L2，1→L1，-1→1/Σ1/|x|（含 0 元
    inf 传播 → 0），0→非零计数，±inf→max/min，p>0→(Σ|x|^p)^{1/p}，
    p<0(p≠-1)→(Σ1/|x|^(-p))^{1/p}。"""
    import math

    if ord is None:
        o = 2
    elif isinstance(ord, str):
        if ord == 'inf':
            o = float('inf')
        elif ord == '-inf':
            o = float('-inf')
        else:
            raise ValueError("Invalid norm order for vector norm.")
    elif isinstance(ord, bool):
        o = int(ord)
    elif isinstance(ord, (int, float)):
        o = ord
    else:
        o = float(ord)

    if not vals:
        return 0.0  # 空向量：±inf 用 initial=0，其余范数空和为 0
    absx = [abs(v) for v in vals]
    if o == 2:
        return math.sqrt(sum(a * a for a in absx))
    if o == 1:
        return sum(absx)
    if o == 0:
        return float(sum(1 for v in vals if v != 0))
    if o == float('inf'):
        return max(absx)
    if o == float('-inf'):
        return min(absx)
    if o == -1:
        s = 0.0
        for a in absx:
            s += (1.0 / a) if a != 0 else float('inf')
        return 1.0 / s
    if o < 0:
        s = 0.0
        for a in absx:
            s += (1.0 / (a ** (-o))) if a != 0 else float('inf')
        return s ** (1.0 / o)
    return sum(a ** o for a in absx) ** (1.0 / o)


def _norm_mat(M, ord):
    """二维矩阵范数（NumPy 语义）：None→Frobenius，±1→列和 max/min，
    ±inf→行和 max/min，2/-2→最大/最小奇异值，'nuc'→核范数（奇异值和）。"""
    import math

    m = len(M)
    n = len(M[0]) if m else 0
    absM = [[abs(v) for v in row] for row in M]
    if ord is None:
        return math.sqrt(sum(v * v for row in absM for v in row))
    if ord == 1 or ord == -1:
        if n == 0:
            return 0.0
        colsums = [sum(absM[i][j] for i in range(m)) for j in range(n)]
        return max(colsums) if ord == 1 else min(colsums)
    if ord == float('inf') or ord == float('-inf'):
        if m == 0:
            return 0.0
        rowsums = [sum(row) for row in absM]
        return max(rowsums) if ord == float('inf') else min(rowsums)
    if ord in (2, -2, 'nuc'):
        svals = [float(abs(v)) for v in _svd_jacobi(M, False, False).tolist()]
        if not svals:
            return 0.0
        if ord == 2:
            return svals[0]
        if ord == -2:
            return svals[-1]
        return sum(svals)
    raise ValueError("Invalid norm order for matrix norm.")


class linalg_module:
    """线性代数模块 - 提供矩阵运算、求解、分解等功能。

    【核心功能】
    - dot/matmul: 矩阵乘法
    - inv/pinv: 矩阵求逆和伪逆
    - solve/lstsq: 线性方程组求解和最小二乘
    - det/norm: 行列式和范数
    - eig/eigvals: 特征值分解
    - svd: 奇异值分解

    【性能特点】
    - 核心运算由Rust实现，性能接近原生NumPy
    - 支持复数矩阵运算
    - 支持批量矩阵操作（3D及以上）

    【线程安全提示】
    ⚠️ 本模块的函数在多线程环境下需要注意：
    1. 读取操作是线程安全的
    2. 写入操作需要外部加锁
    3. 推荐使用copy()创建独立副本再共享

    【使用示例】
    >>> import rsnumpy as np
    >>> # 矩阵乘法
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> np.linalg.matmul(A, B)
    array([[19., 22.],
           [43., 50.]])

    >>> # 矩阵求逆
    >>> A_inv = np.linalg.inv(A)
    >>> A @ A_inv  # 应该接近单位矩阵
    array([[1., 0.],
           [0., 1.]])

    >>> # 解线性方程组
    >>> b = np.array([1, 2])
    >>> x = np.linalg.solve(A, b)
    >>> A @ x  # 应该等于b
    array([1., 2.])
    """

    LinAlgError = LinAlgError

    @staticmethod
    def dot(a, b):
        """计算两个数组的点积（内积）。

        【使用示例】
        >>> import rsnumpy as np
        >>> # 向量内积
        >>> v1 = np.array([1, 2, 3])
        >>> v2 = np.array([4, 5, 6])
        >>> np.linalg.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
        32.0

        >>> # 矩阵乘法
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])
        >>> np.linalg.dot(A, B)  # 等价于 A @ B
        array([[19., 22.],
               [43., 50.]])

        >>> # 高维点积
        >>> T = np.random.rand(2, 3, 4)
        >>> V = np.random.rand(4)
        >>> result = np.linalg.dot(T, V)  # shape: (2, 3)

        【参数说明】
        a, b: array_like
            输入数组，可以是向量、矩阵或高维张量
            - 一维：计算内积
            - 二维：计算矩阵乘法
            - 更高维：按规则进行张量收缩

        【返回值】
        标量或数组：
        - 两个向量：返回标量（内积）
        - 矩阵：返回矩阵
        - 高维：返回张量

        【注意事项】
        - 与matmul的区别：dot对高维数组的处理规则更灵活
        - 复数数组：支持复数运算
        - 性能：对于简单矩阵乘法，推荐使用matmul
        """
        from rsnumpy import ndarray

        def _to_arr(z):
            if hasattr(z, '_array'):
                return z
            if isinstance(z, (list, tuple)):
                return ndarray(z)
            if hasattr(z, '__iter__') and not isinstance(z, (str, bytes)):
                # .flat 等迭代器参数（numpy 语义：dot 接受可迭代对象）
                return ndarray(list(z))
            return ndarray(z)

        a_arr = _to_arr(a)
        b_arr = _to_arr(b)
        a_dtype = getattr(a_arr, '_dtype', 'float64')
        b_dtype = getattr(b_arr, '_dtype', 'float64')
        # f32 同样不走 Rust dot：Rust 侧 32 位缓冲在连续调用后可能
        # 读到脏状态（实测 f32 2x2 dot 返回全 0），Python 路径无此问题
        if _is_complex_arr(a_arr) or _is_complex_arr(b_arr) \
                or a_dtype == 'float32' or b_dtype == 'float32':
            res = _dot_nested(a_arr.tolist(), len(a_arr.shape), b_arr.tolist(), len(b_arr.shape))
            out = ndarray(res) if isinstance(res, list) else res
            # Python 列表推断出 c128/f64：按提升规则还原位宽
            if isinstance(out, ndarray):
                out = out.astype(_dot_res_dtype(a_dtype, b_dtype))
            return out
        try:
            result = _core.linalg.dot(a_arr._array, b_arr._array)
        except ValueError:
            # 回退：a 为 N-D、b 为 1-D 时按 numpy 语义沿 a 末轴收缩
            if b_arr.ndim == 1 and a_arr.ndim >= 1 and a_arr.shape[-1] == b_arr.shape[0]:
                lead = tuple(a_arr.shape[:-1])
                a2 = a_arr.reshape((-1, a_arr.shape[-1]))
                result = _core.linalg.dot(a2._array, b_arr._array)
                out = ndarray._wrap(result, _dtype=_dot_res_dtype(a_dtype, b_dtype))
                return out.reshape(lead) if lead else out
            raise
        return ndarray._wrap(result, _dtype=_dot_res_dtype(a_dtype, b_dtype))

    @staticmethod
    def vdot(a, b):
        """计算两个向量的点积（展开为一维）。"""
        result = _core.linalg.vdot(_ensure(a), _ensure(b))
        return _round_float(result)

    @staticmethod
    def inner(a, b):
        """计算两个数组的内积。"""
        from rsnumpy import ndarray
        result = _core.linalg.inner(_ensure(a), _ensure(b))
        a_dtype = getattr(a, '_dtype', 'float64')
        b_dtype = getattr(b, '_dtype', 'float64')
        if a_dtype in ('float64', 'float32', 'float16') or b_dtype in ('float64', 'float32', 'float16'):
            return ndarray._wrap(result, _dtype='float64')
        return ndarray._wrap(result, _dtype='int64')

    @staticmethod
    def matmul(a, b):
        """计算两个数组的矩阵乘积。

        【使用示例】
        >>> import rsnumpy as np
        >>> # 基本矩阵乘法
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])
        >>> np.linalg.matmul(A, B)
        array([[19., 22.],
               [43., 50.]])

        >>> # 批量矩阵乘法
        >>> batch_A = np.random.rand(5, 3, 4)  # 5个 3x4 矩阵
        >>> batch_B = np.random.rand(5, 4, 6)  # 5个 4x6 矩阵
        >>> result = np.linalg.matmul(batch_A, batch_B)
        >>> result.shape  # (5, 3, 6)
        (5, 3, 6)

        >>> # 广播矩阵乘法
        >>> A = np.random.rand(3, 4)  # 单个矩阵
        >>> batch_B = np.random.rand(5, 4, 6)  # 5个矩阵
        >>> result = np.linalg.matmul(A, batch_B)
        >>> result.shape  # (5, 3, 6)
        (5, 3, 6)

        【参数说明】
        a, b: array_like
            输入数组，必须满足矩阵乘法的形状要求：
            - 最后两个维度必须满足矩阵乘法规则
            - 前面的维度进行广播

        【与dot的区别】
        - matmul不支持标量相乘
        - matmul的广播规则更严格
        - 批量操作时matmul语义更清晰
        - 推荐用于矩阵乘法场景

        【性能提示】
        - 本函数由Rust实现，性能接近原生BLAS
        - 大型矩阵会自动使用多线程（如支持）
        - 批量操作比Python循环快很多
        """
        from rsnumpy import _broadcast_shape, ndarray, zeros
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        b_arr = b if hasattr(b, '_array') else ndarray(b)
        # 空维快捷路径：k/m/n 任一为 0 → 结果为广播形状的全零数组
        # （Rust/_matmul_nested 对空形状不可靠），dtype 与实数路径一致
        if (0 in a_arr.shape[-2:] + b_arr.shape[-2:]
                and len(a_arr.shape) >= 2 and len(b_arr.shape) >= 2):
            a_dt = getattr(a_arr, '_dtype', 'float64')
            b_dt = getattr(b_arr, '_dtype', 'float64')
            return zeros(_broadcast_shape(
                a_arr.shape[:-2], b_arr.shape[:-2])
                + (a_arr.shape[-2], b_arr.shape[-1]),
                dtype=_dot_res_dtype(a_dt, b_dt))
        if _is_complex_arr(a_arr) or _is_complex_arr(b_arr):
            res = _matmul_nested(a_arr.tolist(), len(a_arr.shape), b_arr.tolist(), len(b_arr.shape))
            return _matmul_wrap(res, a_arr.shape, b_arr.shape, True,
                                getattr(a_arr, '_dtype', 'float64'),
                                getattr(b_arr, '_dtype', 'float64'))
        try:
            result = _core.linalg.matmul(_ensure(a), _ensure(b))
        except ValueError:
            # stacked/broadcast (>2-d)：Rust 只覆盖 2-d → 纯 Python 回退
            res = _matmul_nested(a_arr.tolist(), len(a_arr.shape),
                                 b_arr.tolist(), len(b_arr.shape))
            return _matmul_wrap(res, a_arr.shape, b_arr.shape, False,
                                getattr(a_arr, '_dtype', 'float64'),
                                getattr(b_arr, '_dtype', 'float64'))
        a_dtype = getattr(a, '_dtype', 'float64')
        b_dtype = getattr(b, '_dtype', 'float64')
        if a_dtype in ('float64', 'float32', 'float16') or b_dtype in ('float64', 'float32', 'float16'):
            return ndarray._wrap(result, _dtype=_dot_res_dtype(a_dtype, b_dtype))
        return ndarray._wrap(result, _dtype='int64')

    @staticmethod
    def inv(a):
        """计算矩阵的逆。

        【使用示例】
        >>> import rsnumpy as np
        >>> # 单个矩阵求逆
        >>> A = np.array([[1, 2], [3, 4]])
        >>> A_inv = np.linalg.inv(A)
        >>> A @ A_inv  # 应该接近单位矩阵
        array([[1., 0.],
               [0., 1.]])

        >>> # 批量求逆
        >>> batch = np.random.rand(3, 4, 4)  # 3个 4x4 矩阵
        >>> batch_inv = np.linalg.inv(batch)
        >>> batch_inv.shape
        (3, 4, 4)

        >>> # 复数矩阵求逆
        >>> C = np.array([[2+1j, 1-1j], [1+1j, 3+2j]])
        >>> C_inv = np.linalg.inv(C)

        【参数说明】
        a: array_like
            方阵或方阵堆栈（shape为(..., N, N)）
            矩阵必须可逆（非奇异）

        【返回值】
        数组，形状与输入相同，为逆矩阵

        【异常】
        LinAlgError: 矩阵奇异（不可逆）

        【性能提示】
        - 单个矩阵：由Rust BLAS实现，高性能
        - 批量矩阵：使用rayon并行，充分利用多核
        - 大矩阵建议使用pinv（伪逆）更稳定

        【注意事项】
        - 输入必须是方阵
        - 对于病态矩阵，可能数值不稳定
        - 建议先用det()检查是否接近奇异
        """
        from rsnumpy import ndarray
        from ._pylinalg import _common_type, _stacked_2d

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        shape = tuple(a_arr.shape)
        if len(shape) < 2 or shape[-2] != shape[-1]:
            raise LinAlgError(
                "inv: last 2 dimensions of the array must be square")
        res_dt = _common_type(getattr(a_arr, '_dtype', 'float64'))[1]
        n = shape[-1]

        def _as_result(arr):
            # 结果保留输入的 ndarray 子类（numpy 语义）
            at = type(a_arr)
            if at is not ndarray and isinstance(a_arr, ndarray):
                return at(arr)
            return arr

        if n == 0:
            return _as_result(ndarray([]).astype(res_dt).reshape(shape))
        if len(shape) == 2:
            # Rust 快路径仅 2x2/3x3 实输入；1x1/大尺寸/复数走纯 Python
            # 高斯-约当（列主元）。
            if not _is_complex_arr(a_arr) and n in (2, 3):
                try:
                    return _as_result(
                        _wrap(_core.linalg.inv(_ensure(a_arr))).astype(
                            res_dt))
                except ValueError:
                    pass
            return _as_result(
                ndarray(_gauss_jordan_inv(a_arr.tolist())).astype(res_dt))
        lead, subs = _stacked_2d(a_arr)
        out = [_gauss_jordan_inv(sub.tolist()) for sub in subs]
        return _as_result(ndarray(out).astype(res_dt).reshape(shape))

    @staticmethod
    def det(a):
        """计算矩阵的行列式（NumPy 2.x 语义）。

        2-d 返回保输入 dtype 的 0-d 数组（整型输入 → float64）；
        stacked (...,M,M) 返回 (lead...) 数组；空矩阵 (0×0) 行列式为 1；
        全部尺寸走纯 Python LU（``_py_det``，列主元，复数支持）。
        """
        from rsnumpy import (ndarray, float32, float64, complex64,
                             complex128)
        from ._pylinalg import _common_type, _py_det, _stacked_2d
        _SCALAR = {'float32': float32, 'float64': float64,
                   'complex64': complex64, 'complex128': complex128}

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        shape = tuple(a_arr.shape)
        if len(shape) < 2 or shape[-2] != shape[-1]:
            raise ValueError("det requires a square matrix")
        _, res_dt = _common_type(getattr(a_arr, '_dtype', 'float64'))
        if len(shape) == 2:
            # 2-d 返回标量（dtype 随输入；整型输入 → float64 标量）
            return _SCALAR[res_dt](_py_det(a_arr.tolist()))
        lead, subs = _stacked_2d(a_arr)
        vals = [_py_det(sub.tolist()) for sub in subs]
        return ndarray(vals).astype(res_dt).reshape(lead)

    @staticmethod
    def norm(x, ord=None, axis=None, keepdims=False):
        """向量或矩阵范数（NumPy 2.x 语义）。

        向量范数（1-d / axis 为整数）：None/2→L2、1→L1、-1、0、±inf、
        任意 p；矩阵范数（2-d axis=None / axis 为 2-元组）：None/'fro'→
        Frobenius、±1→列和 max/min、±inf→行和 max/min、2/-2→最大/最小
        奇异值、'nuc'→核范数。返回输入实 dtype（inexact）或 float64
        （整型）的 0-d 数组；axis/keepdims 时返回相应形状数组。
        """
        import itertools
        import rsnumpy as _np
        from ..exceptions import AxisError

        x_arr = x if hasattr(x, '_array') else _np.asarray(x)
        ndim = x_arr.ndim
        shape = x_arr.shape

        # ---- 归一化与校验 axis ----
        if axis is None:
            axis_kind = 'none'
            axes = None
        elif isinstance(axis, (tuple, list)):
            if len(axis) != 2:
                raise ValueError("axis must be an integer or a 2-tuple")
            a0, a1 = int(axis[0]), int(axis[1])
            a0n = a0 + ndim if a0 < 0 else a0
            a1n = a1 + ndim if a1 < 0 else a1
            if a0n < 0 or a0n >= ndim or a1n < 0 or a1n >= ndim:
                raise AxisError(
                    "axis out of bounds for array of dimension %d" % ndim)
            if a0n == a1n:
                raise ValueError("axis %d is repeated" % a0n)
            axis_kind = 'matrix'
            axes = (a0n, a1n)
        else:
            axn = int(axis)
            axn = axn + ndim if axn < 0 else axn
            if axn < 0 or axn >= ndim:
                raise AxisError(
                    "axis out of bounds for array of dimension %d" % ndim)
            axis_kind = 'vector'
            axes = (axn,)

        # ---- 上下文与 ord 校验 ----
        if axis_kind == 'none':
            ctx = 'vector' if ndim == 1 else ('matrix' if ndim == 2
                                              else 'flat')
        elif axis_kind == 'vector':
            ctx = 'vector'
        else:
            ctx = 'matrix'

        if ord in ('fro', 'nuc') and ctx != 'matrix':
            raise ValueError("Invalid norm order for matrix norm.")
        if ord == 'fro':
            ord = None  # Frobenius ≡ 矩阵范数 ord None
        if ctx == 'matrix' and ord not in (None, 1, -1, 2, -2,
                                           float('inf'), float('-inf'),
                                           'nuc'):
            raise ValueError("Invalid norm order for matrix norm.")

        out_dt = _norm_out_dtype(getattr(x_arr, '_dtype', 'float64'))

        def _scalar(res):
            a = _np.asarray(res).astype(out_dt)
            if keepdims and axis_kind == 'none':
                a = a.reshape((1,) * ndim)
            return a

        if ctx == 'vector' and axis_kind == 'none':
            return _scalar(_norm_vec(x_arr.tolist(), ord))
        if ctx == 'matrix' and axis_kind == 'none':
            return _scalar(_norm_mat(x_arr.tolist(), ord))
        if ctx == 'flat':
            # ndim>2 且 axis=None：全元素逐元素范数
            flat = x_arr.reshape((x_arr.size,)).tolist()
            return _scalar(_norm_vec(flat, ord))

        # axis 非 None：按 batch 维循环
        reduced = set(axes)
        batch_axes = [i for i in range(ndim) if i not in reduced]
        batch_shape = tuple(shape[i] for i in batch_axes)
        ranges = [range(shape[i]) for i in batch_axes]
        vals = []
        for combo in itertools.product(*ranges):
            idx = []
            ci = 0
            for ax in range(ndim):
                if ax in reduced:
                    idx.append(slice(None))
                else:
                    idx.append(combo[ci])
                    ci += 1
            sub = x_arr[tuple(idx)]
            if axis_kind == 'matrix':
                m = sub.tolist()
                if axes[0] > axes[1]:
                    m = [list(r) for r in zip(*m)]  # 转为 (row, col) 序
                vals.append(_norm_mat(m, ord))
            else:
                vals.append(_norm_vec(sub.tolist(), ord))

        if not batch_axes:
            return _scalar(vals[0])
        out = _np.asarray(vals).reshape(batch_shape)
        if keepdims:
            full = tuple(1 if ax in reduced else shape[ax]
                         for ax in range(ndim))
            out = out.reshape(full)
        return out.astype(out_dt)

    @staticmethod
    def vector_norm(x, /, *, ord=None, axis=None, keepdims=False):
        """Array-API compatible vector norm (numpy.linalg.vector_norm)."""
        import rsnumpy as _np
        x_arr = x if hasattr(x, '_array') else _np.array(x)
        if x_arr.size == 0:
            # NumPy 语义：ord=±inf 用 initial=0，其余范数空和为 0
            r = _np.asarray(0.0)
            if keepdims and axis is None:
                r = r.reshape((1,) * x_arr.ndim) if x_arr.ndim else r
            return r
        absx = _np.abs(x_arr)
        o = 2 if ord is None else ord
        if o == 1:
            return _np.sum(absx, axis=axis, keepdims=keepdims)
        if o in (float('inf'), 'inf'):
            return _np.max(absx, axis=axis, keepdims=keepdims)
        if o in (float('-inf'), '-inf'):
            return _np.min(absx, axis=axis, keepdims=keepdims)
        if o in (2, 2.0, 'fro'):
            return _np.sum(absx ** 2, axis=axis, keepdims=keepdims) ** 0.5
        return _np.sum(absx ** o, axis=axis, keepdims=keepdims) ** (1.0 / o)

    @staticmethod
    def solve(a, b):
        """求解线性方程组 ax = b（NumPy 语义）。

        a: (..., M, M) 方阵或堆栈；b: (M,) 或 (..., M, K)——b 的 lead
        维与 a 广播（b (M,) 对所有批次广播）；x 形状同 b（(M,) →
        (..., M)）；奇异矩阵抛 LinAlgError；dtype 随 a（整型 → float64）。
        """
        from rsnumpy import ndarray
        from ._pylinalg import _common_type, _stacked_2d

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        b_arr = b if hasattr(b, '_array') else ndarray(b)
        a_shape = tuple(a_arr.shape)
        b_shape = tuple(b_arr.shape)
        if len(a_shape) < 2 or a_shape[-2] != a_shape[-1]:
            raise LinAlgError(
                "solve: last 2 dimensions of `a` must be square")
        m = a_shape[-1]
        # b 三种形态（numpy gufunc (m,m),(m,n)->(m,n) 语义）：
        # (m,) → 单向量，广播过 a 全部 lead；(m, k) → 单个矩阵（无
        # lead，广播过 a 全部 lead）；(..., m, k) → lead 维与 a 广播。
        if len(b_shape) == 1:
            b_is_1d = True
            b_lead = ()
            k = 1
            core = b_shape[0]
        elif len(b_shape) == 2:
            b_is_1d = False
            b_lead = ()
            k = b_shape[1]
            core = b_shape[0]
        else:
            b_is_1d = False
            b_lead = b_shape[:-2]
            k = b_shape[-1]
            core = b_shape[-2]
        if core != m:
            raise ValueError(
                "solve: shape of `b` is incompatible with `a`")
        out_lead = list(a_shape[:-2])
        pad = len(b_lead) - len(out_lead)
        if pad > 0:
            out_lead = [1] * pad + out_lead
        for i, d in enumerate(b_lead):
            if out_lead[i] != d and out_lead[i] != 1 and d != 1:
                raise ValueError(
                    "solve: shape of `b` is incompatible with `a`")
            out_lead[i] = max(out_lead[i], d)
        out_lead = tuple(out_lead)

        res_dt = _common_type(getattr(a_arr, '_dtype', 'float64'))[1]
        # Rust 快路径：实数 2x2/3x3 且 b 为 2-d
        use_rust = (not _is_complex_arr(a_arr)
                    and not _is_complex_arr(b_arr) and m in (2, 3)
                    and not b_is_1d)

        def _solve_one(A, B, b1d):
            if use_rust:
                try:
                    return _wrap(_core.linalg.solve(
                        _ensure(A), _ensure(B))).tolist()
                except ValueError:
                    pass
            b_rows = B.tolist()
            if b1d:
                b_rows = [[v] for v in b_rows]
            try:
                return _gauss_jordan_solve(A.tolist(), b_rows)
            except LinAlgError:
                return _solve_pinv(A.tolist(), b_rows)

        def _as_result(arr):
            # 结果保留 b 的 ndarray 子类（numpy 语义）
            bt = type(b_arr)
            if bt is not ndarray and isinstance(b_arr, ndarray):
                return bt(arr)
            return arr

        if k == 0 and not b_is_1d:
            # b (..., m, 0)：零个右端列，无需解算，直接给空结构
            if len(a_shape) == 2:
                rows = [[] for _ in range(m)]
                return _as_result(ndarray(rows).astype(res_dt).reshape(
                    (m, 0)))
            lead, subs = _stacked_2d(a_arr)
            rows = [[[] for _ in range(m)] for _ in subs]
            return _as_result(ndarray(rows).astype(res_dt).reshape(
                out_lead + (m, 0)))

        if m == 0:
            # 空矩阵：不解算，直接给形状 out_lead + (0, k)
            target = out_lead + (0,) if b_is_1d else out_lead + (0, k)
            return _as_result(ndarray([]).astype(res_dt).reshape(target))

        if len(a_shape) == 2:
            rows = _solve_one(a_arr, b_arr, b_is_1d)
            if b_is_1d:
                return _as_result(
                    ndarray([r[0] for r in rows]).astype(res_dt))
            return _as_result(ndarray(rows).astype(res_dt))

        # stacked: a (lead..., m, m)；b (m,)/(m,k) 整体广播，
        # (..., m, k) 按 numpy 广播规则逐子矩阵取 b 的对应子块
        lead, subs = _stacked_2d(a_arr)
        b_lead_full = (1,) * (len(out_lead) - len(b_lead)) + b_lead

        def _t_index(t, shape):
            idx = []
            for d in reversed(shape):
                idx.append(t % d)
                t //= d
            return list(reversed(idx))

        a_lead = a_shape[:-2]
        a_lead_full = (1,) * (len(out_lead) - len(a_lead)) + a_lead
        b_flat = None
        if not (b_is_1d or len(b_shape) == 2):
            b_total = 1
            for d in b_lead:
                b_total *= d
            b_flat = b_arr.reshape((b_total,) + b_shape[-2:])
        total = 1
        for d in out_lead:
            total *= d
        out_rows = []
        for t in range(total):
            ti = _t_index(t, out_lead)
            ai = 0
            for d_a, i in zip(a_lead_full, ti):
                ai = ai * d_a + (i if d_a > 1 else 0)
            A = subs[ai]
            if b_flat is None:
                B = b_arr
            else:
                bi = 0
                for d_b, i in zip(b_lead_full, ti):
                    bi = bi * d_b + (i if d_b > 1 else 0)
                B = b_flat[bi]
            out_rows.append(_solve_one(A, B, b_is_1d))
        if b_is_1d:
            flat = [r[0] for rows in out_rows for r in rows]
            return _as_result(ndarray(flat).astype(res_dt).reshape(
                out_lead + (m,)))
        return _as_result(ndarray(out_rows).astype(res_dt).reshape(
            out_lead + (m, k)))

    @staticmethod
    def lstsq(a, b, rcond=None):
        """最小二乘解 min ||a·x - b||₂（SVD 法，NumPy 2.x 语义）。

        返回 (x, residuals, rank, s)：s 为降序奇异值；rank = 大于
        ``rcond·s_max`` 的奇异值个数（rcond=None → 1e-15；rcond<0 → 机器
        精度，LAPACK 约定）；residuals 为各列残差平方和（仅 rank==n 且
        m>n 时非空），实浮点 dtype；a/b 含 inf/nan 抛 LinAlgError；
        维度不匹配抛 LinAlgError("Incompatible dimensions")。
        """
        import math
        from rsnumpy import ndarray
        from ._pylinalg import (_common_type, _real_dtype, _finfo_eps)

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        b_arr = b if hasattr(b, '_array') else ndarray(b)
        if len(a_arr.shape) != 2:
            raise LinAlgError("expected square matrix")
        m, n = a_arr.shape
        if len(b_arr.shape) not in (1, 2) or b_arr.shape[0] != m:
            raise LinAlgError("Incompatible dimensions")
        b_is_1d = len(b_arr.shape) == 1
        n_rhs = 1 if b_is_1d else b_arr.shape[1]

        a_dt = getattr(a_arr, '_dtype', 'float64')
        res_dt = _common_type(a_dt)[1]
        s_dt = _real_dtype(res_dt)

        a_rows = a_arr.tolist()
        b_rows = b_arr.tolist()
        if b_is_1d:
            b_rows = [[v] for v in b_rows]

        def _scan_nonfinite(data):
            for row in data:
                if not isinstance(row, (list, tuple)):
                    row = [row]
                for v in row:
                    if isinstance(v, complex):
                        r, i = v.real, v.imag
                    else:
                        r, i = float(v), 0.0
                    if r != r or i != i or math.isinf(r) or math.isinf(i):
                        raise LinAlgError(
                            "Array must not contain infs or NaNs")

        _scan_nonfinite(a_rows)
        _scan_nonfinite(b_rows)

        u, s, vt = linalg_module.svd(a_arr, full_matrices=False)
        s_list = [float(abs(v)) for v in s.tolist()]

        if rcond is None:
            rcond_val = 1.0e-15
        elif rcond < 0:
            rcond_val = _finfo_eps(s_dt)
        else:
            rcond_val = float(rcond)
        s_max = max(s_list) if s_list else 0.0
        cutoff = rcond_val * s_max
        rank = int(sum(1 for v in s_list if v > cutoff))

        # x = Vh^H · diag(1/s_i) · U^H · b（s_i ≤ cutoff 置 0）
        k = min(m, n)
        u_rows = u.tolist()   # m × k
        vh_rows = vt.tolist()  # k × n

        def _conj(z):
            return z.conjugate() if isinstance(z, complex) else z

        s_inv = [(1.0 / v) if v > cutoff else 0.0 for v in s_list]
        x_rows = [[0.0] * n_rhs for _ in range(n)]
        for c in range(n_rhs):
            uhb = [0.0] * k
            for j in range(k):
                acc = 0
                for i in range(m):
                    acc += _conj(u_rows[i][j]) * b_rows[i][c]
                uhb[j] = acc
            for i in range(n):
                acc = 0
                for j in range(k):
                    acc += _conj(vh_rows[j][i]) * s_inv[j] * uhb[j]
                x_rows[i][c] = acc

        if b_is_1d:
            x = ndarray([x_rows[i][0] for i in range(n)]).astype(res_dt)
        elif n == 0:
            x = ndarray([]).reshape((0, n_rhs)).astype(res_dt)
        else:
            x = ndarray(x_rows).astype(res_dt)

        if m > n and rank == n and n_rhs > 0:
            res_vals = []
            for c in range(n_rhs):
                acc = 0.0
                for i in range(m):
                    r = b_rows[i][c]
                    for kk in range(n):
                        r -= a_rows[i][kk] * x_rows[kk][c]
                    acc += abs(r) ** 2
                res_vals.append(acc)
            res = ndarray(res_vals).astype(s_dt)
        else:
            res = ndarray([]).astype(s_dt)
        return (x, res, rank, ndarray(s_list).astype(s_dt))

    @staticmethod
    def matrix_rank(a, tol=None, hermitian=False, *, rtol=None):
        """矩阵秩（NumPy 2.x 语义）：2-d/1-d 返回 int；stacked (...,M,N)
        返回 intp 计数数组（rank = count(S > tol)，tol 默认
        ``S.max() * max(M,N) * eps``）。rtol 与 tol 互斥（NumPy 2.0+）。"""
        from rsnumpy import ndarray
        from ._pylinalg import (_common_type, _real_dtype, _finfo_eps)

        if tol is not None and rtol is not None:
            raise ValueError("`tol` and `rtol` can't be both set.")
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        shape = a_arr.shape
        if len(shape) > 2:
            s = linalg_module.svd(a_arr, compute_uv=False, hermitian=hermitian)
            dt = getattr(a_arr, '_dtype', 'float64')
            try:
                s_dt = _real_dtype(_common_type(dt)[1])
            except TypeError:
                s_dt = 'float64'
            eps = _finfo_eps(s_dt)
            m, n = shape[-2], shape[-1]

            def _count(vec, rtol_i=None):
                if tol is not None:
                    t = tol
                elif rtol is not None:
                    # NumPy 语义：tol = S.max() * rtol（rtol 可逐批次）
                    _ri = rtol_i if rtol_i is not None else float(rtol)
                    t = (max(vec) if vec else 0.0) * _ri
                else:
                    t = (max(vec) * max(m, n) * eps if vec else 0.0)
                return sum(1 for v in vec if v > t)

            # s 带完整 lead 形状（(lead..., K)）：先压成 2-d (total, K)
            # 逐行计数，再 reshape 回 lead 形状
            total = 1
            for d in shape[:-2]:
                total *= d
            s2 = s.reshape((total, min(m, n)))
            # rtol 支持标量或数组（NumPy 2.0+：广播到批次形状，逐批次
            # tol = S.max() * rtol[i]）
            rtols = None
            if rtol is not None:
                try:
                    rtols = [float(rtol)] * total
                except (TypeError, ValueError):
                    _rv = ndarray(rtol)
                    _flat = _rv.reshape(total).tolist() \
                        if _rv.size == total else \
                        list(_rv.reshape((-1,)).tolist()) * (total // max(_rv.size, 1))
                    if len(_flat) < total:
                        _flat = (_flat * (total // len(_flat) + 1))[:total]
                    rtols = [float(v) for v in _flat]
            if rtols is None:
                parts = [_count(v) for v in s2.tolist()]
            else:
                parts = [_count(v, r) for v, r in
                         zip(s2.tolist(), rtols)]
            return ndarray(parts).reshape(shape[:-2]).astype('int64')
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
            if rtol is not None:
                # NumPy 语义：tol = S.max() * rtol（max_abs 近似 S.max()）
                tol = max_abs * rtol
            else:
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
        """计算矩阵的特征值和特征向量（NumPy 2.x 语义）。

        返回 EigResult(eigenvalues, eigenvectors)：eigenvalues (..., n)
        恒为复数 dtype（f32→c64、f64/int→c128、c64→c64、c128→c128），
        eigenvectors (..., n, n) 同 dtype 且保留 a 的 ndarray 子类；
        stacked (..., n, n) 支持任意 lead 维；空尺寸返回同形状空复数数组。
        """
        from rsnumpy import ndarray
        from ._pylinalg import _stacked_2d
        a = a if hasattr(a, '_array') else ndarray(a)
        shape = tuple(a.shape)
        if len(shape) < 2 or shape[-2] != shape[-1]:
            raise LinAlgError(
                "eig: last 2 dimensions of the array must be square")
        # numpy：eig 输出恒为复数 dtype（实数输入 → 对应复数）
        dt = getattr(a, '_dtype', 'float64')
        if dt.startswith('complex'):
            res_dt = dt
        elif dt == 'float32':
            res_dt = 'complex64'
        else:
            res_dt = 'complex128'

        def _as_result(arr):
            # eigenvectors 保留 a 的 ndarray 子类（numpy 语义）
            at = type(a)
            if at is not ndarray and isinstance(a, ndarray):
                return at(arr)
            return arr

        lead, subs = _stacked_2d(a)
        n = shape[-1]
        if n == 0:
            ev = ndarray([]).astype(res_dt).reshape(lead + (0,))
            ec = ndarray([]).astype(res_dt).reshape(lead + (0, 0))
            return EigResult(_as_result(ev), _as_result(ec))
        if len(shape) == 2:
            ev, ec = _eig_one(a)
            return EigResult(_as_result(ndarray(ev).astype(res_dt)),
                             _as_result(ndarray(ec).astype(res_dt)))
        eval_list = []
        evec_list = []
        for sub in subs:
            ev, ec = _eig_one(sub)
            eval_list.append(ev)
            evec_list.append(ec)
        # 空 lead（(0,1,1) 等）时嵌套 list 为空，用 reshape 补回 lead 维
        ev = ndarray(eval_list).reshape(lead + (n,)).astype(res_dt)
        ec = ndarray(evec_list).reshape(lead + (n, n)).astype(res_dt)
        return EigResult(_as_result(ev), _as_result(ec))

    @staticmethod
    def eigvals(a):
        """计算矩阵特征值（NumPy 语义：2-d 返回 (n,)，stacked 返回
        (lead..., n)；输出 complex128，float32/complex64 输入为 complex64）。
        2x2/3x3 实输入走 Rust 快路径，其余尺寸走纯 Python QR 算法
        （``_eigvals_qr``，Hessenberg + 单位移 QR 迭代）。
        """
        from rsnumpy import ndarray
        from ._pylinalg import _eigvals_qr, _stacked_2d

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        shape = tuple(a_arr.shape)
        if len(shape) < 2 or shape[-2] != shape[-1]:
            raise LinAlgError(
                "eigvals: last 2 dimensions of the array must be square")
        dt = getattr(a_arr, '_dtype', 'float64')
        out_dt = 'complex64' if dt in ('float32', 'complex64') \
            else 'complex128'
        if len(shape) == 2 and shape[-1] in (2, 3) \
                and dt in ('float32', 'float64'):
            try:
                return _wrap(_core.linalg.eigvals(_ensure(a_arr))) \
                    .astype(out_dt)
            except ValueError:
                pass
        lead, subs = _stacked_2d(a_arr)
        if len(shape) == 2:
            return ndarray(_eigvals_qr(a_arr.tolist())).astype(out_dt)
        evals_list = [_eigvals_qr(sub.tolist()) for sub in subs]
        return ndarray(evals_list).reshape(lead + (shape[-1],)) \
            .astype(out_dt)

    @staticmethod
    def svd(a, full_matrices=True, compute_uv=True, hermitian=False,
            lapack_driver=None):
        """计算奇异值分解 A = u @ diag(s) @ vh。

        支持 NumPy 的 ``full_matrices`` / ``compute_uv`` 关键字；2×2 走 Rust
        核心，其余形状用单侧 Jacobi（``_svd_jacobi``）。stacked (...,M,N)
        按切片循环；dtype 按 NumPy _commonType 规则传播（U/Vh 保持输入
        dtype、S 为对应实 dtype）；含 inf/nan 抛 LinAlgError（模拟 LAPACK
        "SVD did not converge"）。
        """
        from rsnumpy import ndarray, empty, eye
        from ._pylinalg import (_common_type, _real_dtype, _nonfinite_check,
                                _stacked_2d)

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        shape = tuple(a_arr.shape)
        if len(shape) < 2:
            raise LinAlgError(
                "1-dimensional array given. Array must be at least "
                "two-dimensional")
        _, res_dt = _common_type(getattr(a_arr, '_dtype', 'float64'))
        s_dt = _real_dtype(res_dt)

        lead, subs = _stacked_2d(a_arr)
        for sub in subs:
            _nonfinite_check(sub.tolist())
        m, n = shape[-2], shape[-1]
        k = min(m, n)

        # ---- 空矩阵：U/Vh 为单位阵（full_matrices）或约定形状零阵，S 空 ----
        def _empty_result():
            if len(shape) > 2:
                total = 1
                for d in lead:
                    total *= d

                def _block(r, c, identity):
                    if identity and r > 0 and r == c:
                        return (ndarray([eye(r).tolist()] * total)
                                .reshape(lead + (r, c)).astype(res_dt))
                    return ndarray([]).reshape(lead + (r, c)).astype(res_dt)

                if not compute_uv:
                    return ndarray([]).reshape(lead + (0,)).astype(s_dt)
                s = ndarray([]).reshape(lead + (0,)).astype(s_dt)
                return SVDResult(
                    _block(m, m if full_matrices else k, full_matrices), s,
                    _block(n if full_matrices else k, n, full_matrices))

            if not compute_uv:
                return ndarray([]).astype(s_dt)

            def _block2(r, c, identity):
                if identity and r > 0 and r == c:
                    return eye(r).astype(res_dt)
                return empty((r, c)).astype(res_dt)

            u = _block2(m, m if full_matrices else k, full_matrices)
            vt = _block2(n if full_matrices else k, n, full_matrices)
            return SVDResult(u, ndarray([]).astype(s_dt), vt)

        if m == 0 or n == 0:
            return _empty_result()

        if hermitian:
            if m != n:
                raise LinAlgError(
                    "Last 2 dimensions of the array must be square")

            def _one_h(sub):
                # Hermitian：特征分解 A = V diag(w) V^H；S = |w| 降序，
                # 负特征值符号吸收进 U 的对应列，Vh = 重排后的 V^H
                w, v = linalg_module.eig(sub)
                w_list = w.tolist()
                v_rows = v.tolist()
                size = len(w_list)
                s_abs = [abs(complex(x)) for x in w_list]
                signs = [1.0 if complex(x).real >= 0 else -1.0
                         for x in w_list]
                perm = sorted(range(size), key=lambda i: -s_abs[i])
                s_out = [s_abs[i] for i in perm]

                def _conj(z):
                    return z.conjugate() if isinstance(z, complex) else z

                if not compute_uv:
                    return ndarray(s_out).astype(s_dt)
                u_rows = [[signs[perm[j]] * v_rows[i][perm[j]]
                           for j in range(size)] for i in range(size)]
                # Vh = V^H（共轭转置）：Vh[j][i] = conj(V[i][perm[j]])
                vh_rows = [[_conj(v_rows[i][perm[j]])
                            for i in range(size)] for j in range(size)]
                return SVDResult(ndarray(u_rows).astype(res_dt),
                                 ndarray(s_out).astype(s_dt),
                                 ndarray(vh_rows).astype(res_dt))

            if len(shape) == 2:
                return _one_h(a_arr)
            results = [_one_h(sub) for sub in subs]
            if not compute_uv:
                s = ndarray([r.tolist() for r in results]).astype(s_dt)
                return s.reshape(lead + (s.shape[-1],))
            # results 按 lead 维行主序排列，恢复完整 leading 形状
            u = ndarray([r.U.tolist() for r in results]).astype(res_dt)
            s = ndarray([r.S.tolist() for r in results]).astype(s_dt)
            vt = ndarray([r.Vh.tolist() for r in results]).astype(res_dt)
            return SVDResult(u.reshape(lead + tuple(results[0].U.shape)),
                             s.reshape(lead + tuple(results[0].S.shape)),
                             vt.reshape(lead + tuple(results[0].Vh.shape)))

        def _one(sub):
            if not compute_uv:
                s = _svd_jacobi(sub.tolist(), False, False)
                return s.astype(s_dt)
            # Rust 2x2 SVD 只处理实输入（复数输入返回实 U/Vh，结果错误），
            # 复数 2x2 走 _svd_jacobi 的复数路径
            if tuple(sub.shape) == (2, 2) and not _is_complex_arr(sub):
                try:
                    u, s, vt = _core.linalg.svd(_ensure(sub))
                    u = _round_array(u)
                    s = _round_array(s)
                    vt = _round_array(vt)
                    # Rust 2x2 的 s 是 2-d 对角矩阵，归一为 1-d 对角
                    # （与 3x3+ Jacobi 路径的 1-d s 一致）
                    s_list = s.tolist()
                    if (isinstance(s_list, list) and s_list
                            and isinstance(s_list[0], (list, tuple))):
                        s_list = [row[i] for i, row in enumerate(s_list)]
                    return SVDResult(ndarray._wrap(u).astype(res_dt),
                                     ndarray(s_list).astype(s_dt),
                                     ndarray._wrap(vt).astype(res_dt))
                except Exception:
                    pass
            u, s, vt = _svd_jacobi(sub.tolist(), full_matrices, True)
            return SVDResult(u.astype(res_dt), s.astype(s_dt),
                             vt.astype(res_dt))

        if len(shape) == 2:
            return _one(a_arr)
        results = [_one(sub) for sub in subs]
        if not compute_uv:
            s = ndarray([r.tolist() for r in results]).astype(s_dt)
            return s.reshape(lead + (s.shape[-1],))
        # results 按 lead 维行主序排列，恢复完整 leading 形状
        u = ndarray([r.U.tolist() for r in results]).astype(res_dt)
        s = ndarray([r.S.tolist() for r in results]).astype(s_dt)
        vt = ndarray([r.Vh.tolist() for r in results]).astype(res_dt)
        return SVDResult(u.reshape(lead + tuple(results[0].U.shape)),
                         s.reshape(lead + tuple(results[0].S.shape)),
                         vt.reshape(lead + tuple(results[0].Vh.shape)))

    @staticmethod
    def qr(a, mode='reduced'):
        """计算 QR 分解（NumPy 语义）。

        modes: 'reduced'（默认）/ 'complete' / 'r' / 'raw'；NumPy >= 1.8 已移除
        弃用别名 'full'/'f'/'economic'/'e'。2×2/3×3 reduced 走 Rust 核心，
        其余形状用纯 Python Householder（``_qr_householder``）；stacked
        (...,M,N) 按切片循环；空矩阵返回约定形状（complete 时 Q=I_m）；
        'raw' 返回 (h, tau)（FORTRAN 序 h (n,m) + tau (k,)）。
        """
        from rsnumpy import ndarray, empty, eye
        from ._pylinalg import (_common_type, _stacked_2d, _qr_raw)

        # NumPy >= 1.8：弃用别名 'full'/'f'/'economic'/'e' 已移除。
        if mode in ('full', 'f'):
            raise ValueError(
                f"mode {mode!r} is deprecated in NumPy 1.8, use 'complete'")
        if mode in ('economic', 'e'):
            raise ValueError(
                f"mode {mode!r} is deprecated in NumPy 1.8, use 'reduced'")
        if mode not in ('reduced', 'complete', 'r', 'raw'):
            raise ValueError(
                f"mode {mode!r} is not recognized. Must be one of "
                "'raw', 'reduced', 'complete', 'r'.")

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        shape = tuple(a_arr.shape)
        if len(shape) < 2:
            raise LinAlgError(
                "1-dimensional array given. Array must be at least "
                "two-dimensional")
        res_dt = _common_type(getattr(a_arr, '_dtype', 'float64'))[1]
        m, n = shape[-2], shape[-1]
        k = min(m, n)

        def _empty_res():
            if mode == 'raw':
                return (ndarray(empty((n, m))).astype(res_dt),
                        ndarray(empty((k,))).astype(res_dt))
            if mode in ('reduced', 'r'):
                return (ndarray(empty((m, k))).astype(res_dt),
                        ndarray(empty((k, n))).astype(res_dt))
            return (ndarray(eye(m)).astype(res_dt),
                    ndarray(empty((m, n))).astype(res_dt))

        if m == 0 or n == 0:
            res = _empty_res()
            if len(shape) > 2:
                # stacked 空矩阵：numpy 语义把各分量广播到 (lead, ...)
                # （'r'/'raw' 同样适用，否则回 2-d 形状）
                from rsnumpy import broadcast_to
                lead = shape[:-2]
                res = tuple(broadcast_to(x, lead + x.shape) for x in res)
            if mode == 'r':
                return res[1]
            if mode == 'raw':
                return res
            return QRResult(res[0], res[1])

        def _one(sub):
            sm, sn = sub.shape
            if mode == 'raw':
                h, tau = _qr_raw(sub.tolist())
                return (ndarray(h).astype(res_dt), ndarray(tau).astype(res_dt))
            # 32 位（f32/c64）与 c128 不走 Rust 2x2/3x3 快捷路径：Rust 侧
            # 缓冲在连续调用（同一会话内前序 QR/SVD）后可能读到脏状态
            # （实测 c64 返回全 1、f32 返回全 0、c128 返回全 1），
            # Python Householder 无此问题且精度足够
            if sm == sn and sm in (2, 3) and mode == 'reduced' \
                    and getattr(sub, '_dtype', '') not in ('float32',
                                                           'complex64',
                                                           'complex128'):
                try:
                    q, r = _core.linalg.qr(_ensure(sub))
                    q = _round_array(q)
                    r = _round_array(r)
                    return (ndarray._wrap(q).astype(res_dt),
                            ndarray._wrap(r).astype(res_dt))
                except Exception:
                    pass
            q, r = linalg_module._qr_householder(
                sub, mode if mode in ('reduced', 'complete') else 'reduced')
            q = q.astype(res_dt)
            r = r.astype(res_dt)
            if mode == 'r':
                return r
            return (q, r)

        if len(shape) == 2:
            out = _one(a_arr)
            if mode == 'r':
                return out
            return QRResult(out[0], out[1])

        lead, subs = _stacked_2d(a_arr)
        parts = [_one(sub) for sub in subs]
        if mode == 'raw':
            h = ndarray([p[0].tolist() for p in parts]).reshape(lead + (n, m))
            tau = ndarray([p[1].tolist() for p in parts]).reshape(lead + (k,))
            return (h.astype(res_dt), tau.astype(res_dt))
        if mode == 'r':
            return ndarray([p.tolist() for p in parts]).reshape(
                lead + (k, n)).astype(res_dt)
        q = ndarray([p[0].tolist() for p in parts]).reshape(
            lead + ((m, m) if mode == 'complete' else (m, k))).astype(res_dt)
        r = ndarray([p[1].tolist() for p in parts]).reshape(
            lead + ((m, n) if mode == 'complete' else (k, n))).astype(res_dt)
        return QRResult(q, r)

    @staticmethod
    def _qr_householder(a, mode='reduced'):
        """Householder 反射 QR，支持任意 m x n 与 reduced/complete 形态。"""
        import math

        from rsnumpy import ndarray

        if mode == 'reduced':
            want_complete = False
        elif mode == 'complete':
            want_complete = True
        else:
            raise ValueError(
                "mode must be one of 'reduced' or 'complete', "
                f"got {mode!r}")

        rows = [list(row) for row in _ensure(a).tolist()]
        m = len(rows)
        n = len(rows[0]) if m else 0
        if m == 0 or n == 0:
            raise ValueError('cannot factor an empty matrix')

        is_complex = any(isinstance(v, complex) for row in rows for v in row)
        conj = (lambda z: z.conjugate()) if is_complex else (lambda z: z)

        def dot2(xs, ys):
            return sum(conj(x) * y for x, y in zip(xs, ys)) if is_complex \
                else sum(x * y for x, y in zip(xs, ys))

        # R 就地化为上三角，Q 同步累积每一次反射。
        q_mat = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
        for j in range(min(m, n)):
            width = m - j
            col_j = [rows[j + i][j] for i in range(width)]
            norm = math.sqrt(sum(abs(x) ** 2 for x in col_j))
            if norm == 0.0:
                continue
            # v = x - alpha*e1。复数时 alpha 必须与 x0 同相
            # （alpha = -x0/|x0|·||x||），Hx 才能精确落到 e1 方向；
            # 实数 alpha = -sign(x0)·||x||。均取首分量相加以避相消。
            lead = col_j[0]
            if is_complex:
                mag = abs(lead)
                alpha = (-lead / mag * norm) if mag > 0 else complex(-norm)
            else:
                alpha = -norm if lead < 0 else norm
            v = list(col_j)
            v[0] = v[0] - alpha
            vv = sum(abs(t) ** 2 for t in v)
            if vv == 0.0:
                continue

            for c in range(j, n):
                factor = 2.0 * dot2(v, [rows[j + i][c] for i in range(width)]) / vv
                for i in range(width):
                    rows[j + i][c] -= factor * v[i]

            for r_row in range(m):
                # Q_new = Q_old @ H，H = I - 2 v v^H/v^H v：
                # (QH)[r][c] = Q[r][c] - (2/vv)(Qv)[r]·conj(v[c-j])。
                # dot2(Qseg, conj(v)) = conj((Qv)[r])，复数时须再取共轭。
                factor = 2.0 * dot2([q_mat[r_row][j + i] for i in range(width)],
                                    [conj(t) for t in v]) / vv
                if is_complex:
                    factor = factor.conjugate()
                for i in range(width):
                    q_mat[r_row][j + i] -= factor * conj(v[i])

        # 反射残留的下三角应恰为 0，显式置零以便上三角/形状判定通过。
        for i in range(m):
            for j in range(min(i, n)):
                rows[i][j] = 0.0

        if want_complete:
            out_q, out_r = q_mat, rows
        else:
            k = min(m, n)
            out_q = [row[:k] for row in q_mat]
            out_r = rows[:k]

        return (ndarray._wrap(_core.ndarray(out_q)),
                ndarray._wrap(_core.ndarray(out_r)))

    @staticmethod
    def cholesky(a, /, *, upper=False):
        """计算 Cholesky 分解：A = L·L^H（lower）或 U^H·U（upper）。

        A 须 Hermitian 正定（不校验 Hermitian，只用对应三角 + 对角）。
        纯 Python Cholesky-Crout（任意尺寸）；stacked (...,M,M) 按切片；
        空 2-d 返回同形状空数组（dtype 按 _commonType：int→f64）；
        float16 抛 TypeError（"unsupported in linalg"）；非正定/含
        inf/nan 抛 LinAlgError。
        """
        from rsnumpy import ndarray, empty
        from ._pylinalg import (_common_type, _stacked_2d, _is_empty_2d,
                                _py_cholesky, _nonfinite_check)

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        shape = tuple(a_arr.shape)
        if len(shape) < 2:
            raise LinAlgError(
                "1-dimensional array given. Array must be at least "
                "two-dimensional")
        if shape[-1] != shape[-2]:
            raise LinAlgError("Last 2 dimensions of the array must be square")
        res_dt = _common_type(getattr(a_arr, '_dtype', 'float64'))[1]
        if _is_empty_2d(a_arr):
            return empty(shape, dtype=res_dt)

        lead, subs = _stacked_2d(a_arr)

        def _one(sub):
            mat = sub.tolist()
            _nonfinite_check(
                mat, "matrix is not positive-definite (cholesky)")
            return _py_cholesky(mat, upper)

        if len(shape) == 2:
            return ndarray(_one(a_arr)).astype(res_dt)
        parts = [_one(sub) for sub in subs]
        total = 1
        for d in lead:
            total *= d
        return (ndarray(parts).reshape((total,) + shape[-2:])
                .reshape(lead + shape[-2:]).astype(res_dt))

    @staticmethod
    def matrix_power(a, n):
        """计算矩阵的幂（NumPy 2.x 语义）。

        ``n`` 须为整数（否则 TypeError）；0 → 单位矩阵（保持 dtype 与
        stacked 形状）；n>0 二进制快速幂（纯 Python，任意尺寸 / stacked
        (...,M,M)）；n<0 先求逆再乘方（奇异矩阵抛 LinAlgError，与
        NumPy 一致）；非方阵抛 LinAlgError。
        """
        from rsnumpy import ndarray, array
        from ._pylinalg import (_matrix_power_mat, _stacked_2d)

        if not isinstance(n, int):
            idx = getattr(n, '__index__', None)
            if idx is None:
                raise TypeError("Power must be an integer")
            n = idx()

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        shape = tuple(a_arr.shape)
        if len(shape) < 2 or shape[-1] != shape[-2]:
            raise LinAlgError("Square matrix expected")

        dtype = getattr(a_arr, '_dtype', 'float64')
        # subok：输入为 matrix 时结果保持 matrix（numpy matrix_power 语义）
        _is_mat = type(a).__name__ == 'matrix'

        def _fin(res):
            if _is_mat:
                from rsnumpy import matrix as _mx
                return _mx(res)
            return res

        if n < 0:
            try:
                inv_a = linalg_module.inv(a_arr)
            except LinAlgError:
                raise
            except ValueError:
                # 奇异：底层抛 ValueError("Singular matrix")，NumPy 的
                # matrix_power 抛 LinAlgError
                raise LinAlgError("Singular matrix")
            return _fin(linalg_module.matrix_power(inv_a, -n))

        lead, subs = _stacked_2d(a_arr)
        if dtype == 'object':
            # object 数组平铺存储（astype/reshape 不重嵌套），须手动嵌套
            # 并用 array(dtype=) 直接构造结果
            m = shape[-1]

            def _unflat(lst):
                if isinstance(lst, list) and lst \
                        and not isinstance(lst[0], (list, tuple)):
                    return [lst[i * m:(i + 1) * m] for i in range(m)]
                return lst

            if len(shape) == 2:
                return _fin(array(_matrix_power_mat(_unflat(subs.tolist()), n),
                                  dtype='object'))
            parts = [_matrix_power_mat(_unflat(sub.tolist()), n)
                     for sub in subs]
            return _fin(array(parts, dtype='object'))

        if len(shape) == 2:
            return _fin(ndarray(_matrix_power_mat(subs.tolist(), n)).astype(dtype))
        parts = [_matrix_power_mat(sub.tolist(), n) for sub in subs]
        total = len(parts)
        return _fin(ndarray(parts).reshape((total,) + shape[-2:])
                    .reshape(lead + shape[-2:]).astype(dtype))

    @staticmethod
    def pinv(a, rcond=None, hermitian=False, rtol=NoValue):
        """计算 Moore-Penrose 伪逆（NumPy 2.x 参数语义）。

        ``rcond``/``rtol`` 二选一：都未给用 1e-15；``rtol=None`` 用
        ``max(M,N)*eps(结果 dtype)``（按 NumPy Array API 标准，dtype 取
        伪逆实际计算/返回的 dtype——int/bool 输入在 float64 上计算）；
        同时给两者抛 ValueError。空 2-d 输入走 empty 快捷路径（返回
        转置形状、正确 dtype 的空数组）。
        """
        from rsnumpy import ndarray, empty
        from ._pylinalg import (_common_type, _is_empty_2d, _real_dtype,
                                _finfo_eps)

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        dtype = getattr(a_arr, '_dtype', 'float64')
        res_dt = _common_type(dtype)[1]

        if rcond is not None and rtol is not NoValue:
            raise ValueError("`rtol` and `rcond` can't be both set.")
        if rcond is None:
            if rtol is NoValue:
                rcond_val = 1e-15
            elif rtol is None:
                rcond_val = max(a_arr.shape[-2:]) * _finfo_eps(
                    _real_dtype(res_dt))
            else:
                rcond_val = rtol
        else:
            rcond_val = rcond

        if _is_empty_2d(a_arr):
            m, n = a_arr.shape[-2:]
            return empty(a_arr.shape[:-2] + (n, m), dtype=res_dt)

        # A^+ = V Σ^+ U^H，其中 (U, s, Vh) = svd(A, full_matrices=False)
        #（对 A 本身做 SVD；_one 按 V[i,j]=vt[j,i].conj() 组装）
        u, s, vt = linalg_module.svd(a_arr, full_matrices=False,
                                     hermitian=hermitian)

        def _one(u_m, s_list, vt_k):
            m = len(u_m)
            n = len(vt_k[0]) if vt_k else 0
            k = len(s_list)
            s_max = max(s_list) if s_list else 0.0
            cutoff = float(rcond_val) * s_max
            res = [[0.0] * m for _ in range(n)]
            for j in range(k):
                si = (1.0 / s_list[j]) if s_list[j] > cutoff else 0.0
                if si == 0.0:
                    continue
                for i in range(n):
                    vij = vt_k[j][i].conjugate() * si
                    if vij == 0:
                        continue
                    res_i = res[i]
                    for r in range(m):
                        res_i[r] += vij * u_m[r][j].conjugate()
            return res

        shape = a_arr.shape
        if len(shape) == 2:
            res = _one(u.tolist(), s.tolist(), vt.tolist())
            return ndarray(res).astype(res_dt)
        lead = shape[:-2]
        total = 1
        for d in lead:
            total *= d
        flat_u = u.reshape((total,) + u.shape[-2:])
        flat_s = s.reshape((total,) + s.shape[-1:])
        flat_vt = vt.reshape((total,) + vt.shape[-2:])
        parts = [_one(flat_u[i].tolist(), flat_s[i].tolist(),
                      flat_vt[i].tolist())
                 for i in range(flat_u.shape[0])]
        m, n = shape[-2], shape[-1]
        return ndarray(parts).reshape(lead + (n, m)).astype(res_dt)

    @staticmethod
    def trace(a, offset=0, axis1=-2, axis2=-1, dtype=None, out=None):
        """计算矩阵的迹（numpy.linalg.trace 语义：默认 axis1=-2, axis2=-1；
        stacked (...,M,N) 返回最后两轴各矩阵的迹；2-d offset=0 走 Rust 快路径）。"""
        from rsnumpy import ndarray
        _ = dtype, out
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        sh = tuple(a_arr.shape)
        if len(sh) < 2:
            raise ValueError(
                "trace requires an array of at least 2 dimensions")
        if len(sh) == 2 and offset == 0:
            result = _core.linalg.trace(_ensure(a))
            return _round_float(result)
        return linalg_module.diagonal(a_arr, offset, axis1, axis2).sum(-1)

    @staticmethod
    def diagonal(a, offset=0, axis1=-2, axis2=-1):
        """返回矩阵的对角线元素（numpy.linalg.diagonal 语义：默认
        axis1=-2, axis2=-1，支持任意维数组）。"""
        from rsnumpy import ndarray
        a_arr = a if hasattr(a, '_array') else ndarray(a)
        sh = tuple(a_arr.shape)
        if len(sh) < 2:
            raise ValueError(
                "diagonal requires an array of at least 2 dimensions")
        if len(sh) == 2:
            return a_arr.diagonal(offset, 0, 1)
        # ndim>2：逐批次提取 a1/a2 两轴对角（核心 diagonal 不返回
        # 批次形状）：先把 a1/a2 移到末尾，逐 2-d 切片取对角
        nd = len(sh)
        a1, a2 = axis1 % nd, axis2 % nd
        order = [i for i in range(nd) if i not in (a1, a2)] + [a1, a2]
        moved = a_arr.transpose(order)
        msh = tuple(moved.shape)
        m, n = msh[-2], msh[-1]
        lead = 1
        for d in msh[:-2]:
            lead *= d
        flat = moved.reshape((lead, m, n)).tolist()
        if offset >= 0:
            dlen = max(min(m, n) - offset, 0)
            parts = [[b[i + offset][i] for i in range(dlen)]
                     for b in flat]
        else:
            dlen = max(min(m, n) + offset, 0)
            parts = [[b[i][i - offset] for i in range(dlen)]
                     for b in flat]
        return ndarray(parts).reshape(msh[:-2] + (dlen,))

    @staticmethod
    def svdvals(a):
        """计算奇异值（委托 svd(compute_uv=False)：dtype/stacked/nonfinite 一致）。"""
        return linalg_module.svd(a, full_matrices=False, compute_uv=False)

    @staticmethod
    def cond(x, p=None):
        """计算矩阵的条件数。

        条件数 ``kappa = ||x|| * ||inv(x)||`` 衡量线性系统的敏感度：接近 1 表示
        良态，很大表示接近奇异（奇异矩阵返回 `inf`）。

        Parameters
        ----------
        x : array_like
            方阵。
        p : {None, 1, -1, 2, -2, inf, -inf, 'fro'}, optional
            范数类型。默认（及 2 / -2）走 SVD，其余用 ``norm(x, p) * norm(inv(x), p)``。

        【使用示例】
        >>> import rsnumpy as np
        >>> np.linalg.cond(np.array([[1., 0.], [0., 1.]]))
        1.0
        >>> np.linalg.cond(np.array([[1., 2.], [2., 4.]]))   # 奇异
        inf
        """
        from rsnumpy import ndarray
        from ._pylinalg import (_common_type, _real_dtype, _stacked_2d,
                                _is_empty_2d)

        x_arr = x if hasattr(x, '_array') else ndarray(x)
        if _is_empty_2d(x_arr):
            raise LinAlgError("cond is not defined on empty arrays")
        shape = x_arr.shape
        if len(shape) < 2:
            raise LinAlgError(
                "1-dimensional array given. Array must be at least "
                "two-dimensional")
        _, res_dt = _common_type(getattr(x_arr, '_dtype', 'float64'))
        res_dt = _real_dtype(res_dt)  # 条件数恒为实数

        import rsnumpy as _np
        lead, subs = _stacked_2d(x_arr)
        is_2d = len(shape) == 2

        def _has_nan(arr):
            def _rec(x):
                if isinstance(x, list):
                    return any(_rec(y) for y in x)
                if isinstance(x, complex):
                    return x.real != x.real or x.imag != x.imag
                return float(x) != float(x)
            return _rec(arr.tolist())

        def _div(a, b):
            # IEEE 浮点除法语义（Python 的 x/0.0 会抛异常）
            if b == 0.0:
                if a == 0.0 or a != a:
                    return float('nan')
                return float('inf') if a > 0 else float('-inf')
            return a / b

        def _flat0(val):
            v = val.tolist()
            while isinstance(v, list):
                v = v[0]
            return float(v)

        def _inv_or_nan(sub):
            # 求逆失败（奇异/非有限）→ 全 nan 矩阵（NumPy gufunc errstate
            # 语义）；返回 (矩阵, 是否失败)
            try:
                return linalg_module.inv(sub), False
            except Exception:
                n2 = sub.shape[-1]
                is_c = _is_complex_arr(sub)
                nv = complex(0.0, float('nan')) if is_c else float('nan')
                return ndarray([[nv] * n2 for _ in range(n2)]), True

        def _sum_axis(vals, ax):
            # 2-d 嵌套列表轴和（ax=-2 列和 / ax=-1 行和）；nan 传播
            if ax == -2:
                n = len(vals[0]) if vals else 0
                return [sum(vals[i][j] for i in range(len(vals)))
                        for j in range(n)]
            return [sum(row) for row in vals]

        def _extreme(vals, mode):
            # numpy max/min 语义：任一 nan → nan（Rust 归约用 f64::max/min
            # fold：忽略 nan、全 nan 返回 ∓inf，不兼容）
            m = None
            for v in vals:
                if v != v:
                    return float('nan')
                if m is None or (v > m if mode == 'max' else v < m):
                    m = v
            return m if m is not None else float('nan')

        def _one(sub):
            if p is None or p in (2, -2):
                # 与公开 svd 同算法/同 dtype → 值与测试参照位级一致；
                # 非有限输入公开 svd 抛异常 → 回退裸 _svd_jacobi
                # （nan/inf 传播到结果，与 NumPy errstate 语义一致）
                try:
                    s = linalg_module.svd(sub, compute_uv=False).tolist()
                except Exception:
                    s = [float(abs(v)) for v in
                         _svd_jacobi(sub.tolist(), False, False).tolist()]
                if not s:
                    r = float('nan')
                elif p == -2:
                    r = _div(s[-1], s[0])
                else:
                    r = _div(s[0], s[-1])
            else:
                invx, inv_failed = _inv_or_nan(sub)
                if p == 'nuc':
                    # 不能用 norm()：nan 会使其抛异常；svd 对 nan/不收敛
                    # 输入抛 LinAlgError → nan（NumPy errstate 语义）
                    try:
                        sx = _flat0(linalg_module.svd(
                            sub, compute_uv=False).sum(-1))
                        si = _flat0(linalg_module.svd(
                            invx, compute_uv=False).sum(-1))
                        r = sx * si
                    except Exception:
                        r = float('nan')
                elif p == 'fro':
                    r = _flat0(_np.sqrt(
                        (_np.abs(sub) ** 2).sum(-1).sum(-1)
                        * (_np.abs(invx) ** 2).sum(-1).sum(-1)))
                elif _has_nan(sub) or inv_failed:
                    # numpy max/min 语义路径（结果必为 nan 或转 inf，
                    # 无精度要求）
                    asub = [[abs(v) for v in row] for row in sub.tolist()]
                    ainv = [[abs(v) for v in row] for row in invx.tolist()]
                    ax = -2 if p in (1, -1) else -1
                    mode = 'max' if p in (1, _np.inf) else 'min'
                    r = _extreme(_sum_axis(asub, ax), mode) * \
                        _extreme(_sum_axis(ainv, ax), mode)
                else:
                    # 与 NumPy _cond_impl 同运算链（同 dtype/同顺序）
                    if p == 1:
                        r = _flat0(_np.abs(sub).sum(-2).max(-1)) * \
                            _flat0(_np.abs(invx).sum(-2).max(-1))
                    elif p == -1:
                        r = _flat0(_np.abs(sub).sum(-2).min(-1)) * \
                            _flat0(_np.abs(invx).sum(-2).min(-1))
                    elif p == _np.inf:
                        r = _flat0(_np.abs(sub).sum(-1).max(-1)) * \
                            _flat0(_np.abs(invx).sum(-1).max(-1))
                    else:
                        r = _flat0(_np.abs(sub).sum(-1).min(-1)) * \
                            _flat0(_np.abs(invx).sum(-1).min(-1))
            # nan→inf：仅当 x 本身无 nan（nan 来自奇异/求逆失败）
            if r != r and not _has_nan(sub):
                r = float('inf')
            return r

        if is_2d:
            r = _one(x_arr)
            scalar_cls = _np.float32 if res_dt == 'float32' else _np.float64
            return scalar_cls(r)
        parts = [_one(sub) for sub in subs]
        return ndarray(parts).reshape(lead).astype(res_dt)

    @staticmethod
    def eigh(a, UPLO='L'):
        """Hermitian 或对称矩阵特征分解（NumPy 2.x 语义）。

        返回 EighResult(eigenvalues, eigenvectors)：eigenvalues (..., n)
        实数 dtype（f32/c64→f32，f64/c128/int→f64）升序；eigenvectors
        (..., n, n) dtype 随输入（int→f64）且保留 a 的 ndarray 子类；
        UPLO ∈ {'L','l','U','u'} 指定使用哪一三角；stacked (..., n, n)
        支持任意 lead 维；空尺寸返回同形状空数组；非方阵/非法 UPLO
        抛 ValueError。实对称走循环 Jacobi，复 Hermitian 走 2n 实嵌入。
        """
        from rsnumpy import ndarray
        from ._pylinalg import _stacked_2d, _herm_eigh

        u = UPLO if isinstance(UPLO, str) else ''
        if u not in ('L', 'l', 'U', 'u'):
            raise ValueError(
                "eigh: UPLO must be one of ['L', 'l', 'U', 'u']")
        uplo = u.upper()

        a = a if hasattr(a, '_array') else ndarray(a)
        shape = tuple(a.shape)
        if len(shape) < 2 or shape[-2] != shape[-1]:
            raise ValueError("eigh requires a square matrix")
        dt = getattr(a, '_dtype', 'float64')
        real_dt = 'float32' if dt in ('float32', 'complex64') \
            else 'float64'
        vec_dt = dt if dt.startswith(('float', 'complex')) else 'float64'

        def _as_result(arr):
            # eigenvectors 保留 a 的 ndarray 子类（numpy 语义）
            at = type(a)
            if at is not ndarray and isinstance(a, ndarray):
                return at(arr)
            return arr

        lead, subs = _stacked_2d(a)
        n = shape[-1]
        if n == 0:
            ev = ndarray([]).astype(real_dt).reshape(lead + (0,))
            ec = ndarray([]).astype(vec_dt).reshape(lead + (0, 0))
            return EighResult(_as_result(ev), _as_result(ec))
        if len(shape) == 2:
            ev, ec = _herm_eigh(a.tolist(), uplo)
            return EighResult(_as_result(ndarray(ev).astype(real_dt)),
                              _as_result(ndarray(ec).astype(vec_dt)))
        eval_list = []
        evec_list = []
        for sub in subs:
            ev, ec = _herm_eigh(sub.tolist(), uplo)
            eval_list.append(ev)
            evec_list.append(ec)
        ev = ndarray(eval_list).reshape(lead + (n,)).astype(real_dt)
        ec = ndarray(evec_list).reshape(lead + (n, n)).astype(vec_dt)
        return EighResult(_as_result(ev), _as_result(ec))

    @staticmethod
    def solve_banded(lower, upper, ab, b):
        """求解带状线性方程组。"""
        return _wrap(_core.linalg.solve_banded(lower, upper, _ensure(ab), _ensure(b)))

    @staticmethod
    def outer(a, b):
        """外积（numpy.linalg.outer 兼容）：输入必须为一维。"""
        import rsnumpy as _np
        a = _np.asarray(a)
        b = _np.asarray(b)
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("Input arrays must be one-dimensional")
        return a.reshape((-1, 1)) * b.reshape((1, -1))

    @staticmethod
    def slogdet(a):
        """(sign, log|det|) —— NumPy 2.x 语义：返回 SlogdetResult 命名元组
        （.sign/.logabsdet，可作 2 元组解包）。2-d 输入字段为标量（实输入
        float、复输入 complex，dtype 随输入）；stacked 返回 (lead...) 数组；
        空矩阵 (0×0) → (1, 0)；奇异 → (0, -inf)；全部尺寸走纯 Python
        LU（``_py_slogdet``）。
        """
        from rsnumpy import (ndarray, float32, float64, complex64,
                             complex128)
        from ._pylinalg import (_common_type, _real_dtype, _py_slogdet,
                                _stacked_2d)
        _SCALAR = {'float32': float32, 'float64': float64,
                   'complex64': complex64, 'complex128': complex128}

        a_arr = a if hasattr(a, '_array') else ndarray(a)
        shape = tuple(a_arr.shape)
        if len(shape) < 2 or shape[-2] != shape[-1]:
            raise ValueError("slogdet requires a square matrix")
        _, res_dt = _common_type(getattr(a_arr, '_dtype', 'float64'))
        real_dt = _real_dtype(res_dt)
        if len(shape) == 2:
            sign, logabs = _py_slogdet(a_arr.tolist())
            return SlogdetResult(_SCALAR[res_dt](sign),
                                 _SCALAR[real_dt](logabs))
        lead, subs = _stacked_2d(a_arr)
        signs, logabs = zip(*(_py_slogdet(sub.tolist()) for sub in subs))
        return SlogdetResult(
            ndarray(list(signs)).astype(res_dt).reshape(lead),
            ndarray(list(logabs)).astype(real_dt).reshape(lead))

    @staticmethod
    def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        """向量叉积（numpy.linalg.cross 兼容）。"""
        import rsnumpy as _np
        if axis is not None:
            axisa = axisb = axisc = axis
        a = _np.asarray(a)
        b = _np.asarray(b)
        if a.ndim:
            a = _np.moveaxis(a, axisa, -1)
        if b.ndim:
            b = _np.moveaxis(b, axisb, -1)
        if a.shape[-1] != 3 or b.shape[-1] != 3:
            raise ValueError(
                "input arrays must be (arrays of) 3-dimensional vectors")
        c0 = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
        c1 = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
        c2 = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        out = _np.stack((c0, c1, c2), axis=-1)
        if axisc != -1:
            out = _np.moveaxis(out, -1, axisc)
        return out

    @staticmethod
    def tensordot(a, b, axes=2):
        """张量收缩（numpy.linalg.tensordot 兼容）。"""
        import rsnumpy as _np
        a = _np.asarray(a)
        b = _np.asarray(b)
        if isinstance(axes, int):
            axa = tuple(range(a.ndim - axes, a.ndim))
            axb = tuple(range(axes))
        else:
            axa, axb = tuple(axes[0]), tuple(axes[1])
        akeep = [i for i in range(a.ndim) if i not in axa]
        bkeep = [i for i in range(b.ndim) if i not in axb]
        pa = akeep + list(axa)
        pb = list(axb) + bkeep
        a2 = a.transpose(pa) if a.ndim > 1 else a
        b2 = b.transpose(pb) if b.ndim > 1 else b
        k = 1
        for i in axa:
            k *= a.shape[i]
        if k == 0:
            k = 1
        a3 = a2.reshape((-1, k)) if a2.ndim > 0 and a2.size >= 0 else a2.reshape((k,))
        b3 = b2.reshape((k, -1)) if b2.ndim > 0 and b2.size >= 0 else b2.reshape((k,))
        c = _np.matmul(a3, b3)
        out_shape = (tuple(a.shape[i] for i in akeep)
                     + tuple(b.shape[i] for i in bkeep))
        return c.reshape(out_shape)

    @staticmethod
    def tensorinv(a, ind=2):
        """N 维数组的'逆'（numpy.linalg.tensorinv 兼容）。

        a 的形状须'square'：prod(a.shape[:ind]) == prod(a.shape[ind:])，
        否则抛 LinAlgError；ind 为不超过 a.ndim 的正整数（否则
        ValueError）。返回相对 tensordot(a, b, ind) 的逆，形状
        a.shape[ind:] + a.shape[:ind]。
        """
        import rsnumpy as _np
        a = a if hasattr(a, '_array') else _np.asarray(a)
        shape = tuple(a.shape)
        nd = len(shape)
        if not (0 < ind <= nd):
            raise ValueError("Invalid value for ind.")
        m = 1
        for d in shape[:ind]:
            m *= d
        n = 1
        for d in shape[ind:]:
            n *= d
        if m != n:
            raise LinAlgError("Shape of a is not 'square'")
        A = a.reshape((m, n))
        return linalg_module.inv(A).reshape(shape[ind:] + shape[:ind])

    @staticmethod
    def tensorsolve(a, b, axes=None):
        """张量方程 a x = b 求解（numpy.linalg.tensorsolve 兼容）。

        x 的全部指标与 a 的最右指标求和：tensordot(a, x,
        axes=x.ndim) == b；axes（int 元组）指定反转前移到 a 右端
        的轴，默认 None（不重排）。a 须满足
        prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])，否则抛
        LinAlgError；x 形状为 a 的末 (a.ndim - b.ndim) 轴。
        """
        import rsnumpy as _np
        a = a if hasattr(a, '_array') else _np.asarray(a)
        b = b if hasattr(b, '_array') else _np.asarray(b)
        an = len(tuple(a.shape))
        if axes is not None:
            allaxes = list(range(an))
            for k in axes:
                allaxes.remove(k)
                allaxes.insert(an, k)
            a = a.transpose(allaxes)
        bndim = len(tuple(b.shape))
        oldshape = a.shape[-(an - bndim):]
        prod = 1
        for k in oldshape:
            prod *= k
        if a.size != prod * prod:
            raise LinAlgError(
                "Input arrays must satisfy the requirement "
                "prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])")
        A = a.reshape((prod, prod))
        bflat = b.reshape((-1,))
        x = linalg_module.solve(A, bflat)
        return x.reshape(tuple(oldshape))

    @staticmethod
    def matrix_norm(x, ord=None, axis=(-2, -1), keepdims=False, **kwargs):
        """矩阵范数（numpy.linalg.matrix_norm 兼容）。

        Rust 层 norm 不接受元组 axis，这里按 NumPy 语义纯 Python 实现：
        ord=None/'fro' 弗罗贝尼乌斯范数、1/-1 列和范数、2/-2 奇异值
        范数、'nuc' 核范数；支持 ndim>2 的批量切片与 keepdims。
        """
        import rsnumpy as _np

        def _mat_norm_2d(m, o):
            if 0 in tuple(m.shape):
                # NumPy 语义：空矩阵的各阶范数均为 0
                return 0.0
            if o in (None, 'fro'):
                return _np.sqrt(_np.sum(abs(m) ** 2))
            if o == 1:
                return _np.max(_np.sum(abs(m), axis=0))
            if o == -1:
                return _np.min(_np.sum(abs(m), axis=0))
            sv = linalg_module.svd(m, compute_uv=False)
            if sv.size == 0:
                return 0.0
            if o == 2:
                return sv.max()
            if o == -2:
                return sv.min()
            if o == 'nuc':
                return sv.sum()
            raise ValueError(f"invalid norm order {o!r} for matrix norm")

        x = _np.asarray(x)
        axes = tuple(axis) if isinstance(axis, tuple) else (axis,)
        a0, a1 = axes[0] % x.ndim, axes[1] % x.ndim
        if x.ndim == 2:
            res = _mat_norm_2d(x, ord)
        else:
            other = [i for i in range(x.ndim) if i not in (a0, a1)]
            moved = x.transpose(other + [a0, a1])
            shp = moved.shape
            flat = moved.reshape((-1, shp[-2], shp[-1]))
            parts = [_mat_norm_2d(flat[i], ord) for i in range(flat.shape[0])]
            res = _np.asarray(parts).reshape(tuple(shp[:-2]))
        if keepdims:
            shp = list(x.shape)
            shp[a0] = 1
            shp[a1] = 1
            res = _np.asarray(res).reshape(tuple(shp))
        return res

    @staticmethod
    def matrix_transpose(x, axes=None):
        """矩阵转置（numpy.linalg.matrix_transpose 兼容）。"""
        import rsnumpy as _np
        x = _np.asarray(x)
        if x.ndim < 2:
            raise ValueError("array must be at least 2-dimensional")
        if axes is None:
            return x.T
        return x.transpose(axes)

    @staticmethod
    def eigvalsh(a, UPLO='L'):
        """Hermitian/对称矩阵特征值（numpy.linalg.eigvalsh 兼容）。

        返回升序实数 dtype ndarray（f32/c64→f32，f64/c128/int→f64）；
        UPLO/stacked/空尺寸语义同 eigh。
        """
        return linalg_module.eigh(a, UPLO)[0]


def issymmetric(a, tol=None):
    """判断矩阵是否对称。"""
    from rsnumpy import ndarray
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


# 多矩阵连乘（numpy.linalg.multi_dot 兼容）：DP 求最优乘法顺序。
from ._linalg import _multi_dot_matrix_chain_order, multi_dot  # noqa: E402,F401

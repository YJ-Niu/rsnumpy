# SPDX-License-Identifier: MIT
"""纯 Python linalg 回退算法与共享 helper（NumPy 语义对照实现）。

Rust 核心只覆盖小方阵（2×2/3×3）；本模块提供通用形状算法：
Cholesky、LU 行列式、快速矩阵幂、矩阵范数，以及 dtype 规则 /
nonfinite 检查 / stacked 切片等共享 helper。
所有矩阵运算用 Python 浮点（f64）完成，构造结果时再 astype 回
输入对应的输出 dtype（与 NumPy _commonType 规则一致，精度只升不降）。
"""

import cmath
import math

__all__ = [
    'NoValue',
    '_common_type',
    '_result_dtype',
    '_real_dtype',
    '_is_inexact',
    '_nonfinite_check',
    '_stacked_2d',
    '_is_empty_2d',
    '_matmul',
    '_matrix_power_mat',
    '_py_cholesky',
    '_py_det',
    '_py_slogdet',
    '_eigvals_qr',
    '_mat_norm_2d',
    '_finfo_eps',
]


class _NoValue:
    """未提供参数的哨兵（对应 NumPy 的 _NoValue）。"""

    def __repr__(self):
        return '<no value>'


NoValue = _NoValue()


def _finfo_eps(dt):
    """float/complex dtype 对应的机器精度 eps。"""
    if dt in ('float32', 'complex64'):
        return 1.1920928955078125e-07
    if dt in ('float16',):
        return 9.765625e-04
    return 2.220446049250313e-16


def _real_dtype(dt):
    """complex → 对应实 dtype；其余原样返回。"""
    if dt == 'complex64':
        return 'float32'
    if dt == 'complex128':
        return 'float64'
    return dt


def _is_inexact(dt):
    return dt in ('float16', 'float32', 'float64', 'complex64', 'complex128')


def _common_type(a_dtype):
    """(计算 dtype, 返回 dtype)——NumPy _commonType 语义。

    inexact（除 float16）保持原 dtype；float16 → TypeError；
    int/bool/其它 → float64。
    """
    if a_dtype == 'float16':
        raise TypeError("unsupported in linalg.")
    if a_dtype in ('float32', 'float64', 'complex64', 'complex128'):
        return (a_dtype, a_dtype)
    return ('float64', 'float64')


def _result_dtype(a_dtype):
    return _common_type(a_dtype)[1]


def _nonfinite_check(mat, msg="SVD did not converge"):
    """嵌套矩阵含 inf/nan 时抛 LinAlgError（模拟 LAPACK 不收敛）。"""
    from rsnumpy.linalg import LinAlgError

    def scan(v):
        if isinstance(v, (list, tuple)):
            for x in v:
                scan(x)
        else:
            if isinstance(v, complex):
                r, i = v.real, v.imag
            else:
                r, i = float(v), 0.0
            if r != r or i != i or math.isinf(r) or math.isinf(i):
                raise LinAlgError(msg)

    scan(mat)


def _stacked_2d(arr):
    """ndarray (...,M,N) → (lead_shape, [2-d 子数组, ...])。2-d 时 lead=(())。

    注意：不能用 reshape((-1, M, N))——size-0 时 rsnumpy 把 -1 解成 0，
    会丢掉 lead 维（(3,0,5) 变 (0,0,5)）；这里显式算 total。
    """
    shape = arr.shape
    if len(shape) == 2:
        return ((), arr)
    lead = shape[:-2]
    total = 1
    for d in lead:
        total *= d
    last2 = shape[-2:]
    flat = arr.reshape((total,) + last2)
    return (lead, [flat[i] for i in range(total)])


def _is_empty_2d(arr):
    return arr.size == 0 and 0 in arr.shape[-2:]


def _matmul(A, B):
    """嵌套列表矩阵乘 (m×n)·(n×p)，元素可为 int/float/complex/object。"""
    m = len(A)
    n = len(A[0]) if m else 0
    p = len(B[0]) if B and B[0] else 0
    out = [[0] * p for _ in range(m)]
    for i in range(m):
        row_a = A[i]
        row_o = out[i]
        for k in range(n):
            aik = row_a[k]
            if aik == 0:
                continue
            row_b = B[k]
            for j in range(p):
                row_o[j] += aik * row_b[j]
    return out


def _matrix_power_mat(A, n):
    """嵌套列表方阵快速幂（二进制分解），n >= 0（0 → 单位矩阵）。"""
    if n == 0:
        m = len(A)
        return [[1 if i == j else 0 for j in range(m)] for i in range(m)]
    if n == 1:
        return [list(row) for row in A]
    if n == 2:
        return _matmul(A, A)
    if n == 3:
        return _matmul(_matmul(A, A), A)
    z = None
    result = None
    while n > 0:
        z = [list(row) for row in A] if z is None else _matmul(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else _matmul(result, z)
    return result


def _qr_raw(mat):
    """Householder QR（LAPACK geqrf 约定）。

    返回 (h, tau)：h 为 (n, m)（FORTRAN 序，即行主序 (m, n) 的转置：
    上三角 = R，第 j 列下三角 = Householder 向量 v_j/r1j_j），tau (k,)
    （k = min(m, n)）。重建：x' = x - tau_j * w_j * (w_j^H x)，
    w_j = [1, h(j+1:m, j)]^T。
    """
    m = len(mat)
    n = len(mat[0]) if m else 0
    k = min(m, n)
    is_c = any(isinstance(v, complex) for row in mat for v in row)
    H = [list(row) for row in mat]  # (m, n) 行主序工作矩阵
    zero = (0.0 + 0j) if is_c else 0.0
    tau = [zero] * k
    for j in range(k):
        col = [H[i][j] for i in range(j, m)]
        norm = math.sqrt(sum(abs(x) ** 2 for x in col))
        if norm == 0:
            continue
        x0 = col[0]
        if is_c:
            alpha = -(x0 / abs(x0)) * norm if x0 != 0 else -norm
        else:
            alpha = -norm if x0 >= 0 else norm
        r1j = x0 - alpha
        if r1j == 0:
            continue
        if is_c:
            tau[j] = (r1j.conjugate() * alpha
                      / (abs(alpha) ** 2 + abs(r1j) ** 2)) / r1j
        else:
            tau[j] = (r1j * alpha / (alpha * alpha + r1j * r1j)) / r1j
        for i in range(j + 1, m):
            H[i][j] = mat[i][j] / r1j
    # FORTRAN 序：转置为 (n, m)
    h = [[H[i][j] for i in range(m)] for j in range(n)]
    return h, tau


def _py_cholesky(mat, upper=False):
    """Cholesky-Crout 分解：A = L·L^H（lower）或 U^H·U（upper）。

    只用下三角（lower）或上三角（upper）+对角；非正定抛 LinAlgError。
    返回嵌套列表（与输入同 complex/float 性）。
    """
    from rsnumpy.linalg import LinAlgError

    n = len(mat)
    if upper:
        if n == 0:
            return []
        is_c = any(isinstance(v, complex) for row in mat for v in row)
        # 对 A^H 做下三角分解再共轭转置（等价于用 A 的上三角）。
        # A^H[i][j] = conj(A[j][i])——必须取共轭，否则复数矩阵分解错误。
        a_h = [[mat[j][i].conjugate() if is_c else mat[j][i]
                for j in range(n)] for i in range(n)]
        l = _py_cholesky(a_h, False)
        return [[l[j][i].conjugate() if is_c else l[j][i]
                 for j in range(n)] for i in range(n)]

    if n == 0:
        return []
    is_c = any(isinstance(v, complex) for row in mat for v in row)
    zero = 0j if is_c else 0.0

    def _cj(z):
        return z.conjugate() if is_c else z

    L = [[zero] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            # A = L·L^H 约定：A[i][j] = Σ_k L[i][k]·conj(L[j][k])
            s = zero
            for k in range(j):
                s += L[i][k] * _cj(L[j][k])
            if i == j:
                # s = Σ|L[i][k]|²（j=i 时两种约定一致）
                d = mat[i][i] - s
                if is_c:
                    if d.real <= 0:
                        raise LinAlgError(
                            "matrix is not positive-definite (cholesky)")
                    L[i][i] = cmath.sqrt(complex(d))
                else:
                    if d <= 0:
                        raise LinAlgError(
                            "matrix is not positive-definite (cholesky)")
                    L[i][i] = math.sqrt(d)
            else:
                L[i][j] = (mat[i][j] - s) / _cj(L[j][j])
    return L


def _py_det(mat):
    """LU 分解（列主元）求行列式，complex-aware。返回 float/complex 标量。

    空矩阵（0×0）行列式为 1。
    """
    n = len(mat)
    if n == 0:
        return 1.0
    is_c = any(isinstance(v, complex) for row in mat for v in row)
    w = [list(row) for row in mat]
    sign = 1.0
    det = (1.0 + 0j) if is_c else 1.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(w[r][col]))
        if abs(w[pivot][col]) == 0:
            return (0.0 + 0j) if is_c else 0.0
        if pivot != col:
            w[col], w[pivot] = w[pivot], w[col]
            sign = -sign
        det *= w[col][col]
        for r in range(col + 1, n):
            if w[r][col] == 0:
                continue
            f = w[r][col] / w[col][col]
            wr = w[r]
            wc = w[col]
            for c2 in range(col, n):
                wr[c2] -= f * wc[c2]
    return sign * det


def _py_slogdet(mat):
    """LU 分解（列主元）→ (sign, log|det|)。

    实矩阵 sign 为 ±1（奇异为 0.0），复矩阵 sign 为相位（模 1 复数，
    奇异为 0j）；奇异 → logabs 为 -inf；空矩阵 (0×0) → (1, 0)。
    """
    n = len(mat)
    if n == 0:
        return 1.0, 0.0
    is_c = any(isinstance(v, complex) for row in mat for v in row)
    w = [list(row) for row in mat]
    sign = (1.0 + 0j) if is_c else 1.0
    logabs = 0.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(w[r][col]))
        if abs(w[pivot][col]) == 0:
            return (0.0 + 0j) if is_c else 0.0, float('-inf')
        if pivot != col:
            w[col], w[pivot] = w[pivot], w[col]
            sign = -sign
        p = w[col][col]
        mag = abs(p)
        logabs += math.log(mag)
        if is_c:
            sign *= p / mag
        elif p < 0:
            sign = -sign  # 实枢轴符号必须计入——只数行交换会漏掉
        for r in range(col + 1, n):
            if w[r][col] == 0:
                continue
            f = w[r][col] / p
            wr = w[r]
            wc = w[col]
            for c2 in range(col, n):
                wr[c2] -= f * wc[c2]
    return sign, logabs


def _hessenberg_reduce(mat):
    """复数 Householder 相似约化到上 Hessenberg 形式（特征值不变）。"""
    n = len(mat)
    A = [[complex(v) for v in row] for row in mat]
    for j in range(n - 2):
        x = [A[r][j] for r in range(j + 1, n)]
        norm_x = math.sqrt(sum(abs(z) ** 2 for z in x))
        if norm_x == 0:
            continue
        sgn = x[0] / abs(x[0]) if abs(x[0]) > 0 else 1.0
        v = [x[0] + sgn * norm_x] + x[1:]
        nv = math.sqrt(sum(abs(z) ** 2 for z in v))
        if nv == 0:
            continue
        v = [z / nv for z in v]
        lo = j + 1
        # 左乘 P = I - 2vv^H：行 lo..n-1 × 列 j..n-1（前列下段已为零，
        # 限制列范围避免浮点垃圾重新渗入）。w = v^H A[:, j..] 先从
        # 未修改的 A 完整算出——A[r][c] -= 2v[r]w[c] 各行无原地依赖。
        w = [sum(v[t - lo].conjugate() * A[t][c]
                 for t in range(lo, n)) for c in range(j, n)]
        for r in range(lo, n):
            vr = v[r - lo]
            for c in range(j, n):
                A[r][c] -= 2 * vr * w[c - j]
        # 右乘 P：(AP)[r][c] = A[r][c] - 2(Av)[r] conj(v[c])
        avs = [sum(A[r][t] * v[t - lo]
                   for t in range(lo, n)) for r in range(n)]
        for c in range(lo, n):
            vc = v[c - lo].conjugate()
            for r in range(n):
                A[r][c] -= 2 * avs[r] * vc
    return A


def _qr_factor(H, i, m, q):
    """H[i:m, i:m] - qI 的 Householder QR 分解，返回 (Q, R)（B = QR）。"""
    k = m - i
    B = [[H[r][c] - (q if r == c else 0j) for c in range(i, m)]
         for r in range(i, m)]
    R = [row[:] for row in B]
    Q = [[(1.0 + 0j) if r == c else 0j for c in range(k)] for r in range(k)]
    for j in range(k - 1):
        x = [R[r][j] for r in range(j, k)]
        norm_x = math.sqrt(sum(abs(z) ** 2 for z in x))
        if norm_x == 0:
            continue
        sgn = x[0] / abs(x[0]) if abs(x[0]) > 0 else 1.0
        v = [x[0] + sgn * norm_x] + x[1:]
        nv = math.sqrt(sum(abs(z) ** 2 for z in v))
        if nv == 0:
            continue
        v = [z / nv for z in v]
        # R <- P R（P = I - 2vv^H，作用于第 j..k-1 行/列）。w = v^H R[:, j..]
        # 先从未修改的 R 完整算出，避免行方向原地覆盖。
        w = [sum(v[t - j].conjugate() * R[t][c]
                 for t in range(j, k)) for c in range(j, k)]
        for r in range(j, k):
            vr = v[r - j]
            for c in range(j, k):
                R[r][c] -= 2 * vr * w[c - j]
        # Q <- Q P：(QP)[r][c] = Q[r][c] - 2(Qv)[r] conj(v[c])
        for r in range(k):
            qv = sum(Q[r][t] * v[t - j] for t in range(j, k))
            if qv == 0:
                continue
            for c in range(j, k):
                Q[r][c] -= 2 * qv * v[c - j].conjugate()
    return Q, R


def _qr_shift_step(H, i, m, q):
    """对 H[i:m, i:m] 块做一次单位移（q）QR 步骤，原地回写。

    B = block - qI = QR → block' = RQ + qI（相似变换）。
    """
    k = m - i
    Q, R = _qr_factor(H, i, m, q)
    for r in range(k):
        for c in range(k):
            H[i + r][i + c] = sum(R[r][t] * Q[t][c] for t in range(k)) \
                + (q if r == c else 0j)


def _qr_double_step(H, i, m):
    """Francis 双移位步：以尾部 2x2 块的两个特征值 p、q 为连续两次
    单位移 QR 的移位（显式双移位），三次收敛。要求 m - i >= 3。"""
    import cmath
    a, b = H[m - 2][m - 2], H[m - 2][m - 1]
    c, d = H[m - 1][m - 2], H[m - 1][m - 1]
    tr = a + d
    detv = a * d - b * c
    disc = cmath.sqrt(tr * tr - 4.0 * detv)
    _qr_shift_step(H, i, m, (tr + disc) / 2.0)
    _qr_shift_step(H, i, m, (tr - disc) / 2.0)


def _eigvals_fl_dk(mat):
    """特征多项式法（f64 FL + DK 多起点 + c0 校正 + Newton 抛光）。

    f64 FL 系数累积 ~1e-11 相对误差，常数项 c[0] 直接决定根乘积
    （=det），故把 c[0] 校正为 f64 LU 行列式（相对 ~1e-15）得校正
    多项式 p̃'，再对每个 DK 候选根做 3 步 Newton 抛光。n≤8 可靠；
    n≥9 系数量级（~‖A‖^n）超出 f64 有效位，根不可靠。
    """
    import cmath
    n = len(mat)
    c = _fl_coefficients(mat)
    roots = _dk_roots(c)
    target = ((-1.0) ** n) * _py_det(mat)
    dc = target - c[0]
    for i, z in enumerate(roots):
        for _ in range(3):
            f = _poly_eval(c, z) + dc
            dp = _poly_deriv(c, z)
            if dp == 0:
                break
            st = f / dp
            z -= st
            if st == 0:
                break
        roots[i] = z
    return roots


def _cgivens(a, b):
    """复 Givens：返回 (c, s, r)，G = [[c, s], [-conj(s), conj(c)]]，
    G^H [a, b]^T = [r, 0]^T（r ≥ 0 实）。c = a/t, s = -conj(b)/t。"""
    import math
    t = math.sqrt(abs(a) ** 2 + abs(b) ** 2)
    if t == 0:
        return 1.0 + 0j, 0j, 0.0
    return a / t, -complex(b).conjugate() / t, t


def _rot_rows(H, r1, r2, jlo, jhi, c, s):
    """左乘 G^H 于行 r1, r2（列 jlo..jhi）：new1 = conj(c)·x - s·y，
    new2 = conj(s)·x + c·y。"""
    ca = c.conjugate()
    cs = s.conjugate()
    for j in range(jlo, jhi + 1):
        x = H[r1][j]
        y = H[r2][j]
        H[r1][j] = ca * x - s * y
        H[r2][j] = cs * x + c * y


def _rot_cols(H, c1, c2, rlo, rhi, c, s):
    """右乘 G 于列 c1, c2（行 rlo..rhi）：new1 = c·x - conj(s)·y，
    new2 = s·x + conj(c)·y。"""
    cs = s.conjugate()
    cc = c.conjugate()
    for r in range(rlo, rhi + 1):
        x = H[r][c1]
        y = H[r][c2]
        H[r][c1] = c * x - cs * y
        H[r][c2] = s * x + cc * y


def _francis_step(H, lo, hi, exc=False):
    """窗口 [lo, hi]（含端点）上的隐式 Francis 双移位 QR 步
    （bulge chasing，复 Givens；数值稳定，不显式构造奇异因子分解）。
    exc=True 用异常移位（dlahqr 对策：每 10 步无收缩扰动一次，
    打破移位与特征值重合导致的停滞）。"""
    import cmath
    if exc:
        # LAPACK 式异常移位：0.75×次对角范数 + 尾对角
        w1 = w2 = H[hi][hi] + 0.75 * (abs(H[hi][hi - 1])
                                      + abs(H[hi - 1][hi - 2]))
    else:
        a, b = H[hi - 1][hi - 1], H[hi - 1][hi]
        c, d = H[hi][hi - 1], H[hi][hi]
        tr = a + d
        disc = cmath.sqrt(tr * tr - 4.0 * (a * d - b * c))
        w1 = (tr + disc) / 2.0
        w2 = (tr - disc) / 2.0
    # 初始 chase 向量 u = (H-w1·I)(H-w2·I) e_lo 的前两个非零分量
    u0 = (H[lo][lo] - w1) * (H[lo][lo] - w2) \
        + H[lo][lo + 1] * H[lo + 1][lo]
    u1 = H[lo + 1][lo] * ((H[lo][lo] - w2) + (H[lo + 1][lo + 1] - w1))
    c0, s0, _ = _cgivens(u0, u1)
    _rot_rows(H, lo, lo + 1, lo, hi, c0, s0)
    _rot_cols(H, lo, lo + 1, lo, min(lo + 2, hi), c0, s0)
    # 追逐 bulge：第 t 步 bulge 在 (lo+2+t, lo+t)
    for t in range(hi - lo - 1):
        r1 = lo + 1 + t
        r2 = lo + 2 + t
        x = H[r1][lo + t]
        y = H[r2][lo + t]
        ck, sk, _ = _cgivens(x, y)
        # 左旋须含 bulge 列 lo+t：x→r、y→0（漏掉该列会破坏相似性）
        _rot_rows(H, r1, r2, lo + t, hi, ck, sk)
        H[r2][lo + t] = 0j  # 数值保险
        _rot_cols(H, r1, r2, lo, min(r2 + 1, hi), ck, sk)


def _qr_iter_eigvals(H):
    """Hessenberg 矩阵特征值（dlahqr 式驱动）：窗口递归 + 尾块 1x1/2x2
    正确提取 + 隐式 Francis 双移位。H[i+1][i]≈0 时矩阵分裂为两个独立
    块（尾块递归，前导块继续），绝不可把未三角化的尾块当对角读取。"""
    import cmath
    n = len(H)
    eps = 2.220446049250313e-16
    evals = []
    state = {'it': 0}

    def _2x2(lo):
        a, b = H[lo][lo], H[lo][lo + 1]
        c, d = H[lo + 1][lo], H[lo + 1][lo + 1]
        tr = a + d
        disc = cmath.sqrt(tr * tr - 4.0 * (a * d - b * c))
        return (tr + disc) / 2.0, (tr - disc) / 2.0

    def _window(lo, hi):
        # stall：本窗口连续无收缩的步数（每 10 步触发一次异常移位）
        stall = 0
        while hi - lo + 1 > 0:
            size = hi - lo + 1
            if size == 1:
                evals.append(complex(H[lo][lo]))
                return
            if size == 2:
                e1, e2 = _2x2(lo)
                evals.append(e1)
                evals.append(e2)
                return
            # 从底部向上找第一个可收缩次对角元 H[j+1][j]
            split = None
            for j in range(hi - 1, lo - 1, -1):
                if H[j + 1][j] == 0:
                    split = j
                    break
                if abs(H[j + 1][j]) <= eps * (abs(H[j][j])
                                              + abs(H[j + 1][j + 1])):
                    H[j + 1][j] = 0j
                    split = j
                    break
            if split is not None:
                stall = 0
                tsize = hi - split  # 尾块 [split+1, hi] 大小
                if tsize == 1:
                    evals.append(complex(H[hi][hi]))
                    hi = split
                    continue
                if tsize == 2:
                    e1, e2 = _2x2(split + 1)
                    evals.append(e1)
                    evals.append(e2)
                    hi = split
                    continue
                # 尾块 ≥3：仍是独立 Hessenberg 块，递归处理后再回头
                _window(split + 1, hi)
                hi = split
                continue
            # 显式双移位 QR 步（两次单位移 QR，移位取尾部 2x2 特征值）；
            # 每 10 步无收缩改用异常移位（dlahqr 对策）
            if stall % 10 == 9:
                _qr_shift_step(H, lo, hi + 1,
                               H[hi][hi] + 0.75 * (abs(H[hi][hi - 1])
                                                   + abs(H[hi - 1][hi - 2])))
            else:
                _qr_double_step(H, lo, hi + 1)
            stall += 1
            state['it'] += 1
            if state['it'] > 60 * n:
                raise RuntimeError(
                    'QR eigenvalue iteration did not converge')

    _window(0, n - 1)
    # 实输入的实特征值：复算术残留的微小虚部归零
    out = []
    for z in evals:
        if abs(z.imag) < 1e-11 * max(1.0, abs(z.real)):
            z = complex(z.real, 0.0)
        out.append(z)
    return out


def _eigvals_qr(mat):
    """通用特征值（纯 Python），返回 complex 列表（顺序任意，同 LAPACK 语义）。

    n≤2 解析式；n≤8 特征多项式法（FL+DK+c0 校正，精度 ~1e-15 相对）；
    n≥9 FL 系数量级超出 f64 有效位 → Hessenberg + Francis 双移位 QR
    迭代（不收敛回退 FL+DK，保持旧行为下界）。
    """
    import cmath
    n = len(mat)
    if n == 0:
        return []
    if n == 1:
        return [complex(mat[0][0])]
    if n == 2:
        a, b = mat[0][0], mat[0][1]
        c, d = mat[1][0], mat[1][1]
        tr = a + d
        det = a * d - b * c
        disc = cmath.sqrt(tr * tr - 4.0 * det)
        return [(tr + disc) / 2.0, (tr - disc) / 2.0]
    if n >= 9:
        H = _hessenberg_reduce(mat)
        try:
            return _qr_iter_eigvals(H)
        except Exception:
            pass  # 不收敛：回退 FL+DK（旧路径）
    return _eigvals_fl_dk(mat)


def _eigvals_highprec(mat):
    """通用 n×n 特征值：高精度 FL 系数 + f64 DK 候选 + 高精度 Newton。

    复数 Decimal 用 (re, im) 对表示；FL 递推 B_{k+1} = B_k·A + c_{n-k}·I，
    c_{n-k} = −tr(B_k·A)/k（复矩阵迹可为复数）。
    """
    from decimal import Decimal, localcontext
    n = len(mat)
    D = Decimal
    TINY = D(10) ** -30

    def _cadd(x, y):
        return (x[0] + y[0], x[1] + y[1])

    def _cmul(x, y):
        return (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0])

    def _to_cd(v):
        if isinstance(v, complex):
            return (D(repr(v.real)), D(repr(v.imag)))
        return (D(repr(v)), D(0))

    def _peval(c, zr, zi):
        re, im = c[-1]
        for k in range(len(c) - 2, -1, -1):
            cr, ci = c[k]
            re, im = re * zr - im * zi + cr, re * zi + im * zr + ci
        return re, im

    def _pderiv(c, zr, zi):
        re, im = D(0), D(0)
        for k in range(len(c) - 1, 0, -1):
            cr, ci = c[k]
            re, im = (re * zr - im * zi + k * cr,
                      re * zi + im * zr + k * ci)
        return re, im

    with localcontext() as ctx:
        ctx.prec = 45
        A = [[_to_cd(v) for v in row] for row in mat]
        c = [(D(0), D(0))] * (n + 1)
        c[n] = (D(1), D(0))
        B = [[(D(1) if i == j else D(0), D(0)) for j in range(n)]
             for i in range(n)]
        for k in range(1, n + 1):
            BA = [[(D(0), D(0)) for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    s = (D(0), D(0))
                    for t in range(n):
                        s = _cadd(s, _cmul(B[i][t], A[t][j]))
                    BA[i][j] = s
            tre = D(0)
            tim = D(0)
            for i in range(n):
                tre += BA[i][i][0]
                tim += BA[i][i][1]
            c[n - k] = (-tre / k, -tim / k)
            for i in range(n):
                BA[i][i] = _cadd(BA[i][i], c[n - k])
            B = BA
        # f64 舍入系数 → DK 候选（~1e-11 精度，落 Newton 收敛域内）
        cf = [complex(float(c[k][0]), float(c[k][1])) for k in range(n + 1)]
        approx = _dk_roots(cf)
        roots = []
        for z in approx:
            zr = D(repr(z.real))
            zi = D(repr(z.imag))
            for _ in range(100):
                fr, fi = _peval(c, zr, zi)
                dr, di = _pderiv(c, zr, zi)
                den = dr * dr + di * di
                if den == 0:
                    break
                sr = (fr * dr + fi * di) / den
                si = (fi * dr - fr * di) / den
                zr -= sr
                zi -= si
                if max(abs(sr), abs(si)) < TINY:
                    break
            roots.append(complex(float(zr), float(zi)))
    return roots


def _fl_coefficients(mat):
    """Faddeev–LeVerrier：p(λ)=det(λI−A)=Σ c[k]λ^k 的系数（c 长 n+1，c[n]=1）。

    递推：B_1 = I，c_{n-k} = −tr(B_k·A)/k，B_{k+1} = B_k·A + c_{n-k}·I。
    （迹必须取 B_k·A——直接取 tr(B_k) 会双重计数，系数全错。）
    """
    n = len(mat)
    c = [0.0j] * (n + 1)
    c[n] = 1.0
    B = [[1.0 + 0j if i == j else 0.0j for j in range(n)] for i in range(n)]
    for k in range(1, n + 1):
        BA = [[sum(B[i][t] * mat[t][j] for t in range(n))
               for j in range(n)] for i in range(n)]
        c[n - k] = -sum(BA[i][i] for i in range(n)) / k
        for i in range(n):
            BA[i][i] += c[n - k]
        B = BA
    return c


def _poly_eval(coeffs, z):
    """Horner 求 Σ coeffs[k]·z^k（必须从最高次项起迭代——
    从 c[0] 正向迭代会得到倒数多项式 z^d·p(1/z)）。"""
    r = coeffs[-1]
    for k in range(len(coeffs) - 2, -1, -1):
        r = r * z + coeffs[k]
    return r


def _poly_deriv(coeffs, z):
    """Horner 求导数 Σ k·coeffs[k]·z^(k−1)。"""
    r = 0.0j
    for k in range(len(coeffs) - 1, 0, -1):
        r = r * z + k * coeffs[k]
    return r


def _dk_roots(c, dk_iters=60):
    """求 p(λ)=Σ c[k]λ^k 的全体根：逐轮“多起点 DK + Newton 验证 + 降阶”。

    每轮用多组初值相位做 DK 同步迭代收集候选点，按牛顿步长
    |p(z)|/|p'(z)| 评分，对前列候选逐个 Newton 抛光并验证（收敛且
    未逃逸），接受第一个验证通过的根；仅用已验证根做合成除法降阶
    ——绝不接受伪根，避免伪根降阶后的 NaN 级联。
    """
    n = len(c) - 1
    if n == 0:
        return []
    mono = [z / c[n] for z in c]
    roots = []
    cur = mono
    deg = n
    while deg > 0:
        if deg == 1:
            roots.append(complex(-cur[0]))
            break
        R = 1.0 + max(abs(z) for z in cur[:-1])
        cands = []
        for phase in (0.35, 1.7, 3.05, 4.3, 5.5, 0.9):
            w = [cmath.rect(0.7 * R, 2.0 * math.pi * i / deg + phase)
                 for i in range(deg)]
            for _ in range(dk_iters):
                new = [0.0j] * deg
                for i in range(deg):
                    den = 1.0 + 0.0j
                    for j in range(deg):
                        if j != i:
                            den *= w[i] - w[j]
                    step = (_poly_eval(cur, w[i]) / den
                            if abs(den) > 1e-300 else 0.0j)
                    new[i] = w[i] - step
                w = new
                if max(abs(z) for z in w) > 100.0 * R:
                    break  # 逃逸 → 该组初值作废
            cands.extend(w)
        scored = []
        for z in cands:
            if z != z:
                continue
            dp = _poly_deriv(cur, z)
            s = (abs(_poly_eval(cur, z)) / abs(dp)) if dp != 0 \
                else float('inf')
            scored.append((s, z))
        scored.sort(key=lambda t: t[0])
        accepted = None
        for _s, z in scored[:8]:
            r = z
            ok = False
            for _ in range(200):
                f = _poly_eval(cur, r)
                dp = _poly_deriv(cur, r)
                if dp == 0 or r != r or abs(r) > 100.0 * R:
                    break
                st = f / dp
                r -= st
                if abs(st) < 1e-13:
                    ok = True
                    break
            if ok:
                accepted = r
                break
        if accepted is None:
            # 兜底：取评分最低候选（退化模式，随机测试矩阵不会走到）
            accepted = scored[0][1] if scored else 0.0j
        roots.append(complex(accepted))
        # 合成除法降阶：cur = (z − r)·q
        q = [0.0j] * deg
        q[deg - 1] = 1.0
        for k in range(deg - 2, -1, -1):
            q[k] = cur[k + 1] + r * q[k + 1]
        cur = q
        deg -= 1
    return roots


def _solve_pivot(A, b):
    """列主元高斯消元解 Ax = b（奇异矩阵用最大可用枢轴继续，不抛异常）。"""
    n = len(A)
    work = [list(row) for row in A]
    rhs = list(b)
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(work[r][col]))
        if p != col:
            work[col], work[p] = work[p], work[col]
            rhs[col], rhs[p] = rhs[p], rhs[col]
        pv = work[col][col]
        if pv == 0:
            continue
        for r in range(col + 1, n):
            f = work[r][col] / pv
            if f != 0:
                for c2 in range(col, n):
                    work[r][c2] -= f * work[col][c2]
                rhs[r] -= f * rhs[col]
    x = [0.0j] * n
    for r in range(n - 1, -1, -1):
        pv = work[r][r]
        if pv == 0:
            continue
        s = rhs[r]
        for c2 in range(r + 1, n):
            s -= work[r][c2] * x[c2]
        x[r] = s / pv
    return x


def _eigenvectors_general(mat, evals):
    """通用 n×n 特征向量（正规方程 + 列主元高斯消元）。

    解 (A−λI)^H(A−λI) v = e：λ 处该矩阵近奇异，解的方向即零空间
    = 特征向量；若 e 与零空间近正交（解范数过小）换下一坐标方向。
    返回行主序嵌套 list [row][eig_index]（与小矩阵路径格式一致），
    第 i 个特征向量对应 evals[i]（与 eigvals 顺序一致）。
    """
    n = len(mat)
    evecs_cols = []
    for lam in evals:
        B = [[0.0j] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0j
                for k in range(n):
                    aik = mat[k][i] - (lam if k == i else 0.0)
                    akj = mat[k][j] - (lam if k == j else 0.0)
                    s += aik.conjugate() * akj
                B[i][j] = s
        v = None
        for t in range(n):
            b = [0.0j] * n
            b[t] = 1.0
            cand = _solve_pivot(B, b)
            nrm = math.sqrt(sum(abs(z) ** 2 for z in cand))
            if nrm > 1e-8:
                v = [z / nrm for z in cand]
                break
        if v is None:
            v = [1.0 + 0.0j if i == 0 else 0.0j for i in range(n)]
        evecs_cols.append(v)
    return [[col[i] for col in evecs_cols] for i in range(n)]


def _mat_norm_2d(mat, p):
    """2-d 嵌套矩阵范数（p: None/'fro'/1/-1/inf/-inf）。complex-aware。"""
    m = len(mat)
    n = len(mat[0]) if m else 0
    if m == 0 or n == 0:
        return 0.0
    if p in (None, 'fro'):
        return math.sqrt(sum(abs(v) ** 2 for row in mat for v in row))
    if p == 1:
        return max(sum(abs(row[j]) for row in mat) for j in range(n))
    if p == -1:
        return min(sum(abs(row[j]) for row in mat) for j in range(n))
    if p in (float('inf'), 'inf'):
        return max(sum(abs(v) for v in row) for row in mat)
    if p in (float('-inf'), '-inf'):
        return min(sum(abs(v) for v in row) for row in mat)
    raise ValueError(f"invalid norm order {p!r} for matrix norm")


def _jacobi_eigh_real(a):
    """实对称 n×n 矩阵的循环 Jacobi 特征分解（确定性、重特征值稳健）。

    返回 (特征值升序列表, 特征向量 n×n 嵌套列表)——V[i][j] 为第 j 个
    特征向量的第 i 个分量（列正交）。
    """
    n = len(a)
    A = [[float(v) for v in row] for row in a]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _sweep in range(100):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += A[p][q] * A[p][q]
        if off < 1e-300:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = A[p][q]
                if abs(apq) < 1e-300:
                    continue
                app, aqq = A[p][p], A[q][q]
                if abs(app - aqq) < 1e-16 * max(abs(app), abs(aqq), 1.0):
                    t = -1.0
                else:
                    tau = (aqq - app) / (2.0 * apq)
                    t = -(1.0 if tau >= 0 else -1.0) / (
                        abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    if k == p or k == q:
                        continue
                    akp, akq = A[k][p], A[k][q]
                    A[k][p] = A[p][k] = c * akp + s * akq
                    A[k][q] = A[q][k] = -s * akp + c * akq
                A[p][p] = c * c * app + s * s * aqq + 2.0 * s * c * apq
                A[q][q] = s * s * app + c * c * aqq - 2.0 * s * c * apq
                A[p][q] = A[q][p] = 0.0
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p] = c * vkp + s * vkq
                    V[k][q] = -s * vkp + c * vkq
    evals = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: evals[i])
    evals = [evals[i] for i in order]
    V = [[V[i][j] for j in order] for i in range(n)]
    return evals, V


def _herm_eigh(a, uplo):
    """Hermitian（实对称/复 Hermitian）n×n 矩阵特征分解。

    a: n×n 嵌套列表（实数或复数；按 uplo 指定的三角为有效数据，
    另一侧被忽略）；uplo: 'L' 或 'U'。
    返回 (特征值升序实数列表, 特征向量 n×n 嵌套列表——实输入为实、
    复输入为复)。复 Hermitian 走 2n×2n 实对称嵌入
    B = [[Re, -Im], [Im, Re]]：B 的特征值两两成对，特征向量 [x; y]
    对应复特征向量 x + iy。
    """
    n = len(a)

    def _cj(v):
        return v.conjugate() if isinstance(v, complex) else v

    H = []
    for i in range(n):
        row = []
        for j in range(n):
            if (uplo == 'L' and i >= j) or (uplo == 'U' and i <= j):
                row.append(a[i][j])
            else:
                row.append(_cj(a[j][i]))
        H.append(row)
    is_c = any(isinstance(v, complex) for row in H for v in row)
    if not is_c:
        return _jacobi_eigh_real(H)
    R = [[v.real for v in row] for row in H]
    I = [[v.imag for v in row] for row in H]
    B = []
    for i in range(n):
        B.append([R[i][j] for j in range(n)] +
                 [-I[i][j] for j in range(n)])
    for i in range(n):
        B.append([I[i][j] for j in range(n)] +
                 [R[i][j] for j in range(n)])
    evb, Vb = _jacobi_eigh_real(B)
    evals = [0.5 * (evb[2 * j] + evb[2 * j + 1]) for j in range(n)]
    evecs = [[complex(Vb[i][2 * j], Vb[n + i][2 * j]) for j in range(n)]
             for i in range(n)]
    return evals, evecs

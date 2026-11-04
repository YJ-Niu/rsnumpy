"""多项式旧 API 移植（与 numpy 2.x lib/_polynomial_impl.py 行为对齐）。

包含 poly / roots / polyint / polyder / polyfit / polyval / polyadd /
polysub / polymul / polydiv / poly1d，覆盖官方 numpy TestPolynomial 语义：
poly1d 的 str/repr 对齐排版（_raise_power）、power 索引的 __getitem__/
__setitem__、p/q 返回 (商, 余数) 元组、polyfit 的 cov 参数、roots 的
dtype 保持（float32 输入 → float32 根）。
"""
import re as _re
import warnings as _py_warnings


def _NX():
    import rsnumpy
    return rsnumpy


class RankWarning(UserWarning):
    """polyfit 秩不足（病态拟合）告警，对应 numpy.exceptions.RankWarning。"""


def _isscalar(x):
    """numpy isscalar 语义：0-d 数组/Python 标量/标量对象视为标量。"""
    if hasattr(x, 'ndim'):
        return getattr(x, 'ndim', 1) == 0
    return isinstance(x, (int, float, complex, bool))


def _mintypecode(typechars, typeset='GDFgdf', default='d'):
    """最小可容纳类型字符（numpy mintypecode 的 typeset='GDFgdf' 子集语义）。

    输入字符全部在 typeset 内 → 返回其中尺寸最大者（容器须容纳全部）；
    有字符不在 typeset 内（int/bool 等）→ 安全回落到 default。
    """
    if isinstance(typechars, str):
        chars = list(typechars)
    else:
        chars = [c for c in typechars]
    in_set = [c for c in chars if c in typeset]
    if len(in_set) != len(chars):
        return default
    size = {'f': 4, 'd': 8, 'g': 16, 'F': 8, 'D': 16, 'G': 32}
    return max(in_set, key=lambda c: size[c])


def _to_real_if_imag_zero(w, t_dtype):
    """实输入矩阵的特征值若虚部全零 → 转实（numpy 向后兼容语义）。"""
    k = getattr(t_dtype, 'kind', 'f')
    if k != 'c' and hasattr(w, 'imag'):
        im = w.imag
        vals = im.tolist() if hasattr(im, 'tolist') else [im]
        if all(v == 0 for v in vals):
            return w.real
    return w


def _convolve_full(a_list, b_list, dtype=None):
    """1-d 满卷积（纯 Python，N 小；等价 np.convolve(mode='full')）。"""
    NX = _NX()
    out = [0] * (len(a_list) + len(b_list) - 1)
    for i, ai in enumerate(a_list):
        for j, bj in enumerate(b_list):
            out[i + j] += ai * bj
    if dtype is not None:
        return NX.array(out, dtype=dtype)
    return NX.array(out)


def _conjugates_match(roots_list):
    """复根序列是否全部成共轭对（此时系数为实，numpy poly 语义）。"""
    cs = sorted((complex(z) for z in roots_list),
                key=lambda z: (z.real, z.imag))
    cj = sorted((z.conjugate() for z in cs),
                key=lambda z: (z.real, z.imag))
    return all(a == b for a, b in zip(cs, cj))


def poly(seq_of_zeros):
    """由根序列（或方阵）返回多项式系数（按幂次降序，首项恒为 1）。"""
    NX = _NX()
    seq = NX.atleast_1d(seq_of_zeros)
    sh = seq.shape
    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        seq = NX.linalg.eigvals(seq)
    elif len(sh) == 1:
        dt = seq.dtype
        if dt.type is not NX.object_:
            seq = seq.astype(_mintypecode(dt.char))
    else:
        raise ValueError("input must be 1d or non-empty square 2d array.")
    if len(seq) == 0:
        return 1.0
    dt = seq.dtype
    a = NX.ones((1,), dtype=dt)
    for zero in seq:
        a = _convolve_full(a.tolist(), [1, -zero], dtype=dt)
    if issubclass(dt.type, NX.complexfloating):
        # 复根全为共轭对 → 系数为实
        if _conjugates_match(seq.tolist()):
            a = a.real.copy()
    return a


def roots(p):
    """返回多项式的根（系数按幂次降序；伴随矩阵特征值法）。"""
    NX = _NX()
    p = NX.atleast_1d(p)
    if p.ndim != 1:
        raise ValueError("Input must be a rank-1 array.")
    non_zero = NX.nonzero(NX.ravel(p))[0]
    if len(non_zero) == 0:
        return NX.array([])
    trailing_zeros = len(p) - int(non_zero[-1]) - 1
    p = p[int(non_zero[0]):int(non_zero[-1]) + 1]
    # 非浮点输入 → 转浮点（int → float64）
    if not issubclass(p.dtype.type, (NX.floating, NX.complexfloating)):
        p = p.astype(float)
    # 伴随矩阵 + 纯 Python 复 QR（rsnumpy linalg.eigvals 复特征值精度不足）
    p_list = p.tolist()
    while len(p_list) > 1 and p_list[0] == 0:
        p_list = p_list[1:]
    deg = len(p_list) - 1
    if deg == 0:
        r_list = []
    elif deg == 1:
        r_list = [-complex(p_list[1]) / complex(p_list[0])]
    else:
        c0 = complex(p_list[0])
        M = [[0j] * deg for _ in range(deg)]
        for j in range(deg):
            M[0][j] = -complex(p_list[j + 1]) / c0
        for i in range(1, deg):
            M[i][i - 1] = 1 + 0j
        r_list = _eigvals_qr(M)
    # 实输入且根虚部全 ~0 → 实数组（保持输入 dtype）；否则复数
    if p.dtype.kind != 'c':
        if r_list and all(abs(z.imag) <= 1e-9 * max(1.0, abs(z.real))
                          for z in r_list):
            r = NX.array([z.real for z in r_list], dtype=p.dtype)
        else:
            r = NX.array([complex(z) for z in r_list])
    else:
        r = NX.array([complex(z) for z in r_list], dtype=p.dtype)
    if trailing_zeros:
        z = NX.zeros(trailing_zeros, r.dtype if len(r) else p.dtype)
        r = NX.hstack((r, z))
    return r


def polyint(p, m=1, k=None):
    """返回多项式的 m 阶不定积分（常数 k，默认 0）。"""
    NX = _NX()
    m = int(m)
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    if k is None:
        k = NX.zeros(m, float)
    k = NX.atleast_1d(k)
    if len(k) == 1 and m > 1:
        k = k[0] * NX.ones(m, float)
    if len(k) < m:
        raise ValueError(
            "k must be a scalar or a rank-1 array of length 1 or >m.")
    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p)
    if m == 0:
        return poly1d(p) if truepoly else p
    # 整数/对象数组也须工作：真除法产生浮点
    y = NX.concatenate((p / NX.arange(len(p), 0, -1),
                        NX.array([k[0]])))
    val = polyint(y, m - 1, k=k[1:])
    return poly1d(val) if truepoly else val


def polyder(p, m=1):
    """返回多项式的 m 阶导数。"""
    NX = _NX()
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")
    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p)
    n = len(p) - 1
    y = p[:-1] * NX.arange(n, 0, -1)
    if m == 0:
        val = p
    else:
        val = polyder(y, m - 1)
    return poly1d(val) if truepoly else val


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """最小二乘多项式拟合（支持 full 诊断与 cov 协方差矩阵）。"""
    NX = _NX()
    order = int(deg) + 1
    x = NX.asarray(x) + 0.0
    y = NX.asarray(y) + 0.0
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    if rcond is None:
        rcond = len(x) * NX.finfo(x.dtype).eps
    lhs = NX.vander(x, order)
    rhs = y
    if w is not None:
        w = NX.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        lhs = lhs * w[:, NX.newaxis]
        if rhs.ndim == 2:
            rhs = rhs * w[:, NX.newaxis]
        else:
            rhs = rhs * w
    # 按列缩放改善条件数
    scale = NX.sqrt((lhs * lhs).sum(axis=0))
    lhs = lhs / scale
    c, resids, rank, s = NX.linalg.lstsq(lhs, rhs, rcond)
    c = (c.T / scale).T
    if rank != order and not full:
        _py_warnings.warn("Polyfit may be poorly conditioned",
                          RankWarning, stacklevel=2)
    if full:
        return c, resids, rank, s, rcond
    elif cov:
        Vbase = _inv_square(NX.dot(lhs.T, lhs))
        Vbase = Vbase / NX.outer(scale, scale)
        if cov == "unscaled":
            fac = 1
        else:
            if len(x) <= order:
                raise ValueError("the number of data points must exceed "
                                 "order to scale the covariance matrix")
            fac = resids / (len(x) - order)
        if y.ndim == 1:
            return c, Vbase * fac
        return c, Vbase[:, :, NX.newaxis] * fac
    return c


def polyval(p, x):
    """Horner 法求值；x 为 poly1d 时返回复合多项式。"""
    NX = _NX()
    p = NX.asarray(p)
    if isinstance(x, poly1d):
        y = 0
    else:
        x = NX.asanyarray(x)
        y = NX.zeros_like(x)
    for pv in p:
        y = y * x + pv
    return y


def _inv_square(a):
    """N×N 逆矩阵（高斯消元 + 部分选主元；rsnumpy linalg.inv 仅 2x2/3x3）。"""
    NX = _NX()
    n = len(a)
    M = [list(row) for row in a.tolist()]
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            raise ValueError("Singular matrix in polyfit covariance.")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            I[col], I[piv] = I[piv], I[col]
        d = M[col][col]
        M[col] = [v / d for v in M[col]]
        I[col] = [v / d for v in I[col]]
        for r2 in range(n):
            if r2 != col:
                f = M[r2][col]
                if f != 0:
                    M[r2] = [v - f * u for v, u in zip(M[r2], M[col])]
                    I[r2] = [v - f * u for v, u in zip(I[r2], I[col])]
    return NX.array(I)


def _cvec_norm(v):
    """复向量 2-范数。"""
    s = 0.0
    for z in v:
        s += z.real * z.real + z.imag * z.imag
    return s ** 0.5


def _qr_decomp(A):
    """n×n 复矩阵 QR 分解（修正 Gram-Schmidt；n 小）。"""
    n = len(A)
    Q = [[0j] * n for _ in range(n)]
    R = [[0j] * n for _ in range(n)]
    for j in range(n):
        v = [complex(A[i][j]) for i in range(n)]
        for k in range(j):
            rq = 0j
            for i in range(n):
                rq += Q[i][k].conjugate() * v[i]
            R[k][j] = rq
            for i in range(n):
                v[i] -= Q[i][k] * rq
        norm = _cvec_norm(v)
        R[j][j] = norm
        if norm > 0:
            for i in range(n):
                Q[i][j] = v[i] / norm
    return Q, R


def _matmul(A, B):
    n = len(A)
    p = len(B)
    m = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)]
            for i in range(n)]


def _eigvals_qr(M0, maxiter=3000):
    """n×n 矩阵特征值（复 QR + Rayleigh 移位 + 逐层消减；2×2 闭式解）。"""
    import cmath as _cmath
    n = len(M0)
    M = [[complex(v) for v in row] for row in M0]
    eig = []
    size = n
    while size > 2:
        converged = None
        for _it in range(maxiter):
            mu = M[size - 1][size - 1]
            A = [[M[i][j] - (mu if i == j else 0j) for j in range(size)]
                 for i in range(size)]
            Q, R = _qr_decomp(A)
            RQ = _matmul(R, Q)
            for i in range(size):
                for j in range(size):
                    M[i][j] = RQ[i][j] + (mu if i == j else 0j)
            scale = max(abs(M[size - 2][size - 2]),
                        abs(M[size - 1][size - 1]), 1e-300)
            if abs(M[size - 1][size - 2]) < 1e-13 * scale:
                converged = M[size - 1][size - 1]
                break
        eig.append(converged if converged is not None
                   else M[size - 1][size - 1])
        size -= 1
    if size == 2:
        a, b = M[0][0], M[0][1]
        c, d = M[1][0], M[1][1]
        tr = a + d
        s = _cmath.sqrt(tr * tr - 4 * (a * d - b * c))
        eig.append((tr + s) / 2)
        eig.append((tr - s) / 2)
    else:
        eig.append(M[0][0])
    return eig


def polyadd(a1, a2):
    """两个多项式相加（任一为 poly1d → 结果 poly1d）。"""
    NX = _NX()
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1 = NX.atleast_1d(a1)
    a2 = NX.atleast_1d(a2)
    diff = len(a2) - len(a1)
    if diff == 0:
        val = a1 + a2
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) + a2
    else:
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 + NX.concatenate((zr, a2))
    return poly1d(val) if truepoly else val


def polysub(a1, a2):
    """两个多项式相减（任一为 poly1d → 结果 poly1d）。"""
    NX = _NX()
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1 = NX.atleast_1d(a1)
    a2 = NX.atleast_1d(a2)
    diff = len(a2) - len(a1)
    if diff == 0:
        val = a1 - a2
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) - a2
    else:
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 - NX.concatenate((zr, a2))
    return poly1d(val) if truepoly else val


def polymul(a1, a2):
    """两个多项式相乘（满卷积；任一为 poly1d → 结果 poly1d）。"""
    NX = _NX()
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1, a2 = poly1d(a1), poly1d(a2)
    val = _convolve_full(a1.coeffs.tolist(), a2.coeffs.tolist())
    return poly1d(val) if truepoly else val


def polydiv(u, v):
    """多项式相除，返回 (商, 余数)（任一为 poly1d → 均为 poly1d）。"""
    NX = _NX()
    truepoly = (isinstance(u, poly1d) or isinstance(v, poly1d))
    u = NX.atleast_1d(u) + 0.0
    v = NX.atleast_1d(v) + 0.0
    # w 承载公共类型
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    scale = 1. / v[0]
    q = NX.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.astype(w.dtype)
    for k in range(m - n + 1):
        d = scale * r[k]
        q[k] = d
        r[k:k + n + 1] = r[k:k + n + 1] - d * v
    while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        r = r[1:]
    if truepoly:
        return poly1d(q), poly1d(r)
    return q, r


_poly_mat = _re.compile(r"\*\*([0-9]*)")


def _raise_power(astr, wrap=70):
    """numpy poly1d 对齐排版：把 "c x**p" 拆成两行（指数行 + 系数行）。"""
    n = 0
    line1 = ''
    line2 = ''
    output = ' '
    while True:
        mat = _poly_mat.search(astr, n)
        if mat is None:
            break
        span = mat.span()
        power = mat.groups()[0]
        partstr = astr[n:span[0]]
        n = span[1]
        toadd2 = partstr + ' ' * (len(power) - 1)
        toadd1 = ' ' * (len(partstr) - 1) + power
        if ((len(line2) + len(toadd2) > wrap)
                or (len(line1) + len(toadd1) > wrap)):
            output += line1 + "\n" + line2 + "\n "
            line1 = toadd1
            line2 = toadd2
        else:
            line2 += partstr + ' ' * (len(power) - 1)
            line1 += ' ' * (len(partstr) - 1) + power
    output += line1 + "\n" + line2
    return output + astr[n:]


def _coef_parts(coeff):
    """系数 → (是否复数非零虚部, 实部, 虚部)（str 格式化用）。"""
    v = coeff.tolist() if hasattr(coeff, 'tolist') else coeff
    if isinstance(v, complex):
        if v.imag == 0:
            return False, v.real, 0.0
        return True, v.real, v.imag
    return False, float(v), 0.0


class poly1d:
    """一维多项式类（兼容 numpy.poly1d 接口：power 索引、str 对齐排版、
    p/q 返回 (商, 余数) 元组、可变系数 c）。"""

    __hash__ = None

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        # numpy 中 p.coeffs 是自身缓冲区视图：+= 就地（无 setter），
        # setattr 整体替换 → AttributeError。rsnumpy 算术恒产生新对象，
        # 以 shape 区分：与现系数同 shape 视为 += / *= 类运算结果，接受；
        # 异 shape 整体替换 → AttributeError。
        NX = _NX()
        arr = value if hasattr(value, '_array') else NX.asarray(value)
        if tuple(arr.shape) == tuple(self._coeffs.shape):
            self._coeffs = arr
        else:
            raise AttributeError("Cannot set attribute")

    @property
    def variable(self):
        return self._variable

    @property
    def order(self):
        return len(self._coeffs) - 1

    @property
    def roots(self):
        return roots(self._coeffs)

    c = coef = coefficients = coeffs
    r = roots
    o = order

    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, poly1d):
            self._variable = c_or_r._variable
            self._coeffs = c_or_r._coeffs
            if variable is not None:
                self._variable = variable
            return
        NX = _NX()
        if r:
            c_or_r = poly(c_or_r)
        c_or_r = NX.atleast_1d(c_or_r)
        if c_or_r.ndim > 1:
            raise ValueError("Polynomial must be 1d only.")
        _dt = c_or_r.dtype  # trim 后空数组 dtype 可能丢失
        c_or_r = NX.trim_zeros(c_or_r, trim='f')
        if len(c_or_r) == 0:
            c_or_r = NX.array([0], dtype=_dt)
        self._coeffs = c_or_r
        self._variable = 'x' if variable is None else variable

    def __array__(self, t=None, copy=None):
        # rsnumpy asarray 无 copy 关键字：copy=None 语义 = 按需转换
        NX = _NX()
        if t:
            return NX.asarray(self.coeffs, t)
        return NX.asarray(self.coeffs)

    def __repr__(self):
        vals = repr(self.coeffs)
        # rsnumpy 数组 repr 为 "array([...])"，与 numpy 同款切片
        if vals.startswith('array(') and vals.endswith(')'):
            vals = vals[6:-1]
        return f"poly1d({vals})"

    def __len__(self):
        return self.order

    def __str__(self):
        NX = _NX()
        thestr = "0"
        var = self.variable
        # 移除前导零（首个非零系数起）；object 数组用 Python 逐元素比较
        full = self.coeffs
        if getattr(full.dtype, 'kind', 'f') == 'O':
            nz_list = [c != 0 for c in full.tolist()]
        else:
            nz_list = (full != 0).tolist()
        idx = next((i for i, v in enumerate(nz_list) if v), len(nz_list))
        coeffs = full[idx:]
        N = len(coeffs) - 1

        def fmt_float(q):
            s = f'{q:.4g}'
            s = s.removesuffix('.0000')
            return s

        for k, coeff in enumerate(coeffs):
            is_c, re_, im_ = _coef_parts(coeff)
            if not is_c:
                coefstr = fmt_float(re_)
            elif re_ == 0:
                coefstr = f'{fmt_float(im_)}j'
            else:
                coefstr = f'({fmt_float(re_)} + {fmt_float(im_)}j)'
            power = (N - k)
            if power == 0:
                if coefstr != '0':
                    newstr = f'{coefstr}'
                elif k == 0:
                    newstr = '0'
                else:
                    newstr = ''
            elif power == 1:
                if coefstr == '0':
                    newstr = ''
                else:
                    newstr = f'{coefstr} {var}'
            elif coefstr == '0':
                newstr = ''
            else:
                newstr = f'{coefstr} {var}**{power}'
            if k > 0:
                if newstr != '':
                    if newstr.startswith('-'):
                        thestr = f"{thestr} - {newstr[1:]}"
                    else:
                        thestr = f"{thestr} + {newstr}"
            else:
                thestr = newstr
        return _raise_power(thestr)

    def __call__(self, val):
        return polyval(self.coeffs, val)

    def __neg__(self):
        return poly1d(-self.coeffs)

    def __pos__(self):
        return self

    def __mul__(self, other):
        if _isscalar(other):
            return poly1d(self.coeffs * other)
        other = poly1d(other)
        return poly1d(polymul(self.coeffs, other.coeffs))

    def __rmul__(self, other):
        if _isscalar(other):
            return poly1d(other * self.coeffs)
        other = poly1d(other)
        return poly1d(polymul(self.coeffs, other.coeffs))

    def __add__(self, other):
        return poly1d(polyadd(self.coeffs, poly1d(other).coeffs))

    def __radd__(self, other):
        return poly1d(polyadd(self.coeffs, poly1d(other).coeffs))

    def __pow__(self, val):
        if not _isscalar(val) or int(val) != val or val < 0:
            raise ValueError("Power to non-negative integers only.")
        res = [1]
        for _ in range(val):
            res = polymul(self.coeffs, res)
        return poly1d(res)

    def __sub__(self, other):
        return poly1d(polysub(self.coeffs, poly1d(other).coeffs))

    def __rsub__(self, other):
        return poly1d(polysub(poly1d(other).coeffs, self.coeffs))

    def __truediv__(self, other):
        if _isscalar(other):
            return poly1d(self.coeffs / other)
        return polydiv(self, poly1d(other))

    def __rtruediv__(self, other):
        if _isscalar(other):
            return poly1d(other / self.coeffs)
        return polydiv(poly1d(other), self)

    def __eq__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        if self.coeffs.shape != other.coeffs.shape:
            return False
        return bool((self.coeffs == other.coeffs).all())

    def __ne__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        return not self.__eq__(other)

    def __getitem__(self, val):
        # val 是幂次（power），非数组下标
        ind = self.order - val
        if val > self.order:
            return self.coeffs.dtype.type(0)
        if val < 0:
            return self.coeffs.dtype.type(0)
        return self.coeffs[ind]

    def __setitem__(self, key, val):
        ind = self.order - key
        if key < 0:
            raise ValueError("Does not support negative powers.")
        if key > self.order:
            NX = _NX()
            zr = NX.zeros(key - self.order, self.coeffs.dtype)
            self._coeffs = NX.concatenate((zr, self.coeffs))
            ind = 0
        self._coeffs[ind] = val

    def __iter__(self):
        return iter(self.coeffs)

    def integ(self, m=1, k=0):
        """不定积分（等价 polyint(self, m, k)）。"""
        return poly1d(polyint(self.coeffs, m=m, k=k))

    def deriv(self, m=1):
        """m 阶导数（等价 polyder(self, m)）。"""
        return poly1d(polyder(self.coeffs, m=m))

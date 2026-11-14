# SPDX-License-Identifier: MIT
"""numpy.polynomial.polynomial 兼容模块（新 API：升序系数，低次在前）。

数值内核复用 Rust _core（polyadd/polysub/polymul/polydiv/polyroots，
均为降序系数约定，边界处反转）；求值/积分/导数/拟合走 rsnumpy 数组
运算或 lstsq（保持 dtype 与 object 语义）；校验、lbnd/k/scl、axis、
deg 幂列表、__array_function__ 派发按 numpy 2.x 语义实现。
"""

import builtins
import itertools as _itertools

__all__ = [
    'polyzero', 'polyone', 'polyx', 'polydomain',
    'polyval', 'polyvalfromroots', 'polyval2d', 'polyval3d', 'polyvalnd',
    'polygrid2d', 'polygrid3d',
    'polyder', 'polyint',
    'polyroots', 'polyfromroots', 'polyfit',
    'polyadd', 'polysub', 'polymul', 'polymulx', 'polydiv', 'polypow',
    'polytrim', 'polycompanion', 'polyline',
    'polyvander', 'polyvander2d', 'polyvander3d',
    'Polynomial',
]


def _NX():
    import rsnumpy
    return rsnumpy


def _core():
    import rsnumpy.num_core as _c
    return _c


def _nd():
    from rsnumpy import ndarray
    return ndarray


def _wrap(raw):
    return _nd()(raw)


def _raw(a):
    if hasattr(a, '_array'):
        return a._array
    return _core().ndarray(a)


def _frac_type():
    from fractions import Fraction
    return Fraction


def _isfrac(x):
    return isinstance(x, _frac_type())


def _is_py_scalar(x):
    return isinstance(x, (int, float, complex)) or _isfrac(x)


def _pyval(c, x):
    """Horner 求值（升序系数 c）。object dtype（Fraction）标量精确。"""
    y = 0
    for pv in reversed(c):
        y = y * x + pv
    return y


def _flat_shape(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _unflat(vals, shape):
    """1-D 列表 → 嵌套列表（C 序）；shape=() 返回标量。"""
    shape = tuple(shape)
    if not shape:
        return vals[0]
    sz = _flat_shape(shape[1:])
    if sz == 0:
        return [_unflat([], shape[1:]) for _ in range(shape[0])]
    return [_unflat(vals[i * sz:(i + 1) * sz], shape[1:])
            for i in range(shape[0])]


def _flat_list(data):
    """嵌套列表 → 1-D 列表（C 序）。"""
    if not isinstance(data, (list, tuple)):
        return [data]
    out = []
    for v in data:
        out.extend(_flat_list(v))
    return out


def _idx_unflat(gi, shape):
    """flat C 序索引 → 多下标元组。"""
    idx = []
    for s in reversed(shape):
        idx.append(int(gi % s))
        gi //= s
    return tuple(reversed(idx))


def _multiget(nested, idx):
    v = nested
    for i in idx:
        v = v[i]
    return v


def _mk_array(vals):
    """构造 ndarray；含 Fraction 时走 object dtype（保精确值）。"""
    NX = _NX()
    if isinstance(vals, (list, tuple)):
        flat = _flat_list(list(vals))
    else:
        flat = [vals]
    if any(_isfrac(v) for v in flat):
        seq = list(vals) if isinstance(vals, (list, tuple)) else [vals]
        return NX.array(seq, dtype='object')
    return NX.array(vals)


def _dispatch(args, func, kwargs=None):
    """__array_function__ 派发（numpy 2.x：第一个实现该协议的参数类型，
    在实例上调用 type(a).__array_function__(a, func, (t,), args, kwargs)）。"""
    if kwargs is None:
        kwargs = {}
    for a in args:
        t = builtins.type(a)
        af = getattr(t, '__array_function__', None)
        if af is not None:
            return af(a, func, (t,), args, kwargs)
    return None


def __getattr__(name):
    """常量惰性构造：包在 rsnumpy 部分初始化期被 import 时（rsnumpy/__init__
    早期 from .polynomial import ...）ndarray 尚不可用，推迟到运行时。"""
    if name == 'polyzero':
        return _nd()([0.])
    if name == 'polyone':
        return _nd()([1.])
    if name == 'polyx':
        return _nd()([0., 1.])
    if name == 'polydomain':
        return _nd()([-1., 1.])
    raise AttributeError("module 'rsnumpy.polynomial.polynomial' has no "
                         "attribute %r" % (name,))


def _py_mul(a, b):
    """升序系数相乘（纯 Python，object dtype 安全）。"""
    n, m = len(a), len(b)
    out = [0] * (n + m - 1)
    for i in range(n):
        for j in range(m):
            out[i + j] = out[i + j] + a[i] * b[j]
    return out


def _py_add(a, b):
    """升序系数相加（纯 Python，object dtype 安全）。"""
    n = builtins.max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
            for i in range(n)]


# ========== 求值 ==========

def polyval(x, c, tensor=True):
    """按升序系数在 x 处求值。c 多维时末轴为批量维：
    tensor=True 输出形状 c.shape[1:] + x.shape，否则 x.shape + c.shape[1:]。"""
    r = _dispatch((x, c), polyval, {'tensor': tensor})
    if r is not None:
        return r
    NX = _NX()
    c = NX.asarray(c)
    cshape = tuple(c.shape)
    cvals = c.tolist()
    if c.ndim == 1:
        if not cvals:
            return NX.array([])
        if _is_py_scalar(x):
            return _pyval(cvals, x)
        xarr = NX.asanyarray(x)
        y = NX.zeros_like(xarr)
        for pv in reversed(cvals):
            y = y * xarr + pv
        return y
    # c 多维：首轴系数轴，其余批量轴
    xarr = NX.asanyarray(x)
    xshape = tuple(xarr.shape)
    xflat = _flat_list(xarr.tolist())
    npts = _flat_shape(xshape) if xshape else 1
    nb = _flat_shape(cshape[1:])
    res = [0.0] * (nb * npts)
    for g in range(nb):
        gidx = _idx_unflat(g, cshape[1:])
        ccol = [_multiget(cvals, (i,) + gidx) for i in range(cshape[0])]
        for k in range(npts):
            v = _pyval(ccol, xflat[k])
            if tensor:
                res[g * npts + k] = v
            else:
                res[k * nb + g] = v
    out_shape = cshape[1:] + xshape if tensor else xshape + cshape[1:]
    return NX.array(_unflat(res, out_shape))


def polyvalfromroots(x, roots, tensor=True):
    """Πᵢ (x - roots[i])；roots 多维时末轴组分别求值。
    tensor=True：输出形状 roots.shape[1:] + x.shape；
    tensor=False：x 与根组逐组配对（第 i 个 x 点配第 i 组根），
    输出形状 roots.shape[1:]（roots 须 ≥2 维）。"""
    NX = _NX()
    x = NX.asanyarray(x)
    roots = NX.asarray(roots)
    if roots.ndim == 0:
        # 0 维根 = 单根（numpy 语义）
        rshape = (1,)
        rval = [roots.tolist()]
    else:
        rshape = tuple(roots.shape)
        rval = roots.tolist()
    xshape = tuple(x.shape)
    if len(rshape) > 1:
        out_shape = rshape[1:] + xshape if tensor else rshape[1:]
    elif tensor:
        out_shape = xshape
    else:
        raise ValueError("too few dimensions in `roots`")
    xval = x.tolist()
    ndx = x.ndim
    if ndx == 0:
        xflat = [xval]
        xshape = (1,)
    else:
        xflat = _flat_list(xval)
    npts = _flat_shape(xshape)
    if len(rshape) > 1:
        gshape = rshape[1:]
        ng = _flat_shape(gshape)
        if tensor:
            res = [0.0] * (ng * npts)
            for gi in range(ng):
                gidx = _idx_unflat(gi, gshape)
                for k in range(npts):
                    v = 1.0
                    for i in range(rshape[0]):
                        v *= (xflat[k] - _multiget(rval, (i,) + gidx))
                    res[gi * npts + k] = v
        else:
            # 逐组配对：x[i] 与第 i 组根
            if npts < ng:
                raise ValueError("x must have at least as many points "
                                 "as root groups for tensor=False")
            res = [0.0] * ng
            for gi in range(ng):
                gidx = _idx_unflat(gi, gshape)
                v = 1.0
                for i in range(rshape[0]):
                    v *= (xflat[gi] - _multiget(rval, (i,) + gidx))
                res[gi] = v
        return NX.array(_unflat(res, out_shape))
    # 1 维 roots：单组
    res = []
    for xv in xflat:
        v = 1.0
        for rr in rval:
            v *= (xv - rr)
        res.append(v)
    return NX.array(_unflat(res, out_shape))


# ========== 导数 / 积分 ==========

def _apply_along_axis(c, fn, axis):
    """对 ndarray c 沿系数轴 axis 逐段应用 fn(list)->list（transpose 实现，
    任意维：先转置把系数轴放最后，折叠前置维后逐行应用）。"""
    NX = _NX()
    shape = tuple(c.shape)
    nd = builtins.len(shape)
    if nd <= 1:
        return NX.array(fn(c.tolist()))
    ax = axis if axis >= 0 else axis + nd
    perm = [a for a in range(nd) if a != ax]
    perm.append(ax)
    t = c.transpose(tuple(perm))
    tshape = tuple(shape[p] for p in perm)
    lead = _flat_shape(tshape[:-1])
    t2 = t.reshape((lead, tshape[-1]))
    fn_out = [fn(row) for row in t2.tolist()]
    new_len = builtins.len(fn_out[0])
    o = NX.array(fn_out).reshape(tshape[:-1] + (new_len,))
    inv = [0] * nd
    for i, p in enumerate(perm):
        inv[p] = i
    return o.transpose(tuple(inv))


def _der1d(cvals, m, scl):
    cur = list(cvals)
    for _ in range(m):  # m 为 float 时自然 TypeError
        if len(cur) <= 1:
            cur = [0.0]
        else:
            cur = [(j + 1) * scl * cur[j + 1] for j in range(len(cur) - 1)]
    return cur


def _int1d(cvals, m, k, lbnd, scl):
    cur = list(cvals)
    for i in range(m):
        cur = [v * scl for v in cur]  # scl 每次积分都乘（numpy 语义）
        if builtins.len(cur) == 1 and cur[0] == 0:
            # 长度为 1 的零多项式：保持长度，只累加积分常数（numpy 语义）
            cur = [cur[0] + k[i]]
        else:
            nxt = [0] + [cur[j] / (j + 1) for j in range(len(cur))]
            # lbnd 语义：第 i+1 次积分后的多项式在 lbnd 处取值恰为 k[i]
            if lbnd is not None:
                nxt[0] = nxt[0] + k[i] - _pyval(nxt, lbnd)
            else:
                nxt[0] = nxt[0] + k[i]
            cur = nxt
    return cur


def _check_axis(axis):
    if axis is not None and (isinstance(axis, bool) or not isinstance(axis, int)):
        raise TypeError("axis must be an integer")


def polyder(c, m=1, scl=1, axis=0):
    """m 阶导数（每阶乘 scl）。axis 指定系数轴（numpy 默认 0）。"""
    _check_axis(axis)
    NX = _NX()
    if m < 0:
        raise ValueError("m must be non-negative")
    c = NX.asarray(c)
    if c.ndim == 1:
        return NX.array(_der1d(c.tolist(), m, scl))
    ax = 0 if axis is None else axis
    fn = lambda v: _der1d(v, m, scl)
    return _apply_along_axis(c, fn, ax)


def polyint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """m 次积分（numpy 默认 k=[]、lbnd=0、scl=1、axis=0）。
    k 为各次积分常数（标量或长 m 序列，空序列=全零）；
    lbnd 标量：使第 i+1 次积分后多项式在 lbnd 处取 k[i]；scl 整体缩放。"""
    _check_axis(axis)
    NX = _NX()
    if m < 0:
        raise ValueError("m must be non-negative")
    if isinstance(k, (list, tuple)):
        if len(k) > m:
            raise ValueError("Too many integration constants")
        # 短序列零填充（numpy 语义）
        klist = list(k) + [0] * (m - builtins.len(k))
    else:
        klist = [k] * m
    if isinstance(lbnd, (list, tuple)):
        raise ValueError("lbnd must be a scalar")
    if isinstance(scl, (list, tuple)):
        raise ValueError("scl must be a scalar")
    c = NX.asarray(c)
    if c.ndim == 1:
        return NX.array(_int1d(c.tolist(), m, klist, lbnd, scl))
    ax = 0 if axis is None else axis
    fn = lambda v: _int1d(v, m, klist, lbnd, scl)
    return _apply_along_axis(c, fn, ax)


# ========== 根 / 由根构造 / 拟合 ==========

def polyroots(c):
    """多项式根（Rust QR 伴矩阵特征值内核，降序约定）。
    结果按 (实部, 虚部) 排序——实根升序，与 numpy 实根情形一致。"""
    NX = _NX()
    c = NX.asarray(c)
    cvals = c.tolist()
    if len(cvals) <= 1:
        return NX.array([])
    desc = list(reversed(cvals))
    raw = _wrap(_core().polyroots_rs(_raw(desc)))
    vals = raw.tolist()
    cv = [complex(v) for v in vals]
    cv.sort(key=lambda z: (z.real, z.imag))
    if all(z.imag == 0 for z in cv):
        return NX.array([z.real for z in cv])
    return NX.array([z for z in cv])


def polyfromroots(roots):
    """由根序列返回升序系数。"""
    NX = _NX()
    r = NX.asarray(roots)
    if r.size == 0:
        return NX.array([1.0])
    vals = r.tolist()
    out = [1.0]
    for rt in vals:
        # 标准构造：c * (x - rt)：new[i] -= c[i]*rt; new[i+1] += c[i]（升序）
        new = [0.0] * (builtins.len(out) + 1)
        for i, cv in enumerate(out):
            new[i] = new[i] - cv * rt
            new[i + 1] = new[i + 1] + cv
        out = new
    return NX.array(out)


def _is_complex_data(vals):
    """嵌套列表数据是否含 complex 元素。"""
    if isinstance(vals, (list, tuple)):
        flat = vals if all(not isinstance(v, (list, tuple))
                           for v in vals) else _flat_list(vals)
    else:
        flat = [vals]
    return any(isinstance(v, complex) for v in flat)


def _gauss_complex(A, bvec):
    """复数高斯消元（列主元）。"""
    k = builtins.len(bvec)
    M = [list(A[i]) + [bvec[i]] for i in range(k)]
    for i in range(k):
        p = i
        best = builtins.abs(M[i][i])
        for r in range(i + 1, k):
            if builtins.abs(M[r][i]) > best:
                best = builtins.abs(M[r][i])
                p = r
        M[i], M[p] = M[p], M[i]
        piv = M[i][i]
        if piv == 0:
            continue
        for r in range(i + 1, k):
            f = M[r][i] / piv
            if f != 0:
                for c in range(i, k + 1):
                    M[r][c] = M[r][c] - f * M[i][c]
    x = [0j] * k
    for i in range(k - 1, -1, -1):
        s = M[i][k]
        for c in range(i + 1, k):
            s = s - M[i][c] * x[c]
        x[i] = s / M[i][i] if M[i][i] != 0 else 0j
    return x


def _lstsq_normal(V, ycol):
    """正规方程 (V^H V) c = V^H y 最小二乘（复数路径用；
    rsnumpy 复数 lstsq 内核损坏，见 linalg 后续任务）。"""
    n = builtins.len(V)
    k = builtins.len(V[0])
    A = [[0j] * k for _ in range(k)]
    for i in range(n):
        row = V[i]
        for a in range(k):
            ca = row[a].conjugate()
            for b in range(k):
                A[a][b] = A[a][b] + ca * row[b]
    bvec = [sum(V[i][a].conjugate() * ycol[i] for i in range(n))
            for a in range(k)]
    return _gauss_complex(A, bvec)


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """最小二乘拟合。deg 可为 int 或严格递增非负幂列表；
    结果长度 max(deg)+1（缺失幂次补零）；支持复数 x、2-D y（逐列）、权重 w。"""
    NX = _NX()
    x = NX.asarray(x)
    y = NX.asarray(y)
    if x.ndim != 1:
        raise TypeError("x must be 1 dimensional")
    if y.ndim > 2:
        raise TypeError("y must be 1 or 2 dimensional")
    if x.size == 0:
        raise TypeError("x must be non-empty")
    if y.shape[0] != x.size:
        raise TypeError("x and y must have same first dimension")
    if w is not None:
        w = NX.asarray(w)
        if w.ndim != 1 or w.size != x.size:
            raise TypeError("w must be 1 dimensional and same length as x")
    if isinstance(deg, (list, tuple)):
        if len(deg) == 0:
            raise TypeError("deg must be a non-empty sequence or integer")
        for d in deg:
            if isinstance(d, bool) or not isinstance(d, int):
                raise TypeError("deg values must be integers")
            if d < 0:
                raise ValueError("deg must be non-negative")
        if any(deg[i] >= deg[i + 1] for i in range(len(deg) - 1)):
            raise ValueError("deg sequence must be strictly increasing")
        order = builtins.max(deg)
    else:
        if isinstance(deg, bool) or not isinstance(deg, int):
            raise TypeError("deg must be an integer or sequence")
        if deg < 0:
            raise ValueError("deg must be non-negative")
        order = deg
    xvals = x.tolist()
    yvals = y.tolist()
    wvals = w.tolist() if w is not None else None
    rows = []
    for i, xv in enumerate(xvals):
        s = (wvals[i] ** 0.5) if wvals is not None else 1.0
        rows.append([(xv ** d) * s for d in range(order + 1)])
    if _is_complex_data(rows) or _is_complex_data(yvals):
        # 复数路径：正规方程纯 Python 求解（复数 lstsq 内核损坏）
        if y.ndim == 2:
            ncol = builtins.len(yvals[0])
            cols = [[row[i] for row in yvals] for i in range(ncol)]
        else:
            cols = [yvals]
        sols = [_lstsq_normal(rows, col) for col in cols]
        if y.ndim == 1:
            sol = NX.array(sols[0])
        else:
            sol = NX.array([list(row) for row in sols])
        if full:
            return sol, NX.array([]), NX.array([order + 1]), NX.array([])
        return sol
    # 1 维 y → 1 维解（numpy 语义）；2 维 y 逐列
    Y = NX.array(yvals) if y.ndim == 1 else \
        NX.array([list(row) for row in yvals])
    sol, resids, rank, sv = NX.linalg.lstsq(NX.array(rows), Y, rcond=rcond)
    if y.ndim == 1 and sol.ndim == 2:
        # lstsq 对 1-D b 可能返回 (n,1) → 压平为 (n,)
        sol = NX.array([v[0] for v in sol.tolist()])
    if full:
        return sol, resids, rank, sv
    return sol


# ========== 算术族（Rust 内核，降序约定边界反转） ==========

def _binop(name, a, b):
    NX = _NX()
    a = NX.asarray(a)
    b = NX.asarray(b)
    ra = list(reversed(a.tolist()))
    rb = list(reversed(b.tolist()))
    res = getattr(_core(), name)(_raw(ra), _raw(rb))
    return NX.array(list(reversed(res.tolist())))


def polyadd(a1, a2):
    return _binop('polyadd', a1, a2)


def polysub(a1, a2):
    return _binop('polysub', a1, a2)


def polymul(a1, a2):
    return _binop('polymul', a1, a2)


def polymulx(c):
    """乘以 x（升序：前置补零后裁剪尾零）。"""
    NX = _NX()
    c = NX.asarray(c)
    vals = c.tolist()
    out = [0.0] + list(vals)
    i = len(out) - 1
    while i >= 0 and builtins.abs(out[i]) <= 0:
        i -= 1
    if i < 0:
        return NX.array([0.0])
    return NX.array(out[:i + 1])


def polydiv(u, v):
    """多项式除法 → (商, 余数)；v 为零多项式抛 ZeroDivisionError。"""
    NX = _NX()
    u = NX.asarray(u)
    v = NX.asarray(v)
    vv = list(reversed(v.tolist()))
    if not any(builtins.abs(xv) > 0 for xv in vv):
        raise ZeroDivisionError("polynomial division")
    qd, rd = _core().polydiv(_raw(list(reversed(u.tolist()))), _raw(vv))
    return (NX.array(list(reversed(qd.tolist()))),
            NX.array(list(reversed(rd.tolist()))))


def polypow(c, pow, maxpower=None):
    """c 的 pow 次幂（pow=0 → [1]）。maxpower 为 numpy 预留参数。"""
    if isinstance(pow, bool) or not isinstance(pow, int):
        raise TypeError("pow must be an integer")
    if pow < 0:
        raise ValueError("pow must be non-negative")
    NX = _NX()
    c = NX.asarray(c)
    cvals = c.tolist()
    if pow == 0:
        return NX.array([1.0])
    out = [1.0]
    for _ in range(pow):
        out = _py_mul(out, cvals)
    return NX.array(out)


def polytrim(coef, tol=0):
    """自高次端裁剪 |c| <= tol 的系数；全裁完返回 [0]。"""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    NX = _NX()
    c = NX.asarray(coef)
    vals = c.tolist()
    i = len(vals) - 1
    while i >= 0 and builtins.abs(vals[i]) <= tol:
        i -= 1
    if i < 0:
        return NX.array([0.0])
    return NX.array(vals[:i + 1])


def polycompanion(c):
    """伴随矩阵（升序系数；首行为 -c[:n]/c[n]，次对角线为单位）。"""
    NX = _NX()
    c = NX.asarray(c)
    vals = c.tolist()
    if c.ndim != 1 or len(vals) < 2:
        raise ValueError("c must be a 1d array with at least 2 elements")
    lead = vals[-1]
    cv = [v / lead for v in vals]
    n = len(vals) - 1
    comp = [[0.0] * n for _ in range(n)]
    for i in range(1, n):
        comp[i][i - 1] = 1.0
    comp[0] = [-cv[i] for i in range(n)]
    return NX.array(comp)


def polyline(off, scl):
    """线性多项式 off + scl*x 的系数表示（scl=0 时裁剪）。"""
    NX = _NX()
    if scl != 0:
        return NX.array([off, scl])
    return NX.array([off])


# ========== Vandermonde 矩阵族 ==========

def polyvander(x, deg):
    """x**i 沿末轴堆叠；输出形状 x.shape + (deg+1,)。"""
    if deg < 0:
        raise ValueError("deg must be non-negative")
    NX = _NX()
    x = NX.asarray(x)
    xv = x.tolist()

    def rec(v):
        if isinstance(v, (list, tuple)):
            return [rec(u) for u in v]
        return [v ** i for i in range(deg + 1)]

    return NX.array(rec(xv))


def _vander_nd(pts, deg):
    """通用 nd Vandermonde：输出形状 pts[0].shape + tuple(d+1 for d in deg)。"""
    NX = _NX()
    pts = [NX.asarray(p) for p in pts]
    shape = tuple(pts[0].shape)
    for p in pts[1:]:
        if tuple(p.shape) != shape:
            raise ValueError("incompatible point dimensions in polyvandern")
    flats = [p.ravel().tolist() for p in pts]
    n = len(flats[0])
    grid_shape = tuple(d + 1 for d in deg)
    ng = _flat_shape(grid_shape)
    # res[g][k] = Π_a flats[a][k] ** idx_a(g)
    res = [[0.0] * n for _ in range(ng)]
    for g in range(ng):
        gidx = _idx_unflat(g, grid_shape)
        for k in range(n):
            v = 1.0
            for a in range(len(pts)):
                if gidx[a]:
                    v *= flats[a][k] ** gidx[a]
            res[g][k] = v
    # 转置：每点 k 的 ng 个值构成 grid_shape 嵌套
    per_point = [_unflat([res[g][k] for g in range(ng)], grid_shape)
                 for k in range(n)]
    return NX.array(_unflat(per_point, shape))


def _broadcast_flat(arr, bshape):
    """把 arr 按广播规则铺到 bshape 后展平为 1-D 列表。"""
    shape = tuple(arr.shape)
    if shape == bshape:
        return _flat_list(arr.tolist())
    nd = builtins.len(bshape)
    pad = [1] * (nd - builtins.len(shape)) + list(shape)
    vals = _flat_list(arr.tolist())
    npts = _flat_shape(bshape)
    out = []
    for k in range(npts):
        q = _idx_unflat(k, bshape)
        sidx = 0
        for a in range(nd - builtins.len(shape), nd):
            s = int(q[a] % pad[a])
            sidx = sidx * builtins.int(pad[a]) + s
        out.append(vals[builtins.int(sidx)])
    return out


def polyvander2d(x, y, deg):
    """2 点 Vandermonde：输出形状 broadcast(x, y).shape + (扁平网格,)，
    网格按 (deg0, deg1) C 序展开。"""
    if builtins.len(deg) != 2:
        raise ValueError("deg must be a length-2 sequence")
    NX = _NX()
    x = NX.asarray(x)
    y = NX.asarray(y)
    xs, ys = tuple(x.shape), tuple(y.shape)
    if not _can_broadcast((xs, ys)):
        raise ValueError("incompatible point dimensions in polyvander2d")
    bshape = _broadcast_shape((xs, ys))
    npts = _flat_shape(bshape)
    gridlen = (deg[0] + 1) * (deg[1] + 1)
    xv = _broadcast_flat(x, bshape)
    yv = _broadcast_flat(y, bshape)
    data = []
    for k in range(npts):
        for i in range(deg[0] + 1):
            for j in range(deg[1] + 1):
                data.append(xv[k] ** i * yv[k] ** j)
    return NX.array(_unflat(data, bshape + (gridlen,)))


def polyvander3d(x, y, z, deg):
    """3 点 Vandermonde：输出形状 broadcast(x, y, z).shape + (扁平网格,)，
    网格按 (deg0, deg1, deg2) C 序展开。"""
    if builtins.len(deg) != 3:
        raise ValueError("deg must be a length-3 sequence")
    NX = _NX()
    x = NX.asarray(x)
    y = NX.asarray(y)
    z = NX.asarray(z)
    shapes = (tuple(x.shape), tuple(y.shape), tuple(z.shape))
    if not _can_broadcast(shapes):
        raise ValueError("incompatible point dimensions in polyvander3d")
    bshape = _broadcast_shape(shapes)
    npts = _flat_shape(bshape)
    gridlen = (deg[0] + 1) * (deg[1] + 1) * (deg[2] + 1)
    xv = _broadcast_flat(x, bshape)
    yv = _broadcast_flat(y, bshape)
    zv = _broadcast_flat(z, bshape)
    data = []
    for k in range(npts):
        for i in range(deg[0] + 1):
            for j in range(deg[1] + 1):
                for t in range(deg[2] + 1):
                    data.append(xv[k] ** i * yv[k] ** j * zv[k] ** t)
    return NX.array(_unflat(data, bshape + (gridlen,)))


# ========== 多维求值 / 网格 ==========

def _can_broadcast(shapes):
    m = builtins.max(builtins.len(s) for s in shapes)
    padded = [list(reversed(s)) + [1] * (m - builtins.len(s))
              for s in shapes]
    for d in range(m):
        sizes = builtins.set(p[d] for p in padded)
        sizes.discard(1)
        if builtins.len(sizes) > 1:
            return False
    return True


def _broadcast_shape(shapes):
    m = builtins.max(builtins.len(s) for s in shapes)
    b = [1] * m
    for s in shapes:
        for d, sz in enumerate(reversed(s)):
            b[m - 1 - d] = builtins.max(b[m - 1 - d], sz)
    return tuple(b)


def _polyvalnd(pts, c):
    """Σ c[idx] Π p_a**idx[a]（点数组按 numpy 广播规则）。"""
    NX = _NX()
    pts = [NX.asarray(p) for p in pts]
    if not pts:
        raise ValueError("pts must be non-empty")
    c = NX.asarray(c)
    if c.ndim != builtins.len(pts):
        raise ValueError("coef array must have the same ndim as "
                         "number of points")
    shapes = [tuple(p.shape) for p in pts]
    if not _can_broadcast(shapes):
        raise ValueError("incompatible point dimensions in polyvalnd: "
                         + " ".join(repr(s) for s in shapes))
    bshape = _broadcast_shape(shapes)
    cvals = c.tolist()
    res = NX.zeros(bshape)
    for idx in _itertools.product(*[range(d) for d in c.shape]):
        cv = _multiget(cvals, idx)
        if cv == 0:
            continue
        term = NX.array(cv)
        for a, p in enumerate(pts):
            if idx[a]:
                term = term * (p ** idx[a])
        res = res + term
    return res


def polyval2d(x, y, c):
    """Σ c[i,j] x**i y**j。"""
    r = _dispatch((x, y, c), polyval2d)
    if r is not None:
        return r
    return _polyvalnd((x, y), c)


def polyval3d(x, y, z, c):
    """Σ c[i,j,k] x**i y**j z**k。"""
    r = _dispatch((x, y, z, c), polyval3d)
    if r is not None:
        return r
    return _polyvalnd((x, y, z), c)


def polyvalnd(pts, c):
    """Σ c[idx] Π p_a**idx[a]（pts 为点数组序列）。"""
    r = _dispatch(tuple(pts) + (c,), polyvalnd)
    if r is not None:
        return r
    return _polyvalnd(tuple(pts), c)


def polygrid2d(x1, x2, c):
    """配对网格求值（xy 索引：输出形状 (len(x2), len(x1))）。"""
    r = _dispatch((x1, x2, c), polygrid2d)
    if r is not None:
        return r
    NX = _NX()
    a1 = NX.asarray(x1)
    a2 = NX.asarray(x2)
    # 输出 res[i, j] = f(x1[i], x2[j])（保留输入形状，测试规格 einsum 'i,j->ij'）
    p1 = a1.reshape(a1.shape + (1,) * builtins.len(a2.shape))
    p2 = a2.reshape((1,) * builtins.len(a1.shape) + a2.shape)
    return _polyvalnd((p1, p2), c)


def polygrid3d(x1, x2, x3, c):
    """三维配对网格求值（输出形状 (len(x3), len(x2), len(x1))）。"""
    NX = _NX()
    a1 = NX.asarray(x1)
    a2 = NX.asarray(x2)
    a3 = NX.asarray(x3)
    # 输出形状 = s1 + s2 + s3：各输入占据互不重叠的轴块（其余轴 pad 1）
    s1, s2, s3 = a1.shape, a2.shape, a3.shape
    R = builtins.len(s1) + builtins.len(s2) + builtins.len(s3)
    p1 = a1.reshape(s1 + (1,) * (R - builtins.len(s1)))
    p2 = a2.reshape((1,) * builtins.len(s1) + s2
                    + (1,) * (R - builtins.len(s1) - builtins.len(s2)))
    p3 = a3.reshape((1,) * (builtins.len(s1) + builtins.len(s2)) + s3)
    return _polyvalnd((p1, p2, p3), c)


# ========== Polynomial 类 ==========

def _py_der(cvals, m):
    cur = list(cvals)
    for _ in range(m):
        if len(cur) <= 1:
            cur = [0]
        else:
            cur = [(j + 1) * cur[j + 1] for j in range(len(cur) - 1)]
    return cur


class Polynomial:
    """numpy.polynomial.Polynomial 兼容类（升序系数；默认 domain/window
    [-1, 1]；求值先把 x 从 domain 线性映射到 window）。"""

    def __init__(self, c, domain=None, window=None, symbol='x'):
        NX = _NX()
        c = _mk_array(c) if isinstance(c, (list, tuple)) else NX.asarray(c)
        if c.ndim > 1:
            c = c.ravel()
        elif c.ndim == 0:
            c = NX.array([c.tolist()])
        self._coef = c
        if domain is None:
            domain = NX.array([-1., 1.])
        if window is None:
            window = NX.array([-1., 1.])
        self._domain = _mk_array(domain) if isinstance(domain, (list, tuple)) \
            else NX.asarray(domain)
        self._window = _mk_array(window) if isinstance(window, (list, tuple)) \
            else NX.asarray(window)
        self._symbol = symbol

    # ---- 属性 ----
    @property
    def coef(self):
        return self._coef

    @property
    def domain(self):
        return self._domain

    @property
    def window(self):
        return self._window

    @property
    def symbol(self):
        return self._symbol

    @property
    def degree(self):
        return builtins.max(len(self._coef.tolist()) - 1, 0)

    # ---- domain → window 映射参数：mapped = x * sc + ofst ----
    def _mapparms(self):
        # 用数组元素访问保 dtype（f4 domain → f4 参数，弱标量语义）；
        # object（Fraction）元素访问返回 Python 对象，走精确算术
        d, w = self._domain, self._window
        sc = (w[1] - w[0]) / (d[1] - d[0])
        ofst = (w[0] * d[1] - w[1] * d[0]) / (d[1] - d[0])
        return ofst, sc

    def __call__(self, x):
        ofst, sc = self._mapparms()
        if _isfrac(x):
            # Fraction 路径：全程 Python 精确算术
            return _pyval(self._coef.tolist(), x * sc + ofst)
        NX = _NX()
        # x 为 Python 标量时保持弱标量语义（dtype 跟随 domain/coef）
        mapped = (x * sc + ofst) if _is_py_scalar(x) else \
            NX.asarray(x) * sc + ofst
        mapped = NX.asarray(mapped)
        y = NX.zeros_like(mapped)
        for pv in reversed(self._coef.tolist()):
            y = y * mapped + pv
        return y

    # ---- 算术（object dtype 安全，纯 Python 系数运算） ----
    def _same_basiss(self, other):
        return (self._domain.tolist() == other._domain.tolist()
                and self._window.tolist() == other._window.tolist())

    def __add__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        if not self._same_basiss(other):
            raise ValueError("domains or windows must be the same")
        c = _py_add(self._coef.tolist(), other._coef.tolist())
        return Polynomial(_mk_array(c), domain=self._domain,
                          window=self._window, symbol=self._symbol)

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        if not self._same_basiss(other):
            raise ValueError("domains or windows must be the same")
        a = self._coef.tolist()
        b = other._coef.tolist()
        n = builtins.max(len(a), len(b))
        c = [(a[i] if i < builtins.len(a) else 0)
             - (b[i] if i < builtins.len(b) else 0)
             for i in range(n)]
        return Polynomial(_mk_array(c), domain=self._domain,
                          window=self._window, symbol=self._symbol)

    def __neg__(self):
        return Polynomial(_mk_array([-v for v in self._coef.tolist()]),
                          domain=self._domain, window=self._window,
                          symbol=self._symbol)

    def _scaled(self, s):
        NX = _NX()
        vals = self._coef.tolist()
        if any(_isfrac(v) for v in vals):
            c = NX.array([s * v for v in vals], dtype='object')
        else:
            c = self._coef * s
        return Polynomial(c, domain=self._domain, window=self._window,
                          symbol=self._symbol)

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            c = _py_mul(self._coef.tolist(), other._coef.tolist())
            return Polynomial(_mk_array(c), domain=self._domain,
                              window=self._window, symbol=self._symbol)
        if _is_py_scalar(other):
            return self._scaled(other)
        return NotImplemented

    __rmul__ = __mul__

    def __pow__(self, n):
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("pow must be an integer")
        if n < 0:
            raise ValueError("pow must be non-negative")
        if n == 0:
            c = [1]
        else:
            c = [1]
            for _ in range(n):
                c = _py_mul(c, self._coef.tolist())
        return Polynomial(_mk_array(c), domain=self._domain,
                          window=self._window, symbol=self._symbol)

    # ---- 导数 / 根 ----
    def deriv(self, m=1):
        c = _py_der(self._coef.tolist(), m)
        return Polynomial(_mk_array(c), domain=self._domain,
                          window=self._window, symbol=self._symbol)

    def roots(self):
        return polyroots(self._coef)

    # ---- 比较（numpy Polynomial 无 __eq__；此处提供以便测试比较） ----
    def __eq__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        return (self._coef.tolist() == other._coef.tolist()
                and self._same_basiss(other))

    def __ne__(self, other):
        r = self.__eq__(other)
        if r is NotImplemented:
            return r
        return not r

    __hash__ = None

    def __repr__(self):
        return ('Polynomial(' + repr(self._coef.tolist()) + ', domain='
                + repr(self._domain.tolist()) + ', window='
                + repr(self._window.tolist()) + ')')

    __str__ = __repr__

    # ---- 拟合 ----
    @builtins.classmethod
    def fit(cls, x, y, deg=1, rcond=None, w=None, full=False,
            domain=None, window=None, symbol='x'):
        """最小二乘拟合（x 先从 domain 映射到 window）；秩亏时 RankWarning。"""
        NX = _NX()
        if domain is None:
            domain = NX.array([-1., 1.])
        if window is None:
            window = NX.array([-1., 1.])
        domain = NX.asarray(domain)
        window = NX.asarray(window)
        d = domain.tolist()
        wv = window.tolist()
        sc = (wv[1] - wv[0]) / (d[1] - d[0])
        ofst = (wv[0] * d[1] - wv[1] * d[0]) / (d[1] - d[0])
        xarr = NX.asarray(x)
        xm = xarr * sc + ofst
        sol, _resids, rank, _sv = polyfit(xm, y, deg, rcond=rcond,
                                          full=True, w=w)
        # 秩亏阈值 = Vandermonde 列数 = 次数 + 1（deg 列表时取 max+1）
        ncoef = (deg + 1) if isinstance(deg, int) else (builtins.max(deg) + 1)
        if rank < ncoef:
            import warnings
            warnings.warn("Polyfit may be poorly conditioned",
                          NX.exceptions.RankWarning, stacklevel=2)
        return cls(sol, domain=domain, window=window, symbol=symbol)

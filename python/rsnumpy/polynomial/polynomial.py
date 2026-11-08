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
    tensor=False：x.shape + roots.shape[1:]（roots 须 ≥2 维）。"""
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
        out_shape = rshape[1:] + xshape if tensor else xshape + rshape[1:]
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
        res = [0.0] * (ng * npts)
        for gi in range(ng):
            gidx = _idx_unflat(gi, gshape)
            for k in range(npts):
                v = 1.0
                for i in range(rshape[0]):
                    v *= (xflat[k] - _multiget(rval, (i,) + gidx))
                if tensor:
                    res[gi * npts + k] = v
                else:
                    res[k * ng + gi] = v
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
    """多项式根（Rust QR 伴矩阵特征值内核，降序约定）。"""
    NX = _NX()
    c = NX.asarray(c)
    cvals = c.tolist()
    if len(cvals) <= 1:
        return NX.array([])
    desc = list(reversed(cvals))
    return _wrap(_core().polyroots_rs(_raw(desc)))


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
    # 1 维 y → 1 维解（numpy 语义）；2 维 y 逐列
    Y = NX.array(yvals) if y.ndim == 1 else \
        NX.array([list(row) for row in yvals])
    sol, resids, rank, sv = NX.linalg.lstsq(NX.array(rows), Y, rcond=rcond)
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
    shape = tuple(x.shape)

    def rec(v, d):
        if d == len(shape) - 1:
            return [v ** i for i in range(deg + 1)]
        return [rec(u, d + 1) for u in v]

    data = rec(xv, 0) if shape else [[xv ** i for i in range(deg + 1)]]
    return NX.array(data)


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


def polyvander2d(x, y, deg):
    if len(deg) != 2:
        raise ValueError("deg must be a length-2 sequence")
    return _vander_nd((x, y), (deg[0], deg[1]))


def polyvander3d(x, y, z, deg):
    if len(deg) != 3:
        raise ValueError("deg must be a length-3 sequence")
    return _vander_nd((x, y, z), (deg[0], deg[1], deg[2]))


# ========== 多维求值 / 网格 ==========

def _polyvalnd(pts, c):
    NX = _NX()
    pts = [NX.asarray(p) for p in pts]
    if not pts:
        raise ValueError("pts must be non-empty")
    shape = tuple(pts[0].shape)
    for p in pts[1:]:
        if tuple(p.shape) != shape:
            raise ValueError("incompatible point dimensions in polyvalnd")
    c = NX.asarray(c)
    if c.ndim != len(pts):
        raise ValueError("coef array must have the same ndim as "
                         "number of points")
    cvals = c.tolist()
    degs = list(c.shape)
    flats = [p.ravel().tolist() for p in pts]
    n = len(flats[0])
    res = [0.0] * n
    for idx in _itertools.product(*[range(d) for d in degs]):
        cv = _multiget(cvals, idx)
        if cv == 0:
            continue
        for k in range(n):
            v = 1.0
            for a in range(len(pts)):
                if idx[a]:
                    v *= flats[a][k] ** idx[a]
            res[k] = res[k] + cv * v
    return NX.array(_unflat(res, shape))


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
    r = _dispatch((x1, x2, c), polygrid2d)
    if r is not None:
        return r
    NX = _NX()
    g1, g2 = NX.meshgrid(NX.asarray(x1), NX.asarray(x2))
    return polyval2d(g1, g2, c)


def polygrid3d(x1, x2, x3, c):
    NX = _NX()
    g1, g2, g3 = NX.meshgrid(NX.asarray(x1), NX.asarray(x2),
                             NX.asarray(x3))
    return polyval3d(g1, g2, g3, c)


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
        c = NX.asarray(c)
        if c.ndim > 1:
            c = c.ravel()
        elif c.ndim == 0:
            c = NX.array([c.tolist()])
        self._coef = c
        if domain is None:
            domain = NX.array([-1., 1.])
        if window is None:
            window = NX.array([-1., 1.])
        self._domain = NX.asarray(domain)
        self._window = NX.asarray(window)
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
        d = self._domain.tolist()
        w = self._window.tolist()
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
        return Polynomial(_NX().array(c), domain=self._domain,
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
        c = [(a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)
             for i in range(n)]
        return Polynomial(_NX().array(c), domain=self._domain,
                          window=self._window, symbol=self._symbol)

    def __neg__(self):
        return Polynomial(_NX().array([-v for v in self._coef.tolist()]),
                          domain=self._domain, window=self._window,
                          symbol=self._symbol)

    def _scaled(self, s):
        NX = _NX()
        vals = self._coef.tolist()
        if str(NX.asarray(vals).dtype) == 'object':
            c = NX.array([s * v for v in vals])
        else:
            c = self._coef * s
        return Polynomial(c, domain=self._domain, window=self._window,
                          symbol=self._symbol)

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            c = _py_mul(self._coef.tolist(), other._coef.tolist())
            return Polynomial(_NX().array(c), domain=self._domain,
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
        return Polynomial(_NX().array(c), domain=self._domain,
                          window=self._window, symbol=self._symbol)

    # ---- 导数 / 根 ----
    def deriv(self, m=1):
        c = _py_der(self._coef.tolist(), m)
        return Polynomial(_NX().array(c), domain=self._domain,
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
        ncoef = deg if isinstance(deg, int) else len(deg)
        if rank < ncoef:
            import warnings
            warnings.warn("Polyfit may be poorly conditioned",
                          NX.exceptions.RankWarning, stacklevel=2)
        return cls(sol, domain=domain, window=window, symbol=symbol)

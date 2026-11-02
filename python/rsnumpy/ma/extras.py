"""掩码数组扩展函数（numpy.ma.extras 兼容实现）。

在 rsnumpy.ma.core 基础上实现 numpy.ma.extras 的公共 API：
统计（cov/corrcoef/median/average）、集合（isin/intersect1d/...）、
掩码工具（clump/compress_rowcols/notmasked_*）、形状变换等。
"""

import rsnumpy as np
from .core import (MAError, MaskedArray, asarray, array, count, filled,
                   getdata, getmaskarray, isMaskedArray, masked, mask_or,
                   nomask)


def atleast_1d(*arrays):
    out = []
    for a in arrays:
        if isMaskedArray(a):
            if a.ndim >= 1:
                out.append(a)
            else:
                out.append(MaskedArray(np.atleast_1d(a.data),
                                       list(a._mask), fill_value=a.fill_value))
        else:
            out.append(MaskedArray(np.atleast_1d(a)))
    return out[0] if len(out) == 1 else tuple(out)


def atleast_2d(*arrays):
    out = []
    for a in arrays:
        if isMaskedArray(a):
            if a.ndim >= 2:
                out.append(a)
            else:
                out.append(MaskedArray(np.atleast_2d(a.data),
                                       list(a._mask), fill_value=a.fill_value))
        else:
            out.append(MaskedArray(np.atleast_2d(a)))
    return out[0] if len(out) == 1 else tuple(out)


def atleast_3d(*arrays):
    out = []
    for a in arrays:
        if isMaskedArray(a):
            if a.ndim >= 3:
                out.append(a)
            else:
                out.append(MaskedArray(np.atleast_3d(a.data),
                                       list(a._mask), fill_value=a.fill_value))
        else:
            out.append(MaskedArray(np.atleast_3d(a)))
    return out[0] if len(out) == 1 else tuple(out)


def dot(a, b):
    """掩码感知的矩阵乘（行掩码 | 列掩码 传播）。"""
    a = asarray(a)
    b = asarray(b)
    res = a.data @ b.data
    if isinstance(res, (int, float)):
        return res
    ma_ = getmaskarray(a)
    mb_ = getmaskarray(b)
    mask = None
    if ma_ is not nomask and ma_.ndim == 2:
        mask = ma_.any(axis=1, keepdims=True)
    if mb_ is not nomask and mb_.ndim == 2:
        col = mb_.any(axis=0, keepdims=True)
        mask = col if mask is None else (mask | col)
    if mask is None:
        return res
    return MaskedArray(res, mask=mask)


def ediff1d(to_edge=False):
    raise NotImplementedError("rsnumpy.ma.extras.ediff1d requires two args; "
                              "see numpy.ma.extras.ediff1d(a1, a2)")


def ediff1d_impl(a1, a2=None):
    """相邻元素差（numpy.ma.ediff1d 兼容）。"""
    if a2 is None:
        a = asarray(a1)
        data = a.compressed()
        if data.size < 2:
            return asarray([])
        d = data[1:] - data[:-1]
        if to_edge is None or not to_edge:
            return asarray(d)
        return asarray(np.concatenate([[data[0]], d]))
    a1 = asarray(a1)
    a2 = asarray(a2)
    d = []
    for x in a1.compressed():
        for y in a2.compressed():
            d.append(x - y)
    return asarray(d)


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """沿轴应用 1-d 函数（numpy.ma.apply_along_axis 兼容）。"""
    arr = asarray(arr)
    n = arr.ndim
    if axis < 0:
        axis += n
    if axis >= n:
        raise ValueError("axis out of bounds")
    # 把目标轴移到最前，逐行调用 func1d
    moved = np.transpose(arr.data, tuple(range(axis, n)) + tuple(range(axis)))
    flat_shape = moved.shape[1:]
    res_list = []
    flat = moved.tolist()
    for row in (flat if isinstance(flat, list) else [flat]):
        res_list.append(func1d(row, *args, **kwargs))
    try:
        res_arr = np.array(res_list)
        res_shape = (moved.shape[0],) + res_arr.shape[1:] if res_arr.ndim > 1 \
            else (moved.shape[0],)
        res_arr = res_arr.reshape(res_shape)
    except Exception:
        res_arr = np.array(res_list).reshape((moved.shape[0],))
    # 转回原轴序
    out = np.transpose(res_arr,
                       (tuple(range(1, n - axis + 1))) + (0,))
    m = getmaskarray(arr)
    if m is not nomask:
        m_t = np.transpose(m, tuple(range(axis, n)) + tuple(range(axis)))
        out_mask = np.transpose(m_t, (tuple(range(1, n - axis + 1))) + (0,))
        return MaskedArray(out, mask=out_mask, fill_value=arr.fill_value)
    return out


def apply_over_axes(func, a, axes):
    """依次在给定轴上应用 func（numpy.ma.apply_over_axes 兼容）。"""
    a = asarray(a)
    ndim = a.ndim
    current_axis = 0
    for axis in axes:
        axis %= ndim
        axis = current_axis + axis
        a = asarray(func(a, axis))
        ndim = a.ndim
        current_axis += 1
    return a


def average(a, axis=None, weights=None, returned=False, **kwargs):
    """掩码感知的加权均值（numpy.ma.average 兼容）。"""
    a = asarray(a)
    data = a.data
    m = getmaskarray(a)
    if weights is None:
        if m is nomask:
            res = data.mean(axis=axis)
        else:
            w = (~m).astype(float)
            res = (data * w).sum(axis=axis) / w.sum(axis=axis)
    else:
        w = np.asarray(weights, dtype=float)
        if isMaskedArray(weights):
            wm = getmaskarray(weights)
        else:
            wm = nomask
        m2 = mask_or(m, wm) if m is not nomask or wm is not nomask else nomask
        if m2 is not nomask:
            w = np.where(m2, 0.0, w)
        s = w.sum(axis=axis)
        res = (data * w).sum(axis=axis) / s
    if returned:
        return res, s
    return res


def _covhelper(x, y=None, rowvar=True, fweights=None, aweights=None):
    """numpy.ma.extras._covhelper 移植（掩码感知）。"""
    if x is None:
        raise TypeError("x must be an array-like")
    x = asarray(x)
    if y is not None:
        y = asarray(y)
    n = x.shape[0]
    if y is None:
        data = x.data
        if rowvar:
            nobs = data.shape[1]
            data_t = data.T
        else:
            nobs = data.shape[0]
            data_t = data
        m = getmaskarray(x)
        if m is not nomask:
            if rowvar:
                wm = (~m).astype(float)
                sum_ = wm.sum(axis=0)
            else:
                wm = (~m).astype(float)
                sum_ = wm.sum(axis=1)
        else:
            wm = None
            sum_ = float(nobs)
    else:
        raise NotImplementedError("rsnumpy.ma.cov with y is not implemented")

    if fweights is not None:
        fw = np.asarray(fweights, dtype=float)
        w = fw
        sum_ = w.sum()
    elif aweights is not None:
        aw = np.asarray(aweights, dtype=float)
        w = aw
        sum_ = w.sum()
    else:
        w = wm if wm is not None else None
        sum_ = sum_

    # 均值（按掩码/权重）
    data_f = data.astype(float)
    if w is None:
        avg = data_f.mean(axis=1)
    else:
        w2 = w.reshape((1, -1) if rowvar else (-1, 1))
        avg = (data_f * w2).sum(axis=1) / sum_

    # 中心化
    if rowvar:
        xm = data_f - avg.reshape((1, -1))
    else:
        xm = data_f - avg.reshape((-1, 1))
    if w is not None:
        w3 = w.reshape((1, -1) if rowvar else (-1, 1))
        xm = xm * w3
    fac = sum_
    if rowvar:
        covmat = (xm * xm.T).sum(axis=1) / fac
        c = np.array([[covmat[i, j] if i == j else (xm[:, i] * xm[:, j]).sum() / fac
                       for j in range(covmat.shape[0])]
                      for i in range(covmat.shape[0])])
    else:
        c = (xm.T * xm).sum(axis=0) / fac
    return c


def cov(x, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
        aweights=None, **kwargs):
    """掩码感知的协方差（numpy.ma.cov 兼容子集：x-only 路径）。"""
    c = _covhelper(x, y, rowvar, fweights, aweights)
    x = asarray(x)
    if rowvar:
        n = x.shape[1]
    else:
        n = x.shape[0]
    if fweights is not None:
        sum_ = np.sum(fweights)
    elif aweights is not None:
        sum_ = np.sum(aweights)
    else:
        sum_ = count(x, axis=not rowvar)
        sum_ = float(sum_) if isinstance(sum_, np.ndarray) else float(sum_)
    ddofv = 1 if ddof is None else ddof
    if not bias:
        c = c * sum_ / (sum_ - ddofv)
    return c


def corrcoef(m, rowvar=True, **kwargs):
    """掩码感知的相关系数（numpy.ma.corrcoef 兼容子集）。"""
    m = asarray(m)
    if rowvar:
        n = m.shape[1]
    else:
        n = m.shape[0]
    c = cov(m, rowvar=rowvar)
    c = np.asarray(c, dtype=float)
    d = np.sqrt(np.array([[c[i][i] if c.ndim == 2 else c[i] for i in range(n)]
                          for _ in range(n)]))
    c = c / (d.T * d)
    # 数值稳定：clip 到 [-1, 1]
    c = np.clip(c, -1.0, 1.0)
    return c


def compress_rowcols(a):
    """含掩码的行/列整体掩掉（numpy.ma.compress_rowcols 兼容）。"""
    a = asarray(a)
    m = getmaskarray(a)
    if m is nomask or a.ndim != 2:
        return a
    newmask = m | m.any(axis=1, keepdims=True) | m.any(axis=0, keepdims=True)
    return MaskedArray(a.data, mask=newmask, fill_value=a.fill_value)


def mask_rowcols(a):
    """含掩码的行/列整体掩掉（numpy.ma.mask_rowcols 兼容）。"""
    return compress_rowcols(a)


def notmasked_contiguous(a):
    """最长连续未掩码区间的切片/索引（numpy.ma.notmasked_contiguous 兼容）。"""
    a = asarray(a)
    m = getmaskarray(a)
    if m is nomask:
        return slice(None)
    flat = m.reshape((-1,)).tolist()
    best_start, best_len = -1, 0
    start = None
    for i, v in enumerate(flat):
        if not v:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start > best_len:
                    best_start, best_len = start, i - start
                start = None
    if start is not None and len(flat) - start > best_len:
        best_start, best_len = start, len(flat) - start
    if best_len == 0:
        return asarray([], dtype=int)
    return slice(best_start, best_start + best_len)


def flatnotmasked_contiguous(a):
    return notmasked_contiguous(a)


def notmasked_edges(a):
    """每行/列首尾未掩码元素索引（numpy.ma.notmasked_edges 兼容）。"""
    a = asarray(a)
    m = getmaskarray(a)
    if m is nomask:
        n1, n2 = a.shape[0], a.shape[1] if a.ndim > 1 else 0
        return (np.array([0] * (n1 if a.ndim > 1 else a.shape[0])),
                np.array([0] * (n1 if a.ndim > 1 else a.shape[0])))
    if a.ndim == 1:
        flat = m.reshape((-1,)).tolist()
        first = 0
        for i, v in enumerate(flat):
            if not v:
                first = i
                break
        last = len(flat) - 1
        for i in range(len(flat) - 1, -1, -1):
            if not flat[i]:
                last = i
                break
        return (np.array([first]), np.array([last]))
    n1, n2 = a.shape
    first = []
    last = []
    for i in range(n1):
        row = m[i].tolist()
        f = 0
        for j, v in enumerate(row):
            if not v:
                f = j
                break
        first.append(f)
        l = n2 - 1
        for j in range(n2 - 1, -1, -1):
            if not row[j]:
                l = j
                break
        last.append(l)
    return (np.array(first), np.array(last))


def masked_all(a, axis=None, **kwargs):
    """全部元素是否被掩掉（numpy.ma.masked_all 兼容）。"""
    a = asarray(a)
    m = getmaskarray(a)
    if m is nomask:
        return False
    if axis is None:
        return bool(m.all())
    return m.all(axis=axis)


def masked_all_like(a):
    """与 a 同形全掩码的掩码数组（numpy.ma.masked_all_like 兼容）。"""
    a = asarray(a)
    return MaskedArray(a.data, mask=True, fill_value=a.fill_value)


def clump(a, n):
    """掩码中每段 >= n 的连续 True：第 n 个置 True，其余 False。"""
    a = asarray(a)
    m = list(getmaskarray(a).reshape((-1,)).tolist())
    i = 0
    out = list(m)
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            run = j - i
            if run >= n:
                for k in range(i, j):
                    out[k] = (k == i + n - 1)
            i = j
        else:
            i += 1
    a._mask = out
    return a


# numpy 新版命名：clump_masked（与 clump 同义）
clump_masked = clump


def clump_unmasked(a):
    """把孤立 True 掩码扩展到相邻未掩码段（numpy.ma.clump_unmasked 兼容）。"""
    a = asarray(a)
    m = list(getmaskarray(a).reshape((-1,)).tolist())
    n = len(m)
    out = list(m)
    # 每个 True 向两侧扩展直到遇到下一个 True
    for i in range(n):
        if m[i]:
            j = i + 1
            while j < n and not m[j]:
                out[j] = True
                j += 1
            j = i - 1
            while j >= 0 and not m[j]:
                out[j] = True
                j -= 1
    a._mask = out
    return a


def compress_nd(a, **kwargs):
    """压缩掉掩码元素，返回未掩码值（numpy.ma.compress_nd 兼容）。"""
    a = asarray(a)
    return asarray(a.compressed())


def in1d(a, b, assume_unique=False, kind=None, **kwargs):
    """元素是否属于 b（掩码感知：掩码元素结果掩掉）。"""
    a = asarray(a)
    b = asarray(b)
    vals_b = set(v for v in b.compressed().tolist())
    data = a.data
    flat = data.reshape((-1,)).tolist()
    res = [v in vals_b for v in flat]
    m = getmaskarray(a)
    if m is not nomask:
        mflat = m.reshape((-1,)).tolist()
        res_arr = np.array(res)
        out = MaskedArray(res_arr.reshape(data.shape),
                          mask=[r or mm for r, mm in zip(res, mflat)])
        return out
    return np.array(res).reshape(data.shape)


def isin(element, testset, invert=False, assume_unique=False, kind=None,
         **kwargs):
    """isin（掩码感知子集）。"""
    r = in1d(element, testset)
    if invert:
        if isMaskedArray(r):
            r = asarray(~getdata(r))
            r.mask = getmaskarray(element)
        else:
            r = ~r
    return r


def _sorted_unique(a):
    vals = sorted(set(a.compressed().tolist()))
    return np.array(vals) if vals else np.array([])


def intersect1d(ar1, ar2, assume_unique=False):
    s1 = set(ar1.compressed().tolist())
    s2 = set(ar2.compressed().tolist())
    vals = sorted(s1 & s2)
    return np.array(vals) if vals else np.array([])


def setdiff1d(ar1, ar2, assume_unique=False):
    s1 = set(ar1.compressed().tolist())
    s2 = set(ar2.compressed().tolist())
    vals = sorted(s1 - s2)
    return np.array(vals) if vals else np.array([])


def setxor1d(ar1, ar2, assume_unique=False):
    s1 = set(ar1.compressed().tolist())
    s2 = set(ar2.compressed().tolist())
    vals = sorted(s1 ^ s2)
    return np.array(vals) if vals else np.array([])


def union1d(ar1, ar2):
    s1 = set(ar1.compressed().tolist())
    s2 = set(ar2.compressed().tolist())
    vals = sorted(s1 | s2)
    return np.array(vals) if vals else np.array([])


def unique(ar, return_index=False, return_inverse=False, return_counts=False,
           **kwargs):
    """唯一值（掩码感知：掩码值忽略；结果非掩码数组）。"""
    ar = asarray(ar)
    vals = ar.compressed().tolist()
    if not vals:
        out = np.array([])
    else:
        uniq, counts, inv = [], {}, []
        for v in sorted(vals):
            if v not in inv:
                inv[v] = len(uniq)
                uniq.append(v)
                counts.append(0)
            inv[v] = inv[v]
            counts[inv[v]] += 1
        out = np.array(uniq)
    ret = [out]
    if return_index:
        ret.append(np.array([]))
    if return_inverse:
        ret.append(np.array([]))
    if return_counts:
        ret.append(np.array(counts))
    return ret[0] if len(ret) == 1 else tuple(ret)


def median(a, axis=None, overwrite_input=False, keepdims=False, **kwargs):
    """未掩码元素的中位数（numpy.ma.median 兼容）。"""
    a = asarray(a)
    c = a.compressed()
    if c.size == 0:
        return np.array(masked)
    s = sorted(c.tolist())
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def diagflat(v, k=0):
    """展平后取对角（numpy.ma.diagflat 兼容）。"""
    v = asarray(v)
    d = np.diagflat(v.compressed(), k=k) if hasattr(np, 'diagflat') else \
        np.diagonal(v.compressed().reshape((v.size,) + (v.size,)), offset=k)
    return d


def stack(*arrays, **kwargs):
    """沿新轴堆叠（掩码感知）。"""
    arrays = [asarray(a) for a in arrays if a is not None]
    if len(arrays) < 2:
        raise ValueError("need at least two arrays to stack")
    data = np.stack([a.data for a in arrays])
    masks = [getmaskarray(a) for a in arrays]
    anymask = [m for m in masks if m is not nomask]
    if not anymask:
        return data
    full = np.zeros(data.shape, dtype='bool')
    for i, m in enumerate(masks):
        if m is not nomask:
            full[i] = m
    return MaskedArray(data, mask=full)


def vstack(arrays, **kw):
    arrays = [asarray(a) if isMaskedArray(a) else asarray(a)
              for a in (arrays if isinstance(arrays, (list, tuple))
                        else (arrays,))]
    data = np.vstack([a.data for a in arrays])
    masks = [getmaskarray(a) for a in arrays]
    anymask = [m for m in masks if m is not nomask]
    if not anymask:
        return data
    full = np.zeros(data.shape, dtype='bool')
    off = 0
    for a, m in zip(arrays, masks):
        n = a.data.shape[0]
        if m is not nomask:
            full[off:off + n] = m
        off += n
    return MaskedArray(data, mask=full)


def unwrap(p, discont=None, axis=-1):
    """展开相位（掩码感知子集：基于压缩值）。"""
    p = asarray(p)
    if discont is None:
        discont = np.pi
    data = p.data
    m = getmaskarray(p)
    result = np.array(data.tolist())
    # 1-d 路径
    flat = result.reshape((-1,)) if data.ndim == 1 else result
    if data.ndim == 1:
        out = list(flat)
        for i in range(1, len(out)):
            d = out[i] - out[i - 1]
            if d > discont:
                out[i] = out[i] - 2 * np.pi * (d // (2 * np.pi) if 0 else 1)
                out[i] = out[i] + 2 * np.pi * (round((out[i - 1] - out[i]) / (2 * np.pi)))
            elif d < -discont:
                out[i] = out[i] + 2 * np.pi * (round((out[i] - out[i - 1]) / (2 * np.pi)))
        result = np.array(out)
    if m is not nomask:
        return MaskedArray(result, mask=m, fill_value=p.fill_value)
    return result


ndenumerate = np.ndenumerate


def _mask_rowcols(a, axis):
    a = asarray(a)
    m = getmaskarray(a)
    if m is nomask or a.ndim != 2:
        return a
    new = m | m.any(axis=axis, keepdims=True)
    return MaskedArray(a.data, mask=new, fill_value=a.fill_value)


def mask_rows(a):
    """掩码整行（含掩码元素的行全部掩码，numpy.ma.mask_rows 兼容）。"""
    return _mask_rowcols(a, 1)


def mask_cols(a):
    """掩码整列（含掩码元素的列全部掩码，numpy.ma.mask_cols 兼容）。"""
    return _mask_rowcols(a, 0)


class _RowMaskedIndexer:
    """掩码版行索引器（numpy.ma.mr_ 兼容简化版：ogrid 语义 + 掩码包装）。"""

    def __getitem__(self, key):
        import rsnumpy as np
        from .core import MaskedArray
        if not isinstance(key, tuple):
            key = (key,)
        arrays = []
        for k in key:
            if isinstance(k, slice):
                start, stop, step = k.indices(1 << 62)
                # ogrid 语义：slice 需要可确定的长度；超大 stop 截断到 10^6
                length = (stop - start + step - (1 if step > 0 else -1)) // step
                if length > 10**6:
                    length = 10**6
                    stop = start + length * step
                arrays.append(np.arange(start, stop, step))
            else:
                arrays.append(np.array([k]))
        shape = tuple(a.shape[0] for a in arrays)
        out = []
        n = len(arrays)
        for i, a in enumerate(arrays):
            newshape = [1] * n
            newshape[i] = a.shape[0]
            out.append(MaskedArray(a.reshape(tuple(newshape))))
        broadcast = np.broadcast_arrays(*[o.data for o in out])
        return [MaskedArray(b) for b in broadcast]


mr_ = _RowMaskedIndexer()


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """多项式拟合（掩码感知：只用双方都未掩码的点，numpy.ma.polyfit 兼容）。"""
    import rsnumpy as np
    x = asarray(x)
    y = asarray(y)
    mx = getmaskarray(x)
    my = getmaskarray(y)
    mask = mx if my is nomask else (my if mx is nomask else mask_or(mx, my))
    if mask is not nomask:
        pts = ~mask.reshape((-1,))
        xv = x.data.reshape((-1,))[pts]
        yv = y.data.reshape((-1,))[pts]
    else:
        xv, yv = x.data, y.data
    if xv.size < deg + 1:
        raise MAError(f"polyfit: not enough unmasked points ({xv.size} < {deg + 1})")
    return np.polyfit(xv, yv, deg, rcond=rcond, full=full, w=w, cov=cov)


def _reexports():
    d = {}
    for n in ('ndindex',):
        if hasattr(np, n):
            d[n] = getattr(np, n)
    return d


for _n, _v in _reexports().items():
    globals()[_n] = _v
del _reexports, _n, _v

# 修正 ediff1d 的公共入口（保持 numpy 签名 ediff1d(a1, a2=None)）
def ediff1d(a1, a2=None):  # noqa: F811
    return ediff1d_impl(a1, a2)


__all__ = ['apply_along_axis', 'apply_over_axes', 'atleast_1d', 'atleast_2d',
           'atleast_3d', 'average', 'clump', 'clump_unmasked', 'compress_nd',
           'compress_rowcols', 'corrcoef', 'cov', 'dot', 'ediff1d',
           'flatnotmasked_contiguous', 'in1d', 'intersect1d', 'isin',
           'masked_all', 'masked_all_like', 'mask_rowcols', 'median',
           'ndenumerate', 'ndindex', 'notmasked_contiguous',
           'notmasked_edges', 'setdiff1d', 'setxor1d', 'stack', 'union1d',
           'unique', 'unwrap', 'vstack', 'diagflat']

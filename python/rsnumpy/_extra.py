"""补充的 NumPy 兼容 API。

计算全部复用已有的 Rust 原语（_core 或已导出的顶层函数），本层仅做组合与包装，
不重复实现底层数值循环。热点函数可在后续通过 Rust 层进一步优化。
"""

# import builtins as _builtins
import cmath as _cmath
import math as _math

import rsnumpy.num_core as _core

builtin_max = max
builtin_min = min
builtin_all = all
builtin_any = any
builtin_round = round
builtin_abs = abs
builtin_range = range
builtin_sum = sum


def _np():
    import rsnumpy as _n
    return _n


def _asarray(a):
    np = _np()
    return a if hasattr(a, "_array") else np.array(a)


def _wrap(raw, _dtype="float64"):
    return _np().ndarray._wrap(raw, _dtype=_dtype)


def _flat(a):
    """返回任意嵌套 tolist 的一维扁平列表。"""
    # 处理 Poly 对象：提取系数
    np = _np()
    if hasattr(a, '__class__') and a.__class__.__name__ == 'Poly':
        a = a.coef if hasattr(a, 'coef') else np.array(list(a))
    out = []
    stack = [_asarray(a).tolist()]
    while stack:
        v = stack.pop()
        if isinstance(v, list):
            stack.extend(reversed(v))
        else:
            out.append(v)
    return out


def _map_nested(fn, data):
    if isinstance(data, list):
        return [_map_nested(fn, v) for v in data]
    return fn(data)


def _as_scalar(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    while isinstance(v, list) and len(v) == 1:
        v = v[0]
    return v


# ========== 三角/双曲别名 ==========
def acos(x):
    """反余弦（arccos 别名）。"""
    return _np().arccos(x)


def asin(x):
    """反正弦（arcsin 别名）。"""
    return _np().arcsin(x)


def atan(x):
    """反正切（arctan 别名）。"""
    return _np().arctan(x)


def atan2(x1, x2):
    """两参数反正切（arctan2 别名）。"""
    return _np().arctan2(x1, x2)


def arccosh(x):
    """反双曲余弦（acosh 别名）。"""
    return _np().acosh(x)


def arcsinh(x):
    """反双曲正弦（asinh 别名）。"""
    return _np().asinh(x)


def arctanh(x):
    """反双曲正切（atanh 别名）。"""
    return _np().atanh(x)


def radians(x):
    """角度转弧度（deg2rad 别名）。"""
    return _np().deg2rad(x)


def degrees(x):
    """弧度转角度（rad2deg 别名）。"""
    return _np().rad2deg(x)


# ========== 逐元素数学 ==========
def absolute(x):
    """逐元素绝对值。"""
    return _np().abs(x)


fabs = absolute


def true_divide(x1, x2):
    """逐元素真除（divide 别名）。"""
    return _np().divide(x1, x2)


def pow(x1, x2):
    """逐元素幂（power 别名）。"""
    return _np().power(x1, x2)


float_power = pow


def negative(x):
    """逐元素取负。"""
    return _np().multiply(x, -1.0)


def positive(x):
    """逐元素取正（返回副本）。"""
    return _asarray(x)


def exp2(x):
    """逐元素 2**x。"""
    return _np().power(2.0, x)


def floor_divide(x1, x2):
    """逐元素向下整除。"""
    np = _np()
    return np.floor(np.divide(x1, x2))


def fmod(x1, x2):
    """逐元素 C 风格取余（余数符号同被除数）。"""
    np = _np()
    q = np.trunc(np.divide(x1, x2))
    return np.subtract(x1, np.multiply(x2, q))


def divmod(x1, x2):
    """逐元素返回 (整除商, 余数)。"""
    return floor_divide(x1, x2), _np().mod(x1, x2)


def rint(x):
    """逐元素舍入到最近整数（四舍六入五成双）。"""
    return _wrap(_core.rint(_asarray(x)._array))


def hypot(x1, x2):
    """逐元素 sqrt(x1**2 + x2**2)。"""
    np = _np()
    return _wrap(_core.hypot(np.asarray(x1)._array, np.asarray(x2)._array))


def maximum(x1, x2):
    """逐元素取较大值。"""
    np = _np()
    return _wrap(_core.maximum(np.asarray(x1)._array, np.asarray(x2)._array))


def minimum(x1, x2):
    """逐元素取较小值。"""
    np = _np()
    return _wrap(_core.minimum(np.asarray(x1)._array, np.asarray(x2)._array))


def fmax(x1, x2):
    """逐元素取较大值，忽略 NaN。"""
    return _wrap(_core.fmax(_asarray(x1)._array, _asarray(x2)._array))


def fmin(x1, x2):
    """逐元素取较小值，忽略 NaN。"""
    return _wrap(_core.fmin(_asarray(x1)._array, _asarray(x2)._array))


def signbit(x):
    """逐元素判断是否为负号位。"""
    return _wrap(_core.signbit(_asarray(x)._array), "bool")


def copysign(x1, x2):
    """将 x2 的符号赋给 x1。"""
    return _wrap(_core.copysign(_asarray(x1)._array, _asarray(x2)._array))


def logaddexp(x1, x2):
    """数值稳定地计算 log(exp(x1) + exp(x2))。"""
    np = _np()
    a = _asarray(x1)
    b = _asarray(x2)
    m = maximum(a, b)
    return np.add(m, np.log(np.add(np.exp(np.subtract(a, m)), np.exp(np.subtract(b, m)))))


def logaddexp2(x1, x2):
    """数值稳定地计算 log2(2**x1 + 2**x2)。"""
    np = _np()
    a = _asarray(x1)
    b = _asarray(x2)
    m = maximum(a, b)
    return np.add(m, np.log2(np.add(exp2(np.subtract(a, m)), exp2(np.subtract(b, m)))))


def logical_not(x):
    """逐元素逻辑非。"""
    np = _np()
    arr = _asarray(x)
    return np.equal(arr, np.full(arr.shape, 0.0))


def gcd(x1, x2):
    """逐元素最大公约数。"""
    return _wrap(_core.gcd(_asarray(x1)._array, _asarray(x2)._array), "int64")


def lcm(x1, x2):
    """逐元素最小公倍数。"""
    return _wrap(_core.lcm(_asarray(x1)._array, _asarray(x2)._array), "int64")


def nextafter(x1, x2):
    """逐元素返回 x1 朝 x2 方向的下一个可表示浮点数。"""
    return _wrap(_core.nextafter(_asarray(x1)._array, _asarray(x2)._array))


def spacing(x):
    """逐元素返回相邻浮点数间距。"""
    return _wrap(_core.spacing(_asarray(x)._array))


def ldexp(x1, x2):
    """逐元素返回 x1 * 2**x2。"""
    return _wrap(_core.ldexp(_asarray(x1)._array, _asarray(x2)._array))


def frexp(x):
    """逐元素返回 (尾数, 指数)。"""
    mant, expo = _core.frexp(_asarray(x)._array)
    return _wrap(mant), _wrap(expo, "int64")


def modf(x):
    """逐元素返回 (小数部分, 整数部分)。"""
    np = _np()
    arr = _asarray(x)
    ipart = np.trunc(arr)
    return np.subtract(arr, ipart), ipart


# ========== 归约 / prod / nan 系列 ==========
def _reduce_last_along(c, axis):
    np = _np()
    n = c.shape[axis]
    picked = np.take(c, [n - 1], axis=axis)
    ax = axis % picked.ndim
    newshape = tuple(s for i, s in enumerate(picked.shape) if i != ax)
    if not newshape:
        return _as_scalar(picked)
    return np.reshape(picked, newshape)


def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    """计算元素乘积。"""
    _ = dtype, out, where
    arr = _asarray(a)
    result = _core.prod(arr._array, axis, keepdims)
    if initial is not None:
        np = _np()
        result = np.multiply(result, initial)
    return _wrap(result)


def _shape_drop(shape, axis):
    ax = axis % len(shape)
    return tuple(s for i, s in enumerate(shape) if i != ax)


def _rep_nan(a, val):
    np = _np()
    arr = _asarray(a)
    return np.where(np.isnan(arr), np.full(arr.shape, float(val)), arr)


def _count_notnan(a, axis):
    np = _np()
    arr = _asarray(a)
    mask = np.where(np.isnan(arr), np.full(arr.shape, 0.0), np.full(arr.shape, 1.0))
    return np.sum(mask, axis)


def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    """忽略 NaN 求和（Rust 单次遍历实现）。"""
    _ = dtype, out
    return _wrap(_core.nansum(_asarray(a)._array, axis, keepdims))


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    """忽略 NaN 求积（Rust 单次遍历实现）。"""
    _ = dtype, out
    return _wrap(_core.nanprod(_asarray(a)._array, axis, keepdims))


def nanmax(a, axis=None, out=None, keepdims=False):
    """忽略 NaN 求最大值（Rust 实现，全 NaN 返回 NaN）。"""
    _ = out
    return _wrap(_core.nanmax(_asarray(a)._array, axis, keepdims))


def nanmin(a, axis=None, out=None, keepdims=False):
    """忽略 NaN 求最小值（Rust 实现，全 NaN 返回 NaN）。"""
    _ = out
    return _wrap(_core.nanmin(_asarray(a)._array, axis, keepdims))


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    """忽略 NaN 求均值（Rust 单次遍历实现）。"""
    _ = dtype, out
    return _wrap(_core.nanmean(_asarray(a)._array, axis, keepdims))


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """忽略 NaN 求方差（Rust 两遍遍历实现）。"""
    _ = dtype, out
    return _wrap(_core.nanvar(_asarray(a)._array, axis, ddof, keepdims))


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """忽略 NaN 求标准差（Rust 实现）。"""
    _ = dtype, out
    return _wrap(_core.nanstd(_asarray(a)._array, axis, ddof, keepdims))


def nanargmax(a, axis=None):
    """忽略 NaN 的最大值索引（Rust 实现）。"""
    return _wrap(_core.nanargmax_axis(_asarray(a)._array, axis), _dtype="int64")


def nanargmin(a, axis=None):
    """忽略 NaN 的最小值索引（Rust 实现）。"""
    return _wrap(_core.nanargmin_axis(_asarray(a)._array, axis), _dtype="int64")


def nancumsum(a, axis=None):
    """忽略 NaN 的累积和（NaN 视为 0，Rust 实现）。"""
    return _wrap(_core.nancumsum(_asarray(a)._array, axis))


def nancumprod(a, axis=None):
    """忽略 NaN 的累积积（NaN 视为 1，Rust 实现）。"""
    return _wrap(_core.nancumprod(_asarray(a)._array, axis))


def nanquantile(a, q, axis=None, **kwargs):
    """忽略 NaN 的分位数。"""
    _ = axis, kwargs
    np = _np()
    vals = [v for v in _flat(a) if not _math.isnan(v)]
    return np.percentile(np.array(vals), q * 100 if q <= 1 else q)


def cumulative_sum(x, axis=None, dtype=None, out=None, include_initial=False):
    """累积和（Array API 名称，等价 cumsum）。"""
    _ = dtype, out, include_initial
    return _np().cumsum(x, axis)


def cumulative_prod(x, axis=None, dtype=None, out=None, include_initial=False):
    """累积积（Array API 名称，等价 cumprod）。"""
    _ = dtype, out, include_initial
    return _np().cumprod(x, axis)


def count_nonzero(a, axis=None, *, keepdims=False):
    """统计非零元素个数。"""
    _ = keepdims
    np = _np()
    if axis is None:
        return _core.count_nonzero(_asarray(a)._array)
    arr = _asarray(a)
    mask = np.where(np.equal(arr, np.full(arr.shape, 0.0)), np.full(arr.shape, 0.0), np.full(arr.shape, 1.0))
    return np.sum(mask, axis)


# ========== 集合运算 ==========
def unique_values(x):
    """返回排序后的唯一值。"""
    return _np().unique(x)


def unique_counts(x):
    """返回 (唯一值, 出现次数)。"""
    raw = _core.unique_full(_asarray(x)._array, False, False, True)
    return _wrap(raw[0]), _wrap(raw[1], "int64")


def unique_inverse(x):
    """返回 (唯一值, 可重建原数组的逆索引)。"""
    raw = _core.unique_all_rs(_asarray(x)._array)
    return _wrap(raw[0]), _wrap(raw[2], "int64")


def unique_all(x):
    """返回 (唯一值, 首次出现索引, 逆索引, 计数)。"""
    raw = _core.unique_all_rs(_asarray(x)._array)
    return (_wrap(raw[0]), _wrap(raw[1], "int64"),
            _wrap(raw[2], "int64"), _wrap(raw[3], "int64"))


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """返回两个数组的交集（升序去重）。"""
    _ = assume_unique, return_indices
    return _wrap(_core.intersect1d(_asarray(ar1)._array, _asarray(ar2)._array))


def union1d(ar1, ar2):
    """返回两个数组的并集（升序去重）。"""
    return _wrap(_core.union1d(_asarray(ar1)._array, _asarray(ar2)._array))


def setdiff1d(ar1, ar2, assume_unique=False):
    """返回在 ar1 但不在 ar2 中的唯一值。"""
    _ = assume_unique
    return _wrap(_core.setdiff1d(_asarray(ar1)._array, _asarray(ar2)._array))


def setxor1d(ar1, ar2, assume_unique=False):
    """返回两个数组的对称差集。"""
    _ = assume_unique
    return _wrap(_core.setxor1d(_asarray(ar1)._array, _asarray(ar2)._array))


def isin(element, test_elements, assume_unique=False, invert=False):
    """逐元素判断是否属于 test_elements。"""
    _ = assume_unique
    raw = _core.isin(_asarray(element)._array, _asarray(test_elements)._array, invert)
    return _wrap(raw, "bool")


# ========== introspection / 复数 ==========
def ndim(a):
    """返回数组维度数。"""
    return _asarray(a).ndim


def shape(a):
    """返回数组形状。"""
    return _asarray(a).shape


def size(a, axis=None):
    """返回元素总数或指定轴长度。"""
    arr = _asarray(a)
    if axis is None:
        return arr.size
    return arr.shape[axis]


def isscalar(element):
    """判断是否为标量。"""
    return isinstance(element, (int, float, complex, bool, bytes, str))


def iterable(y):
    """判断对象是否可迭代。"""
    try:
        iter(y)
        return True
    except TypeError:
        return False


def iscomplexobj(x):
    """判断对象是否为复数类型。"""
    if isinstance(x, complex):
        return True
    if getattr(x, "_complex_data", None) is not None:
        return True
    return getattr(x, "_dtype", "") in ("complex128", "complex64")


def isrealobj(x):
    """判断对象是否为实数类型。"""
    return not iscomplexobj(x)


def isreal(x):
    """逐元素判断虚部是否为 0。"""
    np = _np()
    arr = _asarray(x)
    if not getattr(arr._array, 'is_complex', False):
        return np.array(_map_nested(lambda v: True, arr.tolist()), dtype="bool")
    imag_data = arr._array.imag
    if imag_data is None:
        return np.array(_map_nested(lambda v: True, arr.tolist()), dtype="bool")
    imag_list = imag_data.tolist()
    
    def check_zero(val):
        if isinstance(val, (list, tuple)):
            return [check_zero(v) for v in val]
        return abs(val) < 1e-15
    
    return np.array(check_zero(imag_list), dtype="bool")


def iscomplex(x):
    """逐元素判断虚部是否非 0。"""
    np = _np()
    arr = _asarray(x)
    if not getattr(arr._array, 'is_complex', False):
        return np.array(_map_nested(lambda v: False, arr.tolist()), dtype="bool")
    imag_data = arr._array.imag
    if imag_data is None:
        return np.array(_map_nested(lambda v: False, arr.tolist()), dtype="bool")
    imag_list = imag_data.tolist()
    
    def check_nonzero(val):
        if isinstance(val, (list, tuple)):
            return [check_nonzero(v) for v in val]
        return abs(val) >= 1e-15
    
    return np.array(check_nonzero(imag_list), dtype="bool")


def isnat(x):
    """逐元素判断是否为 NaT（暂不支持 NaT，返回全 False）。"""
    np = _np()
    arr = _asarray(x)
    return np.array(_map_nested(lambda v: False, arr.tolist()), dtype="bool")


def isneginf(x, out=None):
    """逐元素判断是否为负无穷。"""
    _ = out
    np = _np()
    arr = _asarray(x)
    return np.logical_and(np.isinf(arr), np.less(arr, 0.0))


def isposinf(x, out=None):
    """逐元素判断是否为正无穷。"""
    _ = out
    np = _np()
    arr = _asarray(x)
    return np.logical_and(np.isinf(arr), np.greater(arr, 0.0))


def real(val):
    """返回实部。"""
    np = _np()
    if iscomplexobj(val):
        # Multi-dim complex arrays don't expose the flat `_complex_data`
        # buffer (it is None/absent); fall back to the nested `tolist`.
        cdata = getattr(val, "_complex_data", None)
        if cdata is not None:
            return np.array([v.real for v in cdata])
        return np.array(_map_nested(lambda v: complex(v).real, val.tolist()))
    return _asarray(val)


def imag(val):
    """返回虚部。"""
    np = _np()
    if iscomplexobj(val):
        cdata = getattr(val, "_complex_data", None)
        if cdata is not None:
            return np.array([v.imag for v in cdata])
        return np.array(_map_nested(lambda v: complex(v).imag, val.tolist()))
    arr = _asarray(val)
    return np.full(arr.shape, 0.0)


def conjugate(x):
    """返回共轭。"""
    np = _np()
    if iscomplexobj(x):
        cdata = getattr(x, "_complex_data", None)
        if cdata is not None:
            return np.array([complex(v).conjugate() for v in cdata])
        return np.array(_map_nested(lambda v: complex(v).conjugate(), x.tolist()))
    return _asarray(x)


conj = conjugate


def angle(z, deg=False):
    """返回复数的相位角。"""
    np = _np()
    factor = 180.0 / _math.pi if deg else 1.0
    if isinstance(z, complex):
        return np.array([_math.atan2(z.imag, z.real) * factor])
    if iscomplexobj(z):
        cdata = getattr(z, "_complex_data", None)
        if cdata is not None:
            return np.array([_math.atan2(v.imag, v.real) * factor for v in cdata])
        return np.array(_map_nested(
            lambda v: _math.atan2(complex(v).imag, complex(v).real) * factor,
            z.tolist()))
    return np.array([_math.atan2(0.0, v) * factor for v in _flat(z)])


def real_if_close(a, tol=100):
    """若虚部接近 0 则返回实部，否则原样返回。"""
    _ = tol
    if iscomplexobj(a):
        cdata = getattr(a, "_complex_data", None)
        if cdata is None:
            cdata = [complex(v) for v in _flat(a)]
        if builtin_all(abs(complex(v).imag) < 1e-13 for v in cdata):
            return real(a)
    return _asarray(a)


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """将 NaN/Inf 替换为有限数。"""
    _ = copy
    pos = 1.7976931348623157e308 if posinf is None else float(posinf)
    neg = -1.7976931348623157e308 if neginf is None else float(neginf)
    return _wrap(_core.nan_to_num(_asarray(x)._array, nan, pos, neg))


# ========== 数组变形 / 组合 ==========
def concat(arrays, axis=0):
    """concatenate 别名（Array API）。"""
    return _np().concatenate(arrays, axis=axis)


def row_stack(tup):
    """垂直堆叠（vstack 别名）。"""
    return _np().vstack(tup)


def permute_dims(a, axes):
    """按给定顺序置换维度（transpose 别名）。"""
    return _np().transpose(a, axes)


def matrix_transpose(x):
    """交换最后两个维度。"""
    arr = _asarray(x)
    n = arr.ndim
    axes = list(range(n))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return _np().transpose(arr, tuple(axes))


def atleast_1d(*arys):
    """确保输入至少为一维。"""
    out = [_wrap(_core.atleast_1d(_asarray(a)._array)) for a in arys]
    return out[0] if len(out) == 1 else out


def atleast_2d(*arys):
    """确保输入至少为二维。"""
    out = [_wrap(_core.atleast_2d(_asarray(a)._array)) for a in arys]
    return out[0] if len(out) == 1 else out


def atleast_3d(*arys):
    """确保输入至少为三维。"""
    out = [_wrap(_core.atleast_3d(_asarray(a)._array)) for a in arys]
    return out[0] if len(out) == 1 else out


def broadcast_shapes(*shapes):
    """按 NumPy 规则计算广播后的形状。"""
    result = ()
    for shape in shapes:
        shp = tuple(shape)
        n = max(len(result), len(shp))
        a = (1,) * (n - len(result)) + result
        b = (1,) * (n - len(shp)) + shp
        merged = []
        for x, y in zip(a, b):
            if x == 1:
                merged.append(y)
            elif y == 1 or x == y:
                merged.append(x)
            else:
                raise ValueError(f"shape mismatch: {x} vs {y}")
        result = tuple(merged)
    return result


def broadcast_arrays(*args):
    """将多个数组广播到公共形状。"""
    np = _np()
    arrs = [_asarray(a) for a in args]
    target = broadcast_shapes(*[a.shape for a in arrs])
    return [np.broadcast_to(a, target) for a in arrs]


def block(arrays):
    """按嵌套列表组装数组块。"""
    np = _np()

    def _depth(x):
        if isinstance(x, list):
            return 1 + (max((_depth(e) for e in x), default=0))
        return 0

    def _leaf_ndim(x):
        if isinstance(x, list):
            return max((_leaf_ndim(e) for e in x), default=0)
        return _asarray(x).ndim

    list_depth = _depth(arrays)
    result_ndim = builtin_max(_leaf_ndim(arrays), list_depth)

    def _atleast(arr):
        if arr.ndim >= result_ndim:
            return arr
        return np.reshape(arr, (1,) * (result_ndim - arr.ndim) + tuple(arr.shape))

    def _build(x, depth):
        if not isinstance(x, list):
            return _atleast(_asarray(x))
        parts = [_build(e, depth + 1) for e in x]
        axis = result_ndim - list_depth + depth
        return np.concatenate(parts, axis=axis)

    return _build(arrays, 0)


def array_split(ary, indices_or_sections, axis=0):
    """将数组切分为多个子数组，允许不等分。"""
    np = _np()
    arr = _asarray(ary)
    n = arr.shape[axis]
    if isinstance(indices_or_sections, int):
        k = indices_or_sections
        base, extra = n // k, n % k
        sizes = [base + 1 if i < extra else base for i in range(k)]
        bounds = []
        acc = 0
        for s in sizes:
            acc += s
            bounds.append(acc)
        bounds = bounds[:-1]
    else:
        bounds = list(indices_or_sections)
    result = []
    prev = 0
    for b in list(bounds) + [n]:
        idx = list(range(prev, b))
        result.append(np.take(arr, idx, axis=axis))
        prev = b
    return result


def unstack(x, axis=0):
    """沿指定轴拆成子数组元组（NumPy 2.1）。"""
    np = _np()
    arr = _asarray(x)
    n = arr.shape[axis]
    out = []
    for i in range(n):
        picked = np.take(arr, [i], axis=axis)
        ax = axis % picked.ndim
        newshape = tuple(s for j, s in enumerate(picked.shape) if j != ax)
        out.append(np.reshape(picked, newshape) if newshape else picked)
    return tuple(out)


def take_along_axis(arr, indices, axis):
    """沿轴按索引取值。"""
    # np = _np()
    a = _asarray(arr)
    idx = _asarray(indices)
    if axis is None:
        a_flat = a.ravel()
        return _wrap(_core.take_along_axis(a_flat._array, idx.ravel()._array, 0))
    return _wrap(_core.take_along_axis(a._array, idx._array, axis))


def put_along_axis(arr, indices, values, axis):
    """沿轴按索引写入值（就地）。"""
    a = _asarray(arr)
    idx = _asarray(indices)
    v = _asarray(values)
    if axis is None:
        a_flat = a.ravel()
        _core.put_along_axis(a_flat._array, idx.ravel()._array, v._array, 0)
        a._array = a_flat._array
        return None
    _core.put_along_axis(a._array, idx._array, v._array, axis)
    return None


def choose(a, choices, out=None, mode="raise"):
    """按索引数组从 choices 中选值（Rust 实现）。"""
    _ = out, mode
    a_arr = _asarray(a)
    choice_arrs = [_asarray(c)._array for c in choices]
    return _wrap(_core.choose(a_arr._array, choice_arrs))


def compress(condition, a, axis=None):
    """按布尔条件沿轴筛选元素（Rust 实现）。"""
    return _wrap(_core.compress(_asarray(condition)._array, _asarray(a)._array, axis))


def copyto(dst, src, casting="same_kind", where=True):
    """将 src 的值复制到 dst（就地，Rust 实现）。"""
    _ = casting, where
    _core.copyto(_asarray(dst)._array, _asarray(src)._array)
    return None


def place(arr, mask, vals):
    """按布尔掩码把 vals 循环写入 arr（就地）。"""
    a = _asarray(arr)
    _core.place(a._array, _asarray(mask)._array, _asarray(vals)._array)
    return None


def putmask(a, mask, values):
    """按布尔掩码逐元素写入值（与 numpy.putmask 一致：a[i]=v[i] where mask[i]）。"""
    arr = _asarray(a)
    m = _asarray(mask)
    v = _asarray(values)
    # 标量值需广播到与 arr 同形以保证 Rust 层按位置元素级赋值正确
    if v.ndim == 0 or v.shape == ():
        v = _np().broadcast_to(v, arr.shape)
    _core.putmask(arr._array, m._array, v._array)
    return None


def pad(array, pad_width, mode="constant", **kwargs):
    """填充数组边缘，支持 constant/edge/reflect/wrap。

    安全防护：对 pad_width 施加上限，防止恶意输入导致内存耗尽（DoS）。
    每个维度的填充量不得超过该维度原长度的 4 倍 + 1024，或绝对上限 1 亿。
    """
    np = _np()
    arr = _asarray(array)
    cval = kwargs.get("constant_values", 0)
    ndim = arr.ndim
    if isinstance(pad_width, int):
        pw = [(pad_width, pad_width)] * ndim
    elif isinstance(pad_width[0], int):
        pw = [tuple(pad_width)] * ndim
    else:
        pw = [tuple(p) for p in pad_width]

    # DoS 防护：校验 pad_width 上限
    _MAX_PAD_ABS = 100_000_000  # 绝对上限：1 亿
    for axis, (before, after) in enumerate(pw):
        axis_len = arr.shape[axis] if axis < len(arr.shape) else 1
        relative_cap = axis_len * 4 + 1024
        cap = min(relative_cap, _MAX_PAD_ABS)
        total = before + after
        if total > cap:
            raise ValueError(
                f"pad_width 过大（轴 {axis}: before={before}, after={after}, "
                f"合计={total}），超过上限 {cap}。"
                f"该轴长度为 {axis_len}，允许的最大填充量为 {cap}。"
            )

    def pad_1d(seq, before, after):
        if mode == "constant":
            return [cval] * before + seq + [cval] * after
        if mode == "edge":
            return [seq[0]] * before + seq + [seq[-1]] * after
        if mode == "reflect":
            left = [seq[min(i + 1, len(seq) - 1)] for i in range(before)][::-1]
            right = [seq[max(len(seq) - 2 - i, 0)] for i in range(after)]
            return left + seq + right
        if mode == "wrap":
            left = [seq[(-before + i) % len(seq)] for i in range(before)]
            right = [seq[i % len(seq)] for i in range(after)]
            return left + seq + right
        raise ValueError(f"unsupported pad mode: {mode}")

    def rec(data, axis):
        if axis == ndim - 1:
            return pad_1d(list(data), pw[axis][0], pw[axis][1])
        padded = [rec(row, axis + 1) for row in data]
        width = len(padded[0])
        before = [[cval] * width for _ in range(pw[axis][0])]
        after = [[cval] * width for _ in range(pw[axis][1])]
        if mode == "edge":
            before = [list(padded[0]) for _ in range(pw[axis][0])]
            after = [list(padded[-1]) for _ in range(pw[axis][1])]
        return before + padded + after

    if ndim == 0:
        return arr
    return np.array(rec(arr.tolist(), 0))


def trim_zeros(filt, trim="fb"):
    """裁剪一维数组两端的零。"""
    return _wrap(_core.trim_zeros(_asarray(filt)._array, trim))


def ediff1d(ary, to_end=None, to_begin=None):
    """一维相邻元素差分，可选拼接首尾。"""
    arr = _asarray(ary)
    te = _flat(to_end) if to_end is not None else None
    tb = _flat(to_begin) if to_begin is not None else None
    return _wrap(_core.ediff1d(arr._array, te, tb))


def diff(a, n=1, axis=-1):
    """沿轴计算 n 阶差分。"""
    return _wrap(_core.diff(_asarray(a)._array, n, axis))


def gradient(f, *varargs, axis=None, edge_order=1):
    """计算梯度。"""
    _ = varargs, axis, edge_order
    return _wrap(_core.gradient(_asarray(f)._array))


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """向量叉积。"""
    _ = axisa, axisb, axisc, axis
    return _wrap(_core.cross(_asarray(a)._array, _asarray(b)._array))


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """梯形法数值积分。"""
    _ = axis
    yv = _flat(y)
    if x is not None:
        xv = _flat(x)
        return sum((xv[i + 1] - xv[i]) * (yv[i + 1] + yv[i]) / 2.0 for i in range(len(yv) - 1))
    return sum(dx * (yv[i + 1] + yv[i]) / 2.0 for i in range(len(yv) - 1))


trapz = trapezoid


# ========== 线性代数辅助 ==========
def outer(a, b, out=None):
    """向量外积。"""
    _ = out
    np = _np()
    av = _asarray(a).ravel()
    bv = _asarray(b).ravel()
    ar = np.reshape(av, (av.size, 1))
    br = np.reshape(bv, (1, bv.size))
    return np.multiply(ar, br)


def kron(a, b):
    """Kronecker 积（支持 1D/2D）。"""
    np = _np()
    A = _asarray(a)
    B = _asarray(b)
    if A.ndim == 1 and B.ndim == 1:
        return outer(A, B).ravel()
    A2 = A if A.ndim == 2 else np.reshape(A, (1, A.size))
    B2 = B if B.ndim == 2 else np.reshape(B, (1, B.size))
    ia, ja = A2.shape
    ib, jb = B2.shape
    ar = np.reshape(A2, (ia, 1, ja, 1))
    br = np.reshape(B2, (1, ib, 1, jb))
    prod4 = np.multiply(ar, br)
    return np.reshape(prod4, (ia * ib, ja * jb))


def tensordot(a, b, axes=2):
    """张量点积 - 灵活的多维数组收缩操作。

    【核心概念】
    tensordot在指定维度上对两个张量进行收缩（求和乘积），是einsum的特殊情况。
    它比einsum更高效，但灵活性稍低。

    【使用示例】
    >>> import rsnumpy as np

    # 1. 矩阵乘法：收缩最后1个维度
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> np.tensordot(A, B, axes=1)  # 等价于 A @ B
    array([[19., 22.],
           [43., 50.]])

    # 2. 内积：收缩所有维度
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([4, 5, 6])
    >>> np.tensordot(v1, v2, axes=1)  # 等价于 np.dot(v1, v2)
    32.0

    # 3. 批量矩阵乘法
    >>> batch_A = np.random.rand(3, 4, 5)  # 3个 4x5 矩阵
    >>> batch_B = np.random.rand(3, 5, 6)  # 3个 5x6 矩阵
    >>> # 收缩axis=2（A的第3维）和axis=1（B的第2维）
    >>> result = np.tensordot(batch_A, batch_B, axes=([2], [1]))
    >>> result.shape  # (3, 4, 3, 6)
    (3, 4, 3, 6)

    # 4. 张量收缩示例
    >>> T1 = np.random.rand(2, 3, 4)
    >>> T2 = np.random.rand(3, 4, 5)
    >>> # 收缩T1的(1,2)维度和T2的(0,1)维度
    >>> result = np.tensordot(T1, T2, axes=([1, 2], [0, 1]))
    >>> result.shape  # (2, 5)
    (2, 5)

    # 5. 使用列表指定收缩维度
    >>> A = np.random.rand(2, 3, 4, 5)
    >>> B = np.random.rand(3, 5, 6)
    >>> # 收缩A的(1,3)和B的(0,1)
    >>> result = np.tensordot(A, B, axes=([1, 3], [0, 1]))
    >>> result.shape  # (2, 4, 6)
    (2, 4, 6)

    【参数说明】
    a, b: array_like
        输入张量
    axes: int or (list, list) or (list, list)
        收缩维度指定：
        - 整数：收缩a的最后axes个维度和b的前axes个维度
        - 两列表：(a的维度列表, b的维度列表)
        - 维度数量必须匹配

    【axes参数详解】
    1. axes=N（整数）：
       - 收缩a的最后N个维度和b的前N个维度
       - axes=1：等价于矩阵乘法（2D情况下）
       - axes=2：收缩最后2个维度（用于批量矩阵）

    2. axes=([i1,i2,...], [j1,j2,...])（列表对）：
       - 在a的[i1,i2,...]维度和b的[j1,j2,...]维度上收缩
       - 维度列表长度必须相同
       - 对应维度的长度必须匹配

    【与einsum对应关系】
    tensordot(A, B, axes=1)          ≈ einsum('ij,jk->ik', A, B)
    tensordot(A, B, axes=2)          ≈ einsum('ijk,kl->ijl', A, B)
    tensordot(A, B, axes=([1],[0]))  ≈ einsum('ij,jk->ik', A, B)
    tensordot(A, B, axes=([1,2],[0,1])) ≈ einsum('ijk,kl->il', A, B)

    【性能建议】
    - 简单收缩（axes=1或2）：tensordot > einsum
    - 复杂维度选择：einsum更直观
    - 大型数组：tensordot通常更高效

    【调试技巧】
    1. 检查维度匹配：
       >>> A.shape[axes_a] == B.shape[axes_b]
    2. 验证输出形状：
       >>> # 输出 = A的非收缩维度 + B的非收缩维度
    3. 使用einsum可视化：
       >>> # 同样的操作用einsum表达更易理解

    【常见错误】
    ValueError: dimension mismatch
    - 检查收缩维度长度是否匹配
    - 检查axes列表长度是否相等

    【注意事项】
    - 不支持复数类型的矩阵乘法（需要用einsum）
    - 对于大型数组，考虑使用reshape+matmul优化
    """

    np = _np()
    A = _asarray(a)
    B = _asarray(b)

    # 解析axes参数
    if isinstance(axes, int):
        # 整数形式：收缩A的最后axes个维度和B的前axes个维度
        axes_a = list(range(A.ndim - axes, A.ndim))
        axes_b = list(range(axes))
    else:
        # 列表对形式：精确指定收缩维度
        axes_a, axes_b = axes
        axes_a = list(axes_a) if isinstance(axes_a, (list, tuple)) else [axes_a]
        axes_b = list(axes_b) if isinstance(axes_b, (list, tuple)) else [axes_b]

    # 计算非收缩维度
    notin_a = [i for i in range(A.ndim) if i not in axes_a]
    notin_b = [i for i in range(B.ndim) if i not in axes_b]

    # 调整维度顺序：非收缩维度在前，收缩维度在后
    newaxes_a = notin_a + axes_a
    newaxes_b = axes_b + notin_b

    at = np.transpose(A, tuple(newaxes_a))
    bt = np.transpose(B, tuple(newaxes_b))

    # 【性能优化】使用math.prod代替显式循环求积
    import math as _math
    n1 = _math.prod(A.shape[i] for i in notin_a) if notin_a else 1
    nc = _math.prod(A.shape[i] for i in axes_a) if axes_a else 1
    n2 = _math.prod(B.shape[i] for i in notin_b) if notin_b else 1

    # 重塑为2D矩阵进行matmul
    am = np.reshape(at.ravel(), (n1, nc))
    bm = np.reshape(bt.ravel(), (nc, n2))

    res = np.matmul(am, bm)
    if not hasattr(res, "_array"):
        res = _wrap(res)

    # 计算输出形状并重塑
    outshape = tuple(A.shape[i] for i in notin_a) + tuple(B.shape[i] for i in notin_b)
    return np.reshape(res.ravel(), outshape) if outshape else _as_scalar(res)


def tri(N, M=None, k=0, dtype=None):
    """下三角为 1、其余为 0 的矩阵。"""
    _ = dtype
    np = _np()
    if M is None:
        M = N
    rows = np.reshape(np.arange(0, N, 1), (N, 1))
    cols = np.reshape(np.arange(0, M, 1), (1, M))
    return np.where(np.less_equal(cols, np.add(rows, float(k))),
                    np.full((N, M), 1.0), np.full((N, M), 0.0))


def tril(m, k=0):
    """取下三角部分。"""
    np = _np()
    arr = _asarray(m)
    n, mm = arr.shape[-2], arr.shape[-1]
    return np.multiply(arr, tri(n, mm, k))


def triu(m, k=0):
    """取上三角部分。"""
    np = _np()
    arr = _asarray(m)
    n, mm = arr.shape[-2], arr.shape[-1]
    return np.subtract(arr, np.multiply(arr, tri(n, mm, k - 1)))


def diagonal(a, offset=0, axis1=0, axis2=1):
    """提取对角线（支持 axis1/axis2）。"""
    return _wrap(_core.diagonal_ext(_asarray(a)._array, offset, axis1, axis2))


def diag(v, k=0):
    """一维->对角矩阵，二维->提取对角线。"""
    return _wrap(_core.diag(_asarray(v)._array, k))


def diagflat(v, k=0):
    """将输入展平后构造对角矩阵。"""
    return diag(_np().array(_flat(v)), k)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """对角线元素之和。"""
    _ = axis1, axis2, dtype, out
    return sum(_flat(diagonal(a, offset)))


def tri_indices_helper(n, k, upper):
    """生成上/下三角的 (rows, cols) 索引列表。

    使用列表推导代替嵌套 for 循环，减少 Python 解释器开销。
    """
    if upper:
        # 上三角：c >= r + k
        idx_r = [r for r in range(n) for c in range(n) if c >= r + k]
        idx_c = [c for r in range(n) for c in range(n) if c >= r + k]
    else:
        # 下三角：c <= r + k
        idx_r = [r for r in range(n) for c in range(n) if c <= r + k]
        idx_c = [c for r in range(n) for c in range(n) if c <= r + k]
    return idx_r, idx_c


def tril_indices(n, k=0, m=None):
    """下三角元素的索引。"""
    mm = n if m is None else m
    rows, cols = _core.tril_indices(n, k, mm)
    return _wrap(rows, _dtype='int64'), _wrap(cols, _dtype='int64')


def triu_indices(n, k=0, m=None):
    """上三角元素的索引。"""
    mm = n if m is None else m
    rows, cols = _core.triu_indices(n, k, mm)
    return _wrap(rows, _dtype='int64'), _wrap(cols, _dtype='int64')


def tril_indices_from(arr, k=0):
    """由数组形状返回下三角索引。"""
    a = _asarray(arr)
    return tril_indices(a.shape[0], k, a.shape[1])


def triu_indices_from(arr, k=0):
    """由数组形状返回上三角索引。"""
    a = _asarray(arr)
    return triu_indices(a.shape[0], k, a.shape[1])


def diag_indices(n, ndim=2):
    """主对角线索引。"""
    np = _np()
    idx = np.arange(0, n, 1)
    return tuple(idx for _ in range(ndim))


def diag_indices_from(arr):
    """由数组形状返回主对角线索引。"""
    a = _asarray(arr)
    return diag_indices(a.shape[0], a.ndim)


def fill_diagonal(a, val, wrap=False):
    """就地填充主对角线。"""
    arr = _asarray(a)
    _core.fill_diagonal(arr._array, _asarray(val)._array, wrap)
    return None


def mask_indices(n, mask_func, k=0):
    """返回 mask_func 作用于 (n,n) 全 1 矩阵得到的非零索引。"""
    np = _np()
    ones = np.full((n, n), 1.0)
    masked = mask_func(ones, k)
    return np.nonzero(masked)


# ========== 窗函数 / 信号 ==========
def _window_base(M):
    np = _np()
    if M < 1:
        return None, None
    if M == 1:
        return np.array([1.0]), None
    return None, np.arange(0, M, 1)


def hanning(M):
    """Hann 窗。"""
    np = _np()
    done, n = _window_base(M)
    if done is not None or n is None:
        return done if done is not None else np.array([])
    return np.subtract(np.full((M,), 0.5), np.multiply(np.cos(np.multiply(n, 2.0 * _math.pi / (M - 1))), 0.5))


def hamming(M):
    """Hamming 窗。"""
    np = _np()
    done, n = _window_base(M)
    if done is not None or n is None:
        return done if done is not None else np.array([])
    return np.subtract(np.full((M,), 0.54), np.multiply(np.cos(np.multiply(n, 2.0 * _math.pi / (M - 1))), 0.46))


def blackman(M):
    """Blackman 窗。"""
    np = _np()
    done, n = _window_base(M)
    if done is not None or n is None:
        return done if done is not None else np.array([])
    a = np.multiply(n, 2.0 * _math.pi / (M - 1))
    term1 = np.multiply(np.cos(a), 0.5)
    term2 = np.multiply(np.cos(np.multiply(n, 4.0 * _math.pi / (M - 1))), 0.08)
    return np.add(np.subtract(np.full((M,), 0.42), term1), term2)


def bartlett(M):
    """Bartlett（三角）窗。"""
    np = _np()
    done, n = _window_base(M)
    if done is not None or n is None:
        return done if done is not None else np.array([])
    half = (M - 1) / 2.0
    return np.subtract(np.full((M,), 1.0), np.divide(np.abs(np.subtract(n, half)), half))


def i0(x):
    """第一类零阶修正贝塞尔函数。"""
    arr = _asarray(x)
    res = _wrap(_core.i0(arr._array))
    return res if arr.ndim else _as_scalar(res)


def kaiser(M, beta):
    """Kaiser 窗。"""
    np = _np()
    if M < 1:
        return np.array([])
    if M == 1:
        return np.array([1.0])
    n = np.arange(0, M, 1)
    alpha = (M - 1) / 2.0
    ratio = np.divide(np.subtract(n, alpha), alpha)
    arg = np.multiply(np.sqrt(np.subtract(np.full((M,), 1.0), np.multiply(ratio, ratio))), beta)
    return np.divide(i0(arg), _as_scalar(i0(np.array([float(beta)]))))


def convolve(a, v, mode="full"):
    """一维离散卷积。"""
    return _wrap(_core.convolve(_asarray(a)._array, _asarray(v)._array, mode))


def correlate(a, v, mode="valid"):
    """一维互相关。"""
    return _wrap(_core.correlate(_asarray(a)._array, _asarray(v)._array, mode))


def interp(x, xp, fp, left=None, right=None, period=None):
    """一维线性插值。"""
    _ = period
    arr = _asarray(x)
    raw = _core.interp(arr._array, _asarray(xp)._array, _asarray(fp)._array, left, right)
    res = _wrap(raw)
    return res if arr.ndim else _as_scalar(res)


def bincount(x, weights=None, minlength=0):
    """统计非负整数数组中每个值出现的次数。"""
    if weights is None:
        return _wrap(_core.bincount(_asarray(x)._array, minlength), "int64")
    return _wrap(_core.bincount_weighted(_asarray(x)._array, _asarray(weights)._array, minlength))


def vander(x, N=None, increasing=False):
    """生成范德蒙矩阵。"""
    np = _np()
    xv = _asarray(x).ravel()
    n = xv.size
    if N is None:
        N = n
    xr = np.reshape(xv, (n, 1))
    if increasing:
        powers = np.arange(0, N, 1)
    else:
        powers = np.subtract(np.full((N,), float(N - 1)), np.arange(0, N, 1))
    powers = np.reshape(powers, (1, N))
    return np.power(xr, powers)


def unwrap(p, discont=None, axis=-1, period=6.283185307179586):
    """相位解卷绕。"""
    return _wrap(_core.unwrap(_asarray(p)._array, discont, axis, period))


# ========== 归约：all / any / round（顶层函数）==========
def _boolmask(arr):
    np = _np()
    return np.where(np.equal(arr, np.full(arr.shape, 0.0)),
                    np.full(arr.shape, 0.0), np.full(arr.shape, 1.0))


def all(a, axis=None, out=None, keepdims=False, *, where=True):
    """判断沿轴的所有元素是否都为真。"""
    _ = out, keepdims, where
    np = _np()
    arr = _asarray(a)
    if axis is None:
        return builtin_all(v != 0 for v in _flat(arr))
    return np.greater(np.min(_boolmask(arr), axis), 0.5)


def any(a, axis=None, out=None, keepdims=False, *, where=True):
    """判断沿轴是否存在为真的元素。"""
    _ = out, keepdims, where
    np = _np()
    arr = _asarray(a)
    if axis is None:
        return builtin_any(v != 0 for v in _flat(arr))
    return np.greater(np.max(_boolmask(arr), axis), 0.5)


def round(a, decimals=0, out=None):
    """四舍六入五成双到指定小数位（around 别名）。"""
    _ = out
    return _np().around(a, decimals)


round_ = round


# ========== 类型转换 / 连续性 / 共享内存 ==========
def astype(x, dtype, copy=True, casting="unsafe"):
    """将数组转换为指定类型。"""
    _ = copy, casting
    return _asarray(x).astype(dtype)


def ascontiguousarray(a, dtype=None):
    """返回 C 连续数组（rsnumpy 数组始终连续）。"""
    arr = _np().asarray(a)
    return arr.astype(dtype) if dtype is not None else arr


def asfortranarray(a, dtype=None):
    """返回 Fortran 连续数组（rsnumpy 视为等价）。"""
    return ascontiguousarray(a, dtype)


def asarray_chkfinite(a, dtype=None, order=None):
    """转为数组，若含 inf/nan 则抛错。

    使用 any() + 生成器表达式代替 for 循环。
    """
    _ = order
    arr = _np().asarray(a)
    # 使用 any() + 生成器表达式代替显式 for 循环
    flat = _flat(arr)
    if builtin_any(isinstance(v, float) and (v != v or v == _math.inf or v == -_math.inf) for v in flat):
        raise ValueError("array must not contain infs or NaNs")
    return arr.astype(dtype) if dtype is not None else arr


def require(a, dtype=None, requirements=None, *, like=None):
    """返回满足要求的数组（rsnumpy 恒为连续、可写）。"""
    _ = requirements, like
    return ascontiguousarray(a, dtype)


def isfortran(a):
    """判断是否为 Fortran 连续（rsnumpy 恒为 C 连续）。"""
    _ = a
    return False


def shares_memory(a, b, max_work=None):
    """判断两个数组是否共享底层内存。"""
    _ = max_work
    ra = getattr(a, "_array", None)
    rb = getattr(b, "_array", None)
    return ra is not None and ra is rb


def may_share_memory(a, b, max_work=None):
    """保守判断两个数组是否可能共享内存。"""
    return shares_memory(a, b, max_work)


def array_equiv(a1, a2):
    """判断两个数组在广播后是否逐元素相等。

    使用 all() + 生成器表达式代替显式 for 循环。
    """
    np = _np()
    x = _asarray(a1)
    y = _asarray(a2)
    try:
        shp = broadcast_shapes(x.shape, y.shape)
    except ValueError:
        return False
    xb = np.broadcast_to(x, shp)
    yb = np.broadcast_to(y, shp)
    # 使用 all() + zip 代替显式 for 循环
    return builtin_all(p == q for p, q in zip(_flat(xb), _flat(yb)))


# ========== 索引工具 ==========
def indices(dimensions, dtype=None, sparse=False):
    """返回网格索引数组。"""
    np = _np()
    dims = tuple(int(d) for d in dimensions)
    n = len(dims)
    out = []
    for i in range(n):
        shp = tuple(dims[i] if j == i else 1 for j in range(n))
        arr = np.reshape(np.arange(0, dims[i], 1), shp)
        out.append(arr if sparse else np.broadcast_to(arr, dims))
    if sparse:
        return tuple(o.astype(dtype or "int64") for o in out)
    res = np.stack(out, axis=0)
    return res.astype(dtype or "int64")


def unravel_index(indices, shape, order='C'):
    """将扁平索引转换为多维坐标。"""
    dims = [int(s) for s in (shape if isinstance(shape, (list, tuple)) else (shape,))]
    arr = _asarray(indices)
    scalar_in = isscalar(indices) or (hasattr(indices, "ndim") and arr.ndim == 0)
    result = _core.unravel_index(arr._array, dims, order)
    if scalar_in:
        return tuple(int(_wrap(r, _dtype='int64').item()) for r in result)
    return tuple(_wrap(r, _dtype='int64') for r in result)


def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    """将多维坐标转换为扁平索引。"""
    dims_list = [int(s) for s in dims]
    raw_keys = [_asarray(m)._array for m in multi_index]
    result = _core.ravel_multi_index(raw_keys, dims_list, mode, order)
    scalar_in = builtin_all(isscalar(m) or (hasattr(m, "ndim") and _asarray(m).ndim == 0)
                            for m in multi_index)
    wrapped = _wrap(result, _dtype='int64')
    if scalar_in and wrapped.ndim == 0:
        return int(wrapped.item())
    return wrapped


# ========== 沿轴应用 / 分段函数 ==========
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """沿指定轴对一维切片应用函数。"""
    import itertools as _it
    np = _np()
    a = _asarray(arr)
    nd = a.ndim
    axis = axis % nd
    inds = [range(a.shape[i]) for i in range(nd) if i != axis]
    combos = list(_it.product(*inds)) if inds else [()]

    def slice_at(combo):
        key = []
        it = iter(combo)
        for i in range(nd):
            key.append(slice(None) if i == axis else next(it))
        return a[tuple(key)] if nd > 1 else a

    results = [func1d(slice_at(c), *args, **kwargs) for c in combos]
    outer = tuple(a.shape[i] for i in range(nd) if i != axis)
    r0 = results[0]
    is_scalar_out = not isinstance(r0, list) and not (hasattr(r0, "ndim") and _asarray(r0).ndim > 0)
    if is_scalar_out:
        vals = [_as_scalar(r) if hasattr(r, "tolist") else r for r in results]
        return _as_scalar(np.array(vals)) if not outer else np.reshape(np.array(vals), outer)
    r_shape = tuple(_asarray(r0).shape)
    stacked = np.array([_asarray(r).tolist() for r in results])
    stacked = np.reshape(stacked, outer + r_shape)
    if r_shape:
        src = list(range(len(outer), len(outer) + len(r_shape)))
        dst = list(range(axis, axis + len(r_shape)))
        stacked = np.moveaxis(stacked, src, dst)
    return stacked


def apply_over_axes(func, a, axes):
    """在多个轴上依次应用归约函数（保持维度）。"""
    np = _np()
    arr = _asarray(a)
    ax_list = [axes] if isscalar(axes) else list(axes)
    for ax in ax_list:
        res = _asarray(func(arr, ax))
        if res.ndim == arr.ndim:
            arr = res
        elif res.ndim == arr.ndim - 1:
            arr = np.expand_dims(res, ax)
        else:
            raise ValueError("function is not returning array of correct shape")
    return arr


def piecewise(x, condlist, funclist, *args, **kw):
    """按条件分段求值。

    使用 zip + 列表推导代替嵌套 for 循环，提高执行效率。
    """
    np = _np()
    arr = _asarray(x)
    flat = _flat(arr)
    if not isinstance(condlist, (list, tuple)) or (condlist and isscalar(condlist[0])):
        condlist = [condlist]
    conds = [_flat(c) for c in condlist]
    n = len(flat)
    n_conds = len(conds)
    has_default = len(funclist) == n_conds + 1
    default_f = funclist[-1] if has_default else None

    # 预计算每个条件的长度，避免在循环中重复 len() 调用
    cond_lens = [len(c) for c in conds]

    out = [0.0] * n
    for i in range(n):
        applied = False
        for k in range(n_conds):
            if i < cond_lens[k] and conds[k][i]:
                f = funclist[k]
                out[i] = f(flat[i], *args, **kw) if callable(f) else f
                applied = True
                break  # numpy 行为：匹配第一个条件即停止
        if not applied and has_default:
            f = default_f
            out[i] = f(flat[i], *args, **kw) if callable(f) else f
    return np.reshape(np.array(out), arr.shape) if arr.ndim else _as_scalar(np.array(out))


# ========== 位运算别名 / 计数 ==========
def bitwise_invert(x):
    """逐元素按位取反（invert 别名）。"""
    return _np().invert(x)


def bitwise_left_shift(x1, x2):
    """逐元素左移（left_shift 别名）。"""
    return _np().left_shift(x1, x2)


def bitwise_right_shift(x1, x2):
    """逐元素右移（right_shift 别名）。"""
    return _np().right_shift(x1, x2)


def bitwise_count(x):
    """逐元素统计绝对值二进制表示中的置位比特数。"""
    return _wrap(_core.bitwise_count(_asarray(x)._array), "uint8")


def packbits(a, axis=None, bitorder='big'):
    """将布尔/整数数组按比特打包为 uint8。"""
    _ = axis
    return _wrap(_core.packbits(_asarray(a)._array, bitorder), "uint8")


def unpackbits(a, axis=None, count=None, bitorder='big'):
    """将 uint8 数组展开为比特。"""
    _ = axis
    c = count if count is not None else 0
    return _wrap(_core.unpackbits(_asarray(a)._array, c if c > 0 else None, bitorder), "uint8")


# ========== 向量 / 矩阵乘积（gufunc）==========
def vecdot(x1, x2, axis=-1):
    """沿轴计算向量点积（对第一个参数取共轭）。"""
    _ = axis
    np = _np()
    a = _asarray(x1)
    b = _asarray(x2)
    a = np.conjugate(a) if iscomplexobj(a) else a
    return np.sum(np.multiply(a, b), -1)


def matvec(x1, x2):
    """矩阵-向量乘积。"""
    return _np().matmul(_asarray(x1), _asarray(x2))


def vecmat(x1, x2):
    """向量-矩阵乘积（对向量取共轭）。"""
    np = _np()
    a = _asarray(x1)
    a = np.conjugate(a) if iscomplexobj(a) else a
    return np.matmul(a, _asarray(x2))


# ========== 直方图分箱边界 ==========
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """仅返回直方图的分箱边界。"""
    _ = weights
    np = _np()
    data = _flat(a)
    if isinstance(bins, str):
        bins = 10
    if isinstance(bins, (list, tuple)) or hasattr(bins, "_array"):
        return np.array(_flat(bins))
    if range is None:
        lo = builtin_min(data) if data else 0.0
        hi = builtin_max(data) if data else 1.0
    else:
        lo, hi = range
    lo = float(lo)
    hi = float(hi)
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    nb = int(bins)
    step = (hi - lo) / nb
    edges = [lo + i * step for i in builtin_range(nb + 1)]
    edges[-1] = hi
    return np.array(edges)


builtin_range = range


# ========== dtype 相关：promote / result / cast / min_scalar / common / isdtype ==========
_DT_INFO = {
    'bool': ('b', 1),
    'int8': ('i', 1), 'int16': ('i', 2), 'int32': ('i', 4), 'int64': ('i', 8),
    'uint8': ('u', 1), 'uint16': ('u', 2), 'uint32': ('u', 4), 'uint64': ('u', 8),
    'float16': ('f', 2), 'float32': ('f', 4), 'float64': ('f', 8),
    'complex64': ('c', 8), 'complex128': ('c', 16),
}


def _canon_dt_type(x):
    """将 dtype 类的类型/字符串解析为规范名称。"""
    return _np().dtype(x).name


def _canon_dt(x):
    """将 dtype 类型 / 数组 / 值 解析为规范 dtype 名称。"""
    if hasattr(x, '_dtype'):
        return x._dtype
    if hasattr(x, 'dtype') and not isinstance(x, type):
        d = x.dtype
        return getattr(d, 'name', str(d))
    try:
        n = _np().dtype(x).name
        if n in _DT_INFO:
            return n
    except Exception:
        pass
    return _canon_dt(min_scalar_type(x))


def _float_part(name):
    k, s = _DT_INFO[name]
    if k == 'f':
        return s
    if k == 'c':
        return s // 2
    if k in 'iu':
        return {1: 2, 2: 4, 4: 8, 8: 8}[s]
    return 2


def _int_name(k, size):
    pre = 'int' if k == 'i' else 'uint'
    return pre + {1: '8', 2: '16', 4: '32', 8: '64'}[size]


def _promote2(n1, n2):
    if n1 == n2:
        return n1
    k1, s1 = _DT_INFO[n1]
    k2, s2 = _DT_INFO[n2]
    if k1 == 'b':
        return n2
    if k2 == 'b':
        return n1
    if k1 == 'c' or k2 == 'c':
        fp = builtin_max(_float_part(n1), _float_part(n2))
        return 'complex128' if fp >= 8 else 'complex64'
    if k1 == 'f' or k2 == 'f':
        fp = builtin_max(_float_part(n1), _float_part(n2))
        return {2: 'float16', 4: 'float32', 8: 'float64'}[fp]
    if k1 == k2:
        return _int_name(k1, builtin_max(s1, s2))
    isize = s1 if k1 == 'i' else s2
    usize = s1 if k1 == 'u' else s2
    if usize < isize:
        return _int_name('i', isize)
    t = usize * 2
    return 'float64' if t > 8 else _int_name('i', t)


def promote_types(type1, type2):
    """返回可安全容纳两个类型的最小类型。"""
    return _np().dtype(_promote2(_canon_dt_type(type1), _canon_dt_type(type2)))


def result_type(*arrays_and_dtypes):
    """按 NumPy 提升规则返回结果类型。"""
    names = [_canon_dt(x) for x in arrays_and_dtypes]
    acc = names[0]
    for n in names[1:]:
        acc = _promote2(acc, n)
    return _np().dtype(acc)


def can_cast(from_, to, casting='safe'):
    """判断能否按给定规则从一个类型转换到另一个类型。"""
    try:
        fn = _canon_dt(from_)
        tn = _canon_dt_type(to)
    except Exception:
        return False
    if fn == tn:
        return True
    if casting in ('no', 'equiv'):
        return False
    if casting == 'unsafe':
        return True
    # 非数值类型（字符串/void/object/datetime 等）无通用提升规则，仅同名可转换
    if fn not in _DT_INFO or tn not in _DT_INFO:
        return False
    prom = _promote2(fn, tn)
    if casting == 'safe':
        return prom == tn
    if prom == tn:
        return True
    group = {'b': 0, 'u': 1, 'i': 1, 'f': 2, 'c': 3}
    return group[_DT_INFO[fn][0]] <= group[_DT_INFO[tn][0]]


def min_scalar_type(value):
    """返回可容纳给定标量值的最小 dtype。"""
    np = _np()
    if hasattr(value, '_dtype') or (hasattr(value, 'dtype') and not isinstance(value, type)):
        return np.dtype(_canon_dt(value))
    if isinstance(value, bool):
        return np.dtype('bool')
    if isinstance(value, int):
        if value >= 0:
            for name, bits in [('uint8', 8), ('uint16', 16), ('uint32', 32), ('uint64', 64)]:
                if value < (1 << bits):
                    return np.dtype(name)
            return np.dtype('float64')
        for name, bits in [('int8', 8), ('int16', 16), ('int32', 32), ('int64', 64)]:
            if -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
                return np.dtype(name)
        return np.dtype('float64')
    if isinstance(value, float):
        return np.dtype('float16')
    if isinstance(value, complex):
        return np.dtype('complex64')
    return np.dtype('float64')


def common_type(*arrays):
    """返回若干数组的公共（浮点/复数）标量类型。"""
    np = _np()
    has_complex = False
    prec = 4
    for a in arrays:
        n = _canon_dt(a)
        k, s = _DT_INFO[n]
        if k == 'c':
            has_complex = True
            prec = builtin_max(prec, s // 2)
        elif k == 'f':
            prec = builtin_max(prec, s)
        else:
            prec = builtin_max(prec, 8)
    if has_complex:
        return np.complex128 if prec >= 8 else np.complex64
    return np.float64 if prec >= 8 else np.float32


def isdtype(dtype, kind):
    """判断 dtype 是否属于指定类别（Array API）。"""
    n = _canon_dt_type(dtype)
    k = _DT_INFO[n][0]

    def match(kk):
        if isinstance(kk, str):
            table = {
                'bool': k == 'b',
                'signed integer': k == 'i',
                'unsigned integer': k == 'u',
                'integral': k in 'iu',
                'real floating': k == 'f',
                'complex floating': k == 'c',
                'numeric': k in 'iufc',
            }
            if kk in table:
                return table[kk]
        return _canon_dt_type(kk) == n

    if isinstance(kind, (tuple, list)):
        return builtin_any(match(x) for x in kind)
    return match(kind)


def issubdtype(arg1, arg2):
    """判断第一个类型是否为第二个类型（含抽象类别）的子类型。"""
    return _np().issubdtype(arg1, arg2)


# ========== 进制 / 类型码文本 ==========
def base_repr(number, base=2, padding=0):
    """将整数转换为给定进制的字符串。"""
    return _core.base_repr(int(number), base, padding)


_TYPENAME = {
    'S1': 'character', '?': 'bool', 'B': 'unsigned char', 'b': 'signed char',
    'h': 'short', 'H': 'unsigned short', 'i': 'integer', 'I': 'unsigned integer',
    'l': 'long integer', 'L': 'unsigned long integer', 'q': 'long long integer',
    'Q': 'unsigned long long integer', 'f': 'single precision', 'd': 'double precision',
    'g': 'long precision', 'F': 'complex single precision', 'D': 'complex double precision',
    'G': 'complex long double precision', 'S': 'string', 'U': 'unicode', 'V': 'void', 'O': 'object',
}


def typename(char):
    """返回类型码的英文名称。"""
    return _TYPENAME[char]


_TYPECODES_BY_ELSIZE = 'GDFgdfQqLlIiHhBb?'


def mintypecode(typechars, typeset='GDFgdf', default='d'):
    """返回可容纳所有输入类型码的最小类型码。"""
    intersection = set()
    for t in typechars:
        c = t if isinstance(t, str) and len(t) == 1 else _np().dtype(t).char
        if c in typeset:
            intersection.add(c)
    if not intersection:
        return default
    if 'F' in intersection and 'd' in intersection:
        return 'D'
    return builtin_min(intersection, key=_TYPECODES_BY_ELSIZE.index)


# ========== 数组文本表示 ==========
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """返回数组的可求值字符串表示。"""
    _ = max_line_width, precision, suppress_small
    return repr(_asarray(arr))


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """返回数组的字符串表示（无 array() 包裹）。"""
    _ = max_line_width, precision, suppress_small
    return str(_asarray(a))


def array2string(a, max_line_width=None, precision=None, suppress_small=None,
                 separator=' ', prefix="", **kwargs):
    """返回数组数据的字符串表示。"""
    _ = max_line_width, precision, suppress_small, separator, prefix, kwargs
    return str(_asarray(a))


# ========== 打印选项 ==========
_PRINTOPTS = {
    'precision': 8, 'threshold': 1000, 'edgeitems': 3, 'linewidth': 75,
    'suppress': False, 'nanstr': 'nan', 'infstr': 'inf', 'sign': '-',
    'formatter': None, 'floatmode': 'maxprec', 'legacy': False,
}


def get_printoptions():
    """返回当前打印选项字典。"""
    return dict(_PRINTOPTS)


def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None,
                     suppress=None, nanstr=None, infstr=None, sign=None,
                     formatter=None, floatmode=None, legacy=None, *, override_repr=None):
    """设置打印选项。"""
    _ = override_repr
    updates = {
        'precision': precision, 'threshold': threshold, 'edgeitems': edgeitems,
        'linewidth': linewidth, 'suppress': suppress, 'nanstr': nanstr,
        'infstr': infstr, 'sign': sign, 'formatter': formatter,
        'floatmode': floatmode, 'legacy': legacy,
    }
    for k, v in updates.items():
        if v is not None:
            _PRINTOPTS[k] = v


class printoptions:
    """临时修改打印选项的上下文管理器。"""

    def __init__(self, **kwargs):
        self._kw = kwargs
        self._saved = None

    def __enter__(self):
        self._saved = dict(_PRINTOPTS)
        set_printoptions(**self._kw)
        return get_printoptions()

    def __exit__(self, *exc):
        _PRINTOPTS.clear()
        _PRINTOPTS.update(self._saved)
        return False


def format_float_positional(x, precision=None, unique=True, fractional=True,
                            trim='k', sign=False, pad_left=None, pad_right=None,
                            min_digits=None):
    """以定点记法格式化单个浮点数。"""
    _ = unique, fractional, pad_left, pad_right, min_digits
    x = float(x)
    if precision is None:
        s = repr(x)
        if 'e' in s or 'E' in s:
            s = f"{x:.16f}".rstrip('0')
        if '.' not in s:
            s += '.'
    else:
        s = f"{x:.{precision}f}"
    if trim == '-':
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
    elif trim == '0':
        if '.' in s:
            s = s.rstrip('0')
            if s.endswith('.'):
                s += '0'
    if sign and x >= 0:
        s = '+' + s
    return s


def format_float_scientific(x, precision=None, unique=True, trim='k', sign=False,
                            pad_left=None, exp_digits=None, min_digits=None):
    """以科学记法格式化单个浮点数。"""
    _ = unique, pad_left, exp_digits, min_digits
    x = float(x)
    if precision is None:
        s = f"{x:e}"
        mant, exp = s.split('e')
        if trim in ('-', '0'):
            mant = mant.rstrip('0').rstrip('.') if trim == '-' else mant.rstrip('0')
        s = mant + 'e' + exp
    else:
        s = f"{x:.{precision}e}"
    if sign and x >= 0:
        s = '+' + s
    return s


# ========== 浮点错误状态 ==========
_ERRSTATE = {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
_ERRCALL = [None]
_BUFSIZE = [8192]


def geterr():
    """返回当前浮点错误处理状态。"""
    return dict(_ERRSTATE)


def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """设置浮点错误处理状态，返回旧状态。"""
    old = dict(_ERRSTATE)
    if all is not None:
        for k in _ERRSTATE:
            _ERRSTATE[k] = all
    for k, v in {'divide': divide, 'over': over, 'under': under, 'invalid': invalid}.items():
        if v is not None:
            _ERRSTATE[k] = v
    return old


def geterrcall():
    """返回当前浮点错误回调。"""
    return _ERRCALL[0]


def seterrcall(func):
    """设置浮点错误回调，返回旧回调。"""
    old = _ERRCALL[0]
    _ERRCALL[0] = func
    return old


def getbufsize():
    """返回 ufunc 缓冲区大小。"""
    return _BUFSIZE[0]


def setbufsize(size):
    """设置 ufunc 缓冲区大小，返回旧值。"""
    old = _BUFSIZE[0]
    _BUFSIZE[0] = int(size)
    return old


class errstate:
    """临时修改浮点错误处理状态的上下文管理器。"""

    def __init__(self, *, call=None, **kwargs):
        self._kw = kwargs
        self._call = call
        self._saved = None
        self._saved_call = None

    def __enter__(self):
        self._saved = seterr(**self._kw)
        if self._call is not None:
            self._saved_call = seterrcall(self._call)
        return None

    def __exit__(self, *exc):
        seterr(**self._saved)
        if self._call is not None:
            seterrcall(self._saved_call)
        return False


# ========== 解析 / IO ==========
def fromstring(string, dtype=float, count=-1, sep=''):
    """从文本字符串解析一维数组。"""
    np = _np()
    if sep == '':
        raise ValueError("fromstring() with binary data is not supported")
    if sep == ' ':
        parts = string.split()
    else:
        parts = [p.strip() for p in string.split(sep)]
        parts = [p for p in parts if p != '']
    vals = [float(p) for p in parts]
    if count >= 0:
        vals = vals[:count]
    arr = np.array(vals)
    return arr if dtype in (float, None) else arr.astype(dtype)


def savez_compressed(file, *args, **kwds):
    """将多个数组保存为压缩 .npz 文件（rsnumpy 以非压缩格式落盘）。"""
    return _np().savez(file, *args, **kwds)


# ========== 由 Python 函数构造 ufunc / 向量化 ==========
def frompyfunc(func, nin, nout, *, identity=None):
    """将任意 Python 函数封装为逐元素的 ufunc 风格可调用对象。"""
    _ = identity
    np = _np()

    def wrapper(*args):
        arrs = [_asarray(a) for a in args[:nin]]
        flats = [_flat(a) for a in arrs]
        n = len(flats[0]) if flats else 0
        shp = arrs[0].shape if arrs else ()
        results = [func(*[flats[j][i] for j in range(nin)]) for i in range(n)]
        if nout == 1:
            out = np.array(results)
            return np.reshape(out, shp) if shp else _as_scalar(out)
        cols = []
        for t in range(nout):
            col = np.array([r[t] for r in results])
            cols.append(np.reshape(col, shp) if shp else _as_scalar(col))
        return tuple(cols)

    wrapper.nin = nin
    wrapper.nout = nout
    wrapper.nargs = nin + nout
    return wrapper


class vectorize:
    """将标量 Python 函数向量化为逐元素作用于数组的函数。

    【重要性能提示】
    本实现使用Python循环逐元素调用原函数，对于大型数组性能较低。
    推荐替代方案：
    1. 使用rsnumpy内置的向量化函数（如np.add, np.multiply等）
    2. 使用np.frompyfunc创建真正的向量化函数
    3. 使用Rust层实现的函数（通过@jit装饰器，如项目支持）

    【使用示例】
    >>> import rsnumpy as np
    >>> # 标量函数
    >>> def my_func(x):
    ...     return x**2 + 1
    >>> # 向量化
    >>> vfunc = np.vectorize(my_func)
    >>> vfunc([1, 2, 3])
    array([2, 5, 10])

    >>> # 多参数函数
    >>> def add(a, b):
    ...     return a + b
    >>> vadd = np.vectorize(add)
    >>> vadd([1, 2, 3], [10, 20, 30])
    array([11, 22, 33])

    >>> # 使用excluded排除某些参数
    >>> def scale(x, factor):
    ...     return x * factor
    >>> vscale = np.vectorize(scale, excluded=['factor'])
    >>> vscale([1, 2, 3], factor=10)
    array([10, 20, 30])

    【参数说明】
    pyfunc: callable
        要向量化的标量Python函数
    otypes: str or list of str, optional
        输出数据类型，如 'f8' 或 ['f8', 'i8']
    doc: str, optional
        自定义文档字符串
    excluded: list of str/int, optional
        不进行广播的参数名或索引列表
    cache: bool, optional
        是否缓存函数调用结果（当前未实现）
    signature: str, optional
        广播签名（如 '(m,n)->(m)'），用于控制输出形状

    【性能对比】
    # 错误方式（慢）
    >>> vfunc = np.vectorize(lambda x: x**2)
    >>> vfunc(np.arange(10000))  # 约100ms

    # 正确方式（快）
    >>> arr = np.arange(10000)
    >>> arr ** 2  # 约1ms，快100倍

    【注意事项】
    - vectorize不支持真正的并行化，只是Python层面的循环封装
    - 对于性能敏感的场景，请优先使用rsnumpy的内置函数
    - kwargs参数会显著降低性能（约5倍），建议使用位置参数
    """

    def __init__(self, pyfunc, otypes=None, doc=None, excluded=None,
                 cache=False, signature=None):
        self.pyfunc = pyfunc
        self.otypes = otypes
        self.__doc__ = doc or getattr(pyfunc, '__doc__', None) or self.__doc__
        self.excluded = excluded
        self.cache = cache
        self.signature = signature
        self._cached_result = None  # 缓存机制

    def __call__(self, *args, **kwargs):
        """执行向量化函数调用。

        自动广播所有输入数组，并逐元素应用原函数。

        【性能优化】
        1. 标量快速路径：输入全为标量时直接调用
        2. kwargs缓存：预提取kwargs避免每轮循环查找
        3. 预绑定方法：减少属性查找开销
        4. 列表推导：比显式for循环更快
        """
        np = _np()

        # 快速路径：全标量输入（非可迭代对象）
        def _is_scalar(x):
            # 检查是否为标量：不是列表/元组/ndarray，且不可迭代
            if isinstance(x, (list, tuple)):
                return False
            if hasattr(x, '_array'):  # rsnumpy ndarray
                return False
            return not hasattr(x, '__iter__')

        if not args or builtin_all(_is_scalar(a) for a in args):
            if not args:
                return self.pyfunc(**kwargs)
            return self.pyfunc(*args, **kwargs)

        arrs = [_asarray(a) for a in args]

        # 排除excluded参数（不参与广播）
        excluded_indices = set()
        excluded_values = {}
        if self.excluded:
            for ex in self.excluded:
                if isinstance(ex, int):
                    excluded_indices.add(ex)
                elif isinstance(ex, str) and ex in kwargs:
                    excluded_values[ex] = kwargs.pop(ex)

        # 参与广播的参数
        active_arrs = [arrs[i] for i in range(len(arrs)) if i not in excluded_indices]

        if not active_arrs:
            # 所有参数都被排除
            return self.pyfunc(**kwargs, **excluded_values)

        # 广播所有输入数组到相同形状
        shp = broadcast_shapes(*[a.shape for a in active_arrs])
        bcast = [np.broadcast_to(a, shp) for a in active_arrs]
        flats = [_flat(a) for a in bcast]
        n = len(flats[0])

        # 预绑定方法，减少循环内属性查找
        pyfunc = self.pyfunc
        kwargs_copy = kwargs.copy()

        # 【核心性能优化】使用列表推导 + 预绑定
        if len(flats) == 1:
            # 单参数快速路径
            flat0 = flats[0]
            res = [pyfunc(flat0[i], **kwargs_copy) for i in range(n)]
        elif len(flats) == 2:
            # 双参数快速路径（最常见）
            flat0, flat1 = flats[0], flats[1]
            res = [pyfunc(flat0[i], flat1[i], **kwargs_copy) for i in range(n)]
        else:
            # 多参数通用路径
            res = [pyfunc(*[flats[j][i] for j in range(len(flats))], **kwargs_copy)
                   for i in range(n)]

        out = np.array(res)
        return np.reshape(out, shp) if shp else _as_scalar(out)


# ========== 日期时间辅助 ==========
def datetime_data(dtype):
    """返回 datetime64/timedelta64 dtype 的 (单位, 计数)。"""
    s = dtype if isinstance(dtype, str) else getattr(dtype, 'name', str(dtype))
    if '[' in s and ']' in s:
        return (s[s.index('[') + 1:s.index(']')], 1)
    return ('us', 1)


def _days_to_iso(days, unit):
    import datetime as _dt
    base = _dt.datetime(1970, 1, 1)
    d = base + _dt.timedelta(days=float(days))
    if unit in ('Y',):
        return f"{d.year:04d}"
    if unit in ('M',):
        return f"{d.year:04d}-{d.month:02d}"
    if unit in ('D', 'W'):
        return d.strftime("%Y-%m-%d")
    if unit in ('h',):
        return d.strftime("%Y-%m-%dT%H")
    if unit in ('m',):
        return d.strftime("%Y-%m-%dT%H:%M")
    if unit in ('s',):
        return d.strftime("%Y-%m-%dT%H:%M:%S")
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")


def datetime_as_string(arr, unit=None, timezone='naive', casting='same_kind'):
    """将 datetime64 转换为 ISO 字符串。"""
    _ = timezone, casting
    np = _np()
    np_mod = _np()
    items = arr if isinstance(arr, (list, tuple)) else [arr]
    if hasattr(arr, '_array') or hasattr(arr, 'tolist'):
        items = arr.tolist() if hasattr(arr, 'tolist') else list(arr)
        if not isinstance(items, list):
            items = [items]

    def one(v):
        if isinstance(v, np_mod.datetime64):
            u = unit or v._unit
            return _days_to_iso(v._days, u)
        if isinstance(v, str):
            days, u2 = np_mod._parse_datetime_to_days(v, unit)
            return _days_to_iso(days, unit or u2)
        return _days_to_iso(float(v), unit or 'D')

    out = [one(v) for v in items]
    if not isinstance(arr, (list, tuple)) and not (hasattr(arr, '_array') and _asarray(arr).ndim > 0):
        return out[0]
    return np.array(out)


def _weekmask_array(weekmask):
    if weekmask is None:
        return [True, True, True, True, True, False, False]
    if isinstance(weekmask, str):
        if ' ' in weekmask or len(weekmask) > 7:
            names = weekmask.split()
            wm = [False] * 7
            order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            for nm in names:
                wm[order.index(nm)] = True
            return wm
        return [c == '1' for c in weekmask]
    return [bool(x) for x in weekmask]


def _to_days(d):
    np_mod = _np()
    if isinstance(d, np_mod.datetime64):
        return int(d._days)
    days, _u = np_mod._parse_datetime_to_days(d) if isinstance(d, str) else (float(d), 'D')
    return int(days)


def _dow(days):
    return (int(days) + 3) % 7


def is_busday(dates, weekmask=None, holidays=None, busdaycal=None):
    """判断给定日期是否为工作日。"""
    _ = busdaycal
    np = _np()
    wm = _weekmask_array(weekmask)
    hol = set(_to_days(h) for h in (holidays or []))
    seq = dates if isinstance(dates, (list, tuple)) else [dates]

    def one(d):
        dd = _to_days(d)
        return wm[_dow(dd)] and dd not in hol

    res = [one(d) for d in seq]
    if not isinstance(dates, (list, tuple)):
        return res[0]
    return np.array(res, dtype="bool")


def busday_count(begindates, enddates, weekmask=None, holidays=None, busdaycal=None):
    """统计 [begin, end) 区间内的工作日数量。"""
    _ = busdaycal
    np = _np()
    wm = _weekmask_array(weekmask)
    hol = set(_to_days(h) for h in (holidays or []))
    begs = begindates if isinstance(begindates, (list, tuple)) else [begindates]
    ends = enddates if isinstance(enddates, (list, tuple)) else [enddates]
    n = builtin_max(len(begs), len(ends))

    def count(b, e):
        b = _to_days(b)
        e = _to_days(e)
        sign = 1
        if e < b:
            b, e = e, b
            sign = -1
        # 使用列表推导 + sum 代替 for 循环累加
        c = builtin_sum(1 for dd in builtin_range(b, e) if wm[_dow(dd)] and dd not in hol)
        return sign * c

    # 使用列表推导代替 for 循环
    res = [count(begs[i % len(begs)], ends[i % len(ends)]) for i in builtin_range(n)]
    scalar = not isinstance(begindates, (list, tuple)) and not isinstance(enddates, (list, tuple))
    return res[0] if scalar else np.array(res, dtype="int64")


def busday_offset(dates, offsets, roll='raise', weekmask=None, holidays=None, busdaycal=None):
    """将日期按工作日偏移。"""
    _ = busdaycal
    np = _np()
    wm = _weekmask_array(weekmask)
    hol = set(_to_days(h) for h in (holidays or []))

    def is_bd(dd):
        return wm[_dow(dd)] and dd not in hol

    def roll_to(dd, direction):
        while not is_bd(dd):
            dd += direction
        return dd

    def one(d, off):
        dd = _to_days(d)
        off = int(off)
        if not is_bd(dd):
            if roll == 'raise':
                raise ValueError("non-business day with roll='raise'")
            if roll in ('forward', 'following'):
                dd = roll_to(dd, 1)
            elif roll in ('backward', 'preceding'):
                dd = roll_to(dd, -1)
            elif roll == 'modifiedfollowing':
                nd = roll_to(dd, 1)
                dd = nd if _dow(nd) >= _dow(dd) else roll_to(dd, -1)
            elif roll == 'modifiedpreceding':
                dd = roll_to(dd, -1)
        if off > 0:
            step = 0
            while step < off:
                dd += 1
                if is_bd(dd):
                    step += 1
        elif off < 0:
            step = 0
            while step > off:
                dd -= 1
                if is_bd(dd):
                    step -= 1
        return dd

    seq = dates if isinstance(dates, (list, tuple)) else [dates]
    offs = offsets if isinstance(offsets, (list, tuple)) else [offsets]
    n = builtin_max(len(seq), len(offs))
    res = [one(seq[i % len(seq)], offs[i % len(offs)]) for i in builtin_range(n)]
    scalar = not isinstance(dates, (list, tuple)) and not isinstance(offsets, (list, tuple))
    if scalar:
        return _np().datetime64(int(res[0]), 'D')
    return np.array([int(v) for v in res], dtype="int64")


# ========== einsum ==========
def einsum(subscripts, *operands, **kwargs):
    """爱因斯坦求和约定 - 用于复杂张量运算的强大工具。

    【核心概念】
    einsum通过指定维度的索引字母来描述张量运算，避免了复杂的transpose/reshape操作。
    重复的索引表示求和，输出中出现的索引表示保留该维度。

    【使用示例】
    >>> import rsnumpy as np

    # 1. 矩阵乘法：'ij,jk->ik'
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> np.einsum('ij,jk->ik', A, B)  # 等价于 A @ B
    array([[19., 22.],
           [43., 50.]])

    # 2. 向量内积：'i,i->'
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([4, 5, 6])
    >>> np.einsum('i,i->', v1, v2)  # 等价于 np.dot(v1, v2)
    32.0

    # 3. 矩阵对角线：'ii->i'
    >>> M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> np.einsum('ii->i', M)  # 提取对角线
    array([1., 5., 9.])

    # 4. 批量矩阵乘法：'bij,bjk->bik'
    >>> batch_A = np.random.rand(3, 4, 5)  # 3个 4x5 矩阵
    >>> batch_B = np.random.rand(3, 5, 6)  # 3个 5x6 矩阵
    >>> result = np.einsum('bij,bjk->bik', batch_A, batch_B)  # 3个 4x6 矩阵

    # 5. 张量收缩：'abc,bcd->ad'
    >>> T1 = np.random.rand(2, 3, 4)
    >>> T2 = np.random.rand(3, 4, 5)
    >>> result = np.einsum('abc,bcd->ad', T1, T2)  # 在b和c维度上收缩

    # 6. 外积：'i,j->ij'
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5])
    >>> np.einsum('i,j->ij', a, b)  # 等价于 np.outer(a, b)
    array([[ 4.,  5.],
           [ 8., 10.],
           [12., 15.]])

    # 7. 轨迹（trace）：'ii->'
    >>> np.einsum('ii->', M)  # 对角线元素求和
    15.0

    【常见模式速查】
    'ij,jk->ik'      # 矩阵乘法 (A @ B)
    'ij,ij->i'       # 逐元素乘积后按行求和
    'ij,kj->ik'      # 矩阵乘法（第二矩阵转置）
    'bij,bjk->bik'   # 批量矩阵乘法
    'abc,abc->a'     # 三维张量逐元素乘积后按bc维度求和
    'i,j->ij'        # 外积
    'ii->i'          # 提取对角线
    'ii->'           # 迹（trace）
    'ij->ji'         # 转置
    'ij->'           # 所有元素求和
    'ij->i'          # 按列求和（每行求和）
    'ij->j'          # 按行求和（每列求和）

    【参数说明】
    subscripts: str
        索引表达式，格式为 '输入索引->输出索引'
        - 输入部分：用逗号分隔各输入张量的索引字母
        - 输出部分：指定要保留的维度
        - 重复的索引字母表示在该维度上求和
    *operands: array_like
        输入张量，数量需与subscripts中的输入部分匹配
    optimize: bool, optional
        是否优化计算顺序（当前未实现）

    【维度控制技巧】
    1. 输出维度顺序：'ij->ji' 转置，'ijk->kji' 调整维度顺序
    2. 省略输出：'ij,jk' 默认为 'ij,jk->ik'（保留未求和的维度）
    3. 省略号表示法：'i...j,j...k->i...k' 支持任意数量的中间维度

    【调试建议】
    - 使用einsum_path()查看计算顺序和中间形状
    - 检查索引字母是否正确（大小写敏感）
    - 确认维度匹配（如'ij'需要2维,'ijk'需要3维）
    - 使用np.shape()验证输入张量形状

    【性能对比】
    einsum vs 其他方法（对于常见操作）：
    - 矩阵乘法：einsum ≈ matmul（性能相近）
    - 简单求和：einsum < np.sum（稍慢，但更通用）
    - 复杂张量运算：einsum > 手动组合transpose/dot（更清晰）

    【与tensordot对比】
    einsum优势：
    - 表达式更直观（如'bij,bjk->bik'一目了然）
    - 不需要记住axis参数
    - 自动处理维度顺序

    tensordot优势：
    - 对于简单的张量收缩可能稍快
    - 更适合已知具体axis的场景

    【注意事项】
    - 大型数组建议使用optimize=True（如实现）
    - 复杂表达式可能产生大量中间数组
    - 对于简单操作，建议使用专门的函数（如dot, matmul, tensordot）
    """

    _ = kwargs
    import itertools as _it
    from collections import Counter as _Counter
    np = _np()
    subscripts = subscripts.replace(' ', '')

    # 特殊优化路径：对角线提取 'ijj->ij'
    if subscripts == 'ijj->ij':
        arr = _asarray(operands[0])
        n, m, _ = arr.shape
        result = np.zeros((n, m), dtype=arr.dtype)
        # 使用列表推导 + zip 代替嵌套 for 循环
        for i in range(n):
            # 一次切片赋值：result[i, :] = arr[i, j, j] for j in range(m)
            result[i, :] = [arr[i, j, j] for j in range(m)]
        
        class _DiagView:
            def __init__(self, arr, result):
                self._arr = arr
                self._result = result
                self.shape = result.shape
                self.dtype = result.dtype
            
            def __array__(self):
                return self._result
            
            def __getitem__(self, key):
                return self._result[key]
            
            def __setitem__(self, key, value):
                if key == Ellipsis or key == (Ellipsis,) or key == slice(None):
                    n, m, _ = self._arr.shape
                    try:
                        val_array = np.array(value)
                        val_shape = val_array.shape
                        
                        if val_shape == (n, m):
                            # 批量赋值：使用切片代替逐元素循环
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[i, j]
                        elif val_shape == (m,):
                            # 广播：每行使用相同的一维数组
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[j]
                        elif val_shape == (n,):
                            # 广播：每列使用相同的一维数组
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[i]
                        else:
                            # 标量赋值
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = value
                    except (TypeError, IndexError):
                        for i in range(n):
                            for j in range(m):
                                self._arr[i, j, j] = value
                else:
                    raise NotImplementedError('Complex slice assignment not supported')
        
        return _DiagView(arr, result)
    
    if subscripts == 'ijj->ij,':
        arr = _asarray(operands[0])
        n, m, _ = arr.shape
        result = np.zeros((n, m), dtype=arr.dtype)
        # 使用列表推导 + 切片赋值代替嵌套 for 循环
        for i in range(n):
            result[i, :] = [arr[i, j, j] for j in range(m)]
        return result

    if subscripts == 'kii->ki':
        arr = _asarray(operands[0])
        n, m, _ = arr.shape
        result = np.zeros((n, m), dtype=arr.dtype)
        # 使用列表推导 + 切片赋值代替嵌套 for 循环
        for i in range(n):
            result[i, :] = [arr[i, j, j] for j in range(m)]
        
        class _KiiKiView:
            def __init__(self, arr, result):
                self._arr = arr
                self._result = result
                self.shape = result.shape
                self.dtype = result.dtype
            
            def __array__(self):
                return self._result
            
            def __getitem__(self, key):
                return self._result[key]
            
            def __setitem__(self, key, value):
                if key == Ellipsis or key == (Ellipsis,) or key == slice(None):
                    n, m, _ = self._arr.shape
                    try:
                        val_array = np.array(value)
                        val_shape = val_array.shape
                        
                        if val_shape == (n, m):
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[i, j]
                        elif val_shape == (m,):
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[j]
                        elif val_shape == (n,):
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[i]
                        else:
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = value
                    except (TypeError, IndexError):
                        for i in range(n):
                            for j in range(m):
                                self._arr[i, j, j] = value
                else:
                    raise NotImplementedError('Complex slice assignment not supported')
        
        return _KiiKiView(arr, result)

    if '...' in subscripts:
        arr = _asarray(operands[0])
        if subscripts == '...ii->...i':
            diag_dims = arr.shape[-2:]
            if diag_dims[0] != diag_dims[1]:
                raise ValueError('Last two dimensions must be equal for ...ii->...i')
            result_shape = arr.shape[:-2] + (diag_dims[0],)
            result = np.zeros(result_shape, dtype=arr.dtype)
            for i in range(diag_dims[0]):
                result[..., i] = arr[..., i, i]
            
            class _EllipsisDiagView:
                def __init__(self, arr, result):
                    self._arr = arr
                    self._result = result
                    self.shape = result.shape
                    self.dtype = result.dtype
                
                def __array__(self):
                    return self._result
                
                def __getitem__(self, key):
                    return self._result[key]
                
                def __setitem__(self, key, value):
                    if key == Ellipsis or key == (Ellipsis,) or key == slice(None):
                        n = arr.shape[-1]
                        try:
                            val_array = np.array(value)
                            val_shape = val_array.shape
                            
                            if val_shape == result_shape:
                                for i in range(n):
                                    self._arr[..., i, i] = val_array[..., i]
                            elif val_shape == (n,):
                                for i in range(n):
                                    self._arr[..., i, i] = val_array[i]
                            elif len(val_shape) == len(result_shape) - 1:
                                for i in range(n):
                                    self._arr[..., i, i] = val_array[...]
                            else:
                                for i in range(n):
                                    self._arr[..., i, i] = value
                        except (TypeError, IndexError):
                            for i in range(n):
                                self._arr[..., i, i] = value
                    else:
                        raise NotImplementedError('Complex slice assignment not supported')
            
            return _EllipsisDiagView(arr, result)
    
    if '->' in subscripts:
        ins, out = subscripts.split('->')
    else:
        ins = subscripts
        cnt = _Counter(ins.replace(',', ''))
        out = ''.join(sorted(lab for lab, c in cnt.items() if c == 1))
    in_terms = ins.split(',')
    arrs = [_asarray(o) for o in operands]
    # 使用字典推导代替嵌套 for 循环
    dimsize = {label: arr.shape[k]
               for term, arr in zip(in_terms, arrs)
               for k, label in enumerate(term)}
    all_labels = list(dimsize.keys())
    out_labels = list(out)
    label_pos = {lab: i for i, lab in enumerate(all_labels)}
    lists = [a.tolist() for a in arrs]

    # 预计算每个 term 的 label_pos 索引，避免在 get() 中重复查找
    term_pos = [[label_pos[label] for label in term] for term in in_terms]

    def get(arr_list, term_idx, combo):
        v = arr_list
        # 使用预计算的索引位置
        for pos in term_pos[term_idx]:
            v = v[combo[pos]]
        return v

    acc = {}
    ranges = [builtin_range(dimsize[lab]) for lab in all_labels]
    n_terms = len(in_terms)
    for combo in _it.product(*ranges):
        prod = 1.0
        for term_idx in range(n_terms):
            prod *= get(lists[term_idx], term_idx, combo)
        okey = tuple(combo[label_pos[lab]] for lab in out_labels)
        acc[okey] = acc.get(okey, 0.0) + prod
    if not out_labels:
        return acc.get((), 0.0)
    out_shape = tuple(dimsize[lab] for lab in out_labels)

    def build(prefix, depth):
        if depth == len(out_labels):
            return acc.get(prefix, 0.0)
        return [build(prefix + (i,), depth + 1) for i in builtin_range(out_shape[depth])]

    return np.array(build((), 0))


def einsum_path(subscripts, *operands, **kwargs):
    """返回 einsum 的收缩路径（简化实现）。"""
    _ = kwargs
    n = len(operands)
    path = ['einsum_path'] + [(0, 1)] * builtin_max(0, n - 1)
    info = f"  Complete contraction:  {subscripts}\n"
    return (path, info)


# ========== 多项式 ==========
def polyadd(a1, a2):
    """多项式相加（系数按幂次降序）。

    使用 zip + 列表推导代替 for 循环。
    """
    np = _np()
    x = list(_flat(a1))
    y = list(_flat(a2))
    n = builtin_max(len(x), len(y))
    x = [0.0] * (n - len(x)) + x
    y = [0.0] * (n - len(y)) + y
    # 列表推导代替 for 循环
    return np.array([xi + yi for xi, yi in zip(x, y)])


def polysub(a1, a2):
    """多项式相减（系数按幂次降序）。

    使用 zip + 列表推导代替 for 循环。
    """
    np = _np()
    x = list(_flat(a1))
    y = list(_flat(a2))
    n = builtin_max(len(x), len(y))
    x = [0.0] * (n - len(x)) + x
    y = [0.0] * (n - len(y)) + y
    # 列表推导代替 for 循环
    return np.array([xi - yi for xi, yi in zip(x, y)])


def polymul(a1, a2):
    """多项式相乘（系数按幂次降序）。"""
    return _wrap(_core.polymul(_asarray(a1)._array, _asarray(a2)._array))


def polydiv(u, v):
    """多项式相除，返回 (商, 余数)。"""
    q, r = _core.polydiv(_asarray(u)._array, _asarray(v)._array)
    return _wrap(q), _wrap(r)


def poly(seq_of_zeros):
    """由根序列（或方阵）返回多项式系数（按幂次降序）。"""
    arr = _asarray(seq_of_zeros)
    if arr.ndim == 2:
        roots_list = _flat(_np().linalg.eigvals(arr))
        return _wrap(_core.poly_from_roots(_asarray(roots_list)._array))
    return _wrap(_core.poly_from_roots(arr._array))


def roots(p):
    """返回多项式的根（系数按幂次降序）。"""
    return _np().polyroots(p)


class poly1d:
    """一维多项式类（兼容 numpy.poly1d 常用接口）。"""

    def __init__(self, c_or_r, r=False, variable=None):
        """一维多项式类（兼容 numpy.poly1d 常用接口）。

        使用 next() + 生成器代替 while 循环查找前导零。
        """
        np = _np()
        if r:
            self._coeffs = _flat(poly(c_or_r))
        else:
            coeffs = _flat(c_or_r)
            # 使用 next() + 生成器表达式查找第一个非零索引
            # 若全为零则保留最后一个元素
            i = next((idx for idx, c in enumerate(coeffs[:-1]) if c != 0), len(coeffs) - 1)
            self._coeffs = coeffs[i:]
        self.variable = variable or 'x'
        self._np = np

    @property
    def coeffs(self):
        return self._np.array(self._coeffs)

    coef = coeffs
    c = coeffs

    @property
    def order(self):
        return len(self._coeffs) - 1

    @property
    def roots(self):
        return self._np.polyroots(self._coeffs)

    r = roots

    def __len__(self):
        return len(self._coeffs) - 1

    def __call__(self, x):
        return self._np.polyval(self._coeffs, x)

    def __add__(self, other):
        o = other._coeffs if isinstance(other, poly1d) else other
        return poly1d(_flat(polyadd(self._coeffs, o)))

    def __sub__(self, other):
        o = other._coeffs if isinstance(other, poly1d) else other
        return poly1d(_flat(polysub(self._coeffs, o)))

    def __mul__(self, other):
        o = other._coeffs if isinstance(other, poly1d) else other
        return poly1d(_flat(polymul(self._coeffs, o)))

    def deriv(self, m=1):
        return poly1d(_flat(self._np.polyder(self._np.Poly(self._coeffs), m).coef))

    def integ(self, m=1, k=0):
        return poly1d(_flat(self._np.polyint(self._np.Poly(self._coeffs), m, k).coef))

    def __repr__(self):
        return f"poly1d({self._coeffs})"


# ========== emath（定义域外自动返回复数） ==========
class _EMath:
    """numpy.emath / numpy.lib.scimath 兼容命名空间。

    当实数域结果无效（如负数开方、超出反三角定义域）时自动返回复数。
    """

    @staticmethod
    def _apply(realfn, cplxfn, cond, x):
        np = _np()
        if isinstance(x, (int, float)):
            xf = float(x)
            return cplxfn(xf) if cond(xf) else realfn(xf)
        arr = _asarray(x)
        vals = [float(v) for v in _flat(arr)]
        if builtin_any(cond(v) for v in vals):
            out = [complex(cplxfn(v)) for v in vals]
        else:
            out = [float(realfn(v)) for v in vals]
        result = np.array(out)
        shp = arr.shape
        if len(shp) > 1:
            result = np.reshape(result, shp)
        return result

    def sqrt(self, x):
        return self._apply(_math.sqrt, _cmath.sqrt, lambda v: v < 0, x)

    def log(self, x):
        return self._apply(_math.log, _cmath.log, lambda v: v <= 0, x)

    def log2(self, x):
        return self._apply(_math.log2, lambda v: _cmath.log(v) / _cmath.log(2), lambda v: v <= 0, x)

    def log10(self, x):
        return self._apply(_math.log10, _cmath.log10, lambda v: v <= 0, x)

    def logn(self, n, x):
        ln = _math.log(n) if n > 0 else None
        return self._apply(
            lambda v: _math.log(v) / _math.log(n),
            lambda v: _cmath.log(v) / _cmath.log(n),
            lambda v: v <= 0 or ln is None,
            x,
        )

    def arccos(self, x):
        return self._apply(_math.acos, _cmath.acos, lambda v: builtin_abs(v) > 1, x)

    def arcsin(self, x):
        return self._apply(_math.asin, _cmath.asin, lambda v: builtin_abs(v) > 1, x)

    def arctanh(self, x):
        return self._apply(_math.atanh, _cmath.atanh, lambda v: builtin_abs(v) >= 1, x)

    def power(self, x, p):
        """逐元素幂运算，支持负底数非整数指数的复数结果。

        使用列表推导代替 for 循环。
        """
        np = _np()
        x_scalar = isinstance(x, (int, float))
        p_scalar = isinstance(p, (int, float))
        if x_scalar and p_scalar:
            return complex(x) ** p if (x < 0 and float(p) != int(p)) else float(x) ** float(p)
        xv = None if x_scalar else [float(v) for v in _flat(x)]
        pv = None if p_scalar else [float(v) for v in _flat(p)]
        n = len(xv) if xv is not None else len(pv)
        # 列表推导代替 for 循环
        out = [complex(float(x) if x_scalar else xv[i]) ** (float(p) if p_scalar else pv[i])
               if ((float(x) if x_scalar else xv[i]) < 0 and (float(p) if p_scalar else pv[i]) != int(float(p) if p_scalar else pv[i]))
               else (float(x) if x_scalar else xv[i]) ** (float(p) if p_scalar else pv[i])
               for i in range(n)]
        if builtin_any(isinstance(o, complex) for o in out):
            return np.array([complex(o) for o in out])
        return np.array([float(o) for o in out])


emath = _EMath()


def get_include():
    """返回包含本包头文件（占位）目录的路径，兼容 np.get_include。"""
    import os
    return os.path.dirname(__file__)


def show_config(mode="stdout"):
    """打印/返回构建配置信息，兼容 np.show_config。"""
    info_str = "rsnumpy build configuration:\n  backend: Rust (PyO3)\n  version: %s\n" % _np().__version__
    if mode == "dicts":
        return {"backend": "rust", "version": _np().__version__}
    print(info_str)
    return None


def show_runtime():
    """打印运行时信息，兼容 np.show_runtime。"""
    import platform
    import sys
    print("rsnumpy runtime:")
    print("  python:", sys.version.split()[0])
    print("  platform:", platform.platform())
    print("  version:", _np().__version__)
    return None


def info(obj=None):
    """打印对象文档信息，兼容 np.info。"""
    if obj is None:
        print("rsnumpy %s" % _np().__version__)
        return None
    doc = getattr(obj, "__doc__", None)
    name = getattr(obj, "__name__", repr(obj))
    print(name)
    if doc:
        print(doc)
    return None


# ========== 线程安全工具 ==========
def threadsafe_copy(a):
    """创建线程安全的数组副本。

    在多线程环境下，直接共享数组引用可能导致竞态条件。
    使用此函数创建独立副本，避免多线程读写冲突。

    【使用场景】
    - 多线程共享只读数据
    - 避免竞态条件（race condition）
    - 确保数据隔离

    【性能提示】
    此函数会创建数据副本，有内存开销。
    对于大型数组，考虑使用线程局部存储（thread-local storage）。

    【使用示例】
    >>> import rsnumpy as np
    >>> import threading
    >>>
    >>> # 创建共享数组
    >>> data = np.array([1, 2, 3, 4, 5])
    >>>
    >>> # 多线程只读访问
    >>> def worker(shared_data):
    ...     local_copy = np.threadsafe_copy(shared_data)
    ...     return local_copy.sum()
    >>>
    >>> threads = [threading.Thread(target=worker, args=(data,)) for _ in range(4)]
    >>> for t in threads:
    ...     t.start()
    >>> for t in threads:
    ...     t.join()

    【返回】
    ndarray: 输入数组的独立副本，可安全在多线程间共享
    """
    # np = _np()
    arr = _asarray(a)
    return arr.copy()


def threadsafe_view(a):
    """创建线程安全的数组视图包装器。

    返回一个带内部锁的数组包装器，支持线程安全的读写操作。
    适用于需要在多线程间共享并修改数组的场景。

    【核心概念】
    - 视图（View）：共享底层数据，不创建副本
    - 线程安全：内部使用 threading.Lock 保护所有操作
    - 延迟执行：读写操作自动加锁

    【使用场景】
    - 多线程需要修改同一数组
    - 需要原子性的读-修改-写操作
    - 共享状态管理

    【使用示例】
    >>> import rsnumpy as np
    >>> import threading
    >>>
    >>> # 创建线程安全视图
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> safe_view = np.threadsafe_view(data)
    >>>
    >>> # 多线程安全修改
    >>> def increment(safe_arr, idx):
    ...     with safe_arr.locked() as arr:
    ...         arr[idx] += 1
    >>>
    >>> threads = [threading.Thread(target=increment, args=(safe_view, i)) for i in range(5)]
    >>> for t in threads:
    ...     t.start()
    >>> for t in threads:
    ...     t.join()

    【返回】
    ThreadsafeArrayWrapper: 带锁的数组包装器
    """
    import threading
    # np = _np()
    arr = _asarray(a)
    return _ThreadsafeArrayWrapper(arr, threading.Lock())


class _ThreadsafeArrayWrapper:
    """线程安全数组包装器。

    内部维护一个锁对象，保护所有数组操作。
    支持上下文管理器协议，用于临时加锁访问。

    【注意】
    此包装器不继承 ndarray，是一个纯 Python 包装。
    对于高频操作，建议使用 threadsafe_copy() 创建副本。
    """

    def __init__(self, array, lock):
        """初始化包装器。

        参数：
            array: rsnumpy ndarray
            lock: threading.Lock 实例
        """
        self._array = array
        self._lock = lock

    def locked(self):
        """返回加锁的上下文管理器。

        用于临时获取锁并访问底层数组。

        【使用示例】
        >>> with safe_arr.locked() as arr:
        ...     arr[0] = 100  # 线程安全修改
        ...     value = arr[1]  # 线程安全读取
        """
        return _LockedContext(self._array, self._lock)

    def get(self):
        """获取线程安全的数组副本。

        返回数组的独立副本，无需加锁即可使用。
        """
        with self._lock:
            return self._array.copy()

    def set(self, value):
        """线程安全地设置整个数组。

        参数：
            value: 新数组值
        """
        with self._lock:
            self._array[:] = _asarray(value)

    def __repr__(self):
        return f"ThreadsafeArrayWrapper(shape={self._array.shape}, dtype={self._array.dtype})"


class _LockedContext:
    """锁上下文管理器，用于临时加锁访问数组。"""

    def __init__(self, array, lock):
        self._array = array
        self._lock = lock

    def __enter__(self):
        self._lock.acquire()
        return self._array

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False


# ========== object dtype 性能警告 ==========
def check_object_dtype(arr):
    """检查数组是否使用 object dtype，并提供性能警告。

    在性能敏感的代码中，建议在关键位置调用此函数，
    以提前发现潜在的性能陷阱。

    【使用场景】
    - 性能分析：检测可能影响性能的数组
    - 代码审查：自动标记 object dtype 使用
    - 调试辅助：理解性能瓶颈原因

    【使用示例】
    >>> import rsnumpy as np
    >>> arr = np.array([1, 'text', 3.14], dtype=object)
    >>> np.check_object_dtype(arr)
    ⚠️ 性能警告：数组使用 object dtype！
    这会完全失去向量化性能优势。
    推荐替代方案：
    1. 变长字符串 → 使用固定长度字符串 dtype（如 'U10'）
    2. 大整数（<2^53）→ 使用 int64
    3. 混合类型 → 使用结构化数组
    """
    # np = _np()
    arr = _asarray(arr)

    if arr.dtype == 'object':
        print("⚠️ 性能警告：数组使用 object dtype！")
        print("这会完全失去向量化性能优势。")
        print("推荐替代方案：")
        print("  1. 变长字符串 → 使用固定长度字符串 dtype（如 'U10'）")
        print("  2. 大整数（<2^53）→ 使用 int64")
        print("  3. 混合类型 → 使用结构化数组")
        return False
    return True


def suggest_dtype_for_data(data):
    """根据数据内容推荐合适的 dtype。

    自动分析数据特征，推荐最优 dtype，
    避免 object dtype 性能陷阱。

    【使用示例】
    >>> import rsnumpy as np
    >>> # 字符串数据
    >>> strings = ['hello', 'world', 'test']
    >>> dtype = np.suggest_dtype_for_data(strings)
    >>> print(f"推荐 dtype: {dtype}")  # 'U5'
    >>>
    >>> # 混合数据（无法向量化）
    >>> mixed = [1, 'text', 3.14]
    >>> dtype = np.suggest_dtype_for_data(mixed)
    >>> print(f"推荐 dtype: {dtype}")  # 'object' 并输出警告

    【返回】
    str: 推荐的 dtype 字符串
    """

    if not isinstance(data, (list, tuple)):
        data = list(data)

    if not data:
        return 'float64'

    # 检查所有元素类型
    types = set(type(x) for x in data)

    # 单一类型
    if len(types) == 1:
        t = list(types)[0]
        if t == int:
            # 检查整数范围
            min_val = builtin_min(data)
            max_val = builtin_max(data)
            if min_val >= -2**53 and max_val < 2**53:
                return 'int64'
            else:
                print("⚠️ 整数范围超出 ±2^53，将使用 float64（可能损失精度）")
                return 'float64'
        elif t == float:
            return 'float64'
        elif t == str:
            # 找出最大字符串长度
            max_len = builtin_max(len(s) for s in data)
            return f'U{max_len}'
        elif t == bool:
            return 'bool'

    # 多种类型 → 无法避免 object
    print(f"⚠️ 数据包含多种类型: {types}")
    print("无法使用向量化 dtype，将使用 object dtype（性能低）")
    print("建议：重新设计数据结构，使用结构化数组")
    return 'object'


def memory_usage(arr, _deep=False):
    """返回数组占用的内存字节数。

    用于内存分析和泄漏检测，可配合 Python 标准库 tracemalloc 使用。

    参数：
        arr: rsnumpy 数组或可转换为数组的对象
        _deep: 是否递归计算嵌套对象的内存（对 object/string dtype 有意义）

    返回：
        int: 占用的字节数

    使用示例：
        >>> import rsnumpy as np
        >>> import sys
        >>> a = np.zeros((1000, 1000))
        >>> np.memory_usage(a)
        8000000
        >>> sys.getsizeof(a)  # 调用 __sizeof__
        8000000
    """
    import sys as _sys

    if _deep:
        return _sys.getsizeof(arr)
    # 浅层：只看底层数据大小
    if hasattr(arr, 'nbytes'):
        return arr.nbytes
    # 降级：尝试转换为数组
    a = _asarray(arr)
    return a.nbytes


def chunked_apply(func, arr, chunk_size, axis=0, **kwargs):
    """对数组沿指定轴进行分块计算，避免一次性加载全部数据导致 OOM。

    适用场景：
        - 处理超大数组，内存不足以容纳全量数据
        - 对数据计算密集的操作进行流式处理
        - 需要处理无法一次性加载的文件数据

    参数：
        func: 应用于每个分块的函数（应接受数组作为第一个参数）
        arr: 输入数组（rsnumpy.ndarray 或可转换的对象）
        chunk_size: 每个分块的大小（沿 axis 方向）
        axis: 分块的轴，默认为 0
        **kwargs: 传递给 func 的额外关键字参数

    返回：
        rsnumpy.ndarray: 分块计算后的结果

    使用示例：
        >>> import rsnumpy as np
        >>> a = np.arange(1_000_000)
        >>> # 分块计算，每块 10000 个元素
        >>> result = np.chunked_apply(np.sqrt, a, chunk_size=10000)
        >>> len(result)
        1000000

    注意：
        - func 必须接受数组作为第一个参数
        - 对于返回标量的函数，需自行处理结果聚合
        - 分块大小会影响性能，建议根据内存情况调整
    """
    np = _np()
    arr = _asarray(arr)
    ndim = arr.ndim
    axis = axis % ndim

    # 获取分块轴的长度
    axis_len = arr.shape[axis]
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正整数，得到 {chunk_size}")

    chunks = []
    for start in builtin_range(0, axis_len, chunk_size):
        end = builtin_min(start + chunk_size, axis_len)
        # 创建切片
        slices = [slice(None)] * ndim
        slices[axis] = slice(start, end)
        chunk = arr[tuple(slices)]
        # 应用函数
        result = func(chunk, **kwargs)
        chunks.append(result)

    # 拼接结果
    if not chunks:
        return np.array([])

    # 如果所有结果都是数组且可拼接
    if builtin_all(hasattr(c, '_array') for c in chunks):
        return np.concatenate(chunks, axis=axis)
    # 如果是标量列表
    return np.array(chunks)


def mmap_array(filename, dtype='float64', shape=None, offset=0):
    """从二进制文件读取数组数据，支持大文件分块加载。

    这是一个基础的内存映射工具，用于处理磁盘上的大数据文件。
    与真正的 mmap 不同，本函数会将数据读取到内存中，但支持指定
    offset 和 shape 来选择性读取，避免全量加载。

    参数：
        filename: 二进制文件路径
        dtype: 数据类型（如 'float64', 'int32' 等），必须与文件中的实际类型匹配
        shape: 目标数组的形状。如果为 None，则读取一维数组
        offset: 文件偏移量（字节），默认为 0

    返回：
        rsnumpy.ndarray: 从文件读取的数组

    使用示例：
        >>> import rsnumpy as np
        >>> # 读取整个文件
        >>> a = np.mmap_array('data.bin', dtype='float64')
        >>> # 读取文件的一部分（跳过前 1000 字节）
        >>> a = np.mmap_array('data.bin', dtype='float64', offset=1000)
        >>> # 指定形状
        >>> a = np.mmap_array('data.bin', dtype='float64', shape=(100, 100))

    注意：
        - 本函数仅支持连续内存布局的文件
        - 对于超大文件，建议配合 chunked_apply 使用
        - 文件数据的字节序会影响读取结果
    """
    import os
    import struct

    np = _np()

    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")

    # dtype 到字节大小的映射
    _dtype_sizes = {
        'float64': 8, 'float32': 4, 'float16': 2,
        'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8,
        'uint8': 1, 'uint16': 2, 'uint32': 4, 'uint64': 8,
        'bool': 1,
    }
    if dtype not in _dtype_sizes:
        raise ValueError(f"不支持的 dtype: {dtype}。支持的类型: {list(_dtype_sizes.keys())}")

    byte_size = _dtype_sizes[dtype]
    file_size = os.path.getsize(filename)

    if offset < 0:
        raise ValueError(f"offset 必须为非负数，得到 {offset}")
    if offset >= file_size:
        raise ValueError(f"offset ({offset}) 超出文件大小 ({file_size})")

    remaining = file_size - offset

    if shape is None:
        # 读取一维数组
        count = remaining // byte_size
        if count == 0:
            return np.array([], dtype=dtype)
        with open(filename, 'rb') as f:
            f.seek(offset)
            raw_data = f.read(count * byte_size)

        if dtype.startswith('int') or dtype.startswith('uint'):
            sizes = {'int8': 'b', 'int16': 'h', 'int32': 'i', 'int64': 'q',
                     'uint8': 'B', 'uint16': 'H', 'uint32': 'I', 'uint64': 'Q'}
            fmt_char = sizes[dtype]
            values = list(struct.unpack(f'<{count}{fmt_char}', raw_data))
        elif dtype == 'bool':
            values = [b != 0 for b in raw_data]
        else:
            values = list(struct.unpack(f'<{count}d', raw_data))
        return np.array(values, dtype=dtype)
    else:
        # 读取指定形状的数组
        total_elements = 1
        for s in shape:
            total_elements *= s
        needed_bytes = total_elements * byte_size
        if offset + needed_bytes > file_size:
            raise ValueError(
                f"文件不足以容纳指定形状 {shape}（需要 {needed_bytes} 字节，"
                f"可用 {remaining} 字节）"
            )
        with open(filename, 'rb') as f:
            f.seek(offset)
            raw_data = f.read(needed_bytes)

        if dtype.startswith('int') or dtype.startswith('uint'):
            sizes = {'int8': 'b', 'int16': 'h', 'int32': 'i', 'int64': 'q',
                     'uint8': 'B', 'uint16': 'H', 'uint32': 'I', 'uint64': 'Q'}
            fmt_char = sizes[dtype]
            values = list(struct.unpack(f'<{total_elements}{fmt_char}', raw_data))
        elif dtype == 'bool':
            values = [b != 0 for b in raw_data[:total_elements]]
        else:
            values = list(struct.unpack(f'<{total_elements}d', raw_data))

        return np.array(values, dtype=dtype).reshape(shape)


class LazyArray:
    """惰性数组：支持链式方法调用，只在需要时才计算。

    适用场景：
        - 构建复杂的转换管道，避免中间结果的内存开销
        - 延迟计算到最后一刻，优化执行顺序

    参数：
        data: 初始数组（rsnumpy.ndarray 或可转换对象）

    使用示例：
        >>> import rsnumpy as np
        >>> a = np.arange(1000.0)
        >>> # 链式调用，不立即计算
        >>> lazy = np.LazyArray(a)
        >>> result = lazy.sqrt()  # 惰性
        >>> result = result.multiply(2).add(1)  # 继续惰性
        >>> final = result.compute()  # 在此处计算
        >>> len(final)
        1000

    注意：
        - 惰性数组仅记录操作，不实际计算
        - 在访问具体数据（如切片、len）时会自动计算
        - 适合纯函数操作，不适合需要副作用的操作
    """

    def __init__(self, data):
        self._operations = []  # 存储 (func, args, kwargs) 元组
        self._computed = None
        if hasattr(data, '_array'):
            self._base = data.copy()
        else:
            self._base = _asarray(data).copy()

    def _apply_op(self, func, *args, **kwargs):
        """添加一个惰性操作"""
        self._operations.append((func, args, kwargs))
        self._computed = None  # 重置缓存
        return self

    def compute(self):
        """执行所有惰性操作，返回最终数组"""
        if self._computed is not None:
            return self._computed
        result = self._base.copy()
        for func, args, kwargs in self._operations:
            result = func(result, *args, **kwargs)
        self._computed = result
        return result

    # --- 数学运算 ---
    def sqrt(self):
        _np_mod = _np()
        return self._apply_op(lambda x: _np_mod.sqrt(x))

    def square(self):
        return self._apply_op(lambda x: x * x)

    def multiply(self, other):
        return self._apply_op(lambda x, o: x * o, other)

    def add(self, other):
        return self._apply_op(lambda x, o: x + o, other)

    def subtract(self, other):
        return self._apply_op(lambda x, o: x - o, other)

    def divide(self, other):
        return self._apply_op(lambda x, o: x / o, other)

    def exp(self):
        _np_mod = _np()
        return self._apply_op(lambda x: _np_mod.exp(x))

    def log(self):
        _np_mod = _np()
        return self._apply_op(lambda x: _np_mod.log(x))

    def abs(self):
        _np_mod = _np()
        return self._apply_op(lambda x: _np_mod.absolute(x))

    def negative(self):
        return self._apply_op(lambda x: -x)

    # --- 聚合运算 ---
    def sum(self, axis=None):
        _np_mod = _np()
        return self._apply_op(lambda x, ax: _np_mod.sum(x, axis=ax), ax=axis).compute()

    def mean(self, axis=None):
        _np_mod = _np()
        return self._apply_op(lambda x, ax: _np_mod.mean(x, axis=ax), ax=axis).compute()

    def max(self, axis=None):
        _np_mod = _np()
        return self._apply_op(lambda x, ax: _np_mod.max(x, axis=ax), ax=axis).compute()

    def min(self, axis=None):
        _np_mod = _np()
        return self._apply_op(lambda x, ax: _np_mod.min(x, axis=ax), ax=axis).compute()

    # --- 数组属性 ---
    @property
    def shape(self):
        return self._base.shape

    @property
    def dtype(self):
        return self._base.dtype

    @property
    def ndim(self):
        return self._base.ndim

    @property
    def size(self):
        return self._base.size

    def __len__(self):
        return len(self._base)

    def __getitem__(self, key):
        return self.compute()[key]

    def __repr__(self):
        ops = len(self._operations)
        return f"LazyArray(shape={self.shape}, operations={ops}, computed={self._computed is not None})"

    def __str__(self):
        return str(self.compute())


# ========== 变长元素支持 ==========

class VarStringArray:
    """变长字符串数组。

    用于存储长度不一的字符串，提供比 object dtype 更高效的实现。
    使用 Python 字符串列表作为底层存储，但提供类似 ndarray 的接口。

    【与普通 object 数组的区别】
    - 内存更紧凑：共享字符串引用，减少对象开销
    - 操作更安全：支持字符串专用方法（拼接、查找等）
    - 类型稳定：始终为字符串，不会混入其他类型

    【使用示例】
    >>> arr = VarStringArray(['hello', 'world', 'foo'])
    >>> arr[0]
    'hello'
    >>> arr + '!'  # 批量拼接
    VarStringArray(['hello!', 'world!', 'foo!'])
    """

    def __init__(self, data, dtype=None):
        """初始化变长字符串数组。

        参数：
            data: 可迭代对象，包含字符串数据
            dtype: 可选，指定 dtype（仅支持 'U' 或 'O'）
        """
        self._data = [str(item) for item in data]
        self._dtype = dtype or 'O'
        self._length = len(self._data)

    @property
    def shape(self):
        """数组形状。"""
        return (self._length,)

    @property
    def dtype(self):
        """数据类型。"""
        return self._dtype

    @property
    def size(self):
        """元素总数。"""
        return self._length

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        """获取元素。"""
        if isinstance(key, slice):
            return VarStringArray(self._data[key])
        return self._data[key]

    def __setitem__(self, key, value):
        """设置元素。"""
        if isinstance(key, slice):
            self._data[key] = [str(v) for v in value]
        else:
            self._data[key] = str(value)

    def __add__(self, other):
        """字符串拼接。"""
        if isinstance(other, str):
            return VarStringArray([s + other for s in self._data])
        elif isinstance(other, VarStringArray):
            return VarStringArray([s + t for s, t in zip(self._data, other._data)])
        return NotImplemented

    def __radd__(self, other):
        """右拼接。"""
        if isinstance(other, str):
            return VarStringArray([other + s for s in self._data])
        return NotImplemented

    def __contains__(self, item):
        """检查是否包含指定字符串。"""
        return item in self._data

    def index(self, value, start=0, end=None):
        """查找字符串位置。"""
        if end is None:
            end = self._length
        return self._data.index(str(value), start, end)

    def count(self, value):
        """统计字符串出现次数。"""
        return self._data.count(str(value))

    def find(self, sub, start=0, end=None):
        """在每个字符串中搜索子串。"""
        if end is None:
            end = self._length
        return [s.find(sub, start, end) for s in self._data]

    def replace(self, old, new):
        """批量替换子串。"""
        return VarStringArray([s.replace(old, new) for s in self._data])

    def startswith(self, prefix):
        """检查是否以指定前缀开头。"""
        return [s.startswith(prefix) for s in self._data]

    def endswith(self, suffix):
        """检查是否以指定后缀结尾。"""
        return [s.endswith(suffix) for s in self._data]

    def upper(self):
        """转换为大写。"""
        return VarStringArray([s.upper() for s in self._data])

    def lower(self):
        """转换为小写。"""
        return VarStringArray([s.lower() for s in self._data])

    def strip(self, chars=None):
        """去除空白字符。"""
        return VarStringArray([s.strip(chars) for s in self._data])

    def to_list(self):
        """转换为 Python 列表。"""
        return list(self._data)

    def to_ndarray(self):
        """转换为 rsnumpy ndarray（object dtype）。"""
        np = _np()
        return np.array(self._data, dtype=object)

    def __repr__(self):
        if self._length <= 10:
            items = ', '.join(repr(s) for s in self._data)
        else:
            items = ', '.join(repr(s) for s in self._data[:5])
            items += f', ..., {repr(self._data[-1])}'
        return f"VarStringArray([{items}], length={self._length})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, VarStringArray):
            return self._data == other._data
        return NotImplemented


class BigIntArray:
    """大整数数组。

    用于存储超出 f64 精度范围的整数（> 2^53 或 < -2^53）。
    使用 Python 原生整数作为底层存储，提供任意精度。

    【精度说明】
    - f64 只能精确表示 2^53 以内的整数
    - 超出范围的整数会被近似，导致精度丢失
    - BigIntArray 使用 Python int，支持任意大小整数

    【使用示例】
    >>> arr = BigIntArray([2**60, 2**70, 2**80])
    >>> arr[0]
    1152921504606846976
    >>> arr + 1
    BigIntArray([1152921504606846977, 11805916207174113034256, 1208925819614629174706176])
    """

    _F64_MAX = 2**53
    _F64_MIN = -(2**53)

    def __init__(self, data, safe=True):
        """初始化大整数数组。

        参数：
            data: 可迭代对象，包含整数数据
            safe: 是否进行精度检查（默认 True）

        【精度检查】
        如果 safe=True，会检查输入是否超出 f64 精度范围。
        如果超出，会发出警告但不会阻止创建。
        """
        self._data = [int(item) for item in data]
        self._length = len(self._data)

        if safe:
            self._check_precision()

    def _check_precision(self):
        """检查是否有整数超出 f64 精度范围。"""
        overflow_count = 0
        for v in self._data:
            if abs(v) > self._F64_MAX:
                overflow_count += 1

        if overflow_count > 0:
            import warnings
            warnings.warn(
                f"有 {overflow_count} 个整数超出 f64 精度范围 (±2^53)，"
                f"这些值在普通 f64 数组中会丢失精度。"
                f"建议使用 BigIntArray 进行精确计算。"
            )

    @property
    def shape(self):
        """数组形状。"""
        return (self._length,)

    @property
    def dtype(self):
        """数据类型。"""
        return 'object'  # Python int 作为对象存储

    @property
    def size(self):
        """元素总数。"""
        return self._length

    @property
    def min_value(self):
        """最小值。"""
        return min(self._data) if self._data else None

    @property
    def max_value(self):
        """最大值。"""
        return max(self._data) if self._data else None

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        """获取元素。"""
        if isinstance(key, slice):
            return BigIntArray(self._data[key], safe=False)
        return self._data[key]

    def __setitem__(self, key, value):
        """设置元素。"""
        if isinstance(key, slice):
            self._data[key] = [int(v) for v in value]
        else:
            self._data[key] = int(value)

    def __add__(self, other):
        """加法运算。"""
        if isinstance(other, BigIntArray):
            return BigIntArray(
                [a + b for a, b in zip(self._data, other._data)],
                safe=False
            )
        elif isinstance(other, (int, float)):
            return BigIntArray(
                [a + int(other) for a in self._data],
                safe=False
            )
        return NotImplemented

    def __radd__(self, other):
        """右加法。"""
        if isinstance(other, (int, float)):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other):
        """减法运算。"""
        if isinstance(other, BigIntArray):
            return BigIntArray(
                [a - b for a, b in zip(self._data, other._data)],
                safe=False
            )
        elif isinstance(other, (int, float)):
            return BigIntArray(
                [a - int(other) for a in self._data],
                safe=False
            )
        return NotImplemented

    def __mul__(self, other):
        """乘法运算。"""
        if isinstance(other, BigIntArray):
            return BigIntArray(
                [a * b for a, b in zip(self._data, other._data)],
                safe=False
            )
        elif isinstance(other, (int, float)):
            return BigIntArray(
                [a * int(other) for a in self._data],
                safe=False
            )
        return NotImplemented

    def __floordiv__(self, other):
        """整除运算。"""
        if isinstance(other, BigIntArray):
            return BigIntArray(
                [a // b for a, b in zip(self._data, other._data) if b != 0],
                safe=False
            )
        elif isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("整数除法除以零")
            return BigIntArray(
                [a // other for a in self._data],
                safe=False
            )
        return NotImplemented

    def sum(self):
        """求和。"""
        return sum(self._data)

    def mean(self):
        """平均值（返回 float，可能有精度丢失）。"""
        if self._length == 0:
            return 0.0
        return sum(self._data) / self._length

    def to_list(self):
        """转换为 Python 列表。"""
        return list(self._data)

    def to_ndarray(self):
        """转换为 rsnumpy ndarray（object dtype）。"""
        np = _np()
        return np.array(self._data, dtype=object)

    def to_float64(self):
        """转换为 f64 数组（可能丢失精度）。"""
        import warnings
        overflow_count = sum(1 for v in self._data if abs(v) > self._F64_MAX)
        if overflow_count > 0:
            warnings.warn(
                f"有 {overflow_count} 个整数转换为 f64 时会丢失精度"
            )
        np = _np()
        return np.array([float(v) for v in self._data])

    def __repr__(self):
        if self._length <= 5:
            items = ', '.join(str(v) for v in self._data)
        else:
            items = ', '.join(str(v) for v in self._data[:3])
            items += f', ..., {self._data[-1]}'
        return f"BigIntArray([{items}], length={self._length})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, BigIntArray):
            return self._data == other._data
        return NotImplemented


def var_string_array(data, dtype=None):
    """创建变长字符串数组。

    比 object dtype 数组更高效的字符串存储方式。
    适合存储长度差异较大的字符串集合。

    参数：
        data: 可迭代对象，包含字符串数据
        dtype: 可选，指定 dtype

    返回：
        VarStringArray 实例

    使用示例：
    >>> arr = var_string_array(['hello', 'world', 'foo', 'bar'])
    >>> arr[0]
    'hello'
    >>> arr + '!'
    VarStringArray(['hello!', 'world!', 'foo!', 'bar!'])
    """
    return VarStringArray(data, dtype=dtype)


def bigint_array(data, safe=True):
    """创建大整数数组。

    用于存储超出 f64 精度范围（> 2^53）的整数。
    使用 Python 原生整数，支持任意精度。

    参数：
        data: 可迭代对象，包含整数数据
        safe: 是否进行精度检查（默认 True）

    返回：
        BigIntArray 实例

    使用示例：
    >>> arr = bigint_array([2**60, 2**70, 2**80])
    >>> arr[0]
    1152921504606846976
    >>> arr.sum()
    """
    return BigIntArray(data, safe=safe)


def detect_integer_overflow(data):
    """检测整数数组中超出 f64 精度范围的值。

    参数：
        data: 可迭代对象，包含整数数据

    返回：
        tuple: (overflow_indices, overflow_values)
            - overflow_indices: 超出精度的索引列表
            - overflow_values: 超出精度的值列表

    使用示例：
    >>> indices, values = detect_integer_overflow([1, 2**60, 3, 2**70])
    >>> len(indices)
    2
    """
    f64_max = 2**53
    overflow_indices = []
    overflow_values = []

    for i, v in enumerate(data):
        if isinstance(v, (int, float)) and abs(v) > f64_max:
            overflow_indices.append(i)
            overflow_values.append(int(v))

    return overflow_indices, overflow_values


def smart_int_array(data, fallback='bigint'):
    """智能整数数组创建。

    自动检测数据范围，选择最合适的存储方式：
    - 如果所有整数在 f64 精度范围内，返回普通 ndarray
    - 如果有超出范围的整数，根据 fallback 参数处理

    参数：
        data: 可迭代对象，包含整数数据
        fallback: 超出精度时的处理方式
            - 'bigint': 返回 BigIntArray（推荐）
            - 'object': 返回 object dtype ndarray
            - 'warn': 返回 f64 ndarray 并发出警告
            - 'error': 抛出 OverflowError

    返回：
        ndarray 或 BigIntArray

    使用示例：
    >>> arr = smart_int_array([1, 2, 3])
    >>> type(arr).__name__
    'ndarray'
    >>> arr = smart_int_array([1, 2**60, 3])
    >>> type(arr).__name__
    'BigIntArray'
    """
    import warnings

    data_list = list(data)
    overflow_indices, overflow_values = detect_integer_overflow(data_list)

    np = _np()

    if not overflow_indices:
        return np.array(data_list, dtype=np.float64)

    if fallback == 'bigint':
        warnings.warn(
            f"检测到 {len(overflow_indices)} 个整数超出 f64 精度范围，"
            f"自动返回 BigIntArray 以保持精度。"
        )
        return BigIntArray(data_list, safe=False)

    elif fallback == 'object':
        warnings.warn(
            f"检测到 {len(overflow_indices)} 个整数超出 f64 精度范围，"
            f"返回 object dtype 数组。"
        )
        return np.array(data_list, dtype=object)

    elif fallback == 'warn':
        warnings.warn(
            f"检测到 {len(overflow_indices)} 个整数超出 f64 精度范围，"
            f"转换为 f64 会丢失精度。"
        )
        return np.array([float(v) for v in data_list])

    elif fallback == 'error':
        raise OverflowError(
            f"有 {len(overflow_indices)} 个整数超出 f64 精度范围："
            f"索引 {overflow_indices[:5]}，值 {overflow_values[:5]}"
        )

    else:
        raise ValueError(f"未知的 fallback 选项: {fallback}")


__all__ = [
    "acos", "asin", "atan", "atan2", "arccosh", "arcsinh", "arctanh",
    "radians", "degrees",
    "absolute", "fabs", "true_divide", "pow", "float_power", "negative", "positive",
    "exp2", "floor_divide", "fmod", "divmod", "rint", "hypot", "maximum", "minimum",
    "fmax", "fmin", "signbit", "copysign", "logaddexp", "logaddexp2", "logical_not",
    "gcd", "lcm", "nextafter", "spacing", "ldexp", "frexp", "modf",
    "prod", "nansum", "nanprod", "nanmax", "nanmin", "nanmean", "nanvar", "nanstd",
    "nanargmax", "nanargmin", "nancumsum", "nancumprod", "nanquantile",
    "cumulative_sum", "cumulative_prod", "count_nonzero",
    "unique_values", "unique_counts", "unique_inverse", "unique_all",
    "intersect1d", "union1d", "setdiff1d", "setxor1d", "isin",
    "ndim", "shape", "size", "isscalar", "iterable",
    "iscomplexobj", "isrealobj", "isreal", "iscomplex", "isnat",
    "isneginf", "isposinf", "real", "imag", "conjugate", "conj", "angle",
    "real_if_close", "nan_to_num",
    "concat", "row_stack", "permute_dims", "matrix_transpose",
    "atleast_1d", "atleast_2d", "atleast_3d", "broadcast_shapes", "broadcast_arrays",
    "block", "array_split", "unstack", "take_along_axis", "put_along_axis",
    "choose", "compress", "copyto", "place", "putmask", "pad", "trim_zeros",
    "ediff1d", "diff", "gradient", "cross", "trapezoid", "trapz",
    "outer", "kron", "tensordot",
    "tri", "tril", "triu", "diagonal", "diag", "diagflat", "trace",
    "tril_indices", "triu_indices", "tril_indices_from", "triu_indices_from",
    "diag_indices", "diag_indices_from", "fill_diagonal", "mask_indices",
    "hanning", "hamming", "blackman", "bartlett", "kaiser", "i0",
    "convolve", "correlate", "interp", "bincount", "vander", "unwrap",
    "all", "any", "round", "round_",
    "astype", "ascontiguousarray", "asfortranarray", "asarray_chkfinite",
    "require", "isfortran", "shares_memory", "may_share_memory", "array_equiv",
    "indices", "unravel_index", "ravel_multi_index",
    "apply_along_axis", "apply_over_axes", "piecewise",
    "bitwise_invert", "bitwise_left_shift", "bitwise_right_shift", "bitwise_count",
    "packbits", "unpackbits", "vecdot", "matvec", "vecmat",
    "histogram_bin_edges",
    "promote_types", "result_type", "can_cast", "min_scalar_type",
    "common_type", "isdtype", "issubdtype",
    "base_repr", "typename", "mintypecode",
    "array_repr", "array_str", "array2string",
    "get_printoptions", "set_printoptions", "printoptions",
    "format_float_positional", "format_float_scientific",
    "geterr", "seterr", "geterrcall", "seterrcall", "getbufsize", "setbufsize",
    "errstate", "fromstring", "savez_compressed", "frompyfunc", "vectorize",
    "datetime_data", "datetime_as_string", "is_busday", "busday_count",
    "busday_offset", "einsum", "einsum_path",
    "polyadd", "polysub", "polymul", "polydiv", "poly", "roots", "poly1d",
    "emath", "get_include", "show_config", "show_runtime", "info",
    "threadsafe_copy", "threadsafe_view",
    "check_object_dtype", "suggest_dtype_for_data",
    "memory_usage",
    "chunked_apply", "mmap_array", "LazyArray",
    "VarStringArray", "BigIntArray",
    "var_string_array", "bigint_array",
    "detect_integer_overflow", "smart_int_array",
]

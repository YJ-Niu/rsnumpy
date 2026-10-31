"""数学函数模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import builtins
import math

import rsnumpy.num_core as _core


def _nd():
    from .__init__ import ndarray as _n
    return _n


def _wrap(result):
    dt = "complex128" if getattr(result, 'is_complex', False) else "float64"
    return _nd()._wrap(result, _dtype=dt)


def _wrap_bool(result):
    """比较/逻辑运算结果按布尔数组返回（与 numpy 一致，可直接用作掩码索引）。"""
    return _nd()._wrap(result, _dtype="bool")


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
    if hasattr(a, '__class__') and a.__class__.__name__ == 'ndarray':
        return a
    return _core.ndarray(a)


# ========== 三角函数 ==========
def sin(x):
    """计算正弦值。"""
    return _wrap(_core.sin(_ensure_raw(x)))


def cos(x):
    """计算余弦值。"""
    return _wrap(_core.cos(_ensure_raw(x)))


def tan(x):
    """计算正切值。"""
    return _wrap(_core.tan(_ensure_raw(x)))


def arcsin(x):
    """计算反正弦值。"""
    return _wrap(_core.asin(_ensure_raw(x)))


def arccos(x):
    """计算反余弦值。"""
    return _wrap(_core.acos(_ensure_raw(x)))


def arctan(x):
    """计算反正切值。"""
    return _wrap(_core.atan(_ensure_raw(x)))


def arctan2(x1, x2):
    """计算 x1/x2 的反正切值。"""
    return _wrap(_core.arctan2(_ensure_raw(x1), _ensure_raw(x2)))


def deg2rad(x):
    """将角度转换为弧度。"""
    return _wrap(_core.deg2rad(_ensure_raw(x)))


def rad2deg(x):
    """将弧度转换为角度。"""
    return _wrap(_core.rad2deg(_ensure_raw(x)))


# ========== 双曲函数 ==========
def sinh(x):
    """计算双曲正弦值。"""
    return _wrap(_core.sinh(_ensure_raw(x)))


def cosh(x):
    """计算双曲余弦值。"""
    return _wrap(_core.cosh(_ensure_raw(x)))


def tanh(x):
    """计算双曲正切值。"""
    return _wrap(_core.tanh(_ensure_raw(x)))


def arcsinh(x):
    """计算反双曲正弦值。"""
    return _wrap(_core.asinh(_ensure_raw(x)))


def arccosh(x):
    """计算反双曲余弦值。"""
    return _wrap(_core.acosh(_ensure_raw(x)))


def arctanh(x):
    """计算反双曲正切值。"""
    return _wrap(_core.atanh(_ensure_raw(x)))


# ========== 指数和对数函数 ==========
def exp(x):
    """计算指数值 e^x。"""
    return _wrap(_core.exp(_ensure_raw(x)))


def expm1(x):
    """计算 exp(x) - 1。"""
    return _wrap(_core.expm1(_ensure_raw(x)))


def log(x):
    """计算自然对数。"""
    return _wrap(_core.log(_ensure_raw(x)))


def log10(x):
    """计算以 10 为底的对数。"""
    return _wrap(_core.log10(_ensure_raw(x)))


def log2(x):
    """计算以 2 为底的对数。"""
    return _wrap(_core.log2(_ensure_raw(x)))


def log1p(x):
    """计算 log(1 + x)。"""
    return _wrap(_core.log1p(_ensure_raw(x)))


# ========== 舍入函数 ==========
def around(a, decimals=0):
    """四舍六入五成双（numpy 语义：rint(x*10^d)/10^d 的 half-even）。

    与 Python round 的十进制正确舍入算法不同：6.35*10 恰好二进制舍入到
    63.5 → rint half-even → 64 → 6.4（numpy 行为）。
    """
    import builtins as _b
    arr = a if hasattr(a, '_array') else _nd()(a)
    _dt = str(getattr(arr, '_dtype', 'float64'))
    if _dt in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
               'uint32', 'uint64', 'bool') and decimals >= 0:
        # numpy 语义：整数数组 decimals>=0 时 around 为无操作
        return arr.copy()

    def _rnd(v):
        if decimals >= 0:
            _f = 10 ** decimals
            return _b.round(v * _f) / _f
        _f = 10 ** (-decimals)
        return _b.round(v / _f) * _f

    def _rec(x):
        out = []
        for v in x:
            if isinstance(v, list):
                out.append(_rec(v))
            else:
                out.append(_rnd(v))
        return out

    try:
        tl = arr.tolist()
        if not isinstance(tl, list):
            tl = [tl]
        res = _nd()(_rec(tl))
        if getattr(arr, 'ndim', 0) == 0:
            res = res.reshape(())
            return res.astype(_dt) if _dt != 'float64' else res
        return res.astype(_dt) if _dt != 'float64' else res
    except Exception:
        return arr.round(decimals)


def floor(x):
    """向下取整。"""
    return _wrap(_core.floor(_ensure_raw(x)))


def ceil(x):
    """向上取整。"""
    return _wrap(_core.ceil(_ensure_raw(x)))


def trunc(x):
    """截断小数部分。"""
    return _wrap(_core.trunc(_ensure_raw(x)))


def fix(x):
    """向零取整。"""
    return _wrap(_core.fix(_ensure_raw(x)))


# ========== 特殊函数 ==========
def sqrt(x):
    """计算平方根。"""
    return _wrap(_core.sqrt(_ensure_raw(x)))


def square(x):
    """计算平方。"""
    return _wrap(_core.square(_ensure_raw(x)))


def cbrt(x):
    """计算立方根。"""
    return _wrap(_core.cbrt(_ensure_raw(x)))


def absolute(x):
    """计算绝对值。"""
    import builtins
    if isinstance(x, complex):
        from .__init__ import ndarray
        return ndarray([builtins.abs(x)])
    return _wrap(_core.abs(_ensure_raw(x)))


abs = absolute


def sign(x):
    """计算符号函数。"""
    return _wrap(_core.sign(_ensure_raw(x)))


def clip(a, a_min, a_max):
    """限制数组值在指定范围内。"""
    return _wrap(_core.clip(_ensure_raw(a), a_min, a_max))


def sinc(x):
    """计算 sinc 函数。"""
    return _wrap(_core.sinc(_ensure_raw(x)))


def heaviside(x, h0=0.5):
    """计算 Heaviside 阶跃函数。"""
    return _wrap(_core.heaviside(_ensure_raw(x), h0))


# ========== 算术运算 ==========
def add(x1, x2, out=None, **kwargs):
    """逐元素加法。"""
    res = _wrap(_core.add(_ensure_raw(x1), _ensure_raw(x2)))
    if out is not None:
        try:
            out[...] = res
        except Exception:
            pass
    return res


def subtract(x1, x2, out=None, **kwargs):
    """逐元素减法。"""
    res = _wrap(_core.subtract(_ensure_raw(x1), _ensure_raw(x2)))
    if out is not None:
        try:
            out[...] = res
        except Exception:
            pass
    return res


def multiply(x1, x2, out=None, **kwargs):
    """逐元素乘法。"""
    res = _wrap(_core.multiply(_ensure_raw(x1), _ensure_raw(x2)))
    if out is not None:
        try:
            out[...] = res
        except Exception:
            pass
    return res


def divide(x1, x2, out=None, **kwargs):
    """逐元素除法。"""
    res = _wrap(_core.divide(_ensure_raw(x1), _ensure_raw(x2)))
    if out is not None:
        try:
            out[...] = res
        except Exception:
            pass
    return res


_INT_BITS = {
    'int8': (8, True), 'int16': (16, True), 'int32': (32, True), 'int64': (64, True),
    'uint8': (8, False), 'uint16': (16, False), 'uint32': (32, False), 'uint64': (64, False),
}

_FLOAT_NAMES = frozenset(('float16', 'float32', 'float64', 'float', 'half', 'single', 'double'))


def _dtype_name(dtype):
    """将 dtype 类型对象/字符串解析为名称。"""
    if isinstance(dtype, str):
        return dtype
    return getattr(dtype, '__name__', str(dtype))


def _is_arr(x):
    return hasattr(x, '_array') or (hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray')


def _wrap_int(val, bits, signed):
    """按位宽将整数按二补数回绕（模拟定宽整数溢出）。"""
    val &= (1 << bits) - 1
    if signed and (val >> (bits - 1)):
        val -= (1 << bits)
    return val


def power(x1, x2, dtype=None):
    """逐元素幂运算。

    指定 dtype 且 x1/x2 均为标量时，按目标类型语义计算：整数类型精确幂后
    按位宽二补数回绕（与 numpy 定宽整数溢出一致），浮点类型返回浮点标量。
    """
    if dtype is not None and not _is_arr(x1) and not _is_arr(x2):
        name = _dtype_name(dtype)
        if name in _INT_BITS:
            bits, signed = _INT_BITS[name]
            return _wrap_int(int(x1) ** int(x2), bits, signed)
        if name in _FLOAT_NAMES:
            return float(x1) ** float(x2)
    raw_result = _core.power(_ensure_raw(x1), _ensure_raw(x2))
    dtype = getattr(x1, '_dtype', 'float64') if hasattr(x1, '_dtype') else 'float64'
    return _nd()(raw_result, _dtype=dtype)


def mod(x1, x2):
    """逐元素取模。"""
    raw_result = _core.mod_(_ensure_raw(x1), _ensure_raw(x2))
    dtype = getattr(x1, '_dtype', 'float64') if hasattr(x1, '_dtype') else 'float64'
    return _nd()(raw_result, _dtype=dtype)


remainder = mod  # mod 的别名


# ========== 比较运算 ==========
def greater(x1, x2):
    """逐元素大于比较。"""
    return _wrap_bool(_core.greater(_ensure_raw(x1), _ensure_raw(x2)))


def less(x1, x2):
    """逐元素小于比较。"""
    return _wrap_bool(_core.less(_ensure_raw(x1), _ensure_raw(x2)))


def equal(x1, x2):
    """逐元素相等比较。"""
    return _wrap_bool(_core.equal(_ensure_raw(x1), _ensure_raw(x2)))


def not_equal(x1, x2):
    """逐元素不相等比较。"""
    return _wrap_bool(_core.not_equal(_ensure_raw(x1), _ensure_raw(x2)))


def greater_equal(x1, x2):
    """逐元素大于等于比较。"""
    return _wrap_bool(_core.greater_equal(_ensure_raw(x1), _ensure_raw(x2)))


def less_equal(x1, x2):
    """逐元素小于等于比较。"""
    return _wrap_bool(_core.less_equal(_ensure_raw(x1), _ensure_raw(x2)))


def logical_and(x1, x2, out=None, **kwargs):
    """逐元素逻辑与。"""
    res = _wrap_bool(_core.logical_and(_ensure_raw(x1), _ensure_raw(x2)))
    if out is not None:
        try:
            out[...] = res
        except Exception:
            pass
    return res


def logical_or(x1, x2, out=None, **kwargs):
    """逐元素逻辑或。"""
    res = _wrap_bool(_core.logical_or(_ensure_raw(x1), _ensure_raw(x2)))
    if out is not None:
        try:
            out[...] = res
        except Exception:
            pass
    return res


def logical_xor(x1, x2, out=None, **kwargs):
    """逐元素逻辑异或。"""
    res = _wrap_bool(_core.logical_xor(_ensure_raw(x1), _ensure_raw(x2)))
    if out is not None:
        try:
            out[...] = res
        except Exception:
            pass
    return res


def _isclose_arrays(arr_a, arr_b, rtol, atol, equal_nan):
    """isclose 的数值核心（numpy 语义，Python 逐元素实现）。

    底层 Rust 数组比较对非有限值有缺口（inf==inf → False、
    nan != nan → False），故展平后用 Python 原生浮点语义逐元素判断：
    两者皆有限 → |a-b| <= atol + rtol*|b|；非有限 → 仅 a == b 时 True；
    双 NaN → equal_nan。atol/rtol 支持 list/tuple 数组。
    形状不可广播 → ValueError（numpy 语义，避免直达 Rust panic）。
    """
    import builtins as _b
    import math as _m
    nd = _nd()

    def _num(x):
        if isinstance(x, (list, tuple)):
            return nd(list(x))
        return x

    def _flat(x):
        if not isinstance(x, (list, tuple)):
            # 0-d 数组 tolist() 返回标量
            return [x]
        o = []
        for v in x:
            if isinstance(v, list):
                o.extend(_flat(v))
            else:
                o.append(v)
        return o

    _rt, _at = _num(rtol), _num(atol)
    # 广播前置校验（形状不匹配 → ValueError）
    try:
        _sa = builtins.tuple(arr_a.shape)
    except Exception:
        _sa = ()
    for _x in (arr_b, _at, _rt):
        try:
            _sx = builtins.tuple(_x.shape)
        except Exception:
            _sx = ()
        for _da, _db in zip(builtins.reversed(_sa), builtins.reversed(_sx)):
            if _da != _db and _da != 1 and _db != 1:
                raise ValueError(
                    "operands could not be broadcast together with shapes "
                    f"{_sa} {_sx}")
    # 0-d 数组提升为 1-d：Rust 广播对 0-d 参与运算会静默降维
    # （0d - (3,) → 0-d），先 reshape 成 (1,) 让 (1,) 广播语义生效
    if getattr(arr_a, 'shape', None) == () and hasattr(arr_a, 'reshape'):
        arr_a = arr_a.reshape((1,))
    if getattr(arr_b, 'shape', None) == () and hasattr(arr_b, 'reshape'):
        arr_b = arr_b.reshape((1,))
    # 全 Python 广播 + 逐元素计算：Rust 广播算术对 (1,) vs (N,) 也会
    # 静默产出 size-1 结果，不可信
    def _shp(x):
        try:
            return builtins.tuple(x.shape)
        except Exception:
            return ()

    def _flat(x):
        if not isinstance(x, (list, tuple)):
            # 0-d 数组 tolist() 返回标量
            return [x]
        o = []
        for v in x:
            if isinstance(v, list):
                o.extend(_flat(v))
            else:
                o.append(v)
        return o

    _sa, _sb = _shp(arr_a), _shp(arr_b)
    _at_is_arr = hasattr(_at, 'shape')
    _rt_is_arr = hasattr(_rt, 'shape')
    _st = _shp(_at) if _at_is_arr else ()
    _sr = _shp(_rt) if _rt_is_arr else ()
    _ndim = builtins.max(len(_sa), len(_sb), len(_st), len(_sr))
    _out_shape = []
    for _d in range(_ndim):
        _dims = []
        for _s in (_sa, _sb, _st, _sr):
            _l = len(_s)
            _dims.append(_s[_l - _ndim + _d] if _d >= _ndim - _l else 1)
        _mx = builtins.max(_dims)
        if builtins.any(x != 1 and x != _mx for x in _dims):
            raise ValueError(
                "operands could not be broadcast together with shapes "
                f"{_sa} {_sb} {_st} {_sr}")
        _out_shape.append(_mx)
    _total = 1
    for _s in _out_shape:
        _total *= _s

    def _bcast(flat, shp):
        """把 flat 值按 shp 广播到 _out_shape，返回展平列表。"""
        shp = (1,) * (_ndim - len(shp)) + tuple(shp)
        _strides = [1] * _ndim
        for _i in range(_ndim - 2, -1, -1):
            _strides[_i] = _strides[_i + 1] * _out_shape[_i + 1]
        _ost = [0 if shp[_i] == 1 else _strides[_i] for _i in range(_ndim)]
        _res = []
        _idx = [0] * _ndim
        for _ in range(_total):
            _off = builtins.sum(_idx[_k] * _ost[_k] for _k in range(_ndim))
            _res.append(flat[_off])
            for _k in range(_ndim - 1, -1, -1):
                _idx[_k] += 1
                if _idx[_k] < _out_shape[_k]:
                    break
                _idx[_k] = 0
        return _res

    _bva = _bcast(_flat(arr_a.tolist()), _sa)
    _bvb = _bcast(_flat(arr_b.tolist()), _sb)
    _bat = _bcast(_flat(_at.tolist()), _st) if _at_is_arr else None
    _brt = _bcast(_flat(_rt.tolist()), _sr) if _rt_is_arr else None
    _out = []
    for _i in range(_total):
        _av, _bv = _bva[_i], _bvb[_i]
        _atv = _bat[_i] if _bat is not None else atol
        _rtv = _brt[_i] if _brt is not None else rtol
        if _av != _av and _bv != _bv:
            # 双 NaN
            _out.append(builtins.bool(equal_nan))
        elif _av == _bv:
            # 相等值（含 inf==inf / -inf==-inf）恒 close
            _out.append(True)
        elif _m.isfinite(_av) and _m.isfinite(_bv):
            _out.append(builtins.bool(abs(_av - _bv) <=
                                      _atv + _rtv * abs(_bv)))
        else:
            # 非有限对（inf/-inf 不相等）→ False
            _out.append(False)
    _shape = builtins.tuple(_out_shape)
    res = nd([1.0 if v else 0.0 for v in _out])
    if _shape:
        res = res.reshape(_shape)
    res._dtype = 'bool'
    return res


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """逐元素判断是否接近（numpy 语义，Python 层实现）。

    |a-b| <= atol + rtol*|b|；相等值（含 inf==inf）恒 close；
    NaN 默认 False，equal_nan=True 时同位 NaN 为 True。
    标量输入返回 np.bool 标量单例（np.True_/np.False_）。
    """
    import math as _m
    try:
        if (not _m.isfinite(rtol) or rtol < 0 or
                not _m.isfinite(atol) or atol < 0):
            import warnings as _warn_is
            _warn_is.warn(
                f"One of rtol or atol is not valid, atol: {atol}, "
                f"rtol: {rtol}", RuntimeWarning, stacklevel=3)
    except (TypeError, ValueError):
        pass
    nd = _nd()
    _a_arr = hasattr(a, '_array')
    _b_arr = hasattr(b, '_array')
    if not _a_arr and not _b_arr and \
            not isinstance(a, (list, tuple)) and \
            not isinstance(b, (list, tuple)):
        # 标量 × 标量：纯 Python 计算（绕开 0-d 数组 item/bool 缺口），
        # 返回 np.bool 标量单例（numpy 语义）
        try:
            _va, _vb = a, b
            if isinstance(_va, builtins.bool) or isinstance(_vb, builtins.bool):
                _va, _vb = builtins.bool(_va), builtins.bool(_vb)
            else:
                _va, _vb = builtins.complex(_va), builtins.complex(_vb)
            if _va != _va and _vb != _vb:
                _val = builtins.bool(equal_nan)
            elif _va == _vb:
                # 相等值（含 inf==inf / -inf==-inf）恒 close
                _val = True
            else:
                # numpy 语义：非有限对（含 inf/-inf）仅在相等时 close
                _va_f = builtins.bool(_va == _va and
                                      abs(_va) != builtins.float('inf'))
                _vb_f = builtins.bool(_vb == _vb and
                                      abs(_vb) != builtins.float('inf'))
                if _va_f and _vb_f:
                    _val = builtins.bool(abs(_va - _vb) <=
                                         atol + rtol * abs(_vb))
                else:
                    _val = False
        except Exception:
            _val = False
        try:
            from .__init__ import True_ as _t_one, False_ as _f_one
        except Exception:
            return _val
        return _t_one if _val else _f_one
    arr_a = a if _a_arr else nd(a)
    arr_b = b if _b_arr else nd(b)
    # NEP50 弱类型：f32/f16 数组与 Python float（type 精确匹配；float64/
    # float32 标量为强类型不降级）比较 → float 降到数组精度
    _dt_a = getattr(arr_a, '_dtype', '')
    if _dt_a in ('float32', 'float16'):
        try:
            from .__init__ import _round_scalar_to_f
            if not _b_arr and type(b) is float:
                b = _round_scalar_to_f(_dt_a, b)
                arr_b = nd(b)
            if type(atol) is float:
                atol = _round_scalar_to_f(_dt_a, atol)
            if type(rtol) is float:
                rtol = _round_scalar_to_f(_dt_a, rtol)
        except Exception:
            pass
    res = _isclose_arrays(arr_a, arr_b, rtol, atol, equal_nan)
    return res


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False, *args, **kwargs):
    """判断两个数组是否全部元素级接近（numpy 语义）。

    与 NumPy 对齐：``equal_nan=True`` 时同位置的 NaN 视为相等。
    标量输入返回 Python bool。
    """
    _ = args, kwargs
    res = isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    try:
        if hasattr(res, 'all'):
            return builtins.bool(res.all())
        return builtins.bool(res)
    except Exception:
        return False


# 反双曲函数别名
asinh = arcsinh
acosh = arccosh
atanh = arctanh


# 其他数学函数
def reciprocal(x):
    """逐元素返回倒数（1/x）。"""
    return _wrap(_core.reciprocal(_ensure_raw(x)))

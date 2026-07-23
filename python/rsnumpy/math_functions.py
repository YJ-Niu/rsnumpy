"""数学函数模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

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
    """四舍五入到指定小数位（使用 Rust round）。"""
    arr = a if hasattr(a, '_array') else _nd()(a)
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
    if hasattr(x, '_array'):
        x_np = _core.array(x)
    else:
        x_np = _core.array(x)
    result_np = _core.sqrt(x_np)
    return _nd()(result_np.tolist())


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
def add(x1, x2):
    """逐元素加法。"""
    return _wrap(_core.add(_ensure_raw(x1), _ensure_raw(x2)))


def subtract(x1, x2):
    """逐元素减法。"""
    return _wrap(_core.subtract(_ensure_raw(x1), _ensure_raw(x2)))


def multiply(x1, x2):
    """逐元素乘法。"""
    return _wrap(_core.multiply(_ensure_raw(x1), _ensure_raw(x2)))


def divide(x1, x2):
    """逐元素除法。"""
    return _wrap(_core.divide(_ensure_raw(x1), _ensure_raw(x2)))


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


def logical_and(x1, x2):
    """逐元素逻辑与。"""
    return _wrap_bool(_core.logical_and(_ensure_raw(x1), _ensure_raw(x2)))


def logical_or(x1, x2):
    """逐元素逻辑或。"""
    return _wrap_bool(_core.logical_or(_ensure_raw(x1), _ensure_raw(x2)))


def logical_xor(x1, x2):
    """逐元素逻辑异或。"""
    return _wrap_bool(_core.logical_xor(_ensure_raw(x1), _ensure_raw(x2)))


def isclose(a, b, rtol=1e-05, atol=1e-08):
    """逐元素判断是否接近。"""
    return _wrap_bool(_core.isclose(_ensure_raw(a), _ensure_raw(b), rtol, atol))


def allclose(a, b, rtol=1e-05, atol=1e-08):
    """判断两个数组是否元素级接近。"""
    return _core.allclose(_ensure_raw(a), _ensure_raw(b), rtol, atol)


# 反双曲函数别名
asinh = arcsinh
acosh = arccosh
atanh = arctanh


# 其他数学函数
def reciprocal(x):
    """逐元素返回倒数（1/x）。"""
    return _wrap(_core.reciprocal(_ensure_raw(x)))

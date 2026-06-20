"""数学函数模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnumpy._core as _core


def _nd():
    from .__init__ import ndarray as _n
    return _n


def _wrap(result):
    return _nd()(result)


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
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
    return _wrap(_core.sqrt(_ensure_raw(x)))


def square(x):
    """计算平方。"""
    return _wrap(_core.square(_ensure_raw(x)))


def cbrt(x):
    """计算立方根。"""
    return _wrap(_core.cbrt(_ensure_raw(x)))


def absolute(x):
    """计算绝对值。"""
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


def power(x1, x2):
    """逐元素幂运算。"""
    return _wrap(_core.power(_ensure_raw(x1), _ensure_raw(x2)))


def mod(x1, x2):
    """逐元素取模。"""
    return _wrap(_core.mod_(_ensure_raw(x1), _ensure_raw(x2)))


# ========== 比较运算 ==========
def greater(x1, x2):
    """逐元素大于比较。"""
    return _wrap(_core.greater(_ensure_raw(x1), _ensure_raw(x2)))


def less(x1, x2):
    """逐元素小于比较。"""
    return _wrap(_core.less(_ensure_raw(x1), _ensure_raw(x2)))


def equal(x1, x2):
    """逐元素相等比较。"""
    return _wrap(_core.equal(_ensure_raw(x1), _ensure_raw(x2)))


def not_equal(x1, x2):
    """逐元素不相等比较。"""
    return _wrap(_core.not_equal(_ensure_raw(x1), _ensure_raw(x2)))


def greater_equal(x1, x2):
    """逐元素大于等于比较。"""
    return _wrap(_core.greater_equal(_ensure_raw(x1), _ensure_raw(x2)))


def less_equal(x1, x2):
    """逐元素小于等于比较。"""
    return _wrap(_core.less_equal(_ensure_raw(x1), _ensure_raw(x2)))


def logical_and(x1, x2):
    """逐元素逻辑与。"""
    return _wrap(_core.logical_and(_ensure_raw(x1), _ensure_raw(x2)))


def logical_or(x1, x2):
    """逐元素逻辑或。"""
    return _wrap(_core.logical_or(_ensure_raw(x1), _ensure_raw(x2)))


def logical_xor(x1, x2):
    """逐元素逻辑异或。"""
    return _wrap(_core.logical_xor(_ensure_raw(x1), _ensure_raw(x2)))


def isclose(a, b, rtol=1e-05, atol=1e-08):
    """逐元素判断是否接近。"""
    return _wrap(_core.isclose(_ensure_raw(a), _ensure_raw(b), rtol, atol))


def allclose(a, b, rtol=1e-05, atol=1e-08):
    """判断两个数组是否元素级接近。"""
    return _core.allclose(_ensure_raw(a), _ensure_raw(b), rtol, atol)


# 反双曲函数别名
asinh = arcsinh
acosh = arccosh
atanh = arctanh

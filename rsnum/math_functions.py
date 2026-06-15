"""
数学函数模块

此模块包含三角函数、指数对数、舍入等逐元素数学函数。
"""

import math

def _apply_math_func(func, x):
    """
    对数组逐元素应用数学函数。
    
    参数:
        func: 要应用的数学函数。
        x: 输入数组。
    
    返回:
        ndarray: 结果数组。
    """
    from .__init__ import ndarray
    arr = ndarray(x)
    data = arr.tolist()
    
    def apply(data):
        if isinstance(data, list):
            return [apply(item) for item in data]
        return func(data)
    
    result_data = apply(data)
    return ndarray(result_data)


# ========== 三角函数 ==========

def sin(x):
    """
    计算正弦值。
    
    参数:
        x: 输入数组（弧度）。
    
    返回:
        ndarray: 正弦值数组。
    """
    return _apply_math_func(math.sin, x)


def cos(x):
    """
    计算余弦值。
    
    参数:
        x: 输入数组（弧度）。
    
    返回:
        ndarray: 余弦值数组。
    """
    return _apply_math_func(math.cos, x)


def tan(x):
    """
    计算正切值。
    
    参数:
        x: 输入数组（弧度）。
    
    返回:
        ndarray: 正切值数组。
    """
    return _apply_math_func(math.tan, x)


def arcsin(x):
    """
    计算反正弦值。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 反正弦值数组（弧度）。
    """
    return _apply_math_func(math.asin, x)


def arccos(x):
    """
    计算反余弦值。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 反余弦值数组（弧度）。
    """
    return _apply_math_func(math.acos, x)


def arctan(x):
    """
    计算反正切值。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 反正切值数组（弧度）。
    """
    return _apply_math_func(math.atan, x)


def arctan2(y, x):
    """
    计算 y/x 的反正切值。
    
    参数:
        y: 分子数组。
        x: 分母数组。
    
    返回:
        ndarray: 反正切值数组（弧度）。
    """
    from .__init__ import ndarray
    y_arr = ndarray(y)
    x_arr = ndarray(x)
    y_data = y_arr.tolist()
    x_data = x_arr.tolist()
    
    def apply(y_data, x_data):
        if isinstance(y_data, list) and isinstance(x_data, list):
            return [apply(yd, xd) for yd, xd in zip(y_data, x_data)]
        return math.atan2(y_data, x_data)
    
    result_data = apply(y_data, x_data)
    return ndarray(result_data)


def deg2rad(x):
    """
    将角度转换为弧度。
    
    参数:
        x: 输入数组（角度）。
    
    返回:
        ndarray: 弧度数组。
    """
    return _apply_math_func(lambda v: math.radians(v), x)


def rad2deg(x):
    """
    将弧度转换为角度。
    
    参数:
        x: 输入数组（弧度）。
    
    返回:
        ndarray: 角度数组。
    """
    return _apply_math_func(lambda v: math.degrees(v), x)


# ========== 双曲函数 ==========

def sinh(x):
    """
    计算双曲正弦值。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 双曲正弦值数组。
    """
    return _apply_math_func(math.sinh, x)


def cosh(x):
    """
    计算双曲余弦值。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 双曲余弦值数组。
    """
    return _apply_math_func(math.cosh, x)


def tanh(x):
    """
    计算双曲正切值。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 双曲正切值数组。
    """
    return _apply_math_func(math.tanh, x)


def asinh(x):
    """
    计算反双曲正弦值。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 反双曲正弦值数组。
    """
    return _apply_math_func(math.asinh, x)


def acosh(x):
    """
    计算反双曲余弦值。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 反双曲余弦值数组。
    """
    return _apply_math_func(math.acosh, x)


def atanh(x):
    """
    计算反双曲正切值。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 反双曲正切值数组。
    """
    return _apply_math_func(math.atanh, x)


# ========== 指数和对数函数 ==========

def exp(x):
    """
    计算指数值 e^x。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 指数值数组。
    """
    return _apply_math_func(math.exp, x)


def expm1(x):
    """
    计算 exp(x) - 1。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: exp(x) - 1 的结果数组。
    """
    return _apply_math_func(math.expm1, x)


def log(x):
    """
    计算自然对数。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 自然对数值数组。
    """
    return _apply_math_func(math.log, x)


def log10(x):
    """
    计算以10为底的对数。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 以10为底的对数值数组。
    """
    return _apply_math_func(math.log10, x)


def log2(x):
    """
    计算以2为底的对数。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 以2为底的对数值数组。
    """
    return _apply_math_func(math.log2, x)


def log1p(x):
    """
    计算 log(1 + x)。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: log(1 + x) 的结果数组。
    """
    return _apply_math_func(math.log1p, x)


# ========== 舍入函数 ==========

def around(a, decimals=0):
    """
    四舍五入到指定小数位数。
    
    参数:
        a: 输入数组。
        decimals: 小数位数。
    
    返回:
        ndarray: 四舍五入后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.round(decimals)


def floor(x):
    """
    向下取整。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 向下取整后的数组。
    """
    return _apply_math_func(math.floor, x)


def ceil(x):
    """
    向上取整。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 向上取整后的数组。
    """
    return _apply_math_func(math.ceil, x)


def trunc(x):
    """
    截断小数部分。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 截断后的数组。
    """
    return _apply_math_func(math.trunc, x)


def fix(x):
    """
    向零取整。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 向零取整后的数组。
    """
    return trunc(x)


# ========== 特殊函数 ==========

def sqrt(x):
    """
    计算平方根。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 平方根数组。
    """
    return _apply_math_func(math.sqrt, x)


def square(x):
    """
    计算平方。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 平方数组。
    """
    from .__init__ import ndarray
    arr = ndarray(x)
    return arr * arr


def cbrt(x):
    """
    计算立方根。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 立方根数组。
    """
    return _apply_math_func(lambda v: v ** (1.0 / 3.0), x)


def abs(x):
    """
    计算绝对值。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 绝对值数组。
    """
    return _apply_math_func(math.fabs, x)


def sign(x):
    """
    计算符号函数。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: 符号数组（-1, 0, 或 1）。
    """
    from .__init__ import ndarray
    arr = ndarray(x)
    return (arr > 0).astype(float) - (arr < 0).astype(float)


def clip(a, a_min, a_max):
    """
    限制数组值在指定范围内。
    
    参数:
        a: 输入数组。
        a_min: 最小值。
        a_max: 最大值。
    
    返回:
        ndarray: 裁剪后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.where(arr < a_min, a_min).where(arr > a_max, a_max)


def sinc(x):
    """
    计算 sinc 函数。
    
    参数:
        x: 输入数组。
    
    返回:
        ndarray: sinc 值数组。
    """
    from .__init__ import ndarray
    arr = ndarray(x)
    sin_arr = sin(arr)
    return sin_arr.where(arr != 0, sin_arr, arr.where(arr != 0, arr, 1.0))


def heaviside(x, y):
    """
    计算 Heaviside 阶跃函数。
    
    参数:
        x: 输入数组。
        y: x=0 时的值。
    
    返回:
        ndarray: Heaviside 函数值数组。
    """
    from .__init__ import ndarray
    x_arr = ndarray(x)
    return (x_arr > 0).astype(float) + (x_arr == 0).astype(float) * y


# ========== 算术运算 ==========

def add(x1, x2):
    """
    逐元素加法。
    
    参数:
        x1: 第一个数组。
        x2: 第二个数组。
    
    返回:
        ndarray: 加法结果数组。
    """
    from .__init__ import ndarray
    return ndarray(x1) + ndarray(x2)


def subtract(x1, x2):
    """
    逐元素减法。
    
    参数:
        x1: 被减数数组。
        x2: 减数数组。
    
    返回:
        ndarray: 减法结果数组。
    """
    from .__init__ import ndarray
    return ndarray(x1) - ndarray(x2)


def multiply(x1, x2):
    """
    逐元素乘法。
    
    参数:
        x1: 第一个数组。
        x2: 第二个数组。
    
    返回:
        ndarray: 乘法结果数组。
    """
    from .__init__ import ndarray
    return ndarray(x1) * ndarray(x2)


def divide(x1, x2):
    """
    逐元素除法。
    
    参数:
        x1: 被除数数组。
        x2: 除数数组。
    
    返回:
        ndarray: 除法结果数组。
    """
    from .__init__ import ndarray
    return ndarray(x1) / ndarray(x2)


def power(x1, x2):
    """
    逐元素幂运算。
    
    参数:
        x1: 底数数组。
        x2: 指数数组。
    
    返回:
        ndarray: 幂运算结果数组。
    """
    from .__init__ import ndarray
    return ndarray(x1) ** ndarray(x2)


def mod(x1, x2):
    """
    逐元素取模。
    
    参数:
        x1: 被除数数组。
        x2: 除数数组。
    
    返回:
        ndarray: 取模结果数组。
    """
    from .__init__ import ndarray
    return ndarray(x1) % ndarray(x2)


# ========== 比较运算 ==========

def greater(x1, x2):
    """
    逐元素大于比较。
    
    参数:
        x1: 第一个数组。
        x2: 第二个数组。
    
    返回:
        ndarray: 布尔数组。
    """
    from .__init__ import ndarray
    return ndarray(x1) > ndarray(x2)


def less(x1, x2):
    """
    逐元素小于比较。
    
    参数:
        x1: 第一个数组。
        x2: 第二个数组。
    
    返回:
        ndarray: 布尔数组。
    """
    from .__init__ import ndarray
    return ndarray(x1) < ndarray(x2)


def equal(x1, x2):
    """
    逐元素相等比较。
    
    参数:
        x1: 第一个数组。
        x2: 第二个数组。
    
    返回:
        ndarray: 布尔数组。
    """
    from .__init__ import ndarray
    return ndarray(x1) == ndarray(x2)


def logical_and(x1, x2):
    """
    逐元素逻辑与。
    
    参数:
        x1: 第一个数组。
        x2: 第二个数组。
    
    返回:
        ndarray: 布尔数组。
    """
    from .__init__ import ndarray
    arr1 = ndarray(x1).astype(bool)
    arr2 = ndarray(x2).astype(bool)
    return arr1 * arr2


def logical_or(x1, x2):
    """
    逐元素逻辑或。
    
    参数:
        x1: 第一个数组。
        x2: 第二个数组。
    
    返回:
        ndarray: 布尔数组。
    """
    from .__init__ import ndarray
    arr1 = ndarray(x1).astype(bool)
    arr2 = ndarray(x2).astype(bool)
    return (arr1 + arr2) > 0


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    逐元素判断是否接近。
    
    参数:
        a: 第一个数组。
        b: 第二个数组。
        rtol: 相对容差。
        atol: 绝对容差。
        equal_nan: 是否将 NaN 视为相等。
    
    返回:
        ndarray: 布尔数组。
    """
    from .__init__ import ndarray
    a_arr = ndarray(a)
    b_arr = ndarray(b)
    diff = (a_arr - b_arr).abs()
    return diff <= (atol + rtol * b_arr.abs())


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    判断两个数组是否元素级接近。
    
    参数:
        a: 第一个数组。
        b: 第二个数组。
        rtol: 相对容差。
        atol: 绝对容差。
        equal_nan: 是否将 NaN 视为相等。
    
    返回:
        bool: 是否所有元素都接近。
    """
    return isclose(a, b, rtol, atol, equal_nan).all()
"""rsnumpy.fft - 快速傅里叶变换模块。

与 numpy.fft 对齐：既支持 `import rsnumpy as np; np.fft.fftshift(...)`，
也支持 `from rsnumpy.fft import fft, ifft, rfft, irfft, fftshift, ifftshift`。

底层一维变换由 Rust (`rsnumpy._core`) 实现，复数结果以形状 ``[n, 2]``
的实数数组表示（每行为 ``[实部, 虚部]``）。fftshift/ifftshift/fftfreq/rfftfreq
则参考 numpy 的实现用纯 Python 组合现有原语完成。
"""

import math as _math

import rsnumpy.num_core as _core

__all__ = [
    'fft', 'ifft', 'rfft', 'irfft',
    'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq',
]


def _np():
    """惰性获取父包，避免导入期循环依赖。"""
    import rsnumpy as _rsnp
    return _rsnp


def _raw(a):
    """取得底层 `_core.ndarray`。"""
    if hasattr(a, '_array'):
        return a._array
    return _core.ndarray(a)


def _scale(result, factor):
    """对（可能是复数 [n, 2] 表示的）结果整体乘以标量。"""
    if factor == 1.0:
        return result
    return result * factor


def _norm_forward_factor(n, norm):
    """正变换（fft/rfft）在给定 norm 下相对 numpy 默认(无缩放)的系数。"""
    if not n:
        return 1.0
    if norm == 'ortho':
        return 1.0 / _math.sqrt(n)
    if norm == 'forward':
        return 1.0 / n
    return 1.0


def _norm_inverse_factor(n, norm):
    """逆变换（ifft/irfft）相对 `_core` 默认(已含 1/n)的补偿系数。"""
    if not n:
        return 1.0
    if norm == 'ortho':
        return _math.sqrt(n)
    if norm == 'forward':
        return float(n)
    return 1.0


def fft(a, n=None, axis=-1, norm=None):
    """计算一维离散傅里叶变换。"""
    _ = (n, axis)
    nd = _np().ndarray
    arr = a if hasattr(a, '_array') else nd(a)
    result = nd._wrap(_core.py_fft_ndarray(arr._array))
    return _scale(result, _norm_forward_factor(arr.size, norm))


def ifft(a, n=None, axis=-1, norm=None):
    """计算一维逆离散傅里叶变换。"""
    _ = (n, axis)
    nd = _np().ndarray
    arr = a if hasattr(a, '_array') else nd(a)
    result = nd._wrap(_core.py_ifft_ndarray(arr._array))
    return _scale(result, _norm_inverse_factor(result.shape[0], norm))


def rfft(a, n=None, axis=-1, norm=None):
    """计算实输入的一维离散傅里叶变换。"""
    _ = (n, axis)
    nd = _np().ndarray
    arr = a if hasattr(a, '_array') else nd(a)
    result = nd._wrap(_core.py_rfft_ndarray(arr._array))
    return _scale(result, _norm_forward_factor(arr.size, norm))


def irfft(a, n=None, axis=-1, norm=None):
    """计算 rfft 的逆变换（返回实数数组）。"""
    _ = axis
    nd = _np().ndarray
    arr = a if hasattr(a, '_array') else nd(a)
    result = nd._wrap(_core.py_irfft_ndarray(arr._array, n))
    return _scale(result, _norm_inverse_factor(result.shape[0], norm))


def _shift(x, axes, inverse):
    np = _np()
    x = np.asarray(x)
    ndim = len(x.shape)
    if axes is None:
        axis_list = range(ndim)
    elif isinstance(axes, int):
        axis_list = (axes,)
    else:
        axis_list = axes
    for ax in axis_list:
        n = x.shape[ax]
        half = n // 2
        shift = -half if inverse else half
        x = np.roll(x, shift, axis=ax)
    return x


def fftshift(x, axes=None):
    """将零频分量移到频谱中心。"""
    return _shift(x, axes, inverse=False)


def ifftshift(x, axes=None):
    """fftshift 的逆操作。"""
    return _shift(x, axes, inverse=True)


def fftfreq(n, d=1.0):
    """返回离散傅里叶变换的采样频率。"""
    np = _np()
    val = 1.0 / (n * d)
    half = (n - 1) // 2 + 1
    p1 = np.arange(0, half)
    p2 = np.arange(-(n // 2), 0)
    return np.concatenate([p1, p2]) * val


def rfftfreq(n, d=1.0):
    """返回 rfft 的采样频率（仅非负频率）。"""
    np = _np()
    val = 1.0 / (n * d)
    half = n // 2 + 1
    return np.arange(0, half) * val

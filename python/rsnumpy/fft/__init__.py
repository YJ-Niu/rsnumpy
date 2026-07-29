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


def _fft_1d(arr_core, n, kind):
    """对一维核心数组执行 FFT。

    kind: 'fft', 'ifft', 'rfft', 'irfft'
    """
    if kind == 'fft':
        return _core.py_fft_ndarray(arr_core)
    elif kind == 'ifft':
        return _core.py_ifft_ndarray(arr_core)
    elif kind == 'rfft':
        return _core.py_rfft_ndarray(arr_core)
    elif kind == 'irfft':
        return _core.py_irfft_ndarray(arr_core, n)
    raise ValueError(f"未知 FFT 类型: {kind}")


def _fft_nd(arr, n, axis, norm, kind, is_inverse):
    """沿指定轴对多维数组执行 FFT。

    kind: 'fft', 'ifft', 'rfft', 'irfft'
    is_inverse: True 表示逆变换（使用 _norm_inverse_factor）
    """
    np = _np()
    nd = np.ndarray
    arr = arr if hasattr(arr, '_array') else nd(arr)

    # 一维数组直接处理
    if arr.ndim <= 1:
        result = nd._wrap(_fft_1d(arr._array, n, kind))
        if is_inverse:
            return _scale(result, _norm_inverse_factor(result.shape[0], norm))
        else:
            return _scale(result, _norm_forward_factor(arr.size, norm))

    # 多维数组：沿 axis 处理
    # 标准化 axis
    ax = axis % arr.ndim if axis is not None else -1
    ax = ax if ax >= 0 else arr.ndim + ax

    # 将 axis 交换到最后一维
    if ax != arr.ndim - 1:
        arr_swapped = np.swapaxes(arr, ax, arr.ndim - 1)
    else:
        arr_swapped = arr

    # 展平前面的维度
    original_shape = arr_swapped.shape
    last_dim = original_shape[-1]
    arr_2d = arr_swapped.reshape(-1, last_dim)

    # 对每行做 1D FFT，使用列表推导代替显式 for 循环
    rows = [nd._wrap(_fft_1d(arr_2d[i]._array, n, kind)) for i in range(arr_2d.shape[0])]

    # 合并结果
    if not rows:
        result_2d = np.array([])
    else:
        result_2d = np.stack(rows)

    # 计算新形状
    new_last_dim = result_2d.shape[1] if result_2d.ndim >= 2 else result_2d.shape[0]
    new_shape = original_shape[:-1] + (new_last_dim,)
    result = result_2d.reshape(new_shape)

    # 将 axis 交换回原位
    if ax != arr.ndim - 1:
        result = np.swapaxes(result, arr.ndim - 1, ax)

    # 应用归一化
    transform_len = result.shape[ax]
    if is_inverse:
        return _scale(result, _norm_inverse_factor(transform_len, norm))
    else:
        return _scale(result, _norm_forward_factor(arr.size // last_dim * transform_len if transform_len else arr.size, norm))


def fft(a, n=None, axis=-1, norm=None):
    """计算一维离散傅里叶变换。"""
    return _fft_nd(a, n, axis, norm, 'fft', is_inverse=False)


def ifft(a, n=None, axis=-1, norm=None):
    """计算一维逆离散傅里叶变换。"""
    return _fft_nd(a, n, axis, norm, 'ifft', is_inverse=True)


def rfft(a, n=None, axis=-1, norm=None):
    """计算实输入的一维离散傅里叶变换。"""
    return _fft_nd(a, n, axis, norm, 'rfft', is_inverse=False)


def irfft(a, n=None, axis=-1, norm=None):
    """计算 rfft 的逆变换（返回实数数组）。"""
    return _fft_nd(a, n, axis, norm, 'irfft', is_inverse=True)


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

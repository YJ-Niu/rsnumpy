"""
ndarray 对象方法模块

此模块包含 rsnumpy.ndarray 类的所有对象方法和属性。
"""

import rsnumpy._core as _core
from .__init__ import ndarray, _ensure


def _wrap_result(result, dtype="float64"):
    """包装结果为 ndarray 对象"""
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return ndarray._wrap(result, _dtype=dtype)
    return result


class NdArrayMethods:
    """ndarray 对象方法集合"""
    
    @staticmethod
    def astype(arr, dtype):
        """将数组转换为指定的数据类型。"""
        name = dtype if isinstance(dtype, str) else getattr(dtype, "__name__", str(dtype))
        return _wrap_result(arr._array.astype(name), dtype=name)
    
    @staticmethod
    def reshape(arr, *shape):
        """改变数组形状而不改变数据。"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        else:
            shape = shape
        try:
            result = arr._array.reshape(shape)
        except ValueError:
            result = arr._array.flatten().reshape(shape)
        return ndarray._wrap(result, _dtype=getattr(arr, '_dtype', "float64"), _fields=getattr(arr, '_fields', None), _raw_data=getattr(arr, '_raw_data', None))
    
    @staticmethod
    def resize(arr, new_shape):
        """改变数组形状和大小。"""
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        elif isinstance(new_shape, list):
            new_shape = tuple(new_shape)
        arr._array.resize(new_shape)
    
    @staticmethod
    def ravel(arr, order='C'):
        """将数组展平为一维。"""
        if order == 'A' or order == 'K':
            order = 'C'
        return NdArrayMethods.flatten(arr, order)
    
    @staticmethod
    def flatten(arr, order='C'):
        """展平数组为一维。"""
        dtype = getattr(arr, '_dtype', "float64")
        result = _core.flatten_full(arr._array, order)
        return _wrap_result(result, dtype)
    
    @staticmethod
    def transpose(arr, *axes):
        """转置数组。"""
        dtype = getattr(arr, '_dtype', "float64")
        fields = getattr(arr, '_fields', None)
        raw = getattr(arr, '_raw_data', None)
        if len(axes) == 0:
            result = arr._array.T
        else:
            result = _core.transpose(arr._array)
        return ndarray._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw)
    
    @staticmethod
    def swapaxes(arr, axis1, axis2):
        """交换两个轴的位置。"""
        dtype = getattr(arr, '_dtype', "float64")
        fields = getattr(arr, '_fields', None)
        raw_data = getattr(arr, '_raw_data', None)
        result = _core.swapaxes(arr._array, axis1, axis2)
        return ndarray._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)
    
    @staticmethod
    def squeeze(arr, axis=None):
        """移除长度为1的轴。"""
        return _wrap_result(arr._array.squeeze())
    
    @staticmethod
    def max(arr, axis=None):
        """返回数组的最大值。"""
        result = arr._array.max(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def min(arr, axis=None):
        """返回数组的最小值。"""
        result = arr._array.min(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def mean(arr, axis=None):
        """计算数组的平均值。"""
        result = arr._array.mean(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def std(arr, axis=None):
        """计算数组的标准差。"""
        result = arr._array.std(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def var(arr, axis=None):
        """计算数组的方差。"""
        result = arr._array.var(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def sum(arr, axis=None):
        """计算数组元素的和。"""
        result = arr._array.sum(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def prod(arr, axis=None):
        """计算数组元素的乘积。"""
        result = arr._array.prod(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def cumsum(arr, axis=None):
        """计算累积和。"""
        result = arr._array.cumsum(axis)
        return _wrap_result(result)
    
    @staticmethod
    def cumprod(arr, axis=None):
        """计算累积乘积。"""
        result = arr._array.cumprod(axis)
        return _wrap_result(result)
    
    @staticmethod
    def argmax(arr, axis=None):
        """返回最大值的索引。"""
        result = _wrap_result(_core.argmax_axis(arr._array, axis))
        if result.ndim == 0:
            return int(result.item())
        return result
    
    @staticmethod
    def argmin(arr, axis=None):
        """返回最小值的索引。"""
        result = _wrap_result(_core.argmin_axis(arr._array, axis))
        if result.ndim == 0:
            return int(result.item())
        return result
    
    @staticmethod
    def argsort(arr, axis=-1):
        """返回排序后的索引。"""
        result = arr._array.argsort(axis)
        return _wrap_result(result)
    
    @staticmethod
    def sort(arr, axis=-1):
        """原地排序数组。"""
        result = arr._array.sort(axis)
        if hasattr(result, 'shape') and result.shape == arr.shape:
            arr._array = result
        else:
            arr._array.data = result.data
    
    @staticmethod
    def diagonal(arr, offset=0, axis1=0, axis2=1):
        """返回数组的对角线元素。"""
        result = arr._array.diagonal(offset, axis1, axis2)
        return _wrap_result(result)
    
    @staticmethod
    def trace(arr, offset=0, axis1=0, axis2=1):
        """计算数组的迹（对角线元素之和）。"""
        return arr._array.trace(offset, axis1, axis2)
    
    @staticmethod
    def fill(arr, value):
        """用指定值填充数组。"""
        arr._array.fill(value)
    
    @staticmethod
    def item(arr, *args):
        """获取数组的单个元素。"""
        return arr._array.item(*args)
    
    @staticmethod
    def tolist(arr):
        """将数组转换为 Python 列表。"""
        return arr._array.tolist()
    
    @staticmethod
    def take(arr, indices, axis=None):
        """根据索引获取元素。"""
        result = arr._array.take(_ensure(indices), axis)
        return _wrap_result(result)
    
    @staticmethod
    def put(arr, indices, values):
        """将值放入数组的指定位置。"""
        arr._array.put(_ensure(indices), _ensure(values))
    
    @staticmethod
    def repeat(arr, repeats, axis=None):
        """重复数组元素。"""
        result = arr._array.repeat(repeats, axis)
        return _wrap_result(result)
    
    @staticmethod
    def nonzero(arr):
        """返回非零元素的索引。"""
        raw = _core.nonzero_arrs(arr._array)
        return tuple(ndarray._wrap(r) for r in raw)

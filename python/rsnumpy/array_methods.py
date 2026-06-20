"""
ndarray 对象方法模块

此模块包含 rsnumpy.ndarray 类的所有对象方法和属性。
"""

import rsnumpy._core as _core
from .__init__ import ndarray, _ensure


def _wrap_result(result):
    """包装结果为 ndarray 对象"""
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return ndarray._wrap(result)
    return result


class NdArrayMethods:
    """ndarray 对象方法集合"""
    
    @staticmethod
    def astype(arr, dtype):
        """
        将数组转换为指定的数据类型。

        参数:
            dtype: 目标数据类型。

        返回:
            ndarray: 转换后的数组。
        """
        return _wrap_result(arr._array.astype(dtype))
    
    @staticmethod
    def reshape(arr, *shape):
        """
        改变数组形状而不改变数据。

        参数:
            *shape: 新的形状，可以是元组或多个参数。-1 表示自动计算该维度。

        返回:
            ndarray: 重新塑形后的数组。
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = shape
        
        if -1 in shape:
            total_size = arr.size
            known_size = 1
            unknown_idx = -1
            for i, s in enumerate(shape):
                if s == -1:
                    unknown_idx = i
                elif s > 0:
                    known_size *= s
            
            if unknown_idx != -1:
                new_shape = list(shape)
                new_shape[unknown_idx] = total_size // known_size
                shape = tuple(new_shape)
        
        result = arr._array.reshape(shape)
        return _wrap_result(result)
    
    @staticmethod
    def resize(arr, new_shape):
        """
        改变数组形状和大小。

        参数:
            new_shape: 新的形状（int, tuple 或 list）。
        """
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        elif isinstance(new_shape, list):
            new_shape = tuple(new_shape)
        arr._array.resize(new_shape)
    
    @staticmethod
    def ravel(arr, order='C'):
        """
        将数组展平为一维。

        参数:
            order: 展平顺序，'C'表示C风格（行优先），'F'表示Fortran风格（列优先）。

        返回:
            ndarray: 展平后的数组。
        """
        return _wrap_result(arr._array.ravel())
    
    @staticmethod
    def flatten(arr, order='C'):
        """
        返回数组的一维副本。

        参数:
            order: 展平顺序。

        返回:
            ndarray: 展平后的数组副本。
        """
        return _wrap_result(arr._array.flatten())
    
    @staticmethod
    def transpose(arr, *axes):
        """
        转置数组。

        参数:
            *axes: 轴的顺序。

        返回:
            ndarray: 转置后的数组。
        """
        if len(axes) == 0:
            result = arr._array.T
        else:
            result = _core.transpose(arr._array)
        return _wrap_result(result)
    
    @staticmethod
    def swapaxes(arr, axis1, axis2):
        """
        交换两个轴的位置。

        参数:
            axis1: 第一个轴。
            axis2: 第二个轴。

        返回:
            ndarray: 交换轴后的数组。
        """
        arr_obj = ndarray(arr._array)
        axes = list(range(arr_obj.ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return arr_obj.transpose(*axes)
    
    @staticmethod
    def squeeze(arr, axis=None):
        """
        移除长度为1的轴。

        参数:
            axis: 可选，指定要移除的轴。

        返回:
            ndarray: 压缩后的数组。
        """
        return _wrap_result(arr._array.squeeze())
    
    @staticmethod
    def max(arr, axis=None):
        """
        返回数组的最大值。

        参数:
            axis: 计算最大值的轴。

        返回:
            标量或 ndarray: 最大值。
        """
        result = arr._array.max(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def min(arr, axis=None):
        """
        返回数组的最小值。

        参数:
            axis: 计算最小值的轴。

        返回:
            标量或 ndarray: 最小值。
        """
        result = arr._array.min(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def mean(arr, axis=None):
        """
        计算数组的平均值。

        参数:
            axis: 计算平均值的轴。

        返回:
            标量或 ndarray: 平均值。
        """
        result = arr._array.mean(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def std(arr, axis=None):
        """
        计算数组的标准差。

        参数:
            axis: 计算标准差的轴。

        返回:
            标量或 ndarray: 标准差。
        """
        result = arr._array.std(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def var(arr, axis=None):
        """
        计算数组的方差。

        参数:
            axis: 计算方差的轴。

        返回:
            标量或 ndarray: 方差。
        """
        result = arr._array.var(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def sum(arr, axis=None):
        """
        计算数组元素的和。

        参数:
            axis: 求和的轴。

        返回:
            标量或 ndarray: 和。
        """
        result = arr._array.sum(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def prod(arr, axis=None):
        """
        计算数组元素的乘积。

        参数:
            axis: 求乘积的轴。

        返回:
            标量或 ndarray: 乘积。
        """
        result = arr._array.prod(axis)
        if axis is None and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def cumsum(arr, axis=None):
        """
        计算累积和。

        参数:
            axis: 计算累积和的轴。

        返回:
            ndarray: 累积和数组。
        """
        result = arr._array.cumsum(axis)
        return _wrap_result(result)
    
    @staticmethod
    def cumprod(arr, axis=None):
        """
        计算累积乘积。

        参数:
            axis: 计算累积乘积的轴。

        返回:
            ndarray: 累积乘积数组。
        """
        result = arr._array.cumprod(axis)
        return _wrap_result(result)
    
    @staticmethod
    def argmax(arr, axis=None):
        """
        返回最大值的索引。

        参数:
            axis: 查找最大值的轴。

        返回:
            标量或 ndarray: 索引。
        """
        return _wrap_result(_core.argmax_axis(arr._array, axis))
    
    @staticmethod
    def argmin(arr, axis=None):
        """
        返回最小值的索引。

        参数:
            axis: 查找最小值的轴。

        返回:
            标量或 ndarray: 索引。
        """
        return _wrap_result(_core.argmin_axis(arr._array, axis))
    
    @staticmethod
    def argsort(arr, axis=-1):
        """
        返回排序后的索引。

        参数:
            axis: 排序的轴。

        返回:
            ndarray: 索引数组。
        """
        result = arr._array.argsort(axis)
        return _wrap_result(result)
    
    @staticmethod
    def sort(arr, axis=-1):
        """
        原地排序数组。

        参数:
            axis: 排序的轴。
        """
        result = arr._array.sort(axis)
        if hasattr(result, 'shape') and result.shape == arr.shape:
            arr._array = result
        else:
            arr._array.data = result.data
    
    @staticmethod
    def diagonal(arr, offset=0, axis1=0, axis2=1):
        """
        返回数组的对角线元素。

        参数:
            offset: 对角线偏移量。
            axis1: 第一个轴。
            axis2: 第二个轴。

        返回:
            ndarray: 对角线元素。
        """
        result = arr._array.diagonal(offset, axis1, axis2)
        return _wrap_result(result)
    
    @staticmethod
    def trace(arr, offset=0, axis1=0, axis2=1):
        """
        计算数组的迹（对角线元素之和）。

        参数:
            offset: 对角线偏移量。
            axis1: 第一个轴。
            axis2: 第二个轴。

        返回:
            标量: 迹。
        """
        return arr._array.trace(offset, axis1, axis2)
    
    @staticmethod
    def fill(arr, value):
        """
        用指定值填充数组。

        参数:
            value: 填充值。
        """
        arr._array.fill(value)
    
    @staticmethod
    def item(arr, *args):
        """
        获取数组的单个元素。

        参数:
            *args: 索引。

        返回:
            标量: 元素值。
        """
        return arr._array.item(*args)
    
    @staticmethod
    def tolist(arr):
        """
        将数组转换为 Python 列表。

        返回:
            list: Python 列表表示。
        """
        return arr._array.tolist()
    
    @staticmethod
    def take(arr, indices, axis=None):
        """
        根据索引获取元素。

        参数:
            indices: 索引数组。
            axis: 获取元素的轴。

        返回:
            ndarray: 选取的元素。
        """
        result = arr._array.take(_ensure(indices), axis)
        return _wrap_result(result)
    
    @staticmethod
    def put(arr, indices, values):
        """
        将值放入数组的指定位置。

        参数:
            indices: 索引数组。
            values: 要放入的值。
        """
        arr._array.put(_ensure(indices), _ensure(values))
    
    @staticmethod
    def repeat(arr, repeats, axis=None):
        """
        重复数组元素。

        参数:
            repeats: 重复次数。
            axis: 重复的轴。

        返回:
            ndarray: 重复后的数组。
        """
        result = arr._array.repeat(repeats, axis)
        return _wrap_result(result)
    
    @staticmethod
    def nonzero(arr):
        """
        返回非零元素的索引。

        返回:
            tuple: 非零元素的索引元组。
        """
        raw = arr._array.nonzero()
        # raw is list of lists (Vec<Vec<usize>>), convert to tuple of 1D arrays
        result = []
        for indices in raw:
            result.append(ndarray(list(indices)))
        return tuple(result)

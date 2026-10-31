"""
ndarray 对象方法模块

此模块包含 rsnumpy.ndarray 类的所有对象方法和属性。
"""

import rsnumpy.num_core as _core


def _get_deps():
    """延迟获取 ndarray 和 _ensure，避免循环导入导致的 import 缓慢。"""
    import rsnumpy
    return rsnumpy.ndarray, rsnumpy._ensure


def _wrap_result(result, dtype="float64"):
    """包装结果为 ndarray 对象"""
    ndarray, _ = _get_deps()
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return ndarray._wrap(result, _dtype=dtype)
    return result


class NdArrayMethods:
    """ndarray 对象方法集合"""
    
    @staticmethod
    def astype(arr, dtype, copy=True):
        """将数组转换为指定的数据类型。"""
        if isinstance(dtype, str):
            try:
                import rsnumpy as _rn_asm
                name = str(_rn_asm._resolve_dtype(dtype))
            except Exception:
                name = dtype
        else:
            _dn = getattr(dtype, 'name', None)
            name = _dn if isinstance(_dn, str) else getattr(dtype, "__name__", str(dtype))
            # DType 类本身（type(np.dtype(...))）：按裸 S 处理
            if isinstance(dtype, type):
                try:
                    import rsnumpy._dtypes as _dts_asm
                    if issubclass(dtype, _dts_asm.DType):
                        name = "bytes_"
                except Exception:
                    pass
        name = str(name)
        # 提取目标 dtype 的字节序（保留非 native 标记，结果上还原）
        _bo = None
        try:
            if isinstance(dtype, str):
                if dtype and dtype[0] in '<>':
                    _bo = dtype[0]
            else:
                _dbo = getattr(dtype, 'byteorder', None)
                if isinstance(_dbo, str) and _dbo not in ('|', '=', '@', 'n', 'native'):
                    _bo = _dbo
        except Exception:
            _bo = None
        ndarray_cls, _ = _get_deps()

        def _fin(res):
            if _bo is not None and res is not None:
                try:
                    import rsnumpy._dtypes as _dts_bo
                    _d = _dts_bo.DType(name)
                    _d._byteorder = _bo
                    res._dtype_obj = _d
                except Exception:
                    pass
            return res
        # numpy 语义：目标 dtype 相同且 copy=False → 返回自身（视图）
        if copy is False and name == str(getattr(arr, '_dtype', '')):
            return _fin(arr)
        # raw-data 数组（object/complex 标量/字符串）的底层存储是占位存储而非
        # 真实值，经构造器重建以保证转换正确
        _raw = getattr(arr, '_raw_data', None)
        if _raw is not None:
            try:
                if getattr(arr, 'ndim', 0) == 0:
                    res = ndarray_cls(_raw[0] if _raw else None, _dtype=name)
                else:
                    res = ndarray_cls(_raw, _dtype=name)
                    if res.ndim < arr.ndim and arr.shape:
                        res = res.reshape(arr.shape)
                return _fin(res)
            except Exception:
                pass
        # Rust astype 对整数目标转换不正确（int→int 回绕错误、float→int
        # 只重标注）：统一用 Python 语义计算——NaN→0、float 向零截断、
        # int 按目标位宽补码回绕（与 numpy same-kind cast 一致）
        _INT_BITS = {'int8': 8, 'int16': 16, 'int32': 32, 'int64': 64,
                     'uint8': 8, 'uint16': 16, 'uint32': 32, 'uint64': 64}
        if name in _INT_BITS:
            _storage = str(getattr(arr._array, 'dtype', '') or '')
            if _storage in ("float16", "float32", "float64") or _storage in _INT_BITS:
                try:
                    def _rec(x):
                        out = []
                        for v in x:
                            if isinstance(v, list):
                                out.extend(_rec(v))
                            else:
                                out.append(v)
                        return out
                    _t = arr.tolist()
                    fv = _rec(_t) if isinstance(_t, list) else [_t]
                    bits = _INT_BITS[name]
                    signed = name.startswith('int')
                    mask = (1 << bits) - 1
                    half = 1 << (bits - 1)
                    out = []
                    for v in fv:
                        if isinstance(v, float) and v != v:
                            out.append(0)  # NaN→0（numpy 未定义行为，与 array() 路径一致）
                        else:
                            iv = int(v) & mask
                            if iv >= half:
                                # 高位值（有符号越界 / u64 超出 i64 构造范围）：
                                # 存 i64 补码负值，tolist 按位宽掩码还原
                                iv -= 1 << bits
                            out.append(iv)
                    if getattr(arr, 'ndim', 0) == 0:
                        tmp = ndarray_cls(out[0])
                    else:
                        tmp = ndarray_cls(out)
                        if tmp.ndim < arr.ndim and arr.shape:
                            tmp = tmp.reshape(arr.shape)
                    return _fin(ndarray_cls._wrap(tmp._array, _dtype=name))
                except Exception:
                    pass
        # float 目标（f64→f32/f16）：真正舍入到目标精度（NEP50 语义）
        if name in ('float32', 'float16'):
            _storage = str(getattr(arr._array, 'dtype', '') or '')
            if _storage in ('float64', 'float32'):
                try:
                    import struct as _struct
                    fmt = 'f' if name == 'float32' else 'e'
                    lim = 3.4028235e38 if fmt == 'f' else 65504.0

                    def _rec2(x):
                        o = []
                        for v in x:
                            if isinstance(v, list):
                                o.extend(_rec2(v))
                            else:
                                o.append(v)
                        return o

                    def _rnd(v):
                        if isinstance(v, float) and v == v and abs(v) < lim:
                            try:
                                return _struct.unpack(fmt, _struct.pack(fmt, v))[0]
                            except Exception:
                                return v
                        return v

                    _t = arr.tolist()
                    fv = _rec2(_t) if isinstance(_t, list) else [_t]
                    rv = [_rnd(v) for v in fv]
                    if getattr(arr, 'ndim', 0) == 0:
                        tmp = ndarray_cls(rv[0])
                    else:
                        tmp = ndarray_cls(rv)
                        if tmp.ndim < arr.ndim and arr.shape:
                            tmp = tmp.reshape(arr.shape)
                    return _fin(ndarray_cls._wrap(tmp._array, _dtype=name))
                except Exception:
                    pass
        # 复数目标：用复数值列表重建（避免 Rust 复数转换缺口）
        if name in ('complex64', 'complex128'):
            try:
                def _flatc(x):
                    if isinstance(x, list):
                        out = []
                        for v in x:
                            out.extend(_flatc(v))
                        return out
                    return [x]
                _t = arr.tolist()
                fv = [complex(v) for v in (_flatc(_t) if isinstance(_t, list) else [_t])]
                tmp = ndarray_cls(fv)
                if getattr(arr, 'ndim', 0) == 0:
                    tmp = tmp.reshape(())
                elif tmp.ndim < arr.ndim and arr.shape:
                    tmp = tmp.reshape(arr.shape)
                return ndarray_cls._wrap(tmp._array, _dtype=name)
            except Exception:
                pass
        return _fin(_wrap_result(arr._array.astype(name), dtype=name))

    @staticmethod
    def reshape(arr, *shape):
        """改变数组形状而不改变数据。"""
        ndarray, _ = _get_deps()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        else:
            shape = shape
        try:
            result = arr._array.reshape(shape)
        except ValueError:
            result = arr._array.flatten().reshape(shape)
        out = ndarray._wrap(result, _dtype=getattr(arr, '_dtype', "float64"), _fields=getattr(arr, '_fields', None), _raw_data=getattr(arr, '_raw_data', None))
        try:
            out._rs_parent = arr
        except Exception:
            pass
        return out
    
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
        ndarray, _ = _get_deps()
        dtype = getattr(arr, '_dtype', "float64")
        fields = getattr(arr, '_fields', None)
        raw = getattr(arr, '_raw_data', None)
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        if len(axes) == 0:
            result = arr._array.T
        else:
            result = _core.transpose_axes(arr._array, list(axes))
        return ndarray._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw)
    
    @staticmethod
    def swapaxes(arr, axis1, axis2):
        """交换两个轴的位置。"""
        ndarray, _ = _get_deps()
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
    def max(arr, axis=None, out=None, keepdims=False):
        """返回数组的最大值。"""
        _ = out
        result = arr._array.max(axis, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def min(arr, axis=None, out=None, keepdims=False):
        """返回数组的最小值。"""
        _ = out
        result = arr._array.min(axis, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def mean(arr, axis=None, dtype=None, out=None, keepdims=False):
        """计算数组的平均值。"""
        _ = dtype, out
        result = arr._array.mean(axis, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def std(arr, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """计算数组的标准差。"""
        _ = dtype, out
        result = arr._array.std(axis, ddof, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def var(arr, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """计算数组的方差。"""
        _ = dtype, out
        result = arr._array.var(axis, ddof, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def sum(arr, axis=None, dtype=None, out=None, keepdims=False):
        """计算数组元素的和。"""
        _ = dtype, out
        result = arr._array.sum(axis, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def prod(arr, axis=None, dtype=None, out=None, keepdims=False):
        """计算数组元素的乘积。"""
        _ = dtype, out
        result = arr._array.prod(axis, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
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
        if isinstance(result, int):
            return result
        if hasattr(result, 'ndim') and result.ndim == 0:
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
        _cpx_i = getattr(arr, '_complex_data', None)
        if _cpx_i is not None:
            return _cpx_i[0] if _cpx_i else 0j
        v = arr._array.item(*args)
        # 无符号高位值以 i64 补码负值存储：还原无符号值
        _u_bits = {'uint8': 8, 'uint16': 16, 'uint32': 32,
                   'uint64': 64}.get(str(getattr(arr, '_dtype', '')))
        if _u_bits and isinstance(v, (int, float)) and v < 0:
            v = int(v) & ((1 << _u_bits) - 1)
        return v

    @staticmethod
    def tolist(arr):
        """将数组转换为 Python 列表。"""
        # 超大广播 override 视图：物化会 OOM abort 进程 → MemoryError
        _ovr_t = getattr(arr, '_rs_shape_override', None)
        if _ovr_t is not None:
            _n_t = 1
            for _v_t in _ovr_t:
                _n_t *= int(_v_t)
                if _n_t > 2**40:
                    raise MemoryError(
                        "rsnumpy: cannot materialize huge broadcast array via tolist()")
        # _complex_data（Python 级复数）优先：_array 只有实部
        _cpx_t = getattr(arr, '_complex_data', None)
        if _cpx_t is not None:
            shp = getattr(arr, 'shape', None) or ()
            if not shp:
                return _cpx_t[0] if _cpx_t else 0j

            def _nest_c(vals, dims):
                if len(dims) == 1:
                    return list(vals)
                n = dims[0] or 0
                step = (len(vals) // n) if n else 0
                return [_nest_c(vals[i * step:(i + 1) * step], dims[1:])
                        for i in range(n)]
            return _nest_c(list(_cpx_t), shp)
        v = arr._array.tolist()
        # 无符号高位值以 i64 补码负值存储：还原无符号值
        _u_bits = {'uint8': 8, 'uint16': 16, 'uint32': 32,
                   'uint64': 64}.get(str(getattr(arr, '_dtype', '')))
        if _u_bits:
            _mask = (1 << _u_bits) - 1

            def _mk(x):
                if isinstance(x, list):
                    return [_mk(y) for y in x]
                if isinstance(x, (int, float)) and not isinstance(x, bool) \
                        and x < 0:
                    return int(x) & _mask
                return x
            return _mk(v)
        return v
    
    @staticmethod
    def take(arr, indices, axis=None):
        """根据索引获取元素。"""
        _, _ensure = _get_deps()
        result = arr._array.take(_ensure(indices), axis)
        return _wrap_result(result)
    
    @staticmethod
    def put(arr, indices, values):
        """将值放入数组的指定位置。"""
        _, _ensure = _get_deps()
        arr._array.put(_ensure(indices), _ensure(values))
    
    @staticmethod
    def repeat(arr, repeats, axis=None):
        """重复数组元素。"""
        result = arr._array.repeat(repeats, axis)
        return _wrap_result(result)
    
    @staticmethod
    def nonzero(arr):
        """返回非零元素的索引。"""
        ndarray, _ = _get_deps()
        raw = _core.nonzero_arrs(arr._array)
        return tuple(ndarray._wrap(r) for r in raw)

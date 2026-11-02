"""
ndarray 对象方法模块

此模块包含 rsnumpy.ndarray 类的所有对象方法和属性。
"""

import builtins

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
        # S/U 字符串目标：Rust astype 不支持字符串存储。numpy 语义：
        # 值转字符串后取最小宽度；裸 'S'/'U' 按数据推断（astype('S')→'S3'），
        # 显式宽度（'S4'）直接用（np.array 自带截断）。
        _su_kind = None
        _su_width = None
        if isinstance(dtype, str):
            _core = dtype[1:] if dtype and dtype[0] in '<>|=' else dtype
            if _core and _core[0] in 'SU':
                _su_kind = _core[0]
                _tail = _core[1:]
                _su_width = builtins.int(_tail) if _tail.isdigit() else None
        elif name in ('bytes_', 'string_'):
            _su_kind = 'S' if name == 'bytes_' else 'U'
        if _su_kind is not None:
            try:
                import rsnumpy as _rn_su

                def _flats(x):
                    if isinstance(x, list):
                        out = []
                        for v in x:
                            out.extend(_flats(v))
                        return out
                    return [x]

                _t = arr.tolist()
                _vals = _flats(_t) if isinstance(_t, list) else [_t]
                _strs = [str(v) for v in _vals]
                if _su_width is None:
                    if _su_kind == 'S':
                        _w = builtins.max(
                            (len(s.encode('utf-8')) for s in _strs), default=0)
                    else:
                        _w = builtins.max((len(s) for s in _strs), default=0)
                    _su_width = _w
                _spec = f'{_su_kind}{_su_width}'
                if getattr(arr, 'ndim', 0) == 0:
                    # 0-d 直接标量构造：reshape(()) 会抹掉字符串宽度
                    return _fin(_rn_su.array(_strs[0], dtype=_spec))
                _res = _rn_su.array(_strs, dtype=_spec)
                return _fin(_res)
            except Exception:
                pass
        # object 目标：Rust 无 object 存储，按 Python 值重建（numpy 语义：
        # 对象原样保留）。0-d object 源（_raw_data 单元素）与数值源都覆盖。
        if name in ('object', 'O'):
            try:
                def _flo(x):
                    if isinstance(x, list):
                        out = []
                        for v in x:
                            out.extend(_flo(v))
                        return out
                    return [x]
                src = getattr(arr, '_raw_data', None)
                if src is not None:
                    fv = list(src) if isinstance(src, (list, tuple)) else [src]
                else:
                    _t = arr.tolist()
                    fv = _flo(_t) if isinstance(_t, list) else [_t]
                tmp = ndarray_cls(fv, _dtype='object')
                if getattr(arr, 'ndim', 0) == 0:
                    tmp = tmp.reshape(())
                elif tmp.ndim < arr.ndim and arr.shape:
                    tmp = tmp.reshape(arr.shape)
                return _fin(tmp)
            except Exception:
                pass
        return _fin(_wrap_result(arr._array.astype(name), dtype=name))

    @staticmethod
    def reshape(arr, *shape):
        """改变数组形状而不改变数据。"""
        ndarray, _ = _get_deps()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = builtins.tuple(shape[0])
        else:
            shape = builtins.tuple(shape)
        # 0 尺寸数组：Rust 不接受 0 维——占位 + shape override（numpy 语义）。
        _sz = 1
        for _d in shape:
            _sz *= _d
        if _sz == 0:
            import rsnumpy as _rn
            out = ndarray._wrap(_rn.zeros((1,)),
                                _dtype=getattr(arr, '_dtype', "float64"),
                                _fields=getattr(arr, '_fields', None),
                                _raw_data=getattr(arr, '_raw_data', None))
            out._rs_shape_override = shape
            try:
                out._rs_parent = arr
            except Exception:
                pass
            return out
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
        # 逻辑空数组：Rust 层对空数据 resize 会死循环（copy_len 恒 0）；
        # numpy 语义：扩大时新元素以 0 填充 → Python 层直接重建。
        if arr.size == 0:
            try:
                arr._array = _core.zeros(list(new_shape))
            except Exception:
                pass
            _ov = getattr(arr, '_rs_shape_override', None)
            if _ov is not None:
                arr._rs_shape_override = None
            return
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
        """转置数组（视图：标记 base 供 flags.OWNDATA 判定）。"""
        ndarray, _ = _get_deps()
        dtype = getattr(arr, '_dtype', "float64")
        fields = getattr(arr, '_fields', None)
        raw = getattr(arr, '_raw_data', None)
        if len(axes) == 1:
            a0 = axes[0]
            if isinstance(a0, (tuple, list)):
                axes = a0
            elif hasattr(a0, 'ndim') and getattr(a0, 'ndim', 1) == 1:
                # numpy 接受一维数组作为轴置换
                axes = [int(x) for x in a0.reshape((-1,)).tolist()]
        if len(axes) == 0:
            result = arr._array.T
        else:
            _ax = [int(x) for x in axes]
            _nd = arr.ndim
            if len(_ax) != _nd:
                # numpy 语义：轴数必须与数组维度一致
                raise ValueError("axes don't match array")
            _norm = [x + _nd if x < 0 else x for x in _ax]
            if len(set(_norm)) != len(_norm):
                # numpy 语义：轴不可重复
                raise ValueError("repeated axis in transpose")
            for x in _norm:
                if x < 0 or x >= _nd:
                    raise ValueError(
                        f"axis {x} is out of bounds for array of "
                        f"dimension {_nd}")
            result = _core.transpose_axes(arr._array, _norm)
        out = ndarray._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw)
        if raw is None:
            out.base = arr
        return out
    
    @staticmethod
    def swapaxes(arr, axis1, axis2):
        """交换两个轴的位置（视图：标记 base 供 flags.OWNDATA 判定）。"""
        ndarray, _ = _get_deps()
        dtype = getattr(arr, '_dtype', "float64")
        fields = getattr(arr, '_fields', None)
        raw_data = getattr(arr, '_raw_data', None)
        result = _core.swapaxes(arr._array, axis1, axis2)
        out = ndarray._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)
        if raw_data is None:
            out.base = arr
        return out
    
    @staticmethod
    def squeeze(arr, axis=None):
        """移除长度为1的轴（视图：标记 base 供 flags.OWNDATA 判定）。"""
        result = arr._array.squeeze()
        if getattr(arr, '_raw_data', None) is not None:
            return _wrap_result(result)
        out = _wrap_result(result)
        if hasattr(out, 'base'):
            out.base = arr
        return out
    
    @staticmethod
    def _reduce_axis_check(arr, axis):
        """归约 axis 校验（numpy 语义：越界 → AxisError；0-d 允许 axis=0）。

        越界 axis 直达 Rust 会 panic abort 整个进程；0-d + axis=0
        归一化为 None（全归约）——Rust 对 0-d 的 axis=0 也会 panic。
        """
        if axis is None:
            return None
        _dim = int(arr.ndim)
        _ax = axis + _dim if axis < 0 else axis
        if _dim == 0:
            if _ax != 0:
                from rsnumpy.exceptions import AxisError
                raise AxisError(
                    f"axis {axis} is out of bounds for array of "
                    f"dimension {_dim}")
            return None
        if _ax < 0 or _ax >= _dim:
            from rsnumpy.exceptions import AxisError
            raise AxisError(
                f"axis {axis} is out of bounds for array of "
                f"dimension {_dim}")
        return axis

    @staticmethod
    def max(arr, axis=None, out=None, keepdims=False):
        """返回数组的最大值。"""
        _ = out
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = arr._array.max(axis, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def min(arr, axis=None, out=None, keepdims=False):
        """返回数组的最小值。"""
        _ = out
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = arr._array.min(axis, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def mean(arr, axis=None, dtype=None, out=None, keepdims=False):
        """计算数组的平均值。"""
        _ = dtype, out
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = arr._array.mean(axis, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def std(arr, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """计算数组的标准差。"""
        _ = dtype, out
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = arr._array.std(axis, ddof, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def var(arr, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """计算数组的方差。"""
        _ = dtype, out
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = arr._array.var(axis, ddof, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def sum(arr, axis=None, dtype=None, out=None, keepdims=False):
        """计算数组元素的和。"""
        _ = dtype, out
        # 0-d 短路：Rust 归约对 0-d 输入会 abort；numpy 返回保持原 dtype 的 0-d
        if not tuple(getattr(arr._array, 'shape', ())):
            nd = _nd()
            return nd(arr._array, _dtype=getattr(arr, '_dtype', 'float64'))
        # complex 短路：Rust 归约对 complex 会 abort，纯 Python 求和
        _k = getattr(getattr(arr, '_dtype_obj', None), 'kind', None)
        if _k is None:
            _dn = str(getattr(arr, '_dtype', ''))
            _k = _dn[0] if _dn else '?'
        if _k == 'c':
            _vals = arr.tolist()

            def _cs(x):
                if isinstance(x, (list, tuple)):
                    return _cs(x[0]) + sum(_cs(y) for y in x[1:])
                return x

            _tot = _cs(_vals)
            return _tot
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = arr._array.sum(axis, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)

    @staticmethod
    def prod(arr, axis=None, dtype=None, out=None, keepdims=False):
        """计算数组元素的乘积。"""
        _ = dtype, out
        # 0-d 短路：Rust 归约对 0-d 输入会 abort；numpy 返回保持原 dtype 的 0-d
        if not tuple(getattr(arr._array, 'shape', ())):
            nd = _nd()
            return nd(arr._array, _dtype=getattr(arr, '_dtype', 'float64'))
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = arr._array.prod(axis, keepdims)
        if axis is None and not keepdims and hasattr(result, 'tolist'):
            return result.tolist()
        return _wrap_result(result)
    
    @staticmethod
    def cumsum(arr, axis=None):
        """计算累积和。"""
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = arr._array.cumsum(axis)
        return _wrap_result(result)
    
    @staticmethod
    def cumprod(arr, axis=None):
        """计算累积乘积。"""
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = arr._array.cumprod(axis)
        return _wrap_result(result)
    
    @staticmethod
    def argmax(arr, axis=None):
        """返回最大值的索引。"""
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = _wrap_result(_core.argmax_axis(arr._array, axis))
        if result.ndim == 0:
            return int(result.item())
        return result
    
    @staticmethod
    def argmin(arr, axis=None):
        """返回最小值的索引。"""
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
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
    def take(arr, indices, axis=None, mode='raise', out=None):
        """根据索引获取元素（保留输入数组 dtype；numpy 兼容 mode 语义）。

        越界索引抛 IndexError（直达 Rust 会 panic abort 整个进程）。
        全部索引合法时走 Rust 快路径，否则 Python 回退（含 clip/wrap）。
        """
        ndarray, _ensure = _get_deps()
        dt = getattr(arr, '_dtype', 'float64')

        def _norm(i, size, ax):
            if size == 0:
                raise IndexError(
                    f"index {i} is out of bounds for axis {ax} "
                    f"with size {size}")
            if mode == 'clip':
                if i > size - 1:
                    i = size - 1
                if i < -size:
                    i = -size
            elif mode != 'wrap':
                if i < -size or i >= size:
                    raise IndexError(
                        f"index {i} is out of bounds for axis {ax} "
                        f"with size {size}")
            return i % size

        if isinstance(indices, int) and not isinstance(indices, bool):
            # 标量索引：保持既有 Rust 行为
            result = arr._array.take(_ensure(indices), axis)
            return _wrap_result(result, dt)
        idxs = indices.tolist() if hasattr(indices, 'tolist') else list(indices)
        if axis is None:
            n = 1
            for d in arr.shape:
                n *= d
            norm = [_norm(i, n, 0) for i in idxs]
            if mode == 'raise' and all(0 <= i < n for i in idxs):
                result = arr._array.take(_ensure(list(idxs)), None)
                return _wrap_result(result, dt)
            flat = arr.ravel().tolist()
            if not isinstance(flat, list):
                flat = [flat]
            return ndarray([flat[i] for i in norm], _dtype=dt)
        ax = axis % arr.ndim
        size = arr.shape[ax]
        norm = [_norm(i, size, axis) for i in idxs]
        if mode == 'raise' and all(0 <= i < size for i in idxs):
            result = arr._array.take(_ensure(list(idxs)), axis)
            return _wrap_result(result, dt)
        other = tuple(arr.shape[i] for i in range(arr.ndim) if i != ax)
        rows = (arr.moveaxis(ax, -1).reshape((-1, size)).tolist()
                if size > 0 else [])
        out_rows = [[row[i] for i in norm] for row in rows]
        return ndarray(out_rows, _dtype=dt).reshape(other + (len(idxs),))
    
    @staticmethod
    def put(arr, indices, values, mode='raise'):
        """将值放入数组的指定位置（numpy 兼容 mode 语义）。

        空 values 为 no-op；越界索引抛 IndexError（避免 Rust panic abort）。
        """
        ndarray, _ensure = _get_deps()
        vals = values.tolist() if hasattr(values, 'tolist') else list(values)
        if not vals:
            # numpy 语义：空 values 不修改数组
            return
        n = 1
        for d in arr.shape:
            n *= d
        idxs = [int(v) for v in
                (indices.tolist() if hasattr(indices, 'tolist')
                 else list(indices))]

        def _norm(i):
            if n == 0:
                raise IndexError(
                    f"index {i} is out of bounds for axis 0 with size 0")
            if mode == 'clip':
                if i > n - 1:
                    i = n - 1
                if i < -n:
                    i = -n
            elif mode != 'wrap':
                if i < -n or i >= n:
                    raise IndexError(
                        f"index {i} is out of bounds for axis 0 "
                        f"with size {n}")
            return i % n

        norm = [_norm(i) for i in idxs]
        if mode == 'raise' and all(0 <= i < n for i in idxs):
            arr._array.put(_ensure(idxs), _ensure(values))
            return
        # 越界/clip/wrap：就地修改扁平缓冲
        flat = arr.ravel().tolist()
        if not isinstance(flat, list):
            flat = [flat]
        for i, v in zip(norm, vals):
            flat[i] = v
        dt = getattr(arr, '_dtype', 'float64')
        arr._array = ndarray(flat, _dtype=dt).reshape(
            tuple(arr.shape))._array
    
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

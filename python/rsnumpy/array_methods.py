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
        # 结构化目标含 object 字段：底层 float 存储无法表示，显式拒绝
        # （静默降级会损坏数据；numpy 对 object 字段的 axis 运算同样抛
        # TypeError）。
        if isinstance(dtype, (list, tuple)) and dtype and builtins.all(
                isinstance(f, (list, tuple)) and len(f) >= 2 for f in dtype):
            if builtins.any(f[1] is object or f[1] in ('object', 'O')
                            for f in dtype):
                raise TypeError(
                    "cannot astype to a structured dtype containing object "
                    "fields (rsnumpy cannot represent object fields)")
        # 非结构化 → 结构化：逐元素逐字段 cast（numpy astype 语义：
        # 每个标量按各字段 dtype 转换成一个记录）
        try:
            import rsnumpy as _rn_ast
            _dst_dt = _rn_ast.dtype(dtype)
            _dst_names = getattr(_dst_dt, 'names', None)
            if _dst_names is not None:
                _src_dt = getattr(arr, '_dtype_obj', None)
                _src_structured = (_src_dt is not None
                                   and getattr(_src_dt, '_names', None)
                                   is not None)
                if not _src_structured:
                    _recs = []
                    for _v in arr.tolist():
                        _recs.append(tuple(
                            _rn_ast._coerce_field_value(
                                _v, _dst_dt._fields[_n][0])
                            for _n in _dst_names))
                    res = _rn_ast.array(_recs, dtype=dtype)
                    if res.ndim < arr.ndim and arr.shape:
                        res = res.reshape(arr.shape)
                    return res
        except Exception:
            pass
        # 字符串目标（S/U）：_resolve_dtype 会把 S/U 降解成 float64，须单独
        # 处理（numpy 语义：S 源解码并在 null 截断、U 源编码、宽度变更截断/
        # 补零；裸 S/U（无宽度）= 自动宽度）
        _str_tgt = None
        try:
            import rsnumpy as _rn_st
            import rsnumpy._dtypes as _dts_st2
            _DType_st = _dts_st2.DType
            if isinstance(dtype, str):
                _core_st = dtype[1:] if dtype and dtype[0] in '<>|=' else dtype
                _st_exp_w = (bool(_core_st) and _core_st[0] in 'SU'
                             and _core_st[1:].isdigit())
            else:
                _st_exp_w = True
            _tdt_st = dtype if isinstance(dtype, _DType_st) \
                else _rn_st.dtype(dtype)
            if getattr(_tdt_st, 'kind', None) in ('S', 'U'):
                _str_tgt = (getattr(_tdt_st, '_kind', None), _st_exp_w)
        except Exception:
            _str_tgt = None
        if _str_tgt is not None and getattr(arr, '_raw_data', None) is not None \
                and getattr(arr, '_str_dtype', None) is not None:
            _tk, _st_exp_w = _str_tgt
            _st_vals = arr.tolist()
            if not isinstance(_st_vals, list):
                _st_vals = [_st_vals]
            _st_norm = []
            for _v in _st_vals:
                if isinstance(_v, (bytes, bytearray)):
                    _st_norm.append(
                        bytes(_v).decode('latin-1').split('\x00')[0])
                else:
                    _st_norm.append(str(_v).split('\x00')[0])
            if _tk == 'U':
                _st_w = (_tdt_st._itemsize // 4) if _st_exp_w else \
                    max([len(_s) for _s in _st_norm] + [1])
                _st_out = [_s[:_st_w] for _s in _st_norm]
            else:
                _st_w = _tdt_st._itemsize if _st_exp_w else \
                    max([len(_s) for _s in _st_norm] + [1])
                _st_out = [_s.encode('latin-1')[:_st_w] for _s in _st_norm]
            _st_bo = getattr(_tdt_st, '_byteorder', None)
            _st_prefix = '>' if _st_bo == '>' else '<'
            _st_res = _rn_st.array(_st_out, dtype=f'{_tk}{max(_st_w, 1)}')
            if _st_bo in ('<', '>'):
                try:
                    import rsnumpy._dtypes as _dts_st
                    _st_res._dtype_obj = _dts_st._clone_with_byteorder(
                        _dts_st.DType(f'{_tk}{max(_st_w, 1)}'), _st_bo)
                except Exception:
                    pass
            if _st_res.ndim < arr.ndim and arr.shape:
                _st_res = _st_res.reshape(arr.shape)
            return _st_res
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
                    res._dtype_obj = _dts_bo._clone_with_byteorder(_d, _bo)
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
                    signed = name.startswith('int')  # noqa: F841
                    mask = (1 << bits) - 1
                    half = 1 << (bits - 1)
                    _signed = name.startswith('int')
                    out = []
                    for v in fv:
                        if isinstance(v, float) and v != v:
                            out.append(0)  # NaN→0（numpy 未定义行为，与 array() 路径一致）
                        elif _signed:
                            iv = int(v) & mask
                            if iv >= half:
                                # 有符号越界：存 i64 补码负值（与 asarray 一致）
                                iv -= 1 << bits
                            out.append(iv)
                        else:
                            # 无符号 <2^63：存值本身（与 asarray 约定一致）；符号扩展
                            # 会把 u8 255 存成 -1.0，== 比较（f64 原值）必挂。
                            # uint64 高位（≥2^63）f64 舍入会越界（2^64-5→2^64.0，
                            # 下游 _check_int_coerce 抛 OverflowError），回落补码负值
                            iv = int(v) & mask
                            if bits == 64 and iv >= half:
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
    def reshape(arr, *shape, copy=None):
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
            # numpy 语义：零尺寸 reshape 保留目标形状（如 (1,0,1)）；
            # Rust 拒绝零尺寸 → 经 zeros 按目标形状构造（继承元数据）。
            import rsnumpy as _rn
            _dt = getattr(arr, '_dtype', "float64")
            try:
                _z = _rn.zeros(shape, dtype=_dt)
                _z._dtype_obj = getattr(arr, '_dtype_obj', None)
                _z._str_dtype = getattr(arr, '_str_dtype', None)
                _z._fields = getattr(arr, '_fields', None)
                _z._raw_data = getattr(arr, '_raw_data', None)
                try:
                    _z._rs_parent = arr
                except Exception:
                    pass
                return _z
            except Exception:
                pass
            out = ndarray._wrap(_rn.zeros((1,)),
                                _dtype=_dt,
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
        # 继承结构化 DType（缺失会导致字段访问/属性查找失效）
        out._dtype_obj = getattr(arr, '_dtype_obj', None)
        # 字符串数组：宽度存 _str_dtype（_dtype 是 'string_' 占位），缺失时
        # 结果退化成裸 |S0（如 str 数组 reshape 后 dtype 失真）
        out._str_dtype = getattr(arr, '_str_dtype', None)
        try:
            out._rs_parent = arr
        except Exception:
            pass
        # 子类保持（numpy 语义：recarray.reshape → recarray；其余子类
        # 如 matrix/chararray 有各自语义，不在此统一保持）
        _src_cls = builtins.type(arr)
        if _src_cls is not ndarray:
            try:
                import rsnumpy as _rn_rs
                _rec_cls = _rn_rs.recarray
                if issubclass(_src_cls, _rec_cls):
                    out.__class__ = _src_cls
            except Exception:
                pass
        if copy is True:
            out = out.copy()
            try:
                out._rs_parent = None
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
        """返回最大值的索引（全归约返回 intp 标量，numpy 语义）。"""
        axis = NdArrayMethods._reduce_axis_check(arr, axis)
        result = _wrap_result(_core.argmax_axis(arr._array, axis))
        if result.ndim == 0:
            import rsnumpy as _rsn
            return _rsn.int64(result.item())
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
        kind = getattr(getattr(arr, 'dtype', None), 'kind', None)
        if kind in ('S', 'U'):
            # Rust 层 fill 只收数值；字符串 dtype 走 setitem 强转
            if arr.ndim == 0:
                arr[()] = value
            else:
                for _i in range(int(arr.size)):
                    arr[_i] = value
            return
        arr._array.fill(value)
    
    @staticmethod
    def item(arr, *args):
        """获取数组的单个元素。"""
        _cpx_i = getattr(arr, '_complex_data', None)
        if _cpx_i is not None:
            return _cpx_i[0] if _cpx_i else 0j
        # numpy item 语义：item(0, 0) 与 item((0, 0)) 等价；越界索引抛
        # IndexError（Python 层预检——Rust item 只收扁平 int，传 tuple 会
        # TypeError 而非 numpy 的 IndexError）。
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            idx = tuple(int(i) for i in args[0])
        else:
            idx = tuple(int(i) for i in args)
        shp = tuple(arr.shape)
        if len(idx) > len(shp):
            raise IndexError(
                f"too many indices for array: array is {len(shp)}-dimensional, "
                f"but {len(idx)} were indexed")
        for _ai, (_i, _d) in enumerate(zip(idx, shp)):
            if _d == 0 or _i < -_d or _i >= _d:
                raise IndexError(
                    f"index {_i} is out of bounds for axis {_ai} with size {_d}")
        # object 数组：返回 _raw_data 存储的 Python 引用（numpy 保持对象身份；
        # Rust 侧 f64 占位值对 object 无意义）
        if str(getattr(arr, '_dtype', '')) == 'object':
            _raw_i = getattr(arr, '_raw_data', None)
            if _raw_i is not None:
                if not idx:
                    if int(arr.size) != 1:
                        raise ValueError(
                            "can only convert an array of size 1 "
                            "to a Python scalar")
                    cur = _raw_i
                    while isinstance(cur, list) and len(cur) == 1:
                        cur = cur[0]
                    return cur
                if len(idx) == 1 and isinstance(_raw_i, list):
                    _i0 = idx[0]
                    if _i0 < 0:
                        _i0 += shp[0]
                    if 0 <= _i0 < len(_raw_i) \
                            and not isinstance(_raw_i[_i0], list):
                        return _raw_i[_i0]
        if len(idx) == 1:
            v = arr._array.item(idx[0])
        elif idx:
            _flat = 0
            for _i, _d in zip(idx, shp):
                _flat = _flat * _d + _i
            v = arr._array.item(_flat)
        else:
            v = arr._array.item()
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
            # indices 可以是多维：结果与 indices 同形状（numpy 语义：沿展平取）。
            _idx_shape = getattr(indices, 'shape', None)
            _is_nd_idx = _idx_shape is not None and len(tuple(_idx_shape)) > 1
            _flat_idxs = []

            def _rec_idx(x):
                if isinstance(x, (list, tuple)):
                    for w in x:
                        _rec_idx(w)
                else:
                    _flat_idxs.append(x)

            _rec_idx(idxs)
            norm = [_norm(i, n, 0) for i in _flat_idxs]
            if (mode == 'raise' and all(0 <= i < n for i in _flat_idxs)
                    and not _is_nd_idx):
                result = arr._array.take(_ensure(list(_flat_idxs)), None)
                return _wrap_result(result, dt)
            flat = arr.ravel().tolist()
            if not isinstance(flat, list):
                flat = [flat]
            res = [flat[i] for i in norm]
            if _is_nd_idx:
                def _shp(flatv, sh):
                    if len(sh) == 1:
                        return list(flatv)
                    _sub = 1
                    for _x in sh[1:]:
                        _sub *= _x
                    return [_shp(flatv[_i * _sub:(_i + 1) * _sub], sh[1:])
                            for _i in range(sh[0])]
                return ndarray(_shp(res, tuple(_idx_shape)), _dtype=dt)
            return ndarray(res, _dtype=dt)
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
        n = 1
        for d in arr.shape:
            n *= d
        # indices 可为标量 int / ndarray 标量 / 数组 / list（numpy put 语义）
        if hasattr(indices, 'tolist'):
            _t = indices.tolist()
            if isinstance(_t, (bool, int, float)):
                idxs = [int(_t)]
            else:
                idxs = [int(v) for v in _t]
        elif isinstance(indices, (bool, int)):
            idxs = [int(indices)]
        else:
            idxs = [int(v) for v in list(indices)]
        # values 可为标量（填到全部索引，numpy put 语义）——先展开，避免
        # list(scalar) 报 'int' object is not iterable
        if hasattr(values, 'tolist'):
            _vt = values.tolist()
        else:
            _vt = values
        if isinstance(_vt, (bool, int, float, complex)):
            values = [_vt] * len(idxs)
        vals = values.tolist() if hasattr(values, 'tolist') else list(values)
        if not vals:
            # numpy 语义：空 values 不修改数组
            return

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
        # 数组 repeats：Rust 仅支持标量 → Python 展开（1-D；2-D 轴展开）
        if (isinstance(repeats, (list, tuple))
                or (hasattr(repeats, 'ndim')
                    and getattr(repeats, 'ndim', 0) > 0)):
            from rsnumpy import ndarray as _ND
            shp = tuple(arr.shape)
            cnts = repeats.tolist() if hasattr(repeats, 'tolist') \
                else list(repeats)
            if not isinstance(cnts, (list, tuple)):
                cnts = [cnts]
            cnts = [int(c) for c in cnts]
            if len(shp) <= 1:
                flat = arr.ravel().tolist()
                if not isinstance(flat, list):
                    flat = [flat]
                out = []
                for v, c in zip(flat, cnts):
                    out.extend([v] * c)
                return _ND(out, _dtype=getattr(arr, '_dtype', 'float64'))
            rows = arr.tolist()
            if axis == 0:
                out = []
                for i, row in enumerate(rows):
                    out.extend([row] * cnts[i])
                return _ND(out, _dtype=getattr(arr, '_dtype', 'float64'))
            out = []
            for row in rows:
                nr = []
                for v, c in zip(row, cnts):
                    nr.extend([v] * c)
                out.append(nr)
            return _ND(out, _dtype=getattr(arr, '_dtype', 'float64'))
        # 字符串数组：Rust repeat 作用于 f64 占位存储（产出 float 零、丢
        # 字符串数据）——标量 repeats 路径须在 Python 层按元素重复
        #（数组 repeats 已在上方分支 Python 展开）
        _sd_r = getattr(arr, '_str_dtype', None)
        if _sd_r is not None and getattr(arr, '_raw_data', None) is not None:
            import rsnumpy as _rn_r
            _rdt = f'{_sd_r[0]}{max(int(_sd_r[1]), 1)}'
            _r = int(repeats)
            shp = [int(s) for s in arr.shape]
            if not shp:
                _out = [arr.tolist()] * _r
            elif axis is None:
                # tolist 平铺 C 序（ravel 走 Rust 占位存储会丢字符串数据）
                _t = arr.tolist()
                if not isinstance(_t, list):
                    _t = [_t]

                def _fl(x):
                    o = []
                    for v in x:
                        o.extend(_fl(v) if isinstance(v, list) else [v])
                    return o
                _flat = _fl(_t)
                _out = [v for v in _flat for _ in range(_r)]
            else:
                _ax = int(axis) % len(shp)

                def _ra(nested, a):
                    if a == 0:
                        return [v for v in nested for _ in range(_r)]
                    return [_ra(row, a - 1) for row in nested]

                def _unflat(flat, shape):
                    if not shape or len(shape) == 1:
                        return list(flat)
                    _cnt = 1
                    for _s in shape[1:]:
                        _cnt *= _s
                    return [_unflat(flat[i * _cnt:(i + 1) * _cnt], shape[1:])
                            for i in range(shape[0])]
                _t = arr.tolist()
                if not isinstance(_t, list):
                    _t = [[_t]]
                # 多维字符串 tolist 可能返回平铺（C 序）：先重建嵌套
                if len(shp) > 1 and _t and not isinstance(_t[0], list):
                    _t = _unflat(_t, shp)
                _out = _ra(_t, _ax)
                shp[_ax] *= _r
            # 模块级 array() 走真正的字符串构造路径（ndarray() 构造器
            # 对 _dtype='U1' 只做重标注，会丢 _str_dtype/_raw_data）
            _res = _rn_r.array(_out, dtype=_rdt)
            if shp and axis is not None and _res.ndim < len(shp):
                _res = _res.reshape(tuple(shp))
            return _res
        result = arr._array.repeat(repeats, axis)
        return _wrap_result(result)
    
    @staticmethod
    def nonzero(arr):
        """返回非零元素的索引。"""
        ndarray, _ = _get_deps()
        raw = _core.nonzero_arrs(arr._array)
        return tuple(ndarray._wrap(r) for r in raw)

import builtins
import itertools
import math

"""数组操作模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnumpy.num_core as _core


def _nd():
    from .__init__ import ndarray as _n
    return _n


def _np():
    import rsnumpy as _m
    return _m


def _wrap(result):
    return _nd()(result)


def _scalar_or_wrap(a):
    """Python 标量按 numpy 语义保 dtype（bool→bool、int→int64）；
    ndarray 构造器默认 float64（Rust f64 存储），直接 nd() 会丢 int 语义。"""
    if isinstance(a, builtins.bool):
        return _nd()(a, _dtype='bool')
    if isinstance(a, builtins.int):
        return _nd()(a, _dtype='int64')
    return _wrap(a)


def _nested_to_list(x):
    """Recursively turn nested arrays / lists / tuples into pure nested lists."""
    if hasattr(x, '_array'):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_nested_to_list(i) for i in x]
    return x


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
    if isinstance(a, (list, tuple)):
        # Rust core cannot parse lists that contain array objects; promote
        # them to pure nested lists first (e.g. concatenate([[X[0]], [X[1]]])).
        return _core.ndarray(_nested_to_list(a))
    return _core.ndarray(a)


def _scalar_idx(x):
    """标量下标/形状参数：Python int/bool 或 rsnumpy 整数标量（非序列/数组）。
    rsnumpy 整型标量不再是 int 子类，需用 __index__ 协议识别。"""
    if isinstance(x, int):
        return True
    return (hasattr(x, '__index__') and not hasattr(x, '__iter__')
            and not hasattr(x, '_array'))


def reshape(a, shape=None, order='C', *, newshape=None, copy=None):
    """改变数组形状而不改变数据。

    shape 为新形状（整数或整数元组）。newshape 为已弃用的旧参数名，
    order 支持 'C'（行优先）与 'F'（列优先）。
    """
    if shape is None:
        shape = newshape
    if shape is None:
        raise TypeError("reshape() missing 1 required positional argument: 'shape'")
    if isinstance(shape, int) or _scalar_idx(shape):
        shape = (int(shape),)
    else:
        shape = tuple(shape)
    arr = a if hasattr(a, '_array') else _wrap(a)
    _tot = 1
    for _s in shape:
        _tot *= int(_s)
    if _tot == 0:
        # Rust _core.reshape 拒绝零尺寸：直接构造同 dtype 零尺寸数组
        # （复用非零 raw 占位 + shape 覆盖会导致 tolist 读取不一致）。
        return _np().array([], dtype=getattr(arr, '_dtype', 'float64'))
    if order == 'F':
        flat = arr.ravel(order='F')
        out = _wrap(_core.reshape(flat._array, shape))
        # numpy 语义:F 序输入 + F 序 reshape = 视图（保血缘）;
        # C 序输入 + F 序 reshape = 新拷贝（断血缘,shares_memory False）。
        # numpy 语义:copy=False 要求视图,无法视图时必须拷贝 → ValueError;
        # copy=None 时 F 序输入 = 视图（保血缘）,C 序输入 = 新拷贝（断血缘）。
        _in_f = getattr(arr, '_rs_order', None) == 'F'
        if copy is False and not _in_f:
            raise ValueError("Unable to avoid creating a copy while reshaping.")
        if _in_f:
            try:
                out._rs_parent = arr
                out._rs_order = 'F'
            except Exception:
                pass
        else:
            try:
                out._rs_parent = None
            except Exception:
                pass
            if getattr(out, '_raw_data', None) is getattr(arr, '_raw_data', None):
                try:
                    out._raw_data = None
                except Exception:
                    pass
    else:
        _in_c = getattr(arr, '_rs_order', None) != 'F'
        if copy is False and not _in_c:
            raise ValueError("Unable to avoid creating a copy while reshaping.")
        out = _wrap(_core.reshape(_ensure_raw(a), shape))
        try:
            out._rs_parent = arr
        except Exception:
            pass
    if copy is True:
        # numpy 语义:copy=True 恒返回新数组(不与输入共享内存)
        out = out.copy()
        try:
            out._rs_parent = None
        except Exception:
            pass
    return out


def ravel(a, order='C'):
    """展平数组。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    return arr.ravel(order=order)


def moveaxis(a, source, destination):
    """将数组的轴从 source 移动到 destination（numpy 语义，纯 Python 实现）。

    校验与 numpy normalize_axis_tuple 一致：AxisError（source/destination
    越界）、ValueError（repeated axis、source/destination 数量不一致）。
    非 ndarray 输入（MaskedArray/list）计算后转换回原类型。
    """
    np = _np()
    if getattr(a, 'ndim', None) is None:
        a = np.array(a)
    ndim = a.ndim

    def _norm(axes, what):
        if isinstance(axes, int) or _scalar_idx(axes):
            axes = [int(axes)]
        else:
            axes = [int(x) for x in axes]
        if len(set(axes)) != len(axes):
            raise ValueError(f"repeated axis in `{what}`")
        for ax in axes:
            if not (-ndim <= ax < ndim):
                raise np.AxisError(
                    f"{what} axis {ax} is out of bounds for array "
                    f"of dimension {ndim}")
        return axes

    source = _norm(source, 'source')
    destination = _norm(destination, 'destination')
    if len(source) != len(destination):
        raise ValueError("source and destination must have the same number of values")
    # numpy 官方算法（先归一化非负）：剩余轴 + 按 destination 位置插入 source 轴
    source_n = [s % ndim for s in source]
    dest_n = [d % ndim for d in destination]
    order = [n for n in range(ndim) if n not in source_n]
    for dest, src in sorted(zip(dest_n, source_n)):
        order.insert(dest, src)
    if isinstance(a, _nd()):
        return a.transpose(order)
    # 非 ndarray（MaskedArray/list 等）：ndarray 计算后转回原类型
    result = np.array(a).transpose(order)
    if not isinstance(a, (list, tuple)):
        try:
            result = type(a)(result)
        except Exception:
            pass
    return result


def rollaxis(a, axis, start=0):
    """滚动指定轴到 start 位置（numpy rollaxis 语义）。

    与 moveaxis 的区别（numpy 语义）：start 可取 [ -ndim, ndim ]；
    归一化后若 start > axis 则 start -= 1（轴"越过自己"时不占位）。
    等价：moveaxis(a, axis, start if start <= axis else start - 1)。
    """
    np = _np()
    arr = a if getattr(a, 'ndim', None) is not None else np.array(a)
    ndim = arr.ndim
    if not (-ndim <= axis < ndim):
        raise np.AxisError(
            f"axis {axis} is out of bounds for array of dimension {ndim}")
    if not (-ndim <= start <= ndim):
        raise np.AxisError(
            f"start {start} is out of bounds for array of dimension {ndim}")
    axis_n = axis % ndim
    # numpy 语义:start 归一化到 [0, ndim]——负 start 按 +ndim 折算
    # (-1 → ndim-1),而非模 ndim+1(那会把 -1 错映成 ndim)。
    start_n = start + ndim if start < 0 else start
    if start_n > axis_n:
        start_n -= 1
    return moveaxis(arr, axis_n, start_n)


def broadcast_to(a, shape, subok=None, device=None):
    """将数组广播到新形状。subok/device 仅为兼容 numpy/array-api 签名，忽略。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    # Handle broadcasting to scalar shape ()
    if shape == () or shape == []:
        result = arr._array  # 0-d array is already scalar-shaped
    else:
        _shp = ((int(shape),) if (isinstance(shape, int) or _scalar_idx(shape))
                else tuple(int(s) for s in shape))
        _tot = 1
        for _s in _shp:
            _tot *= _s
        if _tot > 2**28:  # 2GB 量级：Rust 物化会 OOM / capacity overflow abort
            # 超大广播：零分配 Python 层视图（Rust 物化会 OOM / capacity overflow
            # abort 进程）。numpy 语义：broadcast_to 从不分配；物化时抛 MemoryError。
            res = nd._wrap(arr._array, _dtype=dtype, _fields=fields, _raw_data=raw_data)
            res._rs_shape_override = _shp
            return res
        result = _core.broadcast_to(arr._array, shape)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def transpose(a, axes=None):
    """转置数组。axes 为 None 时反转所有轴，否则按给定顺序置换。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    result = _core.transpose_axes(arr._array, axes)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def swapaxes(a, axis1, axis2):
    """交换数组的两个轴。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    result = _core.swapaxes(arr._array, axis1, axis2)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def _flat_to_nested(flat, shape):
    """扁平列表按 C 序重塑为嵌套列表（插 1 维不改 C 序 flat，掩码用）。"""
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    _sub = 1
    for _x in shape[1:]:
        _sub *= _x
    return [_flat_to_nested(flat[_i * _sub:(_i + 1) * _sub], shape[1:])
            for _i in range(shape[0])]


def expand_dims(a, axis):
    """扩展数组的形状。"""
    # ma.MaskedArray 是独立类（非 ndarray 子类）：numpy subok 语义 →
    # 数据与掩码一起扩展，重建为 MaskedArray（插 1 维不改 C 序 flat）。
    _MA = getattr(getattr(_np(), 'ma', None), 'MaskedArray', None)
    if _MA is not None and type(a) is _MA:
        _ed = expand_dims(a._data, axis)
        _m = _flat_to_nested(list(a._mask), tuple(getattr(_ed, 'shape', ())))
        return _MA(_ed, mask=_m)
    arr = a if hasattr(a, '_array') else _scalar_or_wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    _shp = tuple(getattr(arr, 'shape', ()))
    axes = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
    # numpy 官方语义（_shape_base_impl.expand_dims）：axis 在
    # out_ndim = len(axis)+ndim 上归一化；重复 axis → ValueError；
    # 越界 → AxisError（勿让越界值直达 Rust 崩溃）。
    out_ndim = len(axes) + len(_shp)
    _norm = []
    for _ax in axes:
        _n = _ax + out_ndim if _ax < 0 else _ax
        if not 0 <= _n < out_ndim:
            from rsnumpy.exceptions import AxisError as _AxisError
            raise _AxisError(
                f"axis {_ax} is out of bounds for array of dimension {len(_shp)}")
        _norm.append(_n)
    if len(set(_norm)) != len(_norm):
        raise ValueError(f"repeated axis {axes}")
    # 保持子类型（numpy subok 语义）；matrix 除外——matrix 恒为 2-d，
    # expand_dims 必须降级为 ndarray（numpy 语义）。
    _t = type(arr)
    if _t is not nd and hasattr(_t, '_wrap') and \
            _t is not getattr(_np(), 'matrix', None):
        _cls = _t
    else:
        _cls = nd
    if len(axes) == 1:
        result = _core.expand_dims(arr._array, axes[0])
    else:
        _rs = list(_shp)
        for _n in sorted(_norm):
            _rs.insert(_n, 1)
        result = _core.reshape(arr._array, tuple(_rs))
    return _cls._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def squeeze(a, axis=None):
    """从数组形状中删除单维度条目（axis 指定时只删除指定轴）。"""
    # list 输入（可能含 ndarray 元素）：先递归转纯嵌套列表再构造，
    # 使 np.squeeze([np.zeros((3, 1))]) 得到 (1, 3, 1) 而非 object 数组
    arr = a if hasattr(a, '_array') else _nd()(_nested_to_list(a))
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    if axis is None:
        result = _core.squeeze(arr._array)
        return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)
    # numpy 语义：只删除指定轴（该轴大小须为 1）
    shape = tuple(arr.shape)
    ndim = len(shape)
    if isinstance(axis, int):
        _axes = (axis,)
    else:
        _axes = tuple(axis)
    _axes = tuple(_x % ndim for _x in _axes)
    new_shape = []
    for _i, _s in enumerate(shape):
        if _i in _axes:
            if _s != 1:
                raise ValueError(
                    "cannot select an axis to squeeze out which has size "
                    "not equal to one")
        else:
            new_shape.append(_s)
    result = arr.reshape(tuple(new_shape))
    return nd._wrap(result._array, _dtype=dtype, _fields=fields,
                    _raw_data=raw_data)


_CASTING_VALUES = ("no", "equiv", "safe", "same_kind", "unsafe")


def _check_casting(casting):
    """numpy 语义：casting 合法值校验（非法值抛 ValueError）。"""
    if casting is not None and casting not in _CASTING_VALUES:
        raise ValueError(
            "casting must be one of 'no', 'equiv', 'safe', 'same_kind', or "
            f"'unsafe'; got {casting!r}")


def _apply_dtype(res, dtype):
    """显式 dtype 转换；非 native 字节序经 _dtype_obj 保留（astype 同款）。"""
    _name = _dtype_name(dtype)
    # S/U 类目标：astype 不支持字符串存储，经 array 构造重建
    _kind = None
    try:
        from .__init__ import dtype as _dtype_fn
        _kind = getattr(_dtype_fn(dtype), 'kind', None)
    except Exception:
        pass
    if _kind in ('S', 'U'):
        from .__init__ import array as _array_fn
        try:
            _spec = str(_dtype_fn(dtype))
        except Exception:
            _spec = _name
        # 值先字符串化：rsnumpy 的 S-dtype 数组对数值输入保留数值占位存储，
        # 不转会让 out 路径与 dtype 路径（真字符串）不一致
        try:
            _flat = res.tolist()
            if not isinstance(_flat, list):
                _flat = [_flat]

            def _flatten_all(x):
                out = []
                for v in x:
                    if isinstance(v, list):
                        out.extend(_flatten_all(v))
                    else:
                        out.append(v)
                return out

            _flat = _flatten_all(_flat)
            return _array_fn([str(v) for v in _flat], dtype=_spec)
        except Exception:
            return _array_fn(res.tolist(), dtype=_spec)
    res = res.astype(_name)
    try:
        from .__init__ import dtype as _dtype_fn
        import rsnumpy._dtypes as _dts
        _dbo = getattr(_dtype_fn(dtype), 'byteorder', None)
        if isinstance(_dbo, str) and _dbo not in ('|', '=', '@', 'n', 'native'):
            _d = _dts.DType(_name)
            res._dtype_obj = _dts._clone_with_byteorder(_d, _dbo)
        elif _name == 'complex64' and str(getattr(res, '_dtype', '')) != 'complex64':
            # astype 复数路径强制 complex128 标签；按 np.array 语义重挂 complex64
            res._dtype = 'complex64'
    except Exception:
        pass
    return res


def _check_cast_allowed(arr_list, dtype, casting):
    """numpy casting 语义：non-unsafe 转换需 can_cast 许可，否则 TypeError。

    numpy 检查**所有**输入数组（不止第一个）——如 stack((int, float),
    dtype=int64, casting='safe') 必须因 float→int64 被拒。arr_list 可传单个
    数组或数组列表。"""
    if casting == "unsafe" or casting is None:
        return
    try:
        from .__init__ import dtype as _dtype_fn, can_cast as _can_cast
        _dst_dt = _dtype_fn(dtype)
        _dst = _dst_dt.name
        _srcs = arr_list if isinstance(arr_list, (builtins.list, builtins.tuple)) \
            else [arr_list]
        # 纯 Python 标量走 copyto 标量语义（pyscalar casting），不参与 can_cast
        for _arr in _srcs:
            if not hasattr(_arr, '_dtype'):
                continue
            _src = getattr(_arr, '_dtype', 'float64')
            if _src in ('object', 'O'):
                continue  # object 数组可自由转换（numpy 语义）
            try:
                _src_kind = getattr(_dtype_fn(_src), 'kind', None)
            except Exception:
                _src_kind = None
            if _src_kind in ('S', 'U') and getattr(_dst_dt, 'kind', None) in ('S', 'U'):
                continue  # 字符串类别间转换自由（string_→U5 等名称差异不算降级）
            # 传 DType 实例（保留字节序）：'no' 模式须字节序敏感（f8→>f8 拒）
            if not _can_cast(_src, _dst_dt, casting=casting):
                raise TypeError(
                    f"Cannot cast array data from dtype('{_src}') to "
                    f"dtype('{_dst}') according to the rule '{casting}'")
    except TypeError:
        raise
    except Exception:
        pass


def _dtype_name(dtype):
    """把 dtype 参数解析为 dtype 名字符串（np.float32 类/DType 实例/字符串）。"""
    from .__init__ import dtype as _dtype_fn
    try:
        return _dtype_fn(dtype).name
    except Exception:
        return builtins.str(dtype)


def _str_parses(s):
    """字符串能否解析为数值（copyto 宽松转换的前置判定）。"""
    try:
        builtins.float(s)
        return True
    except Exception:
        return False


def _out_dtype_of(out):
    """取 out 数组的富 dtype（_dtype_obj 优先，否则存储标签）。"""
    dt = getattr(out, '_dtype_obj', None)
    if dt is None:
        dt = getattr(out, '_dtype', 'float64')
    return dt


def _finalize_out(res, out, arr_list, casting):
    """stack 家族与 concatenate 共用的 out= 收尾：
    ① casting 许可检查（TypeError）；② 写入前按 out.dtype 精度转换
    （如 f32 舍入）；③ S/U 类字符串 out 经 _raw_data 直搬（浮点 setitem
    写不进字符串占位存储）；④ out 与 dtype 同传由调用方先行抛 TypeError。"""
    _out_dt = _out_dtype_of(out)
    if not hasattr(_out_dt, 'kind') and _out_dt in ('string_', 'bytes_'):
        # 字符串数组构造（_string_filled）只带 _str_dtype 宽度信息：
        # 还原显式 spec（('S',4)→'S4'），避免裸 'string_' 探出 S3
        _sd = getattr(out, '_str_dtype', None)
        if _sd and _sd[0] in ('S', 'U') and isinstance(_sd[1], int):
            _out_dt = f'{_sd[0]}{_sd[1]}'
    _check_cast_allowed(arr_list, _out_dt, casting)
    # numpy 语义：out 形状与结果不一致 → ValueError（不可静默吞掉）
    try:
        _res_shape = builtins.tuple(res.shape)
    except Exception:
        _res_shape = None
    if _res_shape is not None \
            and builtins.tuple(out.shape) != _res_shape:
        raise ValueError(
            f"output array is not the expected shape "
            f"{_res_shape}, got {builtins.tuple(out.shape)}")
    try:
        res = _apply_dtype(res, _out_dt)
    except Exception:
        pass
    _handled = False
    try:
        from .__init__ import dtype as _dtype_fn
        if getattr(_dtype_fn(_out_dt), 'kind', None) in ('S', 'U'):
            _vals = getattr(res, '_raw_data', None)
            if _vals is not None:
                out._raw_data = list(_vals)
                out._str_dtype = getattr(res, '_str_dtype', None)
                _handled = True
    except Exception:
        _handled = False
    if not _handled:
        try:
            out[...] = res
        except Exception:
            pass
    return out


def concatenate(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    """沿着指定轴连接数组（支持 numpy 的 out/dtype 参数）。"""
    _check_casting(casting)
    if out is not None and dtype is not None:
        raise TypeError(
            "concatenate() only takes `out` or `dtype` as an argument, "
            "not both")
    # 子数组 dtype（如 '(2,)i'，DType 解析为 kind 'V' + shape）不支持拼接
    if dtype is not None:
        try:
            from .__init__ import dtype as _dtype_fn
            _pdt = _dtype_fn(dtype)
            if getattr(_pdt, 'kind', None) == 'V' \
                    and builtins.tuple(getattr(_pdt, 'shape', ()) or ()):
                raise TypeError(
                    f"concatenate() does not support subarray dtype "
                    f"{builtins.str(dtype)!r}")
        except TypeError:
            raise
        except Exception:
            pass
    # numpy 语义：len 超过 intc 上限须在迭代（访问任何元素）之前抛
    try:
        _max_intc = (1 << 31) - 1
        _arr_len = builtins.len(arrays)
        if _arr_len > _max_intc:
            raise ValueError(
                f"concatenate() only supports up to {_max_intc} arrays "
                f"but got {_arr_len}.")
    except ValueError:
        raise
    except Exception:
        pass
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("concatenate requires at least one array")
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    # S/U 目标 dtype：Rust concatenate 不支持字符串存储，改走 Python
    # 字符串重建。numpy 语义：裸 'S'/'U' 的宽度 = double 转字符串的宽度
    # （测试注释 "identical to a cast (of a double array)" → 'S3'），
    # 显式宽度（'S0'/'S4'）直接用（np.array 自带截断）。
    _su_spec = None
    _su_zero_width = False
    if dtype is not None:
        try:
            from .__init__ import dtype as _dtype_fn
            _pd = _dtype_fn(dtype)
            if getattr(_pd, 'kind', None) in ('S', 'U'):
                _su_spec = str(dtype)
                _kind = getattr(_pd, 'kind', None)
                _its = getattr(_pd, 'itemsize', None)
                # DType 实例带宽度（S4 itemsize=4 / U5 itemsize=20=5×4）→
                # 还原显式宽度 spec，避免裸 'string_' 语义探出 S3
                if isinstance(_its, int) and _its > 0:
                    if _kind == 'S':
                        _su_spec = f'S{_its}'
                    elif _kind == 'U' and _its % 4 == 0:
                        _su_spec = f'U{_its // 4}'
                # 裸 'S'/'U' 与零宽度 'S0'/'U0' 同规：宽度 = double 转字符串
                # 的宽度（numpy 测试语义 "identical to a cast of a double
                # array" → 'S3'），而非按数据推断
                if _su_spec.lstrip('<>|=')[1:] in ('', '0'):
                    _su_zero_width = True
                    from .__init__ import array as _array_fn
                    _probe = _array_fn([1.0], dtype=_su_spec)
                    _su_spec = str(_probe.dtype)
        except Exception:
            _su_spec = None
    if _su_spec is not None:
        from .__init__ import array as _array_fn
        # casting 检查先行（如 no/equiv/safe 下 float64→S4 必须 TypeError）；
        # 零宽度目标豁免——numpy 对裸 'U'/'S' 的数值→字符串转换按 safe 对待
        if not _su_zero_width:
            _check_cast_allowed(arr_list, dtype, casting)
        _vals = []

        def _flatten_all(x):
            out = []
            for v in x:
                if isinstance(v, list):
                    out.extend(_flatten_all(v))
                else:
                    out.append(v)
            return out

        for a in arr_list:
            if isinstance(a, (builtins.list, builtins.tuple)):
                # Python list 元素保留原始值（int 0 → '0' 而非数值化 '0.0'）
                _vals.extend(_flatten_all(builtins.list(a)))
                continue
            arr = a if hasattr(a, '_array') else _wrap(a)
            _t = arr.tolist()
            _vals.extend(_flatten_all(_t) if isinstance(_t, list) else [_t])
        # numpy 语义：dtype='S' 不检查/字符串化 None → TypeError
        if builtins.any(v is None for v in _vals):
            raise TypeError(
                "cannot concatenate None values with a string dtype")
        _strs = [str(v) for v in _vals]
        return _array_fn(_strs, dtype=_su_spec)
    nd = _nd()
    _dt = dtype if dtype is not None else getattr(first, '_dtype', "float64")
    # numpy 语义：0-d 数组不能直接拼接（"Scalars cannot be concatenated"）
    if axis is not None and builtins.any(
            (hasattr(a, '_array') and getattr(a, 'ndim', 0) == 0)
            or (not hasattr(a, '_array') and not isinstance(
                a, (builtins.list, builtins.tuple, builtins.str)))
            for a in arr_list):
        raise ValueError("zero-dimensional arraylets cannot be concatenated")
    # numpy 语义：ndim 不一致时报 "all the input arrays must have same
    # number of dimensions..."（Rust 层只有笼统的 incompatible shapes）
    if axis is not None:
        _ndims = []
        for _a in arr_list:
            _wa = _a if hasattr(_a, '_array') else _wrap(_a)
            _ndims.append(getattr(_wa, 'ndim', 0))
        if _ndims and builtins.len(builtins.set(_ndims)) > 1:
            _i1 = builtins.next(
                (i for i, d in builtins.enumerate(_ndims) if d != _ndims[0]), 1)
            raise ValueError(
                "all the input arrays must have same number of dimensions, "
                f"but the array at index 0 has {_ndims[0]} dimension(s) "
                f"and the array at index {_i1} has {_ndims[_i1]} dimension(s)")
        # 同 ndim 但拼接轴外维度尺寸不一致 → numpy 格式消息
        if axis is not None and _ndims and builtins.len(
                builtins.set(_ndims)) == 1:
            _ax = axis % max(_ndims[0], 1)
            _shapes = [
                builtins.tuple(
                    (_a if hasattr(_a, '_array') else _wrap(_a)).shape)
                for _a in arr_list]
            for _i in range(1, builtins.len(_shapes)):
                for _d in range(_ndims[0]):
                    if _d == _ax:
                        continue
                    if _shapes[_i][_d] != _shapes[0][_d]:
                        raise ValueError(
                            "all the input array dimensions except for the "
                            "concatenation axis must match exactly, but along "
                            f"dimension {_d}, the array at index 0 has size "
                            f"{_shapes[0][_d]} and the array at index {_i} "
                            f"has size {_shapes[_i][_d]}")
    # 负 axis 归一化：越界直达 Rust 会 abort，须在 Python 边界校验
    if axis is not None:
        from .exceptions import AxisError
        _first_ndim = getattr(first, 'ndim', 0)
        if axis < 0:
            axis += _first_ndim
        if not 0 <= axis < _first_ndim:
            raise AxisError(
                f'axis {axis} is out of bounds for array of '
                f'dimension {_first_ndim}')
    # copyto 语义对齐：str 标量元素按宽松转换（无法解析按 0），
    # 与 np.copyto(dst, 'x', casting=...) 成功的行为一致
    if builtins.any(isinstance(a, builtins.str) for a in arr_list):
        arr_list = [
            (builtins.float(a) if _str_parses(a) else 0.0)
            if isinstance(a, builtins.str) else a
            for a in arr_list
        ]
    # numpy 语义：pyscalar 越界（如 300→int8、2**64→int64）构造时抛
    # OverflowError——仅在无显式 dtype= 时（目标由输入推断）；显式 dtype=
    # 时标量走 copyto 宽松语义（wrap，不抛）。first 为标量时推断出
    # float64 不可信，out 的 dtype 也参与检查。
    if dtype is None and casting != "unsafe":
        try:
            from .__init__ import _INT_RANGES, dtype as _dtype_fn
            _tgt_candidates = [_dt]
            if out is not None:
                _tgt_candidates.append(_out_dtype_of(out))
            for _tgt_dt in _tgt_candidates:
                _tgt_name = getattr(_dtype_fn(_tgt_dt), 'name', None) \
                    if not isinstance(_tgt_dt, str) else _tgt_dt
                if _tgt_name not in _INT_RANGES:
                    continue
                _lo, _hi = _INT_RANGES[_tgt_name]
                for _a in arr_list:
                    if isinstance(_a, builtins.int) \
                            and not isinstance(_a, builtins.bool) \
                            and not (_lo <= _a <= _hi):
                        raise OverflowError(
                            f"Python int too large to convert to {_tgt_name}")
        except OverflowError:
            raise
        except Exception:
            pass
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [_ensure_raw(a) for a in arr_list]
    if axis is None:
        # numpy 语义：axis=None 时各数组展平后做一维拼接
        flat_list = []
        for a in arr_list:
            arr = a if hasattr(a, '_array') else _wrap(a)
            flat_list.append(_ensure_raw(arr.reshape((arr.size,))))
        result = _core.concatenate(flat_list, 0)
    else:
        result = _core.concatenate(raw_list, axis)
    res = nd._wrap(result, _dtype=getattr(first, '_dtype', "float64"),
                   _fields=fields, _raw_data=raw_data)
    if dtype is not None:
        _check_cast_allowed(arr_list, dtype, casting)
        res = _apply_dtype(res, dtype)
    if out is not None:
        return _finalize_out(res, out, arr_list, casting)
    return res


# numpy 的 __array_function__ dispatcher 会访问 ._implementation 直调实现
concatenate._implementation = concatenate


def stack(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    """沿着新轴堆叠数组（支持 numpy 的 out/dtype 参数）。"""
    _check_casting(casting)
    if out is not None and dtype is not None:
        raise TypeError(
            "stack() only takes `out` or `dtype` as an argument, not both")
    # numpy 语义：ndarray 本身是合法序列（按其元素堆叠）；标量/生成器拒绝
    if not isinstance(arrays, (builtins.list, builtins.tuple)) \
            and not hasattr(arrays, '_array'):
        raise TypeError(
            f"arrays to stack must be array_like sequences, not "
            f"{builtins.type(arrays).__name__}")
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("need at least one array to stack")
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    nd = _nd()
    _dt0 = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [_ensure_raw(a) for a in arr_list]
    # axis 归一化与越界校验（越界直接传 Rust 会 abort，不可捕获）：
    # numpy 允许 axis ∈ [-(nd+1), nd]，结果 ndim = 输入 nd+1
    from .exceptions import AxisError
    _in_ndim = getattr(first, 'ndim', 0)
    if axis < 0:
        axis += _in_ndim + 1
    if not 0 <= axis <= _in_ndim:
        raise AxisError(f'axis {axis} is out of bounds for array of '
                        f'dimension {_in_ndim + 1}')
    result = _core.stack(raw_list, axis)
    res = nd._wrap(result, _dtype=_dt0, _fields=fields, _raw_data=raw_data)
    if dtype is not None:
        _check_cast_allowed(arr_list, dtype, casting)
        res = _apply_dtype(res, dtype)
    if out is not None:
        return _finalize_out(res, out, arr_list, casting)
    return res


def vstack(arrays, out=None, dtype=None, casting="same_kind"):
    """垂直堆叠数组（沿行）。"""
    _check_casting(casting)
    if out is not None and dtype is not None:
        raise TypeError(
            "vstack() only takes `out` or `dtype` as an argument, not both")
    # numpy 语义：ndarray 本身是合法序列；标量/生成器拒绝
    if not isinstance(arrays, (builtins.list, builtins.tuple)) \
            and not hasattr(arrays, '_array'):
        raise TypeError(
            f"arrays to stack must be array_like sequences, not "
            f"{builtins.type(arrays).__name__}")
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("vstack requires at least one array")
    # numpy 语义：0-d 数组先按 (1,) 展平再堆叠
    for _i, _a in builtins.enumerate(arr_list):
        if hasattr(_a, '_array') and getattr(_a, 'ndim', 0) == 0:
            arr_list[_i] = _a.reshape((1,))
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    nd = _nd()
    dtype0 = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [_ensure_raw(a) for a in arr_list]
    result = _core.vstack(raw_list)
    res = nd._wrap(result, _dtype=dtype0, _fields=fields, _raw_data=raw_data)
    if dtype is not None:
        _check_cast_allowed(arr_list, dtype, casting)
        res = _apply_dtype(res, dtype)
    if out is not None:
        return _finalize_out(res, out, arr_list, casting)
    return res


def hstack(arrays, out=None, dtype=None, casting="same_kind"):
    """水平堆叠数组（沿列）。"""
    _check_casting(casting)
    if out is not None and dtype is not None:
        raise TypeError(
            "hstack() only takes `out` or `dtype` as an argument, not both")
    # numpy 语义：ndarray 本身是合法序列；标量/生成器拒绝
    if not isinstance(arrays, (builtins.list, builtins.tuple)) \
            and not hasattr(arrays, '_array'):
        raise TypeError(
            f"arrays to stack must be array_like sequences, not "
            f"{builtins.type(arrays).__name__}")
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("hstack requires at least one array")
    # numpy 语义：0-d 数组先按 (1,) 展平再堆叠
    for _i, _a in builtins.enumerate(arr_list):
        if hasattr(_a, '_array') and getattr(_a, 'ndim', 0) == 0:
            arr_list[_i] = _a.reshape((1,))
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    nd = _nd()
    dtype0 = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [_ensure_raw(a) for a in arr_list]
    result = _core.hstack(raw_list)
    res = nd._wrap(result, _dtype=dtype0, _fields=fields, _raw_data=raw_data)
    if dtype is not None:
        _check_cast_allowed(arr_list, dtype, casting)
        res = _apply_dtype(res, dtype)
    if out is not None:
        return _finalize_out(res, out, arr_list, casting)
    return res


def dstack(arrays):
    """沿深度方向堆叠数组（numpy 语义：1-D→(1,n,1)、2-D→(m,n,1) 后
    沿新 axis=2 拼接）。"""
    if not isinstance(arrays, (list, tuple)) and not hasattr(arrays, '_array'):
        raise TypeError("arrays to stack must be array_like or sequences")
    arr_list = [_ensure_raw(a) if not hasattr(a, '_array') else a
                for a in arrays]
    if not arr_list:
        raise ValueError("dstack requires at least one array")
    first = arr_list[0]
    nd = _nd()
    dtype = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    # numpy 语义：先统一升到 3-D（避免对低维数组沿 axis=2 拼接越界）
    lifted = []
    for a in arr_list:
        sh = tuple(a.shape)
        if len(sh) == 1:
            a = a.reshape((1,) + sh + (1,))
        elif len(sh) == 2:
            a = a.reshape(sh + (1,))
        elif len(sh) == 0:
            a = a.reshape((1, 1, 1))
        lifted.append(a)
    result = _core.concatenate([_ensure_raw(a) for a in lifted], 2)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def column_stack(arrays):
    """将 1-D 数组作为列堆叠成 2-D 数组。"""
    if not isinstance(arrays, (list, tuple)) and not hasattr(arrays, '_array'):
        raise TypeError("arrays to stack must be array_like or sequences")
    arr_list = list(arrays)
    if not arr_list:
        raise ValueError("column_stack requires at least one array")
    first = arr_list[0] if hasattr(arr_list[0], '_array') else _wrap(arr_list[0])
    nd = _nd()
    dtype = getattr(first, '_dtype', "float64")
    fields = getattr(first, '_fields', None)
    raw_data = getattr(first, '_raw_data', None)
    raw_list = [_ensure_raw(a) for a in arr_list]
    result = _core.column_stack(raw_list)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def split(ary, indices_or_sections, axis=0):
    """将数组拆分为多个子数组（使用 Rust 实现）。"""
    arr = ary if hasattr(ary, '_array') else _wrap(ary)
    # numpy 语义：整数份数不能整除该轴长度时抛 ValueError
    if isinstance(indices_or_sections, (int, builtins.int)) and not isinstance(indices_or_sections, bool):
        axis_len = arr.shape[axis] if arr.ndim > axis else 0
        if axis_len % indices_or_sections != 0:
            raise ValueError(
                "array split does not result in an equal division")
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    result_raw = _core.split_rs(arr._array, indices_or_sections, axis)
    return [nd._wrap(r, _dtype=dtype, _fields=fields, _raw_data=raw_data) for r in result_raw]


def hsplit(ary, indices_or_sections):
    """水平拆分数组（numpy 语义：1-D 数组沿唯一轴 axis=0 拆）。"""
    arr = ary if hasattr(ary, '_array') else _wrap(ary)
    if arr.ndim == 1:
        return split(ary, indices_or_sections, axis=0)
    return split(ary, indices_or_sections, axis=1)


def vsplit(ary, indices_or_sections):
    """垂直拆分数组（numpy 语义：1-D 数组无轴 1，抛 ValueError）。"""
    arr = ary if hasattr(ary, '_array') else _wrap(ary)
    if arr.ndim < 2:
        raise ValueError(
            "vsplit only works on arrays of 2 or more dimensions")
    return split(ary, indices_or_sections, axis=0)


def dsplit(ary, indices_or_sections):
    """沿深度拆分数组。"""
    return split(ary, indices_or_sections, axis=2)


def tile(A, reps):
    """重复数组（numpy 语义：reps 短于 ndim 时前置补 1，即沿末轴重复；
    长于 ndim 时前导补新轴）。"""
    arr = A if hasattr(A, '_array') else _wrap(A)
    if isinstance(reps, int) or _scalar_idx(reps):
        reps = [int(reps)]
    reps = [int(r) for r in reps]
    sh = tuple(arr.shape)
    nd = len(sh)
    if len(reps) < nd:
        reps = [1] * (nd - len(reps)) + reps
    out = arr
    if len(reps) > nd:
        out = out.reshape((1,) * (len(reps) - nd) + sh)
        nd = len(reps)
    # 空数组：直接构造目标形状的空结果
    if getattr(arr, 'size', 0) == 0:
        new_sh = tuple(r * s for r, s in zip(reps, tuple(out.shape)))
        return _np().zeros(new_sh, dtype=getattr(arr, '_dtype', None) or 'float64')
    for i, r in enumerate(reps):
        if r == 1:
            continue
        if r == 0:
            sl = [slice(None)] * nd
            sl[i] = slice(0, 0)
            out = out[tuple(sl)]
        else:
            out = _np().concatenate([out] * r, axis=i)
    return _wrap(_ensure_raw(out))


def repeat(a, repeats, axis=None):
    """重复数组元素。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    return arr.repeat(repeats, axis)


def flip(m, axis=None):
    """沿指定轴翻转数组。"""
    arr = m if hasattr(m, '_array') else _wrap(m)
    return _wrap(_core.flip(_ensure_raw(arr), axis))


def fliplr(m):
    """左右翻转数组。"""
    return flip(m, axis=1)


def flipud(m):
    """上下翻转数组。"""
    return flip(m, axis=0)


def roll(a, shift, axis=None):
    """沿指定轴滚动数组（numpy 语义）。

    axis=None：滚动整个展平数组；axis 为 list/tuple：各轴独立滚动
    （shift 可为标量广播或等长 tuple）。标量 shift 先在 Python 层按
    size 取模（size 是各轴长度的公倍数，(s % size) % len_i == s % len_i），
    规避 Rust 侧 Python 大整数 / uint64(>2^63) 转 C long 的溢出。
    """
    arr = a if hasattr(a, '_array') else _wrap(a)
    # 空数组 roll 是恒等映射：Rust 侧 shift % 0 会 panic（除零）。
    if getattr(arr, 'size', 0) == 0:
        return arr.copy()

    def _to_int(v):
        return int(v) if (isinstance(v, int) or _scalar_idx(v)) else v

    if isinstance(axis, (list, tuple)):
        axes = [_to_int(x) % arr.ndim for x in axis]
        if isinstance(shift, (list, tuple)):
            shifts = [_to_int(s) for s in shift]
        else:
            shifts = [_to_int(shift)] * len(axes)
        if len(shifts) != len(axes):
            raise ValueError("shift and axis must have the same length")
        result = arr
        for sh, ax in zip(shifts, axes):
            result = _roll_axis(result, sh, ax)
        return result
    if axis is None:
        # numpy 语义：滚动整个展平数组
        flat = arr.ravel()
        s = _to_int(shift)
        if s % flat.size != 0:
            r = _wrap(_core.roll(_ensure_raw(flat), s % flat.size, None))
            return r.reshape(arr.shape)
        return arr.copy()
    return _roll_axis(arr, _to_int(shift), _to_int(axis) % arr.ndim)


def _roll_axis(arr, shift, axis):
    """沿单轴滚动（Rust 实现 + Python 层取模）。"""
    size = arr.size
    if shift % size != 0:
        return _wrap(_core.roll(_ensure_raw(arr), shift % size, axis))
    return arr.copy()


def rot90(m, k=1, axes=(0, 1)):
    """将数组旋转 90 度。"""
    return _wrap(_core.rot90(_ensure_raw(m), k, axes[0], axes[1]))


def resize(a, new_shape):
    """返回指定形状的新数组，数据重复填充至新形状。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    if isinstance(new_shape, int) or _scalar_idx(new_shape):
        new_shape = (int(new_shape),)
    new_shape = [builtins.int(d) for d in new_shape]
    # numpy 语义：负尺寸非法
    if builtins.any(d < 0 for d in new_shape):
        raise ValueError("negative dimensions are not allowed")
    _size = 1
    for d in new_shape:
        _size *= d
    if _size == 0 or getattr(arr, 'size', 0) == 0:
        # 目标含 0 维或源为空数组 → 零填充（numpy 语义）
        _dt = getattr(arr, 'dtype', None)
        result = _np().zeros(tuple(new_shape), dtype=_dt)
    else:
        result = nd._wrap(_core.resize_rs(arr._array, list(new_shape)),
                          _dtype=dtype, _fields=fields, _raw_data=raw_data)
    # numpy 语义：resize 保留输入的子类类型
    if isinstance(a, nd) and builtins.type(a) is not nd:
        try:
            result = result.view(builtins.type(a))
        except Exception:
            pass
    return result


def append(arr, values, axis=None):
    """将值附加到数组末尾。"""
    arr = arr if hasattr(arr, '_array') else _wrap(arr)
    values_nd = values if hasattr(values, '_array') else _wrap(values)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)
    if axis is None:
        a_flat = arr.ravel()
        v_flat = values_nd.ravel()
        return concatenate((a_flat, v_flat))
    raw_list = [arr._array, values_nd._array]
    result = _core.concatenate(raw_list, axis)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def delete(arr, obj, axis=None):
    """删除指定位置的元素。"""
    arr = arr if hasattr(arr, '_array') else _wrap(arr)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    if isinstance(obj, slice):
        start, stop, step = obj.indices(arr.size if axis is None else arr.shape[axis])
        obj = list(range(start, stop, step))
    elif isinstance(obj, int) or _scalar_idx(obj):
        obj = [int(obj)]
    obj = sorted(set(obj))
    
    result = _core.delete_rs(arr._array, obj, axis)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def insert(arr, obj, values, axis=None):
    """沿指定轴在指定下标前插入值。"""
    arr = arr if hasattr(arr, '_array') else _wrap(arr)
    nd = _nd()
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    raw_data = getattr(arr, '_raw_data', None)

    if isinstance(obj, int):
        obj = [obj]

    if not hasattr(values, '_array'):
        if isinstance(values, (list, tuple)):
            values_nd = _nd()(values)
        else:
            values_nd = _nd()([values])
    else:
        values_nd = values

    values_flat = values_nd.ravel().tolist()

    # 检查数据中是否包含复数（即使 dtype 声明为 float64）
    def _contains_complex(data):
        """递归检查数据中是否包含复数"""
        if isinstance(data, complex):
            return True
        if isinstance(data, (list, tuple)):
            return any(_contains_complex(x) for x in data)
        return False

    # 处理复数数据：展平为 [real, imag, real, imag, ...]
    if dtype == 'complex128' or _contains_complex(values_flat):
        # 递归展平嵌套复数列表
        def _flatten_complex_nested(lst):
            result = []
            for x in lst:
                if isinstance(x, complex):
                    result.extend([x.real, x.imag])
                elif isinstance(x, (list, tuple)):
                    result.extend(_flatten_complex_nested(x))
                else:
                    result.extend([float(x), 0.0])
            return result

        # 列表推导 + 闭包，避免在循环中重复定义函数
        flat_real_imag = []
        for item in values_flat:
            if isinstance(item, complex):
                flat_real_imag.extend([item.real, item.imag])
            elif isinstance(item, (list, tuple)):
                flat_real_imag.extend(_flatten_complex_nested(item))
            else:
                flat_real_imag.extend([float(item), 0.0])
        values_flat = flat_real_imag

    result = _core.insert_rs(arr._array, obj, values_flat, axis)
    return nd._wrap(result, _dtype=dtype, _fields=fields, _raw_data=raw_data)


def unique(a, return_index=False, return_inverse=False, return_counts=False,
           axis=None, *, equal_nan=True, sorted=True):
    """查找数组内的唯一元素（纯 Python 实现，对齐 numpy 语义）。

    axis=None 展平后去重；axis=k 将该轴上的每个子数组视为一个元素去重
    （按元素字典序）。equal_nan=True 时全部 NaN 视为相等只留一个；
    False 时各 NaN 互不相等各占一席。return_inverse 在 axis=None 时
    按输入形状展平，axis=k 时为沿该轴的一维数组。
    """
    npmod = _np()
    nd = _nd()

    # MaskedArray 输入：对 data 计算 unique，结果再挂回掩码。
    _ma_input = None
    if (not hasattr(a, '_array')) and type(a).__name__ == 'MaskedArray':
        _ma_input = a
        a = a.data
    arr = a if hasattr(a, '_array') else _wrap(a)
    dtype = getattr(arr, '_dtype', "float64")
    fields = getattr(arr, '_fields', None)
    shape = tuple(getattr(arr, 'shape', ()))
    ndim = len(shape)

    # ---------- axis 校验（numpy 语义：非整数 -> TypeError，越界 -> AxisError） ----------
    if axis is not None:
        if not (isinstance(axis, int) and not isinstance(axis, bool)):
            if hasattr(axis, '__index__'):
                axis = axis.__index__()
            else:
                raise TypeError(
                    f"unique(): axis={axis!r} cannot be interpreted as an integer")
        if axis >= ndim or axis < -ndim:
            from .exceptions import AxisError
            raise AxisError(
                f"axis {axis} is out of bounds for array of dimension {ndim}")
        _ax = axis + ndim if axis < 0 else axis
    else:
        _ax = None

    if int(getattr(arr, 'size', 0)) == 0:
        # numpy 语义（issue 15559）：零尺寸数组沿轴的各切片全为空且相等；
        # Python 层处理避免 Rust 崩溃。
        _shp = shape
        _intp = "int64"
        if _ax is None:
            uniq = npmod.empty((0,), dtype=dtype)
            idx = inv = cnt = npmod.empty((0,), dtype=_intp)
        else:
            n_axis = _shp[_ax]
            if n_axis == 0:
                uniq = npmod.empty(_shp, dtype=dtype)
                idx = inv = cnt = npmod.empty((0,), dtype=_intp)
            else:
                _us = list(_shp)
                _us[_ax] = 1
                uniq = npmod.empty(tuple(_us), dtype=dtype)
                idx = npmod.empty((1,), dtype=_intp)
                idx[0] = 0
                inv = npmod.zeros((n_axis,), dtype=_intp)
                cnt = npmod.empty((1,), dtype=_intp)
                cnt[0] = n_axis
        if not (return_index or return_inverse or return_counts):
            return uniq
        outputs = [uniq]
        if return_index:
            outputs.append(idx)
        if return_inverse:
            outputs.append(inv)
        if return_counts:
            outputs.append(cnt)
        return tuple(outputs)

    # object 类型（及含 object 字段的结构化）不支持 axis 路径（numpy 抛 TypeError）。
    if _ax is not None:
        _obj = (dtype in ('object', 'O')
                or getattr(dtype, 'name', '') in ('object', 'O'))
        if not _obj and fields:
            for _nm, _fd in fields:
                if _fd in ('object', 'O') or 'object' in str(_fd):
                    _obj = True
                    break
        if _obj:
            raise TypeError("unique() with axis is not supported for object arrays")

    # ---------- 取行：axis=k 时行 = 沿轴的子数组（C 序下为连续块）；None 时每个标量一行 ----------
    def _flat_of(x):
        out = []

        def rec(v):
            if isinstance(v, (list, tuple)):
                for w in v:
                    rec(w)
            else:
                out.append(v)

        rec(x.tolist())
        return out

    if fields is not None:
        flat = [tuple(r) for r in arr.tolist()]
    else:
        flat = _flat_of(arr)
    if _ax is None:
        rows = [[v] for v in flat]
        n_rows = len(rows)
    else:
        # axis=k 的元素 = “固定第 k 轴位置的切片”（如 2D 数组 axis=1 是列），
        # C 序下不一定连续，按索引分解采样。
        _strides = [1] * ndim
        for _d in range(ndim - 2, -1, -1):
            _strides[_d] = _strides[_d + 1] * shape[_d + 1]
        n_rows = shape[_ax]
        rows = []
        for _p in range(n_rows):
            _row = []
            for _idx in itertools.product(*[range(_s) for _s in shape]):
                if _idx[_ax] == _p:
                    _row.append(
                        flat[sum(_i * _st for _i, _st in zip(_idx, _strides))])
            rows.append(_row)

    # ---------- NA / NaN 判定与相等性 ----------
    # StringDType（可空字符串）：NA 对象可为 None / 任意 NaN / 任意身份对象；
    # 字符串 na_object 按普通字符串值处理（无 NA 语义）。
    _vstr_dt = getattr(arr, '_dtype_obj', None)
    _is_vstr = _vstr_dt is not None and type(_vstr_dt).__name__ == 'StrDType'
    _na_obj = getattr(arr, '_na_object', None)
    _na_semantics = _is_vstr and not isinstance(_na_obj, str)
    # numpy 语义：na_object=None 时 equal_nan=False 仅支持 hash 路径
    # （sorted=False）；默认排序路径抛 ValueError。
    if _na_semantics and not equal_nan and _na_obj is None and sorted:
        raise ValueError(
            "unique with equal_nan=False is not supported for StringDType "
            "arrays with na_object=None when sorted")

    def _is_nan(v):
        if isinstance(v, float):
            return v != v
        if isinstance(v, complex):
            return v.real != v.real or v.imag != v.imag
        return False

    def _is_na_obj(v):
        if not _na_semantics:
            return False
        if v is _na_obj or v is None:
            return True
        try:
            if v != v:  # 任意浮点型 NaN 都视为 NA
                return True
        except Exception:
            pass
        return False

    def _same_elem(x, y):
        # NA 对象 / NaN：仅 equal_nan=True 时相等——先于身份短路判定，
        # 使 equal_nan=False 时多个 NA 保持独立（即使是同一实例）
        if _is_na_obj(x) or _is_na_obj(y):
            return bool(equal_nan) and _is_na_obj(x) and _is_na_obj(y)
        if x is y:
            return True
        if _is_nan(x) or _is_nan(y):
            return bool(equal_nan) and _is_nan(x) and _is_nan(y)
        try:
            r = x == y
        except Exception:
            return False
        try:
            return bool(r)
        except Exception:
            return False

    def _same_row(ra, rb):
        return all(_same_elem(x, y) for x, y in zip(ra, rb))

    # ---------- 去重（记录首次出现位置） ----------
    _uniq_rows, _first, _inv, _cnt = [], [], [], []
    for _i, _row in enumerate(rows):
        for _j, _urow in enumerate(_uniq_rows):
            if _same_row(_row, _urow):
                _inv.append(_j)
                _cnt[_j] += 1
                break
        else:
            _inv.append(len(_uniq_rows))
            _uniq_rows.append(_row)
            _first.append(_i)
            _cnt.append(1)
    _k = len(_uniq_rows)

    # ---------- 排序（NaN 最后；-0.0 在 0.0 前；复数按 (实部, 虚部) 字典序） ----------
    def _key_elem(v):
        if _is_na_obj(v) or _is_nan(v):
            return (1, 0.0)
        if isinstance(v, complex):
            return (0, 0.0, _key_elem(v.real), _key_elem(v.imag))
        return (0, 0.0, v)

    def _key_row(row):
        return tuple(_key_elem(v) for v in row)

    if sorted:
        order = list(range(_k))
        try:
            order.sort(key=lambda _j: _key_row(_uniq_rows[_j]))
        except TypeError:
            order = list(range(_k))  # 不可比较类型（object 等）：保持首次出现序
    else:
        order = list(range(_k))
    _pos = [0] * _k
    for _new, _old in enumerate(order):
        _pos[_old] = _new
    uniq_rows = [_uniq_rows[_j] for _j in order]
    idx_arr = [_first[_j] for _j in order]
    inv_arr = [_pos[_j] for _j in _inv]
    cnt_arr = [_cnt[_j] for _j in order]

    # ---------- 结果数组 ----------
    # 结果数组布局：第 k 轴换成唯一数，其余位置按 C 序收集各唯一“行”的值。
    def _layout_axis(res_rows):
        _So = 1
        for _d in shape[:_ax]:
            _So *= _d
        _St = 1
        for _d in shape[_ax + 1:]:
            _St *= _d
        _L = _So * _St
        _fr = [None] * (_k * _L)
        for _j, _row in enumerate(res_rows):
            for _r in range(_L):
                _o = _r // _St
                _t = _r % _St
                _fr[_o * (_k * _St) + _j * _St + _t] = _row[_r]
        return _fr

    if fields is not None:
        # 结构化：每条记录是一个元素
        _recs = [list(r) for r in uniq_rows]
        if _ax is None:
            uniq_arr = npmod.array(_recs, dtype=list(fields))
        else:
            _shp_us = shape[:_ax] + (_k,) + shape[_ax + 1:]
            uniq_arr = npmod.array(_flat_to_nested(_layout_axis(_recs), _shp_us),
                                   dtype=list(fields))
    else:
        if _ax is None:
            if _is_vstr:
                # 保留 NA 对象身份，并把原始 dtype 实例挂回结果
                uniq_arr = nd([v for r in uniq_rows for v in r],
                              _dtype='string_', _na_object=_na_obj,
                              _vstring=True)
                uniq_arr._dtype_obj = _vstr_dt
            else:
                uniq_arr = nd([v for r in uniq_rows for v in r], _dtype=dtype)
        else:
            _shp_us = shape[:_ax] + (_k,) + shape[_ax + 1:]
            uniq_arr = nd(_flat_to_nested(_layout_axis(uniq_rows), _shp_us),
                          _dtype=dtype)

    def _intp_arr(lst, shp=None):
        if shp and len(shp) > 1:
            return nd(_flat_to_nested(list(lst), shp), _dtype='int64')
        return nd(list(lst), _dtype='int64')

    # numpy 语义：结果保持输入的数组子类型（如 Subclass）
    if uniq_arr.__class__ is not arr.__class__:
        uniq_arr.__class__ = arr.__class__

    def _assemble(res0):
        if not (return_index or return_inverse or return_counts):
            return res0
        _outs = [res0]
        if return_index:
            _outs.append(_intp_arr(idx_arr))
        if return_inverse:
            _outs.append(_intp_arr(inv_arr, shape if _ax is None else None))
        if return_counts:
            _outs.append(_intp_arr(cnt_arr))
        return _outs[0] if len(_outs) == 1 else tuple(_outs)

    if _ma_input is not None:
        from .ma.core import MaskedArray
        # 每个唯一值的掩码：其任一出现位置被掩码则整体被掩码
        _mflat = list(_ma_input._mask)
        _um_old = [builtins.any(_mflat[_i] for _i, _j in enumerate(_inv)
                                if _j == _t) for _t in range(_k)]
        _umask = [_um_old[_j] for _j in order]
        return _assemble(MaskedArray(uniq_arr, mask=_umask))

    return _assemble(uniq_arr)


# ========== Array API 风格的 unique 包装 ==========
from collections import namedtuple

_UniqueInv = namedtuple("UniqueAll", ["values", "inverse_indices"])
_UniqueAll = namedtuple("UniqueAll", ["values", "indices",
                                      "inverse_indices", "counts"])


def unique_values(x):
    """Array API：唯一值（equal_nan=False 语义）。"""
    return unique(x, equal_nan=False)


def unique_counts(x):
    """Array API：唯一值与计数（equal_nan=False 语义）。"""
    v, c = unique(x, return_counts=True, equal_nan=False)
    return v, c


def unique_inverse(x):
    """Array API：唯一值与逆映射（按输入形状，equal_nan=False 语义）。"""
    v, inv = unique(x, return_inverse=True, equal_nan=False)
    return _UniqueInv(values=v, inverse_indices=inv)


def unique_all(x):
    """Array API：唯一值 / 索引 / 逆映射 / 计数（equal_nan=False 语义）。"""
    v, idx, inv, c = unique(x, return_index=True, return_inverse=True,
                            return_counts=True, equal_nan=False)
    return _UniqueAll(values=v, indices=idx, inverse_indices=inv, counts=c)

# SPDX-License-Identifier: MIT
"""掩码数组核心（ma.core 兼容子集）。

包装底层 ndarray 与逐元素布尔掩码；提供 ma.core 移植测试
直接引用的公共名与辅助函数。
"""

import builtins
import warnings

import rsarray as np


def _flatten(seq):
    out = []
    for x in seq:
        if isinstance(x, (list, tuple)):
            out.extend(_flatten(x))
        else:
            out.append(x)
    return out


def _record_count(data):
    dt = getattr(data, '_dtype_obj', None)
    if dt is not None:
        return len(np._flatten_records(data._raw_data))
    return data.size


def _fields_leaf_count(dt):
    """结构化 dtype 的每记录叶子数（嵌套字段/子数组递归展开）。"""
    per = 0
    for _n in dt.names:
        fdt = dt.fields[_n][0]
        if getattr(fdt, '_names', None) is not None:
            per += _fields_leaf_count(fdt)
        elif getattr(fdt, '_subdtype', None) is not None:
            _b, _s = fdt._subdtype
            c = 1
            for s2 in (_s or ()):
                c *= s2
            if getattr(_b, '_names', None) is not None:
                per += c * _fields_leaf_count(_b)
            else:
                per += c
        else:
            per += 1
    return per


def _leaf_count(data):
    """掩码长度语义：普通数组 = size；结构化数组 = size × 每记录字段叶子数
    （—结构化 ma 的掩码为字段级）。"""
    dt = getattr(data, 'dtype', None)
    names = getattr(dt, 'names', None)
    size = int(getattr(data, 'size', 0) or 0)
    if not names:
        return size
    return size * _fields_leaf_count(dt)


def _is_ma(a):
    return isinstance(a, MaskedArray)


class MAError(ValueError):
    """掩码数组错误（ma.core.MAError 兼容）。"""


class MaskError(Exception):
    """掩码相关错误（ma.core.MaskError 兼容）。"""


class MaskedArrayFutureWarning(FutureWarning):
    """ma.core.MaskedArrayFutureWarning 兼容占位。"""


_ma_argsort_default = object()
_ma_reduce_default = object()


def _ma_mask_or(m1, m2, size):
    """掩码按位或，双方各自广播到 size（广播语义：1 → 全部重复，
    n < size → 平铺裁剪）。"""
    def bc(m):
        if len(m) == size:
            return list(m)
        if len(m) == 1:
            return list(m) * size if size else []
        if len(m) < size:
            reps = (size + len(m) - 1) // len(m)
            return (list(m) * reps)[:size]
        return list(m)[:size]
    a = bc(m1)
    b = bc(m2)
    return [x or y for x, y in zip(a, b)]


class MaskedArray:
    """带掩码的数组：包装底层 ndarray 与逐元素布尔掩码。"""

    def __init__(self, data=None, mask=False, dtype=None, fill_value=None,
                 copy=False, **kwargs):
        _ = copy, kwargs
        _src_ma = None
        if isinstance(data, MaskedArray):
            base_mask = list(data._mask)
            _src_ma = data
            data = data._data
        else:
            base_mask = None
            if not hasattr(data, '_array'):
                data = np.array(data, dtype=dtype) if dtype is not None else np.array(data)
            elif dtype is not None:
                data = data.view(dtype)
        self._data = data
        # 掩码长度语义：普通数组 = size；结构化数组 = size ×
        # 每记录字段叶子数（字段级掩码）。展平后按叶子数对齐：
        # 单值广播、不足补 False、超长截断——不做记录级截断（那会
        # 丢失字段级掩码信息）。
        n = _leaf_count(data)
        if mask is None or mask is False:
            m = base_mask if base_mask is not None else [False]
        elif mask is True:
            m = [True]
        else:
            mm = mask.tolist() if hasattr(mask, 'tolist') else list(mask)
            m = [bool(x) for x in _flatten(mm)]
        if len(m) == 1 and n > 1:
            m = m * n
        if len(m) > n:
            m = m[:n]
        if len(m) < n:
            m = m + [False] * (n - len(m))
        self._mask = m
        self.fill_value = self._as_fill(fill_value)

    def _as_fill(self, fv):
        """fill_value 规范化：按 data dtype 存为 0-d 数组（ma 语义：
        x.fill_value.dtype == x.dtype；不可转换则存原值）。"""
        if fv is None:
            return None
        try:
            if hasattr(fv, 'astype'):
                # np.array(0-d ndarray, dtype=...) 不改 dtype，须用 astype
                return fv.astype(self._data.dtype)
            return np.array(fv, dtype=self._data.dtype)
        except Exception:
            try:
                return np.array(fv)
            except Exception:
                return fv

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def dtype(self):
        return self._data.dtype

    @dtype.setter
    def dtype(self, value):
        # gh-31192：设置 dtype 为弃用行为（发警告但仍生效）
        warnings.warn(
            "Setting the dtype on a MaskedArray is deprecated and will be "
            "removed in a future version.",
            DeprecationWarning, stacklevel=2)
        self._data = self._data.astype(value)

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    @property
    def data(self):
        return self._data

    @property
    def mask(self):
        m = np.array(self._mask, dtype='bool')
        if len(self._mask) == int(self.size):
            # 普通数组：掩码形状与数据形状一致
            try:
                return m.reshape(self.shape)
            except Exception:
                return m
        # 结构化字段级掩码：长度 = 叶子数（size × 每记录字段数），平坦表示
        return m

    @mask.setter
    def mask(self, value):
        if value is None:
            self._mask = [False] * _leaf_count(self._data)
            return
        mask = np.asarray(value, dtype='bool')
        if mask.ndim == 0:
            self._mask = [bool(mask)] * _leaf_count(self._data)
            return
        self._mask = [bool(item) for item in mask.reshape((-1,)).tolist()]

    def _mask_flat(self):
        return list(self._mask)

    def tolist(self):
        """ma 语义：返回值（masked 位置保留原数据）。"""
        return self._data.tolist()

    def copy(self):
        """返回独立副本（data 与 mask 均拷贝）。"""
        return MaskedArray(self._data.copy(), list(self._mask))

    def _as_fill(self, fv):
        """fill_value 规范化：按 data dtype 存为 0-d 数组（ma 语义：
        x.fill_value.dtype == x.dtype；不可转换则存原值）。"""
        if fv is None:
            return None
        try:
            if hasattr(fv, 'astype'):
                # np.array(0-d ndarray, dtype=...) 不改 dtype，须用 astype
                return fv.astype(self._data.dtype)
            return np.array(fv, dtype=self._data.dtype)
        except Exception:
            try:
                return np.array(fv)
            except Exception:
                return fv

    def _fill_lossy(self, fv):
        """fill 值向当前 dtype 的转换是否 lossy（浮点静默回绕视为 lossy）。"""
        dt = str(self._data.dtype)
        try:
            if hasattr(fv, 'astype'):
                cast = fv.astype(dt)
            else:
                cast = np.array(fv, dtype=dt)
        except Exception:
            return True
        if isinstance(fv, float) and 'int' in dt and not (abs(fv) < 2 ** 63):
            return True
        try:
            return bool(isinstance(fv, float) and 'int' in dt
                        and float(cast) != fv)
        except Exception:
            return False

    def _inherit_fill(self, src):
        """从源 MaskedArray 继承 fill（exact 保留，lossy 忽略）。"""
        if _is_ma(src) and getattr(src, '_fill_value', None) is not None \
                and not self._fill_lossy(src._fill_value):
            self._fill_value = src._fill_value

    def astype(self, dtype, **kwargs):
        """dtype 转换：data 走 ndarray.astype，掩码原样保留。"""
        _ = kwargs
        return MaskedArray(self._data.astype(dtype), mask=list(self._mask),
                           fill_value=self.fill_value)

    def _ma_extreme(self, which, axis=None):
        """掩码感知 min/max：掩码位填 ±inf 后归约（ma 语义：忽略
        masked 元素）。axis=None 返回 0-d MaskedArray；axis 返回轴归约。"""
        import math as _math
        d = self._data
        shp = tuple(d.shape)
        msk = np.array(self._mask, dtype='bool')
        if len(msk) == int(self.size):
            msk = msk.reshape(shp)
        flat_v = d.reshape((-1,)).tolist() if d.ndim else [d.tolist()]
        flat_m = msk.reshape((-1,)).tolist()
        fillv = float('inf') if which == 'min' else float('-inf')
        fv = [fillv if mv else v for v, mv in zip(flat_v, flat_m)]
        if axis is None:
            if not fv:
                return MaskedArray(np.array(0.0), mask=[True])
            r = builtins.min(fv) if which == 'min' else builtins.max(fv)
            if r == fillv or (isinstance(r, float) and r != r):
                # 全掩码（或 nan）：masked 标量
                return MaskedArray(np.array(flat_v[0]), mask=[True])
            return MaskedArray(np.array(r), mask=[False])
        nd = len(shp)
        ax = axis % nd
        order = [i for i in range(nd) if i != ax] + [ax]

        def build(flat, dims):
            if not dims:
                return flat[0]
            step = 1
            for s in dims[1:]:
                step *= s
            return [build(flat[i * step:(i + 1) * step], dims[1:])
                    for i in range(dims[0])]

        nest = build(fv, [shp[i] for i in order])

        def reduce_last(lst):
            if lst and isinstance(lst[0], list):
                return [reduce_last(x) for x in lst]
            return builtins.min(lst) if which == 'min' \
                else builtins.max(lst)

        red = reduce_last(nest)
        out_shp = tuple(shp[i] for i in order[:-1])
        res = np.array(red)
        if out_shp:
            res = res.reshape(out_shp)
            # 全掩码行 → 掩码位为 True
            _of = [1 if v == fillv else 0 for v in red] if not out_shp \
                else None
            # 逐行掩码：构造行掩码嵌套
            def rowmask(flat, dims):
                if not dims:
                    return flat[0]
                step = 1
                for s in dims[1:]:
                    step *= s
                return [rowmask(flat[i * step:(i + 1) * step], dims[1:])
                        for i in range(dims[0])]

            fm = [1 if v == fillv else 0 for v in fv]
            nm = build(fm, [shp[i] for i in order])

            def mask_last(lst):
                if lst and isinstance(lst[0], list):
                    return [mask_last(x) for x in lst]
                return 1 if builtins.all(lst) else 0

            rm = mask_last(nm)
            rm = np.array(rm, dtype='bool').reshape(out_shp)
            out = MaskedArray(res, mask=False)
            out._mask = [bool(x) for x
                         in rm.reshape((-1,)).tolist()]
            out.fill_value = self.fill_value
            return out
        if red == fillv:
            return MaskedArray(np.array(flat_v[0]), mask=[True])
        return MaskedArray(np.array(red), mask=[False])

    def reshape(self, *args, **kwargs):
        """reshape：data 变形，平坦掩码直接沿用（元素总数不变）。"""
        return MaskedArray(self._data.reshape(*args, **kwargs),
                           mask=list(self._mask),
                           fill_value=self.fill_value)

    def min(self, axis=None, out=None, keepdims=False):
        """掩码感知最小值（ma 语义：忽略 masked 元素）。"""
        _ = out, keepdims
        return self._ma_extreme('min', axis)

    def max(self, axis=None, out=None, keepdims=False):
        """掩码感知最大值（ma 语义：忽略 masked 元素）。"""
        _ = out, keepdims
        return self._ma_extreme('max', axis)

    # ------------------------------------------------------------------
    # 转置族（ma 语义：结果保留掩码）
    # ------------------------------------------------------------------
    def _ma_rebuild(self, data, mask_flat):
        out = MaskedArray(data, mask=False)
        out._mask = list(mask_flat)
        out.fill_value = self.fill_value
        return out

    def transpose(self, *axes):
        data_t = self._data.transpose(*axes) if axes else self._data.T
        m = np.asarray(self._mask, dtype='bool').reshape(self.shape)
        mask_t = m.transpose(*axes) if axes else m.T
        return self._ma_rebuild(data_t, mask_t.reshape((-1,)).tolist())

    def swapaxes(self, axis1, axis2):
        data_t = self._data.swapaxes(axis1, axis2)
        m = np.asarray(self._mask, dtype='bool').reshape(self.shape)
        mask_t = m.swapaxes(axis1, axis2)
        return self._ma_rebuild(data_t, mask_t.reshape((-1,)).tolist())

    @property
    def T(self):
        return self.transpose()

    @property
    def mT(self):
        if self.ndim < 2:
            raise ValueError("matrix transpose with ndim < 2 is undefined")
        return self.swapaxes(self.ndim - 2, self.ndim - 1)

    def argsort(self, axis=_ma_argsort_default, kind=None, order=None):
        """掩码感知 argsort（ma.argsort 兼容，gh-8701 警告语义）。"""
        return _ma_argsort_impl(self, axis, kind, order)

    # ------------------------------------------------------------------
    # 算术/比较 dunder：掩码按位或传播（ma 语义的核心子集）
    # ------------------------------------------------------------------
    def _ma_binop(self, other, op, reflected=False):
        if isinstance(other, _MaskedConstant):
            other = other._as_ma()
        a = self._data
        b = other._data if _is_ma(other) else other
        res = op(b, a) if reflected else op(a, b)
        if not hasattr(res, '_array'):
            res = np.asarray(res)
        if _is_ma(other):
            m = _ma_mask_or(self._mask, other._mask,
                            int(getattr(res, 'size', len(self._mask))))
        else:
            m = list(self._mask)
        fill = self.fill_value if self.fill_value is not None else None
        return MaskedArray(res, mask=m, fill_value=fill)

    def _ma_refop(self, other, op):
        if not _is_ma(other):
            return NotImplemented
        return other._ma_binop(self, op, reflected=True)

    def __add__(self, other):
        return self._ma_binop(other, np.add)

    def __radd__(self, other):
        return self._ma_refop(other, np.add)

    def __sub__(self, other):
        return self._ma_binop(other, np.subtract)

    def __rsub__(self, other):
        return self._ma_refop(other, np.subtract)

    def __mul__(self, other):
        return self._ma_binop(other, np.multiply)

    def __rmul__(self, other):
        return self._ma_refop(other, np.multiply)

    def __truediv__(self, other):
        return self._ma_binop(other, np.divide)

    def __rtruediv__(self, other):
        return self._ma_refop(other, np.divide)

    def __floordiv__(self, other):
        return self._ma_binop(other, np.floor_divide)

    def __rfloordiv__(self, other):
        return self._ma_refop(other, np.floor_divide)

    def __mod__(self, other):
        return self._ma_binop(other, np.mod)

    def __rmod__(self, other):
        return self._ma_refop(other, np.mod)

    def __pow__(self, other):
        return self._ma_binop(other, np.power)

    def __rpow__(self, other):
        return self._ma_refop(other, np.power)

    def __neg__(self):
        return self._ma_binop(-1, np.multiply)

    def __pos__(self):
        return MaskedArray(self._data, list(self._mask),
                           fill_value=self.fill_value)

    def __abs__(self):
        # abs 是一元算子：不能走二元 _ma_binop（会双参调用 op）。
        # mask 恒等传播（|x| 不改变掩码）。
        res = np.abs(self._data)
        fill = self.fill_value if self.fill_value is not None else None
        return MaskedArray(res, mask=list(self._mask), fill_value=fill)

    def _ma_cmp(self, other, op):
        # ma _comparison 权威语义：
        # - mask = self.mask OR other.mask
        # - data = op(raw data)
        # - 仅 ==/!=：掩码位的 data 改为 op(self_mask, other_mask)
        #   （双掩码视为相等，单掩码视为不等）；其余比较 data 保持 raw
        # - fill_value 转 bool（默认 True）
        a = self._data
        if isinstance(other, _MaskedConstant):
            other = other._as_ma()
        b = other._data if _is_ma(other) else other
        res = op(a, b)
        if not hasattr(res, '_array'):
            res = np.asarray(res)
        omask = list(other._mask) if _is_ma(other) else None
        if omask is not None:
            m = _ma_mask_or(self._mask, omask,
                            int(getattr(res, 'size', len(self._mask))))
        else:
            m = list(self._mask)
        is_eqne = op is np.equal or op is np.not_equal
        if is_eqne:
            try:
                if res.ndim == 0:
                    sm = bool(self._mask[0]) if self._mask else False
                    om = bool(omask[0]) if omask else False
                    v = (sm == om) if op is np.equal else (sm != om)
                    res = np.array([v], dtype='bool').reshape(())
                else:
                    flat = res.reshape((-1,)).tolist()
                    nf = []
                    for i, mv in enumerate(m):
                        if mv:
                            sm = bool(self._mask[i]) \
                                if i < len(self._mask) else False
                            om = bool(omask[i]) \
                                if omask is not None and i < len(omask) \
                                else False
                            nf.append((sm == om) if op is np.equal
                                      else (sm != om))
                        else:
                            nf.append(flat[i])
                    res = np.array(nf, dtype='bool').reshape(res.shape)
            except Exception:
                pass
        if self.fill_value is None:
            fv = True
        else:
            try:
                fv = bool(self.fill_value)
            except Exception:
                fv = True
        return MaskedArray(res, mask=m, fill_value=fv)

    def __eq__(self, other):
        return self._ma_cmp(other, np.equal)

    def __ne__(self, other):
        return self._ma_cmp(other, np.not_equal)

    def __lt__(self, other):
        return self._ma_cmp(other, np.less)

    def __le__(self, other):
        return self._ma_cmp(other, np.less_equal)

    def __gt__(self, other):
        return self._ma_cmp(other, np.greater)

    def __ge__(self, other):
        return self._ma_cmp(other, np.greater_equal)

    def __and__(self, other):
        return self._ma_binop(other, np.logical_and)

    def __or__(self, other):
        return self._ma_binop(other, np.logical_or)

    def __xor__(self, other):
        return self._ma_binop(other, np.logical_xor)

    def __rand__(self, other):
        return self._ma_refop(other, np.logical_and)

    def __ror__(self, other):
        return self._ma_refop(other, np.logical_or)

    def __rxor__(self, other):
        return self._ma_refop(other, np.logical_xor)

    def view(self, dtype=None, type=None):
        _ = type
        if dtype is None:
            return MaskedArray(self._data, list(self._mask))
        newdata = self._data.view(dtype)
        newn = _leaf_count(newdata)
        newmask = list(self._mask) if len(self._mask) == newn else [False] * newn
        return MaskedArray(newdata, newmask)

    def filled(self, fill_value=None):
        value = self.fill_value if fill_value is None else fill_value
        if value is None:
            value = 0
        result = self._data.copy()
        result[self.mask] = value
        return result

    def compressed(self):
        return self._data[~self.mask].copy()

    def __getitem__(self, key):
        res = self._data[key]
        if hasattr(res, 'ndim') and getattr(res, 'ndim', 0) > 0:
            m = np.array(self._mask, dtype='bool')
            _sz = int(self.size)
            try:
                if len(self._mask) == _sz:
                    m = m.reshape(self.shape)
                    m_res = m[key]
                else:
                    m_res = None
            except Exception:
                m_res = None
            if m_res is not None:
                m_flat = [bool(x) for x
                          in np.asarray(m_res).reshape((-1,)).tolist()]
            else:
                m_flat = [False] * (int(res.size)
                                    if hasattr(res, 'size') else 0)
            return MaskedArray(res, m_flat)
        # 标量索引返回 0-d MaskedArray（ma 语义：掩码信息不能丢，
        # 否则后续比较走 ndarray 标量分支抛 Unsupported operand type）
        try:
            mv = bool(self._mask[int(key)]) if int(key) < len(self._mask) \
                else False
        except Exception:
            mv = False
        return MaskedArray(res, [mv])

    def __setitem__(self, key, value):
        self._data[key] = value

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return "masked_array(data=%s,\n             mask=%s)" % (self._data, self._mask)

    def __str__(self):
        return self.__repr__()


def array(data=None, mask=False, dtype=None, fill_value=None, copy=False, **kwargs):
    """构造一个掩码数组。"""
    return MaskedArray(data, mask=mask, dtype=dtype, fill_value=fill_value,
                       copy=copy, **kwargs)


masked_array = MaskedArray


def getdata(a):
    """返回掩码数组的底层数据。"""
    return a._data if _is_ma(a) else a


def getmask(a):
    """ma.getmask 兼容：非掩码数组返回 nomask。"""
    if _is_ma(a):
        return a._mask
    return nomask


def getmaskarray(a):
    """返回掩码数组的布尔掩码。"""
    if _is_ma(a):
        return a.mask
    return np.zeros(getattr(a, 'shape', (0,)), dtype='bool')


def asarray(a, dtype=None):
    if _is_ma(a):
        if dtype is not None and a.dtype != np.dtype(dtype):
            return MaskedArray(a.data.astype(dtype), a.mask, fill_value=a.fill_value)
        return a
    return MaskedArray(np.asarray(a, dtype=dtype))


def isMaskedArray(a):
    return _is_ma(a)


def masked_values(data, value, rtol=1e-5, atol=1e-8, copy=True):
    data = np.array(data, copy=copy)
    mask = np.abs(data - value) <= atol + rtol * np.abs(value)
    return MaskedArray(data, mask=mask, fill_value=value)


def masked_invalid(data, copy=True):
    data = np.array(data, copy=copy)
    return MaskedArray(data, mask=~np.isfinite(data))


class _MaskedConstant:
    """np.ma.masked 单例：按 数组语义表现为 0-d float64 MaskedArray
    （值 0.0、全掩码）。委托给惰性创建的 MaskedArray 实例。"""

    # 哨兵：np.add 等核心 ufunc 据此短路返回常量自身（避免循环导入）
    _ma_masked_self = True

    def __repr__(self):
        return 'masked'

    def __str__(self):
        return 'masked'

    def __bool__(self):
        return False

    def _as_ma(self):
        if '_ma' not in self.__dict__:
            self._ma = MaskedArray(np.array(0.0), mask=[True])
        return self._ma

    def __getattr__(self, name):
        # 常规查找失败后委托给 0-d MaskedArray（dtype/_data/_mask/算术等）
        return getattr(self._as_ma(), name)

    def __int__(self):
        # 数组语义：int(masked) 对整型目标抛 MaskError（不可静默取值）
        raise MaskError("Cannot convert masked constant to int")

    def __float__(self):
        return float(self._as_ma()._data)

    def __index__(self):
        return 0

    # 数组语义：与 masked 常量的任何二元运算都返回 masked 单例自身
    def _masked_ret(self, *args, **kwargs):
        return self

    __add__ = _masked_ret
    __radd__ = _masked_ret
    __sub__ = _masked_ret
    __rsub__ = _masked_ret
    __mul__ = _masked_ret
    __rmul__ = _masked_ret
    __truediv__ = _masked_ret
    __rtruediv__ = _masked_ret
    __floordiv__ = _masked_ret
    __rfloordiv__ = _masked_ret
    __mod__ = _masked_ret
    __rmod__ = _masked_ret
    __pow__ = _masked_ret
    __rpow__ = _masked_ret
    __and__ = _masked_ret
    __rand__ = _masked_ret
    __or__ = _masked_ret
    __ror__ = _masked_ret
    __xor__ = _masked_ret
    __rxor__ = _masked_ret
    __eq__ = _masked_ret
    __ne__ = _masked_ret
    __lt__ = _masked_ret
    __le__ = _masked_ret
    __gt__ = _masked_ret
    __ge__ = _masked_ret


masked = _MaskedConstant()
nomask = False
MaskType = np.bool_


# ---------------------------------------------------------------------------
# 掩码构造器（ma.core.masked_* 兼容）
# ---------------------------------------------------------------------------

def masked_equal(a, value, copy=True):
    a = np.array(a, copy=copy) if not hasattr(a, '_array') else a
    return MaskedArray(a, mask=(a == value))


def masked_not_equal(a, value, copy=True):
    a = np.array(a, copy=copy) if not hasattr(a, '_array') else a
    return MaskedArray(a, mask=(a != value))


def masked_greater(a, t, copy=True):
    a = np.array(a, copy=copy) if not hasattr(a, '_array') else a
    return MaskedArray(a, mask=(a > t))


def masked_greater_equal(a, t, copy=True):
    a = np.array(a, copy=copy) if not hasattr(a, '_array') else a
    return MaskedArray(a, mask=(a >= t))


def masked_less(a, t, copy=True):
    a = np.array(a, copy=copy) if not hasattr(a, '_array') else a
    return MaskedArray(a, mask=(a < t))


def masked_less_equal(a, t, copy=True):
    a = np.array(a, copy=copy) if not hasattr(a, '_array') else a
    return MaskedArray(a, mask=(a <= t))


def masked_inside(a, left, right, copy=True):
    a = np.array(a, copy=copy) if not hasattr(a, '_array') else a
    return MaskedArray(a, mask=((a >= left) & (a <= right)))


def masked_outside(a, left, right, copy=True):
    a = np.array(a, copy=copy) if not hasattr(a, '_array') else a
    return MaskedArray(a, mask=((a < left) | (a > right)))


def masked_object(a, copy=True):
    """掩掉 None 与 NaN 元素（ma.masked_object 兼容）。"""
    a = np.array(a, copy=copy) if not hasattr(a, '_array') else a
    vals = a.tolist()
    flat = _flatten(vals)
    mask = [v is None or (isinstance(v, float) and v != v) for v in flat]
    return MaskedArray(a, mask=mask)


def masked_where(condition, a, **kwargs):
    """按条件生成掩码数组（ma.masked_where 兼容）。"""
    cond = np.array(condition, dtype=bool) if not hasattr(condition, '_array') else condition
    return MaskedArray(a, mask=cond)


def mvoid(a):
    """全部掩码的掩码数组（ma.mvoid 兼容）。"""
    return MaskedArray(a, mask=True)


# ---------------------------------------------------------------------------
# 掩码/填充值辅助
# ---------------------------------------------------------------------------

def common_fill_value(dtype):
    """dtype 的默认填充值（ma.common_fill_value 兼容）。"""
    dt = np.dtype(dtype)
    kind = dt.kind
    if kind in ('U', 'S'):
        return ''
    if kind == 'c':
        return complex(np.inf, np.inf)
    if kind == 'f':
        return np.inf
    if kind == 'b':
        return False
    if kind in ('i', 'u'):
        return 999999
    if kind in ('M', 'm'):
        return 'NaT'
    return 0


def default_fill_value(a):
    if _is_ma(a):
        if a.fill_value is not None:
            return a.fill_value
    return common_fill_value(getattr(a, 'dtype', 'f'))


def maximum_fill_value(dt):
    dt = np.dtype(dt)
    kind = dt.kind
    if kind == 'b':
        return True
    if kind == 'f':
        return np.inf
    if kind in ('i', 'u'):
        return int(np.iinfo(dt).max)
    if kind in ('M', 'm'):
        return np.iinfo(np.dtype('int64')).max
    return 0


def minimum_fill_value(dt):
    dt = np.dtype(dt)
    kind = dt.kind
    if kind == 'b':
        return False
    if kind == 'f':
        return -np.inf
    if kind in ('i', 'u'):
        return int(np.iinfo(dt).min)
    if kind in ('M', 'm'):
        return int(np.iinfo(np.dtype('int64')).min)
    return 0


def make_mask(a, shrink=True, mask=nomask):
    """为 a 构造布尔掩码（ma.make_mask 兼容）。"""
    if mask is nomask:
        if _is_ma(a):
            return a._mask if a._mask.__class__ is list else nomask
        return nomask
    if _is_ma(a) and mask is a._mask:
        return mask
    m = np.asarray(mask, dtype='bool')
    if m.shape != tuple(getattr(a, 'shape', ())) and m.size == 1:
        m = np.broadcast_to(m, tuple(getattr(a, 'shape', ())))
    if shrink:
        if not bool(m.any()):
            return nomask
    return m


def make_mask_descr(dtype):
    """掩码的 dtype 描述（结构化 → 字段描述列表；否则 bool）。"""
    dt = np.dtype(dtype)
    if dt.names:
        return [(n, 'bool') for n in dt.names]
    return 'bool'


def flatten_mask(mask):
    if mask is nomask:
        return nomask
    m = np.asarray(mask, dtype='bool')
    return m.reshape((-1,))


def mask_or(a, b):
    """两个掩码按位或（ma.mask_or 兼容）。"""
    ma_ = getmaskarray(a)
    mb_ = getmaskarray(b)
    if ma_ is nomask:
        return mb_
    if mb_ is nomask:
        return ma_
    return ma_ | mb_


def fix_invalid(a):
    """把 inf/NaN 掩掉（ma.fix_invalid 兼容）。"""
    a = asarray(a)
    data = a.data
    bad = ~np.isfinite(data)
    newmask = list(a._mask)
    for i, b in enumerate(bad.reshape((-1,)).tolist()):
        if i < len(newmask):
            newmask[i] = newmask[i] or bool(b)
    a._mask = newmask[:_leaf_count(data)]
    return a


def count(a, axis=None, **kwargs):
    """未掩码元素计数（ma.count 兼容）。"""
    a = asarray(a) if not _is_ma(a) else a
    m = getmaskarray(a)
    if m is nomask:
        return a.size if axis is None else a.shape[axis]
    unmasked = ~m
    if axis is None:
        return int(unmasked.sum())
    return unmasked.sum(axis=axis)


def filled(a, fill_value=None):
    """掩码位置填值后返回普通数组（ma.filled 兼容）。"""
    if _is_ma(a):
        return a.filled(fill_value)
    return a


def anom(a, tol=1e-05):
    """掩掉 |x| <= tol 的"异常零"值（ma.anom 兼容）。"""
    a = np.asarray(a)
    return MaskedArray(a, mask=(np.abs(a) <= tol))


def minmax(a):
    """返回 (未掩码最小, 未掩码最大)（ma.minmax 兼容）。"""
    a = asarray(a)
    c = a.compressed()
    if c.size == 0:
        raise MAError("No minimal/maximal values")
    return (c.min(), c.max())


def masked_print_option(legacy_print=False):
    import warnings
    warnings.warn("ma.masked_print_option is deprecated and will be "
                  "removed; it is a no-op.", DeprecationWarning, stacklevel=2)

    def f(x):
        return x
    return f


def fromflex(fld, **kwargs):
    raise NotImplementedError(
        "rsarray.ma.fromflex is not implemented (flexible file format)")


def flatten_structured_array(a, out_dtype=None):
    """单字段结构化数组 → 普通数组（ma.flatten_structured_array 兼容）。"""
    dt = np.dtype(getattr(a, 'dtype', a))
    if dt.names and len(dt.names) == 1:
        return a[dt.names[0]]
    return np.asarray(a)


def anomalies(a, axis=None, dtype=None):
    """标准分数（z-score），掩码感知（ma.anomalies 兼容）。

    结果 = (a - mean)/std，沿 axis 归一化；std == 0 的位置与输入掩码
    位置一并掩码。
    """
    a = asarray(a)
    if axis is None:
        mean = a.mean()
        std = a.std()
        out = (a - mean) / std
        if bool(std == 0):
            return MaskedArray(out.data, mask=[True] * a.size,
                               fill_value=a.fill_value)
        return out
    mean = a.mean(axis=axis)
    std = a.std(axis=axis)
    if a.ndim > 0 and getattr(mean, 'ndim', 0) == a.ndim - 1:
        shape = [1] * a.ndim
        shape[axis] = a.shape[axis]
        mean = mean.reshape(tuple(shape))
        std = std.reshape(tuple(shape))
    out = (a - mean) / std
    bad = (std == 0)
    bad_data = bad.data if hasattr(bad, 'data') else bad
    if bool(np.asarray(bad_data).any()):
        try:
            bad_b = np.broadcast_to(np.asarray(bad_data), a.shape)
        except Exception:
            bad_b = np.asarray(bad_data)
        m = getmaskarray(out)
        m_data = m.data if isinstance(m, MaskedArray) else m
        m_final = bad_b if m is nomask else (m_data | bad_b)
        flat = list(np.asarray(m_final).reshape((-1,)).tolist())
        return MaskedArray(out.data, mask=flat, fill_value=a.fill_value)
    return out


def allequal(a, b):
    """两数组逐元素相等（只在双方都未掩码的位置比较，ma.allequal 兼容）。"""
    a = asarray(a)
    b = asarray(b)
    if a.shape != b.shape:
        return False
    ma_ = getmaskarray(a)
    mb_ = getmaskarray(b)
    if ma_ is nomask and mb_ is nomask:
        return bool((a.data == b.data).all())
    m = mask_or(ma_, mb_)
    if m is nomask or not bool(m.any()):
        return bool((a.data == b.data).all())
    unmasked = ~m
    sub_a = a.data[unmasked]
    sub_b = b.data[unmasked]
    if sub_a.size == 0:
        return True
    return bool((sub_a == sub_b).all())


def alltrue(a, axis=None, **kwargs):
    """未掩码元素全为真（掩码元素忽略，ma.alltrue 兼容）。"""
    a = asarray(a)
    if axis is None:
        c = a.compressed()
        return bool((c != 0).all()) if c.size else True
    m = getmaskarray(a)
    ok = (a.data != 0)
    if m is not nomask:
        ok = ok & ~m
    return ok.all(axis=axis)


def sometrue(a, axis=None, **kwargs):
    """任一未掩码元素为真（掩码元素忽略，ma.sometrue 兼容）。"""
    a = asarray(a)
    if axis is None:
        c = a.compressed()
        return bool((c != 0).any()) if c.size else False
    m = getmaskarray(a)
    ok = (a.data != 0)
    if m is not nomask:
        ok = ok & ~m
    return ok.any(axis=axis)


# ---------------------------------------------------------------------------
# 构造器与 re-export（ma.core 公共名从 rsarray 顶层取回）
# ---------------------------------------------------------------------------

def zeros(shape, dtype=float, **kwargs):
    """全零掩码数组（ma.zeros 兼容）。"""
    return MaskedArray(np.zeros(shape, dtype=dtype))


def ones(shape, dtype=float, **kwargs):
    """全一掩码数组（ma.ones 兼容）。"""
    return MaskedArray(np.ones(shape, dtype=dtype))


def empty(shape, dtype=float, **kwargs):
    """未初始化掩码数组（ma.empty 兼容）。"""
    return MaskedArray(np.empty(shape, dtype=dtype))


def full(shape, fill_value, dtype=None, **kwargs):
    """定值掩码数组（ma.full 兼容）。"""
    return MaskedArray(np.full(shape, fill_value, dtype=dtype))


def ones_like(a, dtype=None):
    return MaskedArray(np.ones_like(a.data if _is_ma(a) else a, dtype=dtype),
                       mask=list(asarray(a)._mask) if _is_ma(a) else None)


def zeros_like(a, dtype=None):
    return MaskedArray(np.zeros_like(a.data if _is_ma(a) else a, dtype=dtype),
                       mask=list(asarray(a)._mask) if _is_ma(a) else None)


def empty_like(a, dtype=None):
    return MaskedArray(np.empty_like(a.data if _is_ma(a) else a, dtype=dtype),
                       mask=list(asarray(a)._mask) if _is_ma(a) else None)


# ma 专属构造器：本地定义返回 MaskedArray（np.ma.zeros 等语义），
# 不可被 np 同名普通数组构造器覆盖。
_MA_OWN_CONSTRUCTORS = (
    'empty', 'empty_like', 'full', 'full_like',
    'ones', 'ones_like', 'zeros', 'zeros_like',
)


def _copy_names():
    names = (
        'abs', 'absolute', 'add', 'all', 'allclose', 'allequal', 'alltrue',
        'angle', 'any', 'arange', 'arccos', 'arccosh', 'arcsin', 'arctan',
        'arctan2', 'argsort', 'choose', 'concatenate', 'conjugate', 'cos',
        'cosh', 'deg2rad', 'diag', 'divide', 'empty', 'empty_like', 'equal',
        'exp', 'expm1', 'fabs', 'fix', 'floor', 'full', 'full_like',
        'greater', 'greater_equal', 'hstack', 'identity', 'inner', 'less',
        'less_equal', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logical_and',
        'logical_or', 'logical_xor', 'maximum', 'minimum', 'mod', 'multiply',
        'ndim', 'not_equal', 'ones', 'ones_like', 'outer', 'power', 'prod',
        'product', 'rad2deg', 'ravel', 'repeat', 'reshape', 'resize',
        'rint', 'shape', 'sin', 'sinh', 'sometrue', 'sort', 'sqrt',
        'subtract', 'sum', 'tan', 'tanh', 'transpose', 'trunc', 'where',
        'vstack', 'zeros', 'zeros_like', 'min', 'max', 'mean', 'std',
        'var', 'cumsum', 'cumprod', 'ptp', 'take', 'put', 'putmask',
        'swapaxes', 'newaxis', 'nonzero', 'hypot',
    )
    d = {}
    for n in names:
        if n in _MA_OWN_CONSTRUCTORS:
            continue
        if hasattr(np, n):
            d[n] = getattr(np, n)
    return d


for _n, _v in _copy_names().items():
    globals()[_n] = _v
del _copy_names, _n, _v

# 二元 ufunc 对 masked 常量短路（数组语义：add(masked, x) → masked）
_MA_BIN_UFUNCS = (
    'add', 'subtract', 'divide', 'equal', 'not_equal',
    'greater', 'greater_equal', 'less', 'less_equal',
    'logical_and', 'logical_or', 'logical_xor',
    'arctan2', 'hypot', 'logaddexp', 'maximum', 'minimum',
    'mod', 'power', 'outer', 'inner',
)


def _ma_ufunc_shortcircuit(f):
    def _w(*args, **kwargs):
        for _a in args:
            if isinstance(a := _a, _MaskedConstant):
                return a
        return f(*args, **kwargs)
    return _w


for _n in _MA_BIN_UFUNCS:
    if _n in globals():
        globals()[_n] = _ma_ufunc_shortcircuit(globals()[_n])
del _MA_BIN_UFUNCS, _ma_ufunc_shortcircuit, _n


# ---- 掩码感知 argsort / minmax（ma 语义；覆盖上面从 np 拷贝的同名函数）----

def _ma_argsort_impl(a, axis, kind=None, order=None):
    # gh-8701：>1-d 的默认 axis（None/展平）未来将变为 -1
    if axis is _ma_argsort_default:
        axis = None
        if a.ndim > 1:
            warnings.warn(
                "the default axis of argsort for 2-d arrays is not going to "
                "be the same as the axis of sort in the future; values will "
                "be sorted along the last axis instead.",
                MaskedArrayFutureWarning, stacklevel=3)
    if axis is not None:
        ax = axis % a.ndim
        res = np.argsort(a.data, axis=ax)
        return MaskedArray(res, mask=False, fill_value=a.fill_value)
    # axis=None：展平 argsort，masked 元素排末尾
    flat = a.data.reshape((-1,)).tolist()
    if not isinstance(flat, list):
        flat = [flat]
    mask_flat = ([bool(x) for x in a._mask]
                 if a._mask is not nomask else [False] * len(flat))
    idx = sorted(range(len(flat)), key=lambda i: (mask_flat[i], flat[i]))
    return MaskedArray(np.array(idx, dtype='int64'), mask=False,
                       fill_value=a.fill_value)


def argsort(a, axis=_ma_argsort_default, kind=None, order=None):
    """掩码感知 argsort（ma.argsort 兼容；gh-8701 警告语义）。"""
    return _ma_argsort_impl(asarray(a), axis, kind, order)


class _ma_minmax:
    """ma.minimum/maximum：可调二元算子 + .reduce（1.13 axis 警告语义）。"""

    def __init__(self, name, use_min):
        self.__name__ = name
        self._use_min = use_min
        self.__doc__ = "mask-aware %s (ma.%s compatible)" % (name, name)

    def __call__(self, a, b, **kw):
        return (np.minimum(a, b, **kw) if self._use_min
                else np.maximum(a, b, **kw))

    def reduce(self, a, axis=_ma_reduce_default, **kw):
        a = asarray(a)
        if axis is _ma_reduce_default:
            axis = None
            if a.ndim > 1:
                warnings.warn(
                    "the default axis of reduce for 2-d arrays is not going "
                    "to be the same as the axis of sum in the future.",
                    MaskedArrayFutureWarning, stacklevel=3)
        op = builtins.min if self._use_min else builtins.max
        fill = a.fill_value

        def _fill0():
            return fill if fill is not None else 0

        if a.ndim <= 1 or axis is None:
            flat = a.data.reshape((-1,)).tolist()
            if not isinstance(flat, list):
                flat = [flat]
            mask_flat = ([bool(x) for x in a._mask]
                         if a._mask is not nomask else [False] * len(flat))
            vals = [v for v, mm in zip(flat, mask_flat) if not mm]
            if not vals:
                return _fill0()
            return op(vals)
        # per-axis（2-d 简化：沿行/列取 min/max，masked 元素不参与）
        ax = axis % a.ndim
        d = a.data.tolist()
        rows, cols = len(d), len(d[0])
        mask_flat = ([bool(x) for x in a._mask]
                     if a._mask is not nomask else [False] * (rows * cols))
        out = []
        if ax == 1:
            for i in range(rows):
                vals = [d[i][j] for j in range(cols)
                        if not mask_flat[i * cols + j]]
                out.append(op(vals) if vals else _fill0())
        else:
            for j in range(cols):
                vals = [d[i][j] for i in range(rows)
                        if not mask_flat[i * cols + j]]
                out.append(op(vals) if vals else _fill0())
        return np.array(out, dtype=a.data.dtype)


minimum = _ma_minmax('minimum', True)
maximum = _ma_minmax('maximum', False)

# 遗留别名：product = prod。np.product 在顶层于 ma 导入之后才挂载，
# _copy_names 时尚不可见，这里显式补齐。
product = prod


# ---- 精确签名函数（测试用 inspect.signature 断言；须位于 _copy_names
# ---- 循环之后以覆盖从 np 拷贝的同名函数）

def nonzero(a):
    """掩码感知 nonzero：返回未掩码非零元素的索引元组（ma.nonzero）。"""
    a = asarray(a)
    m = getmaskarray(a)
    ok = a.data != 0
    if m is not nomask:
        m_data = m.data if isinstance(m, MaskedArray) else m
        ok = ok & ~m_data
    idx = np.nonzero(ok)
    return tuple(idx)


def cumsum(a, axis=None, dtype=None, out=None):
    """掩码感知累计和（ma.cumsum 兼容；掩码保留）。"""
    a = asarray(a)
    d = a.data
    res = d.cumsum() if axis is None else d.cumsum(axis=axis)
    if dtype is not None:
        res = res.astype(dtype)
    mask = list(a._mask) if a._mask is not nomask else False
    return MaskedArray(res, mask=mask, fill_value=a.fill_value)


def compress(condition, a, axis=None, out=None):
    """按条件提取元素（ma.compress 兼容；掩码位置不选入）。"""
    a = asarray(a)
    cond = np.asarray(condition)
    if axis is not None:
        res = np.compress(cond, a.data, axis=axis)
        return MaskedArray(res, mask=False, fill_value=a.fill_value)
    cond_flat = [bool(c) for c in list(np.asarray(cond).reshape((-1,)).tolist())]
    data_flat = list(a.data.reshape((-1,)).tolist())
    if a._mask is not nomask:
        mask_flat = list(a._mask)
    else:
        mask_flat = [False] * a.size
    vals = [d for c, d, m in zip(cond_flat, data_flat, mask_flat)
            if c and not m]
    return MaskedArray(np.array(vals, dtype=a.data.dtype), mask=False,
                       fill_value=a.fill_value)


def empty(shape, dtype=None, order='C', *, device=None, like=None,
          fill_value=None, hardmask=False):
    """掩码空数组（ma.empty 兼容；精确签名）。"""
    _ = order, device, like, hardmask
    arr = np.empty(shape, dtype=dtype)
    fv = common_fill_value(arr.dtype) if fill_value is None else fill_value
    try:
        arr.fill(fv)
    except Exception:
        pass
    return MaskedArray(arr, mask=False, fill_value=fv)


def empty_like(prototype, /, dtype=None, order='K', subok=True,
               shape=None, *, device=None):
    """同形状掩码空数组（ma.empty_like 兼容；精确签名）。"""
    _ = order, subok, device
    proto = prototype.data if isinstance(prototype, MaskedArray) else prototype
    if shape is not None:
        arr = np.empty(shape, dtype=dtype if dtype is not None
                       else proto.dtype)
    else:
        arr = np.empty_like(proto, dtype=dtype)
    return MaskedArray(arr, mask=False, fill_value=common_fill_value(arr.dtype))


def squeeze(a, axis=None, *, fill_value=None, hardmask=False):
    """压缩单元素轴（ma.squeeze 兼容；精确签名）。"""
    _ = hardmask
    a = asarray(a)
    res = a.data.squeeze(axis=axis) if axis is not None else a.data.squeeze()
    m = getmaskarray(a)
    mask = False
    if m is not nomask:
        m_arr = m.data if isinstance(m, MaskedArray) else m
        try:
            ms = (np.asarray(m_arr).squeeze(axis=axis) if axis is not None
                  else np.asarray(m_arr).squeeze())
            mask = list(ms.reshape((-1,)).tolist())
        except Exception:
            mask = False
    fv = a.fill_value if fill_value is None else fill_value
    return MaskedArray(res, mask=mask, fill_value=fv)


def identity(n, dtype=None, *, like=None, fill_value=None, hardmask=False):
    """单位矩阵（ma.identity 兼容；精确签名）。"""
    _ = like, hardmask
    arr = np.identity(n, dtype=dtype)
    fv = common_fill_value(arr.dtype) if fill_value is None else fill_value
    return MaskedArray(arr, mask=False, fill_value=fv)


__all__ = ['MAError', 'MaskError', 'MaskType', 'MaskedArray',
           'MaskedArrayFutureWarning', 'array', 'masked_array', 'masked',
           'nomask', 'masked_print_option', 'anom', 'minmax', 'mvoid',
           'fromflex', 'flatten_structured_array', 'getdata', 'getmask',
           'getmaskarray', 'asarray', 'isMaskedArray', 'masked_values',
           'masked_invalid', 'masked_equal', 'masked_not_equal',
           'masked_greater', 'masked_greater_equal', 'masked_less',
           'masked_less_equal', 'masked_inside', 'masked_outside',
           'masked_object', 'masked_where', 'common_fill_value',
           'default_fill_value', 'maximum_fill_value', 'minimum_fill_value',
           'make_mask', 'make_mask_descr', 'flatten_mask', 'mask_or',
           'fix_invalid', 'count', 'filled', 'zeros', 'ones', 'empty',
           'full', 'ones_like', 'zeros_like', 'empty_like']

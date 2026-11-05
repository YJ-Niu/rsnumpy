"""结构化数组辅助函数（numpy.lib.recfunctions 的兼容子集）。

实现直接操作 rsnumpy 结构化数组的纯 Python 记录模型（_dtype_obj / _raw_data），
复用顶层模块暴露的内部辅助函数。
"""

import math as _math

import rsnumpy as np


# ========== 内部辅助 ==========

def _rich_dtype(x):
    """从数组或 dtype-like 取得富 DType。"""
    if isinstance(x, np.DType):
        return x
    d = getattr(x, 'dtype', None)
    if isinstance(d, np.DType):
        return d
    if d is not None:
        return np.dtype(d)
    return np.dtype(x)


def _records(arr):
    """展平结构化数组为记录 tuple 列表（MaskedArray 先解包底层数据）。"""
    data, _mask = _rec_info(arr)
    return np._flatten_records(data._raw_data)


def _default_field(fdt):
    return np._default_field(fdt, np._zero_scalar_for)


def _project_value(ival, in_fdt, out_fdt):
    """按名称将输入字段值投影到输出字段类型（支持嵌套结构）。"""
    if out_fdt.names is not None and in_fdt is not None and in_fdt.names is not None:
        in_names = in_fdt.names
        # 列表推导代替显式 for 循环
        return tuple(
            _project_value(ival[in_names.index(on)], in_fdt[on], out_fdt[on])
            if on in in_names else _default_field(out_fdt[on])
            for on in out_fdt.names
        )
    return np._coerce_field_value(ival, out_fdt)


def _project_record(rec, in_dt, out_dt):
    """从输入记录按名称构造符合 out_dt 的记录，缺失字段用默认值。"""
    in_names = in_dt.names or ()
    # 列表推导代替显式 for 循环
    return tuple(
        _project_value(rec[in_names.index(on)], in_dt[on], out_dt[on])
        if on in in_names else _default_field(out_dt[on])
        for on in out_dt.names
    )


def _iter_flat(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _iter_flat(x)
    else:
        yield v


def _collect_leaves(val, fdt, out):
    if fdt.names is not None:
        for i, n in enumerate(fdt.names):
            _collect_leaves(val[i], fdt[n], out)
    elif fdt.subdtype is not None:
        base = fdt.subdtype[0]
        for x in _iter_flat(val):
            _collect_leaves(x, base, out)
    else:
        out.append(val)


def _record_leaves(rec, dt):
    # 列表推导代替显式 for 循环，但需扩展而非追加
    out = []
    for i, n in enumerate(dt.names):
        leaves = []
        _collect_leaves(rec[i], dt[n], leaves)
        out.extend(leaves)
    return out


def _count_leaves(fdt):
    if fdt.names is not None:
        return sum(_count_leaves(fdt[n]) for n in fdt.names)
    if fdt.subdtype is not None:
        base, shape = fdt.subdtype
        # 使用 math.prod 代替显式 for 循环求积
        return _math.prod(shape) * _count_leaves(base)
    return 1


def _build_from_leaves(flat, idx, fdt):
    if fdt.names is not None:
        # 顺序累积 idx，使用列表推导
        vals = []
        for n in fdt.names:
            v, idx = _build_from_leaves(flat, idx, fdt[n])
            vals.append(v)
        return tuple(vals), idx
    if fdt.subdtype is not None:
        base, shape = fdt.subdtype
        # 使用 math.prod 代替显式 for 循环求积
        total = _math.prod(shape)
        elems = []
        for _ in range(total):
            v, idx = _build_from_leaves(flat, idx, base)
            elems.append(v)
        return np._reshape_flat(elems, shape), idx
    return np._coerce_scalar_by_dt(flat[idx], fdt), idx + 1


def _fix_output(output, usemask=False, asrecarray=False):
    if asrecarray:
        output = np._as_recarray(output)
    if usemask:
        output = np.ma.MaskedArray(output)
    return output


# ========== dtype 内省 ==========

def get_names(adtype):
    """返回结构化 dtype 的字段名（嵌套字段以嵌套元组表示）。"""
    adtype = _rich_dtype(adtype)
    # 列表推导代替显式 for 循环
    return tuple(
        (name, tuple(get_names(current))) if (current := adtype[name]).names is not None else name
        for name in adtype.names
    )


def get_names_flat(adtype):
    """返回结构化 dtype 的全部字段名（扁平）。"""
    adtype = _rich_dtype(adtype)
    listnames = []
    # 使用 extend + 生成器推导替代嵌套 for 循环
    for name in adtype.names:
        listnames.append(name)
        current = adtype[name]
        if current.names is not None:
            listnames.extend(get_names_flat(current))
    return tuple(listnames)


def flatten_descr(ndtype):
    """将（可能嵌套的）结构化 dtype 展平为 (name, dtype) 列表。"""
    ndtype = _rich_dtype(ndtype)
    names = ndtype.names
    if names is None:
        return (('', ndtype),)
    descr = []
    for field in names:
        typ = ndtype[field]
        if typ.names is not None:
            descr.extend(flatten_descr(typ))
        else:
            descr.append((field, typ))
    return tuple(descr)


def get_fieldstructure(adtype, lastname=None, parents=None):
    """返回字段到其父字段列表的映射。"""
    adtype = _rich_dtype(adtype)
    if parents is None:
        parents = {}
    names = adtype.names
    for name in names:
        current = adtype[name]
        if current.names is not None:
            if lastname:
                parents[name] = [lastname]
            else:
                parents[name] = []
            parents.update(get_fieldstructure(current, name, parents))
        else:
            lastparent = list(parents.get(lastname, []) or [])
            if lastparent:
                lastparent.append(lastname)
            elif lastname:
                lastparent = [lastname]
            parents[name] = lastparent or []
    return parents


# ========== dtype 重排 ==========

def _repack_dtype(dt, align, recurse):
    # 列表推导代替显式 for 循环，一次提取 (name, fdt)
    def _process(name):
        fdt = dt.fields[name][0]
        if recurse and fdt.names is not None:
            fdt = _repack_dtype(fdt, align, recurse)
        return name, fdt

    pairs = [_process(name) for name in dt.names]
    names = [p[0] for p in pairs]
    formats = [p[1] for p in pairs]
    return np._build_struct(names, formats, None, None, None, align)


def repack_fields(a, align=False, recurse=False):
    """移除结构化 dtype/数组中字段之间的填充（可选按对齐重排）。"""
    if isinstance(a, np.DType):
        if a.names is None:
            return a
        return _repack_dtype(a, align, recurse)
    dt = _rich_dtype(a)
    newdt = _repack_dtype(dt, align, recurse)
    recs = _records(a)
    return np._wrap_structured(newdt, np._reshape_flat(recs, a.shape))


def _rename_dtype(dt, namemapper):
    # 列表推导代替显式 for 循环
    def _process(name):
        newname = namemapper.get(name, name)
        fdt = dt[name]
        if fdt.names is not None:
            fdt = _rename_dtype(fdt, namemapper)
        return newname, fdt

    pairs = [_process(name) for name in dt.names]
    names = [p[0] for p in pairs]
    formats = [p[1] for p in pairs]
    return np._build_struct(names, formats, None, None, None, dt.isalignedstruct)


def rename_fields(base, namemapper):
    """重命名结构化数组的字段（支持嵌套字段）。"""
    dt = _rich_dtype(base)
    newdt = _rename_dtype(dt, namemapper)
    recs = _records(base)
    return np._wrap_structured(newdt, np._reshape_flat(recs, base.shape))


# ========== 字段增删 ==========

def _drop_descr(ndtype, drop_names):
    names = ndtype.names
    # 列表推导：一次提取 (name, fdt_or_sub)，过滤掉 drop_names 与 None sub
    def _process(name):
        if name in drop_names:
            return None
        current = ndtype[name]
        if current.names is not None:
            sub = _drop_descr(current, drop_names)
            if sub is None:
                return None
            return (name, sub)
        return (name, current)

    pairs = [p for p in (_process(name) for name in names) if p is not None]
    if not pairs:
        return None
    newnames = [p[0] for p in pairs]
    newformats = [p[1] for p in pairs]
    return np._build_struct(newnames, newformats, None, None, None, False)


def drop_fields(base, drop_names, usemask=True, asrecarray=False):
    """删除结构化数组的指定字段（支持嵌套字段名）。"""
    if isinstance(drop_names, str):
        drop_names = {drop_names}
    else:
        drop_names = set(drop_names)
    dt = _rich_dtype(base)
    newdt = _drop_descr(dt, drop_names)
    recs = _records(base)
    newrecs = [_project_record(r, dt, newdt) for r in recs]
    output = np._wrap_structured(newdt, np._reshape_flat(newrecs, base.shape))
    return _fix_output(output, usemask=False, asrecarray=asrecarray)


def require_fields(a, required_dtype):
    """返回仅含指定字段的数组，缺失字段以默认值填充，已有字段做类型转换。"""
    newdt = _rich_dtype(required_dtype)
    dt = _rich_dtype(a)
    recs = _records(a)
    newrecs = [_project_record(r, dt, newdt) for r in recs]
    return np._wrap_structured(newdt, np._reshape_flat(newrecs, a.shape))


def recursive_fill_fields(input, output):
    """将 input 的同名字段递归填入 output（就地修改 output 并返回）。"""
    in_dt = _rich_dtype(input)
    out_dt = _rich_dtype(output)
    in_recs = _records(input)
    out_recs = [list(r) for r in _records(output)]
    in_names = in_dt.names or ()
    out_names = out_dt.names
    # 预计算字段索引映射 (out_idx, in_idx, on_name)
    name_pairs = [
        (oi, in_names.index(on), on)
        for oi, on in enumerate(out_names)
        if on in in_names
    ]
    n = min(len(in_recs), len(out_recs))
    # 外层循环仅取字段名/索引，内层列表推导填充记录
    for oi, ii, on in name_pairs:
        in_field_dt = in_dt[on]
        out_field_dt = out_dt[on]
        for k in range(n):
            out_recs[k][oi] = _project_value(in_recs[k][ii], in_field_dt, out_field_dt)
    _out_data, _out_mask = _rec_info(output)
    _out_data._raw_data = np._reshape_flat([tuple(r) for r in out_recs], _out_data.shape)
    return output


# ========== 结构化 <-> 非结构化 ==========

def structured_to_unstructured(arr, dtype=None, copy=False, casting='unsafe'):
    """将结构化数组转换为常规（多一维）数组。"""
    _ = copy, casting
    dt = _rich_dtype(arr)
    recs = _records(arr)
    rows = [_record_leaves(r, dt) for r in recs]
    nested = np._reshape_flat(rows, arr.shape)
    out = np.array(nested)
    if dtype is not None:
        out = out.astype(dtype)
    return out


def unstructured_to_structured(arr, dtype=None, names=None, **kwargs):
    """将常规数组转换为结构化数组。"""
    _ = kwargs
    if dtype is None and names is not None:
        raise ValueError("either dtype or names must be provided")
    dt = _rich_dtype(dtype)
    rows = arr.tolist()

    def build(row):
        rec = []
        idx = 0
        for n in dt.names:
            v, idx = _build_from_leaves(row, idx, dt[n])
            rec.append(v)
        return tuple(rec)

    outer_shape = arr.shape[:-1]
    flat_rows = np._flatten_records([rows]) if False else _flatten_rows(rows)
    recs = [build(r) for r in flat_rows]
    return np._wrap_structured(dt, np._reshape_flat(recs, outer_shape))


def _flatten_rows(rows):
    """将嵌套的行列表展平到最内层（每行为标量列表）。"""
    if rows and isinstance(rows[0], list) and rows[0] and isinstance(rows[0][0], list):
        out = []
        for r in rows:
            out.extend(_flatten_rows(r))
        return out
    return rows


# ========== 逐字段应用 ==========

def apply_along_fields(func, arr):
    """将归约函数沿字段轴应用于结构化数组。"""
    uarr = structured_to_unstructured(arr)
    return func(uarr, axis=-1)


# ========== 合并 / 堆叠 ==========

def _scalar_dtype_of(arr):
    d = arr.dtype
    if isinstance(d, np.DType):
        return d
    return np.dtype(d)


def merge_arrays(seqarrays, fill_value=-1, flatten=False, usemask=False,
                 asrecarray=False):
    """将多个数组合并为一个结构化数组。"""
    _ = flatten
    arrays = list(seqarrays)
    field_specs = []  # (name, fdt, values)
    counter = 0
    for arr in arrays:
        if not hasattr(arr, 'dtype'):
            arr = np.asarray(arr)
        dt = _rich_dtype(arr)
        if isinstance(dt, np.DType) and dt.names is not None:
            recs = _records(arr)
            # 嵌套列表推导：一次提取 (name, dt[n], values)
            field_specs.extend(
                (n, dt[n], [r[i] for r in recs])
                for i, n in enumerate(dt.names)
            )
        else:
            name = 'f%d' % counter
            counter += 1
            vals = arr.tolist()
            if not isinstance(vals, list):
                vals = [vals]
            field_specs.append((name, _scalar_dtype_of(arr), vals))
    maxlen = max((len(v) for _, _, v in field_specs), default=0)
    names = [s[0] for s in field_specs]
    formats = [s[1] for s in field_specs]
    newdt = np._build_struct(names, formats, None, None, None, False)
    # 嵌套列表推导：一次生成所有记录
    recs = [
        tuple(np._coerce_field_value(vals[k], fdt) if k < len(vals)
              else np._coerce_field_value(fill_value, fdt)
              for _, fdt, vals in field_specs)
        for k in range(maxlen)
    ]
    output = np._wrap_structured(newdt, recs)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)


def stack_arrays(arrays, defaults=None, usemask=True, asrecarray=False,
                 autoconvert=False):
    """将一组结构化数组按字段并集堆叠。"""
    _ = defaults, autoconvert
    if hasattr(arrays, '_array'):
        return arrays
    arrays = list(arrays)
    if len(arrays) == 1:
        return arrays[0]
    if not arrays:
        return np.array([])
    field_order = []
    field_dts = {}
    # 全为无名数组：直接拼接（numpy 语义，不做字段包装）
    if not any(_rich_dtype(arr).names is not None for arr in arrays):
        return np.concatenate([np.array(a) for a in arrays])
    # 非结构化数组包装成单字段记录：字段名按其在字段并集中的插入
    # 位置命名（numpy 语义：stack (x, z)→'f0'，stack (z, x)→'f2'）
    for ai, arr in enumerate(arrays):
        dt = _rich_dtype(arr)
        if dt.names is not None:
            # 使用 for 循环保持字段顺序（dict 去重）
            for n in dt.names:
                field_dts.setdefault(n, []).append(dt[n])
                if n not in field_dts or len(field_dts[n]) == 1:
                    field_order.append(n)
        else:
            fname = 'f%d' % len(field_order)
            arrays[ai] = np.array([(v,) for v in arr.tolist()],
                                  dtype=[(fname, dt)])
            field_dts.setdefault(fname, []).append(dt)
            if fname not in field_order:
                field_order.append(fname)
    # 字段 dtype：autoconvert 时取各来源中最宽者；不一致且未开则 TypeError
    _KIND_RANK = {'b': 0, 'i': 1, 'u': 1, 'f': 2, 'c': 3}

    def _widen(dts):
        cur = dts[0]
        for d in dts[1:]:
            if str(d) == str(cur):
                continue
            kc = getattr(cur, '_kind', None)
            kd = getattr(d, '_kind', None)
            if kd != kc:
                if _KIND_RANK.get(kd, -1) > _KIND_RANK.get(kc, -1):
                    cur = d
            elif getattr(d, 'itemsize', 0) > getattr(cur, 'itemsize', 0):
                cur = d
        return cur

    names = field_order
    formats = []
    for n in names:
        dts = field_dts[n]
        if len(dts) == 1:
            formats.append(dts[0])
        else:
            _w = _widen(dts)
            if not autoconvert and any(str(d) != str(_w) for d in dts):
                raise TypeError(
                    f'incompatible field dtype for {n!r}: '
                    f'{[str(d) for d in dts]} (use autoconvert=True)')
            formats.append(_w)
    newdt = np._build_struct(names, formats, None, None, None, False)
    # 逐记录投影到字段并集；缺失字段填默认值并置掩码；MaskedArray
    # 输入的掩码按记录传播（rsnumpy 结构化掩码为按记录形式）
    out_recs = []
    out_mask = []
    for arr in arrays:
        data, msk = _rec_info(arr)
        dt = _rich_dtype(data)
        in_names = dt.names or ()
        for ri, r in enumerate(_records(data)):
            row, mrow = [], []
            for on in names:
                if on in in_names:
                    row.append(_project_value(
                        r[in_names.index(on)], dt[on], newdt[on]))
                    mrow.append(bool(msk[ri]) if msk else False)
                else:
                    row.append(_default_field(newdt[on]))
                    mrow.append(True)
            out_recs.append(tuple(row))
            out_mask.append(mrow)
    output = np._wrap_structured(newdt, out_recs)
    if usemask:
        import rsnumpy.ma as _ma
        # 传每记录掩码列表（与 ma.array 结构化掩码构造路径一致）
        return _ma.MaskedArray(output, mask=out_mask)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)


# ========== 去重 ==========

def _hashable_key(v):
    """把记录值转为可哈希的键（list 递归转 tuple）。"""
    if isinstance(v, list):
        return tuple(_hashable_key(x) for x in v)
    if isinstance(v, tuple):
        return tuple(_hashable_key(x) for x in v)
    return v


def find_duplicates(a, key=None, ignoremask=True, return_index=False):
    """查找结构化（可选掩码）数组中的重复记录。"""
    if hasattr(a, '_data'):
        base = a._data
        mask = list(a._mask)
    else:
        base = a
        mask = None
    dt = _rich_dtype(base)
    recs = _records(base)
    if key is None:
        # numpy 语义：key=None 按"整条记录"比较（非第一个字段）
        ki = None
        vals = [_hashable_key(r) for r in recs]
    else:
        ki = dt.names.index(key)
        # 列表推导：一次性提取所有键值
        vals = [r[ki] for r in recs]
    if mask is None:
        mask = [False] * len(vals)

    # 字典推导构建 positions：ignoremask=True 跳过掩码记录；
    # ignoremask=False 时掩码记录视为各自独立（masked 值与任何值
    # 都不构成重复，numpy 语义）
    positions = {}
    for i, v in enumerate(vals):
        masked_i = i < len(mask) and mask[i]
        if ignoremask and masked_i:
            continue
        k = ('__masked__', i) if masked_i else v
        positions.setdefault(k, []).append(i)

    # 生成器推导：从 positions.values() 拉平并排序重复索引
    dup_idx = sorted(i for idxs in positions.values() if len(idxs) > 1 for i in idxs)
    dup_recs = [recs[i] for i in dup_idx]
    dup_arr = np._wrap_structured(dt, dup_recs)
    if hasattr(a, '_data'):
        dup_arr = np.ma.MaskedArray(dup_arr, [mask[i] for i in dup_idx])
    if return_index:
        return (dup_arr, np.array(dup_idx, dtype='int64'))
    return dup_arr


# ---------------------------------------------------------------------------
# append_fields / assign_fields_by_name / join_by（numpy.lib.recfunctions 兼容）
# ---------------------------------------------------------------------------

def _get_fieldspec(adtype):
    """dtype → [(name, dtype, title), ...]（numpy.lib.recfunctions._get_fieldspec 兼容）。"""
    dt = adtype if hasattr(adtype, 'names') else np.dtype(adtype)
    if dt.names:
        return [(n, dt.fields[n][0],
                 dt.fields[n][2] if len(dt.fields[n]) > 2 else None)
                for n in dt.names]
    return [(None, dt, None)]


def _zip_descr(seqarrays, flatten=False):
    """由一维数组列表构造记录字段描述（numpy.lib.recfunctions._zip_descr 兼容）。"""
    _ = flatten
    if len(seqarrays) == 0:
        return []
    if len(np.asarray(seqarrays[0]).shape[1:]) != 0:
        raise ValueError("only 1-dimensional arrays are supported")
    return [('r%d' % i, np.asarray(a).dtype)
            for i, a in enumerate(seqarrays)]


def _zip_dtype(seqarrays, flatten=False):
    """由一维数组列表构造记录 dtype（numpy.lib.recfunctions._zip_dtype 兼容）。"""
    return np.dtype(_zip_descr(seqarrays, flatten=flatten))

def _rec_info(a):
    """返回 (底层数据, 平坦掩码列表或 None)。"""
    import rsnumpy.ma as _ma
    if _ma.isMaskedArray(a):
        return a.data, list(a._mask)
    return a, None


def _append_fields_values(base, names, data, usemask=True, fill_value=-1):
    """numpy 语义的 append_fields：按字段名 + 值序列追加新字段。

    输出行数为 max(len(base), len(data))；不足的行以 fill_value 填充
    并置掩码。base 为普通数组时字段名取 'f0'（numpy 兼容）。
    """
    import rsnumpy
    import rsnumpy.ma as ma
    if isinstance(names, str):
        names = [names]
        fields_vals = [data]
    else:
        names = list(names)
        fields_vals = [list(f) for f in data]
    # 基数组字段行与掩码
    b_names_dt = getattr(getattr(base, 'dtype', None), 'names', None)
    if b_names_dt:
        binfo, bmask = _rec_info(base)
        b_names = list(b_names_dt)
        b_specs = [(n, binfo.dtype.fields[n][0]) for n in b_names]
        b_rows = [[binfo[i][n] for n in b_names] for i in range(binfo.shape[0])]
        b_masks = [[bmask[i * len(b_names) + k] if bmask else False
                    for k in range(len(b_names))]
                   for i in range(binfo.shape[0])]
    else:
        b_names = ['f0']
        b_specs = [('f0', base.dtype)]
        b_rows = [[v] for v in base.tolist()]
        b_masks = [[False] for _ in b_rows]
    n_out = max([len(b_rows)] + [len(f) for f in fields_vals])
    _new_specs = [(nm, np.array(f if len(f) else [0]).dtype)
                  for nm, f in zip(names, fields_vals)]
    # 缺失行/缺失值的填充：嵌套结构/子数组字段不能直接填标量
    # （_coerce_record 会因序列长度不匹配越界），按字段 dtype 取默认结构
    def _pad_val(fdt):
        if (getattr(fdt, '_names', None) is not None
                or getattr(fdt, '_subdtype', None) is not None):
            try:
                return np._default_field(fdt, lambda f: fill_value)
            except Exception:
                pass
        return fill_value
    out_rows, out_mask = [], []
    for i in range(n_out):
        row, mrow = [], []
        if i < len(b_rows):
            row += b_rows[i]
            mrow += b_masks[i]
        else:
            row += [_pad_val(s[1]) for s in b_specs]
            mrow += [True] * len(b_names)
        for nm, fdt_i, f in zip(names, _new_specs, fields_vals):
            if i < len(f):
                row.append(f[i])
                mrow.append(False)
            else:
                row.append(_pad_val(fdt_i))
                mrow.append(True)
        out_rows.append(row)
        out_mask.append(mrow)
    out_dtype = list(b_specs) + list(_new_specs)
    arr = rsnumpy.array(out_rows, dtype=out_dtype)
    if usemask:
        flat = [m for rowm in out_mask for m in rowm]
        return ma.MaskedArray(arr, mask=flat)
    return arr


def append_fields(rec1, rec2=None, dk1=None, dk2=None, usemask=True,
                  fill_value=-1, names=None, dtype=None, tobegin=False,
                  data=None, as_recarray=False):
    """按关键字段把 rec2 的字段追加到 rec1（numpy.append_fields 兼容）。

    numpy 2.x 语义：append_fields(base, names, data=None, fill_value=-1,
    usemask=True, as_recarray=False)——第二位置参数是字段名，data 是
    各字段的值序列；输出行数为 max(len(base), len(data))，不足的行以
    fill_value 填充并掩码。
    旧签名（按关键字段合并记录数组，dk1/dk2）保持兼容。
    """
    import rsnumpy
    import rsnumpy.ma as ma
    if data is not None:
        # numpy 形式：第二位置参数是字段名，data 是字段值序列
        if names is None:
            names = rec2
        rec2 = data
        return _append_fields_values(rec1, names, rec2, usemask=usemask,
                                     fill_value=fill_value)
    dk1 = list(dk1) if dk1 else []
    dk2 = list(dk2) if dk2 else []
    data1, mask1 = _rec_info(rec1)
    data2, mask2 = _rec_info(rec2)
    n1 = data1.dtype.names or ()
    n2 = data2.dtype.names or ()
    add_names = [n for n in n2 if n not in n1]
    out_names = list(n1) + add_names if not tobegin else add_names + list(n1)

    idx = {}
    for j in range(data2.shape[0]):
        key = tuple(data2[j][k] for k in dk2) if dk2 else j
        idx.setdefault(key, []).append(j)

    out_rows, out_mask = [], []
    for i in range(data1.shape[0]):
        row1 = data1[i]
        key = tuple(row1[k] for k in dk1) if dk1 else i
        j = idx.get(key, [None])[0]
        vals, ms = {}, {}
        for k, n in enumerate(n1):
            vals[n] = row1[n]
            ms[n] = (mask1[i * len(n1) + k] if mask1 else False)
        if j is not None:
            row2 = data2[j]
            for k, n in enumerate(n2):
                if n in add_names:
                    vals[n] = row2[n]
                    ms[n] = (mask2[j * len(n2) + k] if mask2 else False)
        else:
            for n in add_names:
                vals[n] = fill_value
                ms[n] = True
        out_rows.append([vals[n] for n in out_names])
        out_mask.append([ms[n] for n in out_names])

    out_dtype = [(n, (data1.dtype.fields[n][0] if n in n1
                      else data2.dtype.fields[n][0])) for n in out_names]
    arr = rsnumpy.array(out_rows, dtype=out_dtype)
    if usemask:
        flat = [m for row in out_mask for m in row]
        return ma.MaskedArray(arr, mask=flat)
    return arr


def assign_fields_by_name(rec1, rec2, usemask=True, overwrite=False,
                          extra_fields='drop', zero_unassigned=True):
    """按字段名把 rec2 的值赋给 rec1（第 i 行 ↔ 第 i 行，numpy 兼容）。

    原地写回 rec1 中与 rec2 同名的字段（嵌套结构按字段名递归）；
    rec1 独有字段：zero_unassigned=True（numpy 默认）清零，False 保留原值。
    rec2 独有字段按 extra_fields 丢弃（'drop'）。
    """
    import rsnumpy
    n1 = getattr(getattr(rec1, 'dtype', None), 'names', None)
    n2 = getattr(getattr(rec2, 'dtype', None), 'names', None)
    if n1 is None:
        # 非结构化数组（含 0-d）：整体赋值（numpy 测试的 unstructured 路径）
        try:
            rec1[...] = rec2
        except Exception:
            pass
        return rec1

    def _zero_field(arr, name):
        fdt = arr.dtype.fields[name][0]
        sub = getattr(fdt, '_names', None)
        if sub:
            fv = arr[name]
            for s in sub:
                _zero_field(fv, s)
        else:
            arr[name] = 0

    for n in n1:
        if n2 and n in n2:
            v = rec2[n]
            fdt = rec1.dtype.fields[n][0]
            if getattr(fdt, '_names', None) is not None:
                # 嵌套结构：按字段名递归赋值（子字段视图写回父数组）
                assign_fields_by_name(rec1[n], v, usemask=usemask,
                                      overwrite=overwrite,
                                      extra_fields=extra_fields,
                                      zero_unassigned=zero_unassigned)
            else:
                rec1[n] = v
        elif zero_unassigned:
            _zero_field(rec1, n)
    return rec1


def join_by(left, right, keys, jointype='inner', fill_value=None,
            r1postfix='1', r2postfix='2', defaults=None, usemask=True,
            as_recarray=False, lsuffix=None, rsuffix=None,
            join_type=None):
    """按键字段联接两个记录数组（SQL 风格，numpy.join_by 兼容）。

    jointype: 'inner' / 'left' / 'right' / 'outer'（同 'full'）。
    重叠的非键字段：左表字段加 r1postfix、右表字段加 r2postfix
    （numpy 默认 '1'/'2'）。
    """
    import rsnumpy
    # numpy 顺序：join_by(key, r1, r2, ...)——第一参数是键，后两个是数组；
    # 旧顺序 (left, right, keys) 以"首参是数组"识别并保持兼容。
    if (not hasattr(left, 'dtype') and hasattr(right, 'dtype')
            and hasattr(keys, 'dtype')):
        left, right, keys = right, keys, left
    if join_type is None:
        join_type = jointype
    join_type = str(join_type).lower()
    if join_type == 'outer':
        join_type = 'full'
    if join_type == 'leftouter':
        join_type = 'left'
    if join_type == 'rightouter':
        join_type = 'right'
    dl, ml = _rec_info(left)
    dr, mr = _rec_info(right)
    nl = dl.dtype.names or ()
    nr = dr.dtype.names or ()
    if isinstance(keys, str):
        keys = [keys]
    keys = list(keys)
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate key field(s): {keys}")

    _lp = lsuffix if lsuffix else (r1postfix if r1postfix is not None else '1')
    _rp = rsuffix if rsuffix else (r2postfix if r2postfix is not None else '2')
    left_out = {}
    for n in nl:
        if n in nr and n not in keys:
            left_out[n] = f'{n}{_lp}'
        else:
            left_out[n] = n
    right_out = {}
    for n in nr:
        if n in nl and n not in keys:
            right_out[n] = f'{n}{_rp}'
        else:
            right_out[n] = n
    # spec 项: (输出名, 'l'|'r', 源字段名, fdt)。
    # numpy 语义：键字段排最前（按左表字段顺序，不后缀，只出一次）；
    # 同名非键字段左/右相邻输出（b1,b2,c1,c2）；两表同名字段 dtype
    # 不同时取宽者（如 S5/S10 → S10）。
    def _wider_dt(dt_l, dt_r):
        ls_, rs_ = getattr(dt_l, 'itemsize', 0), getattr(dt_r, 'itemsize', 0)
        return dt_r if getattr(dt_r, '_kind', None) == getattr(
            dt_l, '_kind', None) and rs_ > ls_ else dt_l

    spec = []
    for n in nl:
        if n in keys:
            _fdt = dl.dtype.fields[n][0]
            if n in nr:
                _fdt = _wider_dt(_fdt, dr.dtype.fields[n][0])
            spec.append((n, 'l', n, _fdt))
    for n in nl:
        if n in keys:
            continue
        spec.append((left_out[n], 'l', n, dl.dtype.fields[n][0]))
        if n in nr:
            spec.append((right_out[n], 'r', n, dr.dtype.fields[n][0]))
    for n in nr:
        if n not in nl:
            spec.append((right_out[n], 'r', n, dr.dtype.fields[n][0]))
    out_names = [s[0] for s in spec]

    def _key_val(rec, k):
        # 子数组字段值（ndarray）不可哈希，转为 tuple
        v = rec[k]
        if hasattr(v, 'tolist'):
            t = v.tolist()
            return tuple(t) if isinstance(t, list) else t
        return tuple(v) if isinstance(v, list) else v

    idx = {}
    for j in range(dr.shape[0]):
        key = tuple(_key_val(dr[j], k) for k in keys)
        idx.setdefault(key, []).append(j)
    right_lists = {k: list(v) for k, v in idx.items()}
    consumed_r, consumed_l = set(), set()
    out_rows, out_mask = [], []

    # 缺省值：fill_value=None 时按字段 dtype 取类型默认值（值被掩码，
    # numpy 语义：defaults 未给时数值取 0、字符串取 ''）
    def _default_for(fdt):
        if fill_value is not None:
            return fill_value
        k = getattr(fdt, '_kind', None)
        if k in ('S', 'U'):
            return ''
        if k == 'c':
            return 0j
        if k in ('i', 'u', 'f', 'b'):
            return 0
        return None

    # 字段名重复（如空后缀导致重命名失效）→ ValueError（numpy 语义）
    if len(set(out_names)) != len(out_names):
        dupes = sorted({n for n in out_names if out_names.count(n) > 1})
        raise ValueError(f'field name(s) duplicated in join result: {dupes}')

    def _emit(i, j):
        row, mrow = [], []
        for out_n, side, src_n, fdt in spec:
            if side == 'l':
                if i is None:
                    row.append(_default_for(fdt))
                    mrow.append(True)
                else:
                    row.append(dl[i][src_n])
                    mrow.append(ml[i * len(nl) + nl.index(src_n)]
                                if ml else False)
            else:
                if j is None:
                    row.append(_default_for(fdt))
                    mrow.append(True)
                else:
                    row.append(dr[j][src_n])
                    mrow.append(mr[j * len(nr) + nr.index(src_n)]
                                if mr else False)
        out_rows.append(row)
        out_mask.append(mrow)

    if join_type in ('inner', 'left', 'full'):
        for i in range(dl.shape[0]):
            key = tuple(_key_val(dl[i], k) for k in keys)
            ms = [j for j in right_lists.get(key, []) if j not in consumed_r]
            if ms:
                for j in ms:
                    consumed_r.add(j)
                    consumed_l.add(i)
                    _emit(i, j)
            elif join_type in ('left', 'full'):
                _emit(i, None)
    if join_type in ('right', 'full'):
        for j in range(dr.shape[0]):
            if j in consumed_r:
                continue
            key = tuple(_key_val(dr[j], k) for k in keys)
            i = next((x for x in range(dl.shape[0])
                      if tuple(_key_val(dl[x], k) for k in keys) == key
                      and x not in consumed_l), None)
            if i is not None:
                consumed_l.add(i)
                _emit(i, j)
            else:
                _emit(None, j)

    out_dtype = [(n, fdt) for n, _side, _src, fdt in spec]
    arr = rsnumpy.array(out_rows, dtype=out_dtype)
    import rsnumpy.ma as ma
    if usemask and (rsnumpy.ma.isMaskedArray(left)
                    or rsnumpy.ma.isMaskedArray(right)
                    or any(any(row) for row in out_mask)):
        flat = [m for row in out_mask for m in row]
        return ma.MaskedArray(arr, mask=flat)
    return arr

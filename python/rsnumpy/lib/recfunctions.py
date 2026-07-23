"""结构化数组辅助函数（numpy.lib.recfunctions 的兼容子集）。

实现直接操作 rsnumpy 结构化数组的纯 Python 记录模型（_dtype_obj / _raw_data），
复用顶层模块暴露的内部辅助函数。
"""

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
    """展平结构化数组为记录 tuple 列表。"""
    return np._flatten_records(arr._raw_data)


def _default_field(fdt):
    return np._default_field(fdt, np._zero_scalar_for)


def _project_value(ival, in_fdt, out_fdt):
    """按名称将输入字段值投影到输出字段类型（支持嵌套结构）。"""
    if out_fdt.names is not None and in_fdt is not None and in_fdt.names is not None:
        out = []
        for on in out_fdt.names:
            if on in in_fdt.names:
                sub = ival[in_fdt.names.index(on)]
                out.append(_project_value(sub, in_fdt[on], out_fdt[on]))
            else:
                out.append(_default_field(out_fdt[on]))
        return tuple(out)
    return np._coerce_field_value(ival, out_fdt)


def _project_record(rec, in_dt, out_dt):
    """从输入记录按名称构造符合 out_dt 的记录，缺失字段用默认值。"""
    in_names = in_dt.names or ()
    out = []
    for on in out_dt.names:
        if on in in_names:
            out.append(_project_value(rec[in_names.index(on)], in_dt[on], out_dt[on]))
        else:
            out.append(_default_field(out_dt[on]))
    return tuple(out)


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
    out = []
    for i, n in enumerate(dt.names):
        _collect_leaves(rec[i], dt[n], out)
    return out


def _count_leaves(fdt):
    if fdt.names is not None:
        return sum(_count_leaves(fdt[n]) for n in fdt.names)
    if fdt.subdtype is not None:
        base, shape = fdt.subdtype
        total = 1
        for s in shape:
            total *= s
        return total * _count_leaves(base)
    return 1


def _build_from_leaves(flat, idx, fdt):
    if fdt.names is not None:
        vals = []
        for n in fdt.names:
            v, idx = _build_from_leaves(flat, idx, fdt[n])
            vals.append(v)
        return tuple(vals), idx
    if fdt.subdtype is not None:
        base, shape = fdt.subdtype
        total = 1
        for s in shape:
            total *= s
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
    listnames = []
    for name in adtype.names:
        current = adtype[name]
        if current.names is not None:
            listnames.append((name, tuple(get_names(current))))
        else:
            listnames.append(name)
    return tuple(listnames)


def get_names_flat(adtype):
    """返回结构化 dtype 的全部字段名（扁平）。"""
    adtype = _rich_dtype(adtype)
    listnames = []
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
    names = []
    formats = []
    for name in dt.names:
        fdt = dt.fields[name][0]
        if recurse and fdt.names is not None:
            fdt = _repack_dtype(fdt, align, recurse)
        names.append(name)
        formats.append(fdt)
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
    names = []
    formats = []
    for name in dt.names:
        newname = namemapper.get(name, name)
        fdt = dt[name]
        if fdt.names is not None:
            fdt = _rename_dtype(fdt, namemapper)
        names.append(newname)
        formats.append(fdt)
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
    newnames = []
    newformats = []
    for name in names:
        current = ndtype[name]
        if name in drop_names:
            continue
        if current.names is not None:
            sub = _drop_descr(current, drop_names)
            if sub is not None:
                newnames.append(name)
                newformats.append(sub)
        else:
            newnames.append(name)
            newformats.append(current)
    if not newnames:
        return None
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
    for oi, on in enumerate(out_dt.names):
        if on in (in_dt.names or ()):
            ii = in_dt.names.index(on)
            for k in range(min(len(in_recs), len(out_recs))):
                out_recs[k][oi] = _project_value(in_recs[k][ii], in_dt[on], out_dt[on])
    output._raw_data = np._reshape_flat([tuple(r) for r in out_recs], output.shape)
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
            for i, n in enumerate(dt.names):
                field_specs.append((n, dt[n], [r[i] for r in recs]))
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
    recs = []
    for k in range(maxlen):
        rec = []
        for _, fdt, vals in field_specs:
            if k < len(vals):
                rec.append(np._coerce_field_value(vals[k], fdt))
            else:
                rec.append(np._coerce_field_value(fill_value, fdt))
        recs.append(tuple(rec))
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
    field_dt = {}
    has_structured = False
    for arr in arrays:
        dt = _rich_dtype(arr)
        if dt.names is not None:
            has_structured = True
            for n in dt.names:
                if n not in field_dt:
                    field_order.append(n)
                    field_dt[n] = dt[n]
    if not has_structured:
        return np.concatenate([np.array(a) for a in arrays])
    names = field_order
    formats = [field_dt[n] for n in names]
    newdt = np._build_struct(names, formats, None, None, None, False)
    out_recs = []
    for arr in arrays:
        dt = _rich_dtype(arr)
        for r in _records(arr):
            out_recs.append(_project_record(r, dt, newdt))
    output = np._wrap_structured(newdt, out_recs)
    return _fix_output(output, usemask=usemask, asrecarray=asrecarray)


# ========== 去重 ==========

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
        key = dt.names[0]
    ki = dt.names.index(key)
    vals = [r[ki] for r in recs]
    if mask is None:
        mask = [False] * len(vals)
    positions = {}
    for i, v in enumerate(vals):
        if ignoremask and i < len(mask) and mask[i]:
            continue
        positions.setdefault(v, []).append(i)
    dup_idx = sorted(i for idxs in positions.values() if len(idxs) > 1 for i in idxs)
    dup_recs = [recs[i] for i in dup_idx]
    dup_arr = np._wrap_structured(dt, dup_recs)
    if hasattr(a, '_data'):
        dup_arr = np.ma.MaskedArray(dup_arr, [mask[i] for i in dup_idx])
    if return_index:
        return (dup_arr, np.array(dup_idx, dtype='int64'))
    return dup_arr

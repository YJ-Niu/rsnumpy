use ndarray::{Array, IxDyn, Slice};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyTuple};
use rayon::prelude::*;

use crate::{NdArray, parse_py_list_to_flat};

#[derive(Clone)]
enum IndexDesc {
    Slice(isize, isize, isize),
    Int(usize),
    Fancy(Vec<usize>),
    /// 多维花式索引：扁平化下标 + 原始形状（用于 numpy 语义下的广播）。
    FancyMulti(Vec<usize>, Vec<usize>),
    /// 完整布尔掩码：覆盖所有维度，扁平化下标。
    FullBoolMask(Vec<usize>),
}

fn parse_single_index(item: &Bound<'_, PyAny>, dim_size: isize) -> PyResult<IndexDesc> {
    if let Ok(idx) = item.extract::<isize>() {
        let actual_idx = if idx < 0 {
            (dim_size + idx) as usize
        } else {
            idx as usize
        };
        return Ok(IndexDesc::Int(actual_idx));
    }

    if let Ok(list) = item.cast::<PyList>() {
        if list.is_empty() {
            return Ok(IndexDesc::Fancy(vec![]));
        }
        let is_bool = list
            .iter()
            .all(|e| e.get_type().name().map(|n| n == "bool").unwrap_or(false));

        if is_bool {
            let fancy: Vec<usize> = list
                .iter()
                .enumerate()
                .filter(|(_, e)| e.extract::<bool>().unwrap_or(false))
                .map(|(j, _)| j)
                .collect();
            return Ok(IndexDesc::Fancy(fancy));
        }

        let mut fancy: Vec<usize> = Vec::with_capacity(list.len());
        let mut all_int = true;
        for e in list.iter() {
            if let Ok(v) = e.extract::<isize>() {
                let actual = if v < 0 {
                    (dim_size + v) as usize
                } else {
                    v as usize
                };
                fancy.push(actual);
            } else {
                all_int = false;
                break;
            }
        }
        if all_int {
            return Ok(IndexDesc::Fancy(fancy));
        }

        let (values, shape) = parse_py_list_to_flat(item)?;
        let nd_shape = if shape.is_empty() {
            IxDyn(&[])
        } else {
            IxDyn(&shape)
        };
        let arr = Array::from_shape_vec(nd_shape, values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(ndarray_to_index_desc(
            &NdArray {
                imag: None,
                data: arr,
            },
            dim_size,
        ));
    }

    if let Ok(arr) = item.extract::<NdArray>() {
        return Ok(ndarray_to_index_desc(&arr, dim_size));
    }

    // Also try to extract from Python ndarray wrapper (_array attribute)
    if let Ok(arr) = item
        .getattr("_array")
        .and_then(|a| a.extract::<NdArray>().map_err(|e| -> PyErr { e.into() }))
    {
        return Ok(ndarray_to_index_desc(&arr, dim_size));
    }

    if let Ok(slice_obj) = item.cast::<PySlice>() {
        let start = slice_obj.getattr("start")?.extract::<Option<isize>>()?;
        let stop = slice_obj.getattr("stop")?.extract::<Option<isize>>()?;
        let step = slice_obj.getattr("step")?.extract::<Option<isize>>()?;
        let (start, stop, step) = normalize_slice(start, stop, step, dim_size)?;
        return Ok(IndexDesc::Slice(start, stop, step));
    }

    Err(PyTypeError::new_err(format!(
        "Unsupported index type: {}",
        item.get_type().name()?
    )))
}

/// 按 CPython `slice.indices(length)` 语义归一化切片，正确处理负步长：
/// 步长为负时缺省起点为末元素、缺省终点越过 0；负索引按 length 折算并夹紧到合法边界。
fn normalize_slice(
    start: Option<isize>,
    stop: Option<isize>,
    step: Option<isize>,
    length: isize,
) -> PyResult<(isize, isize, isize)> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(PyValueError::new_err("slice step cannot be zero"));
    }
    // 起点夹紧区间 [lower, upper]；负步长时下界为 -1（可越过 0 到达末端反向）。
    let (lower, upper) = if step < 0 {
        (-1, length - 1)
    } else {
        (0, length)
    };
    let start = match start {
        Some(mut s) => {
            if s < 0 {
                s += length;
            }
            s.clamp(lower, upper)
        }
        None => {
            if step < 0 {
                upper
            } else {
                lower
            }
        }
    };
    let stop = match stop {
        Some(mut s) => {
            if s < 0 {
                s += length;
            }
            s.clamp(lower, upper)
        }
        None => {
            if step < 0 {
                lower
            } else {
                upper
            }
        }
    };
    Ok((start, stop, step))
}

/// 依据归一化后的 (start, stop, step) 枚举实际下标（含负步长的反向遍历）。
fn slice_indices_vec(start: isize, stop: isize, step: isize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = start;
    if step > 0 {
        while i < stop {
            out.push(i as usize);
            i += step;
        }
    } else if step < 0 {
        while i > stop {
            out.push(i as usize);
            i += step;
        }
    }
    out
}

fn ndarray_to_index_desc(arr: &NdArray, dim_size: isize) -> IndexDesc {
    let vals: Vec<f64> = arr.data.iter().copied().collect();
    let shape = arr.data.shape().to_vec();

    // 注意：rsnumpy 内部所有数组均为 f64，无法区分 bool/整数索引。
    // 布尔掩码索引应在 Python 层（__getitem__）提前处理，不进入 Rust 端。
    // 故此处始终将值视为整数索引，即使值为 0.0/1.0 也不作布尔掩码处理。
    let fancy: Vec<usize> = vals
        .iter()
        .map(|&v| {
            let iv = v as isize;
            if iv < 0 {
                (dim_size + iv) as usize
            } else {
                iv as usize
            }
        })
        .collect();
    if arr.data.ndim() > 1 {
        IndexDesc::FancyMulti(fancy, shape)
    } else {
        IndexDesc::Fancy(fancy)
    }
}

fn parse_indices(key: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<Vec<IndexDesc>> {
    let key_tuple = key.cast::<PyTuple>()?;

    if key_tuple.len() == 1 {
        let item = key_tuple.get_item(0)?;
        let arr = item.extract::<NdArray>().or_else(|_| {
            item.getattr("_array")
                .and_then(|a| a.extract::<NdArray>().map_err(|e| -> PyErr { e.into() }))
        });
        if let Ok(arr) = arr {
            let mask_shape: Vec<usize> = arr.data.shape().to_vec();
            if mask_shape == shape.to_vec() {
                let vals: Vec<f64> = arr.data.iter().copied().collect();
                if vals.iter().all(|&v| v == 0.0 || v == 1.0) {
                    let fancy: Vec<usize> = vals
                        .iter()
                        .enumerate()
                        .filter(|&(_, v)| *v == 1.0)
                        .map(|(i, _)| i)
                        .collect();
                    return Ok(vec![IndexDesc::FullBoolMask(fancy)]);
                }
            }
        }
    }

    let indices: Vec<IndexDesc> = key_tuple
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let dim_size = shape.get(i).copied().unwrap_or(0) as isize;
            parse_single_index(&item, dim_size)
        })
        .collect::<PyResult<_>>()?;

    Ok(indices)
}

fn is_all_int(indices: &[IndexDesc]) -> bool {
    indices.iter().all(|idx| matches!(idx, IndexDesc::Int(_)))
}

fn has_fancy(indices: &[IndexDesc]) -> bool {
    indices.iter().any(|idx| {
        matches!(
            idx,
            IndexDesc::Fancy(_) | IndexDesc::FancyMulti(_, _) | IndexDesc::FullBoolMask(_)
        )
    })
}

#[allow(dead_code)]
fn build_dim_lists(indices: &[IndexDesc]) -> Vec<Vec<usize>> {
    indices
        .iter()
        .map(|idx| match idx {
            IndexDesc::Fancy(v) | IndexDesc::FancyMulti(v, _) | IndexDesc::FullBoolMask(v) => {
                v.clone()
            }
            IndexDesc::Slice(start, stop, step) => slice_indices_vec(*start, *stop, *step),
            IndexDesc::Int(idx) => vec![*idx],
        })
        .collect()
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    if shape.len() > 1 {
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

/// 计算两个形状按 numpy 规则广播后的形状；不兼容时返回 None。
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let av = if i + a.len() < n {
            1
        } else {
            a[i + a.len() - n]
        };
        let bv = if i + b.len() < n {
            1
        } else {
            b[i + b.len() - n]
        };
        *slot = if av == bv {
            av
        } else if av == 1 {
            bv
        } else if bv == 1 {
            av
        } else {
            return None;
        };
    }
    Some(out)
}

fn slice_and_int_index(
    a: &Array<f64, IxDyn>,
    indices: &[IndexDesc],
) -> PyResult<Array<f64, IxDyn>> {
    let mut cur = a.clone();
    // 从后往前处理：逆序时 Int 索引移除高维轴后，低维 Slice 的 dim 仍然有效。
    // 逆序保证了当处理低维索引时，其对应轴位置 ≤ cur.ndim()。
    for (dim, idx) in indices.iter().enumerate().rev() {
        match idx {
            IndexDesc::Slice(start, stop, step) => {
                cur = if *step > 0 {
                    let s = Slice {
                        start: *start,
                        end: Some(*stop),
                        step: *step,
                    };
                    cur.slice_axis(ndarray::Axis(dim), s)
                        .into_owned()
                        .into_dyn()
                } else {
                    let idxs = slice_indices_vec(*start, *stop, *step);
                    cur.select(ndarray::Axis(dim), &idxs).into_dyn()
                };
            }
            IndexDesc::Int(i) => {
                let cur_ndim = cur.ndim();
                let ax = dim.min(cur_ndim.saturating_sub(1));
                let dim_size = cur.shape()[ax];
                if *i >= dim_size {
                    return Err(PyIndexError::new_err("Index out of bounds"));
                }
                cur = cur.index_axis(ndarray::Axis(ax), *i).to_owned().into_dyn();
            }
            _ => unreachable!(),
        }
    }
    Ok(cur)
}

/// 依据已补齐的索引描述从单个 f64 数组中取值，返回结果数组（不含虚部）。
/// 语义遵循 numpy 高级索引：整型轴移除，切片轴保留，花式索引按广播合并成一个块。
fn select_from(
    data: &Array<f64, IxDyn>,
    filled_indices: &[IndexDesc],
    shape: &[usize],
) -> PyResult<Array<f64, IxDyn>> {
    if is_all_int(filled_indices) {
        let strides = compute_strides(shape);
        let d = data.as_standard_layout();
        let flat = d.as_slice().unwrap();
        let flat_idx: usize = filled_indices
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                if let IndexDesc::Int(v) = idx {
                    v * strides[i]
                } else {
                    0
                }
            })
            .sum();
        return Ok(Array::from_elem(IxDyn(&[]), flat[flat_idx]));
    }

    if !has_fancy(filled_indices) {
        return slice_and_int_index(data, filled_indices);
    }

    for idx in filled_indices {
        if let IndexDesc::FullBoolMask(v) = idx {
            let src = data.as_standard_layout();
            let src_flat = src.as_slice().unwrap();
            let result: Vec<f64> = v.iter().map(|&i| src_flat[i]).collect();
            return Array::from_shape_vec(IxDyn(&[result.len()]), result)
                .map_err(|e| PyValueError::new_err(e.to_string()));
        }
    }

    // 每个源轴一种取值方式。存在高级索引时，整型索引按 numpy 语义并入高级组
    // （视为 0 维数组：广播不贡献输出维度，但参与连续性判定并在原位被消费）。
    enum AxisPlan {
        Slice(Vec<usize>),
        Adv { shape: Vec<usize>, flat: Vec<usize> },
    }
    let axes: Vec<AxisPlan> = filled_indices
        .iter()
        .map(|idx| match idx {
            IndexDesc::Slice(a, b, c) => AxisPlan::Slice(slice_indices_vec(*a, *b, *c)),
            IndexDesc::Int(i) => AxisPlan::Adv {
                shape: vec![],
                flat: vec![*i],
            },
            IndexDesc::Fancy(v) => AxisPlan::Adv {
                shape: vec![v.len()],
                flat: v.clone(),
            },
            IndexDesc::FancyMulti(v, sh) => AxisPlan::Adv {
                shape: sh.clone(),
                flat: v.clone(),
            },
            IndexDesc::FullBoolMask(_) => unreachable!(),
        })
        .collect();

    // 高级索引轴位置 + 广播后的块形状。
    let adv_axes: Vec<usize> = axes
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a, AxisPlan::Adv { .. }))
        .map(|(i, _)| i)
        .collect();
    let mut adv_shape: Vec<usize> = vec![];
    for &ai in &adv_axes {
        if let AxisPlan::Adv { shape: sh, .. } = &axes[ai] {
            adv_shape = broadcast_shapes(&adv_shape, sh)
                .ok_or_else(|| PyValueError::new_err("shape mismatch in advanced index"))?;
        }
    }
    let adv_ndim = adv_shape.len();
    // 高级索引轴是否连续：连续则块留在原位，否则按 numpy 规则前置。
    let contiguous = adv_axes.windows(2).all(|w| w[1] == w[0] + 1);

    enum OutAxis {
        Slice(usize),
        AdvBlock,
    }
    let mut out_axes: Vec<OutAxis> = Vec::new();
    if contiguous {
        let first_adv = *adv_axes.first().unwrap();
        for (d, a) in axes.iter().enumerate() {
            match a {
                AxisPlan::Slice(_) => out_axes.push(OutAxis::Slice(d)),
                AxisPlan::Adv { .. } => {
                    if d == first_adv {
                        out_axes.push(OutAxis::AdvBlock);
                    }
                }
            }
        }
    } else {
        out_axes.push(OutAxis::AdvBlock);
        for (d, a) in axes.iter().enumerate() {
            if let AxisPlan::Slice(_) = a {
                out_axes.push(OutAxis::Slice(d));
            }
        }
    }

    let mut out_shape: Vec<usize> = Vec::new();
    for oa in &out_axes {
        match oa {
            OutAxis::Slice(d) => {
                if let AxisPlan::Slice(idxs) = &axes[*d] {
                    out_shape.push(idxs.len());
                }
            }
            OutAxis::AdvBlock => out_shape.extend_from_slice(&adv_shape),
        }
    }

    let src_strides = compute_strides(shape);
    let src = data.as_standard_layout();
    let src_flat = src.as_slice().unwrap();
    let out_strides = compute_strides(&out_shape);
    let total: usize = out_shape.iter().product();

    // 各高级索引数组自身的行主序步长。
    let adv_strides: Vec<Vec<usize>> = adv_axes
        .iter()
        .map(|&ai| {
            if let AxisPlan::Adv { shape: sh, .. } = &axes[ai] {
                compute_strides(sh)
            } else {
                unreachable!()
            }
        })
        .collect();

    let result: Vec<f64> = (0..total)
        .into_par_iter()
        .map(|lin| {
            // 解码输出多下标。
            let mut rem = lin;
            let mut out_coord = vec![0usize; out_shape.len()];
            for (k, oc) in out_coord.iter_mut().enumerate() {
                *oc = rem / out_strides[k];
                rem %= out_strides[k];
            }
            // 组装源多下标：切片/高级块按输出坐标映射（整型轴已并入高级块）。
            let mut src_idx = vec![0usize; axes.len()];
            let mut cursor = 0usize;
            for oa in &out_axes {
                match oa {
                    OutAxis::Slice(d) => {
                        if let AxisPlan::Slice(idxs) = &axes[*d] {
                            src_idx[*d] = idxs[out_coord[cursor]];
                        }
                        cursor += 1;
                    }
                    OutAxis::AdvBlock => {
                        let adv_coord = &out_coord[cursor..cursor + adv_ndim];
                        for (j, &ai) in adv_axes.iter().enumerate() {
                            if let AxisPlan::Adv {
                                shape: sh, flat, ..
                            } = &axes[ai]
                            {
                                let off = adv_ndim - sh.len();
                                let mut fidx = 0usize;
                                for (dd, &sd) in sh.iter().enumerate() {
                                    let c = if sd == 1 { 0 } else { adv_coord[off + dd] };
                                    fidx += c * adv_strides[j][dd];
                                }
                                if !flat.is_empty() && fidx < flat.len() {
                                    src_idx[ai] = flat[fidx];
                                }
                            }
                        }
                        cursor += adv_ndim;
                    }
                }
            }
            let mut src_off = 0usize;
            for (d, &si) in src_idx.iter().enumerate() {
                src_off += si * src_strides[d];
            }
            src_flat[src_off]
        })
        .collect();

    Array::from_shape_vec(IxDyn(&out_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn getitem_multi(a: &NdArray, key: &Bound<'_, PyAny>, shape: Vec<usize>) -> PyResult<NdArray> {
    let indices = parse_indices(key, &shape)?;

    let mut filled_indices = indices;
    let ndim = shape.len();
    while filled_indices.len() < ndim {
        let dim_size = shape[filled_indices.len()] as isize;
        filled_indices.push(IndexDesc::Slice(0, dim_size, 1));
    }

    let data = select_from(&a.data, &filled_indices, &shape)?;
    let imag = match &a.imag {
        Some(im) => Some(select_from(im, &filled_indices, &shape)?),
        None => None,
    };
    Ok(NdArray { imag, data })
}

#[pyfunction]
pub fn getitem_scalar(a: &NdArray, indices: Vec<isize>) -> PyResult<f64> {
    let strides = compute_strides(a.data.shape());
    let data = a.data.as_slice_memory_order().unwrap();

    let mut flat_idx = 0;
    for (i, &idx) in indices.iter().enumerate() {
        let dim = a.data.shape()[i] as isize;
        let actual = if idx < 0 { dim + idx } else { idx };
        if actual < 0 || actual >= dim {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        flat_idx += actual as usize * strides[i];
    }

    Ok(data[flat_idx])
}

/// 将 `values`（长度 1 表示广播）按 dim_lists 的笛卡尔序散射写入 `data`。
#[allow(dead_code)]
fn scatter(data: &mut [f64], dim_lists: &[Vec<usize>], strides: &[usize], values: &[f64]) {
    let broadcast = values.len() == 1;
    let mut indices = vec![0usize; dim_lists.len()];
    let mut counter = 0usize;
    loop {
        let mut flat_idx = 0;
        for (i, &idx) in indices.iter().enumerate() {
            flat_idx += dim_lists[i][idx] * strides[i];
        }
        if flat_idx < data.len() {
            data[flat_idx] = if broadcast {
                values[0]
            } else {
                values[counter]
            };
        }
        counter += 1;

        let mut i = dim_lists.len() as isize - 1;
        while i >= 0 {
            indices[i as usize] += 1;
            if indices[i as usize] < dim_lists[i as usize].len() {
                break;
            }
            indices[i as usize] = 0;
            i -= 1;
        }
        if i < 0 {
            break;
        }
    }
}

/// 按 C 顺序（行优先）展平 ndarray，确保非 C 连续数组也能正确排序。
/// 直接用 `arr.iter()` 会按内存顺序迭代，对转置视图或列切片会产生错误的顺序。
fn flatten_c_order(arr: &Array<f64, IxDyn>) -> Vec<f64> {
    let shape: Vec<usize> = arr.shape().to_vec();
    let total = arr.len();
    if arr.is_standard_layout() {
        // C 连续数组，直接按内存顺序拷贝
        if let Some(s) = arr.as_slice_memory_order() {
            return s.to_vec();
        }
    }
    // 非 C 连续，按 C 顺序逐元素索引
    let strides = compute_strides(&shape);
    let mut result = Vec::with_capacity(total);
    for lin in 0..total {
        let mut rem = lin;
        let mut idx: Vec<usize> = Vec::with_capacity(shape.len());
        for &st in &strides {
            let coord = rem / st;
            rem %= st;
            idx.push(coord);
        }
        let val = arr[IxDyn(&idx)];
        result.push(val);
    }
    result
}

#[pyfunction]
pub fn setitem_multi(
    a: &Bound<'_, NdArray>,
    key: &Bound<'_, PyAny>,
    shape: Vec<usize>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let indices = parse_indices(key, &shape)?;

    let mut filled_indices = indices;
    let ndim = shape.len();
    while filled_indices.len() < ndim {
        let dim_size = shape[filled_indices.len()] as isize;
        filled_indices.push(IndexDesc::Slice(0, dim_size, 1));
    }

    let val_nd = crate::coerce_value_to_nd(value)?;
    // 按 C 顺序（行优先）展平值数组，确保与 scatter_advanced 的值索引一致。
    // 不能直接用 data.iter()，因为非 C 连续数组（如转置视图、列切片）的
    // iter() 顺序是内存顺序而非 C 顺序。
    let re: Vec<f64> = flatten_c_order(&val_nd.data);
    let im: Option<Vec<f64>> = val_nd.imag.as_ref().map(flatten_c_order);

    let target_count = calculate_target_count(&filled_indices);
    let broadcast = re.len() == 1;
    if !broadcast && re.len() != target_count {
        return Err(PyValueError::new_err(format!(
            "could not broadcast input array of size {} into selection of size {}",
            re.len(),
            target_count
        )));
    }

    let mut a_borrow = a.borrow_mut();

    if im.is_some() && a_borrow.imag.is_none() {
        let zeros = Array::zeros(a_borrow.data.raw_dim());
        a_borrow.imag = Some(zeros);
    }

    // 使用数据的实际内存步长（按元素数，非字节），而非总是用 C 顺序计算的 strides。
    // 原因：copy()/clone() 会保留原数组布局，若原数组为 F 连续（如 fix1._s[0]），
    // 用 C 顺序 strides 会写入错误位置。as_slice_memory_order_mut 返回内存顺序切片，
    // 必须配合实际 strides 使用。
    // 注意：ndarray 的 `strides()` 返回的已经是元素步长（非字节），不要再除以 elem_size。
    let write_strides: Vec<usize> = a_borrow
        .data
        .strides()
        .iter()
        .map(|&s| s as usize)
        .collect();

    {
        let data = a_borrow.data.as_slice_memory_order_mut().unwrap();
        scatter_advanced(data, &filled_indices, &write_strides, &re, broadcast);
    }
    if let Some(imag_arr) = a_borrow.imag.as_mut() {
        let idata = imag_arr.as_slice_memory_order_mut().unwrap();
        match &im {
            Some(iv) => scatter_advanced(idata, &filled_indices, &write_strides, iv, broadcast),
            None => scatter_advanced(idata, &filled_indices, &write_strides, &[0.0], true),
        }
    }

    Ok(())
}

fn scatter_advanced(
    data: &mut [f64],
    indices: &[IndexDesc],
    strides: &[usize],
    values: &[f64],
    broadcast: bool,
) {
    for idx in indices {
        if let IndexDesc::FullBoolMask(v) = idx {
            for (i, &flat_idx) in v.iter().enumerate() {
                if flat_idx < data.len() {
                    data[flat_idx] = if broadcast { values[0] } else { values[i] };
                }
            }
            return;
        }
    }

    enum AxisPlan {
        Slice(Vec<usize>),
        Adv { shape: Vec<usize>, flat: Vec<usize> },
    }
    let axes: Vec<AxisPlan> = indices
        .iter()
        .map(|idx| match idx {
            IndexDesc::Slice(a, b, c) => AxisPlan::Slice(slice_indices_vec(*a, *b, *c)),
            IndexDesc::Int(i) => AxisPlan::Adv {
                shape: vec![],
                flat: vec![*i],
            },
            IndexDesc::Fancy(v) => AxisPlan::Adv {
                shape: vec![v.len()],
                flat: v.clone(),
            },
            IndexDesc::FancyMulti(v, sh) => AxisPlan::Adv {
                shape: sh.clone(),
                flat: v.clone(),
            },
            IndexDesc::FullBoolMask(_) => unreachable!(),
        })
        .collect();

    let adv_axes: Vec<usize> = axes
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a, AxisPlan::Adv { .. }))
        .map(|(i, _)| i)
        .collect();

    if adv_axes.is_empty() {
        let slice_dim_lists: Vec<Vec<usize>> = axes
            .iter()
            .filter_map(|a| match a {
                AxisPlan::Slice(v) => Some(v.clone()),
                _ => None,
            })
            .collect();
        let total: usize = slice_dim_lists.iter().map(|d| d.len()).product();
        let slice_strides: Vec<usize> = {
            let mut s = vec![1; slice_dim_lists.len()];
            if slice_dim_lists.len() > 1 {
                for i in (0..slice_dim_lists.len() - 1).rev() {
                    s[i] = s[i + 1] * slice_dim_lists[i + 1].len();
                }
            }
            s
        };
        let slice_axes: Vec<usize> = axes
            .iter()
            .enumerate()
            .filter(|(_, a)| matches!(a, AxisPlan::Slice(_)))
            .map(|(i, _)| i)
            .collect();
        for lin in 0..total {
            let mut rem = lin;
            let mut src_idx = vec![0usize; indices.len()];
            for (k, &si) in slice_axes.iter().enumerate() {
                let coord = rem / slice_strides[k];
                rem %= slice_strides[k];
                if let AxisPlan::Slice(v) = &axes[si] {
                    src_idx[si] = v[coord];
                }
            }
            let mut flat_idx = 0;
            for (d, &si) in src_idx.iter().enumerate() {
                flat_idx += si * strides[d];
            }
            if flat_idx < data.len() {
                data[flat_idx] = if broadcast { values[0] } else { values[lin] };
            }
        }
        return;
    }

    let mut adv_shape: Vec<usize> = vec![];
    for &ai in &adv_axes {
        if let AxisPlan::Adv { shape: sh, .. } = &axes[ai] {
            adv_shape = broadcast_shapes(&adv_shape, sh).unwrap();
        }
    }
    let adv_ndim = adv_shape.len();

    let contiguous = adv_axes.windows(2).all(|w| w[1] == w[0] + 1);

    enum OutAxis {
        Slice(usize),
        AdvBlock,
    }
    let mut out_axes: Vec<OutAxis> = Vec::new();
    if contiguous {
        let first_adv = *adv_axes.first().unwrap();
        for (d, a) in axes.iter().enumerate() {
            match a {
                AxisPlan::Slice(_) => out_axes.push(OutAxis::Slice(d)),
                AxisPlan::Adv { .. } => {
                    if d == first_adv {
                        out_axes.push(OutAxis::AdvBlock);
                    }
                }
            }
        }
    } else {
        out_axes.push(OutAxis::AdvBlock);
        for (d, a) in axes.iter().enumerate() {
            if let AxisPlan::Slice(_) = a {
                out_axes.push(OutAxis::Slice(d));
            }
        }
    }

    let mut out_shape: Vec<usize> = Vec::new();
    for oa in &out_axes {
        match oa {
            OutAxis::Slice(d) => {
                if let AxisPlan::Slice(idxs) = &axes[*d] {
                    out_shape.push(idxs.len());
                }
            }
            OutAxis::AdvBlock => out_shape.extend_from_slice(&adv_shape),
        }
    }

    let out_strides = compute_strides(&out_shape);
    let total: usize = out_shape.iter().product();

    let adv_strides: Vec<Vec<usize>> = adv_axes
        .iter()
        .map(|&ai| {
            if let AxisPlan::Adv { shape: sh, .. } = &axes[ai] {
                compute_strides(sh)
            } else {
                unreachable!()
            }
        })
        .collect();

    for lin in 0..total {
        let mut rem = lin;
        let mut out_coord = vec![0usize; out_shape.len()];
        for (k, oc) in out_coord.iter_mut().enumerate() {
            *oc = rem / out_strides[k];
            rem %= out_strides[k];
        }

        let mut src_idx = vec![0usize; indices.len()];
        let mut cursor = 0usize;
        for oa in &out_axes {
            match oa {
                OutAxis::Slice(d) => {
                    if let AxisPlan::Slice(idxs) = &axes[*d] {
                        src_idx[*d] = idxs[out_coord[cursor]];
                    }
                    cursor += 1;
                }
                OutAxis::AdvBlock => {
                    let adv_coord = &out_coord[cursor..cursor + adv_ndim];
                    for (j, &ai) in adv_axes.iter().enumerate() {
                        if let AxisPlan::Adv {
                            shape: sh, flat, ..
                        } = &axes[ai]
                        {
                            let off = adv_ndim - sh.len();
                            let mut fidx = 0usize;
                            for (dd, &sd) in sh.iter().enumerate() {
                                let c = if sd == 1 { 0 } else { adv_coord[off + dd] };
                                fidx += c * adv_strides[j][dd];
                            }
                            if !flat.is_empty() && fidx < flat.len() {
                                src_idx[ai] = flat[fidx];
                            }
                        }
                    }
                    cursor += adv_ndim;
                }
            }
        }

        let mut flat_idx = 0;
        for (d, &si) in src_idx.iter().enumerate() {
            flat_idx += si * strides[d];
        }

        if flat_idx < data.len() {
            data[flat_idx] = if broadcast { values[0] } else { values[lin] };
        }
    }
}

fn calculate_target_count(indices: &[IndexDesc]) -> usize {
    for idx in indices {
        if let IndexDesc::FullBoolMask(v) = idx {
            return v.len();
        }
    }

    enum AxisPlan {
        Slice(Vec<usize>),
        #[allow(dead_code)]
        Adv {
            shape: Vec<usize>,
            flat: Vec<usize>,
        },
    }
    let axes: Vec<AxisPlan> = indices
        .iter()
        .map(|idx| match idx {
            IndexDesc::Slice(a, b, c) => AxisPlan::Slice(slice_indices_vec(*a, *b, *c)),
            IndexDesc::Int(i) => AxisPlan::Adv {
                shape: vec![],
                flat: vec![*i],
            },
            IndexDesc::Fancy(v) => AxisPlan::Adv {
                shape: vec![v.len()],
                flat: v.clone(),
            },
            IndexDesc::FancyMulti(v, sh) => AxisPlan::Adv {
                shape: sh.clone(),
                flat: v.clone(),
            },
            IndexDesc::FullBoolMask(_) => unreachable!(),
        })
        .collect();

    let adv_axes: Vec<usize> = axes
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a, AxisPlan::Adv { .. }))
        .map(|(i, _)| i)
        .collect();

    let mut adv_shape: Vec<usize> = vec![];
    for &ai in &adv_axes {
        if let AxisPlan::Adv { shape: sh, .. } = &axes[ai] {
            adv_shape = broadcast_shapes(&adv_shape, sh)
                .ok_or_else(|| PyValueError::new_err("shape mismatch in advanced index"))
                .unwrap();
        }
    }

    let adv_size: usize = adv_shape.iter().product();

    let slice_size: usize = axes
        .iter()
        .filter(|a| matches!(a, AxisPlan::Slice(_)))
        .map(|a| {
            if let AxisPlan::Slice(v) = a {
                v.len()
            } else {
                1
            }
        })
        .product();

    adv_size * slice_size
}

#[pyfunction]
pub fn iscomplex_cpx(data: Vec<Py<PyAny>>, py: Python<'_>) -> PyResult<NdArray> {
    let mask: Vec<f64> = data
        .iter()
        .map(|item| {
            let obj = item.bind(py);
            let imag: f64 = obj.getattr("imag").unwrap().extract().unwrap();
            if imag.abs() > 1e-12 { 1.0 } else { 0.0 }
        })
        .collect();

    let arr = Array::from_shape_vec(IxDyn(&[data.len()]), mask)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

/// 布尔掩码选择：掩码覆盖数据前若干维，`block` 为剩余维度元素个数（尾块大小）。
/// 对每个为真的掩码位，按 C 序取出对应的 `block` 个连续元素，拼成扁平结果。
/// 将原先 Python 逐元素循环下沉到 Rust，百万级规模从数百毫秒降到亚毫秒。
#[pyfunction]
pub fn masked_select(a: &NdArray, mask: &NdArray, block: usize) -> PyResult<NdArray> {
    let a_std = a.data.as_standard_layout();
    let data = a_std.as_slice().unwrap();
    let mask_std = mask.data.as_standard_layout();
    let mask_data = mask_std.as_slice().unwrap();

    let selected = mask_data.iter().filter(|&&m| m != 0.0).count();
    let mut out: Vec<f64> = Vec::with_capacity(selected * block.max(1));
    if block == 1 {
        for (&m, &v) in mask_data.iter().zip(data.iter()) {
            if m != 0.0 {
                out.push(v);
            }
        }
    } else {
        for (i, &m) in mask_data.iter().enumerate() {
            if m != 0.0 {
                let start = i * block;
                out.extend_from_slice(&data[start..start + block]);
            }
        }
    }

    let out_len = out.len();
    let arr = Array::from_shape_vec(IxDyn(&[out_len]), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

use ndarray::{Array, IxDyn};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyTuple};

use crate::NdArray;

/// 索引描述符：每个维度上的索引方式
#[derive(Clone)]
enum IndexDesc {
    /// 切片 (start, stop, step)，已处理负索引
    Slice(isize, isize, isize),
    /// 整数索引，已处理负索引
    Int(usize),
    /// 花式索引（整数列表）
    Fancy(Vec<usize>),
    /// 花式索引来自 ndim>1 的 ndarray（ix_ 风格 → 笛卡尔积）
    FancyMulti(Vec<usize>),
}

/// 解析 Python 索引对象，返回 IndexDesc
fn parse_single_index(item: &Bound<'_, PyAny>, dim_size: isize) -> PyResult<IndexDesc> {
    // 先尝试解析为 ndarray（必须在 int 之前，否则 __float__ 会干扰）
    if let Ok(arr) = item.extract::<NdArray>() {
        let vals: Vec<f64> = arr.data.iter().copied().collect();
        let fancy: Vec<usize> = vals.iter().map(|&v| {
            let iv = v as isize;
            if iv < 0 { (dim_size + iv) as usize } else { iv as usize }
        }).collect();
        let ndim = arr.data.ndim();
        if ndim > 1 {
            return Ok(IndexDesc::FancyMulti(fancy));
        }
        return Ok(IndexDesc::Fancy(fancy));
    }

    // 尝试解析为 slice
    if let Ok(slice_obj) = item.cast::<PySlice>() {
        let start = slice_obj
            .getattr("start")?
            .extract::<Option<isize>>()?
            .unwrap_or(0);
        let stop = slice_obj
            .getattr("stop")?
            .extract::<Option<isize>>()?
            .unwrap_or(dim_size);
        let step = slice_obj
            .getattr("step")?
            .extract::<Option<isize>>()?
            .unwrap_or(1);

        let actual_start = if start < 0 { (dim_size + start).max(0) } else { start.min(dim_size) };
        let actual_stop = if stop < 0 { (dim_size + stop).max(0) } else { stop.min(dim_size) };

        return Ok(IndexDesc::Slice(actual_start, actual_stop, step));
    }

    // 尝试解析为整数
    if let Ok(idx) = item.extract::<isize>() {
        let actual_idx = if idx < 0 { (dim_size + idx) as usize } else { idx as usize };
        return Ok(IndexDesc::Int(actual_idx));
    }

    // 尝试解析为列表（整数列表）
    if let Ok(list) = item.cast::<PyList>() {
        // 先判断是否为布尔列表
        let mut is_bool = true;
        for e in list.iter() {
            if e.extract::<bool>().is_err() {
                is_bool = false;
                break;
            }
        }
        if is_bool && list.len() > 0 {
            let mut fancy: Vec<usize> = Vec::new();
            for (j, e) in list.iter().enumerate() {
                if e.extract::<bool>().unwrap_or(false) {
                    fancy.push(j);
                }
            }
            return Ok(IndexDesc::Fancy(fancy));
        }
        // 整数列表
        let mut fancy: Vec<usize> = Vec::new();
        for e in list.iter() {
            let v = e.extract::<isize>()?;
            let actual = if v < 0 { (dim_size + v) as usize } else { v as usize };
            fancy.push(actual);
        }
        return Ok(IndexDesc::Fancy(fancy));
    }

    // 以上都不匹配
    Err(PyTypeError::new_err(format!(
        "Unsupported index type: {}",
        item.get_type().name()?
    )))
}

/// 展开 Ellipsis 并解析索引元组，返回 IndexDesc 列表
///
/// Python 侧已展开 Ellipsis 为切片再调用此函数
/// 此函数只处理元组索引的解析
fn parse_indices(
    key: &Bound<'_, PyAny>,
    shape: &[usize],
) -> PyResult<Vec<IndexDesc>> {
    let key_tuple = key.cast::<PyTuple>()?;

    // Python 侧已经展开了 Ellipsis，这里直接解析
    let mut indices: Vec<IndexDesc> = Vec::new();
    for (i, item) in key_tuple.iter().enumerate() {
        let dim_size = shape.get(i).copied().unwrap_or(0) as isize;
        indices.push(parse_single_index(&item, dim_size)?);
    }

    Ok(indices)
}

/// 判断是否为纯整数索引（返回标量）
fn is_all_int(indices: &[IndexDesc]) -> bool {
    indices.iter().all(|idx| matches!(idx, IndexDesc::Int(_)))
}

/// 判断是否包含花式索引（列表/ndarray）
fn has_fancy(indices: &[IndexDesc]) -> bool {
    indices.iter().any(|idx| matches!(idx, IndexDesc::Fancy(_)) || matches!(idx, IndexDesc::FancyMulti(_)))
}

/// 判断是否包含切片
fn has_slice(indices: &[IndexDesc]) -> bool {
    indices.iter().any(|idx| matches!(idx, IndexDesc::Slice(_, _, _)))
}

/// 判断是否存在 ix_ 风格（来自多维 ndarray）
fn has_fancy_multi(indices: &[IndexDesc]) -> bool {
    indices.iter().any(|idx| matches!(idx, IndexDesc::FancyMulti(_)))
}

/// 提取每个维度的索引列表（用于笛卡尔积或配对）
fn build_dim_lists(indices: &[IndexDesc]) -> Vec<Vec<usize>> {
    indices.iter().map(|idx| match idx {
        IndexDesc::Fancy(v) | IndexDesc::FancyMulti(v) => v.clone(),
        IndexDesc::Slice(start, stop, step) => {
            if *step > 0 {
                let mut v: Vec<usize> = Vec::new();
                let mut i = *start;
                while i < *stop {
                    v.push(i as usize);
                    i += *step;
                }
                v
            } else if *step < 0 {
                let mut v: Vec<usize> = Vec::new();
                let mut i = *start;
                while i > *stop {
                    v.push(i as usize);
                    i += *step;
                }
                v
            } else {
                vec![]
            }
        }
        IndexDesc::Int(idx) => vec![*idx],
    }).collect()
}

/// 纯花式索引：逐元素配对（如 x[[0,1,2], [0,1,0]]）
fn fancy_pairwise(a: &Array<f64, IxDyn>, dim_lists: &[Vec<usize>]) -> Array<f64, IxDyn> {
    let n = dim_lists[0].len();
    let mut result_vals: Vec<f64> = Vec::new();
    for i in 0..n {
        let mut cur = a.clone();
        for d in 0..dim_lists.len() {
            let idx = dim_lists[d][i];
            cur = cur.index_axis(ndarray::Axis(0), idx).to_owned().into_dyn();
        }
        result_vals.push(*cur.first().unwrap_or(&0.0));
    }
    Array::from_shape_vec(IxDyn(&[n]), result_vals)
        .unwrap_or_else(|_| Array::from_elem(IxDyn(&[0]), 0.0))
}

/// 笛卡尔积花式索引
fn fancy_cartesian(a: &Array<f64, IxDyn>, dim_lists: &[Vec<usize>]) -> Vec<f64> {
    let mut result_vals: Vec<f64> = Vec::new();

    fn recurse(a: &Array<f64, IxDyn>, dim_lists: &[Vec<usize>], depth: usize,
               coords: &[usize], result_vals: &mut Vec<f64>) {
        if depth == dim_lists.len() {
            let mut cur = a.clone();
            for &c in coords {
                cur = cur.index_axis(ndarray::Axis(0), c).to_owned().into_dyn();
            }
            result_vals.push(*cur.first().unwrap_or(&0.0));
            return;
        }
        for &val in &dim_lists[depth] {
            let mut new_coords = coords.to_vec();
            new_coords.push(val);
            recurse(a, dim_lists, depth + 1, &new_coords, result_vals);
        }
    }

    recurse(a, dim_lists, 0, &[], &mut result_vals);
    result_vals
}

/// 对纯 slice+int 混合索引进行切片
fn slice_and_int_index(a: &Array<f64, IxDyn>, indices: &[IndexDesc]) -> PyResult<Array<f64, IxDyn>> {
    let mut cur = a.clone();
    for (dim, idx) in indices.iter().enumerate().rev() {
        match idx {
            IndexDesc::Slice(start, stop, step) => {
                let s = ndarray::Slice { start: *start, end: Some(*stop), step: *step };
                let dim_axis = ndarray::Axis(dim);
                cur = cur.slice_axis(dim_axis, s).into_owned().into_dyn();
            }
            IndexDesc::Int(i) => {
                cur = cur.index_axis(ndarray::Axis(dim), *i).to_owned().into_dyn();
            }
            _ => unreachable!(),
        }
    }
    Ok(cur)
}

/// 主索引函数：处理所有索引类型
///
/// Python 调用: _core.getitem_multi(a, key_tuple, shape)
/// - a: NdArray 对象
/// - key_tuple: Python 元组（Ellipsis 已在 Python 侧展开）
/// - shape: 数组形状
#[pyfunction]
pub fn getitem_multi(
    a: &NdArray,
    key: &Bound<'_, PyAny>,
    shape: Vec<usize>,
) -> PyResult<NdArray> {
    let indices = parse_indices(key, &shape)?;

    // 填充缺失维度：如果索引数少于形状维度，补充完整切片
    let mut filled_indices = indices;
    let ndim = shape.len();
    while filled_indices.len() < ndim {
        let dim_size = shape[filled_indices.len()] as isize;
        filled_indices.push(IndexDesc::Slice(0, dim_size, 1));
    }

    // 情况 1: 纯整数索引 → 标量
    if is_all_int(&filled_indices) {
        let mut cur = a.data.clone();
        for idx in &filled_indices {
            if let IndexDesc::Int(i) = idx {
                cur = cur.index_axis(ndarray::Axis(0), *i).to_owned().into_dyn();
            }
        }
        return Ok(NdArray { data: cur });
    }

    // 情况 2: 无花式索引（纯 slice + 整数混合）
    let has_fancy_flag = has_fancy(&filled_indices);
    let has_slice_flag = has_slice(&filled_indices);

    if !has_fancy_flag {
        let result = slice_and_int_index(&a.data, &filled_indices)?;
        return Ok(NdArray { data: result });
    }

    // 有花式索引
    let dim_lists = build_dim_lists(&filled_indices);
    let ix_style = has_fancy_multi(&filled_indices);

    if !has_slice_flag && !ix_style {
        // 纯花式索引
        if dim_lists.len() > 1 && dim_lists[1..].iter().all(|dl| dl.len() == dim_lists[0].len()) {
            let result_vals = fancy_pairwise(&a.data, &dim_lists);
            Ok(NdArray { data: result_vals })
        } else {
            let result_vals = fancy_cartesian(&a.data, &dim_lists);
            let out_shape: Vec<usize> = dim_lists.iter().map(|dl| dl.len()).collect();
            let arr = Array::from_shape_vec(IxDyn(&out_shape), result_vals)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: arr })
        }
    } else {
        // 混合索引（slice + fancy）：笛卡尔积
        let result_vals = fancy_cartesian(&a.data, &dim_lists);
        let out_shape: Vec<usize> = dim_lists.iter().map(|dl| dl.len()).collect();
        let arr = Array::from_shape_vec(IxDyn(&out_shape), result_vals)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }
}

/// 获取元素（供标量索引用）
#[pyfunction]
pub fn getitem_scalar(a: &NdArray, indices: Vec<isize>) -> PyResult<f64> {
    let mut cur = a.data.clone();
    for &idx in &indices {
        let dim = cur.shape()[0] as isize;
        let actual = if idx < 0 { dim + idx } else { idx };
        if actual < 0 || actual >= dim {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        cur = cur.index_axis(ndarray::Axis(0), actual as usize).to_owned().into_dyn();
    }
    Ok(*cur.first().unwrap_or(&0.0))
}

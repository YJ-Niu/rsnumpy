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
    FancyMulti(Vec<usize>),
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
        return Ok(ndarray_to_index_desc(&NdArray { data: arr }, dim_size));
    }

    if let Ok(arr) = item.extract::<NdArray>() {
        return Ok(ndarray_to_index_desc(&arr, dim_size));
    }

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

        let actual_start = if start < 0 {
            (dim_size + start).max(0)
        } else {
            start.min(dim_size)
        };
        let actual_stop = if stop < 0 {
            (dim_size + stop).max(0)
        } else {
            stop.min(dim_size)
        };

        return Ok(IndexDesc::Slice(actual_start, actual_stop, step));
    }

    Err(PyTypeError::new_err(format!(
        "Unsupported index type: {}",
        item.get_type().name()?
    )))
}

fn ndarray_to_index_desc(arr: &NdArray, dim_size: isize) -> IndexDesc {
    let vals: Vec<f64> = arr.data.iter().copied().collect();
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
        IndexDesc::FancyMulti(fancy)
    } else {
        IndexDesc::Fancy(fancy)
    }
}

fn parse_indices(key: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<Vec<IndexDesc>> {
    let key_tuple = key.cast::<PyTuple>()?;

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
    indices
        .iter()
        .any(|idx| matches!(idx, IndexDesc::Fancy(_) | IndexDesc::FancyMulti(_)))
}

fn has_slice(indices: &[IndexDesc]) -> bool {
    indices
        .iter()
        .any(|idx| matches!(idx, IndexDesc::Slice(_, _, _)))
}

fn has_fancy_multi(indices: &[IndexDesc]) -> bool {
    indices
        .iter()
        .any(|idx| matches!(idx, IndexDesc::FancyMulti(_)))
}

fn build_dim_lists(indices: &[IndexDesc]) -> Vec<Vec<usize>> {
    indices
        .iter()
        .map(|idx| match idx {
            IndexDesc::Fancy(v) | IndexDesc::FancyMulti(v) => v.clone(),
            IndexDesc::Slice(start, stop, step) => {
                if *step > 0 {
                    (*start..*stop)
                        .step_by(*step as usize)
                        .map(|i| i as usize)
                        .collect()
                } else if *step < 0 {
                    (*start..*stop)
                        .rev()
                        .step_by((-*step) as usize)
                        .map(|i| i as usize)
                        .collect()
                } else {
                    vec![]
                }
            }
            IndexDesc::Int(idx) => vec![*idx],
        })
        .collect()
}

fn fancy_pairwise(a: &Array<f64, IxDyn>, dim_lists: &[Vec<usize>]) -> Array<f64, IxDyn> {
    let n = dim_lists[0].len();
    let strides = compute_strides(a.shape());

    let result_vals: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut flat_idx = 0;
            for (d, dl) in dim_lists.iter().enumerate() {
                flat_idx += dl[i] * strides[d];
            }
            a.as_slice_memory_order().unwrap()[flat_idx]
        })
        .collect();

    Array::from_shape_vec(IxDyn(&[n]), result_vals)
        .unwrap_or_else(|_| Array::from_elem(IxDyn(&[0]), 0.0))
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn fancy_cartesian(a: &Array<f64, IxDyn>, dim_lists: &[Vec<usize>]) -> Vec<f64> {
    let strides = compute_strides(a.shape());
    let data = a.as_slice_memory_order().unwrap();

    let mut indices: Vec<usize> = vec![0; dim_lists.len()];
    let total_result: usize = dim_lists.iter().map(|dl| dl.len()).product();
    let mut result = Vec::with_capacity(total_result);

    loop {
        let mut flat_idx = 0;
        for (i, &idx) in indices.iter().enumerate() {
            flat_idx += dim_lists[i][idx] * strides[i];
        }
        result.push(data[flat_idx]);

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

    result
}

fn slice_and_int_index(
    a: &Array<f64, IxDyn>,
    indices: &[IndexDesc],
) -> PyResult<Array<f64, IxDyn>> {
    let mut cur = a.clone();
    for (dim, idx) in indices.iter().enumerate().rev() {
        match idx {
            IndexDesc::Slice(start, stop, step) => {
                let s = Slice {
                    start: *start,
                    end: Some(*stop),
                    step: *step,
                };
                let dim_axis = ndarray::Axis(dim);
                cur = cur.slice_axis(dim_axis, s).into_owned().into_dyn();
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

#[pyfunction]
pub fn getitem_multi(a: &NdArray, key: &Bound<'_, PyAny>, shape: Vec<usize>) -> PyResult<NdArray> {
    let indices = parse_indices(key, &shape)?;

    let mut filled_indices = indices;
    let ndim = shape.len();
    while filled_indices.len() < ndim {
        let dim_size = shape[filled_indices.len()] as isize;
        filled_indices.push(IndexDesc::Slice(0, dim_size, 1));
    }

    if is_all_int(&filled_indices) {
        let strides = compute_strides(&shape);
        let data = a.data.as_slice_memory_order().unwrap();

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

        let val = data[flat_idx];
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), val),
        });
    }

    let has_fancy_flag = has_fancy(&filled_indices);
    let has_slice_flag = has_slice(&filled_indices);

    if !has_fancy_flag {
        let result = slice_and_int_index(&a.data, &filled_indices)?;
        return Ok(NdArray { data: result });
    }

    let dim_lists = build_dim_lists(&filled_indices);
    let ix_style = has_fancy_multi(&filled_indices);

    if !has_slice_flag && !ix_style {
        if dim_lists.len() > 1
            && dim_lists[1..]
                .iter()
                .all(|dl| dl.len() == dim_lists[0].len())
        {
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
        let result_vals = fancy_cartesian(&a.data, &dim_lists);
        let out_shape: Vec<usize> = dim_lists.iter().map(|dl| dl.len()).collect();
        let arr = Array::from_shape_vec(IxDyn(&out_shape), result_vals)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }
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

    let dim_lists = build_dim_lists(&filled_indices);
    let strides = compute_strides(&shape);

    let val: f64 = if let Ok(v) = value.extract::<f64>() {
        v
    } else if let Ok(v) = value.extract::<i32>() {
        v as f64
    } else if let Ok(v) = value.extract::<bool>() {
        if v { 1.0 } else { 0.0 }
    } else {
        return Err(PyTypeError::new_err(
            "Unsupported value type for assignment",
        ));
    };

    let mut a_borrow = a.borrow_mut();
    let data = a_borrow.data.as_slice_memory_order_mut().unwrap();

    let mut indices = vec![0; dim_lists.len()];

    loop {
        let mut flat_idx = 0;
        for (i, &idx) in indices.iter().enumerate() {
            flat_idx += dim_lists[i][idx] * strides[i];
        }
        if flat_idx < data.len() {
            data[flat_idx] = val;
        }

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

    Ok(())
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
    Ok(NdArray { data: arr })
}

use crate::*;

#[pyfunction]
#[pyo3(name = "where", signature = (condition, x=None, y=None))]
fn where_<'py>(
    py: Python<'py>,
    condition: &NdArray,
    x: Option<&NdArray>,
    y: Option<&NdArray>,
) -> PyResult<Bound<'py, PyAny>> {
    match (x, y) {
        (None, None) => {
            let shape = condition.data.shape();
            let ndim = shape.len();
            let total_elems = condition.data.len();
            let mut indices: Vec<Vec<usize>> = (0..ndim.max(1))
                .map(|_| Vec::with_capacity(total_elems))
                .collect();
            for (flat_idx, &val) in condition.data.iter().enumerate() {
                if val != 0.0 {
                    if ndim == 0 {
                        indices[0].push(0);
                    } else {
                        let mut remaining = flat_idx;
                        for d in (0..ndim).rev() {
                            let dim_size = shape[d];
                            indices[d].push(remaining % dim_size);
                            remaining /= dim_size;
                        }
                    }
                }
            }
            let tuples: Vec<Bound<'py, PyAny>> = indices
                .into_iter()
                .map(|idx_vec| {
                    let data: Vec<f64> = idx_vec.iter().map(|&i| i as f64).collect();
                    let arr = Array::from_shape_vec(IxDyn(&[data.len()]), data).unwrap();
                    Bound::new(
                        py,
                        NdArray {
                            imag: None,
                            data: arr,
                        },
                    )
                    .unwrap()
                    .into_any()
                })
                .collect();
            Ok(PyTuple::new(py, tuples)?.into_any())
        }
        (Some(xv), Some(yv)) => {
            let cond_vec: Vec<f64> = condition.data.iter().copied().collect();
            let x_vec: Vec<f64> = xv.data.iter().copied().collect();
            let y_vec: Vec<f64> = yv.data.iter().copied().collect();
            let n = cond_vec.len();
            let result: Vec<f64> = if n >= crate::par_threshold_cheap() {
                crate::threadpool::with_pool(|| {
                    cond_vec
                        .into_par_iter()
                        .zip(x_vec.into_par_iter().zip(y_vec.into_par_iter()))
                        .map(|(c, (xv, yv))| if c != 0.0 { xv } else { yv })
                        .collect()
                })
            } else {
                cond_vec
                    .into_iter()
                    .zip(x_vec.into_iter().zip(y_vec))
                    .map(|(c, (xv, yv))| if c != 0.0 { xv } else { yv })
                    .collect()
            };
            let arr = Array::from_shape_vec(condition.data.dim(), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let nd = NdArray {
                imag: None,
                data: arr.into_dyn(),
            };
            let bound = Bound::new(py, nd)?;
            Ok(bound.into_any())
        }
        _ => Err(PyTypeError::new_err(
            "Both x and y must be provided together",
        )),
    }
}

#[pyfunction]
fn extract(condition: &NdArray, a: &NdArray) -> PyResult<NdArray> {
    let cond_flat: Vec<f64> = condition.data.iter().copied().collect();
    let data_flat: Vec<f64> = a.data.iter().copied().collect();

    let min_len = cond_flat.len().min(data_flat.len());
    let mut result = Vec::with_capacity(min_len);

    for i in 0..min_len {
        if cond_flat[i] != 0.0 {
            result.push(data_flat[i]);
        }
    }

    let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
fn nonzero(a: &NdArray) -> Vec<Vec<usize>> {
    a.nonzero()
}

/// 直接在 Rust 中将 nonzero 结果转换为 NdArray 数组，避免 Python 层 for 循环
#[pyfunction]
fn nonzero_arrs<'py>(py: Python<'py>, a: &NdArray) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let raw = a.nonzero();
    let mut result = Vec::with_capacity(raw.len());
    for indices in raw {
        let data: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&[data.len()]), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nd = NdArray {
            imag: None,
            data: arr.into_dyn(),
        };
        result.push(Bound::new(py, nd)?.into_any());
    }
    Ok(result)
}

/// ix_ 的 Rust 实现：从多个序列构造开放网格，消除 Python 层 for 循环
#[pyfunction]
fn ix_rs<'py>(py: Python<'py>, args: &Bound<'_, PyTuple>) -> PyResult<Bound<'py, PyTuple>> {
    let n = args.len();
    if n == 0 {
        return PyTuple::new(py, Vec::<Bound<'py, PyAny>>::new());
    }
    let mut arrays: Vec<NdArray> = Vec::with_capacity(n);
    for i in 0..n {
        let item = args.get_item(i)?;
        // 直接解析 Python list/tuple 为 NdArray（支持任意可迭代数据）
        let (values, shape) = parse_py_list_to_flat(&item)?;
        let nd_shape = if shape.is_empty() {
            IxDyn(&[])
        } else {
            IxDyn(&shape)
        };
        let arr = Array::from_shape_vec(nd_shape, values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        arrays.push(NdArray {
            imag: None,
            data: arr,
        });
    }
    let mut results: Vec<Bound<'py, PyAny>> = Vec::with_capacity(n);
    for i in 0..n {
        let size = arrays[i].data.len();
        let mut shape = vec![1usize; n];
        shape[i] = size;
        let reshaped = arrays[i]
            .data
            .clone()
            .into_shape_with_order(IxDyn(&shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        results.push(
            Bound::new(
                py,
                NdArray {
                    imag: None,
                    data: reshaped.into_dyn(),
                },
            )?
            .into_any(),
        );
    }
    PyTuple::new(py, results)
}

/// 从 shape 直接创建 arange 数组元组（用于 fromfunction 等），避免 Python 层 list(range(d)) 循环
#[pyfunction]
fn _arange_arrays<'py>(py: Python<'py>, shape: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let s = shape_to_vec(shape)?;
    let mut arrays: Vec<Bound<'py, PyAny>> = Vec::with_capacity(s.len());
    for &dim in &s {
        let data: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&[dim]), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nd = NdArray {
            imag: None,
            data: arr.into_dyn(),
        };
        arrays.push(Bound::new(py, nd)?.into_any());
    }
    PyTuple::new(py, arrays)
}

/// select 的 Rust 实现 — 单次遍历多条件选择，避免 Python 层 for 循环多次调用 where
#[pyfunction]
#[pyo3(signature = (condlist, choicelist, default=0.0))]
fn select_rs<'py>(
    py: Python<'py>,
    condlist: &Bound<'_, PyList>,
    choicelist: &Bound<'_, PyList>,
    default: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let n = condlist.len();
    if n == 0 {
        return Err(PyValueError::new_err("condlist must not be empty"));
    }
    if choicelist.len() != n {
        return Err(PyValueError::new_err(
            "condlist and choicelist must have the same length",
        ));
    }

    // 提取条件和选择数组（保持和 Python 原有代码一致的语义）
    let mut conds: Vec<NdArray> = Vec::with_capacity(n);
    let mut choices: Vec<NdArray> = Vec::with_capacity(n);

    // 先用第一个条件确定 shape
    let first_cond: NdArray = condlist.get_item(0)?.extract::<NdArray>()?;
    let result_shape = first_cond.data.shape().to_vec();
    let total_size: usize = result_shape.iter().product();
    conds.push(first_cond);

    for i in 1..n {
        conds.push(condlist.get_item(i)?.extract::<NdArray>()?);
    }
    for i in 0..n {
        choices.push(choicelist.get_item(i)?.extract::<NdArray>()?);
    }

    // 用第一个条件的 shape 初始化结果（默认值填充）
    let mut result = vec![default; total_size];

    // 遍历每个 (cond, choice) 对，最后一个匹配的条件获胜（与原有 Python 语义一致）
    for (cond, choice) in conds.iter().zip(choices.iter()) {
        let cond_slice = cond
            .data
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("Non-contiguous condition array"))?;
        let choice_slice = choice
            .data
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("Non-contiguous choice array"))?;
        let min_len = total_size.min(cond_slice.len()).min(choice_slice.len());
        for i in 0..min_len {
            if cond_slice[i] != 0.0 {
                result[i] = choice_slice[i];
            }
        }
    }

    let arr = Array::from_shape_vec(IxDyn(&result_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let nd = NdArray {
        imag: None,
        data: arr.into_dyn(),
    };
    let bound = Bound::new(py, nd)?;
    Ok(bound.into_any())
}

#[pyfunction]
fn argwhere(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape();
    let ndim = shape.len();
    if ndim == 0 {
        if a.data.iter().any(|&v| v != 0.0) {
            let arr = Array::from_shape_vec(IxDyn(&[1, 0]), vec![])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(NdArray {
                imag: None,
                data: arr,
            });
        }
        let arr = Array::from_shape_vec(IxDyn(&[0, 0]), vec![])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            imag: None,
            data: arr,
        });
    }
    let mut indices: Vec<Vec<usize>> = (0..ndim)
        .map(|_| Vec::with_capacity(a.data.len()))
        .collect();
    for (flat_idx, &val) in a.data.iter().enumerate() {
        if val != 0.0 {
            let mut remaining = flat_idx;
            for d in (0..ndim).rev() {
                let dim_size = shape[d];
                indices[d].push(remaining % dim_size);
                remaining /= dim_size;
            }
        }
    }
    let num_nonzero = indices[0].len();
    let mut flat_result = Vec::with_capacity(num_nonzero * ndim);
    for i in 0..num_nonzero {
        for row in indices.iter() {
            flat_result.push(row[i] as f64);
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[num_nonzero, ndim]), flat_result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
fn count_nonzero(a: &NdArray) -> usize {
    a.data.iter().filter(|&&v| v != 0.0).count()
}

#[pyfunction]
fn flatnonzero(a: &NdArray) -> PyResult<NdArray> {
    let indices: Vec<f64> = a
        .data
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0.0)
        .map(|(i, _)| i as f64)
        .collect();
    let arr = Array::from_shape_vec(IxDyn(&[indices.len()]), indices)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(where_, m)?)?;
    m.add_function(wrap_pyfunction!(nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(nonzero_arrs, m)?)?;
    m.add_function(wrap_pyfunction!(argwhere, m)?)?;
    m.add_function(wrap_pyfunction!(flatnonzero, m)?)?;
    m.add_function(wrap_pyfunction!(count_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    m.add_function(wrap_pyfunction!(select_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ix_rs, m)?)?;
    m.add_function(wrap_pyfunction!(_arange_arrays, m)?)?;
    Ok(())
}

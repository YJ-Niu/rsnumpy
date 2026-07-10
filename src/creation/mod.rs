use crate::*;

/// 在 Rust 中单次完成展平 + dtype 推断，构造数组并返回 dtype 编码。
/// 非数值/不规则数据返回错误，由 Python 侧回退。
#[pyfunction]
fn build_array(data: &Bound<'_, PyAny>) -> PyResult<(NdArray, u8)> {
    let (values, shape, flags) = parse_py_categorized(data)?;
    let arr = if shape.is_empty() {
        Array::from_shape_vec(IxDyn(&[]), values)
    } else {
        Array::from_shape_vec(IxDyn(&shape), values)
    }
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((NdArray { data: arr }, flags.dtype_code()))
}

#[pyfunction]
fn array(data: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    NdArray::new(data)
}

#[pyfunction]
fn zeros(shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let s = shape_to_vec(shape)?;
    Ok(NdArray {
        data: Array::zeros(IxDyn(&s)),
    })
}

#[pyfunction]
fn ones(shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let s = shape_to_vec(shape)?;
    Ok(NdArray {
        data: Array::ones(IxDyn(&s)),
    })
}

#[pyfunction]
#[pyo3(signature = (n, m=None, k=0))]
fn eye(n: usize, m: Option<usize>, k: i32) -> PyResult<NdArray> {
    let cols = m.unwrap_or(n);
    let mut data = vec![0.0_f64; n * cols];
    if k >= 0 {
        let k_u = k as usize;
        for i in 0..n {
            let j = i + k_u;
            if j < cols {
                data[i * cols + j] = 1.0;
            }
        }
    } else {
        let k_u = (-k) as usize;
        for j in 0..cols {
            let i = j + k_u;
            if i < n {
                data[i * cols + j] = 1.0;
            }
        }
    }
    let arr =
        Array::from_shape_vec((n, cols), data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        data: arr.into_dyn(),
    })
}

#[pyfunction]
#[pyo3(signature = (start, stop, step=1.0))]
fn arange(start: f64, stop: f64, step: f64) -> PyResult<NdArray> {
    if step == 0.0 {
        return Err(PyValueError::new_err("Step cannot be zero"));
    }
    let n = if step > 0.0 {
        ((stop - start) / step).ceil() as usize
    } else {
        ((start - stop) / (-step)).ceil() as usize
    };
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let val = start + step * i as f64;
        if step > 0.0 {
            if val < stop {
                values.push(val);
            } else {
                break;
            }
        } else {
            if val > stop {
                values.push(val);
            } else {
                break;
            }
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[values.len()]), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=true))]
fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> PyResult<NdArray> {
    if num == 0 {
        return Err(PyValueError::new_err("Number of samples must be positive"));
    }
    if num == 1 {
        return Ok(NdArray {
            data: Array::from_shape_vec(IxDyn(&[1]), vec![start])
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        });
    }
    let step = if endpoint {
        (stop - start) / (num - 1) as f64
    } else {
        (stop - start) / num as f64
    };
    let values: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
    let arr = Array::from_shape_vec(IxDyn(&[num]), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
fn full(shape: &Bound<'_, PyAny>, fill_value: f64) -> PyResult<NdArray> {
    let s = shape_to_vec(shape)?;
    Ok(NdArray {
        data: Array::from_elem(IxDyn(&s), fill_value),
    })
}

#[pyfunction]
fn empty(shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let s = shape_to_vec(shape)?;
    let size: usize = s.iter().product();
    // 仅分配内存，不初始化，比 zeros 更快
    let v = vec![0.0f64; size];
    let data =
        Array::from_shape_vec(IxDyn(&s), v).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data })
}

#[pyfunction]
fn zeros_like(a: &NdArray) -> PyResult<NdArray> {
    Ok(NdArray {
        data: Array::from_elem(a.data.shape(), 0.0),
    })
}

#[pyfunction]
fn ones_like(a: &NdArray) -> PyResult<NdArray> {
    Ok(NdArray {
        data: Array::from_elem(a.data.shape(), 1.0),
    })
}

#[pyfunction]
fn empty_like(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    let size: usize = shape.iter().product();
    let v = vec![0.0f64; size];
    let data = Array::from_shape_vec(IxDyn(&shape), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data })
}

#[pyfunction]
fn full_like(a: &NdArray, fill_value: f64) -> PyResult<NdArray> {
    Ok(NdArray {
        data: Array::from_elem(a.data.shape(), fill_value),
    })
}

#[pyfunction]
#[pyo3(signature = (*args, indexing="xy"))]
fn meshgrid<'py>(
    py: Python<'py>,
    args: &Bound<'_, PyTuple>,
    indexing: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let n = args.len();
    if n == 0 {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let mut arrays: Vec<NdArray> = Vec::with_capacity(n);
    for item in args.iter() {
        let arr = item.extract::<NdArray>()?;
        if arr.data.ndim() != 1 {
            return Err(PyValueError::new_err("All inputs must be 1-D arrays"));
        }
        arrays.push(arr);
    }
    let dims: Vec<usize> = arrays.iter().map(|a| a.data.len()).collect();
    let (swap_first_two, out_dims): (bool, Vec<usize>) = if indexing == "ij" {
        (false, dims.clone())
    } else {
        let mut out = dims.clone();
        if out.len() >= 2 {
            out.swap(0, 1);
        }
        (true, out)
    };
    let mut results: Vec<NdArray> = Vec::with_capacity(n);
    for (i, arr) in arrays.iter().enumerate() {
        let mut shape = vec![1usize; n];
        let actual_idx = if swap_first_two && n >= 2 {
            if i == 0 {
                1
            } else if i == 1 {
                0
            } else {
                i
            }
        } else {
            i
        };
        shape[actual_idx] = arr.data.len();
        let reshaped = arr
            .data
            .clone()
            .into_shape_with_order(IxDyn(&shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let broadcast = reshaped
            .broadcast(IxDyn(&out_dims))
            .ok_or_else(|| PyValueError::new_err("Broadcasting failed"))?
            .to_owned();
        results.push(NdArray {
            data: broadcast.into_dyn(),
        });
    }
    let py_results: Vec<Bound<'py, PyAny>> = results
        .into_iter()
        .map(|nd| Bound::new(py, nd).unwrap().into_any())
        .collect();
    Ok(PyTuple::new(py, py_results)?.into_any())
}

#[pyfunction]
fn logspace(start: f64, stop: f64, num: usize, base: f64) -> PyResult<NdArray> {
    linspace(start, stop, num, true).map(|nd| NdArray {
        data: nd.data.mapv(|v| base.powf(v)),
    })
}

#[pyfunction]
fn geomspace(start: f64, stop: f64, num: usize) -> PyResult<NdArray> {
    if num == 0 {
        return Err(PyValueError::new_err("Number of samples must be positive"));
    }
    if num == 1 {
        return Ok(NdArray {
            data: Array::from_shape_vec(IxDyn(&[1]), vec![start])
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        });
    }
    let log_start = start.ln();
    let log_stop = stop.ln();
    let step = (log_stop - log_start) / (num - 1) as f64;
    let values: Vec<f64> = (0..num)
        .map(|i| (log_start + i as f64 * step).exp())
        .collect();
    let arr = Array::from_shape_vec(IxDyn(&[num]), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(build_array, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(empty_like, m)?)?;
    m.add_function(wrap_pyfunction!(full_like, m)?)?;
    m.add_function(wrap_pyfunction!(logspace, m)?)?;
    m.add_function(wrap_pyfunction!(geomspace, m)?)?;
    m.add_function(wrap_pyfunction!(meshgrid, m)?)?;
    Ok(())
}

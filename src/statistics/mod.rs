use crate::*;

#[pyfunction]
#[pyo3(signature = (x, axis=None, keepdims=false))]
fn sum(_py: Python<'_>, x: &NdArray, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    _py.detach(move || x.sum(axis, keepdims))
}

#[pyfunction]
#[pyo3(signature = (x, axis=None, keepdims=false))]
fn prod(_py: Python<'_>, x: &NdArray, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    _py.detach(move || x.prod(axis, keepdims))
}

#[pyfunction]
#[pyo3(signature = (x, axis=None, keepdims=false))]
fn mean(_py: Python<'_>, x: &NdArray, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    _py.detach(move || x.mean(axis, keepdims))
}

#[pyfunction]
#[pyo3(signature = (x, axis=None, ddof=0, keepdims=false), name = "std")]
fn std_dev(
    _py: Python<'_>,
    x: &NdArray,
    axis: Option<isize>,
    ddof: usize,
    keepdims: bool,
) -> PyResult<NdArray> {
    _py.detach(move || x.std(axis, ddof, keepdims))
}

#[pyfunction]
#[pyo3(signature = (x, axis=None, ddof=0, keepdims=false))]
fn var(
    _py: Python<'_>,
    x: &NdArray,
    axis: Option<isize>,
    ddof: usize,
    keepdims: bool,
) -> PyResult<NdArray> {
    _py.detach(move || x.var(axis, ddof, keepdims))
}

#[pyfunction]
#[pyo3(signature = (x, axis=None, keepdims=false))]
fn min(_py: Python<'_>, x: &NdArray, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    _py.detach(move || x.min(axis, keepdims))
}

#[pyfunction]
#[pyo3(signature = (x, axis=None, keepdims=false))]
fn max(_py: Python<'_>, x: &NdArray, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    _py.detach(move || x.max(axis, keepdims))
}

#[pyfunction]
fn argmin(_py: Python<'_>, x: &NdArray) -> PyResult<usize> {
    _py.detach(move || x.argmin())
}

#[pyfunction]
fn argmax(_py: Python<'_>, x: &NdArray) -> PyResult<usize> {
    _py.detach(move || x.argmax())
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn median(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    match axis {
        None => {
            let mut values: Vec<f64> = x.data.iter().copied().collect();
            if values.is_empty() {
                return Err(PyValueError::new_err(
                    "Cannot compute median of empty array",
                ));
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let len = values.len();
            let median_val = if len.is_multiple_of(2) {
                (values[len / 2 - 1] + values[len / 2]) / 2.0
            } else {
                values[len / 2]
            };
            Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), median_val),
            })
        }
        Some(ax) => {
            let ndim = x.data.ndim();
            let ax = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };

            let nrows = x.data.shape()[0];
            let ncols = x.data.shape()[1];

            let result: Vec<f64> = if ax == 0 {
                (0..ncols)
                    .map(|c| {
                        let mut values: Vec<f64> = (0..nrows).map(|r| x.data[[r, c]]).collect();
                        values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let len = values.len();
                        if len.is_multiple_of(2) {
                            (values[len / 2 - 1] + values[len / 2]) / 2.0
                        } else {
                            values[len / 2]
                        }
                    })
                    .collect()
            } else {
                (0..nrows)
                    .map(|r| {
                        let mut values: Vec<f64> = (0..ncols).map(|c| x.data[[r, c]]).collect();
                        values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let len = values.len();
                        if len.is_multiple_of(2) {
                            (values[len / 2 - 1] + values[len / 2]) / 2.0
                        } else {
                            values[len / 2]
                        }
                    })
                    .collect()
            };

            let mut new_shape: Vec<usize> = x.data.shape().to_vec();
            new_shape.remove(ax);
            let result_arr = Array::from_shape_vec(IxDyn(&new_shape), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: result_arr,
            })
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, weights=None, returned=false))]
fn average(
    a: &NdArray,
    axis: Option<isize>,
    weights: Option<&NdArray>,
    returned: bool,
) -> PyResult<NdArray> {
    match axis {
        None => {
            let values: Vec<f64> = a.data.iter().copied().collect();

            let (result_val, sum_weights): (f64, f64) = match weights {
                None => {
                    let sum: f64 = values.iter().sum();
                    let avg = sum / values.len() as f64;
                    if returned {
                        (avg, values.len() as f64)
                    } else {
                        (avg, 0.0)
                    }
                }
                Some(w) => {
                    let w_values: Vec<f64> = w.data.iter().copied().collect();
                    if values.len() != w_values.len() {
                        return Err(PyValueError::new_err(
                            "weights must have the same length as data",
                        ));
                    }
                    let weighted_sum: f64 =
                        values.iter().zip(w_values.iter()).map(|(v, w)| v * w).sum();
                    let sum_w: f64 = w_values.iter().sum();
                    if sum_w == 0.0 {
                        return Err(PyValueError::new_err("sum of weights must not be zero"));
                    }
                    let avg = weighted_sum / sum_w;
                    if returned { (avg, sum_w) } else { (avg, 0.0) }
                }
            };

            if returned {
                let data = vec![result_val, sum_weights];
                Ok(NdArray {
                    imag: None,
                    data: Array::from_shape_vec(IxDyn(&[2]), data)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                })
            } else {
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), result_val),
                })
            }
        }
        Some(ax) => {
            let ndim = a.data.ndim();
            let ax = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };

            let w_values: Option<Vec<f64>> = weights.map(|w| w.data.iter().copied().collect());

            let axis_len = a.data.shape()[ax];
            let other_dims: Vec<usize> = a
                .data
                .shape()
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &s)| s)
                .collect();
            let num_other: usize = other_dims.iter().product();

            let mut avg_results: Vec<f64> = Vec::with_capacity(num_other);
            let mut sum_w_results: Vec<f64> = Vec::with_capacity(num_other);

            for idx in 0..num_other {
                let mut indices: Vec<usize> = Vec::with_capacity(ndim);
                let mut temp = idx;
                for &dim in other_dims.iter().rev() {
                    indices.push(temp % dim);
                    temp /= dim;
                }
                indices.reverse();

                let mut values: Vec<f64> = Vec::with_capacity(axis_len);
                for i in 0..axis_len {
                    let mut full_indices = indices.clone();
                    full_indices.insert(ax, i);
                    values.push(a.data[full_indices.as_slice()]);
                }

                let (avg, sum_w) = match &w_values {
                    None => {
                        let sum: f64 = values.iter().sum();
                        (sum / values.len() as f64, values.len() as f64)
                    }
                    Some(w) => {
                        if values.len() != w.len() {
                            return Err(PyValueError::new_err(
                                "weights must have the same length as the axis dimension",
                            ));
                        }
                        let weighted_sum: f64 =
                            values.iter().zip(w.iter()).map(|(v, w)| v * w).sum();
                        let sum_w_val: f64 = w.iter().sum();
                        if sum_w_val == 0.0 {
                            return Err(PyValueError::new_err("sum of weights must not be zero"));
                        }
                        (weighted_sum / sum_w_val, sum_w_val)
                    }
                };

                avg_results.push(avg);
                sum_w_results.push(sum_w);
            }

            if returned {
                let combined_len = avg_results.len() * 2;
                let mut combined: Vec<f64> = Vec::with_capacity(combined_len);
                for i in 0..avg_results.len() {
                    combined.push(avg_results[i]);
                    combined.push(sum_w_results[i]);
                }
                let mut result_shape = other_dims.clone();
                result_shape.push(2);
                Ok(NdArray {
                    imag: None,
                    data: Array::from_shape_vec(IxDyn(&result_shape), combined)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                })
            } else {
                Ok(NdArray {
                    imag: None,
                    data: Array::from_shape_vec(IxDyn(&other_dims), avg_results)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                })
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (x, q, axis=None, keepdims=false))]
fn percentile(x: &NdArray, q: f64, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    if !(0.0..=100.0).contains(&q) {
        return Err(PyValueError::new_err(
            "Percentile must be between 0 and 100",
        ));
    }

    match axis {
        None => {
            let mut values: Vec<f64> = x.data.iter().copied().collect();
            if values.is_empty() {
                return Err(PyValueError::new_err(
                    "Cannot compute percentile of empty array",
                ));
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let len = values.len();
            let idx = q / 100.0 * (len - 1) as f64;
            let lower = idx.floor() as usize;
            let upper = idx.ceil() as usize;
            if lower == upper || upper >= len {
                let val = values[lower.min(len - 1)];
                return Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), val),
                });
            }
            let frac = idx - lower as f64;
            let val = values[lower] * (1.0 - frac) + values[upper] * frac;
            Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), val),
            })
        }
        Some(ax) => {
            let ndim = x.data.ndim();
            let ax = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };

            let nrows = x.data.shape()[0];
            let ncols = x.data.shape()[1];

            let result: Vec<f64> = if ax == 0 {
                (0..ncols)
                    .map(|c| {
                        let mut values: Vec<f64> = (0..nrows).map(|r| x.data[[r, c]]).collect();
                        values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let len = values.len();
                        let idx = q / 100.0 * (len - 1) as f64;
                        let lower = idx.floor() as usize;
                        let upper = idx.ceil() as usize;
                        if lower == upper || upper >= len {
                            values[lower.min(len - 1)]
                        } else {
                            let frac = idx - lower as f64;
                            values[lower] * (1.0 - frac) + values[upper] * frac
                        }
                    })
                    .collect()
            } else {
                (0..nrows)
                    .map(|r| {
                        let mut values: Vec<f64> = (0..ncols).map(|c| x.data[[r, c]]).collect();
                        values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let len = values.len();
                        let idx = q / 100.0 * (len - 1) as f64;
                        let lower = idx.floor() as usize;
                        let upper = idx.ceil() as usize;
                        if lower == upper || upper >= len {
                            values[lower.min(len - 1)]
                        } else {
                            let frac = idx - lower as f64;
                            values[lower] * (1.0 - frac) + values[upper] * frac
                        }
                    })
                    .collect()
            };

            let mut new_shape: Vec<usize> = x.data.shape().to_vec();
            if keepdims {
                new_shape[ax] = 1;
            } else {
                new_shape.remove(ax);
            }
            let result_arr = Array::from_shape_vec(IxDyn(&new_shape), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: result_arr,
            })
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, bins=10, range=None))]
fn histogram<'py>(
    py: Python<'py>,
    a: &NdArray,
    bins: usize,
    range: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let values: Vec<f64> = a.data.iter().copied().collect();
    if values.is_empty() {
        return Err(PyValueError::new_err("Empty array"));
    }
    let (min_val, max_val) = if let Some(r) = range {
        let range_tuple = r
            .cast::<PyTuple>()
            .map_err(|_| PyTypeError::new_err("Range must be a tuple"))?;
        if range_tuple.len() != 2 {
            return Err(PyValueError::new_err("Range must have exactly 2 elements"));
        }
        let low = range_tuple.get_item(0)?.extract::<f64>()?;
        let high = range_tuple.get_item(1)?.extract::<f64>()?;
        (low, high)
    } else {
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    };
    let bin_width = (max_val - min_val) / bins as f64;
    if bin_width == 0.0 {
        let hist_arr = Array::from_shape_vec(IxDyn(&[1]), vec![values.len() as f64])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let edges_arr = Array::from_shape_vec(IxDyn(&[2]), vec![min_val, max_val])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return PyTuple::new(
            py,
            vec![
                Bound::new(
                    py,
                    NdArray {
                        imag: None,
                        data: hist_arr.into_dyn(),
                    },
                )?
                .into_any(),
                Bound::new(
                    py,
                    NdArray {
                        imag: None,
                        data: edges_arr.into_dyn(),
                    },
                )?
                .into_any(),
            ],
        );
    }
    let mut hist = vec![0.0_f64; bins];
    let bin_edges: Vec<f64> = (0..=bins).map(|i| min_val + i as f64 * bin_width).collect();
    for &v in &values {
        if v < min_val || v > max_val {
            continue;
        }
        let mut idx = ((v - min_val) / bin_width) as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        hist[idx] += 1.0;
    }
    let hist_arr = Array::from_shape_vec(IxDyn(&[bins]), hist)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let edges_arr = Array::from_shape_vec(IxDyn(&[bins + 1]), bin_edges)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyTuple::new(
        py,
        vec![
            Bound::new(
                py,
                NdArray {
                    imag: None,
                    data: hist_arr.into_dyn(),
                },
            )?
            .into_any(),
            Bound::new(
                py,
                NdArray {
                    imag: None,
                    data: edges_arr.into_dyn(),
                },
            )?
            .into_any(),
        ],
    )
}

#[pyfunction]
fn gradient(f: &NdArray) -> PyResult<NdArray> {
    let n = f.data.len();
    if n == 0 {
        return Err(PyValueError::new_err("Empty array"));
    }
    if n == 1 {
        return Ok(NdArray {
            imag: None,
            data: Array::from_elem(IxDyn(&[]), 0.0),
        });
    }

    // 计算实部的梯度
    let re_values: Vec<f64> = f.data.iter().copied().collect();
    let mut re_grad = Vec::with_capacity(n);
    re_grad.push(re_values[1] - re_values[0]);
    for i in 1..n - 1 {
        re_grad.push((re_values[i + 1] - re_values[i - 1]) / 2.0);
    }
    re_grad.push(re_values[n - 1] - re_values[n - 2]);

    // 如果是复数数组，计算虚部的梯度
    if let Some(ref im_array) = f.imag {
        let im_values: Vec<f64> = im_array.iter().copied().collect();
        let mut im_grad = Vec::with_capacity(n);
        im_grad.push(im_values[1] - im_values[0]);
        for i in 1..n - 1 {
            im_grad.push((im_values[i + 1] - im_values[i - 1]) / 2.0);
        }
        im_grad.push(im_values[n - 1] - im_values[n - 2]);

        let re_arr = Array::from_shape_vec(IxDyn(&[n]), re_grad)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let im_arr = Array::from_shape_vec(IxDyn(&[n]), im_grad)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        return Ok(NdArray {
            imag: Some(im_arr),
            data: re_arr,
        });
    }

    // 实数数组
    let arr = Array::from_shape_vec(IxDyn(&[n]), re_grad)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
#[pyo3(signature = (a, n=1, axis=-1))]
fn diff(a: &NdArray, n: isize, axis: isize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(PyValueError::new_err("Axis out of bounds"));
    }
    let mut current = a.data.clone();
    for _ in 0..n {
        if current.shape()[ax] <= 1 {
            return Err(PyValueError::new_err(
                "Cannot compute diff: axis has less than 2 elements",
            ));
        }
        let shape = current.shape().to_vec();
        let data_vec: Vec<f64> = current.iter().copied().collect();
        let pre_size: usize = shape.iter().take(ax).product();
        let axis_size = shape[ax];
        let post_size: usize = shape.iter().skip(ax + 1).product();
        let block_size = axis_size * post_size;
        let mut new_data = Vec::with_capacity(data_vec.len() - pre_size * post_size);
        for outer in 0..pre_size {
            for inner in 0..post_size {
                let base = outer * block_size + inner;
                for k in 0..axis_size - 1 {
                    new_data.push(
                        data_vec[base + (k + 1) * post_size] - data_vec[base + k * post_size],
                    );
                }
            }
        }
        let mut new_shape = shape.clone();
        new_shape[ax] = axis_size - 1;
        current = Array::from_shape_vec(IxDyn(&new_shape), new_data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    Ok(NdArray {
        imag: None,
        data: current,
    })
}

#[pyfunction]
#[pyo3(signature = (y, dx=1.0))]
fn trapz(y: &NdArray, dx: f64) -> PyResult<NdArray> {
    let values: Vec<f64> = y.data.iter().copied().collect();
    let n = values.len();
    if n < 2 {
        return Ok(NdArray {
            imag: None,
            data: Array::from_elem(IxDyn(&[]), 0.0),
        });
    }
    let mut sum = 0.0;
    for i in 0..n - 1 {
        sum += (values[i] + values[i + 1]) / 2.0 * dx;
    }
    Ok(NdArray {
        imag: None,
        data: Array::from_elem(IxDyn(&[]), sum),
    })
}

#[pyfunction]
#[pyo3(signature = (x, minlength=0))]
fn bincount(x: &NdArray, minlength: usize) -> NdArray {
    let mut max_val: i64 = -1;
    for &v in x.data.iter() {
        let iv = v as i64;
        if iv > max_val {
            max_val = iv;
        }
    }
    let n = ::std::cmp::max(minlength as i64, max_val + 1).max(0) as usize;
    let mut counts = vec![0.0f64; n];
    for &v in x.data.iter() {
        let iv = v as i64;
        if iv >= 0 && (iv as usize) < n {
            counts[iv as usize] += 1.0;
        }
    }
    NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&[n]), counts).unwrap(),
    }
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn ptp(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    match axis {
        None => {
            let (min_val, max_val) = x
                .data
                .iter()
                .cloned()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
                    (min.min(v), max.max(v))
                });
            Ok(NdArray {
                imag: None,
                data: Array::from_elem(IxDyn(&[]), max_val - min_val),
            })
        }
        Some(ax) => {
            let ndim = x.data.ndim();
            let ax = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };
            let min_vals: Array<f64, IxDyn> = x
                .data
                .fold_axis(Axis(ax), f64::INFINITY, |acc, &v| {
                    let a = *acc;
                    if v < a { v } else { a }
                })
                .into_dyn();
            let max_vals: Array<f64, IxDyn> = x
                .data
                .fold_axis(Axis(ax), f64::NEG_INFINITY, |acc, &v| {
                    let a = *acc;
                    if v > a { v } else { a }
                })
                .into_dyn();
            let result: Vec<f64> = min_vals
                .iter()
                .zip(max_vals.iter())
                .map(|(min_val, max_val)| max_val - min_val)
                .collect();
            let result_arr = Array::from_shape_vec(IxDyn(min_vals.shape()), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: result_arr,
            })
        }
    }
}

#[pyfunction]
fn digitize(x: &NdArray, bins: &NdArray) -> PyResult<NdArray> {
    let x_vals: Vec<f64> = x.data.iter().copied().collect();
    let mut bin_edges: Vec<f64> = bins.data.iter().copied().collect();
    bin_edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // 二分查找 + 并行计算：O(n*log m)，原先线性 O(n*m)
    let result: Vec<f64> = x_vals
        .par_iter()
        .map(|&v| bin_edges.partition_point(|&edge| v >= edge) as f64)
        .collect();
    let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== 协方差矩阵 ==========
#[pyfunction]
#[pyo3(signature = (a, rowvar=true))]
fn cov(a: &NdArray, rowvar: bool) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    let data_vec: Vec<f64> = a.data.iter().copied().collect();

    let (n_vars, n_obs) = if shape.len() == 1 {
        (1, shape[0])
    } else if rowvar {
        (shape[0], shape[1])
    } else {
        (shape[1], shape[0])
    };

    let rows: Vec<Vec<f64>> = if shape.len() == 1 {
        vec![data_vec.clone()]
    } else if rowvar {
        (0..shape[0])
            .map(|i| (0..shape[1]).map(|j| data_vec[i * shape[1] + j]).collect())
            .collect()
    } else {
        (0..shape[1])
            .map(|j| (0..shape[0]).map(|i| data_vec[i * shape[1] + j]).collect())
            .collect()
    };

    let means: Vec<f64> = rows
        .iter()
        .map(|r| r.iter().sum::<f64>() / n_obs as f64)
        .collect();
    let centered: Vec<Vec<f64>> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| r.iter().map(|&v| v - means[i]).collect())
        .collect();

    let mut cov_matrix = vec![vec![0.0; n_vars]; n_vars];
    let factor = 1.0 / (n_obs as f64 - 1.0);
    for k in 0..n_vars {
        for j in 0..n_vars {
            let mut s = 0.0;
            for (ci, cj) in centered[k].iter().zip(centered[j].iter()) {
                s += ci * cj;
            }
            cov_matrix[k][j] = s * factor;
        }
    }

    let mut flat = vec![0.0; n_vars * n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            flat[i * n_vars + j] = cov_matrix[i][j];
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[n_vars, n_vars]), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== 2D 直方图 ==========
/// 2D 直方图返回类型：主数组 + 两个边缘数组
type Histogram2DResult = (NdArray, (Vec<f64>, Vec<f64>));

#[pyfunction]
#[pyo3(signature = (x, y, bins=10))]
fn histogram2d_rs(x: &NdArray, y: &NdArray, bins: usize) -> PyResult<Histogram2DResult> {
    let x_vals: Vec<f64> = x.data.iter().copied().collect();
    let y_vals: Vec<f64> = y.data.iter().copied().collect();
    let n = x_vals.len().min(y_vals.len());

    let x_min = *x_vals
        .iter()
        .fold(&f64::INFINITY, |a, b| if a < b { a } else { b });
    let x_max = *x_vals
        .iter()
        .fold(&f64::NEG_INFINITY, |a, b| if a > b { a } else { b });
    let y_min = *y_vals
        .iter()
        .fold(&f64::INFINITY, |a, b| if a < b { a } else { b });
    let y_max = *y_vals
        .iter()
        .fold(&f64::NEG_INFINITY, |a, b| if a > b { a } else { b });

    let mut hist = vec![vec![0.0; bins]; bins];
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    for i in 0..n {
        let xi = if x_range == 0.0 {
            0
        } else {
            let idx = ((x_vals[i] - x_min) / x_range * bins as f64) as usize;
            idx.min(bins - 1)
        };
        let yi = if y_range == 0.0 {
            0
        } else {
            let idx = ((y_vals[i] - y_min) / y_range * bins as f64) as usize;
            idx.min(bins - 1)
        };
        hist[xi][yi] += 1.0;
    }

    let flat: Vec<f64> = hist.iter().flatten().cloned().collect();
    let arr = Array::from_shape_vec(IxDyn(&[bins, bins]), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let x_edges: Vec<f64> = (0..=bins)
        .map(|i| x_min + (x_max - x_min) * i as f64 / bins as f64)
        .collect();
    let y_edges: Vec<f64> = (0..=bins)
        .map(|i| y_min + (y_max - y_min) * i as f64 / bins as f64)
        .collect();

    Ok((
        NdArray {
            imag: None,
            data: arr,
        },
        (x_edges, y_edges),
    ))
}

// ========== 相关系数 ==========
#[pyfunction]
fn corrcoef_rs(a: &NdArray) -> PyResult<NdArray> {
    let cov_arr = cov(a, true)?;
    let shape = cov_arr.data.shape().to_vec();
    let n = shape[0];
    let data_vec: Vec<f64> = cov_arr.data.iter().copied().collect();

    let diag: Vec<f64> = (0..n).map(|i| data_vec[i * n + i]).collect();
    let diag_sqrt: Vec<f64> = diag.iter().map(|&v| v.sqrt()).collect();

    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let denom = diag_sqrt[i] * diag_sqrt[j];
            result[i * n + j] = if denom == 0.0 {
                0.0
            } else {
                data_vec[i * n + j] / denom
            };
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[n, n]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== argmax/argmin 沿指定轴 ==========
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn argmax_axis(a: &NdArray, axis: Option<i32>) -> PyResult<NdArray> {
    if axis.is_none() {
        let idx = a.argmax()?;
        let arr = Array::from_elem(IxDyn(&[]), idx as f64);
        return Ok(NdArray {
            imag: None,
            data: arr,
        });
    }
    let ax = axis.unwrap() as usize;
    let shape = a.data.shape().to_vec();
    if ax >= shape.len() {
        return Err(PyValueError::new_err("axis out of bounds"));
    }
    let outer: usize = shape.iter().take(ax).product();
    let inner: usize = shape.iter().skip(ax + 1).product();
    let dim_size = shape[ax];
    let mut result_shape = shape.clone();
    result_shape.remove(ax);
    if result_shape.is_empty() {
        result_shape = vec![1];
    }
    let total: usize = result_shape.iter().product();
    let mut values = vec![0.0f64; total];
    let data: Vec<f64> = a.data.iter().copied().collect();
    let mut out_idx = 0;
    for o in 0..outer {
        for inn in 0..inner {
            let mut best = 0usize;
            let mut best_val = f64::NEG_INFINITY;
            for i in 0..dim_size {
                let idx = o * dim_size * inner + i * inner + inn;
                let v = data[idx];
                if v > best_val {
                    best_val = v;
                    best = i;
                }
            }
            values[out_idx] = best as f64;
            out_idx += 1;
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&result_shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn argmin_axis(a: &NdArray, axis: Option<i32>) -> PyResult<NdArray> {
    if axis.is_none() {
        let idx = a.argmin()?;
        let arr = Array::from_elem(IxDyn(&[]), idx as f64);
        return Ok(NdArray {
            imag: None,
            data: arr,
        });
    }
    let ax = axis.unwrap() as usize;
    let shape = a.data.shape().to_vec();
    if ax >= shape.len() {
        return Err(PyValueError::new_err("axis out of bounds"));
    }
    let outer: usize = shape.iter().take(ax).product();
    let inner: usize = shape.iter().skip(ax + 1).product();
    let dim_size = shape[ax];
    let mut result_shape = shape.clone();
    result_shape.remove(ax);
    if result_shape.is_empty() {
        result_shape = vec![1];
    }
    let total: usize = result_shape.iter().product();
    let mut values = vec![0.0f64; total];
    let data: Vec<f64> = a.data.iter().copied().collect();
    let mut out_idx = 0;
    for o in 0..outer {
        for inn in 0..inner {
            let mut best = 0usize;
            let mut best_val = f64::INFINITY;
            for i in 0..dim_size {
                let idx = o * dim_size * inner + i * inner + inn;
                let v = data[idx];
                if v < best_val {
                    best_val = v;
                    best = i;
                }
            }
            values[out_idx] = best as f64;
            out_idx += 1;
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&result_shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== NaN 替换为数值 ==========
#[pyfunction]
#[pyo3(signature = (x, nan=0.0, posinf=None, neginf=None))]
fn nan_to_num(x: &NdArray, nan: f64, posinf: Option<f64>, neginf: Option<f64>) -> NdArray {
    // 将 NaN 替换为 nan，正无穷替换为 posinf（默认 f64::MAX），负无穷替换为 neginf（默认 f64::MIN）
    let pos = posinf.unwrap_or(f64::MAX);
    let neg = neginf.unwrap_or(f64::MIN);
    let result: Vec<f64> = x
        .data
        .iter()
        .map(|&v| {
            if v.is_nan() {
                nan
            } else if v.is_infinite() && v > 0.0 {
                pos
            } else if v.is_infinite() && v < 0.0 {
                neg
            } else {
                v
            }
        })
        .collect();
    let shape = x.data.shape().to_vec();
    NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&shape), result).unwrap(),
    }
}

// ========== 数组相邻元素差分（带可选首尾追加） ==========
#[pyfunction]
#[pyo3(signature = (ary, to_end=None, to_begin=None))]
fn ediff1d(ary: &NdArray, to_end: Option<Vec<f64>>, to_begin: Option<Vec<f64>>) -> NdArray {
    // 计算一阶差分，可选拼接前缀和后缀
    let data: Vec<f64> = ary.data.iter().copied().collect();
    let n = data.len();
    let diffs: Vec<f64> = if n > 1 {
        (0..n - 1).map(|i| data[i + 1] - data[i]).collect()
    } else {
        Vec::new()
    };
    let mut result = Vec::new();
    if let Some(prepend) = to_begin {
        result.extend(prepend);
    }
    result.extend(diffs);
    if let Some(append) = to_end {
        result.extend(append);
    }
    NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&[result.len()]), result).unwrap(),
    }
}

// ========== 去除首尾零元素 ==========
#[pyfunction]
#[pyo3(signature = (filt, trim="fb"))]
fn trim_zeros(filt: &NdArray, trim: &str) -> NdArray {
    // trim 字符串包含 'f' 去除前导零，包含 'b' 去除末尾零
    let data: Vec<f64> = filt.data.iter().copied().collect();
    let mut start = 0;
    let mut end = data.len();
    if trim.contains('f') {
        while start < end && data[start] == 0.0 {
            start += 1;
        }
    }
    if trim.contains('b') {
        while end > start && data[end - 1] == 0.0 {
            end -= 1;
        }
    }
    let result = data[start..end].to_vec();
    NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&[result.len()]), result).unwrap(),
    }
}

// ========== 带权重的 bincount ==========
#[pyfunction]
#[pyo3(signature = (x, weights, minlength=0))]
fn bincount_weighted(x: &NdArray, weights: &NdArray, minlength: usize) -> NdArray {
    // 对每个非负整数索引累加对应权重
    let xv: Vec<usize> = x.data.iter().map(|&v| v as usize).collect();
    let wv: Vec<f64> = weights.data.iter().copied().collect();
    let max_val = xv.iter().copied().max().unwrap_or(0);
    let n = (max_val + 1).max(minlength).max(1);
    let mut out = vec![0.0_f64; n];
    for (i, &v) in xv.iter().enumerate() {
        if v < n {
            out[v] += wv[i];
        }
    }
    NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&[n]), out).unwrap(),
    }
}

// ========== NaN 忽略的统计函数 ==========
//
// 设计：所有 nan_* 函数复用同一套「按 axis 分块」的扁平遍历模式：
//   outer × dim_size × inner 三层循环，对每个 (outer, inner) 在 dim_size 维度上做归约。
// axis=None 时按全数组归约，axis 为正/负整数时沿指定轴。

/// 计算沿 axis 归约时的几何参数 (outer, dim_size, inner, result_shape)。
/// keepdims=true 时 result_shape[ax] = 1，否则去掉该维。
fn axis_reduce_layout(
    shape: &[usize],
    axis: Option<isize>,
    keepdims: bool,
) -> PyResult<(usize, usize, usize, Vec<usize>)> {
    match axis {
        None => {
            let total: usize = shape.iter().product();
            let result_shape = if keepdims {
                vec![1usize; shape.len()]
            } else {
                Vec::new()
            };
            Ok((1, total, 1, result_shape))
        }
        Some(ax) => {
            let ndim = shape.len();
            let ax = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };
            if ax >= ndim {
                return Err(PyValueError::new_err("axis out of bounds"));
            }
            let outer: usize = shape.iter().take(ax).product();
            let inner: usize = shape.iter().skip(ax + 1).product();
            let dim_size = shape[ax];
            let result_shape = if keepdims {
                let mut s = shape.to_vec();
                s[ax] = 1;
                s
            } else {
                let mut s = shape.to_vec();
                s.remove(ax);
                s
            };
            Ok((outer, dim_size, inner, result_shape))
        }
    }
}

/// 沿轴求和（NaN 视为 0）。
#[pyfunction]
#[pyo3(signature = (x, axis=None, keepdims=false))]
fn nansum(x: &NdArray, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    let shape = x.data.shape().to_vec();
    let (outer, dim_size, inner, result_shape) = axis_reduce_layout(&shape, axis, keepdims)?;
    let data: Vec<f64> = x.data.iter().copied().collect();
    let total_out = if result_shape.is_empty() {
        1
    } else {
        result_shape.iter().product::<usize>().max(1)
    };

    // 并行优化：当外层循环足够大时使用并行
    let out = if outer * inner >= PAR_THRESHOLD_CHEAP / 4 {
        // 并行路径
        let out_vec: Vec<f64> = (0..outer * inner)
            .into_par_iter()
            .map(|o| {
                let out_o = o / inner;
                let inn = o % inner;
                let base = out_o * dim_size * inner;
                let mut s = 0.0f64;
                for i in 0..dim_size {
                    let v = data[base + i * inner + inn];
                    if !v.is_nan() {
                        s += v;
                    }
                }
                s
            })
            .collect();
        out_vec
    } else {
        // 串行路径
        let mut out = vec![0.0f64; total_out];
        let mut o = 0usize;
        for out_o in 0..outer {
            for inn in 0..inner {
                let base = out_o * dim_size * inner;
                let mut s = 0.0f64;
                for i in 0..dim_size {
                    let v = data[base + i * inner + inn];
                    if !v.is_nan() {
                        s += v;
                    }
                }
                out[o] = s;
                o += 1;
            }
        }
        let _ = o;
        out
    };

    let arr = Array::from_shape_vec(IxDyn(&result_shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

/// 沿轴求积（NaN 视为 1）。
#[pyfunction]
#[pyo3(signature = (x, axis=None, keepdims=false))]
fn nanprod(x: &NdArray, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    let shape = x.data.shape().to_vec();
    let (outer, dim_size, inner, result_shape) = axis_reduce_layout(&shape, axis, keepdims)?;
    let data: Vec<f64> = x.data.iter().copied().collect();
    let total_out = if result_shape.is_empty() {
        1
    } else {
        result_shape.iter().product::<usize>().max(1)
    };
    let mut out = vec![1.0f64; total_out];
    let mut o = 0usize;
    for out_o in 0..outer {
        for inn in 0..inner {
            let base = out_o * dim_size * inner;
            let mut p = 1.0f64;
            for i in 0..dim_size {
                let v = data[base + i * inner + inn];
                if !v.is_nan() {
                    p *= v;
                }
            }
            out[o] = p;
            o += 1;
        }
    }
    let _ = o;
    let arr = Array::from_shape_vec(IxDyn(&result_shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

/// 沿轴求最大值（忽略 NaN；若全为 NaN 则返回 NaN）。
#[pyfunction]
#[pyo3(signature = (x, axis=None, keepdims=false))]
fn nanmax(x: &NdArray, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    let shape = x.data.shape().to_vec();
    let (outer, dim_size, inner, result_shape) = axis_reduce_layout(&shape, axis, keepdims)?;
    let data: Vec<f64> = x.data.iter().copied().collect();
    let total_out = if result_shape.is_empty() {
        1
    } else {
        result_shape.iter().product::<usize>().max(1)
    };
    let mut out = vec![f64::NAN; total_out];
    let mut o = 0usize;
    for out_o in 0..outer {
        for inn in 0..inner {
            let base = out_o * dim_size * inner;
            let mut best: Option<f64> = None;
            for i in 0..dim_size {
                let v = data[base + i * inner + inn];
                if !v.is_nan() {
                    best = Some(match best {
                        Some(b) if b >= v => b,
                        _ => v,
                    });
                }
            }
            out[o] = best.unwrap_or(f64::NAN);
            o += 1;
        }
    }
    let _ = o;
    let arr = Array::from_shape_vec(IxDyn(&result_shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

/// 沿轴求最小值（忽略 NaN；若全为 NaN 则返回 NaN）。
#[pyfunction]
#[pyo3(signature = (x, axis=None, keepdims=false))]
fn nanmin(x: &NdArray, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    let shape = x.data.shape().to_vec();
    let (outer, dim_size, inner, result_shape) = axis_reduce_layout(&shape, axis, keepdims)?;
    let data: Vec<f64> = x.data.iter().copied().collect();
    let total_out = if result_shape.is_empty() {
        1
    } else {
        result_shape.iter().product::<usize>().max(1)
    };
    let mut out = vec![f64::NAN; total_out];
    let mut o = 0usize;
    for out_o in 0..outer {
        for inn in 0..inner {
            let base = out_o * dim_size * inner;
            let mut best: Option<f64> = None;
            for i in 0..dim_size {
                let v = data[base + i * inner + inn];
                if !v.is_nan() {
                    best = Some(match best {
                        Some(b) if b <= v => b,
                        _ => v,
                    });
                }
            }
            out[o] = best.unwrap_or(f64::NAN);
            o += 1;
        }
    }
    let _ = o;
    let arr = Array::from_shape_vec(IxDyn(&result_shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

/// 沿轴求均值（忽略 NaN；若全为 NaN 则返回 NaN）。
#[pyfunction]
#[pyo3(signature = (x, axis=None, keepdims=false))]
fn nanmean(x: &NdArray, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    let shape = x.data.shape().to_vec();
    let (outer, dim_size, inner, result_shape) = axis_reduce_layout(&shape, axis, keepdims)?;
    let data: Vec<f64> = x.data.iter().copied().collect();
    let total_out = if result_shape.is_empty() {
        1
    } else {
        result_shape.iter().product::<usize>().max(1)
    };
    let mut out = vec![f64::NAN; total_out];
    let mut o = 0usize;
    for out_o in 0..outer {
        for inn in 0..inner {
            let base = out_o * dim_size * inner;
            let mut s = 0.0f64;
            let mut cnt = 0usize;
            for i in 0..dim_size {
                let v = data[base + i * inner + inn];
                if !v.is_nan() {
                    s += v;
                    cnt += 1;
                }
            }
            out[o] = if cnt == 0 { f64::NAN } else { s / cnt as f64 };
            o += 1;
        }
    }
    let _ = o;
    let arr = Array::from_shape_vec(IxDyn(&result_shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

/// 沿轴求方差（忽略 NaN；若有效元素数 <= ddof 则返回 NaN）。
#[pyfunction]
#[pyo3(signature = (x, axis=None, ddof=0, keepdims=false))]
fn nanvar(x: &NdArray, axis: Option<isize>, ddof: usize, keepdims: bool) -> PyResult<NdArray> {
    let shape = x.data.shape().to_vec();
    let (outer, dim_size, inner, result_shape) = axis_reduce_layout(&shape, axis, keepdims)?;
    let data: Vec<f64> = x.data.iter().copied().collect();
    let total_out = if result_shape.is_empty() {
        1
    } else {
        result_shape.iter().product::<usize>().max(1)
    };
    let mut out = vec![f64::NAN; total_out];
    let mut o = 0usize;
    for out_o in 0..outer {
        for inn in 0..inner {
            let base = out_o * dim_size * inner;
            let mut s = 0.0f64;
            let mut cnt = 0usize;
            // 第一遍：求均值
            for i in 0..dim_size {
                let v = data[base + i * inner + inn];
                if !v.is_nan() {
                    s += v;
                    cnt += 1;
                }
            }
            if cnt > ddof {
                let mu = s / cnt as f64;
                // 第二遍：求平方差和
                let mut sq = 0.0f64;
                for i in 0..dim_size {
                    let v = data[base + i * inner + inn];
                    if !v.is_nan() {
                        let d = v - mu;
                        sq += d * d;
                    }
                }
                let n = (cnt - ddof) as f64;
                out[o] = sq / n;
            } else {
                out[o] = f64::NAN;
            }
            o += 1;
        }
    }
    let _ = o;
    let arr = Array::from_shape_vec(IxDyn(&result_shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

/// 沿轴求标准差（忽略 NaN）。
#[pyfunction]
#[pyo3(signature = (x, axis=None, ddof=0, keepdims=false))]
fn nanstd(x: &NdArray, axis: Option<isize>, ddof: usize, keepdims: bool) -> PyResult<NdArray> {
    let mut v = nanvar(x, axis, ddof, keepdims)?;
    v.data.mapv_inplace(|x| x.sqrt());
    Ok(v)
}

/// 沿轴求最大值索引（忽略 NaN；若全为 NaN 则返回 0）。
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn nanargmax_axis(a: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    let (outer, dim_size, inner, result_shape) = axis_reduce_layout(&shape, axis, false)?;
    let data: Vec<f64> = a.data.iter().copied().collect();
    let total_out = if result_shape.is_empty() {
        1
    } else {
        result_shape.iter().product::<usize>().max(1)
    };
    let mut out = vec![0.0f64; total_out];
    let mut o = 0usize;
    for out_o in 0..outer {
        for inn in 0..inner {
            let base = out_o * dim_size * inner;
            let mut best_idx = 0usize;
            let mut best_val: Option<f64> = None;
            for i in 0..dim_size {
                let v = data[base + i * inner + inn];
                if !v.is_nan() {
                    match best_val {
                        Some(b) if b >= v => {}
                        _ => {
                            best_val = Some(v);
                            best_idx = i;
                        }
                    }
                }
            }
            out[o] = best_idx as f64;
            o += 1;
        }
    }
    let _ = o;
    let arr = Array::from_shape_vec(IxDyn(&result_shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

/// 沿轴求最小值索引（忽略 NaN；若全为 NaN 则返回 0）。
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn nanargmin_axis(a: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    let (outer, dim_size, inner, result_shape) = axis_reduce_layout(&shape, axis, false)?;
    let data: Vec<f64> = a.data.iter().copied().collect();
    let total_out = if result_shape.is_empty() {
        1
    } else {
        result_shape.iter().product::<usize>().max(1)
    };
    let mut out = vec![0.0f64; total_out];
    let mut o = 0usize;
    for out_o in 0..outer {
        for inn in 0..inner {
            let base = out_o * dim_size * inner;
            let mut best_idx = 0usize;
            let mut best_val: Option<f64> = None;
            for i in 0..dim_size {
                let v = data[base + i * inner + inn];
                if !v.is_nan() {
                    match best_val {
                        Some(b) if b <= v => {}
                        _ => {
                            best_val = Some(v);
                            best_idx = i;
                        }
                    }
                }
            }
            out[o] = best_idx as f64;
            o += 1;
        }
    }
    let _ = o;
    let arr = Array::from_shape_vec(IxDyn(&result_shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

/// 累积和（NaN 视为 0）。axis=None 时按扁平展开。
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn nancumsum(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    let shape = x.data.shape().to_vec();
    let data: Vec<f64> = x.data.iter().copied().collect();
    if let Some(ax) = axis {
        let ndim = shape.len();
        let ax = if ax < 0 {
            (ndim as isize + ax) as usize
        } else {
            ax as usize
        };
        if ax >= ndim {
            return Err(PyValueError::new_err("axis out of bounds"));
        }
        let outer: usize = shape.iter().take(ax).product();
        let inner: usize = shape.iter().skip(ax + 1).product();
        let dim_size = shape[ax];
        let mut out = vec![0.0f64; data.len()];
        for o in 0..outer {
            for inn in 0..inner {
                let mut acc = 0.0f64;
                for i in 0..dim_size {
                    let idx = o * dim_size * inner + i * inner + inn;
                    let v = data[idx];
                    if !v.is_nan() {
                        acc += v;
                    }
                    out[idx] = acc;
                }
            }
        }
        let arr = Array::from_shape_vec(IxDyn(&shape), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    } else {
        // 扁平展开
        let mut acc = 0.0f64;
        let out: Vec<f64> = data
            .iter()
            .map(|&v| {
                if !v.is_nan() {
                    acc += v;
                }
                acc
            })
            .collect();
        let arr = Array::from_shape_vec(IxDyn(&[out.len()]), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }
}

/// 累积积（NaN 视为 1）。axis=None 时按扁平展开。
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn nancumprod(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    let shape = x.data.shape().to_vec();
    let data: Vec<f64> = x.data.iter().copied().collect();
    if let Some(ax) = axis {
        let ndim = shape.len();
        let ax = if ax < 0 {
            (ndim as isize + ax) as usize
        } else {
            ax as usize
        };
        if ax >= ndim {
            return Err(PyValueError::new_err("axis out of bounds"));
        }
        let outer: usize = shape.iter().take(ax).product();
        let inner: usize = shape.iter().skip(ax + 1).product();
        let dim_size = shape[ax];
        let mut out = vec![0.0f64; data.len()];
        for o in 0..outer {
            for inn in 0..inner {
                let mut acc = 1.0f64;
                for i in 0..dim_size {
                    let idx = o * dim_size * inner + i * inner + inn;
                    let v = data[idx];
                    if !v.is_nan() {
                        acc *= v;
                    }
                    out[idx] = acc;
                }
            }
        }
        let arr = Array::from_shape_vec(IxDyn(&shape), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    } else {
        let mut acc = 1.0f64;
        let out: Vec<f64> = data
            .iter()
            .map(|&v| {
                if !v.is_nan() {
                    acc *= v;
                }
                acc
            })
            .collect();
        let arr = Array::from_shape_vec(IxDyn(&[out.len()]), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(prod, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(average, m)?)?;
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(ptp, m)?)?;
    m.add_function(wrap_pyfunction!(histogram, m)?)?;
    m.add_function(wrap_pyfunction!(histogram2d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(corrcoef_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cov, m)?)?;
    m.add_function(wrap_pyfunction!(bincount, m)?)?;
    m.add_function(wrap_pyfunction!(gradient, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(trapz, m)?)?;
    m.add_function(wrap_pyfunction!(digitize, m)?)?;
    m.add_function(wrap_pyfunction!(argmax_axis, m)?)?;
    m.add_function(wrap_pyfunction!(argmin_axis, m)?)?;
    m.add_function(wrap_pyfunction!(nan_to_num, m)?)?;
    m.add_function(wrap_pyfunction!(ediff1d, m)?)?;
    m.add_function(wrap_pyfunction!(trim_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(bincount_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(nansum, m)?)?;
    m.add_function(wrap_pyfunction!(nanprod, m)?)?;
    m.add_function(wrap_pyfunction!(nanmax, m)?)?;
    m.add_function(wrap_pyfunction!(nanmin, m)?)?;
    m.add_function(wrap_pyfunction!(nanmean, m)?)?;
    m.add_function(wrap_pyfunction!(nanvar, m)?)?;
    m.add_function(wrap_pyfunction!(nanstd, m)?)?;
    m.add_function(wrap_pyfunction!(nanargmax_axis, m)?)?;
    m.add_function(wrap_pyfunction!(nanargmin_axis, m)?)?;
    m.add_function(wrap_pyfunction!(nancumsum, m)?)?;
    m.add_function(wrap_pyfunction!(nancumprod, m)?)?;
    Ok(())
}

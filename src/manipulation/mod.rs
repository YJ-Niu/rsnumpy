use crate::*;

#[pyfunction]
fn concatenate(arrays: &Bound<'_, PyAny>, axis: usize) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut ndarrays: Vec<Array<f64, IxDyn>> = Vec::with_capacity(list.len());
    for item in list.iter() {
        let nd = item.extract::<NdArray>()?;
        ndarrays.push(nd.data);
    }
    if ndarrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let views: Vec<_> = ndarrays.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None, data: result.into_dyn(),
    })
}

#[pyfunction]
fn stack(arrays: &Bound<'_, PyAny>, axis: usize) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut ndarrays: Vec<NdArray> = Vec::with_capacity(list.len());
    for item in list.iter() {
        ndarrays.push(item.extract::<NdArray>()?);
    }
    if ndarrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let orig_shape = ndarrays[0].data.shape().to_vec();
    for arr in &ndarrays[1..] {
        if arr.data.shape() != orig_shape.as_slice() {
            return Err(PyValueError::new_err("All arrays must have the same shape"));
        }
    }
    // 一次性收集所有 expanded view，避免逐个 concatenate（O(n²) → O(n)）
    let expanded_views: Vec<_> = ndarrays
        .iter()
        .map(|arr| {
            let mut s = orig_shape.clone();
            s.insert(axis, 1);
            arr.data
                .clone()
                .into_shape_with_order(IxDyn(&s))
                .unwrap()
                .into_dyn()
        })
        .collect();
    let views: Vec<_> = expanded_views.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None, data: result.into_dyn(),
    })
}

#[pyfunction]
fn transpose(a: &NdArray) -> PyResult<NdArray> {
    a.t()
}

#[pyfunction]
fn swapaxes(a: &NdArray, axis1: usize, axis2: usize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    if axis1 >= ndim || axis2 >= ndim {
        return Err(PyValueError::new_err(format!(
            "axis out of bounds: arr.ndim={}, axis1={}, axis2={}",
            ndim, axis1, axis2
        )));
    }
    let mut arr = a.data.clone();
    arr.swap_axes(axis1, axis2);
    let imag = a.imag.as_ref().map(|im| {
        let mut m = im.clone();
        m.swap_axes(axis1, axis2);
        m
    });
    Ok(NdArray { imag, data: arr })
}

#[pyfunction]
#[pyo3(signature = (a, new_shape))]
fn resize_rs(a: &NdArray, new_shape: Vec<usize>) -> PyResult<NdArray> {
    let new_size: usize = new_shape.iter().product();
    let flat_data: Vec<f64> = a.data.iter().copied().collect();

    if flat_data.is_empty() {
        let arr = Array::from_shape_vec(IxDyn(&new_shape), Vec::new())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { imag: None, data: arr });
    }

    let mut result = Vec::with_capacity(new_size);
    for i in 0..new_size {
        result.push(flat_data[i % flat_data.len()]);
    }

    let arr = Array::from_shape_vec(IxDyn(&new_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { imag: None, data: arr })
}

#[pyfunction]
#[pyo3(signature = (a, indices, axis=None))]
fn delete_rs(a: &NdArray, indices: Vec<isize>, axis: Option<isize>) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = axis.map(|x| {
        if x < 0 {
            (ndim as isize + x) as usize
        } else {
            x as usize
        }
    });

    let idx_set: std::collections::BTreeSet<usize> = indices
        .iter()
        .map(|&i| {
            let len = if let Some(axis_idx) = ax {
                a.data.shape()[axis_idx]
            } else {
                a.data.len()
            };
            let idx = if i < 0 {
                (len as isize + i) as usize
            } else {
                i as usize
            };
            idx.min(len - 1)
        })
        .collect();

    if ax.is_none() {
        let flat_data: Vec<f64> = a.data.iter().copied().collect();
        let result: Vec<f64> = flat_data
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !idx_set.contains(i))
            .map(|(_, v)| v)
            .collect();
        let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { imag: None, data: arr });
    }

    let ax = ax.unwrap();
    let shape = a.data.shape().to_vec();
    let axis_size = shape[ax];
    let pre_size: usize = shape.iter().take(ax).product();
    let post_size: usize = shape.iter().skip(ax + 1).product();

    let flat_data: Vec<f64> = a.data.iter().copied().collect();
    let mut result = Vec::with_capacity(pre_size * (axis_size - idx_set.len()) * post_size);
    let block_size = axis_size * post_size;

    for outer in 0..pre_size {
        let base = outer * block_size;
        for i in 0..axis_size {
            if !idx_set.contains(&i) {
                result.extend_from_slice(
                    &flat_data[base + i * post_size..base + (i + 1) * post_size],
                );
            }
        }
    }

    let mut new_shape = shape;
    new_shape[ax] = axis_size - idx_set.len();
    let arr = Array::from_shape_vec(IxDyn(&new_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { imag: None, data: arr })
}

#[pyfunction]
#[pyo3(signature = (a, indices, values, axis=None))]
fn insert_rs(
    a: &NdArray,
    indices: Vec<isize>,
    values: Vec<f64>,
    axis: Option<isize>,
) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = axis.map(|x| {
        if x < 0 {
            (ndim as isize + x) as usize
        } else {
            x as usize
        }
    });

    let flat_data: Vec<f64> = a.data.iter().copied().collect();

    if ax.is_none() {
        let mut result = flat_data.clone();
        let mut offset = 0;
        let mut sorted_indices: Vec<isize> = indices.clone();
        sorted_indices.sort_unstable();

        for &idx in &sorted_indices {
            let pos = if idx < 0 {
                (result.len() as isize + idx + offset) as usize
            } else {
                (idx + offset) as usize
            };
            let pos = pos.min(result.len());

            for &v in values.iter().rev() {
                result.insert(pos, v);
            }
            offset += values.len() as isize;
        }

        let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { imag: None, data: arr });
    }

    let ax = ax.unwrap();
    let shape = a.data.shape().to_vec();
    let axis_size = shape[ax];
    let pre_size: usize = shape.iter().take(ax).product();
    let post_size: usize = shape.iter().skip(ax + 1).product();

    let needed = pre_size * post_size;
    let mut vals = Vec::with_capacity(needed);
    for i in 0..needed {
        vals.push(values[i % values.len()]);
    }

    let idx_set: std::collections::BTreeSet<usize> = indices
        .iter()
        .map(|&i| {
            let idx = if i < 0 {
                (axis_size as isize + i) as usize
            } else {
                i as usize
            };
            idx.min(axis_size)
        })
        .collect();

    let new_axis_size = axis_size + idx_set.len();
    let mut result = Vec::with_capacity(pre_size * new_axis_size * post_size);
    let block_size = axis_size * post_size;

    for outer in 0..pre_size {
        let base = outer * block_size;
        let mut src_idx = 0;

        for dest_idx in 0..new_axis_size {
            if idx_set.contains(&dest_idx) {
                let val_base = outer * post_size;
                result.extend_from_slice(&vals[val_base..val_base + post_size]);
            } else {
                result.extend_from_slice(
                    &flat_data[base + src_idx * post_size..base + (src_idx + 1) * post_size],
                );
                src_idx += 1;
            }
        }
    }

    let mut new_shape = shape;
    new_shape[ax] = new_axis_size;
    let arr = Array::from_shape_vec(IxDyn(&new_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { imag: None, data: arr })
}

#[pyfunction]
fn vstack(arrays: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut ndarrays: Vec<NdArray> = Vec::with_capacity(list.len());
    for item in list.iter() {
        ndarrays.push(item.extract::<NdArray>()?);
    }
    if ndarrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }

    // Check if all inputs are 1D
    let all_1d = ndarrays.iter().all(|a| a.data.ndim() == 1);

    if all_1d {
        // For 1D arrays, stack as rows to create a 2D array
        let ncols = ndarrays[0].data.len();
        for arr in &ndarrays[1..] {
            if arr.data.len() != ncols {
                return Err(PyValueError::new_err(
                    "All arrays must have the same length",
                ));
            }
        }
        let nrows = ndarrays.len();
        let mut data = Vec::with_capacity(nrows * ncols);
        for arr in &ndarrays {
            for val in arr.data.iter() {
                data.push(*val);
            }
        }
        let arr = Array::from_shape_vec((nrows, ncols), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            imag: None, data: arr.into_dyn(),
        });
    }

    let ncols = ndarrays[0].data.shape().last().copied().unwrap_or(1);
    for arr in &ndarrays[1..] {
        let cols = arr.data.shape().last().copied().unwrap_or(1);
        if cols != ncols {
            return Err(PyValueError::new_err(
                "All arrays must have the same number of columns",
            ));
        }
    }
    let views: Vec<_> = ndarrays.iter().map(|a| a.data.view()).collect();
    let result =
        ndarray::concatenate(Axis(0), &views).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None, data: result.into_dyn(),
    })
}

#[pyfunction]
fn hstack(arrays: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut ndarrays: Vec<NdArray> = Vec::with_capacity(list.len());
    for item in list.iter() {
        ndarrays.push(item.extract::<NdArray>()?);
    }
    if ndarrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let ndim = ndarrays[0].data.ndim();
    let axis = if ndim <= 1 { 0 } else { 1 };
    let views: Vec<_> = ndarrays.iter().map(|a| a.data.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None, data: result.into_dyn(),
    })
}

#[pyfunction]
fn tile(a: &NdArray, reps: Vec<usize>) -> PyResult<NdArray> {
    if reps.is_empty() {
        return Ok(NdArray {
            imag: None, data: a.data.clone(),
        });
    }
    let shape = a.data.shape().to_vec();
    let mut result = a.data.clone();
    if reps.len() > shape.len() {
        let mut new_shape = vec![1usize; reps.len() - shape.len()];
        new_shape.extend_from_slice(&shape);
        result = result
            .into_shape_with_order(IxDyn(&new_shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    for (i, &rep) in reps.iter().enumerate() {
        if rep > 1 {
            let mut pieces = Vec::with_capacity(rep);
            for _ in 0..rep {
                pieces.push(result.view());
            }
            result = ndarray::concatenate(Axis(i), &pieces)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    Ok(NdArray {
        imag: None, data: result.into_dyn(),
    })
}

#[pyfunction]
fn squeeze(a: &NdArray) -> PyResult<NdArray> {
    a.squeeze()
}

#[pyfunction]
fn flatten(a: &NdArray) -> NdArray {
    a.flatten()
}

#[pyfunction]
#[pyo3(signature = (a, order="C"))]
fn flatten_full(a: &NdArray, order: &str) -> PyResult<NdArray> {
    if order == "F" && a.data.ndim() > 1 {
        let shape = a.data.shape().to_vec();
        let ndim = shape.len();
        let flat_c: Vec<f64> = a.data.iter().copied().collect();

        let mut strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let mut flat_f = Vec::with_capacity(flat_c.len());

        fn walk_f(
            order: usize,
            pos: usize,
            shape: &[usize],
            strides: &[usize],
            flat_c: &[f64],
            flat_f: &mut Vec<f64>,
        ) {
            if order == 0 {
                flat_f.push(flat_c[pos]);
            } else {
                for v in 0..shape[order - 1] {
                    walk_f(
                        order - 1,
                        pos + v * strides[order - 1],
                        shape,
                        strides,
                        flat_c,
                        flat_f,
                    );
                }
            }
        }

        walk_f(ndim, 0, &shape, &strides, &flat_c, &mut flat_f);

        let arr = Array::from_shape_vec(IxDyn(&[flat_f.len()]), flat_f)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { imag: None, data: arr })
    } else {
        Ok(a.flatten())
    }
}

#[pyfunction]
fn reshape(a: &NdArray, shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    a.reshape(shape)
}

#[pyfunction]
fn expand_dims(a: &NdArray, axis: isize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = if axis < 0 {
        (ndim as isize + axis + 1) as usize
    } else {
        axis as usize
    };
    let mut shape = a.data.shape().to_vec();
    shape.insert(ax, 1);
    let arr = a
        .data
        .clone()
        .into_shape_with_order(IxDyn(&shape))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { imag: None, data: arr })
}

#[pyfunction]
fn column_stack(arrays: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut ndarrays: Vec<NdArray> = Vec::with_capacity(list.len());
    for item in list.iter() {
        ndarrays.push(item.extract::<NdArray>()?);
    }
    if ndarrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let reshaped: Vec<_> = ndarrays
        .iter()
        .map(|arr| {
            if arr.data.ndim() == 1 {
                let shape = vec![arr.data.len(), 1];
                arr.data
                    .clone()
                    .into_shape_with_order(IxDyn(&shape))
                    .unwrap()
                    .into_dyn()
            } else {
                arr.data.clone()
            }
        })
        .collect();
    let views: Vec<_> = reshaped.iter().map(|a| a.view()).collect();
    let result =
        ndarray::concatenate(Axis(1), &views).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None, data: result.into_dyn(),
    })
}

#[pyfunction]
fn roll(a: &NdArray, shift: isize, axis: Option<isize>) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = match axis {
        Some(ax) => {
            if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            }
        }
        None => 0,
    };
    if ndim == 0 {
        return Ok(NdArray {
            imag: None, data: a.data.clone(),
        });
    }
    let shape = a.data.shape().to_vec();
    let axis_size = shape[ax] as isize;
    let shift = shift.rem_euclid(axis_size);
    let data_vec: Vec<f64> = a.data.iter().copied().collect();
    let pre_size: usize = shape.iter().take(ax).product();
    let post_size: usize = shape.iter().skip(ax + 1).product();
    let block_size = axis_size as usize * post_size;
    let mut result = Vec::with_capacity(data_vec.len());
    for outer in 0..pre_size {
        for k in 0..axis_size as usize {
            let src = ((k as isize - shift).rem_euclid(axis_size)) as usize;
            for inner in 0..post_size {
                result.push(data_vec[outer * block_size + src * post_size + inner]);
            }
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { imag: None, data: arr })
}

#[pyfunction]
fn rot90(a: &NdArray, k: isize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    if ndim < 2 {
        return Err(PyValueError::new_err("rot90 requires at least 2D array"));
    }
    let k = k.rem_euclid(4);
    let mut result = a.data.clone();
    for _ in 0..k {
        let shape = result.shape().to_vec();
        let rows = shape[0];
        let cols = shape[1];
        let mut new_data = vec![0.0; rows * cols];
        // 逆时针旋转 90°：new[i][j] = old[j][cols-1-i]
        for i in 0..cols {
            for j in 0..rows {
                new_data[i * rows + j] = result[[j, cols - 1 - i]];
            }
        }
        result = Array::from_shape_vec((cols, rows), new_data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_dyn();
    }
    Ok(NdArray { imag: None, data: result })
}

#[pyfunction]
fn broadcast_to(a: &NdArray, shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let s = shape_to_vec(shape)?;
    let broadcast = a
        .data
        .broadcast(IxDyn(&s))
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "Cannot broadcast shape {:?} to {:?}",
                a.data.shape(),
                s
            ))
        })?
        .to_owned();
    Ok(NdArray { imag: None, data: broadcast })
}

// ========== 翻转数组 ==========
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn flip(a: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    let ndim = shape.len();
    let data_vec: Vec<f64> = a.data.iter().copied().collect();

    match axis {
        None => {
            let mut result = vec![0.0; data_vec.len()];
            for (i, &v) in data_vec.iter().rev().enumerate() {
                result[i] = v;
            }
            let arr = Array::from_shape_vec(IxDyn(&shape), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { imag: None, data: arr })
        }
        Some(ax) => {
            let ax = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };
            if ax >= ndim {
                return Err(PyValueError::new_err(format!("axis {} out of bounds", ax)));
            }
            let axis_size = shape[ax];
            let pre: usize = shape.iter().take(ax).product();
            let post: usize = shape.iter().skip(ax + 1).product();
            let mut result = vec![0.0; data_vec.len()];

            for p in 0..pre {
                for i in 0..axis_size {
                    let src = p * axis_size * post + i * post;
                    let dst = p * axis_size * post + (axis_size - 1 - i) * post;
                    result[dst..dst + post].copy_from_slice(&data_vec[src..src + post]);
                }
            }
            let arr = Array::from_shape_vec(IxDyn(&shape), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { imag: None, data: arr })
        }
    }
}

// ========== 拆分数组 ==========
#[pyfunction]
#[pyo3(signature = (a, indices_or_sections, axis=0))]
fn split_rs(
    a: &NdArray,
    indices_or_sections: &Bound<'_, PyAny>,
    axis: isize,
) -> PyResult<Vec<NdArray>> {
    let shape = a.data.shape().to_vec();
    let ndim = shape.len();
    let ax = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(PyValueError::new_err(format!(
            "axis {} out of bounds",
            axis
        )));
    }
    let axis_size = shape[ax];

    let sections: Vec<usize> = if let Ok(n) = indices_or_sections.extract::<usize>() {
        (1..n).map(|i| (axis_size * i) / n).collect()
    } else if let Ok(indices) = indices_or_sections.extract::<Vec<isize>>() {
        indices
            .iter()
            .map(|&v| {
                if v < 0 {
                    (axis_size as isize + v) as usize
                } else {
                    v as usize
                }
            })
            .filter(|&v| v <= axis_size)
            .collect()
    } else if let Ok(indices) = indices_or_sections.extract::<Vec<i64>>() {
        indices
            .iter()
            .map(|&v| {
                if v < 0 {
                    (axis_size as i64 + v) as usize
                } else {
                    v as usize
                }
            })
            .filter(|&v| v <= axis_size)
            .collect()
    } else {
        vec![]
    };

    let mut split_points = sections.clone();
    split_points.push(axis_size);
    split_points.insert(0, 0);
    // 去重并保持顺序
    split_points.dedup();

    let pre: usize = shape.iter().take(ax).product();
    let post: usize = shape.iter().skip(ax + 1).product();
    let data_vec: Vec<f64> = a.data.iter().copied().collect();

    let mut result = Vec::with_capacity(split_points.len() - 1);
    for i in 0..split_points.len() - 1 {
        let start = split_points[i].min(axis_size);
        let end = split_points[i + 1].min(axis_size);
        if start >= end {
            // 生成空的切片
            let mut empty_shape = shape.clone();
            empty_shape[ax] = 0;
            let section_data: Vec<f64> = Vec::new();
            let arr = Array::from_shape_vec(IxDyn(&empty_shape), section_data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            result.push(NdArray { imag: None, data: arr });
            continue;
        }
        let section_size = end - start;
        let mut new_shape = shape.clone();
        new_shape[ax] = section_size;
        let total: usize = pre * section_size * post;
        let mut section_data = vec![0.0; total];

        for p in 0..pre {
            for s in 0..section_size {
                let src = p * axis_size * post + (start + s) * post;
                let dst = p * section_size * post + s * post;
                section_data[dst..dst + post].copy_from_slice(&data_vec[src..src + post]);
            }
        }
        let arr = Array::from_shape_vec(IxDyn(&new_shape), section_data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        result.push(NdArray { imag: None, data: arr });
    }
    Ok(result)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(swapaxes, m)?)?;
    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    m.add_function(wrap_pyfunction!(column_stack, m)?)?;
    m.add_function(wrap_pyfunction!(tile, m)?)?;
    m.add_function(wrap_pyfunction!(squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(flatten, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_full, m)?)?;
    m.add_function(wrap_pyfunction!(reshape, m)?)?;
    m.add_function(wrap_pyfunction!(expand_dims, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_to, m)?)?;
    m.add_function(wrap_pyfunction!(roll, m)?)?;
    m.add_function(wrap_pyfunction!(rot90, m)?)?;
    m.add_function(wrap_pyfunction!(flip, m)?)?;
    m.add_function(wrap_pyfunction!(resize_rs, m)?)?;
    m.add_function(wrap_pyfunction!(delete_rs, m)?)?;
    m.add_function(wrap_pyfunction!(insert_rs, m)?)?;
    m.add_function(wrap_pyfunction!(split_rs, m)?)?;
    Ok(())
}

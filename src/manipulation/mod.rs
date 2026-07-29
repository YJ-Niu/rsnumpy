use crate::*;

#[pyfunction]
fn concatenate(arrays: &Bound<'_, PyAny>, axis: usize) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut re_arrays: Vec<Array<f64, IxDyn>> = Vec::with_capacity(list.len());
    let mut im_arrays: Vec<Option<Array<f64, IxDyn>>> = Vec::with_capacity(list.len());
    let mut has_imag = false;
    for item in list.iter() {
        let nd = item.extract::<NdArray>()?;
        re_arrays.push(nd.data);
        let has_im = nd.imag.is_some();
        im_arrays.push(nd.imag);
        if has_im {
            has_imag = true;
        }
    }
    if re_arrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let re_views: Vec<_> = re_arrays.iter().map(|a| a.view()).collect();
    let result_re = ndarray::concatenate(Axis(axis), &re_views)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if has_imag {
        let mut imag_sized: Vec<Array<f64, IxDyn>> = Vec::with_capacity(im_arrays.len());
        for (i, im_opt) in im_arrays.iter().enumerate() {
            if let Some(im) = im_opt {
                imag_sized.push(im.clone());
            } else {
                imag_sized.push(Array::zeros(re_arrays[i].raw_dim()));
            }
        }
        let im_views: Vec<_> = imag_sized.iter().map(|a| a.view()).collect();
        let result_im = ndarray::concatenate(Axis(axis), &im_views)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: Some(result_im.into_dyn()),
            data: result_re.into_dyn(),
        })
    } else {
        Ok(NdArray {
            imag: None,
            data: result_re.into_dyn(),
        })
    }
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
    // 检测是否存在复数数组
    let has_imag = ndarrays.iter().any(|arr| arr.imag.is_some());

    // 辅助闭包：对给定分量数组执行 expand + concatenate
    let stack_component = |comps: Vec<Array<f64, IxDyn>>| -> PyResult<Array<f64, IxDyn>> {
        let expanded_views: Vec<_> = comps
            .iter()
            .map(|arr| {
                let mut s = orig_shape.clone();
                s.insert(axis, 1);
                arr.clone()
                    .into_shape_with_order(IxDyn(&s))
                    .unwrap()
                    .into_dyn()
            })
            .collect();
        let views: Vec<_> = expanded_views.iter().map(|a| a.view()).collect();
        let result = ndarray::concatenate(Axis(axis), &views)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_dyn())
    };

    let re_comps: Vec<_> = ndarrays.iter().map(|a| a.data.clone()).collect();
    let result_re = stack_component(re_comps)?;
    if has_imag {
        let im_comps: Vec<_> = ndarrays
            .iter()
            .map(|a| {
                a.imag
                    .clone()
                    .unwrap_or_else(|| Array::zeros(a.data.raw_dim()))
            })
            .collect();
        let result_im = stack_component(im_comps)?;
        Ok(NdArray {
            imag: Some(result_im),
            data: result_re,
        })
    } else {
        Ok(NdArray {
            imag: None,
            data: result_re,
        })
    }
}

#[pyfunction]
fn transpose(a: &NdArray) -> PyResult<NdArray> {
    a.t()
}

#[pyfunction]
#[pyo3(signature = (a, axes=None))]
fn transpose_axes(a: &NdArray, axes: Option<Vec<isize>>) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let axes = match axes {
        Some(ax) => ax
            .iter()
            .map(|&x| {
                if x < 0 {
                    (ndim as isize + x) as usize
                } else {
                    x as usize
                }
            })
            .collect::<Vec<_>>(),
        None => (0..ndim).rev().collect(),
    };
    if axes.len() != ndim {
        return Err(PyValueError::new_err("axes don't match array dimensions"));
    }
    let result = a
        .data
        .view()
        .permuted_axes(axes.clone())
        .into_owned()
        .into_dyn();
    let imag = a
        .imag
        .as_ref()
        .map(|im| im.view().permuted_axes(axes).into_owned().into_dyn());
    Ok(NdArray { imag, data: result })
}

#[pyfunction]
#[pyo3(signature = (a, source, destination))]
fn moveaxis(a: &NdArray, source: Vec<isize>, destination: Vec<isize>) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let norm_source: Vec<usize> = source
        .iter()
        .map(|&s| {
            if s < 0 {
                (ndim as isize + s) as usize
            } else {
                s as usize
            }
        })
        .collect();
    let norm_dest: Vec<usize> = destination
        .iter()
        .map(|&d| {
            if d < 0 {
                (ndim as isize + d) as usize
            } else {
                d as usize
            }
        })
        .collect();
    if norm_source.len() != norm_dest.len() {
        return Err(PyValueError::new_err(
            "source and destination arguments must have the same number of elements",
        ));
    }
    let mut axes: Vec<usize> = (0..ndim).collect();
    let mut removed = Vec::with_capacity(norm_source.len());
    for &src in norm_source.iter().rev() {
        if src >= axes.len() {
            return Err(PyValueError::new_err("axis out of bounds"));
        }
        removed.push(axes.remove(src));
    }
    removed.reverse();
    for (dest, ax) in norm_dest.iter().zip(removed.iter()) {
        if *dest > axes.len() {
            return Err(PyValueError::new_err("axis out of bounds"));
        }
        axes.insert(*dest, *ax);
    }
    let result = a
        .data
        .view()
        .permuted_axes(axes.clone())
        .into_owned()
        .into_dyn();
    let imag = a
        .imag
        .as_ref()
        .map(|im| im.view().permuted_axes(axes).into_owned().into_dyn());
    Ok(NdArray { imag, data: result })
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
        return Ok(NdArray {
            imag: None,
            data: arr,
        });
    }

    let mut result = Vec::with_capacity(new_size);
    for i in 0..new_size {
        result.push(flat_data[i % flat_data.len()]);
    }

    let arr = Array::from_shape_vec(IxDyn(&new_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
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
        return Ok(NdArray {
            imag: None,
            data: arr,
        });
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
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
#[pyo3(signature = (a, indices, values, axis=None))]
fn insert_rs(
    a: &NdArray,
    indices: Vec<isize>,
    values: Vec<f64>,
    axis: Option<isize>,
) -> PyResult<NdArray> {
    // 处理复数数组：values 为 [real0, imag0, real1, imag1, ...]
    if a.has_imag() {
        // 将 values 分离为实部和虚部
        let mut re_values = Vec::new();
        let mut im_values = Vec::new();
        for i in (0..values.len()).step_by(2) {
            re_values.push(values[i]);
            let im_val = if i + 1 < values.len() {
                values[i + 1]
            } else {
                0.0
            };
            im_values.push(im_val);
        }

        // 对实部和虚部分别调用 insert_rs_internal
        let re_result = insert_rs_internal(&a.data, &indices, &re_values, axis)?;
        let im_array = a.imag.as_ref().unwrap();
        let im_result = insert_rs_internal(im_array, &indices, &im_values, axis)?;

        return Ok(NdArray {
            imag: Some(im_result),
            data: re_result,
        });
    }

    // 实数数组：直接处理
    let result = insert_rs_internal(&a.data, &indices, &values, axis)?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

fn insert_rs_internal(
    a: &Array<f64, IxDyn>,
    indices: &[isize],
    values: &[f64],
    axis: Option<isize>,
) -> PyResult<Array<f64, IxDyn>> {
    let ndim = a.ndim();
    let ax = axis.map(|x| {
        if x < 0 {
            (ndim as isize + x) as usize
        } else {
            x as usize
        }
    });

    let flat_data: Vec<f64> = a.iter().copied().collect();

    if ax.is_none() {
        let mut result = flat_data.clone();
        let mut offset = 0;
        let mut sorted_indices: Vec<isize> = indices.to_vec();
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
        return Ok(arr);
    }

    let ax = ax.unwrap();
    let shape = a.shape().to_vec();
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
    Ok(arr)
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
            imag: None,
            data: arr.into_dyn(),
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
        imag: None,
        data: result.into_dyn(),
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
        imag: None,
        data: result.into_dyn(),
    })
}

#[pyfunction]
fn tile(a: &NdArray, reps: Vec<usize>) -> PyResult<NdArray> {
    if reps.is_empty() {
        return Ok(NdArray {
            imag: None,
            data: a.data.clone(),
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
        imag: None,
        data: result.into_dyn(),
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
        Ok(NdArray {
            imag: None,
            data: arr,
        })
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
    Ok(NdArray {
        imag: None,
        data: arr,
    })
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
        imag: None,
        data: result.into_dyn(),
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
        return Ok(a.clone());
    }
    let shape = a.data.shape().to_vec();
    let axis_size = shape[ax] as isize;
    let shift = shift.rem_euclid(axis_size);
    let pre_size: usize = shape.iter().take(ax).product();
    let post_size: usize = shape.iter().skip(ax + 1).product();
    let block_size = axis_size as usize * post_size;

    let data_vec: Vec<f64> = a.data.iter().copied().collect();
    let mut result_re = Vec::with_capacity(data_vec.len());
    for outer in 0..pre_size {
        for k in 0..axis_size as usize {
            let src = ((k as isize - shift).rem_euclid(axis_size)) as usize;
            for inner in 0..post_size {
                result_re.push(data_vec[outer * block_size + src * post_size + inner]);
            }
        }
    }
    let arr_re = Array::from_shape_vec(IxDyn(&shape), result_re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if let Some(im) = &a.imag {
        let im_vec: Vec<f64> = im.iter().copied().collect();
        let mut result_im = Vec::with_capacity(im_vec.len());
        for outer in 0..pre_size {
            for k in 0..axis_size as usize {
                let src = ((k as isize - shift).rem_euclid(axis_size)) as usize;
                for inner in 0..post_size {
                    result_im.push(im_vec[outer * block_size + src * post_size + inner]);
                }
            }
        }
        let arr_im = Array::from_shape_vec(IxDyn(&shape), result_im)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: Some(arr_im),
            data: arr_re,
        })
    } else {
        Ok(NdArray {
            imag: None,
            data: arr_re,
        })
    }
}

#[pyfunction]
#[pyo3(signature = (a, k=1, axis1=0, axis2=1))]
fn rot90(a: &NdArray, k: isize, axis1: usize, axis2: usize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    if ndim < 2 {
        return Err(PyValueError::new_err("rot90 requires at least 2D array"));
    }
    if axis1 >= ndim || axis2 >= ndim {
        return Err(PyValueError::new_err("axis out of bounds"));
    }
    if axis1 == axis2 {
        return Err(PyValueError::new_err("axes must be different"));
    }
    let k = k.rem_euclid(4);
    if k == 0 {
        return Ok(a.clone());
    }

    fn flip_axis(data: &Array<f64, IxDyn>, axis: usize) -> Array<f64, IxDyn> {
        let shape = data.shape().to_vec();
        let axis_size = shape[axis];
        let pre: usize = shape.iter().take(axis).product();
        let post: usize = shape.iter().skip(axis + 1).product();
        let data_vec: Vec<f64> = data.iter().copied().collect();
        let mut result = vec![0.0; data_vec.len()];
        for p in 0..pre {
            for i in 0..axis_size {
                let src = p * axis_size * post + i * post;
                let dst = p * axis_size * post + (axis_size - 1 - i) * post;
                result[dst..dst + post].copy_from_slice(&data_vec[src..src + post]);
            }
        }
        Array::from_shape_vec(IxDyn(&shape), result).unwrap()
    }

    let mut result = a.data.clone();
    let mut imag = a.imag.clone();

    match k {
        1 => {
            result.swap_axes(axis1, axis2);
            result = flip_axis(&result, axis2);
            if let Some(im) = imag.as_mut() {
                im.swap_axes(axis1, axis2);
                *im = flip_axis(im, axis2);
            }
        }
        2 => {
            result = flip_axis(&result, axis1);
            result = flip_axis(&result, axis2);
            if let Some(im) = imag.as_mut() {
                *im = flip_axis(im, axis1);
                *im = flip_axis(im, axis2);
            }
        }
        3 => {
            result.swap_axes(axis1, axis2);
            result = flip_axis(&result, axis1);
            if let Some(im) = imag.as_mut() {
                im.swap_axes(axis1, axis2);
                *im = flip_axis(im, axis1);
            }
        }
        _ => {}
    }

    Ok(NdArray { imag, data: result })
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
    Ok(NdArray {
        imag: None,
        data: broadcast,
    })
}

// ========== 翻转数组 ==========
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn flip(a: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    let ndim = shape.len();
    let data_vec: Vec<f64> = a.data.iter().copied().collect();

    let flip_data = |data: &[f64]| -> Vec<f64> {
        match axis {
            None => {
                let mut result = vec![0.0; data.len()];
                for (i, &v) in data.iter().rev().enumerate() {
                    result[i] = v;
                }
                result
            }
            Some(ax) => {
                let ax = if ax < 0 {
                    (ndim as isize + ax) as usize
                } else {
                    ax as usize
                };
                if ax >= ndim {
                    return data.to_vec();
                }
                let axis_size = shape[ax];
                let pre: usize = shape.iter().take(ax).product();
                let post: usize = shape.iter().skip(ax + 1).product();
                let mut result = vec![0.0; data.len()];

                for p in 0..pre {
                    for i in 0..axis_size {
                        let src = p * axis_size * post + i * post;
                        let dst = p * axis_size * post + (axis_size - 1 - i) * post;
                        result[dst..dst + post].copy_from_slice(&data[src..src + post]);
                    }
                }
                result
            }
        }
    };

    let result_re = flip_data(&data_vec);
    let arr_re = Array::from_shape_vec(IxDyn(&shape), result_re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if let Some(im) = &a.imag {
        let im_vec: Vec<f64> = im.iter().copied().collect();
        let result_im = flip_data(&im_vec);
        let arr_im = Array::from_shape_vec(IxDyn(&shape), result_im)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: Some(arr_im),
            data: arr_re,
        })
    } else {
        Ok(NdArray {
            imag: None,
            data: arr_re,
        })
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
            result.push(NdArray {
                imag: None,
                data: arr,
            });
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
        result.push(NdArray {
            imag: None,
            data: arr,
        });
    }
    Ok(result)
}

// ========== 对角线操作 ==========

#[pyfunction]
#[pyo3(signature = (v, k=0))]
fn diag(v: &NdArray, k: isize) -> PyResult<NdArray> {
    let shape = v.data.shape();
    if shape.len() == 2 {
        // 二维数组：提取第 k 条对角线
        let n_rows = shape[0];
        let n_cols = shape[1];
        let mut diag_vals = Vec::new();
        let start_row = if k < 0 { (-k) as usize } else { 0 };
        let start_col = if k > 0 { k as usize } else { 0 };
        let mut r = start_row;
        let mut c = start_col;
        while r < n_rows && c < n_cols {
            diag_vals.push(v.data[[r, c]]);
            r += 1;
            c += 1;
        }
        Ok(NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&[diag_vals.len()]), diag_vals)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    } else {
        // 一维数组：构造以 v 为第 k 条对角线的方阵
        let flat: Vec<f64> = v.data.iter().copied().collect();
        let n = flat.len() + k.unsigned_abs();
        let mut out = vec![0.0_f64; n * n];
        for (i, &val) in flat.iter().enumerate() {
            let (r, c) = if k >= 0 {
                (i, i + k as usize)
            } else {
                (i + k.unsigned_abs(), i)
            };
            if r < n && c < n {
                out[r * n + c] = val;
            }
        }
        Ok(NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&[n, n]), out)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }
}

#[pyfunction]
#[pyo3(signature = (n, k=0, m=None))]
fn tril_indices(n: usize, k: isize, m: Option<usize>) -> PyResult<(NdArray, NdArray)> {
    let cols = m.unwrap_or(n);
    let mut rows_out = Vec::new();
    let mut cols_out = Vec::new();
    for i in 0..n {
        // 下三角：j 从 0 到 min(i + k, cols - 1)
        let max_col = (i as isize + k).min(cols as isize - 1);
        if max_col < 0 {
            continue;
        }
        for j in 0..=max_col as usize {
            rows_out.push(i as f64);
            cols_out.push(j as f64);
        }
    }
    Ok((
        NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&[rows_out.len()]), rows_out)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        },
        NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&[cols_out.len()]), cols_out)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        },
    ))
}

#[pyfunction]
#[pyo3(signature = (n, k=0, m=None))]
fn triu_indices(n: usize, k: isize, m: Option<usize>) -> PyResult<(NdArray, NdArray)> {
    let cols = m.unwrap_or(n);
    let mut rows_out = Vec::new();
    let mut cols_out = Vec::new();
    for i in 0..n {
        // 上三角：j 从 max(i + k, 0) 到 cols - 1
        let min_col = (i as isize + k).max(0);
        if min_col >= cols as isize {
            continue;
        }
        for j in min_col as usize..cols {
            rows_out.push(i as f64);
            cols_out.push(j as f64);
        }
    }
    Ok((
        NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&[rows_out.len()]), rows_out)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        },
        NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&[cols_out.len()]), cols_out)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        },
    ))
}

// ========== atleast 系列函数 ==========

#[pyfunction]
fn atleast_1d(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape();
    if shape.is_empty() {
        // 0-d 转为 1-d
        let val = a.data.iter().next().copied().unwrap_or(0.0);
        let imag = a.imag.as_ref().map(|im| {
            let im_val = im.iter().next().copied().unwrap_or(0.0);
            Array::from_elem(IxDyn(&[1]), im_val)
        });
        Ok(NdArray {
            imag,
            data: Array::from_elem(IxDyn(&[1]), val),
        })
    } else {
        // 已是一维及以上，直接返回克隆
        Ok(a.clone())
    }
}

#[pyfunction]
fn atleast_2d(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape();
    if shape.is_empty() {
        // 0-d 转为 2-d：shape [] -> [1, 1]
        let val = a.data.iter().next().copied().unwrap_or(0.0);
        let imag = a.imag.as_ref().map(|im| {
            let im_val = im.iter().next().copied().unwrap_or(0.0);
            Array::from_elem(IxDyn(&[1, 1]), im_val)
        });
        Ok(NdArray {
            imag,
            data: Array::from_elem(IxDyn(&[1, 1]), val),
        })
    } else if shape.len() == 1 {
        // 1-d 转为 2-d：shape [N] -> [1, N]
        let new_shape = vec![1usize, shape[0]];
        let re = a
            .data
            .clone()
            .into_shape_with_order(IxDyn(&new_shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let imag = if let Some(im) = &a.imag {
            Some(
                im.clone()
                    .into_shape_with_order(IxDyn(&new_shape.clone()))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        } else {
            None
        };
        Ok(NdArray { imag, data: re })
    } else {
        // 已是二维及以上，直接返回克隆
        Ok(a.clone())
    }
}

#[pyfunction]
fn atleast_3d(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape();
    if shape.is_empty() {
        // 0-d 转为 3-d：shape [] -> [1, 1, 1]
        let val = a.data.iter().next().copied().unwrap_or(0.0);
        let imag = a.imag.as_ref().map(|im| {
            let im_val = im.iter().next().copied().unwrap_or(0.0);
            Array::from_elem(IxDyn(&[1, 1, 1]), im_val)
        });
        Ok(NdArray {
            imag,
            data: Array::from_elem(IxDyn(&[1, 1, 1]), val),
        })
    } else if shape.len() == 1 {
        // 1-d 转为 3-d：shape [N] -> [1, N, 1]
        let new_shape = vec![1usize, shape[0], 1];
        let re = a
            .data
            .clone()
            .into_shape_with_order(IxDyn(&new_shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let imag = if let Some(im) = &a.imag {
            Some(
                im.clone()
                    .into_shape_with_order(IxDyn(&new_shape.clone()))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        } else {
            None
        };
        Ok(NdArray { imag, data: re })
    } else if shape.len() == 2 {
        // 2-d 转为 3-d：shape [M, N] -> [M, N, 1]
        let new_shape = vec![shape[0], shape[1], 1];
        let re = a
            .data
            .clone()
            .into_shape_with_order(IxDyn(&new_shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let imag = if let Some(im) = &a.imag {
            Some(
                im.clone()
                    .into_shape_with_order(IxDyn(&new_shape.clone()))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        } else {
            None
        };
        Ok(NdArray { imag, data: re })
    } else {
        // 已是三维及以上，直接返回克隆
        Ok(a.clone())
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(transpose_axes, m)?)?;
    m.add_function(wrap_pyfunction!(moveaxis, m)?)?;
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
    m.add_function(wrap_pyfunction!(diag, m)?)?;
    m.add_function(wrap_pyfunction!(tril_indices, m)?)?;
    m.add_function(wrap_pyfunction!(triu_indices, m)?)?;
    m.add_function(wrap_pyfunction!(atleast_1d, m)?)?;
    m.add_function(wrap_pyfunction!(atleast_2d, m)?)?;
    m.add_function(wrap_pyfunction!(atleast_3d, m)?)?;
    m.add_function(wrap_pyfunction!(ravel_multi_index, m)?)?;
    m.add_function(wrap_pyfunction!(unravel_index, m)?)?;
    m.add_function(wrap_pyfunction!(place, m)?)?;
    m.add_function(wrap_pyfunction!(fill_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(putmask, m)?)?;
    m.add_function(wrap_pyfunction!(take_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(put_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal_ext, m)?)?;
    m.add_function(wrap_pyfunction!(choose, m)?)?;
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(copyto, m)?)?;
    Ok(())
}

/// 将多维坐标转换为扁平索引
#[pyfunction]
#[pyo3(signature = (multi_index, dims, mode="raise", order="C"))]
fn ravel_multi_index(
    multi_index: Vec<NdArray>,
    dims: Vec<usize>,
    mode: &str,
    order: &str,
) -> PyResult<NdArray> {
    // 展平每个坐标数组，统一以 Vec<f64> 处理（兼容 0-d 标量输入）
    let keys: Vec<Vec<f64>> = multi_index
        .iter()
        .map(|a| a.data.iter().copied().collect())
        .collect();
    let n = keys[0].len();
    let ndim = dims.len();
    // 计算 strides
    let mut strides = vec![0usize; ndim];
    if order == "C" {
        let mut acc = 1;
        for i in (0..ndim).rev() {
            strides[i] = acc;
            acc *= dims[i];
        }
    } else {
        let mut acc = 1;
        for i in 0..ndim {
            strides[i] = acc;
            acc *= dims[i];
        }
    }
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mut s = 0i64;
        for (i, key) in keys.iter().enumerate() {
            let v = key[k] as i64;
            let v = match mode {
                "wrap" => {
                    if v >= 0 {
                        v % (dims[i] as i64)
                    } else {
                        let d = dims[i] as i64;
                        ((v % d) + d) % d
                    }
                }
                "clip" => v.clamp(0, dims[i] as i64 - 1),
                _ => {
                    if v < 0 || v >= dims[i] as i64 {
                        return Err(PyValueError::new_err("invalid entry in coordinates array"));
                    }
                    v
                }
            };
            s += v * strides[i] as i64;
        }
        out.push(s as f64);
    }
    // 全部 0-d 输入返回 0-d 输出（与 numpy 一致）
    let all_scalar = multi_index.iter().all(|a| a.data.ndim() == 0);
    if all_scalar {
        return Ok(NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&[]), out)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        });
    }
    Ok(NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&[out.len()]), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}

/// 将扁平索引转换为多维坐标
#[pyfunction]
#[pyo3(signature = (indices, shape, order="C"))]
fn unravel_index(indices: &NdArray, shape: Vec<usize>, order: &str) -> PyResult<Vec<NdArray>> {
    let ndim = shape.len();
    let flats: Vec<i64> = indices.data.iter().map(|&v| v as i64).collect();
    // 计算 strides
    let mut strides = vec![0usize; ndim];
    if order == "C" {
        let mut acc = 1;
        for i in (0..ndim).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
    } else {
        let mut acc = 1;
        for i in 0..ndim {
            strides[i] = acc;
            acc *= shape[i];
        }
    }
    let _ = strides; // 仅用于校验形状顺序，后续按 rng 处理
    let n = flats.len();
    let mut cols = vec![vec![0f64; n]; ndim];
    for (k, &flat) in flats.iter().enumerate() {
        let mut f = flat;
        let rng: Vec<usize> = if order == "C" {
            (0..ndim).rev().collect()
        } else {
            (0..ndim).collect()
        };
        for i in rng {
            cols[i][k] = (f % shape[i] as i64) as f64;
            f /= shape[i] as i64;
        }
    }
    // 0-d 输入返回 0-d 输出（与 numpy 一致）
    let is_scalar = indices.data.ndim() == 0;
    let result: Vec<NdArray> = cols
        .into_iter()
        .map(|c| {
            let data = if is_scalar {
                Array::from_shape_vec(IxDyn(&[]), c).expect("形状正确")
            } else {
                Array::from_shape_vec(IxDyn(&[c.len()]), c).expect("形状正确")
            };
            NdArray { imag: None, data }
        })
        .collect();
    Ok(result)
}

/// 按布尔掩码循环写入 vals 到 arr（就地，与 numpy.place 一致）。
#[pyfunction]
fn place(arr: &mut NdArray, mask: &NdArray, vals: &NdArray) -> PyResult<()> {
    let v: Vec<f64> = vals.data.iter().copied().collect();
    if v.is_empty() {
        return Ok(());
    }
    // 必须同形状才能就地写
    if arr.data.shape() != mask.data.shape() {
        return Err(PyValueError::new_err(
            "place: mask and arr must have the same shape",
        ));
    }
    let data_slice = arr
        .data
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("place requires contiguous array"))?;
    let mask_slice = mask
        .data
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("place requires contiguous mask"))?;
    let mut j = 0usize;
    for i in 0..data_slice.len() {
        if mask_slice[i] != 0.0 {
            data_slice[i] = v[j % v.len()];
            j += 1;
        }
    }
    Ok(())
}

/// 就地填充主对角线（支持 wrap 参数，与 numpy.fill_diagonal 一致）。
///
/// numpy 算法：
/// - 2D：step = shape[1] + 1；不 wrap 时 end = shape[1]^2；wrap 时 end = None。
///   沿 flat 视图按 step 步长写入，越界自动停止。
/// - N-D (n>2)：要求所有维度等长；step = 1 + cumprod(shape[:-1]).sum()。
#[pyfunction]
#[pyo3(signature = (arr, vals, wrap=false))]
fn fill_diagonal(arr: &mut NdArray, vals: &NdArray, wrap: bool) -> PyResult<()> {
    let shape: Vec<usize> = arr.data.shape().to_vec();
    if shape.len() < 2 {
        return Err(PyValueError::new_err(
            "fill_diagonal requires at least 2D array",
        ));
    }
    let v: Vec<f64> = vals.data.iter().copied().collect();
    if v.is_empty() {
        return Ok(());
    }
    let total: usize = shape.iter().product();
    if total == 0 {
        return Ok(());
    }
    // 计算 step（按 C-contiguous flat 索引）
    let step: usize = if shape.len() == 2 {
        shape[1] + 1
    } else {
        // N-D：所有维度必须等长
        for d in &shape[1..] {
            if *d != shape[0] {
                return Err(PyValueError::new_err(
                    "All dimensions of input must be of equal length",
                ));
            }
        }
        // step = 1 + sum(cumprod(shape[:-1]))
        let mut acc = 1usize;
        let mut sum = 0usize;
        for d in &shape[..shape.len() - 1] {
            acc *= *d;
            sum += acc;
        }
        1 + sum
    };
    // 不 wrap 时 end = shape[1]^2（2D）或 None（N-D 已经被等长约束保护）
    let end: usize = if shape.len() == 2 && !wrap {
        shape[1] * shape[1]
    } else {
        total
    };
    let data_slice = arr
        .data
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("fill_diagonal requires contiguous array"))?;
    let mut k = 0usize;
    let mut i = 0usize;
    while i < end && i < total {
        data_slice[i] = v[k % v.len()];
        k += 1;
        i += step;
    }
    Ok(())
}

/// 按布尔掩码逐元素写入值（就地，与 numpy.putmask 一致：a[i]=v[i] where mask[i]）。
///
/// 与 numpy.place 不同：putmask 按位置元素级赋值，vals 不循环而是按对应索引取值。
#[pyfunction]
fn putmask(arr: &mut NdArray, mask: &NdArray, vals: &NdArray) -> PyResult<()> {
    let total: usize = arr.data.len();
    if mask.data.len() != total {
        return Err(PyValueError::new_err(
            "putmask: mask and arr must have the same shape",
        ));
    }
    let data_slice = arr
        .data
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("putmask requires contiguous array"))?;
    let mask_slice = mask
        .data
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("putmask requires contiguous mask"))?;
    let v: Vec<f64> = vals.data.iter().copied().collect();
    if v.is_empty() {
        return Ok(());
    }
    let v_len = v.len();
    // 标量：直接广播；数组：按位置元素级赋值
    if v_len == 1 {
        for i in 0..total {
            if mask_slice[i] != 0.0 {
                data_slice[i] = v[0];
            }
        }
    } else {
        // vals 与 arr/mask 同形（按位置对应），mask[i] 真时 a[i]=v[i]
        if v_len != total {
            return Err(PyValueError::new_err(format!(
                "putmask: cannot broadcast {} values to {} positions",
                v_len, total
            )));
        }
        for i in 0..total {
            if mask_slice[i] != 0.0 {
                data_slice[i] = v[i];
            }
        }
    }
    Ok(())
}

/// 沿轴按索引数组取值（与 numpy.take_along_axis 一致）。
#[pyfunction]
#[pyo3(signature = (arr, indices, axis))]
fn take_along_axis(arr: &NdArray, indices: &NdArray, axis: isize) -> PyResult<NdArray> {
    let shape: Vec<usize> = arr.data.shape().to_vec();
    let ndim = shape.len();
    if ndim == 0 {
        return Err(PyValueError::new_err(
            "take_along_axis: 0-d array not supported",
        ));
    }
    let ax = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(PyValueError::new_err("take_along_axis: axis out of bounds"));
    }
    let idx_shape = indices.data.shape();
    // 输出形状：除 ax 维替换为 indices 在该维的长度
    let mut out_shape: Vec<usize> = shape.clone();
    out_shape[ax] = idx_shape.get(ax).copied().unwrap_or(idx_shape[0]);
    let out_len: usize = out_shape.iter().product();
    let mut out = vec![0.0_f64; out_len];

    // 计算 arr 与 indices 在 ax 之外各维度的广播形状
    let mut other_shape = Vec::with_capacity(ndim - 1);
    for (i, &s) in shape.iter().enumerate() {
        if i != ax {
            other_shape.push(s);
        }
    }
    let other_total: usize = other_shape.iter().product::<usize>().max(1);
    let ax_len = shape[ax];
    let idx_ax_len = idx_shape.get(ax).copied().unwrap_or(idx_shape[0]);

    // arr 与 indices 的 ax-stride
    let arr_strides = arr.data.strides();
    let idx_strides = indices.data.strides();
    let arr_ax_stride = arr_strides[ax] as usize;
    let idx_ax_stride = idx_strides.get(ax).copied().unwrap_or(1) as usize;

    // 其他维度的步长（按 ax 分割为前后两段）
    let mut arr_other_strides = Vec::with_capacity(ndim - 1);
    let mut idx_other_strides = Vec::with_capacity(ndim - 1);
    for (i, &s) in arr_strides.iter().enumerate() {
        if i != ax {
            arr_other_strides.push(s as usize);
        }
    }
    for (i, &s) in idx_strides.iter().enumerate() {
        if i != ax {
            idx_other_strides.push(s as usize);
        }
    }

    // 输出数组的步长
    let mut out_strides = vec![0usize; ndim];
    {
        let mut acc = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = acc;
            acc *= out_shape[i];
        }
    }
    let out_ax_stride = out_strides[ax];
    let mut out_other_strides = Vec::with_capacity(ndim - 1);
    for (i, &s) in out_strides.iter().enumerate() {
        if i != ax {
            out_other_strides.push(s);
        }
    }

    // arr / indices 的扁平 slice（要求连续）
    let arr_slice = arr
        .data
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("take_along_axis requires contiguous arr"))?;
    let idx_slice = indices
        .data
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("take_along_axis requires contiguous indices"))?;

    // 遍历其他维度组合
    let mut other_idx = vec![0usize; other_shape.len()];
    for _ in 0..other_total {
        // 计算 arr、indices、out 在其他维度的偏移
        let mut arr_off = 0usize;
        let mut idx_off = 0usize;
        let mut out_off = 0usize;
        for i in 0..other_idx.len() {
            arr_off += other_idx[i] * arr_other_strides[i];
            idx_off += other_idx[i] * idx_other_strides[i];
            out_off += other_idx[i] * out_other_strides[i];
        }
        // 沿 ax 维度取值
        for k in 0..idx_ax_len {
            let idx = idx_slice[idx_off + k * idx_ax_stride] as usize;
            if idx >= ax_len {
                return Err(PyValueError::new_err(
                    "take_along_index: index out of bounds",
                ));
            }
            out[out_off + k * out_ax_stride] = arr_slice[arr_off + idx * arr_ax_stride];
        }
        // 递增 other_idx
        if !other_idx.is_empty() {
            let mut i = other_idx.len() - 1;
            loop {
                other_idx[i] += 1;
                if other_idx[i] < other_shape[i] {
                    break;
                }
                other_idx[i] = 0;
                if i == 0 {
                    break;
                }
                i -= 1;
            }
        }
    }
    Ok(NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&out_shape), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}

/// 沿轴按索引数组写入值（就地，与 numpy.put_along_axis 一致）。
///
/// numpy 语义：indices 与 values 同形（或可广播），按对应位置元素级取值。
/// 即对每个 (其它维度坐标, k)，写入 arr[..., idx[..., k], ...] = values[..., k, ...]。
#[pyfunction]
#[pyo3(signature = (arr, indices, values, axis))]
fn put_along_axis(
    arr: &mut NdArray,
    indices: &NdArray,
    values: &NdArray,
    axis: isize,
) -> PyResult<()> {
    // 先把不可变信息收集到本地变量，避免对 arr.data 的可变/不可变借用冲突
    let shape: Vec<usize> = arr.data.shape().to_vec();
    let arr_strides: Vec<isize> = arr.data.strides().to_vec();
    let ndim = shape.len();
    if ndim == 0 {
        return Err(PyValueError::new_err(
            "put_along_axis: 0-d array not supported",
        ));
    }
    let ax = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(PyValueError::new_err("put_along_axis: axis out of bounds"));
    }
    let idx_shape = indices.data.shape();
    let idx_ax_len = idx_shape.get(ax).copied().unwrap_or(idx_shape[0]);
    let ax_len = shape[ax];

    let mut other_shape = Vec::with_capacity(ndim - 1);
    for (i, &s) in shape.iter().enumerate() {
        if i != ax {
            other_shape.push(s);
        }
    }
    let other_total: usize = other_shape.iter().product::<usize>().max(1);

    let idx_strides = indices.data.strides();
    let val_strides = values.data.strides();
    let arr_ax_stride = arr_strides[ax] as usize;
    let idx_ax_stride = idx_strides.get(ax).copied().unwrap_or(1) as usize;
    let val_ax_stride = val_strides.get(ax).copied().unwrap_or(1) as usize;
    let val_total: usize = values.data.len();
    let val_is_scalar = val_total == 1;

    let mut arr_other_strides = Vec::with_capacity(ndim - 1);
    let mut idx_other_strides = Vec::with_capacity(ndim - 1);
    let mut val_other_strides = Vec::with_capacity(ndim - 1);
    for (i, &s) in arr_strides.iter().enumerate() {
        if i != ax {
            arr_other_strides.push(s as usize);
        }
    }
    for (i, &s) in idx_strides.iter().enumerate() {
        if i != ax {
            idx_other_strides.push(s as usize);
        }
    }
    // values 的 ndim 可能小于 arr；按位置广播对齐到 arr 的其它维度
    let val_ndim = values.data.ndim();
    if val_ndim == ndim {
        for (i, &s) in val_strides.iter().enumerate() {
            if i != ax {
                val_other_strides.push(s as usize);
            }
        }
    } else if val_ndim == 0 {
        // 标量：所有 stride 为 0
        val_other_strides = vec![0usize; ndim - 1];
    } else {
        // 一般广播：低维补 1，高维按 ax 之外的顺序对齐
        // 简化：只支持同形或标量；其它广播由 Python 层先 broadcast_to 处理
        return Err(PyValueError::new_err(format!(
            "put_along_axis: values ndim {} not aligned with arr ndim {}",
            val_ndim, ndim
        )));
    }

    let data_slice = arr
        .data
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("put_along_axis requires contiguous arr"))?;
    let idx_slice = indices
        .data
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("put_along_axis requires contiguous indices"))?;
    let val_slice = values
        .data
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("put_along_axis requires contiguous values"))?;

    let mut other_idx = vec![0usize; other_shape.len()];
    for _ in 0..other_total {
        let mut arr_off = 0usize;
        let mut idx_off = 0usize;
        let mut val_off = 0usize;
        for i in 0..other_idx.len() {
            arr_off += other_idx[i] * arr_other_strides[i];
            idx_off += other_idx[i] * idx_other_strides[i];
            val_off += other_idx[i] * val_other_strides[i];
        }
        for k in 0..idx_ax_len {
            let idx = idx_slice[idx_off + k * idx_ax_stride] as usize;
            if idx >= ax_len {
                return Err(PyValueError::new_err("put_along_axis: index out of bounds"));
            }
            let v = if val_is_scalar {
                val_slice[0]
            } else {
                val_slice[val_off + k * val_ax_stride]
            };
            data_slice[arr_off + idx * arr_ax_stride] = v;
        }
        if !other_idx.is_empty() {
            let mut i = other_idx.len() - 1;
            loop {
                other_idx[i] += 1;
                if other_idx[i] < other_shape[i] {
                    break;
                }
                other_idx[i] = 0;
                if i == 0 {
                    break;
                }
                i -= 1;
            }
        }
    }
    Ok(())
}

/// 提取对角线（支持 axis1/axis2，与 numpy.diagonal 一致）。
#[pyfunction]
#[pyo3(signature = (a, offset=0, axis1=0, axis2=1))]
fn diagonal_ext(a: &NdArray, offset: isize, axis1: isize, axis2: isize) -> PyResult<NdArray> {
    let shape = a.data.shape();
    let ndim = shape.len();
    if ndim < 2 {
        return Err(PyValueError::new_err(
            "diagonal_ext: requires at least 2D array",
        ));
    }
    let ax1 = if axis1 < 0 {
        (ndim as isize + axis1) as usize
    } else {
        axis1 as usize
    };
    let ax2 = if axis2 < 0 {
        (ndim as isize + axis2) as usize
    } else {
        axis2 as usize
    };
    if ax1 == ax2 {
        return Err(PyValueError::new_err(
            "diagonal_ext: axis1 and axis2 cannot be equal",
        ));
    }
    if ax1 >= ndim || ax2 >= ndim {
        return Err(PyValueError::new_err("diagonal_ext: axis out of bounds"));
    }

    let d1 = shape[ax1];
    let d2 = shape[ax2];
    // 对角线长度
    let diag_len = if offset >= 0 {
        d1.min(d2.saturating_sub(offset as usize))
    } else {
        d1.saturating_sub((-offset) as usize).min(d2)
    };

    if diag_len == 0 {
        // 返回空数组（保留其他维度形状）
        let mut out_shape: Vec<usize> = Vec::new();
        for (i, &s) in shape.iter().enumerate() {
            if i != ax1 && i != ax2 {
                out_shape.push(s);
            }
        }
        out_shape.push(0);
        if out_shape.is_empty() {
            out_shape.push(0);
        }
        return Ok(NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&out_shape), Vec::new())
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        });
    }

    // 收集其他维度（非 ax1/ax2）
    let mut other_dims: Vec<usize> = Vec::with_capacity(ndim - 2);
    let mut other_axes: Vec<usize> = Vec::with_capacity(ndim - 2);
    for (i, &s) in shape.iter().enumerate() {
        if i != ax1 && i != ax2 {
            other_dims.push(s);
            other_axes.push(i);
        }
    }
    let other_total: usize = other_dims.iter().product::<usize>().max(1);
    let strides = a.data.strides();
    let arr_slice = a
        .data
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("diagonal_ext requires contiguous array"))?;

    // 其他维度的步长
    let mut other_strides = Vec::with_capacity(other_axes.len());
    for &i in &other_axes {
        other_strides.push(strides[i] as usize);
    }
    let s1 = strides[ax1] as usize;
    let s2 = strides[ax2] as usize;

    // 输出形状：其他维度 + (diag_len,)
    let mut out_shape: Vec<usize> = other_dims.clone();
    out_shape.push(diag_len);
    let out_len: usize = out_shape.iter().product();
    let mut out = vec![0.0_f64; out_len];

    // 输出 stride：其他维度在前，diag_len 在最后
    let mut out_strides = vec![0usize; out_shape.len()];
    {
        let mut acc = 1usize;
        for i in (0..out_shape.len()).rev() {
            out_strides[i] = acc;
            acc *= out_shape[i];
        }
    }
    let diag_out_stride = *out_strides.last().unwrap_or(&1);

    let mut other_idx = vec![0usize; other_dims.len()];
    for _ in 0..other_total {
        let mut base_off = 0usize;
        let mut out_off = 0usize;
        for i in 0..other_idx.len() {
            base_off += other_idx[i] * other_strides[i];
            out_off += other_idx[i] * out_strides[i];
        }
        for k in 0..diag_len {
            let (r, c) = if offset >= 0 {
                (k, (k as isize + offset) as usize)
            } else {
                ((k as isize - offset) as usize, k)
            };
            out[out_off + k * diag_out_stride] = arr_slice[base_off + r * s1 + c * s2];
        }
        if !other_idx.is_empty() {
            let mut i = other_idx.len() - 1;
            loop {
                other_idx[i] += 1;
                if other_idx[i] < other_dims[i] {
                    break;
                }
                other_idx[i] = 0;
                if i == 0 {
                    break;
                }
                i -= 1;
            }
        }
    }
    Ok(NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&out_shape), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}

/// 按索引数组从 choices 中选值（与 numpy.choose 一致）。
/// a 为索引数组（0..n-1），choices 为候选值数组列表。
/// 每个 choice 可以是标量（广播）或与 a 同形的数组（逐元素取值）。
/// choice 会沿 a 的形状广播。
#[pyfunction]
fn choose(a: &NdArray, choices: &Bound<'_, PyList>) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    let ndim = shape.len();
    let n_choices = choices.len();

    if n_choices == 0 {
        return Err(PyValueError::new_err("choose: need at least one choice"));
    }

    // 收集所有候选数组的扁平数据和形状
    let mut choice_flats: Vec<(Vec<f64>, Vec<usize>)> = Vec::with_capacity(n_choices);
    for item in choices.iter() {
        let nd = item.extract::<NdArray>()?;
        let flat: Vec<f64> = nd.data.iter().copied().collect();
        let shp = nd.data.shape().to_vec();
        choice_flats.push((flat, shp));
    }

    let total: usize = a.data.iter().count();
    let mut out = vec![0.0f64; total];

    let a_data = a
        .data
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("choose requires contiguous a"))?;

    for flat_idx in 0..total {
        let multi_idx = flat_to_multi(flat_idx, &shape);
        let idx = a_data[flat_idx] as usize;
        if idx >= n_choices {
            return Err(PyValueError::new_err(format!(
                "choose: index {} out of range [0, {})",
                idx, n_choices
            )));
        }

        let (ref choice_data, ref choice_shape) = choice_flats[idx];
        let choice_len = choice_data.len();

        let val = if choice_len == 1 {
            // 标量广播
            choice_data[0]
        } else if *choice_shape == shape {
            // 同形状，直接按多维索引取值
            let mut flat = 0usize;
            let mut stride = 1usize;
            for dim in (0..ndim).rev() {
                flat += multi_idx[dim] * stride;
                stride *= shape[dim];
            }
            choice_data[flat]
        } else {
            // 广播：沿 a 的每个维度广播 choice
            // 从 a 的最低维度开始，对应 choice 的最低维度
            let c_ndim = choice_shape.len();
            let mut flat = 0usize;
            let mut stride = 1usize;
            for a_dim in 0..ndim {
                let c_dim_idx = a_dim as isize - (ndim as isize - c_ndim as isize);
                let c_idx = if c_dim_idx < 0 {
                    // choice 没有这个维度，广播为 0
                    0
                } else {
                    let c_dim = c_dim_idx as usize;
                    let c_dim_size = choice_shape[c_dim];
                    if c_dim_size == 1 {
                        0
                    } else if a_dim < multi_idx.len() && multi_idx[a_dim] < c_dim_size {
                        multi_idx[a_dim]
                    } else {
                        c_dim_size - 1
                    }
                };
                flat += c_idx * stride;
                if (c_dim_idx as usize) < c_ndim {
                    stride *= choice_shape[c_dim_idx as usize];
                }
            }
            if flat < choice_data.len() {
                choice_data[flat]
            } else {
                0.0
            }
        };

        out[flat_idx] = val;
    }

    Ok(NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&shape), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}

/// 扁平索引转多维索引（C 顺序）。
fn flat_to_multi(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut multi = vec![0usize; ndim];
    let mut remaining = flat_idx;
    for dim in (0..ndim).rev() {
        multi[dim] = remaining % shape[dim];
        remaining /= shape[dim];
    }
    multi
}

/// 按布尔条件沿轴筛选元素（与 numpy.compress 一致）。
#[pyfunction]
#[pyo3(signature = (condition, a, axis=None))]
fn compress(condition: &NdArray, a: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    let cond_data: Vec<f64> = condition.data.iter().copied().collect();
    let cond_bool: Vec<bool> = cond_data.iter().map(|&v| v != 0.0).collect();

    match axis {
        None => {
            // 展平操作
            let flat: Vec<f64> = a.data.iter().copied().collect();
            let result: Vec<f64> = flat
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < cond_bool.len() && cond_bool[*i])
                .map(|(_, &v)| v)
                .collect();
            Ok(NdArray {
                imag: None,
                data: Array::from_shape_vec(IxDyn(&[result.len()]), result)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            })
        }
        Some(ax) => {
            let ndim = a.data.ndim();
            let ax = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };
            let shape = a.data.shape();
            let axis_len = shape[ax];

            // 提取沿 axis 的切片
            let mut kept_indices: Vec<usize> = Vec::new();
            for i in 0..axis_len {
                if i < cond_bool.len() && cond_bool[i] {
                    kept_indices.push(i);
                }
            }

            // 计算输出形状
            let mut out_shape: Vec<usize> = shape.to_vec();
            out_shape[ax] = kept_indices.len();

            // 使用手动索引方式沿 axis 提取
            let result = if kept_indices.is_empty() {
                // 空结果
                Array::from_shape_vec(IxDyn(&out_shape), Vec::new())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else {
                // 手动实现沿 axis 的 take 操作
                let strides = a.data.strides();
                let axis_stride = strides[ax] as usize;
                let dim_sz = shape[ax];
                let inner_size: usize = shape[ax + 1..].iter().product();
                let outer_size: usize = shape[..ax].iter().product::<usize>().max(1);
                let data_slice = a
                    .data
                    .as_slice()
                    .ok_or_else(|| PyValueError::new_err("compress requires contiguous array"))?;

                let new_inner: usize = out_shape[ax + 1..].iter().product();
                let new_axis_len = kept_indices.len();
                let mut out_data = vec![0.0f64; outer_size * new_axis_len * new_inner];

                for o in 0..outer_size {
                    for (new_k, &old_k) in kept_indices.iter().enumerate() {
                        for inn in 0..new_inner {
                            let src_idx = o * dim_sz * inner_size + old_k * axis_stride + inn;
                            let dst_idx = o * new_axis_len * new_inner + new_k * new_inner + inn;
                            out_data[dst_idx] = data_slice[src_idx];
                        }
                    }
                }
                Array::from_shape_vec(IxDyn(&out_shape), out_data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            };

            Ok(NdArray {
                imag: None,
                data: result,
            })
        }
    }
}

/// 将 src 的值复制到 dst（就地，与 numpy.copyto 一致）。
#[pyfunction]
fn copyto(dst: &mut NdArray, src: &NdArray) -> PyResult<()> {
    let src_shape = src.data.shape().to_vec();
    let dst_shape = dst.data.shape().to_vec();

    // 检查是否需要广播
    if src_shape == dst_shape {
        // 直接复制
        let dst_slice = dst
            .data
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("copyto requires contiguous dst"))?;
        let src_slice = src
            .data
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("copyto requires contiguous src"))?;
        dst_slice.copy_from_slice(src_slice);
    } else {
        // 广播复制
        let total: usize = dst.data.iter().count();
        let dst_slice = dst
            .data
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("copyto requires contiguous dst"))?;

        // 如果 src 是标量
        if src.data.len() == 1 {
            let val = *src.data.iter().next().unwrap();
            for v in dst_slice.iter_mut() {
                *v = val;
            }
        } else if src.data.len() == total {
            let src_slice = src
                .data
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("copyto requires contiguous src"))?;
            dst_slice.copy_from_slice(src_slice);
        } else {
            // 需要广播，使用 ndarray 的 broadcast 功能
            let src_broadcast = src.data.broadcast(IxDyn(&dst_shape)).ok_or_else(|| {
                PyValueError::new_err("copyto: cannot broadcast src to dst shape")
            })?;
            let src_slice: Vec<f64> = src_broadcast.iter().copied().collect();
            dst_slice.copy_from_slice(&src_slice);
        }
    }
    Ok(())
}

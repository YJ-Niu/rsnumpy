use crate::*;

#[pyfunction]
#[pyo3(signature = (a, axis=-1))]
fn sort(py: Python<'_>, a: &NdArray, axis: isize) -> PyResult<NdArray> {
    a.sort(py, axis)
}

#[pyfunction]
#[pyo3(signature = (a, axis=-1))]
fn argsort(py: Python<'_>, a: &NdArray, axis: isize) -> PyResult<NdArray> {
    a.argsort(py, axis)
}

#[pyfunction]
#[pyo3(signature = (a, kth, axis=-1))]
fn partition(a: &NdArray, kth: Vec<i64>, axis: isize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };

    if ndim == 0 {
        return Ok(NdArray {
            imag: None,
            data: a.data.clone(),
        });
    }

    let shape = a.data.shape().to_vec();
    let ax_len = shape[ax];

    let mut kths: Vec<usize> = kth
        .iter()
        .map(|&k| {
            let mut kk = k;
            if kk < 0 {
                kk += ax_len as i64;
            }
            if kk < 0 || kk >= ax_len as i64 {
                Err(PyValueError::new_err("kth out of bounds"))
            } else {
                Ok(kk as usize)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    kths.sort();
    kths.dedup();

    // 展平到 Vec 进行线性索引，避免 ndarray 多维下 usize 索引 panic
    let mut flat_data: Vec<f64> = a.data.iter().copied().collect();
    let outer_size: usize = shape[..ax].iter().product();
    let inner_size: usize = shape[ax + 1..].iter().product();
    let outer_strides: usize = inner_size * ax_len;
    let inner_strides = inner_size;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let start = outer * outer_strides + inner;
            let mut values: Vec<f64> = (0..ax_len)
                .map(|i| flat_data[start + i * inner_strides])
                .collect();
            for &k in &kths {
                let kth_value = introselect(&mut values, k);
                let mut left = Vec::new();
                let mut mid = Vec::new();
                let mut right = Vec::new();
                for &v in &values {
                    if v < kth_value {
                        left.push(v);
                    } else if v == kth_value {
                        mid.push(v);
                    } else {
                        right.push(v);
                    }
                }
                let mut merged = left;
                merged.extend(mid);
                merged.extend(right);
                values = merged;
            }
            for (i, v) in values.iter().enumerate() {
                flat_data[start + i * inner_strides] = *v;
            }
        }
    }
    let result = Array::from_shape_vec(IxDyn(&shape), flat_data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
#[pyo3(signature = (a, kth, axis=-1))]
fn argpartition(a: &NdArray, kth: Vec<i64>, axis: isize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };

    if ndim == 0 {
        return Ok(NdArray {
            imag: None,
            data: a.data.clone(),
        });
    }

    let shape = a.data.shape().to_vec();
    let ax_len = shape[ax];

    let mut kths: Vec<usize> = kth
        .iter()
        .map(|&k| {
            let mut kk = k;
            if kk < 0 {
                kk += ax_len as i64;
            }
            if kk < 0 || kk >= ax_len as i64 {
                Err(PyValueError::new_err("kth out of bounds"))
            } else {
                Ok(kk as usize)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    kths.sort();
    kths.dedup();

    let flat_data: Vec<f64> = a.data.iter().copied().collect();
    let mut indices: Vec<f64> = (0..flat_data.len()).map(|i| i as f64).collect();

    let outer_size: usize = shape[..ax].iter().product();
    let inner_size: usize = shape[ax + 1..].iter().product();
    let outer_strides: usize = inner_size * ax_len;
    let inner_strides = inner_size;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let start = outer * outer_strides + inner;
            let mut values: Vec<(f64, usize)> = (0..ax_len)
                .map(|i| {
                    (
                        flat_data[start + i * inner_strides],
                        i, // 局部索引
                    )
                })
                .collect();

            for &k in &kths {
                let mut low = 0;
                let mut high = ax_len - 1;
                while low < high {
                    let pivot_idx = (low + high) / 2;
                    values.swap(pivot_idx, high);
                    let pivot = values[high].0;
                    let mut i = low;
                    for j in low..high {
                        if values[j].0 < pivot {
                            values.swap(i, j);
                            i += 1;
                        }
                    }
                    values.swap(i, high);
                    if i == k {
                        break;
                    } else if i < k {
                        low = i + 1;
                    } else {
                        high = i - 1;
                    }
                }
            }

            for (i, &(_, local_idx)) in values.iter().enumerate() {
                indices[start + i * inner_strides] = local_idx as f64;
            }
        }
    }

    let arr = Array::from_shape_vec(IxDyn(&shape), indices)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
fn lexsort(keys: Vec<NdArray>) -> PyResult<NdArray> {
    if keys.is_empty() {
        return Ok(NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&[0]), Vec::new()).unwrap(),
        });
    }

    let n = keys[0].data.len();
    let mut indices: Vec<usize> = (0..n).collect();

    indices.sort_by(|&a, &b| {
        for key in keys.iter().rev() {
            let va = key.data[a];
            let vb = key.data[b];
            match va.partial_cmp(&vb) {
                Some(std::cmp::Ordering::Equal) => continue,
                Some(ord) => return ord,
                None => return std::cmp::Ordering::Equal,
            }
        }
        std::cmp::Ordering::Equal
    });

    let result: Vec<f64> = indices.into_iter().map(|i| i as f64).collect();
    let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
#[pyo3(signature = (a, axis=-1))]
fn sort_complex(a: &NdArray, axis: isize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };

    if ndim == 0 {
        return Ok(NdArray {
            imag: a.imag.clone(),
            data: a.data.clone(),
        });
    }

    let shape = a.data.shape().to_vec();
    let ax_len = shape[ax];
    let outer_size: usize = shape[..ax].iter().product();
    let inner_size: usize = shape[ax + 1..].iter().product();
    let outer_strides = inner_size * ax_len;
    let inner_strides = inner_size;

    let flat_data: Vec<f64> = a.data.iter().copied().collect();
    let flat_imag: Vec<f64> = match &a.imag {
        Some(im) => im.iter().copied().collect(),
        None => vec![0.0; flat_data.len()],
    };

    let mut result_re = flat_data.clone();
    let mut result_im = flat_imag.clone();

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let start = outer * outer_strides + inner;
            let mut values: Vec<(f64, f64, usize)> = (0..ax_len)
                .map(|i| {
                    (
                        flat_data[start + i * inner_strides],
                        flat_imag[start + i * inner_strides],
                        i,
                    )
                })
                .collect();

            values.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
                Some(std::cmp::Ordering::Equal) => {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                }
                Some(ord) => ord,
                None => std::cmp::Ordering::Equal,
            });

            for (i, (re, im, _)) in values.iter().enumerate() {
                result_re[start + i * inner_strides] = *re;
                result_im[start + i * inner_strides] = *im;
            }
        }
    }

    let arr_re = Array::from_shape_vec(IxDyn(&shape), result_re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let arr_im = Array::from_shape_vec(IxDyn(&shape), result_im)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: Some(arr_im),
        data: arr_re,
    })
}

fn introselect(arr: &mut [f64], k: usize) -> f64 {
    let n = arr.len();
    if n == 0 {
        return 0.0;
    }
    let mut left = 0;
    let mut right = n - 1;
    loop {
        if left == right {
            return arr[left];
        }
        if right - left < 5 {
            insertion_sort(arr, left, right);
            return arr[k];
        }
        let mid = (left + right) / 2;
        median_of_three(arr, left, mid, right);
        let p = hoare_partition(arr, left + 1, right - 1);
        if k <= p {
            right = p;
        } else {
            left = p + 1;
        }
    }
}

fn insertion_sort(arr: &mut [f64], left: usize, right: usize) {
    for i in (left + 1)..=right {
        let key = arr[i];
        let mut j = i;
        while j > left && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

fn median_of_three(arr: &mut [f64], a: usize, b: usize, c: usize) {
    if arr[a] > arr[b] {
        arr.swap(a, b);
    }
    if arr[a] > arr[c] {
        arr.swap(a, c);
    }
    if arr[b] > arr[c] {
        arr.swap(b, c);
    }
}

fn hoare_partition(arr: &mut [f64], left: usize, right: usize) -> usize {
    let pivot = arr[left];
    let mut i = left;
    let mut j = right + 1;
    loop {
        loop {
            i += 1;
            if i > right || arr[i] >= pivot {
                break;
            }
        }
        loop {
            if j == 0 || j <= left {
                break;
            }
            j -= 1;
            if arr[j] <= pivot {
                break;
            }
        }
        if i >= j {
            return j;
        }
        arr.swap(i, j);
    }
}

#[pyfunction]
#[pyo3(signature = (a, v, side="left"))]
fn searchsorted(a: &NdArray, v: f64, side: &str) -> PyResult<usize> {
    let values: Vec<f64> = a.data.iter().copied().collect();
    let pos = match side {
        "left" => values.partition_point(|&x| x < v),
        "right" => values.partition_point(|&x| x <= v),
        _ => return Err(PyValueError::new_err("side must be 'left' or 'right'")),
    };
    Ok(pos)
}

#[pyfunction]
#[pyo3(signature = (a, v, side="left", axis=None))]
fn searchsorted_axis(
    a: &NdArray,
    v: &Bound<'_, PyAny>,
    side: &str,
    axis: Option<isize>,
) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = match axis {
        None => ndim.saturating_sub(1),
        Some(ax) => {
            if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            }
        }
    };
    if ax >= ndim {
        return Err(PyValueError::new_err("axis out of bounds"));
    }

    let is_left = side == "left";
    if side != "left" && side != "right" {
        return Err(PyValueError::new_err("side must be 'left' or 'right'"));
    }

    let shape = a.data.shape().to_vec();
    let ax_len = shape[ax];
    let pre_size: usize = shape[..ax].iter().product();
    let post_size: usize = shape[ax + 1..].iter().product();
    let block_size = ax_len * post_size;
    let a_flat: Vec<f64> = a.data.iter().copied().collect();

    // 尝试提取标量 v
    if let Ok(val) = v.extract::<f64>() {
        let mut result = vec![0.0f64; pre_size * post_size];
        for outer in 0..pre_size {
            for inner in 0..post_size {
                let base = outer * block_size + inner;
                let mut pos = 0usize;
                for i in 0..ax_len {
                    let av = a_flat[base + i * post_size];
                    if (is_left && av >= val) || (!is_left && av > val) {
                        break;
                    }
                    pos = i + 1;
                }
                result[outer * post_size + inner] = pos as f64;
            }
        }
        let mut out_shape = shape.clone();
        out_shape.remove(ax);
        let arr = Array::from_shape_vec(IxDyn(&out_shape), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            imag: None,
            data: arr,
        });
    }

    // 尝试提取 NdArray v
    let v_arr = v.extract::<NdArray>()?;
    let v_flat: Vec<f64> = v_arr.data.iter().copied().collect();
    let v_len = v_flat.len();

    let mut result = vec![0.0f64; pre_size * post_size * v_len];
    for outer in 0..pre_size {
        for inner in 0..post_size {
            let base = outer * block_size + inner;
            for (vi, &val) in v_flat.iter().enumerate() {
                let mut pos = 0usize;
                for i in 0..ax_len {
                    let av = a_flat[base + i * post_size];
                    if (is_left && av >= val) || (!is_left && av > val) {
                        break;
                    }
                    pos = i + 1;
                }
                let out_idx = (outer * post_size + inner) * v_len + vi;
                result[out_idx] = pos as f64;
            }
        }
    }

    let mut out_shape = shape.clone();
    out_shape[ax] = v_len;
    let arr = Array::from_shape_vec(IxDyn(&out_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sort, m)?)?;
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    m.add_function(wrap_pyfunction!(partition, m)?)?;
    m.add_function(wrap_pyfunction!(argpartition, m)?)?;
    m.add_function(wrap_pyfunction!(lexsort, m)?)?;
    m.add_function(wrap_pyfunction!(sort_complex, m)?)?;
    m.add_function(wrap_pyfunction!(searchsorted, m)?)?;
    m.add_function(wrap_pyfunction!(searchsorted_axis, m)?)?;
    Ok(())
}

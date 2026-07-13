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
            imag: None, data: a.data.clone(),
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

    let mut result: Array<f64, IxDyn> = a.data.clone();
    let outer_size: usize = shape[..ax].iter().product();
    let inner_size: usize = shape[ax + 1..].iter().product();
    let outer_strides: usize = inner_size * ax_len;
    let inner_strides = inner_size;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let start = outer * outer_strides + inner;
            let mut values: Vec<f64> = (0..ax_len)
                .map(|i| result[start + i * inner_strides])
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
                result[start + i * inner_strides] = *v;
            }
        }
    }
    Ok(NdArray { imag: None, data: result })
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
            imag: None, data: a.data.clone(),
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
                        start + i * inner_strides,
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

            for (i, &(_, orig_idx)) in values.iter().enumerate() {
                indices[start + i * inner_strides] = orig_idx as f64;
            }
        }
    }

    let arr = Array::from_shape_vec(IxDyn(&shape), indices)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { imag: None, data: arr })
}

#[pyfunction]
fn lexsort(keys: Vec<NdArray>) -> PyResult<NdArray> {
    if keys.is_empty() {
        return Ok(NdArray {
            imag: None, data: Array::from_shape_vec(IxDyn(&[0]), Vec::new()).unwrap(),
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
    Ok(NdArray { imag: None, data: arr })
}

#[pyfunction]
fn sort_complex(a: &NdArray) -> PyResult<NdArray> {
    let flat_data: Vec<f64> = a.data.iter().copied().collect();
    let n = flat_data.len();

    if n == 0 {
        return Ok(NdArray {
            imag: None, data: Array::from_shape_vec(IxDyn(&[0]), Vec::new()).unwrap(),
        });
    }

    let mut indexed: Vec<(f64, f64, usize)> = flat_data
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, 0.0, i))
        .collect();

    indexed.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
        Some(std::cmp::Ordering::Equal) => {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        }
        Some(ord) => ord,
        None => std::cmp::Ordering::Equal,
    });

    let mut result = Vec::with_capacity(n);
    for (val, _, _) in indexed {
        result.push(val);
    }

    let shape = a.data.shape().to_vec();
    let arr = Array::from_shape_vec(IxDyn(&shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { imag: None, data: arr })
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sort, m)?)?;
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    m.add_function(wrap_pyfunction!(partition, m)?)?;
    m.add_function(wrap_pyfunction!(argpartition, m)?)?;
    m.add_function(wrap_pyfunction!(lexsort, m)?)?;
    m.add_function(wrap_pyfunction!(sort_complex, m)?)?;
    m.add_function(wrap_pyfunction!(searchsorted, m)?)?;
    Ok(())
}

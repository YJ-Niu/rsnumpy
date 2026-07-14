use crate::*;

#[pyfunction]
fn unique(a: &NdArray) -> PyResult<NdArray> {
    let mut values: Vec<u64> = a.data.iter().map(|v| v.to_bits()).collect();
    values.sort();
    values.dedup();
    let unique_vals: Vec<f64> = values.into_iter().map(f64::from_bits).collect();
    let arr = Array::from_shape_vec(IxDyn(&[unique_vals.len()]), unique_vals)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
#[pyo3(signature = (a, return_index=false, return_inverse=false, return_counts=false))]
fn unique_full(
    a: &NdArray,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
) -> PyResult<Vec<NdArray>> {
    let flat_data: Vec<f64> = a.data.iter().copied().collect();
    let n = flat_data.len();

    let mut indexed: Vec<(f64, usize)> =
        flat_data.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut unique_vals = Vec::new();
    let mut first_indices = Vec::new();
    let mut counts = Vec::new();

    let mut prev_val = None;
    let mut count = 0;
    let mut first_idx = 0;

    for (val, orig_idx) in indexed.iter() {
        match prev_val {
            None => {
                prev_val = Some(*val);
                first_idx = *orig_idx;
                count = 1;
            }
            Some(pv) => {
                if (val - pv).abs() < 1e-15 {
                    count += 1;
                } else {
                    unique_vals.push(pv);
                    first_indices.push(first_idx as f64);
                    counts.push(count as f64);
                    prev_val = Some(*val);
                    first_idx = *orig_idx;
                    count = 1;
                }
            }
        }
    }

    if let Some(pv) = prev_val {
        unique_vals.push(pv);
        first_indices.push(first_idx as f64);
        counts.push(count as f64);
    }

    let mut inverse = Vec::new();
    if return_inverse {
        inverse = Vec::with_capacity(n);
        for v in &flat_data {
            let mut found = false;
            for (i, uv) in unique_vals.iter().enumerate() {
                if (v - uv).abs() < 1e-15 {
                    inverse.push(i as f64);
                    found = true;
                    break;
                }
            }
            if !found {
                inverse.push(0.0);
            }
        }
    }

    let unique_arr = Array::from_shape_vec(IxDyn(&[unique_vals.len()]), unique_vals)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut results = vec![NdArray {
        imag: None,
        data: unique_arr,
    }];

    if return_index {
        let idx_arr = Array::from_shape_vec(IxDyn(&[first_indices.len()]), first_indices)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        results.push(NdArray {
            imag: None,
            data: idx_arr,
        });
    }

    if return_inverse {
        let inv_arr = Array::from_shape_vec(IxDyn(&[inverse.len()]), inverse)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        results.push(NdArray {
            imag: None,
            data: inv_arr,
        });
    }

    if return_counts {
        let cnt_arr = Array::from_shape_vec(IxDyn(&[counts.len()]), counts)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        results.push(NdArray {
            imag: None,
            data: cnt_arr,
        });
    }

    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (a, axis, return_index=false, return_inverse=false, return_counts=false, sorted=true))]
fn unique_axis(
    a: &NdArray,
    axis: isize,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
    sorted: bool,
) -> PyResult<Vec<NdArray>> {
    let ndim = a.data.ndim();
    if ndim == 0 {
        return Err(PyValueError::new_err(
            "axis argument to unique is not supported for 0-d arrays",
        ));
    }
    let ax = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis, ndim
        )));
    }

    // 将目标轴移到最前，其余轴保持相对顺序
    let mut perm: Vec<usize> = Vec::with_capacity(ndim);
    perm.push(ax);
    for i in 0..ndim {
        if i != ax {
            perm.push(i);
        }
    }

    let moved = a.data.view().permuted_axes(perm.clone()).to_owned();
    let n = moved.shape()[0];
    let rest: Vec<usize> = moved.shape()[1..].to_vec();
    let block: usize = rest.iter().product::<usize>().max(1);
    // 按 C 顺序展平（iter 始终按逻辑顺序遍历，与内存布局无关）
    let flat: Vec<f64> = moved.iter().copied().collect();

    let row = |i: usize| -> &[f64] { &flat[i * block..(i + 1) * block] };
    // 逐元素精确比较；NaN 与任何值（含 NaN）都不相等
    let rows_equal = |i: usize, j: usize| -> bool {
        let (ri, rj) = (row(i), row(j));
        for k in 0..block {
            if ri[k].is_nan() || rj[k].is_nan() || ri[k] != rj[k] {
                return false;
            }
        }
        true
    };

    // 稳定字典序排序（NaN 视为最大），保证同组内保留原始先后顺序
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        let (ri, rj) = (row(i), row(j));
        for k in 0..block {
            let (av, bv) = (ri[k], rj[k]);
            let o = match (av.is_nan(), bv.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal),
            };
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    });

    // 相邻去重：每组代表取组内最小原始下标；同时记录每个原始行的组号
    let mut uniq_first: Vec<usize> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut inverse_by_orig: Vec<usize> = vec![0usize; n];
    let mut cur_group: usize = 0;
    for pos in 0..n {
        let oi = order[pos];
        if pos == 0 || !rows_equal(oi, order[pos - 1]) {
            if pos != 0 {
                cur_group += 1;
            }
            uniq_first.push(oi);
            counts.push(1);
        } else {
            *counts.last_mut().unwrap() += 1;
            if oi < *uniq_first.last().unwrap() {
                *uniq_first.last_mut().unwrap() = oi;
            }
        }
        inverse_by_orig[oi] = cur_group;
    }

    let m = uniq_first.len();
    // sorted=false 时按首次出现顺序（代表下标升序）重排各组
    let mut group_order: Vec<usize> = (0..m).collect();
    if !sorted {
        group_order.sort_by_key(|&g| uniq_first[g]);
    }
    let mut remap: Vec<usize> = vec![0usize; m];
    for (new_g, &old_g) in group_order.iter().enumerate() {
        remap[old_g] = new_g;
    }

    let mut out_flat: Vec<f64> = Vec::with_capacity(m * block);
    let mut index_out: Vec<f64> = Vec::with_capacity(m);
    let mut counts_out: Vec<f64> = Vec::with_capacity(m);
    for &g in &group_order {
        out_flat.extend_from_slice(row(uniq_first[g]));
        index_out.push(uniq_first[g] as f64);
        counts_out.push(counts[g] as f64);
    }

    let mut moved_shape: Vec<usize> = Vec::with_capacity(ndim);
    moved_shape.push(m);
    moved_shape.extend_from_slice(&rest);
    let unique_moved = Array::from_shape_vec(IxDyn(&moved_shape), out_flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // 将轴顺序还原到原始排列
    let inv_perm: Vec<usize> = (0..ndim)
        .map(|i| perm.iter().position(|&x| x == i).unwrap())
        .collect();
    let unique_arr = unique_moved
        .view()
        .permuted_axes(inv_perm)
        .as_standard_layout()
        .to_owned();

    let mut results = vec![NdArray {
        imag: None,
        data: unique_arr,
    }];

    if return_index {
        let arr = Array::from_shape_vec(IxDyn(&[index_out.len()]), index_out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        results.push(NdArray {
            imag: None,
            data: arr,
        });
    }
    if return_inverse {
        let inv: Vec<f64> = inverse_by_orig.iter().map(|&g| remap[g] as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&[inv.len()]), inv)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        results.push(NdArray {
            imag: None,
            data: arr,
        });
    }
    if return_counts {
        let arr = Array::from_shape_vec(IxDyn(&[counts_out.len()]), counts_out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        results.push(NdArray {
            imag: None,
            data: arr,
        });
    }

    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (element, test, invert=false))]
fn isin(element: &NdArray, test: &NdArray, invert: bool) -> NdArray {
    let mut keys: Vec<f64> = test
        .data
        .iter()
        .filter(|v| !v.is_nan())
        .map(|&v| if v == 0.0 { 0.0 } else { v })
        .collect();
    keys.sort_by(|a, b| a.total_cmp(b));
    keys.dedup();
    // 小 test 集：线性 `==` 扫描（缓存友好，避免 total_cmp 位运算），与 numpy 暴力路径一致；
    // 大集用排序 + 二分查找 O(n log m)。
    let small = keys.len() <= 16;
    let data = element.data.mapv(|v| {
        let present = if v.is_nan() {
            false
        } else {
            let nv = if v == 0.0 { 0.0 } else { v };
            if small {
                keys.contains(&nv)
            } else {
                keys.binary_search_by(|p| p.total_cmp(&nv)).is_ok()
            }
        };
        let res = if invert { !present } else { present };
        if res { 1.0 } else { 0.0 }
    });
    NdArray { data, imag: None }
}

// 升序排序 + 去重（-0.0 归一为 +0.0，NaN 视为相等只保留其一）。
fn sorted_unique_vec(vals: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = vals
        .iter()
        .map(|&x| if x == 0.0 { 0.0 } else { x })
        .collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v.dedup_by(|a, b| a == b || (a.is_nan() && b.is_nan()));
    v
}

fn vec_to_1d(vals: Vec<f64>) -> NdArray {
    let n = vals.len();
    NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&[n]), vals).unwrap(),
    }
}

#[pyfunction]
fn intersect1d(ar1: &NdArray, ar2: &NdArray) -> NdArray {
    let a = sorted_unique_vec(&ar1.data.iter().copied().collect::<Vec<f64>>());
    let b = sorted_unique_vec(&ar2.data.iter().copied().collect::<Vec<f64>>());
    let (mut i, mut j) = (0usize, 0usize);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].total_cmp(&b[j]) {
            ::std::cmp::Ordering::Less => i += 1,
            ::std::cmp::Ordering::Greater => j += 1,
            ::std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    vec_to_1d(out)
}

#[pyfunction]
fn union1d(ar1: &NdArray, ar2: &NdArray) -> NdArray {
    let mut all: Vec<f64> = ar1.data.iter().copied().collect();
    all.extend(ar2.data.iter().copied());
    vec_to_1d(sorted_unique_vec(&all))
}

#[pyfunction]
fn setdiff1d(ar1: &NdArray, ar2: &NdArray) -> NdArray {
    let a = sorted_unique_vec(&ar1.data.iter().copied().collect::<Vec<f64>>());
    let b = sorted_unique_vec(&ar2.data.iter().copied().collect::<Vec<f64>>());
    let (mut i, mut j) = (0usize, 0usize);
    let mut out = Vec::new();
    while i < a.len() {
        if j >= b.len() {
            out.push(a[i]);
            i += 1;
            continue;
        }
        match a[i].total_cmp(&b[j]) {
            ::std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            ::std::cmp::Ordering::Greater => j += 1,
            ::std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    vec_to_1d(out)
}

#[pyfunction]
fn setxor1d(ar1: &NdArray, ar2: &NdArray) -> NdArray {
    let a = sorted_unique_vec(&ar1.data.iter().copied().collect::<Vec<f64>>());
    let b = sorted_unique_vec(&ar2.data.iter().copied().collect::<Vec<f64>>());
    let (mut i, mut j) = (0usize, 0usize);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].total_cmp(&b[j]) {
            ::std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            ::std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            ::std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    vec_to_1d(out)
}

// unique 的一次遍历实现：返回 [唯一值, 首次出现索引, 逆索引, 计数]。
// uniq 已升序，逆索引用二分查找 O(n log u)，避免 unique_full 中 O(n*u) 的线性扫描。
#[pyfunction]
fn unique_all_rs(x: &NdArray) -> Vec<NdArray> {
    let data: Vec<f64> = x
        .data
        .iter()
        .map(|&v| if v == 0.0 { 0.0 } else { v })
        .collect();
    let uniq = sorted_unique_vec(&data);
    let mut counts = vec![0.0f64; uniq.len()];
    let mut first = vec![-1.0f64; uniq.len()];
    let mut inverse = Vec::with_capacity(data.len());
    for (idx, &v) in data.iter().enumerate() {
        let i = uniq.partition_point(|p| p.total_cmp(&v) == ::std::cmp::Ordering::Less);
        inverse.push(i as f64);
        counts[i] += 1.0;
        if first[i] < 0.0 {
            first[i] = idx as f64;
        }
    }
    vec![
        vec_to_1d(uniq),
        vec_to_1d(first),
        vec_to_1d(inverse),
        vec_to_1d(counts),
    ]
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unique, m)?)?;
    m.add_function(wrap_pyfunction!(unique_full, m)?)?;
    m.add_function(wrap_pyfunction!(unique_axis, m)?)?;
    m.add_function(wrap_pyfunction!(unique_all_rs, m)?)?;
    m.add_function(wrap_pyfunction!(intersect1d, m)?)?;
    m.add_function(wrap_pyfunction!(union1d, m)?)?;
    m.add_function(wrap_pyfunction!(setdiff1d, m)?)?;
    m.add_function(wrap_pyfunction!(setxor1d, m)?)?;
    m.add_function(wrap_pyfunction!(isin, m)?)?;
    Ok(())
}

use crate::*;

// ========== 多项式求值 ==========
#[pyfunction]
#[pyo3(signature = (coef, x))]
fn polyval_rs(coef: &NdArray, x: &NdArray) -> PyResult<NdArray> {
    let c: Vec<f64> = coef.data.iter().copied().collect();
    let x_shape = x.data.shape().to_vec();
    let x_vals: Vec<f64> = x.data.iter().copied().collect();
    let deg = c.len() - 1;

    let result: Vec<f64> = x_vals
        .iter()
        .map(|&xv| {
            let mut val = 0.0;
            for (i, &ci) in c.iter().enumerate() {
                val += ci * xv.powi((deg - i) as i32);
            }
            val
        })
        .collect();

    let arr = Array::from_shape_vec(IxDyn(&x_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== 多项式求导 ==========
#[pyfunction]
#[pyo3(signature = (coef, m=1))]
fn polyder_rs(coef: &NdArray, m: usize) -> PyResult<NdArray> {
    let mut c: Vec<f64> = coef.data.iter().copied().collect();
    for _ in 0..m {
        let deg = c.len() - 1;
        let mut new_c = Vec::with_capacity(c.len() - 1);
        for (i, ci) in c.iter().enumerate().take(c.len() - 1) {
            new_c.push(ci * (deg - i) as f64);
        }
        if new_c.is_empty() {
            new_c.push(0.0);
        }
        c = new_c;
    }
    let arr = Array::from_shape_vec(IxDyn(&[c.len()]), c)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== 多项式积分 ==========
#[pyfunction]
#[pyo3(signature = (coef, m=1, k=0.0))]
fn polyint_rs(coef: &NdArray, m: usize, k: f64) -> PyResult<NdArray> {
    let mut c: Vec<f64> = coef.data.iter().copied().collect();
    for _ in 0..m {
        let deg = c.len() - 1;
        let mut new_c = Vec::with_capacity(c.len() + 1);
        for (i, ci) in c.iter().enumerate() {
            new_c.push(ci / (deg - i + 1) as f64);
        }
        new_c.push(k);
        c = new_c;
    }
    let arr = Array::from_shape_vec(IxDyn(&[c.len()]), c)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== QR 迭代求特征值（辅助函数） ==========
fn qr_iter_eigvals_rs(companion: Vec<Vec<f64>>) -> Vec<f64> {
    let n = companion.len();
    let mut ak = companion.clone();

    for _ in 0..200 {
        let mut q = vec![vec![0.0; n]; n];
        let mut r = vec![vec![0.0; n]; n];

        for j in 0..n {
            let mut v: Vec<f64> = (0..n).map(|i| ak[i][j]).collect();
            for i in 0..j {
                let s: f64 = (0..n).map(|k| q[k][i] * v[k]).sum();
                r[i][j] = s;
                for k in 0..n {
                    v[k] -= s * q[k][i];
                }
            }
            let norm: f64 = v.iter().map(|vv| vv * vv).sum::<f64>().sqrt();
            r[j][j] = norm;
            if norm > 1e-15 {
                for k in 0..n {
                    q[k][j] = v[k] / norm;
                }
            }
        }

        let mut nxt = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += r[i][k] * q[k][j];
                }
                nxt[i][j] = s;
            }
        }
        ak = nxt;
    }
    (0..n).map(|i| ak[i][i]).collect()
}

// ========== 多项式求根 ==========
#[pyfunction]
fn polyroots_rs(coef: &NdArray) -> PyResult<NdArray> {
    let c: Vec<f64> = coef.data.iter().copied().collect();
    let n = c.len() - 1;
    if n == 0 {
        let arr = Array::from_shape_vec(IxDyn(&[0]), vec![])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            imag: None,
            data: arr,
        });
    }
    if n == 1 {
        let root = -c[1] / c[0];
        let arr = Array::from_shape_vec(IxDyn(&[1]), vec![root])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            imag: None,
            data: arr,
        });
    }
    if n == 2 {
        let disc = c[1] * c[1] - 4.0 * c[0] * c[2];
        if disc >= 0.0 {
            let sq = disc.sqrt();
            let r1 = (-c[1] + sq) / (2.0 * c[0]);
            let r2 = (-c[1] - sq) / (2.0 * c[0]);
            let arr = Array::from_shape_vec(IxDyn(&[2]), vec![r1, r2])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(NdArray {
                imag: None,
                data: arr,
            });
        } else {
            let sq = (-disc).sqrt();
            let r1 = -c[1] / (2.0 * c[0]);
            let r2 = sq / (2.0 * c[0]);
            let arr = Array::from_shape_vec(IxDyn(&[2]), vec![r1, r2])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(NdArray {
                imag: None,
                data: arr,
            });
        }
    }

    // 构造伴随矩阵
    let a_n = c[0];
    let mut comp = vec![vec![0.0; n]; n];
    for i in 0..n - 1 {
        comp[i + 1][i] = 1.0;
    }
    for i in 0..n {
        comp[i][n - 1] = -c[n - i] / a_n;
    }

    let roots = qr_iter_eigvals_rs(comp);
    let arr = Array::from_shape_vec(IxDyn(&[roots.len()]), roots)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== 多项式拟合 ==========
#[pyfunction]
#[pyo3(signature = (x, y, deg))]
fn polyfit_rs(x: &NdArray, y: &NdArray, deg: usize) -> PyResult<NdArray> {
    let x_vals: Vec<f64> = x.data.iter().copied().collect();
    let y_vals: Vec<f64> = y.data.iter().copied().collect();
    let n = x_vals.len();
    let m = deg + 1;

    let mut vander = vec![vec![0.0; m]; n];
    for (i, xv) in x_vals.iter().enumerate() {
        for (j, vand_row) in vander[i].iter_mut().enumerate() {
            *vand_row = xv.powi((deg - j) as i32);
        }
    }

    let mut ata = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for vand_row in vander.iter() {
                s += vand_row[i] * vand_row[j];
            }
            ata[i][j] = s;
        }
    }

    let mut aty = vec![0.0; m];
    for i in 0..m {
        let mut s = 0.0;
        for (vand_row, yv) in vander.iter().zip(y_vals.iter()) {
            s += vand_row[i] * yv;
        }
        aty[i] = s;
    }

    // 高斯消元
    let mut aug = vec![vec![0.0; m + 1]; m];
    for i in 0..m {
        for j in 0..m {
            aug[i][j] = ata[i][j];
        }
        aug[i][m] = aty[i];
    }

    for col in 0..m {
        let mut max_row = col;
        for row in col + 1..m {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            continue;
        }

        for row in 0..m {
            if row != col && aug[row][col] != 0.0 {
                let factor = aug[row][col] / pivot;
                #[allow(clippy::needless_range_loop)]
                for jj in col..m + 1 {
                    aug[row][jj] -= factor * aug[col][jj];
                }
            }
        }
    }

    let mut coef = vec![0.0; m];
    for i in (0..m).rev() {
        let mut val = aug[i][m];
        for j in i + 1..m {
            val -= aug[i][j] * coef[j];
        }
        coef[i] = if aug[i][i].abs() > 1e-15 {
            val / aug[i][i]
        } else {
            0.0
        };
    }

    let arr = Array::from_shape_vec(IxDyn(&[m]), coef)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== 多项式算术运算 ==========
#[pyfunction]
fn polyadd(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let ca: Vec<f64> = a.data.iter().copied().collect();
    let cb: Vec<f64> = b.data.iter().copied().collect();
    let n = std::cmp::max(ca.len(), cb.len());
    let mut out = vec![0.0f64; n];
    for i in 0..ca.len() {
        out[n - ca.len() + i] += ca[i];
    }
    for i in 0..cb.len() {
        out[n - cb.len() + i] += cb[i];
    }
    let arr = Array::from_shape_vec(IxDyn(&[out.len()]), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
fn polysub(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let ca: Vec<f64> = a.data.iter().copied().collect();
    let cb: Vec<f64> = b.data.iter().copied().collect();
    let n = std::cmp::max(ca.len(), cb.len());
    let mut out = vec![0.0f64; n];
    for i in 0..ca.len() {
        out[n - ca.len() + i] += ca[i];
    }
    for i in 0..cb.len() {
        out[n - cb.len() + i] -= cb[i];
    }
    let arr = Array::from_shape_vec(IxDyn(&[out.len()]), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
fn polymul(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let ca: Vec<f64> = a.data.iter().copied().collect();
    let cb: Vec<f64> = b.data.iter().copied().collect();
    let n = ca.len() + cb.len() - 1;
    let mut out = vec![0.0f64; n];
    for i in 0..ca.len() {
        for j in 0..cb.len() {
            out[i + j] += ca[i] * cb[j];
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[out.len()]), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== 多项式除法 ==========
/// 多项式除法，返回 (商, 余数)
#[pyfunction]
fn polydiv(u: &NdArray, v: &NdArray) -> PyResult<(NdArray, NdArray)> {
    let u_vals: Vec<f64> = u.data.iter().copied().collect();
    let v_vals: Vec<f64> = v.data.iter().copied().collect();
    let m = u_vals.len();
    let n = v_vals.len();
    if n == 0 || v_vals[0].abs() < 1e-300 {
        return Err(PyValueError::new_err("除数多项式不能为零或首项系数为零"));
    }
    let scale = 1.0 / v_vals[0];
    let mut r = u_vals.clone();
    let (q, rem) = if m >= n {
        let mut q = vec![0.0; m - n + 1];
        for k in 0..(m - n + 1) {
            let d = r[k] * scale;
            q[k] = d;
            for j in 0..n {
                r[k + j] -= d * v_vals[j];
            }
        }
        let rem = r[m - n + 1..].to_vec();
        (q, rem)
    } else {
        (vec![0.0], r.clone())
    };
    // 裁剪余数前导零
    let mut trimmed_rem = rem;
    while trimmed_rem.len() > 1 && trimmed_rem[0].abs() < 1e-14 {
        trimmed_rem.remove(0);
    }
    let q_arr = Array::from_shape_vec(IxDyn(&[q.len()]), q)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rem_arr = Array::from_shape_vec(IxDyn(&[trimmed_rem.len()]), trimmed_rem)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        NdArray {
            imag: None,
            data: q_arr,
        },
        NdArray {
            imag: None,
            data: rem_arr,
        },
    ))
}

// ========== 由根序列构造多项式 ==========
/// 由根序列返回多项式系数
#[pyfunction]
fn poly_from_roots(roots: &NdArray) -> NdArray {
    let roots_vals: Vec<f64> = roots.data.iter().copied().collect();
    let mut coeffs = vec![1.0_f64];
    for &rt in &roots_vals {
        let mut new_coeffs = vec![0.0; coeffs.len() + 1];
        for (i, &c) in coeffs.iter().enumerate() {
            new_coeffs[i] += c;
            new_coeffs[i + 1] -= c * rt;
        }
        coeffs = new_coeffs;
    }
    let arr = Array::from_shape_vec(IxDyn(&[coeffs.len()]), coeffs)
        .expect("由根序列构造系数数组形状必然合法");
    NdArray {
        imag: None,
        data: arr,
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(polyval_rs, m)?)?;
    m.add_function(wrap_pyfunction!(polyder_rs, m)?)?;
    m.add_function(wrap_pyfunction!(polyint_rs, m)?)?;
    m.add_function(wrap_pyfunction!(polyroots_rs, m)?)?;
    m.add_function(wrap_pyfunction!(polyfit_rs, m)?)?;
    m.add_function(wrap_pyfunction!(polyadd, m)?)?;
    m.add_function(wrap_pyfunction!(polysub, m)?)?;
    m.add_function(wrap_pyfunction!(polymul, m)?)?;
    m.add_function(wrap_pyfunction!(polydiv, m)?)?;
    m.add_function(wrap_pyfunction!(poly_from_roots, m)?)?;
    Ok(())
}

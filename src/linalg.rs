use ndarray::{Array, IxDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::NdArray;

/// 点积
#[pyfunction]
fn dot(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let a_data = &a.data;
    let b_data = &b.data;
    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();

    if a_shape.len() == 1 && b_shape.len() == 1 {
        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for dot product: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let result: f64 = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), result),
        });
    }

    if a_shape.len() == 2 && b_shape.len() == 2 {
        if a_shape[1] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for dot product: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let m = a_shape[0];
        let n = a_shape[1];
        let p = b_shape[1];
        let mut result = vec![0.0_f64; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a_data[[i, k]] * b_data[[k, j]];
                }
                result[i * p + j] = sum;
            }
        }
        let arr = Array::from_shape_vec((m, p), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            data: arr.into_dyn(),
        });
    }

    if a_shape.len() == 1 && b_shape.len() == 2 {
        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for dot product: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let n = a_shape[0];
        let p = b_shape[1];
        let mut result = vec![0.0_f64; p];
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a_data[k] * b_data[[k, j]];
            }
            result[j] = sum;
        }
        let arr = Array::from_shape_vec(IxDyn(&[p]), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { data: arr });
    }

    if a_shape.len() == 2 && b_shape.len() == 1 {
        if a_shape[1] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for dot product: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let m = a_shape[0];
        let n = a_shape[1];
        let mut result = vec![0.0_f64; m];
        for i in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a_data[[i, k]] * b_data[k];
            }
            result[i] = sum;
        }
        let arr = Array::from_shape_vec(IxDyn(&[m]), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { data: arr });
    }

    Err(PyValueError::new_err(format!(
        "Unsupported shapes for dot product: {:?} and {:?}",
        a_shape, b_shape
    )))
}

/// 向量点积（展开为一维）
#[pyfunction]
fn vdot(a: &NdArray, b: &NdArray) -> PyResult<f64> {
    let a_flat: Vec<f64> = a.data.iter().copied().collect();
    let b_flat: Vec<f64> = b.data.iter().copied().collect();
    if a_flat.len() != b_flat.len() {
        return Err(PyValueError::new_err(format!(
            "vdot requires same number of elements: {} vs {}",
            a_flat.len(),
            b_flat.len()
        )));
    }
    let result: f64 = a_flat.iter().zip(b_flat.iter()).map(|(x, y)| x * y).sum();
    Ok(result)
}

/// 内积
#[pyfunction]
fn inner(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let a_data = &a.data;
    let b_data = &b.data;
    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();

    if a_shape.len() == 1 && b_shape.len() == 1 {
        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for inner product: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let result: f64 = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), result),
        });
    }

    if a_shape.len() == 2 && b_shape.len() == 2 {
        if a_shape[1] != b_shape[1] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for inner product: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let m = a_shape[0];
        let n = a_shape[1];
        let p = b_shape[0];
        let mut result = vec![0.0_f64; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a_data[[i, k]] * b_data[[j, k]];
                }
                result[i * p + j] = sum;
            }
        }
        let arr = Array::from_shape_vec((m, p), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            data: arr.into_dyn(),
        });
    }

    Err(PyValueError::new_err(format!(
        "Unsupported shapes for inner product: {:?} and {:?}",
        a_shape, b_shape
    )))
}

/// 矩阵乘法
#[pyfunction]
fn matmul(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let a_data = &a.data;
    let b_data = &b.data;
    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();

    if a_shape.len() == 1 && b_shape.len() == 1 {
        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for matmul: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let result: f64 = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), result),
        });
    }

    if a_shape.len() == 2 && b_shape.len() == 1 {
        if a_shape[1] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for matmul: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let m = a_shape[0];
        let n = a_shape[1];
        let mut result = vec![0.0_f64; m];
        for i in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a_data[[i, k]] * b_data[k];
            }
            result[i] = sum;
        }
        let arr = Array::from_shape_vec(IxDyn(&[m]), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { data: arr });
    }

    if a_shape.len() == 1 && b_shape.len() == 2 {
        if a_shape[0] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for matmul: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let n = a_shape[0];
        let p = b_shape[1];
        let mut result = vec![0.0_f64; p];
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a_data[k] * b_data[[k, j]];
            }
            result[j] = sum;
        }
        let arr = Array::from_shape_vec(IxDyn(&[p]), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { data: arr });
    }

    if a_shape.len() == 2 && b_shape.len() == 2 {
        if a_shape[1] != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for matmul: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let m = a_shape[0];
        let n = a_shape[1];
        let p = b_shape[1];
        let mut result = vec![0.0_f64; m * p];
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a_data[[i, k]] * b_data[[k, j]];
                }
                result[i * p + j] = sum;
            }
        }
        let arr = Array::from_shape_vec((m, p), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            data: arr.into_dyn(),
        });
    }

    if a_shape.len() >= 3 && b_shape.len() == 2 {
        let batch_dims: Vec<usize> = a_shape[..a_shape.len() - 2].to_vec();
        let m = a_shape[a_shape.len() - 2];
        let n = a_shape[a_shape.len() - 1];
        let p = b_shape[1];
        if n != b_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for matmul: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        let batch_size: usize = batch_dims.iter().product();
        let mut result = vec![0.0_f64; batch_size * m * p];
        for b_idx in 0..batch_size {
            for i in 0..m {
                for j in 0..p {
                    let mut sum = 0.0;
                    for k in 0..n {
                        let a_val = if a_shape.len() == 3 {
                            a_data[[b_idx, i, k]]
                        } else {
                            a_data[[0, b_idx, i, k]]
                        };
                        let b_val = b_data[[k, j]];
                        sum += a_val * b_val;
                    }
                    result[b_idx * m * p + i * p + j] = sum;
                }
            }
        }
        let mut result_shape = batch_dims.clone();
        result_shape.extend_from_slice(&[m, p]);
        let arr = Array::from_shape_vec(result_shape, result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            data: arr.into_dyn(),
        });
    }

    Err(PyValueError::new_err(format!(
        "Unsupported shapes for matmul: {:?} and {:?}",
        a_shape, b_shape
    )))
}

/// 矩阵求逆 (2x2, 3x3)
#[pyfunction]
fn inv(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("inv requires a square matrix"));
    }
    let n = shape[0];
    match n {
        2 => {
            let det = a.data[[0, 0]] * a.data[[1, 1]] - a.data[[0, 1]] * a.data[[1, 0]];
            if det == 0.0 {
                return Err(PyValueError::new_err("Singular matrix"));
            }
            let inv_det = 1.0 / det;
            let result = vec![
                a.data[[1, 1]] * inv_det,
                -a.data[[0, 1]] * inv_det,
                -a.data[[1, 0]] * inv_det,
                a.data[[0, 0]] * inv_det,
            ];
            let arr = Array::from_shape_vec((2, 2), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                data: arr.into_dyn(),
            })
        }
        3 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]];
            let a32 = a.data[[2, 1]];
            let a33 = a.data[[2, 2]];

            let det = a11 * (a22 * a33 - a23 * a32)
                - a12 * (a21 * a33 - a23 * a31)
                + a13 * (a21 * a32 - a22 * a31);

            if det == 0.0 {
                return Err(PyValueError::new_err("Singular matrix"));
            }
            let inv_det = 1.0 / det;

            let result = vec![
                (a22 * a33 - a23 * a32) * inv_det,
                (a13 * a32 - a12 * a33) * inv_det,
                (a12 * a23 - a13 * a22) * inv_det,
                (a23 * a31 - a21 * a33) * inv_det,
                (a11 * a33 - a13 * a31) * inv_det,
                (a13 * a21 - a11 * a23) * inv_det,
                (a21 * a32 - a22 * a31) * inv_det,
                (a12 * a31 - a11 * a32) * inv_det,
                (a11 * a22 - a12 * a21) * inv_det,
            ];
            let arr = Array::from_shape_vec((3, 3), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                data: arr.into_dyn(),
            })
        }
        _ => Err(PyValueError::new_err(
            "inv only supports 2x2 and 3x3 matrices",
        )),
    }
}

/// 行列式 (2x2, 3x3)
#[pyfunction]
fn det(a: &NdArray) -> PyResult<f64> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("det requires a square matrix"));
    }
    let n = shape[0];
    match n {
        2 => Ok(a.data[[0, 0]] * a.data[[1, 1]] - a.data[[0, 1]] * a.data[[1, 0]]),
        3 => {
            let a11 = a.data[[0, 0]];
            let a12 = a.data[[0, 1]];
            let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]];
            let a22 = a.data[[1, 1]];
            let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]];
            let a32 = a.data[[2, 1]];
            let a33 = a.data[[2, 2]];
            Ok(a11 * (a22 * a33 - a23 * a32)
                - a12 * (a21 * a33 - a23 * a31)
                + a13 * (a21 * a32 - a22 * a31))
        }
        _ => Err(PyValueError::new_err(
            "det only supports 2x2 and 3x3 matrices",
        )),
    }
}

fn matmul_impl(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    let m = a.shape()[0];
    let n = a.shape()[1];
    let p = b.shape()[1];
    let mut result = vec![0.0; m * p];
    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[[i, k]] * b[[k, j]];
            }
            result[i * p + j] = sum;
        }
    }
    Array::from_shape_vec((m, p), result).unwrap().into_dyn()
}

fn solve2x2(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    let inv_det = 1.0 / (a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]);
    let a11 = a[[1, 1]] * inv_det;
    let a12 = -a[[0, 1]] * inv_det;
    let a21 = -a[[1, 0]] * inv_det;
    let a22 = a[[0, 0]] * inv_det;
    let b_shape = b.shape().to_vec();
    if b_shape.len() == 1 {
        let x0 = a11 * b[0] + a12 * b[1];
        let x1 = a21 * b[0] + a22 * b[1];
        Array::from_shape_vec(IxDyn(&[2]), vec![x0, x1]).unwrap().into_dyn()
    } else {
        let n = b_shape[1];
        let mut result = vec![0.0; 2 * n];
        for j in 0..n {
            result[j] = a11 * b[[0, j]] + a12 * b[[1, j]];
            result[n + j] = a21 * b[[0, j]] + a22 * b[[1, j]];
        }
        Array::from_shape_vec((2, n), result).unwrap().into_dyn()
    }
}

fn solve3x3(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    let a11 = a[[0, 0]]; let a12 = a[[0, 1]]; let a13 = a[[0, 2]];
    let a21 = a[[1, 0]]; let a22 = a[[1, 1]]; let a23 = a[[1, 2]];
    let a31 = a[[2, 0]]; let a32 = a[[2, 1]]; let a33 = a[[2, 2]];
    let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
    let inv_det = 1.0 / det;
    let inv = [
        (a22 * a33 - a23 * a32) * inv_det,
        (a13 * a32 - a12 * a33) * inv_det,
        (a12 * a23 - a13 * a22) * inv_det,
        (a23 * a31 - a21 * a33) * inv_det,
        (a11 * a33 - a13 * a31) * inv_det,
        (a13 * a21 - a11 * a23) * inv_det,
        (a21 * a32 - a22 * a31) * inv_det,
        (a12 * a31 - a11 * a32) * inv_det,
        (a11 * a22 - a12 * a21) * inv_det,
    ];
    let b_shape = b.shape().to_vec();
    if b_shape.len() == 1 {
        let x0 = inv[0] * b[0] + inv[1] * b[1] + inv[2] * b[2];
        let x1 = inv[3] * b[0] + inv[4] * b[1] + inv[5] * b[2];
        let x2 = inv[6] * b[0] + inv[7] * b[1] + inv[8] * b[2];
        Array::from_shape_vec(IxDyn(&[3]), vec![x0, x1, x2]).unwrap().into_dyn()
    } else {
        let n = b_shape[1];
        let mut result = vec![0.0; 3 * n];
        for j in 0..n {
            result[j] = inv[0] * b[[0, j]] + inv[1] * b[[1, j]] + inv[2] * b[[2, j]];
            result[n + j] = inv[3] * b[[0, j]] + inv[4] * b[[1, j]] + inv[5] * b[[2, j]];
            result[2 * n + j] = inv[6] * b[[0, j]] + inv[7] * b[[1, j]] + inv[8] * b[[2, j]];
        }
        Array::from_shape_vec((3, n), result).unwrap().into_dyn()
    }
}

fn eig2x2(a: &Array<f64, IxDyn>) -> (Array<f64, IxDyn>, Array<f64, IxDyn>) {
    let a11 = a[[0, 0]]; let a12 = a[[0, 1]];
    let a21 = a[[1, 0]]; let a22 = a[[1, 1]];
    let tr = a11 + a22;
    let det = a11 * a22 - a12 * a21;
    let disc = tr * tr - 4.0 * det;
    let sqrt_disc = disc.sqrt();
    let lambda1 = (tr + sqrt_disc) / 2.0;
    let lambda2 = (tr - sqrt_disc) / 2.0;
    let (v1_0, v1_1) = if a12.abs() > 1e-12 || (lambda1 - a11).abs() > 1e-12 {
        (a12, lambda1 - a11)
    } else {
        (lambda1 - a22, a21)
    };
    let norm1 = (v1_0 * v1_0 + v1_1 * v1_1).sqrt();
    let (v1_0, v1_1) = (v1_0 / norm1, v1_1 / norm1);
    let (v2_0, v2_1) = if a12.abs() > 1e-12 || (lambda2 - a11).abs() > 1e-12 {
        (a12, lambda2 - a11)
    } else {
        (lambda2 - a22, a21)
    };
    let norm2 = (v2_0 * v2_0 + v2_1 * v2_1).sqrt();
    let (v2_0, v2_1) = (v2_0 / norm2, v2_1 / norm2);
    let eigenvalues = Array::from_shape_vec(IxDyn(&[2]), vec![lambda1, lambda2]).unwrap().into_dyn();
    let eigenvectors = Array::from_shape_vec((2, 2), vec![v1_0, v2_0, v1_1, v2_1]).unwrap().into_dyn();
    (eigenvalues, eigenvectors)
}

fn eig3x3(a: &Array<f64, IxDyn>) -> (Array<f64, IxDyn>, Array<f64, IxDyn>) {
    let a11 = a[[0, 0]]; let a12 = a[[0, 1]]; let a13 = a[[0, 2]];
    let a21 = a[[1, 0]]; let a22 = a[[1, 1]]; let a23 = a[[1, 2]];
    let a31 = a[[2, 0]]; let a32 = a[[2, 1]]; let a33 = a[[2, 2]];
    let tr = a11 + a22 + a33;
    let m = (a11 * a22 - a12 * a21) + (a11 * a33 - a13 * a31) + (a22 * a33 - a23 * a32);
    let d = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
    let p = -tr;
    let q = m;
    let r = -d;
    let p2 = p * p;
    let alpha = q - p2 / 3.0;
    let beta = 2.0 * p2 * p / 27.0 - p * q / 3.0 + r;
    let disc = beta * beta / 4.0 + alpha * alpha * alpha / 27.0;
    if disc > 1e-10 {
        panic!("Complex eigenvalues not supported in eig3x3");
    }
    let sqrt_neg_alpha_3 = (-alpha / 3.0).sqrt();
    let cos_arg = (-beta / (2.0 * sqrt_neg_alpha_3.powi(3))).clamp(-1.0, 1.0);
    let theta = cos_arg.acos() / 3.0;
    let two_sqrt = 2.0 * sqrt_neg_alpha_3;
    let pi = std::f64::consts::PI;
    let t0 = two_sqrt * theta.cos();
    let t1 = two_sqrt * (theta - 2.0 * pi / 3.0).cos();
    let t2 = two_sqrt * (theta - 4.0 * pi / 3.0).cos();
    let shift = p / 3.0;
    let eigenvalues = vec![t0 - shift, t1 - shift, t2 - shift];
    let eig_arr = Array::from_shape_vec(IxDyn(&[3]), eigenvalues.clone()).unwrap().into_dyn();
    let mut eigvecs = vec![0.0; 9];
    for (idx, &lam) in eigenvalues.iter().enumerate() {
        let b00 = a11 - lam; let b01 = a12; let b02 = a13;
        let b10 = a21; let b11 = a22 - lam; let b12 = a23;
        let b20 = a31; let b21 = a32; let b22 = a33 - lam;
        let (vx, vy, vz) = {
            let vx = b01 * b12 - b02 * b11;
            let vy = b02 * b10 - b00 * b12;
            let vz = b00 * b11 - b01 * b10;
            let n = (vx * vx + vy * vy + vz * vz).sqrt();
            if n > 1e-12 {
                (vx / n, vy / n, vz / n)
            } else {
                let vx = b01 * b22 - b02 * b21;
                let vy = b02 * b20 - b00 * b22;
                let vz = b00 * b21 - b01 * b20;
                let n = (vx * vx + vy * vy + vz * vz).sqrt();
                if n > 1e-12 {
                    (vx / n, vy / n, vz / n)
                } else {
                    let vx = b11 * b22 - b12 * b21;
                    let vy = b12 * b20 - b10 * b22;
                    let vz = b10 * b21 - b11 * b20;
                    let n = (vx * vx + vy * vy + vz * vz).sqrt();
                    (vx / n, vy / n, vz / n)
                }
            }
        };
        eigvecs[idx] = vx;
        eigvecs[3 + idx] = vy;
        eigvecs[6 + idx] = vz;
    }
    let evec_arr = Array::from_shape_vec((3, 3), eigvecs).unwrap().into_dyn();
    (eig_arr, evec_arr)
}

#[pyfunction]
#[pyo3(signature = (x, ord=None, axis=None))]
fn norm(x: &NdArray, ord: Option<f64>, axis: Option<isize>) -> PyResult<NdArray> {
    let shape = x.data.shape().to_vec();
    let data = &x.data;
    let ord = ord.unwrap_or(2.0);
    if let Some(ax) = axis {
        let ax = if ax < 0 { (shape.len() as isize + ax) as usize } else { ax as usize };
        if ax >= shape.len() {
            return Err(PyValueError::new_err(format!("axis out of bounds")));
        }
        if shape.len() == 2 {
            let nrows = shape[0];
            let ncols = shape[1];
            if ax == 0 {
                let mut result = vec![0.0; ncols];
                for j in 0..ncols {
                    let mut s = 0.0;
                    for i in 0..nrows {
                        let v = data[[i, j]];
                        if ord == 1.0 { s += v.abs(); }
                        else if ord == 2.0 { s += v * v; }
                        else if ord == f64::INFINITY { s = s.max(v.abs()); }
                        else { return Err(PyValueError::new_err(format!("Unsupported norm order: {}", ord))); }
                    }
                    if ord == 2.0 { s = s.sqrt(); }
                    result[j] = s;
                }
                let arr = Array::from_shape_vec(IxDyn(&[ncols]), result)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(NdArray { data: arr })
            } else {
                let mut result = vec![0.0; nrows];
                for i in 0..nrows {
                    let mut s = 0.0;
                    for j in 0..ncols {
                        let v = data[[i, j]];
                        if ord == 1.0 { s += v.abs(); }
                        else if ord == 2.0 { s += v * v; }
                        else if ord == f64::INFINITY { s = s.max(v.abs()); }
                        else { return Err(PyValueError::new_err(format!("Unsupported norm order: {}", ord))); }
                    }
                    if ord == 2.0 { s = s.sqrt(); }
                    result[i] = s;
                }
                let arr = Array::from_shape_vec(IxDyn(&[nrows]), result)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(NdArray { data: arr })
            }
        } else {
            Err(PyValueError::new_err("norm with axis only supports 2D arrays"))
        }
    } else if shape.len() == 1 {
        if ord == 1.0 {
            let val: f64 = data.iter().map(|v| v.abs()).sum();
            Ok(NdArray { data: Array::from_elem(IxDyn(&[]), val) })
        } else if ord == 2.0 {
            let val: f64 = data.iter().map(|v| v * v).sum::<f64>().sqrt();
            Ok(NdArray { data: Array::from_elem(IxDyn(&[]), val) })
        } else if ord == f64::INFINITY {
            let val = data.iter().fold(0.0f64, |a, b| a.max(b.abs()));
            Ok(NdArray { data: Array::from_elem(IxDyn(&[]), val) })
        } else {
            Err(PyValueError::new_err(format!("Unsupported norm order for 1D array: {}", ord)))
        }
    } else if shape.len() == 2 {
        if ord == 1.0 {
            let ncols = shape[1];
            let mut max_sum = 0.0;
            for j in 0..ncols {
                let s: f64 = (0..shape[0]).map(|i| data[[i, j]].abs()).sum();
                if s > max_sum { max_sum = s; }
            }
            Ok(NdArray { data: Array::from_elem(IxDyn(&[]), max_sum) })
        } else if ord == 2.0 {
            let val: f64 = data.iter().map(|v| v * v).sum::<f64>().sqrt();
            Ok(NdArray { data: Array::from_elem(IxDyn(&[]), val) })
        } else if ord == f64::INFINITY {
            let ncols = shape[1];
            let mut max_sum = 0.0;
            for i in 0..shape[0] {
                let s: f64 = (0..ncols).map(|j| data[[i, j]].abs()).sum();
                if s > max_sum { max_sum = s; }
            }
            Ok(NdArray { data: Array::from_elem(IxDyn(&[]), max_sum) })
        } else {
            Err(PyValueError::new_err(format!("Unsupported norm order for matrix: {}", ord)))
        }
    } else {
        Err(PyValueError::new_err(format!("norm not supported for {}D array", shape.len())))
    }
}

#[pyfunction]
fn solve(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let a_shape = a.data.shape().to_vec();
    let b_shape = b.data.shape().to_vec();
    if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
        return Err(PyValueError::new_err("solve requires a square matrix A"));
    }
    let n = a_shape[0];
    if !(b_shape.len() == 1 || b_shape.len() == 2) {
        return Err(PyValueError::new_err("b must be 1D or 2D"));
    }
    let b_dim0 = b_shape[0];
    if b_dim0 != n {
        return Err(PyValueError::new_err(format!(
            "Incompatible shapes: A {:?} and b {:?}", a_shape, b_shape
        )));
    }
    match n {
        2 => {
            let det = a.data[[0, 0]] * a.data[[1, 1]] - a.data[[0, 1]] * a.data[[1, 0]];
            if det == 0.0 {
                return Err(PyValueError::new_err("Singular matrix"));
            }
            Ok(NdArray { data: solve2x2(&a.data, &b.data) })
        }
        3 => {
            let a11 = a.data[[0, 0]]; let a12 = a.data[[0, 1]]; let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]]; let a22 = a.data[[1, 1]]; let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]]; let a32 = a.data[[2, 1]]; let a33 = a.data[[2, 2]];
            let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
            if det == 0.0 {
                return Err(PyValueError::new_err("Singular matrix"));
            }
            Ok(NdArray { data: solve3x3(&a.data, &b.data) })
        }
        _ => Err(PyValueError::new_err("solve only supports 2x2 and 3x3 matrices")),
    }
}

#[pyfunction]
fn eig(a: &NdArray) -> PyResult<(NdArray, NdArray)> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("eig requires a square matrix"));
    }
    let n = shape[0];
    match n {
        2 => {
            let a11 = a.data[[0, 0]]; let a12 = a.data[[0, 1]];
            let a21 = a.data[[1, 0]]; let a22 = a.data[[1, 1]];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det;
            if disc < 0.0 {
                return Err(PyValueError::new_err("eig does not support complex eigenvalues"));
            }
            let (evals, evecs) = eig2x2(&a.data);
            Ok((NdArray { data: evals }, NdArray { data: evecs }))
        }
        3 => {
            let a11 = a.data[[0, 0]]; let a12 = a.data[[0, 1]]; let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]]; let a22 = a.data[[1, 1]]; let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]]; let a32 = a.data[[2, 1]]; let a33 = a.data[[2, 2]];
            let tr = a11 + a22 + a33;
            let m_val = (a11 * a22 - a12 * a21) + (a11 * a33 - a13 * a31) + (a22 * a33 - a23 * a32);
            let d = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
            let p = -tr;
            let q = m_val;
            let r = -d;
            let p2 = p * p;
            let alpha = q - p2 / 3.0;
            let beta = 2.0 * p2 * p / 27.0 - p * q / 3.0 + r;
            let disc = beta * beta / 4.0 + alpha * alpha * alpha / 27.0;
            if disc > 1e-10 {
                return Err(PyValueError::new_err("eig does not support complex eigenvalues"));
            }
            let (evals, evecs) = eig3x3(&a.data);
            Ok((NdArray { data: evals }, NdArray { data: evecs }))
        }
        _ => Err(PyValueError::new_err("eig only supports 2x2 and 3x3 matrices")),
    }
}

#[pyfunction]
fn eigvals(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("eigvals requires a square matrix"));
    }
    let n = shape[0];
    match n {
        2 => {
            let a11 = a.data[[0, 0]]; let a12 = a.data[[0, 1]];
            let a21 = a.data[[1, 0]]; let a22 = a.data[[1, 1]];
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det;
            if disc < 0.0 {
                return Err(PyValueError::new_err("eigvals does not support complex eigenvalues"));
            }
            let sqrt_disc = disc.sqrt();
            let lambda1 = (tr + sqrt_disc) / 2.0;
            let lambda2 = (tr - sqrt_disc) / 2.0;
            let arr = Array::from_shape_vec(IxDyn(&[2]), vec![lambda1, lambda2])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: arr })
        }
        3 => {
            let a11 = a.data[[0, 0]]; let a12 = a.data[[0, 1]]; let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]]; let a22 = a.data[[1, 1]]; let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]]; let a32 = a.data[[2, 1]]; let a33 = a.data[[2, 2]];
            let tr = a11 + a22 + a33;
            let m_val = (a11 * a22 - a12 * a21) + (a11 * a33 - a13 * a31) + (a22 * a33 - a23 * a32);
            let d = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
            let p = -tr;
            let q = m_val;
            let r = -d;
            let p2 = p * p;
            let alpha = q - p2 / 3.0;
            let beta = 2.0 * p2 * p / 27.0 - p * q / 3.0 + r;
            let disc = beta * beta / 4.0 + alpha * alpha * alpha / 27.0;
            if disc > 1e-10 {
                return Err(PyValueError::new_err("eigvals does not support complex eigenvalues"));
            }
            let sqrt_neg_alpha_3 = (-alpha / 3.0).sqrt();
            let cos_arg = (-beta / (2.0 * sqrt_neg_alpha_3.powi(3))).clamp(-1.0, 1.0);
            let theta = cos_arg.acos() / 3.0;
            let two_sqrt = 2.0 * sqrt_neg_alpha_3;
            let pi = std::f64::consts::PI;
            let t0 = two_sqrt * theta.cos();
            let t1 = two_sqrt * (theta - 2.0 * pi / 3.0).cos();
            let t2 = two_sqrt * (theta - 4.0 * pi / 3.0).cos();
            let shift = p / 3.0;
            let eigenvalues = vec![t0 - shift, t1 - shift, t2 - shift];
            let arr = Array::from_shape_vec(IxDyn(&[3]), eigenvalues)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: arr })
        }
        _ => Err(PyValueError::new_err("eigvals only supports 2x2 and 3x3 matrices")),
    }
}

#[pyfunction]
fn svd(a: &NdArray) -> PyResult<(NdArray, NdArray, NdArray)> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err("svd only supports 2x2 matrices"));
    }
    let a11 = a.data[[0, 0]]; let a12 = a.data[[0, 1]];
    let a21 = a.data[[1, 0]]; let a22 = a.data[[1, 1]];
    let b11 = a11 * a11 + a21 * a21;
    let b12 = a11 * a12 + a21 * a22;
    let b22 = a12 * a12 + a22 * a22;
    let tr = b11 + b22;
    let det = b11 * b22 - b12 * b12;
    let disc = tr * tr - 4.0 * det;
    let sqrt_disc = disc.sqrt();
    let lambda1 = (tr + sqrt_disc) / 2.0;
    let lambda2 = (tr - sqrt_disc) / 2.0;
    let sigma1 = lambda1.sqrt();
    let sigma2 = lambda2.sqrt();
    let (v1x, v1y) = if b12.abs() > 1e-12 {
        (b12, lambda1 - b11)
    } else {
        (1.0, 0.0)
    };
    let v1_norm = (v1x * v1x + v1y * v1y).sqrt();
    let (v1x, v1y) = (v1x / v1_norm, v1y / v1_norm);
    let (v2x, v2y) = if b12.abs() > 1e-12 {
        (b12, lambda2 - b11)
    } else {
        (0.0, 1.0)
    };
    let v2_norm = (v2x * v2x + v2y * v2y).sqrt();
    let (v2x, v2y) = (v2x / v2_norm, v2y / v2_norm);
    let u1x = (a11 * v1x + a12 * v1y) / sigma1;
    let u1y = (a21 * v1x + a22 * v1y) / sigma1;
    let u1_norm = (u1x * u1x + u1y * u1y).sqrt();
    let (u1x, u1y) = (u1x / u1_norm, u1y / u1_norm);
    let u2x = (a11 * v2x + a12 * v2y) / sigma2;
    let u2y = (a21 * v2x + a22 * v2y) / sigma2;
    let u2_norm = (u2x * u2x + u2y * u2y).sqrt();
    let (u2x, u2y) = (u2x / u2_norm, u2y / u2_norm);
    let u_data = vec![u1x, u2x, u1y, u2y];
    let s_data = vec![sigma1, 0.0, 0.0, sigma2];
    let vt_data = vec![v1x, v1y, v2x, v2y];
    let u_arr = Array::from_shape_vec((2, 2), u_data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let s_arr = Array::from_shape_vec((2, 2), s_data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let vt_arr = Array::from_shape_vec((2, 2), vt_data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        NdArray { data: u_arr.into_dyn() },
        NdArray { data: s_arr.into_dyn() },
        NdArray { data: vt_arr.into_dyn() },
    ))
}

#[pyfunction]
fn qr(a: &NdArray) -> PyResult<(NdArray, NdArray)> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("qr requires a 2D array"));
    }
    let m = shape[0];
    let n = shape[1];
    if m != n || (m != 2 && m != 3) {
        return Err(PyValueError::new_err("qr only supports 2x2 and 3x3 matrices"));
    }
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    for j in 0..n {
        let mut col = Vec::with_capacity(m);
        for i in 0..m {
            col.push(a.data[[i, j]]);
        }
        cols.push(col);
    }
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut r_vals = vec![0.0; n * n];
    for j in 0..n {
        let mut v = cols[j].clone();
        for i in 0..j {
            let dot: f64 = (0..m).map(|k| cols[j][k] * q_cols[i][k]).sum();
            r_vals[i * n + j] = dot;
            for k in 0..m {
                v[k] -= dot * q_cols[i][k];
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        r_vals[j * n + j] = norm;
        let q_col: Vec<f64> = v.into_iter().map(|x| x / norm).collect();
        q_cols.push(q_col);
    }
    let mut q_data = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            q_data[i * n + j] = q_cols[j][i];
        }
    }
    let q_arr = Array::from_shape_vec((m, n), q_data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let r_arr = Array::from_shape_vec((n, n), r_vals)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((NdArray { data: q_arr.into_dyn() }, NdArray { data: r_arr.into_dyn() }))
}

#[pyfunction]
fn cholesky(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("cholesky requires a square matrix"));
    }
    let n = shape[0];
    match n {
        2 => {
            let l11 = a.data[[0, 0]].sqrt();
            let l21 = a.data[[1, 0]] / l11;
            let l22 = (a.data[[1, 1]] - l21 * l21).sqrt();
            let result = vec![l11, 0.0, l21, l22];
            let arr = Array::from_shape_vec((2, 2), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: arr.into_dyn() })
        }
        3 => {
            let l11 = a.data[[0, 0]].sqrt();
            let l21 = a.data[[1, 0]] / l11;
            let l31 = a.data[[2, 0]] / l11;
            let l22 = (a.data[[1, 1]] - l21 * l21).sqrt();
            let l32 = (a.data[[2, 1]] - l31 * l21) / l22;
            let l33 = (a.data[[2, 2]] - l31 * l31 - l32 * l32).sqrt();
            let result = vec![l11, 0.0, 0.0, l21, l22, 0.0, l31, l32, l33];
            let arr = Array::from_shape_vec((3, 3), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: arr.into_dyn() })
        }
        _ => Err(PyValueError::new_err("cholesky only supports 2x2 and 3x3 matrices")),
    }
}

#[pyfunction]
fn matrix_power(a: &NdArray, n: i32) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("matrix_power requires a square matrix"));
    }
    let n_dim = shape[0];
    if n_dim != 2 && n_dim != 3 {
        return Err(PyValueError::new_err("matrix_power only supports 2x2 and 3x3 matrices"));
    }
    if n == 0 {
        let mut result = vec![0.0; n_dim * n_dim];
        for i in 0..n_dim {
            result[i * n_dim + i] = 1.0;
        }
        let arr = Array::from_shape_vec((n_dim, n_dim), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { data: arr.into_dyn() });
    }
    if n > 0 {
        let mut result = a.data.clone();
        for _ in 1..n {
            result = matmul_impl(&result, &a.data);
        }
        Ok(NdArray { data: result })
    } else {
        let inv_arr = inv(&NdArray { data: a.data.clone() })?;
        let abs_n = (-n) as u32;
        let mut result = inv_arr.data.clone();
        for _ in 1..abs_n {
            result = matmul_impl(&result, &inv_arr.data);
        }
        Ok(NdArray { data: result })
    }
}

#[pyfunction]
fn pinv(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("pinv requires a 2D array"));
    }
    let m = shape[0];
    let n = shape[1];
    if m == n && m <= 3 {
        return inv(a);
    }
    if m == 2 && n == 2 {
        return inv(a);
    }
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += a.data[[k, i]] * a.data[[k, j]];
            }
            ata[i * n + j] = sum;
        }
    }
    let ata_arr = Array::from_shape_vec((n, n), ata)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ata_nd = NdArray { data: ata_arr.into_dyn() };
    let ata_inv = inv(&ata_nd)?;
    let mut result = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += ata_inv.data[[i, k]] * a.data[[j, k]];
            }
            result[i * m + j] = sum;
        }
    }
    let arr = Array::from_shape_vec((n, m), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr.into_dyn() })
}

#[pyfunction]
fn trace(a: &NdArray) -> PyResult<f64> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("trace requires a square matrix"));
    }
    let n = shape[0];
    let mut result = 0.0;
    for i in 0..n {
        result += a.data[[i, i]];
    }
    Ok(result)
}

#[pyfunction]
fn diagonal(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("diagonal requires a 2D array"));
    }
    let n = shape[0].min(shape[1]);
    let mut result = vec![0.0; n];
    for i in 0..n {
        result[i] = a.data[[i, i]];
    }
    let arr = Array::from_shape_vec(IxDyn(&[n]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
fn svdvals(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("svdvals requires a 2D array"));
    }
    let m = shape[0];
    let n = shape[1];
    if m != n || (m != 2 && m != 3) {
        return Err(PyValueError::new_err("svdvals only supports 2x2 and 3x3 matrices"));
    }
    match m {
        2 => {
            let a11 = a.data[[0, 0]]; let a12 = a.data[[0, 1]];
            let a21 = a.data[[1, 0]]; let a22 = a.data[[1, 1]];
            let b11 = a11 * a11 + a21 * a21;
            let b12 = a11 * a12 + a21 * a22;
            let b22 = a12 * a12 + a22 * a22;
            let tr = b11 + b22;
            let det = b11 * b22 - b12 * b12;
            let disc = tr * tr - 4.0 * det;
            let sqrt_disc = disc.sqrt();
            let lambda1 = (tr + sqrt_disc) / 2.0;
            let lambda2 = (tr - sqrt_disc) / 2.0;
            let sigma1 = lambda1.sqrt();
            let sigma2 = lambda2.sqrt();
            let arr = Array::from_shape_vec(IxDyn(&[2]), vec![sigma1, sigma2])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: arr })
        }
        3 => {
            let a11 = a.data[[0, 0]]; let a12 = a.data[[0, 1]]; let a13 = a.data[[0, 2]];
            let a21 = a.data[[1, 0]]; let a22 = a.data[[1, 1]]; let a23 = a.data[[1, 2]];
            let a31 = a.data[[2, 0]]; let a32 = a.data[[2, 1]]; let a33 = a.data[[2, 2]];
            let b11 = a11*a11 + a21*a21 + a31*a31;
            let b12 = a11*a12 + a21*a22 + a31*a32;
            let b13 = a11*a13 + a21*a23 + a31*a33;
            let b22 = a12*a12 + a22*a22 + a32*a32;
            let b23 = a12*a13 + a22*a23 + a32*a33;
            let b33 = a13*a13 + a23*a23 + a33*a33;
            let tr = b11 + b22 + b33;
            let m_val = b11*b22 - b12*b12 + b11*b33 - b13*b13 + b22*b33 - b23*b23;
            let d = b11*(b22*b33 - b23*b23) - b12*(b12*b33 - b13*b23) + b13*(b12*b23 - b22*b13);
            let p = -tr;
            let q = m_val;
            let r = -d;
            let p2 = p * p;
            let alpha = q - p2 / 3.0;
            let beta = 2.0 * p2 * p / 27.0 - p * q / 3.0 + r;
            let disc = beta * beta / 4.0 + alpha * alpha * alpha / 27.0;
            if disc > 1e-10 {
                return Err(PyValueError::new_err("svdvals does not support complex singular values"));
            }
            let sqrt_neg_alpha_3 = (-alpha / 3.0).sqrt();
            let cos_arg = (-beta / (2.0 * sqrt_neg_alpha_3.powi(3))).clamp(-1.0, 1.0);
            let theta = cos_arg.acos() / 3.0;
            let two_sqrt = 2.0 * sqrt_neg_alpha_3;
            let pi = std::f64::consts::PI;
            let t0 = two_sqrt * theta.cos();
            let t1 = two_sqrt * (theta - 2.0 * pi / 3.0).cos();
            let t2 = two_sqrt * (theta - 4.0 * pi / 3.0).cos();
            let shift = p / 3.0;
            let sigma1 = (t0 - shift).sqrt();
            let sigma2 = (t1 - shift).sqrt();
            let sigma3 = (t2 - shift).sqrt();
            let arr = Array::from_shape_vec(IxDyn(&[3]), vec![sigma1, sigma2, sigma3])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: arr })
        }
        _ => Err(PyValueError::new_err("svdvals only supports 2x2 and 3x3 matrices")),
    }
}

#[pyfunction]
fn eigh(a: &NdArray) -> PyResult<(NdArray, NdArray)> {
    let shape = a.data.shape().to_vec();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err("eigh requires a square matrix"));
    }
    let n = shape[0];
    if n != 2 && n != 3 {
        return Err(PyValueError::new_err("eigh only supports 2x2 and 3x3 matrices"));
    }
    eig(a)
}

#[pyfunction]
fn solve_banded(lower: usize, upper: usize, ab: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let ab_shape = ab.data.shape().to_vec();
    let b_shape = b.data.shape().to_vec();
    
    if ab_shape.len() != 2 {
        return Err(PyValueError::new_err("ab must be a 2D array"));
    }
    if b_shape.len() != 1 && b_shape.len() != 2 {
        return Err(PyValueError::new_err("b must be 1D or 2D"));
    }
    
    let n = ab_shape[1];
    let m = ab_shape[0];
    
    if m != lower + upper + 1 {
        return Err(PyValueError::new_err("ab rows must equal lower + upper + 1"));
    }
    if b_shape[0] != n {
        return Err(PyValueError::new_err("b dimensions must match matrix size"));
    }
    
    if n != 2 && n != 3 {
        return Err(PyValueError::new_err("solve_banded only supports 2x2 and 3x3 systems"));
    }
    
    let mut full = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..m {
            let row_in_full = (j + i) as isize - upper as isize;
            if row_in_full >= 0 && row_in_full < n as isize {
                full[row_in_full as usize * n + j] = ab.data[[i, j]];
            }
        }
    }
    
    let full_arr = Array::from_shape_vec((n, n), full)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let full_nd = NdArray { data: full_arr.into_dyn() };
    
    solve(&full_nd, b)
}

pub fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(vdot, m)?)?;
    m.add_function(wrap_pyfunction!(inner, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(inv, m)?)?;
    m.add_function(wrap_pyfunction!(det, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(eig, m)?)?;
    m.add_function(wrap_pyfunction!(eigvals, m)?)?;
    m.add_function(wrap_pyfunction!(svd, m)?)?;
    m.add_function(wrap_pyfunction!(qr, m)?)?;
    m.add_function(wrap_pyfunction!(cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_power, m)?)?;
    m.add_function(wrap_pyfunction!(pinv, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(svdvals, m)?)?;
    m.add_function(wrap_pyfunction!(eigh, m)?)?;
    m.add_function(wrap_pyfunction!(solve_banded, m)?)?;
    Ok(())
}
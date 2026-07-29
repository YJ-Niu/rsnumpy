use crate::*;

#[pyfunction]
fn invert(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| if v == 0.0 { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn bitwise_and(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) & (b as i64);
        bits as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn bitwise_or(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) | (b as i64);
        bits as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn bitwise_xor(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) ^ (b as i64);
        bits as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn left_shift(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) << (b as i64);
        bits as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn right_shift(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) >> (b as i64);
        bits as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn bitwise_not(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| {
            let bits = !(v as i64);
            bits as f64
        }),
    }
}

#[pyfunction]
fn binary_repr(num: i64, width: Option<usize>) -> String {
    let w = if let Some(w) = width {
        w
    } else {
        let bits = 64 - num.abs().leading_zeros();
        if bits == 0 { 1 } else { bits as usize }
    };

    if num >= 0 {
        format!("{num:0w$b}")
    } else {
        let val = (1 << w) + num;
        format!("{val:0w$b}")
    }
}

/// 逐元素统计绝对值二进制表示中置位比特数（与 numpy.bitwise_count 一致，输出 uint8）。
#[pyfunction]
fn bitwise_count(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| (v.abs() as i64).count_ones() as f64),
    }
}

// ========== 比特打包 ==========
/// 将布尔/整数数组按比特打包为 uint8
#[pyfunction]
#[pyo3(signature = (a, bitorder="big"))]
fn packbits(a: &NdArray, bitorder: &str) -> NdArray {
    let bits: Vec<u8> = a
        .data
        .iter()
        .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
        .collect();
    let n = bits.len().div_ceil(8);
    let mut out = vec![0u8; n];
    for (i, &b) in bits.iter().enumerate() {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if bitorder == "big" {
            out[byte_idx] |= b << (7 - bit_idx);
        } else {
            out[byte_idx] |= b << bit_idx;
        }
    }
    let result: Vec<f64> = out.iter().map(|&v| v as f64).collect();
    let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
        .expect("打包比特后构造数组形状必然合法");
    NdArray {
        imag: None,
        data: arr,
    }
}

// ========== 比特展开 ==========
/// 将 uint8 数组展开为比特
#[pyfunction]
#[pyo3(signature = (a, count=None, bitorder="big"))]
fn unpackbits(a: &NdArray, count: Option<usize>, bitorder: &str) -> NdArray {
    let vals: Vec<u8> = a.data.iter().map(|&v| v as u8).collect();
    let mut out: Vec<u8> = Vec::with_capacity(vals.len() * 8);
    for &iv in &vals {
        let mut bits = [0u8; 8];
        for (j, bit) in bits.iter_mut().enumerate() {
            *bit = if bitorder == "big" {
                (iv >> (7 - j)) & 1
            } else {
                (iv >> j) & 1
            };
        }
        out.extend(bits.iter());
    }
    if let Some(c) = count {
        out.truncate(c);
    }
    let result: Vec<f64> = out.iter().map(|&v: &u8| v as f64).collect();
    let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
        .expect("展开比特后构造数组形状必然合法");
    NdArray {
        imag: None,
        data: arr,
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(invert, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(left_shift, m)?)?;
    m.add_function(wrap_pyfunction!(right_shift, m)?)?;
    m.add_function(wrap_pyfunction!(binary_repr, m)?)?;
    m.add_function(wrap_pyfunction!(packbits, m)?)?;
    m.add_function(wrap_pyfunction!(unpackbits, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_count, m)?)?;
    Ok(())
}

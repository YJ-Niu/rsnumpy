use crate::*;

#[pyfunction]
#[pyo3(signature = (bytes, count=-1))]
fn bytes_to_floats(bytes: &[u8], count: isize) -> PyResult<NdArray> {
    use std::mem;
    let n = bytes.len() / mem::size_of::<f64>();
    let actual = if count < 0 { n } else { count as usize };
    let mut result = Vec::with_capacity(actual);
    for i in 0..actual {
        let start = i * 8;
        if start + 8 > bytes.len() {
            break;
        }
        let chunk = &bytes[start..start + 8];
        let arr_u8: [u8; 8] = [
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ];
        result.push(f64::from_le_bytes(arr_u8));
    }
    let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// ========== 从数组接口协议缓冲区创建数组（如 PIL 图像） ==========
// typestr 形如 "|u1"、"<f8"、">i4"：首字符为字节序，第二字符为类型，其余为字节数。
fn decode_buffer_scalar(kind: char, chunk: &[u8], little: bool) -> Option<f64> {
    let val = match (kind, chunk.len()) {
        ('b', 1) => {
            if chunk[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        ('u', 1) => chunk[0] as f64,
        ('i', 1) => (chunk[0] as i8) as f64,
        ('u', 2) => {
            let b = [chunk[0], chunk[1]];
            (if little {
                u16::from_le_bytes(b)
            } else {
                u16::from_be_bytes(b)
            }) as f64
        }
        ('i', 2) => {
            let b = [chunk[0], chunk[1]];
            (if little {
                i16::from_le_bytes(b)
            } else {
                i16::from_be_bytes(b)
            }) as f64
        }
        ('u', 4) => {
            let b = [chunk[0], chunk[1], chunk[2], chunk[3]];
            (if little {
                u32::from_le_bytes(b)
            } else {
                u32::from_be_bytes(b)
            }) as f64
        }
        ('i', 4) => {
            let b = [chunk[0], chunk[1], chunk[2], chunk[3]];
            (if little {
                i32::from_le_bytes(b)
            } else {
                i32::from_be_bytes(b)
            }) as f64
        }
        ('u', 8) => {
            let b = [
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ];
            (if little {
                u64::from_le_bytes(b)
            } else {
                u64::from_be_bytes(b)
            }) as f64
        }
        ('i', 8) => {
            let b = [
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ];
            (if little {
                i64::from_le_bytes(b)
            } else {
                i64::from_be_bytes(b)
            }) as f64
        }
        ('f', 4) => {
            let b = [chunk[0], chunk[1], chunk[2], chunk[3]];
            (if little {
                f32::from_le_bytes(b)
            } else {
                f32::from_be_bytes(b)
            }) as f64
        }
        ('f', 8) => {
            let b = [
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ];
            if little {
                f64::from_le_bytes(b)
            } else {
                f64::from_be_bytes(b)
            }
        }
        _ => return None,
    };
    Some(val)
}

#[pyfunction]
#[pyo3(signature = (bytes, typestr, shape))]
fn from_buffer_typed(bytes: &[u8], typestr: &str, shape: Vec<usize>) -> PyResult<NdArray> {
    let chars: Vec<char> = typestr.chars().collect();
    let byteorder = chars.first().copied().unwrap_or('|');
    let kind = *chars
        .get(1)
        .ok_or_else(|| PyValueError::new_err("invalid array interface typestr"))?;
    let itemsize: usize = typestr[2..]
        .parse()
        .map_err(|_| PyValueError::new_err("invalid array interface typestr"))?;
    if itemsize == 0 {
        return Err(PyValueError::new_err("invalid array interface typestr"));
    }
    // '>' 为大端；'<'、'|'、'=' 视为小端（构建目标为小端平台）。
    let little = byteorder != '>';

    let count: usize = shape.iter().product();
    if count.saturating_mul(itemsize) > bytes.len() {
        return Err(PyValueError::new_err("buffer smaller than shape requires"));
    }

    let mut result: Vec<f64> = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(itemsize).take(count) {
        let val = decode_buffer_scalar(kind, chunk, little)
            .ok_or_else(|| PyValueError::new_err("unsupported array interface typestr"))?;
        result.push(val);
    }

    let arr = Array::from_shape_vec(IxDyn(&shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

// 供 Python 包装层按其追踪的 dtype 构建数组接口（底层 f64 → 目标类型字节）。
#[pyfunction]
fn array_interface<'py>(arr: &Bound<'py, NdArray>, typestr: &str) -> PyResult<Bound<'py, PyDict>> {
    let py = arr.py();
    let borrowed = arr.borrow();
    build_array_interface(py, &borrowed.data, typestr)
}

// ========== 整数序列归约（供 Python 层替代内置 sum/max，避免依赖 builtins 模块） ==========
#[pyfunction]
fn isum(values: Vec<i64>) -> i64 {
    values.iter().sum()
}

#[pyfunction]
fn imax(values: Vec<i64>) -> PyResult<i64> {
    values
        .into_iter()
        .max()
        .ok_or_else(|| PyValueError::new_err("max() arg is an empty sequence"))
}

// ========== 多维 tuple 索引（key 为 tuple） ==========
#[pyfunction]
#[pyo3(signature = (a, ranges))]
fn tuple_getitem(a: &NdArray, ranges: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    // ranges: list of (start, end, step) tuples
    let list = ranges.cast::<PyList>()?;
    let mut ranges_vec: Vec<(i64, i64, i64)> = Vec::with_capacity(list.len());
    for item in list.iter() {
        let triple = item.extract::<(i64, i64, i64)>()?;
        ranges_vec.push(triple);
    }

    let shape = a.data.shape().to_vec();
    let ndim = shape.len();

    if ranges_vec.len() != ndim {
        return Err(PyValueError::new_err("tuple length must match ndim"));
    }

    // 计算每个维度的输出大小
    let mut out_shape: Vec<usize> = Vec::with_capacity(ndim);
    for &(s, e, st) in ranges_vec.iter() {
        let st = st.max(1);
        if e <= s {
            out_shape.push(0);
        } else {
            out_shape.push(((e - s + st - 1) / st) as usize);
        }
    }

    // 计算 C-order 的总元素数
    let total: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut values = Vec::with_capacity(total);
    let flat_data: Vec<f64> = a.data.iter().copied().collect();

    // 递归生成多维索引
    fn collect(
        dim: usize,
        ndim: usize,
        ranges: &[(i64, i64, i64)],
        shape: &[usize],
        idx: &mut Vec<i64>,
        flat_data: &[f64],
        values: &mut Vec<f64>,
    ) {
        if dim == ndim {
            // 计算 flat_idx
            let mut flat_idx = 0usize;
            let mut stride = 1usize;
            for d in (0..ndim).rev() {
                flat_idx += (idx[d] as usize) * stride;
                stride *= shape[d];
            }
            if flat_idx < flat_data.len() {
                values.push(flat_data[flat_idx]);
            }
            return;
        }
        let (s, e, st) = ranges[dim];
        let mut i = s;
        while i < e {
            idx.push(i);
            collect(dim + 1, ndim, ranges, shape, idx, flat_data, values);
            idx.pop();
            i += st;
        }
    }

    let mut idx = Vec::with_capacity(ndim);
    collect(
        0,
        ndim,
        &ranges_vec,
        &shape,
        &mut idx,
        &flat_data,
        &mut values,
    );

    let arr = if out_shape.is_empty() {
        Array::from_shape_vec(IxDyn(&[]), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        Array::from_shape_vec(IxDyn(&out_shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bytes_to_floats, m)?)?;
    m.add_function(wrap_pyfunction!(from_buffer_typed, m)?)?;
    m.add_function(wrap_pyfunction!(array_interface, m)?)?;
    m.add_function(wrap_pyfunction!(isum, m)?)?;
    m.add_function(wrap_pyfunction!(imax, m)?)?;
    m.add_function(wrap_pyfunction!(tuple_getitem, m)?)?;
    Ok(())
}

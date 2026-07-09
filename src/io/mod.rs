use crate::*;

// ========== I/O: 保存为 .npy ==========
#[pyfunction]
#[pyo3(signature = (filename, a))]
fn save_npy(filename: &str, a: &NdArray) -> PyResult<()> {
    use std::fs::File;
    use std::io::Write;

    let shape = a.data.shape().to_vec();
    let data_vec: Vec<f64> = a.data.iter().copied().collect();

    let mut file = File::create(filename)
        .map_err(|e| PyValueError::new_err(format!("Failed to create file: {}", e)))?;

    file.write_all(b"\x93NUMPY")
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    file.write_all(&[1u8])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    file.write_all(&[0u8])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    let header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}), }}",
        shape_str.join(", ")
    );
    let header_len = header.len();
    let padded_len = (header_len + 10).div_ceil(64) * 64 - 10;
    let padded_header = format!("{:width$}", header, width = padded_len);

    file.write_all(&(padded_len as u16).to_le_bytes())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    file.write_all(padded_header.as_bytes())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    for &v in &data_vec {
        file.write_all(&v.to_le_bytes())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    Ok(())
}

// ========== I/O: 从 .npy 加载 ==========
#[pyfunction]
#[pyo3(signature = (filename))]
fn load_npy(filename: &str) -> PyResult<NdArray> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(filename)
        .map_err(|e| PyValueError::new_err(format!("Failed to open file: {}", e)))?;

    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if &magic != b"\x93NUMPY" {
        return Err(PyValueError::new_err("Not a valid .npy file"));
    }

    let mut version = [0u8; 2];
    file.read_exact(&mut version)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut header_len_bytes = [0u8; 2];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let header_len = u16::from_le_bytes(header_len_bytes) as usize;

    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let header_str = String::from_utf8_lossy(&header);

    let shape: Vec<usize> = if let Some(start) = header_str.find("shape") {
        let rest = &header_str[start + 6..];
        if let Some(open) = rest.find('(') {
            if let Some(close) = rest.find(')') {
                let inside = &rest[open + 1..close];
                if inside.trim().is_empty() {
                    vec![]
                } else {
                    inside
                        .split(',')
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .collect()
                }
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    let total: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let mut values = vec![0.0f64; total];

    for v in values.iter_mut() {
        let mut bytes = [0u8; 8];
        if file.read_exact(&mut bytes).is_err() {
            break;
        }
        *v = f64::from_le_bytes(bytes);
    }

    let arr_shape = if shape.is_empty() {
        IxDyn(&[])
    } else {
        IxDyn(&shape)
    };
    let arr = Array::from_shape_vec(arr_shape, values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

fn format_value(fmt: &str, v: f64) -> String {
    if fmt.contains("%d") || fmt.contains("%i") {
        (v as i64).to_string()
    } else {
        v.to_string()
    }
}

// ========== I/O: 保存为文本文件 ==========
#[pyfunction]
#[pyo3(signature = (filename, a, fmt="%.18e", delimiter=" "))]
fn save_text(filename: &str, a: &NdArray, fmt: &str, delimiter: &str) -> PyResult<()> {
    use std::fs::File;
    use std::io::Write;

    let shape = a.data.shape().to_vec();
    let data_vec: Vec<f64> = a.data.iter().copied().collect();

    let mut file = File::create(filename)
        .map_err(|e| PyValueError::new_err(format!("Failed to create file: {}", e)))?;

    if shape.len() == 1 {
        let line: Vec<String> = data_vec.iter().map(|v| format_value(fmt, *v)).collect();
        writeln!(file, "{}", line.join(delimiter))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    } else if shape.len() == 2 {
        for i in 0..shape[0] {
            let line: Vec<String> = (0..shape[1])
                .map(|j| format_value(fmt, data_vec[i * shape[1] + j]))
                .collect();
            writeln!(file, "{}", line.join(delimiter))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    } else {
        for &v in &data_vec {
            writeln!(file, "{}", format_value(fmt, v))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    // 让编译器忽略未使用的 fmt 参数
    let _ = fmt;
    Ok(())
}

// ========== I/O: 从文本文件加载 ==========
#[pyfunction]
#[pyo3(signature = (filename, delimiter=None, skiprows=0))]
fn load_text(filename: &str, delimiter: Option<&str>, skiprows: usize) -> PyResult<NdArray> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(filename)
        .map_err(|e| PyValueError::new_err(format!("Failed to open file: {}", e)))?;
    let reader = BufReader::new(file);

    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(1000);
    for (idx, line) in reader.lines().enumerate() {
        if idx < skiprows {
            continue;
        }
        let line = line.map_err(|e| PyValueError::new_err(e.to_string()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = match delimiter {
            Some(d) => trimmed.split(d).collect(),
            None => trimmed.split_whitespace().collect(),
        };
        let row: Vec<f64> = parts
            .iter()
            .filter_map(|p| p.trim().parse::<f64>().ok())
            .collect();
        if !row.is_empty() {
            rows.push(row);
        }
    }

    if rows.is_empty() {
        let arr = Array::from_shape_vec(IxDyn(&[0]), vec![])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { data: arr });
    }

    let ncols = rows[0].len();
    let nrows = rows.len();

    if ncols == 1 && rows.iter().all(|r| r.len() == 1) {
        let flat: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let arr = Array::from_shape_vec(IxDyn(&[flat.len()]), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { data: arr });
    }

    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let arr = Array::from_shape_vec(IxDyn(&[nrows, ncols]), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

// ========== savez_npz: 将多个数组保存为 zip 格式 ==========
#[pyfunction]
#[pyo3(signature = (filename, arrays, names))]
fn savez_npz(filename: &str, arrays: &Bound<'_, PyAny>, names: &Bound<'_, PyAny>) -> PyResult<()> {
    use std::fs::File;
    use std::io::Write;

    let arr_list = arrays.cast::<PyList>()?;
    let name_list = names.cast::<PyList>()?;
    if arr_list.len() != name_list.len() {
        return Err(PyValueError::new_err("arrays and names length mismatch"));
    }

    let file = File::create(filename)
        .map_err(|e| PyValueError::new_err(format!("Failed to create file: {}", e)))?;
    let mut zip = zip::ZipWriter::new(file);
    let options: zip::write::FileOptions<'_, ()> =
        zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);

    for (item, name_item) in arr_list.iter().zip(name_list.iter()) {
        let nd = item.extract::<NdArray>()?;
        let name_str = name_item.extract::<String>()?;
        let entry_name = if name_str.ends_with(".npy") {
            name_str
        } else {
            format!("{}.npy", name_str)
        };

        // 序列化 ndarray 为 .npy 格式
        let shape = nd.data.shape().to_vec();
        let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        let header = format!(
            "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}), }}",
            shape_str.join(", ")
        );
        let header_len = header.len();
        let padded_len = (header_len + 10).div_ceil(64) * 64 - 10;
        let data_size: usize = nd.data.len();
        let mut buf: Vec<u8> = Vec::with_capacity(10 + padded_len + data_size * 8);
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        let padded_header = format!("{:width$}", header, width = padded_len);

        let header_len_u16 = padded_len as u16;
        buf.extend_from_slice(&header_len_u16.to_le_bytes());
        buf.extend_from_slice(padded_header.as_bytes());

        for &v in nd.data.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        zip.start_file(&entry_name, options)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        zip.write_all(&buf)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }

    zip.finish()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(())
}

// ========== load_npz: 从 zip 加载多个数组 ==========
// 返回 Vec<(String, NdArray)>
#[pyfunction]
fn load_npz(filename: &str) -> PyResult<Vec<(String, NdArray)>> {
    use std::fs::File;
    use std::io::Read;

    let file = File::open(filename)
        .map_err(|e| PyValueError::new_err(format!("Failed to open file: {}", e)))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut results: Vec<(String, NdArray)> = Vec::with_capacity(zip.len());

    for i in 0..zip.len() {
        let mut entry = zip
            .by_index(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let name = entry.name().to_string();
        if !name.ends_with(".npy") {
            continue;
        }
        let key = name.trim_end_matches(".npy").to_string();

        let mut contents = Vec::with_capacity(entry.size() as usize);
        entry
            .read_to_end(&mut contents)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // 解析 .npy
        if contents.len() < 10 || &contents[0..6] != b"\x93NUMPY" {
            continue;
        }
        let header_len = u16::from_le_bytes([contents[8], contents[9]]) as usize;
        let header_start = 10;
        let header_end = header_start + header_len;
        if contents.len() < header_end {
            continue;
        }
        let header_str = String::from_utf8_lossy(&contents[header_start..header_end]);
        let shape: Vec<usize> = if let Some(start) = header_str.find("shape") {
            let rest = &header_str[start + 6..];
            if let Some(open) = rest.find('(') {
                if let Some(close) = rest.find(')') {
                    let inside = &rest[open + 1..close];
                    if inside.trim().is_empty() {
                        vec![]
                    } else {
                        inside
                            .split(',')
                            .filter_map(|s| s.trim().parse::<usize>().ok())
                            .collect()
                    }
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let data_start = header_end;
        let total: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let mut values = vec![0.0f64; total];
        for (j, v) in values.iter_mut().enumerate() {
            let start = data_start + j * 8;
            if start + 8 > contents.len() {
                break;
            }
            let bytes = [
                contents[start],
                contents[start + 1],
                contents[start + 2],
                contents[start + 3],
                contents[start + 4],
                contents[start + 5],
                contents[start + 6],
                contents[start + 7],
            ];
            *v = f64::from_le_bytes(bytes);
        }

        let arr_shape = if shape.is_empty() {
            IxDyn(&[])
        } else {
            IxDyn(&shape)
        };
        let arr = Array::from_shape_vec(arr_shape, values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nd = NdArray { data: arr };
        results.push((key, nd));
    }

    Ok(results)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_npy, m)?)?;
    m.add_function(wrap_pyfunction!(load_npy, m)?)?;
    m.add_function(wrap_pyfunction!(save_text, m)?)?;
    m.add_function(wrap_pyfunction!(load_text, m)?)?;
    m.add_function(wrap_pyfunction!(savez_npz, m)?)?;
    m.add_function(wrap_pyfunction!(load_npz, m)?)?;
    Ok(())
}

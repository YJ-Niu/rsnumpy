use crate::*;
use std::fmt::Write;

fn format_complex_scalar(val: f64) -> String {
    if val.is_nan() {
        return "nan+nanj".to_string();
    }
    if val.is_infinite() {
        let sign = if val > 0.0 { "" } else { "-" };
        return format!("{}inf+0.j", sign);
    }
    // 整数不显示小数点
    let real_str = if val == val.floor() && val.is_finite() && val.abs() < 1e16 {
        let v = val as i64;
        if v as f64 == val {
            format!("{}.", v)
        } else {
            format!("{}", val)
        }
    } else {
        format!("{}", val)
    };
    format!("{}+0.j", real_str)
}

fn format_complex_array_inner(arr: &Array<f64, IxDyn>, pad_width: usize) -> String {
    if arr.ndim() == 0 {
        return format_complex_scalar(arr.iter().next().copied().unwrap_or(0.0_f64));
    }
    if arr.ndim() == 1 {
        let mut s = String::from("[");
        for (i, val) in arr.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            let val_str = format_complex_scalar(*val);
            if pad_width > 0 {
                s.push_str(&format!("{:>width$}", val_str, width = pad_width));
            } else {
                s.push_str(&val_str);
            }
        }
        s.push(']');
        return s;
    }
    let mut s = String::from("[");
    let n = arr.shape()[0];
    for i in 0..n {
        if i > 0 {
            s.push_str("\n ");
        }
        let sub = arr.index_axis(Axis(0), i).to_owned().into_dyn();
        let row_str = format_complex_array_inner(&sub, pad_width);
        s.push_str(&row_str);
    }
    s.push(']');
    s
}

fn format_complex_array(arr: &Array<f64, IxDyn>) -> String {
    let pad_width = if arr.ndim() >= 2 {
        arr.iter()
            .map(|v| format_complex_scalar(*v).len())
            .max()
            .unwrap_or(1)
    } else {
        0
    };
    format_complex_array_inner(arr, pad_width)
}

#[pyfunction]
fn _format_complex_repr(arr: &NdArray) -> String {
    let inner = format_complex_array(&arr.data);
    format!("rsnumpy.ndarray({}) dtype=complex128", inner)
}

#[pyfunction]
fn _format_complex_str(arr: &NdArray) -> String {
    format_complex_array(&arr.data)
}

fn format_int_scalar(val: f64) -> String {
    // 位重解释：将 f64 的位模式当作 i64 显示。
    // 对于 empty 未初始化内存，匹配 numpy 的垃圾值显示。
    let int_val = val.to_bits() as i64;
    format!("{}", int_val)
}

fn format_int_array_inner(arr: &Array<f64, IxDyn>, pad_width: usize) -> String {
    if arr.ndim() == 0 {
        return format_int_scalar(arr.iter().next().copied().unwrap_or(0.0_f64));
    }
    if arr.ndim() == 1 {
        let mut s = String::from("[");
        for (i, val) in arr.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            let val_str = format_int_scalar(*val);
            if pad_width > 0 {
                let _ = write!(s, "{:>pad_width$}", val_str);
            } else {
                s.push_str(&val_str);
            }
        }
        s.push(']');
        return s;
    }
    let mut s = String::from("[");
    let n = arr.shape()[0];
    for i in 0..n {
        if i > 0 {
            s.push_str("\n ");
        }
        let sub = arr.index_axis(Axis(0), i).to_owned().into_dyn();
        let row_str = format_int_array_inner(&sub, pad_width);
        s.push_str(&row_str);
    }
    s.push(']');
    s
}

fn format_int_array(arr: &Array<f64, IxDyn>) -> String {
    let pad_width = if arr.ndim() >= 2 {
        arr.iter()
            .map(|v| format_int_scalar(*v).len())
            .max()
            .unwrap_or(1)
    } else {
        0
    };
    format_int_array_inner(arr, pad_width)
}

#[pyfunction]
fn _format_int_repr(arr: &NdArray) -> String {
    let inner = format_int_array(&arr.data);
    format!("rsnumpy.ndarray({inner}) dtype=int64")
}

#[pyfunction]
fn _format_int_str(arr: &NdArray) -> String {
    format_int_array(&arr.data)
}

fn compute_max_width_float(arr: &Array<f64, IxDyn>) -> usize {
    arr.iter()
        .map(|v| format_float_scalar(*v).len())
        .max()
        .unwrap_or(1)
}

fn format_float_array_inner(arr: &Array<f64, IxDyn>, pad_width: usize) -> String {
    if arr.ndim() == 0 {
        return format_float_scalar(arr.iter().next().copied().unwrap_or(0.0_f64));
    }
    if arr.ndim() == 1 {
        let mut s = String::from("[");
        for (i, val) in arr.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            let val_str = format_float_scalar(*val);
            if pad_width > 0 {
                let _ = write!(s, "{:>pad_width$}", val_str);
            } else {
                s.push_str(&val_str);
            }
        }
        s.push(']');
        return s;
    }
    let mut s = String::from("[");
    let n = arr.shape()[0];
    for i in 0..n {
        if i > 0 {
            s.push_str("\n ");
        }
        let sub = arr.index_axis(Axis(0), i).to_owned().into_dyn();
        let row_str = format_float_array_inner(&sub, pad_width);
        s.push_str(&row_str);
    }
    s.push(']');
    s
}

fn format_float_array(arr: &Array<f64, IxDyn>) -> String {
    let pad_width = if arr.ndim() >= 2 {
        compute_max_width_float(arr)
    } else {
        0
    };
    format_float_array_inner(arr, pad_width)
}

#[pyfunction]
fn _format_float_repr(arr: &NdArray) -> String {
    let inner = format_float_array(&arr.data);
    format!("rsnumpy.ndarray({inner}) dtype=float64")
}

#[pyfunction]
fn _format_float_str(arr: &NdArray) -> String {
    format_float_array(&arr.data)
}

#[pyfunction]
fn _format_int_val_repr(arr: &NdArray) -> String {
    let inner = format_array_repr(&arr.data, "");
    format!("rsnumpy.ndarray({inner}) dtype=int64")
}

#[pyfunction]
fn _format_int_val_str(arr: &NdArray) -> String {
    format_array_repr(&arr.data, "")
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_format_complex_repr, m)?)?;
    m.add_function(wrap_pyfunction!(_format_complex_str, m)?)?;
    m.add_function(wrap_pyfunction!(_format_int_repr, m)?)?;
    m.add_function(wrap_pyfunction!(_format_int_str, m)?)?;
    m.add_function(wrap_pyfunction!(_format_float_repr, m)?)?;
    m.add_function(wrap_pyfunction!(_format_float_str, m)?)?;
    m.add_function(wrap_pyfunction!(_format_int_val_repr, m)?)?;
    m.add_function(wrap_pyfunction!(_format_int_val_str, m)?)?;
    Ok(())
}

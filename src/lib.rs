pub(crate) use ndarray::{Array, ArrayViewD, Axis, IxDyn, Slice, Zip};
pub(crate) use pyo3::exceptions::{PyBufferError, PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
pub(crate) use pyo3::types::{
    PyBool, PyBytes, PyComplex, PyDict, PyFloat, PyInt, PyList, PySlice, PyTuple,
};
pub(crate) use rayon::prelude::*;
use std::fmt::Write;

// macOS：强制链接 Accelerate BLAS 后端（否则符号会被优化掉）。
#[cfg(target_os = "macos")]
use blas_src as _;

mod arithmetic;
mod bitwise;
mod buffer;
mod creation;
mod fft;
mod formatting;
mod indexing;
mod io;
mod linalg;
mod logic;
mod manipulation;
mod mathematics;
mod poly;
mod random;
mod searching;
mod setops;
mod sorting;
mod statistics;

pub(crate) fn parse_py_list_to_flat(data: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    if let Ok(val) = data.extract::<f64>() {
        return Ok((vec![val], vec![]));
    }
    if let Ok(list) = data.cast::<PyList>() {
        let n = list.len();
        if n == 0 {
            return Ok((vec![], vec![0]));
        }
        // 快速路径：扁平标量列表 —— 避免为每个元素分配小 Vec。
        let mut flat = Vec::with_capacity(n);
        let mut all_scalar = true;
        for item in list.iter() {
            match item.extract::<f64>() {
                Ok(v) => flat.push(v),
                Err(_) => {
                    all_scalar = false;
                    break;
                }
            }
        }
        if all_scalar {
            return Ok((flat, vec![n]));
        }
        // 通用路径：嵌套结构。
        let mut all_values = Vec::with_capacity(n);
        let mut child_shapes: Vec<Vec<usize>> = Vec::with_capacity(n);
        for item in list.iter() {
            let (vals, shape) = parse_py_list_to_flat(&item)?;
            all_values.extend(vals);
            child_shapes.push(shape);
        }
        let first_shape = &child_shapes[0];
        for shape in &child_shapes {
            if *shape != *first_shape {
                return Err(PyValueError::new_err(
                    "All sub-arrays must have the same shape",
                ));
            }
        }
        let mut shape = vec![list.len()];
        if !first_shape.is_empty() && first_shape[0] != 0 {
            shape.extend(first_shape);
        } else if first_shape.len() == 1 && first_shape[0] == 0 {
            shape.push(0);
        }
        Ok((all_values, shape))
    } else if let Ok(val) = data.extract::<i32>() {
        Ok((vec![val as f64], vec![]))
    } else if let Ok(listed) = data.call_method0("tolist") {
        // 数组类对象（如 numpy.ndarray / numpy 标量）：先转为 Python list 再解析。
        parse_py_list_to_flat(&listed)
    } else {
        Err(PyTypeError::new_err("Unsupported data type"))
    }
}

/// 元素类型标记：跟踪展平数据中出现过的 Python 标量类别。
#[derive(Clone, Copy, Default)]
struct TypeFlags {
    has_float: bool,
    has_int: bool,
    has_bool: bool,
}

impl TypeFlags {
    fn merge(&mut self, o: TypeFlags) {
        self.has_float |= o.has_float;
        self.has_int |= o.has_int;
        self.has_bool |= o.has_bool;
    }

    /// 与 Python 侧 `_infer_int_dtype` 语义一致的 dtype 编码：
    /// 0 = float64，1 = int64，2 = bool。
    fn dtype_code(&self) -> u8 {
        if self.has_float {
            0
        } else if self.has_bool && !self.has_int {
            2
        } else {
            1
        }
    }
}

/// 递归解析 Python 序列（list/tuple）为扁平 f64 数据、形状和类型标记。
/// 遇到非数值元素（字符串/复数等）返回 TypeError，遇到不规则嵌套返回 ValueError，
/// 由 Python 侧回退到通用构造路径。
fn parse_py_categorized(data: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>, TypeFlags)> {
    // bool 是 int 的子类，必须先判定 bool。
    if let Ok(b) = data.cast::<PyBool>() {
        let v = if b.is_true() { 1.0 } else { 0.0 };
        return Ok((
            vec![v],
            vec![],
            TypeFlags {
                has_bool: true,
                ..Default::default()
            },
        ));
    }
    if data.cast::<PyInt>().is_ok() {
        let v: f64 = data.extract()?;
        return Ok((
            vec![v],
            vec![],
            TypeFlags {
                has_int: true,
                ..Default::default()
            },
        ));
    }
    if data.cast::<PyFloat>().is_ok() {
        let v: f64 = data.extract()?;
        return Ok((
            vec![v],
            vec![],
            TypeFlags {
                has_float: true,
                ..Default::default()
            },
        ));
    }
    let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = data.cast::<PyList>() {
        list.iter().collect()
    } else if let Ok(tuple) = data.cast::<PyTuple>() {
        tuple.iter().collect()
    } else if let Ok(listed) = data.call_method0("tolist") {
        // 数组类对象（如 numpy.ndarray / numpy 标量）：转为 Python list 后再解析，
        // 由 tolist 产出的 int/float/bool 保留原类别以正确推导 dtype。
        return parse_py_categorized(&listed);
    } else {
        return Err(PyTypeError::new_err("non-numeric element"));
    };
    let n = items.len();
    if n == 0 {
        return Ok((vec![], vec![0], TypeFlags::default()));
    }
    // 快速路径：扁平标量序列。
    let mut flat = Vec::with_capacity(n);
    let mut flags = TypeFlags::default();
    let mut all_scalar = true;
    for item in &items {
        if let Ok(b) = item.cast::<PyBool>() {
            flat.push(if b.is_true() { 1.0 } else { 0.0 });
            flags.has_bool = true;
        } else if item.cast::<PyInt>().is_ok() {
            flat.push(item.extract::<f64>()?);
            flags.has_int = true;
        } else if item.cast::<PyFloat>().is_ok() {
            flat.push(item.extract::<f64>()?);
            flags.has_float = true;
        } else {
            all_scalar = false;
            break;
        }
    }
    if all_scalar {
        return Ok((flat, vec![n], flags));
    }
    // 通用路径：嵌套序列。
    let mut all_values = Vec::with_capacity(n);
    let mut child_shapes: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut merged = TypeFlags::default();
    for item in &items {
        let (vals, shape, f) = parse_py_categorized(item)?;
        all_values.extend(vals);
        child_shapes.push(shape);
        merged.merge(f);
    }
    let first_shape = &child_shapes[0];
    for shape in &child_shapes {
        if *shape != *first_shape {
            return Err(PyValueError::new_err(
                "All sub-arrays must have the same shape",
            ));
        }
    }
    let mut shape = vec![n];
    if !first_shape.is_empty() && first_shape[0] != 0 {
        shape.extend(first_shape);
    } else if first_shape.len() == 1 && first_shape[0] == 0 {
        shape.push(0);
    }
    Ok((all_values, shape, merged))
}

/// 递归解析可能包含复数的 Python 序列，返回 (实部扁平, 虚部扁平, 形状, 是否含复数)。
/// 实部/虚部等长（实数元素的虚部填 0）。numpy 复数数组通过 `tolist()` 回退处理。
fn parse_py_complex(data: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>, bool)> {
    // 标量实数（int/float/bool 均可 extract 为 f64）。
    if let Ok(v) = data.extract::<f64>() {
        return Ok((vec![v], vec![0.0], vec![], false));
    }
    // 标量复数。
    if let Ok(c) = data.cast::<PyComplex>() {
        return Ok((vec![c.real()], vec![c.imag()], vec![], true));
    }
    let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = data.cast::<PyList>() {
        list.iter().collect()
    } else if let Ok(tuple) = data.cast::<PyTuple>() {
        tuple.iter().collect()
    } else if let Ok(listed) = data.call_method0("tolist") {
        // numpy 数组 / 标量等数组类对象：转为 Python 原生对象后再解析。
        return parse_py_complex(&listed);
    } else {
        return Err(PyTypeError::new_err("Unsupported data type"));
    };
    let n = items.len();
    if n == 0 {
        return Ok((vec![], vec![], vec![0], false));
    }
    let mut re_all = Vec::with_capacity(n);
    let mut im_all = Vec::with_capacity(n);
    let mut child_shapes: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut any_complex = false;
    for item in &items {
        let (re, im, shape, c) = parse_py_complex(item)?;
        re_all.extend(re);
        im_all.extend(im);
        child_shapes.push(shape);
        any_complex |= c;
    }
    let first_shape = &child_shapes[0];
    for shape in &child_shapes {
        if *shape != *first_shape {
            return Err(PyValueError::new_err(
                "All sub-arrays must have the same shape",
            ));
        }
    }
    let mut shape = vec![n];
    if !first_shape.is_empty() && first_shape[0] != 0 {
        shape.extend(first_shape);
    } else if first_shape.len() == 1 && first_shape[0] == 0 {
        shape.push(0);
    }
    Ok((re_all, im_all, shape, any_complex))
}

fn shape_to_vec(shape: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(tup) = shape.cast::<PyTuple>() {
        let mut result = Vec::with_capacity(tup.len());
        for item in tup.iter() {
            result.push(item.extract::<usize>()?);
        }
        return Ok(result);
    }
    if let Ok(val) = shape.extract::<usize>() {
        return Ok(vec![val]);
    }
    if let Ok(list) = shape.cast::<PyList>() {
        let mut result = Vec::with_capacity(list.len());
        for item in list.iter() {
            result.push(item.extract::<usize>()?);
        }
        return Ok(result);
    }
    Err(PyTypeError::new_err("Shape must be a tuple or integer"))
}

fn value_to_pyobject(py: Python<'_>, val: f64) -> Bound<'_, PyAny> {
    PyFloat::new(py, val).into_any()
}

/// 标量转 Python 对象：虚部存在返回 Python complex，否则返回 float。
fn scalar_to_py(py: Python<'_>, re: f64, im: Option<f64>) -> Bound<'_, PyAny> {
    match im {
        Some(i) => PyComplex::from_doubles(py, re, i).into_any(),
        None => value_to_pyobject(py, re),
    }
}

/// 将赋值右值解析为 NdArray（支持实数标量、复数标量、NdArray、嵌套复数列表）。
fn coerce_value_to_nd(value: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(v) = value.extract::<f64>() {
        return Ok(NdArray::from_real(Array::from_elem(IxDyn(&[]), v)));
    }
    if let Ok(c) = value.cast::<PyComplex>() {
        return Ok(NdArray::from_complex(
            Array::from_elem(IxDyn(&[]), c.real()),
            Array::from_elem(IxDyn(&[]), c.imag()),
        ));
    }
    if let Ok(arr) = value.extract::<NdArray>() {
        return Ok(arr);
    }
    let (re, im, shape, has_c) = parse_py_complex(value)?;
    let data = Array::from_shape_vec(IxDyn(&shape), re)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if has_c {
        let imag = Array::from_shape_vec(IxDyn(&shape), im)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray::from_complex(data, imag))
    } else {
        Ok(NdArray::from_real(data))
    }
}

/// 把 `val` 的前若干元素按 C 序写入 `dst[start..]`（不广播，与原语义一致）。
fn write_row(dst: &mut [f64], start: usize, row_size: usize, val: &Array<f64, IxDyn>) {
    let max_len = val.len().min(row_size);
    if let Some(vs) = val.as_slice() {
        dst[start..start + max_len].copy_from_slice(&vs[..max_len]);
    } else {
        for (j, v) in val.iter().take(max_len).enumerate() {
            dst[start + j] = *v;
        }
    }
}

fn vec_f64_to_pylist<'a>(py: Python<'a>, vec: &[f64]) -> Bound<'a, PyList> {
    let list = PyList::empty(py);
    for v in vec {
        list.append(PyFloat::new(py, *v)).unwrap();
    }
    list
}

fn vec_usize_to_pytuple<'a>(py: Python<'a>, vec: &[usize]) -> Bound<'a, PyTuple> {
    let items: Vec<Bound<'a, PyAny>> = vec
        .iter()
        .map(|&v| pyo3::IntoPyObject::into_pyobject(v, py).unwrap().into_any())
        .collect();
    PyTuple::new(py, &items).unwrap()
}

fn to_python_list<'a>(py: Python<'a>, arr: &ArrayViewD<'_, f64>) -> PyResult<Bound<'a, PyAny>> {
    if arr.ndim() == 0 {
        return Ok(value_to_pyobject(
            py,
            arr.iter().next().copied().unwrap_or(0.0_f64),
        ));
    }
    if arr.ndim() == 1 {
        let vec: Vec<f64> = arr.iter().copied().collect();
        return Ok(vec_f64_to_pylist(py, &vec).into_any());
    }
    let list = PyList::empty(py);
    for i in 0..arr.shape()[0] {
        let sub = arr.index_axis(Axis(0), i);
        let item = to_python_list(py, &sub.into_dyn())?;
        list.append(item)?;
    }
    Ok(list.into_any())
}

fn to_python_list_complex<'a>(
    py: Python<'a>,
    re: &ArrayViewD<'_, f64>,
    im: &ArrayViewD<'_, f64>,
) -> PyResult<Bound<'a, PyAny>> {
    if re.ndim() == 0 {
        let r = re.iter().next().copied().unwrap_or(0.0_f64);
        let i = im.iter().next().copied().unwrap_or(0.0_f64);
        return Ok(PyComplex::from_doubles(py, r, i).into_any());
    }
    if re.ndim() == 1 {
        let list = PyList::empty(py);
        for (r, i) in re.iter().zip(im.iter()) {
            list.append(PyComplex::from_doubles(py, *r, *i))?;
        }
        return Ok(list.into_any());
    }
    let list = PyList::empty(py);
    for k in 0..re.shape()[0] {
        let sub_re = re.index_axis(Axis(0), k);
        let sub_im = im.index_axis(Axis(0), k);
        list.append(to_python_list_complex(
            py,
            &sub_re.into_dyn(),
            &sub_im.into_dyn(),
        )?)?;
    }
    Ok(list.into_any())
}

fn compute_max_width(arr: &Array<f64, IxDyn>) -> usize {
    arr.iter()
        .map(|v| format_scalar(*v).len())
        .max()
        .unwrap_or(1)
}

fn format_array_repr(arr: &Array<f64, IxDyn>, _prefix: &str) -> String {
    if arr.ndim() == 0 {
        return format_scalar(arr.iter().next().copied().unwrap_or(0.0_f64));
    }
    let pad_width = compute_max_width(arr);
    let render = move |v: f64| {
        let s = format_scalar(v);
        format!("{s:>pad_width$}")
    };
    crate::formatting::wrap_recurse(arr, " ", crate::formatting::LINE_WIDTH, &render, " ")
}

fn format_scalar(val: f64) -> String {
    if val.is_nan() {
        return "nan".to_string();
    }
    if val.is_infinite() {
        return if val > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if val == val.floor() && val.is_finite() && val.abs() < 1e16 {
        let v = val as i64;
        if v as f64 == val {
            return format!("{}", v);
        }
    }
    format!("{:.8}", val)
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn format_float_scalar(val: f64) -> String {
    if val.is_nan() {
        return "nan".to_string();
    }
    if val.is_infinite() {
        return if val > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if val == 0.0 {
        // 保留负零符号，与 numpy 一致（例如 [-0.  0.  1.]）。
        return if val.is_sign_negative() {
            "-0.".to_string()
        } else {
            "0.".to_string()
        };
    }
    if val != 0.0 && val.abs() < 1e-10 {
        return format!("{:.10e}", val);
    }
    let val_rounded = (val * 1e8).round() / 1e8;
    if val_rounded == val_rounded.floor() && val_rounded.is_finite() && val_rounded.abs() < 1e16 {
        let v = val_rounded as i64;
        if v as f64 == val_rounded {
            return format!("{}.", v);
        }
    }
    if val_rounded.abs() >= 1e10 {
        return format!("{:.10e}", val_rounded);
    }
    let s = format!("{:.8}", val_rounded);
    if let Some(pos) = s.find('.') {
        s[..pos + 1 + s[pos + 1..].trim_end_matches('0').len()].to_string()
    } else {
        s
    }
}

/// NdArray: 类似 NumPy ndarray 的多维数组
///
/// 复数支持：`data` 恒为实部（保留形状）；`imag` 为 `Some(虚部)`（与 `data` 同形状）
/// 时表示复数数组，为 `None` 时表示实数数组（向后兼容，原有算子只需补 `imag: None`）。
#[pyclass(name = "ndarray", from_py_object)]
#[derive(Clone)]
pub struct NdArray {
    imag: Option<Array<f64, IxDyn>>,
    data: Array<f64, IxDyn>,
}

impl NdArray {
    /// 构造实数数组（虚部为空）。
    #[inline]
    pub(crate) fn from_real(data: Array<f64, IxDyn>) -> Self {
        NdArray { imag: None, data }
    }

    /// 构造复数数组（实部 + 虚部，需同形状）。
    #[inline]
    pub(crate) fn from_complex(re: Array<f64, IxDyn>, im: Array<f64, IxDyn>) -> Self {
        NdArray {
            imag: Some(im),
            data: re,
        }
    }

    /// 是否为复数数组。
    #[inline]
    pub(crate) fn has_imag(&self) -> bool {
        self.imag.is_some()
    }
}

/// 缓冲协议导出期间随 Py_buffer 存活的 shape/strides（字节步长）。
/// 在 `__getbuffer__` 中装箱、指针写入 view.internal，`__releasebuffer__` 中释放。
struct BufferMeta {
    shape: Vec<pyo3::ffi::Py_ssize_t>,
    strides: Vec<pyo3::ffi::Py_ssize_t>,
}

#[pymethods]
impl NdArray {
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // 优先走原有实数快速路径（含扁平标量优化）；仅当其失败（可能含复数）时才走复数解析。
        if let Ok((values, shape)) = parse_py_list_to_flat(data) {
            let arr = Array::from_shape_vec(IxDyn(&shape), values)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(NdArray::from_real(arr));
        }
        let (re, im, shape, has_complex) = parse_py_complex(data)?;
        let dims = IxDyn(&shape);
        let re_arr = Array::from_shape_vec(dims.clone(), re)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if has_complex {
            let im_arr = Array::from_shape_vec(dims, im)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray::from_complex(re_arr, im_arr))
        } else {
            Ok(NdArray::from_real(re_arr))
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        let arr_str = format_array_repr(&self.data, "");
        Ok(format!("rsnumpy.ndarray({}) dtype=float64", arr_str))
    }

    fn __str__(&self) -> PyResult<String> {
        let arr_str = format_array_repr(&self.data, "");
        Ok(arr_str)
    }

    fn __len__(&self) -> PyResult<usize> {
        if self.data.ndim() == 0 {
            return Err(PyTypeError::new_err("len() of unsized object"));
        }
        Ok(self.data.shape()[0])
    }

    fn __getitem__<'a>(
        slf: &'a Bound<'a, Self>,
        index: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let py = slf.py();
        let borrowed = slf.borrow();
        let data = &borrowed.data;
        let imag = borrowed.imag.as_ref();

        if let Ok(idx) = index.extract::<isize>() {
            if data.ndim() == 0 {
                return Ok(scalar_to_py(
                    py,
                    data.iter().next().copied().unwrap_or(0.0_f64),
                    imag.map(|im| im.iter().next().copied().unwrap_or(0.0_f64)),
                ));
            }
            let dim = data.shape()[0] as isize;
            let actual = if idx < 0 { dim + idx } else { idx };
            if actual < 0 || actual >= dim {
                return Err(PyIndexError::new_err("Index out of bounds"));
            }
            let view = data.index_axis(Axis(0), actual as usize);
            let imag_view = imag.map(|im| im.index_axis(Axis(0), actual as usize));
            if view.ndim() == 0 {
                return Ok(scalar_to_py(
                    py,
                    view.iter().next().copied().unwrap_or(0.0_f64),
                    imag_view.map(|iv| iv.iter().next().copied().unwrap_or(0.0_f64)),
                ));
            }
            let sub = NdArray {
                imag: imag_view.map(|iv| iv.to_owned().into_dyn()),
                data: view.to_owned().into_dyn(),
            };
            let bound = Bound::new(py, sub)?;
            return Ok(bound.into_any());
        }

        if let Ok(list) = index.cast::<PyList>() {
            let mut bools = Vec::with_capacity(list.len());
            for item in list.iter() {
                if let Ok(b) = item.extract::<bool>() {
                    bools.push(b);
                } else {
                    return Err(PyTypeError::new_err(
                        "Boolean index list must contain only booleans",
                    ));
                }
            }
            let flat_data: Vec<f64> = data.iter().copied().collect();
            let flat_imag: Option<Vec<f64>> = imag.map(|im| im.iter().copied().collect());
            let mut result = Vec::with_capacity(bools.len());
            let mut result_im = flat_imag.as_ref().map(|_| Vec::with_capacity(bools.len()));
            for (i, &b) in bools.iter().enumerate() {
                if b && i < flat_data.len() {
                    result.push(flat_data[i]);
                    if let (Some(ri), Some(fi)) = (result_im.as_mut(), flat_imag.as_ref()) {
                        ri.push(fi[i]);
                    }
                }
            }
            let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let imag_arr = match result_im {
                Some(ri) => Some(
                    Array::from_shape_vec(IxDyn(&[ri.len()]), ri)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                None => None,
            };
            let sub = NdArray {
                imag: imag_arr,
                data: arr,
            };
            let bound = Bound::new(py, sub)?;
            return Ok(bound.into_any());
        }

        if let Ok(arr) = index.extract::<NdArray>() {
            let flat_data: Vec<f64> = data.iter().copied().collect();
            let flat_imag: Option<Vec<f64>> = imag.map(|im| im.iter().copied().collect());
            let mut result = Vec::with_capacity(flat_data.len());
            let mut result_im = flat_imag
                .as_ref()
                .map(|_| Vec::with_capacity(flat_data.len()));
            for (i, &v) in arr.data.iter().enumerate() {
                if v != 0.0 && i < flat_data.len() {
                    result.push(flat_data[i]);
                    if let (Some(ri), Some(fi)) = (result_im.as_mut(), flat_imag.as_ref()) {
                        ri.push(fi[i]);
                    }
                }
            }
            let arr_result = Array::from_shape_vec(IxDyn(&[result.len()]), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let imag_arr = match result_im {
                Some(ri) => Some(
                    Array::from_shape_vec(IxDyn(&[ri.len()]), ri)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                None => None,
            };
            let sub = NdArray {
                imag: imag_arr,
                data: arr_result,
            };
            let bound = Bound::new(py, sub)?;
            return Ok(bound.into_any());
        }

        Err(PyTypeError::new_err("Unsupported index type"))
    }

    fn __getitem_slice__(&self, py_slice: &Bound<'_, PySlice>) -> PyResult<NdArray> {
        let dim = self.data.shape()[0] as i64;
        let indices = py_slice
            .call_method1("indices", (dim,))?
            .extract::<(i64, i64, i64)>()?;
        let (start, stop, step) = indices;
        let nd_slice = Slice::new(start as isize, Some(stop as isize), step as isize);
        let view = self
            .data
            .slice_axis(Axis(0), nd_slice)
            .to_owned()
            .into_dyn();
        let imag = self
            .imag
            .as_ref()
            .map(|im| im.slice_axis(Axis(0), nd_slice).to_owned().into_dyn());
        Ok(NdArray { imag, data: view })
    }

    fn __setitem__(&mut self, index: isize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.data.ndim() == 0 {
            return Err(PyIndexError::new_err(
                "Scalar array does not support item assignment",
            ));
        }
        let val_nd = coerce_value_to_nd(value)?;

        let dim = self.data.shape()[0] as isize;
        let actual = if index < 0 { dim + index } else { index };
        if actual < 0 || actual >= dim {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        let row_size = self.data.len() / self.data.shape()[0];
        let start = actual as usize * row_size;

        // 右值为复数而自身为实数时，先分配零虚部完成升级。
        if val_nd.imag.is_some() && self.imag.is_none() {
            self.imag = Some(Array::zeros(self.data.raw_dim()));
        }

        {
            let data_slice = self
                .data
                .as_slice_mut()
                .ok_or_else(|| PyValueError::new_err("Non-contiguous array"))?;
            write_row(data_slice, start, row_size, &val_nd.data);
        }
        if let Some(self_im) = self.imag.as_mut() {
            let im_slice = self_im
                .as_slice_mut()
                .ok_or_else(|| PyValueError::new_err("Non-contiguous array"))?;
            match &val_nd.imag {
                Some(vi) => write_row(im_slice, start, row_size, vi),
                None => im_slice[start..start + row_size].fill(0.0),
            }
        }
        Ok(())
    }

    fn __setitem_slice__(
        &mut self,
        py_slice: &Bound<'_, PySlice>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let val_nd = coerce_value_to_nd(value)?;

        let dim = self.data.shape()[0] as i64;
        let indices = py_slice
            .call_method1("indices", (dim,))?
            .extract::<(i64, i64, i64)>()?;
        let (start, stop, step) = indices;
        let nd_slice = Slice::new(start as isize, Some(stop as isize), step as isize);

        if val_nd.imag.is_some() && self.imag.is_none() {
            self.imag = Some(Array::zeros(self.data.raw_dim()));
        }

        {
            let mut slice_view = self.data.slice_axis_mut(Axis(0), nd_slice);
            let bcast = val_nd
                .data
                .broadcast(slice_view.dim())
                .ok_or_else(|| PyValueError::new_err("Cannot broadcast value to slice shape"))?;
            slice_view.assign(&bcast);
        }
        if let Some(self_im) = self.imag.as_mut() {
            let mut slice_view = self_im.slice_axis_mut(Axis(0), nd_slice);
            match &val_nd.imag {
                Some(vi) => {
                    let bcast = vi.broadcast(slice_view.dim()).ok_or_else(|| {
                        PyValueError::new_err("Cannot broadcast value to slice shape")
                    })?;
                    slice_view.assign(&bcast);
                }
                None => slice_view.fill(0.0),
            }
        }
        Ok(())
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<NdArrayIter> {
        let borrowed = slf.borrow();
        let data = borrowed.data.clone();
        let imag = borrowed.imag.clone();
        let ndim = data.ndim();
        let sub_arrays: Vec<NdArray> = if ndim <= 1 {
            match &imag {
                Some(im) => data
                    .iter()
                    .zip(im.iter())
                    .map(|(&r, &i)| {
                        NdArray::from_complex(
                            Array::from_elem(IxDyn(&[]), r),
                            Array::from_elem(IxDyn(&[]), i),
                        )
                    })
                    .collect(),
                None => data
                    .iter()
                    .map(|&v| NdArray::from_real(Array::from_elem(IxDyn(&[]), v)))
                    .collect(),
            }
        } else {
            (0..data.shape()[0])
                .map(|i| {
                    let sub = data.index_axis(Axis(0), i).to_owned().into_dyn();
                    let sub_im = imag
                        .as_ref()
                        .map(|im| im.index_axis(Axis(0), i).to_owned().into_dyn());
                    NdArray {
                        imag: sub_im,
                        data: sub,
                    }
                })
                .collect()
        };
        Ok(NdArrayIter {
            arrays: sub_arrays,
            index: 0,
        })
    }

    #[getter]
    fn shape<'a>(slf: &'a Bound<'a, Self>) -> PyResult<Bound<'a, PyAny>> {
        let py = slf.py();
        let s = slf.borrow().data.shape().to_vec();
        Ok(vec_usize_to_pytuple(py, &s).into_any())
    }

    // NumPy 数组接口协议：底层恒为 f64，暴露为 '<f8' 供真实 numpy/matplotlib 直接消费。
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        build_array_interface(py, &self.data, "<f8")
    }

    // PEP 3118 缓冲协议：直接暴露底层连续 f64 内存（只读）及其 shape/strides，
    // 供 rsplotlib 等下游用 PyO3 `PyBuffer<f64>` 零拷贝读取，免去序列化为 bytes 的开销。
    // 消费方（如 PyBuffer::get）以 PyBUF_FULL_RO 请求，包含 shape/strides，故非连续数组亦可如实描述。
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::ffi::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        // 仅导出只读视图：避免 Python 侧经缓冲写入而与 Rust 所有权产生别名。
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("Object is not writable"));
        }

        let itemsize = std::mem::size_of::<f64>() as pyo3::ffi::Py_ssize_t;
        let (buf_ptr, len_elems, ndim, is_c, shape, strides) = {
            let borrowed = slf.borrow();
            let data = &borrowed.data;
            let shape: Vec<pyo3::ffi::Py_ssize_t> = data
                .shape()
                .iter()
                .map(|&d| d as pyo3::ffi::Py_ssize_t)
                .collect();
            // ndarray 步长以元素为单位，Py_buffer 需字节步长。
            let strides: Vec<pyo3::ffi::Py_ssize_t> = data
                .strides()
                .iter()
                .map(|&s| (s as pyo3::ffi::Py_ssize_t) * itemsize)
                .collect();
            (
                data.as_ptr() as *mut std::ffi::c_void,
                data.len(),
                data.ndim(),
                data.is_standard_layout(),
                shape,
                strides,
            )
        };

        // 消费方不接受 strides 时只能提供 C 连续布局；显式要求 C 连续时同理。
        let wants_strides = (flags & pyo3::ffi::PyBUF_STRIDES) == pyo3::ffi::PyBUF_STRIDES;
        if !wants_strides && !is_c {
            return Err(PyBufferError::new_err(
                "underlying buffer is not C-contiguous",
            ));
        }
        if (flags & pyo3::ffi::PyBUF_C_CONTIGUOUS) == pyo3::ffi::PyBUF_C_CONTIGUOUS && !is_c {
            return Err(PyBufferError::new_err(
                "underlying buffer is not C-contiguous",
            ));
        }

        // shape/strides 需活到 __releasebuffer__：装箱后把指针交给 view，原始指针存入 internal。
        let meta = Box::new(BufferMeta { shape, strides });
        let shape_ptr = meta.shape.as_ptr() as *mut pyo3::ffi::Py_ssize_t;
        let strides_ptr = meta.strides.as_ptr() as *mut pyo3::ffi::Py_ssize_t;
        let meta_ptr = Box::into_raw(meta) as *mut std::ffi::c_void;

        unsafe {
            // 转移一个强引用给 view.obj，确保缓冲存活期间数组（及其内存）不被回收。
            (*view).obj = slf.into_any().into_ptr();
            (*view).buf = buf_ptr;
            (*view).len = (len_elems as pyo3::ffi::Py_ssize_t) * itemsize;
            (*view).readonly = 1;
            (*view).itemsize = itemsize;
            // f64 的格式字符为 "d"；使用静态 C 字符串，CPython 不会释放它。
            (*view).format = if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
                c"d".as_ptr() as *mut std::ffi::c_char
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = ndim as std::ffi::c_int;
            (*view).shape = if (flags & pyo3::ffi::PyBUF_ND) == pyo3::ffi::PyBUF_ND {
                shape_ptr
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if wants_strides {
                strides_ptr
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = meta_ptr;
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        unsafe {
            if !(*view).internal.is_null() {
                drop(Box::from_raw((*view).internal as *mut BufferMeta));
                (*view).internal = std::ptr::null_mut();
            }
        }
    }

    #[getter]
    fn ndim(&self) -> PyResult<usize> {
        Ok(self.data.ndim())
    }

    #[getter]
    fn size(&self) -> PyResult<usize> {
        Ok(self.data.len())
    }

    #[getter]
    fn dtype(&self) -> PyResult<String> {
        Ok(if self.has_imag() {
            "complex128".to_string()
        } else {
            "float64".to_string()
        })
    }

    #[getter]
    fn flags<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, ArrayFlags>> {
        Bound::new(py, ArrayFlags {})
    }

    #[getter]
    #[pyo3(name = "T")]
    fn t(&self) -> PyResult<NdArray> {
        if self.data.ndim() <= 1 {
            return Ok(self.clone());
        }
        let data = self.data.view().reversed_axes().into_owned().into_dyn();
        let imag = self
            .imag
            .as_ref()
            .map(|im| im.view().reversed_axes().into_owned().into_dyn());
        Ok(NdArray { imag, data })
    }

    fn reshape(&self, shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        // 使用 isize 解析，支持 -1
        let s: Vec<isize> = if let Ok(tup) = shape.cast::<PyTuple>() {
            let mut result = Vec::with_capacity(tup.len());
            for item in tup.iter() {
                result.push(item.extract::<isize>()?);
            }
            result
        } else if let Ok(val) = shape.extract::<isize>() {
            vec![val]
        } else if let Ok(list) = shape.cast::<PyList>() {
            let mut result = Vec::with_capacity(list.len());
            for item in list.iter() {
                result.push(item.extract::<isize>()?);
            }
            result
        } else {
            return Err(PyTypeError::new_err(
                "Shape must be a tuple, list, or integer",
            ));
        };

        let total = self.data.len() as isize;

        // 处理 -1（自动计算维度）
        let unknown_count = s.iter().filter(|&&v| v == -1).count();
        if unknown_count > 1 {
            return Err(PyValueError::new_err(
                "can only specify one unknown dimension",
            ));
        }

        let known: isize = s.iter().filter(|&&v| v > 0).product();
        let out: Vec<usize> = if unknown_count == 1 {
            if known == 0 || total % known != 0 {
                return Err(PyValueError::new_err(format!(
                    "cannot reshape array of size {} into shape {:?}",
                    total, s
                )));
            }
            s.iter()
                .map(|&v| -> PyResult<usize> {
                    if v == -1 {
                        Ok((total / known) as usize)
                    } else if v <= 0 {
                        Err(PyValueError::new_err(format!(
                            "{} is not a valid dimension size",
                            v
                        )))
                    } else {
                        Ok(v as usize)
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            s.iter()
                .map(|&v| -> PyResult<usize> {
                    if v <= 0 {
                        Err(PyValueError::new_err(format!(
                            "{} is not a valid dimension size",
                            v
                        )))
                    } else {
                        Ok(v as usize)
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        };

        // 验证总元素数一致
        if out.iter().product::<usize>() != total as usize {
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {} into shape {:?}",
                total, s
            )));
        }

        let arr = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(&out))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let imag = match &self.imag {
            Some(im) => Some(
                im.clone()
                    .into_shape_with_order(IxDyn(&out))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };
        Ok(NdArray { imag, data: arr })
    }

    fn flatten(&self) -> NdArray {
        let flat: Vec<f64> = self.data.iter().copied().collect();
        let data = Array::from_shape_vec(IxDyn(&[flat.len()]), flat).unwrap();
        let imag = self.imag.as_ref().map(|im| {
            let f: Vec<f64> = im.iter().copied().collect();
            Array::from_shape_vec(IxDyn(&[f.len()]), f).unwrap()
        });
        NdArray { imag, data }
    }

    fn ravel(&self) -> NdArray {
        self.flatten()
    }

    fn squeeze(&self) -> PyResult<NdArray> {
        let new_shape: Vec<usize> = self
            .data
            .shape()
            .iter()
            .filter(|&&d| d > 1)
            .copied()
            .collect();
        let arr = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(&new_shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let imag = match &self.imag {
            Some(im) => Some(
                im.clone()
                    .into_shape_with_order(IxDyn(&new_shape))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };
        Ok(NdArray { imag, data: arr })
    }

    fn copy(&self) -> NdArray {
        NdArray {
            imag: self.imag.clone(),
            data: self.data.clone(),
        }
    }

    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.imag {
            Some(im) => {
                to_python_list_complex(py, &self.data.view().into_dyn(), &im.view().into_dyn())
            }
            None => to_python_list(py, &self.data.view().into_dyn()),
        }
    }

    #[pyo3(signature = (repeats, axis=None))]
    fn repeat(&self, repeats: usize, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let values: Vec<f64> = self.data.iter().copied().collect();
                let mut result = Vec::with_capacity(values.len() * repeats);
                for &v in &values {
                    for _ in 0..repeats {
                        result.push(v);
                    }
                }
                let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(NdArray {
                    imag: None,
                    data: arr,
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 {
                    (ndim as isize + ax) as usize
                } else {
                    ax as usize
                };
                let shape = self.data.shape().to_vec();
                let axis_size = shape[ax];
                let pre_size: usize = shape.iter().take(ax).product();
                let post_size: usize = shape.iter().skip(ax + 1).product();
                let block_size = axis_size * post_size;
                let data_vec: Vec<f64> = self.data.iter().copied().collect();
                let mut result = Vec::with_capacity(data_vec.len() * repeats);
                for outer in 0..pre_size {
                    for _ in 0..repeats {
                        for k in 0..axis_size {
                            for inner in 0..post_size {
                                result.push(data_vec[outer * block_size + k * post_size + inner]);
                            }
                        }
                    }
                }
                let mut new_shape = shape;
                new_shape[ax] = axis_size * repeats;
                let arr = Array::from_shape_vec(IxDyn(&new_shape), result)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(NdArray {
                    imag: None,
                    data: arr,
                })
            }
        }
    }

    fn fill(&mut self, value: f64) {
        self.data.mapv_inplace(|_| value);
    }

    fn astype(&self, dtype: &str) -> PyResult<NdArray> {
        // rsnumpy 内部元素类型固定为 f64，这里只支持把 f64 截断为整数（int/int32/int64）。
        // 对于 float* / 其它类型，保持 f64 行为（与旧版一致）。
        let dt = dtype.to_lowercase();
        let is_complex = matches!(
            dt.as_str(),
            "complex" | "complex128" | "complex64" | "c16" | "c8" | "cdouble" | "cfloat"
        );
        if is_complex {
            // 转为复数：已是复数则保留虚部，否则补零虚部。
            let imag = self
                .imag
                .clone()
                .unwrap_or_else(|| Array::zeros(self.data.raw_dim()));
            return Ok(NdArray::from_complex(self.data.clone(), imag));
        }
        let is_int = matches!(
            dt.as_str(),
            "int"
                | "i"
                | "int8"
                | "int16"
                | "int32"
                | "int64"
                | "intp"
                | "i8"
                | "i16"
                | "i32"
                | "i64"
        );
        if is_int {
            let casted: Vec<f64> = self.data.iter().map(|v| v.trunc()).collect();
            let arr = Array::from_shape_vec(IxDyn(self.data.shape()), casted)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray {
                imag: None,
                data: arr,
            })
        } else {
            // float* 或其它：丢弃虚部（与 numpy astype 到实数类型一致）。
            Ok(NdArray {
                imag: None,
                data: self.data.clone(),
            })
        }
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        dispatch_binop(self, other, CBinOp::Add)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        dispatch_binop(self, other, CBinOp::Sub)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        dispatch_binop(self, other, CBinOp::Mul)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        dispatch_binop(self, other, CBinOp::Div)
    }

    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        _mod: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        binary_op(self, other, |a, b| a.powf(b))
    }

    fn __neg__(&self) -> PyResult<NdArray> {
        match &self.imag {
            Some(im) => Ok(NdArray::from_complex(
                self.data.mapv(|x| -x),
                im.mapv(|x| -x),
            )),
            None => Ok(NdArray::from_real(self.data.mapv(|x| -x))),
        }
    }

    fn __abs__(&self) -> PyResult<NdArray> {
        match &self.imag {
            // 复数取模：sqrt(re^2 + im^2)，结果为实数数组。
            Some(im) => {
                let mag = Zip::from(&self.data)
                    .and(im)
                    .map_collect(|&r, &i| (r * r + i * i).sqrt());
                Ok(NdArray::from_real(mag))
            }
            None => Ok(NdArray::from_real(self.data.mapv(|x| x.abs()))),
        }
    }

    /// 实部（复数数组返回实部，实数数组返回自身副本）。
    #[getter]
    fn real(&self) -> NdArray {
        NdArray::from_real(self.data.clone())
    }

    /// 是否为复数数组（存在虚部）。
    #[getter]
    fn is_complex(&self) -> bool {
        self.imag.is_some()
    }

    /// 虚部（复数数组返回虚部，实数数组返回全零数组）。
    #[getter]
    fn imag(&self) -> NdArray {
        match &self.imag {
            Some(im) => NdArray::from_real(im.clone()),
            None => NdArray::from_real(Array::zeros(self.data.raw_dim())),
        }
    }

    /// 共轭（复数取反虚部；实数返回自身副本）。
    fn conj(&self) -> NdArray {
        match &self.imag {
            Some(im) => NdArray::from_complex(self.data.clone(), im.mapv(|x| -x)),
            None => NdArray::from_real(self.data.clone()),
        }
    }

    fn conjugate(&self) -> NdArray {
        self.conj()
    }

    fn __round__(&self, ndigits: Option<i32>) -> PyResult<NdArray> {
        match ndigits {
            None => Ok(NdArray {
                imag: None,
                data: self.data.mapv(|x| x.round()),
            }),
            Some(n) => {
                let factor = 10.0_f64.powi(n);
                Ok(NdArray {
                    imag: None,
                    data: self.data.mapv(|x| (x * factor).round() / factor),
                })
            }
        }
    }

    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op_lr(self, other, |a, b| if a > b { 1.0 } else { 0.0 })
    }

    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op_lr(self, other, |a, b| if a < b { 1.0 } else { 0.0 })
    }

    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op_lr(self, other, |a, b| if a >= b { 1.0 } else { 0.0 })
    }

    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op_lr(self, other, |a, b| if a <= b { 1.0 } else { 0.0 })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op_lr(
            self,
            other,
            |a, b| if (a - b).abs() < 1e-12 { 1.0 } else { 0.0 },
        )
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op_lr(
            self,
            other,
            |a, b| if (a - b).abs() >= 1e-12 { 1.0 } else { 0.0 },
        )
    }

    // ========== 新增缺失方法 ==========

    #[pyo3(signature = (axis=None))]
    fn prod(&self, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let val = self.data.iter().cloned().fold(1.0_f64, |a, b| a * b);
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 {
                    (ndim as isize + ax) as usize
                } else {
                    ax as usize
                };
                let shape = self.data.shape().to_vec();
                let axis_size = shape[ax];
                let pre_size: usize = shape.iter().take(ax).product();
                let post_size: usize = shape.iter().skip(ax + 1).product();
                let block_size = axis_size * post_size;
                let data_vec: Vec<f64> = self.data.iter().copied().collect();
                // 并行计算每个切片的乘积
                let n_slices = pre_size * post_size;
                let results: Vec<f64> = (0..n_slices)
                    .into_par_iter()
                    .map(|idx| {
                        let outer = idx / post_size;
                        let inner = idx % post_size;
                        let base = outer * block_size + inner;
                        let mut prod = 1.0_f64;
                        for k in 0..axis_size {
                            prod *= data_vec[base + k * post_size];
                        }
                        prod
                    })
                    .collect();
                let new_shape: Vec<usize> = shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ax)
                    .map(|(_, &s)| s)
                    .collect();
                let arr = Array::from_shape_vec(IxDyn(&new_shape), results)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(NdArray {
                    imag: None,
                    data: arr,
                })
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn cumsum(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<NdArray> {
        cumulative(py, &self.data, axis, 0.0, |acc, x| acc + x)
    }

    #[pyo3(signature = (axis=None))]
    fn cumprod(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<NdArray> {
        cumulative(py, &self.data, axis, 1.0, |acc, x| acc * x)
    }

    #[pyo3(signature = (offset=0, axis1=0, axis2=1))]
    fn diagonal(&self, offset: isize, axis1: usize, axis2: usize) -> PyResult<NdArray> {
        let ndim = self.data.ndim();
        if ndim < 2 {
            return Err(PyValueError::new_err(
                "diagonal requires at least 2-D array",
            ));
        }
        let shape = self.data.shape().to_vec();
        let d1 = if axis1 < ndim {
            shape[axis1]
        } else {
            return Err(PyValueError::new_err("axis1 out of bounds"));
        };
        let d2 = if axis2 < ndim {
            shape[axis2]
        } else {
            return Err(PyValueError::new_err("axis2 out of bounds"));
        };
        let offset_abs = offset.unsigned_abs();
        let i_start = if offset >= 0 { 0 } else { offset_abs };
        let j_start = if offset >= 0 { offset_abs } else { 0 };
        let n = d1.min(d2).saturating_sub(offset_abs);
        let mut diag = Vec::with_capacity(n);
        for k in 0..n {
            let mut coords = vec![0usize; ndim];
            for (d, coord) in coords.iter_mut().enumerate() {
                if d == axis1 {
                    *coord = i_start + k;
                } else if d == axis2 {
                    *coord = j_start + k;
                }
            }
            diag.push(self.data[IxDyn(&coords)]);
        }
        let arr = Array::from_shape_vec(IxDyn(&[diag.len()]), diag)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }

    #[pyo3(signature = (offset=0, axis1=0, axis2=1))]
    fn trace(&self, offset: isize, axis1: usize, axis2: usize) -> PyResult<f64> {
        let diag = self.diagonal(offset, axis1, axis2)?;
        Ok(diag.data.iter().sum())
    }

    #[pyo3(signature = (*args))]
    fn item(&self, args: &Bound<'_, PyTuple>) -> PyResult<f64> {
        let n = args.len();
        if n == 0 {
            if self.data.len() != 1 {
                return Err(PyValueError::new_err(
                    "item requires exactly one element array when no indices given",
                ));
            }
            return Ok(*self.data.iter().next().unwrap_or(&0.0));
        }
        let shape = self.data.shape();
        let mut coords = vec![0usize; shape.len()];
        for i in 0..n.min(shape.len()) {
            let idx: isize = args.get_item(i)?.extract()?;
            let dim_size = shape[i] as isize;
            let actual = if idx < 0 { dim_size + idx } else { idx };
            if actual < 0 || actual >= dim_size {
                return Err(PyIndexError::new_err("Index out of bounds"));
            }
            coords[i] = actual as usize;
        }
        Ok(self.data[IxDyn(&coords)])
    }

    fn take(&self, indices: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
        let ndim = self.data.ndim();
        let ax = match axis {
            None => {
                // flat take
                let flat: Vec<f64> = self.data.iter().copied().collect();
                let idx_vals: Vec<usize> = indices.data.iter().map(|&v| v as usize).collect();
                let result: Vec<f64> = idx_vals
                    .iter()
                    .map(|&i| *flat.get(i).unwrap_or(&0.0))
                    .collect();
                let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                return Ok(NdArray {
                    imag: None,
                    data: arr,
                });
            }
            Some(ax) => {
                if ax < 0 {
                    (ndim as isize + ax) as usize
                } else {
                    ax as usize
                }
            }
        };
        let shape = self.data.shape().to_vec();
        if ax >= ndim {
            return Err(PyValueError::new_err("Axis out of bounds"));
        }
        let axis_size = shape[ax];
        let pre_size: usize = shape.iter().take(ax).product();
        let post_size: usize = shape.iter().skip(ax + 1).product();
        let block_size = axis_size * post_size;
        let data_vec: Vec<f64> = self.data.iter().copied().collect();
        let idx_vals: Vec<usize> = indices
            .data
            .iter()
            .map(|&v| {
                let idx = v as isize;
                if idx < 0 {
                    (axis_size as isize + idx) as usize
                } else {
                    idx as usize
                }
            })
            .collect();
        let new_axis_size = idx_vals.len();
        let mut result = Vec::with_capacity(pre_size * new_axis_size * post_size);
        for outer in 0..pre_size {
            for &idx in &idx_vals {
                let idx = if idx < axis_size { idx } else { axis_size - 1 };
                for inner in 0..post_size {
                    result.push(data_vec[outer * block_size + idx * post_size + inner]);
                }
            }
        }
        let mut new_shape = shape.clone();
        new_shape[ax] = new_axis_size;
        let arr = Array::from_shape_vec(IxDyn(&new_shape), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }

    fn put(&mut self, indices: &NdArray, values: &NdArray) -> PyResult<()> {
        let flat_len = self.data.len();
        let idx_vals: Vec<usize> = indices
            .data
            .iter()
            .map(|&v| {
                let idx = v as isize;
                if idx < 0 {
                    (flat_len as isize + idx) as usize
                } else {
                    idx as usize
                }
            })
            .collect();
        let val_vec: Vec<f64> = values.data.iter().copied().collect();
        let data_slice = self
            .data
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("put requires contiguous array"))?;
        for (i, &idx) in idx_vals.iter().enumerate() {
            if idx < flat_len {
                data_slice[idx] = val_vec[i % val_vec.len()];
            }
        }
        Ok(())
    }

    fn resize(&mut self, new_shape: &Bound<'_, PyAny>) -> PyResult<()> {
        let s = shape_to_vec(new_shape)?;
        let new_size: usize = s.iter().product();
        let flat_data: Vec<f64> = self.data.iter().copied().collect();
        let mut new_data = Vec::with_capacity(new_size);
        if new_size <= flat_data.len() {
            new_data.extend_from_slice(&flat_data[..new_size]);
        } else {
            new_data.extend_from_slice(&flat_data);
            while new_data.len() < new_size {
                let remaining = new_size - new_data.len();
                let copy_len = flat_data.len().min(remaining);
                new_data.extend_from_slice(&flat_data[..copy_len]);
            }
        }
        let arr = Array::from_shape_vec(IxDyn(&s), new_data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.data = arr;
        Ok(())
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        dispatch_binop_r(self, other, CBinOp::Add)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        dispatch_binop_r(self, other, CBinOp::Sub)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        dispatch_binop_r(self, other, CBinOp::Mul)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        dispatch_binop_r(self, other, CBinOp::Div)
    }

    #[pyo3(signature = (axis=None))]
    fn sum(&self, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let val = self.data.sum();
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 {
                    (ndim as isize + ax) as usize
                } else {
                    ax as usize
                };
                let result = self.data.sum_axis(Axis(ax));
                Ok(NdArray {
                    imag: None,
                    data: result.into_dyn(),
                })
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn mean(&self, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let val = self.data.mean().unwrap_or(0.0);
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 {
                    (ndim as isize + ax) as usize
                } else {
                    ax as usize
                };
                let result = self.data.mean_axis(Axis(ax));
                match result {
                    Some(arr) => Ok(NdArray {
                        imag: None,
                        data: arr.into_dyn(),
                    }),
                    None => Err(PyValueError::new_err("Invalid axis")),
                }
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn std(&self, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let m = self.data.mean().unwrap_or(0.0);
                let var = self.data.mapv(|x| (x - m).powi(2)).mean().unwrap_or(0.0);
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), var.sqrt()),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 {
                    (ndim as isize + ax) as usize
                } else {
                    ax as usize
                };
                let m = self.data.mean_axis(Axis(ax));
                match m {
                    Some(mean_arr) => {
                        let shape = self.data.shape().to_vec();
                        let axis_size = shape[ax];
                        let pre_size: usize = shape.iter().take(ax).product();
                        let post_size: usize = shape.iter().skip(ax + 1).product();
                        let data_vec: Vec<f64> = self.data.iter().copied().collect();

                        // 并行计算每个切片的方差
                        let n_slices = pre_size * post_size;
                        let results: Vec<f64> = (0..n_slices)
                            .into_par_iter()
                            .map(|idx| {
                                let outer = idx / post_size;
                                let inner = idx % post_size;
                                let base = outer * axis_size * post_size + inner;
                                let mean = mean_arr[outer * post_size + inner];
                                let mut sum_sq = 0.0;
                                for k in 0..axis_size {
                                    let diff = data_vec[base + k * post_size] - mean;
                                    sum_sq += diff * diff;
                                }
                                (sum_sq / axis_size as f64).sqrt()
                            })
                            .collect();

                        let new_shape: Vec<usize> = shape
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != ax)
                            .map(|(_, &s)| s)
                            .collect();

                        if new_shape.is_empty() {
                            Ok(NdArray {
                                imag: None,
                                data: Array::from_elem(IxDyn(&[]), results[0]),
                            })
                        } else {
                            let arr = Array::from_shape_vec(IxDyn(&new_shape), results)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                            Ok(NdArray {
                                imag: None,
                                data: arr,
                            })
                        }
                    }
                    None => Err(PyValueError::new_err("Invalid axis")),
                }
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn var(&self, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let m = self.data.mean().unwrap_or(0.0);
                let var = self.data.mapv(|x| (x - m).powi(2)).mean().unwrap_or(0.0);
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), var),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 {
                    (ndim as isize + ax) as usize
                } else {
                    ax as usize
                };
                let m = self.data.mean_axis(Axis(ax));
                match m {
                    Some(mean_arr) => {
                        let shape = self.data.shape().to_vec();
                        let axis_size = shape[ax];
                        let pre_size: usize = shape.iter().take(ax).product();
                        let post_size: usize = shape.iter().skip(ax + 1).product();
                        let data_vec: Vec<f64> = self.data.iter().copied().collect();

                        // 并行计算每个切片的方差
                        let n_slices = pre_size * post_size;
                        let results: Vec<f64> = (0..n_slices)
                            .into_par_iter()
                            .map(|idx| {
                                let outer = idx / post_size;
                                let inner = idx % post_size;
                                let base = outer * axis_size * post_size + inner;
                                let mean = mean_arr[outer * post_size + inner];
                                let mut sum_sq = 0.0;
                                for k in 0..axis_size {
                                    let diff = data_vec[base + k * post_size] - mean;
                                    sum_sq += diff * diff;
                                }
                                sum_sq / axis_size as f64
                            })
                            .collect();

                        let new_shape: Vec<usize> = shape
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != ax)
                            .map(|(_, &s)| s)
                            .collect();

                        if new_shape.is_empty() {
                            Ok(NdArray {
                                imag: None,
                                data: Array::from_elem(IxDyn(&[]), results[0]),
                            })
                        } else {
                            let arr = Array::from_shape_vec(IxDyn(&new_shape), results)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                            Ok(NdArray {
                                imag: None,
                                data: arr,
                            })
                        }
                    }
                    None => Err(PyValueError::new_err("Invalid axis")),
                }
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn min(&self, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let val = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 {
                    (ndim as isize + ax) as usize
                } else {
                    ax as usize
                };
                let result: Array<f64, IxDyn> = self
                    .data
                    .fold_axis(Axis(ax), f64::INFINITY, |acc, &v| {
                        let a = *acc;
                        if v < a { v } else { a }
                    })
                    .into_dyn();
                Ok(NdArray {
                    imag: None,
                    data: result,
                })
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn max(&self, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let val = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                Ok(NdArray {
                    imag: None,
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 {
                    (ndim as isize + ax) as usize
                } else {
                    ax as usize
                };
                let result: Array<f64, IxDyn> = self
                    .data
                    .fold_axis(Axis(ax), f64::NEG_INFINITY, |acc, &v| {
                        let a = *acc;
                        if v > a { v } else { a }
                    })
                    .into_dyn();
                Ok(NdArray {
                    imag: None,
                    data: result,
                })
            }
        }
    }

    fn argmin(&self) -> PyResult<usize> {
        let (idx, _) = self
            .data
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Equal))
            .ok_or_else(|| PyValueError::new_err("Empty array"))?;
        Ok(idx)
    }

    fn argmax(&self) -> PyResult<usize> {
        let (idx, _) = self
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(::std::cmp::Ordering::Equal))
            .ok_or_else(|| PyValueError::new_err("Empty array"))?;
        Ok(idx)
    }

    fn all(&self) -> bool {
        self.data.iter().all(|&v| v != 0.0)
    }

    fn any(&self) -> bool {
        self.data.iter().any(|&v| v != 0.0)
    }

    fn nonzero(&self) -> Vec<Vec<usize>> {
        let shape = self.data.shape();
        let ndim = shape.len();
        if ndim == 0 {
            if self.data.iter().any(|&v| v != 0.0) {
                return vec![vec![0usize]];
            }
            return vec![vec![]];
        }
        let mut result: Vec<Vec<usize>> = (0..ndim).map(|_| Vec::new()).collect();
        for (flat_idx, &val) in self.data.iter().enumerate() {
            if val != 0.0 {
                let mut remaining = flat_idx;
                for d in (0..ndim).rev() {
                    let dim_size = shape[d];
                    result[d].push(remaining % dim_size);
                    remaining /= dim_size;
                }
            }
        }
        result
    }

    fn clip(&self, a_min: f64, a_max: f64) -> PyResult<NdArray> {
        Ok(NdArray {
            imag: None,
            data: self.data.mapv(|v| v.max(a_min).min(a_max)),
        })
    }

    #[pyo3(signature = (axis=-1))]
    fn sort(&self, py: Python<'_>, axis: isize) -> PyResult<NdArray> {
        let ndim = self.data.ndim();
        let ax = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };
        if ax >= ndim {
            return Err(PyValueError::new_err("Axis out of bounds"));
        }
        // 排序为纯计算，主动释放 GIL，让其它 Python 线程在排序期间可推进。
        let data = &self.data;
        let arr = py
            .detach(|| -> Result<Array<f64, IxDyn>, String> {
                if ndim <= 1 {
                    let mut values: Vec<f64> = data.iter().copied().collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    return Array::from_shape_vec(IxDyn(&[values.len()]), values)
                        .map_err(|e| e.to_string());
                }
                let shape = data.shape().to_vec();
                let mut data_vec: Vec<f64> = data.iter().copied().collect();
                let axis_size = shape[ax];
                let pre_size: usize = shape.iter().take(ax).product();
                let post_size: usize = shape.iter().skip(ax + 1).product();
                let block_size = axis_size * post_size;
                // 并行排序每个轴切片
                let n_slices = pre_size * post_size;
                let sorted_slices: Vec<Vec<f64>> = (0..n_slices)
                    .into_par_iter()
                    .map(|idx| {
                        let outer = idx / post_size;
                        let inner = idx % post_size;
                        let base = outer * block_size + inner;
                        let mut slice: Vec<f64> = (0..axis_size)
                            .map(|k| data_vec[base + k * post_size])
                            .collect();
                        slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        slice
                    })
                    .collect();
                // 写回结果（串行，排序已并行完成）
                for (idx, sorted_slice) in sorted_slices.into_iter().enumerate() {
                    let outer = idx / post_size;
                    let inner = idx % post_size;
                    let base = outer * block_size + inner;
                    for (k, val) in sorted_slice.into_iter().enumerate() {
                        data_vec[base + k * post_size] = val;
                    }
                }
                Array::from_shape_vec(IxDyn(&shape), data_vec).map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }

    #[pyo3(signature = (axis=-1))]
    fn argsort(&self, py: Python<'_>, axis: isize) -> PyResult<NdArray> {
        let ndim = self.data.ndim();
        let ax = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };
        if ax >= ndim {
            return Err(PyValueError::new_err("Axis out of bounds"));
        }
        // 排序为纯计算，主动释放 GIL。
        let data = &self.data;
        let arr = py
            .detach(|| -> Result<Array<f64, IxDyn>, String> {
                if ndim <= 1 {
                    let values: Vec<f64> = data.iter().copied().collect();
                    let mut indices: Vec<usize> = (0..values.len()).collect();
                    indices.sort_by(|&a, &b| {
                        values[a]
                            .partial_cmp(&values[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let result: Vec<f64> = indices.into_iter().map(|i| i as f64).collect();
                    return Array::from_shape_vec(IxDyn(&[result.len()]), result)
                        .map_err(|e| e.to_string());
                }
                let shape = data.shape().to_vec();
                let data_vec: Vec<f64> = data.iter().copied().collect();
                let axis_size = shape[ax];
                let pre_size: usize = shape.iter().take(ax).product();
                let post_size: usize = shape.iter().skip(ax + 1).product();
                let block_size = axis_size * post_size;
                // 并行计算每个轴切片的排序索引
                let n_slices = pre_size * post_size;
                let index_results: Vec<Vec<(usize, f64)>> = (0..n_slices)
                    .into_par_iter()
                    .map(|idx| {
                        let outer = idx / post_size;
                        let inner = idx % post_size;
                        let base = outer * block_size + inner;
                        let mut indexed: Vec<(usize, f64)> = (0..axis_size)
                            .map(|k| (k, data_vec[base + k * post_size]))
                            .collect();
                        indexed.sort_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        indexed
                    })
                    .collect();
                let mut result = data_vec;
                for (idx, indexed) in index_results.into_iter().enumerate() {
                    let outer = idx / post_size;
                    let inner = idx % post_size;
                    let base = outer * block_size + inner;
                    for (pos, (idx_val, _)) in indexed.into_iter().enumerate() {
                        result[base + pos * post_size] = idx_val as f64;
                    }
                }
                Array::from_shape_vec(IxDyn(&shape), result).map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        Ok(NdArray {
            imag: None,
            data: arr,
        })
    }
}

/// 计算密集（transcendental：sin/exp/log…）逐元素并行阈值：
/// 每元素工作量大，较小规模并行即可回本。
pub(crate) const PAR_THRESHOLD: usize = 32_768;

/// 访存密集（add/sub/mul/div、比较、sqrt 等）逐元素并行阈值：
/// 每元素工作量极小，瓶颈在内存带宽；小/中等规模并行的线程调度开销反而拖慢，
/// 需到较大规模（多核带宽叠加）才划算。经基准定位（32k~500k 区间串行更快）而抬高。
pub(crate) const PAR_THRESHOLD_CHEAP: usize = 262_144;

/// 中等代价逐元素并行阈值（sqrt/reciprocal 等：比纯加乘重、比 transcendental 轻）。
/// 基准显示其并行回本点约在 5 万元素，介于 CHEAP 与 PAR_THRESHOLD 之间。
pub(crate) const PAR_THRESHOLD_MEDIUM: usize = 49_152;

fn binary_op<F>(a: &NdArray, b: &Bound<'_, PyAny>, op: F) -> PyResult<NdArray>
where
    F: Fn(f64, f64) -> f64 + Sync + Send,
{
    let py = b.py();
    if let Ok(scalar) = b.extract::<f64>() {
        let data = &a.data;
        // 纯计算，主动释放 GIL 让其它 Python 线程可并行推进。
        let out = py.detach(|| {
            if data.len() >= PAR_THRESHOLD_CHEAP {
                Zip::from(data).par_map_collect(|&x| op(x, scalar))
            } else {
                data.mapv(|x| op(x, scalar))
            }
        });
        return Ok(NdArray {
            imag: None,
            data: out,
        });
    }
    // 常见情形：右操作数就是 ndarray —— 借用而非 extract（后者会整份克隆）。
    if let Ok(other) = b.cast::<NdArray>() {
        let other = other.borrow();
        let a_data = &a.data;
        let other_data = &other.data;
        let out = py
            .detach(|| broadcast_binary_compute(a_data, other_data, op))
            .map_err(PyValueError::new_err)?;
        return Ok(NdArray {
            imag: None,
            data: out,
        });
    }
    // 回退：可被强制转换为 ndarray 的其它序列类型（此路径会克隆，但非热点）。
    if let Ok(other) = b.extract::<NdArray>() {
        let a_data = &a.data;
        let other_data = &other.data;
        let out = py
            .detach(|| broadcast_binary_compute(a_data, other_data, op))
            .map_err(PyValueError::new_err)?;
        return Ok(NdArray {
            imag: None,
            data: out,
        });
    }
    Err(PyTypeError::new_err("Unsupported operand type"))
}

/// 与 binary_op 行为完全一致：保留独立名字以表达调用方“左右操作数顺序”的语义，
/// 实际实现直接转发，避免重复代码。
#[inline]
fn binary_op_lr<F>(a: &NdArray, b: &Bound<'_, PyAny>, op: F) -> PyResult<NdArray>
where
    F: Fn(f64, f64) -> f64 + Sync + Send,
{
    binary_op(a, b, op)
}

/// 广播逐元素计算的核心实现。错误以 `String` 返回（`Send`，可跨 `allow_threads` 边界），
/// 由 `broadcast_binary_op` 包装成 `PyErr`。
fn broadcast_binary_compute<F>(
    a: &Array<f64, IxDyn>,
    b: &Array<f64, IxDyn>,
    op: F,
) -> Result<Array<f64, IxDyn>, String>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();

    if a_shape == b_shape {
        // 形状一致：用 ndarray::Zip 逐元素计算并直接产出结果数组，
        // 省去把两个输入分别物化成 Vec 的额外分配。
        let z = Zip::from(a).and(b);
        return Ok(if a.len() >= PAR_THRESHOLD_CHEAP {
            z.par_map_collect(|&x, &y| op(x, y))
        } else {
            z.map_collect(|&x, &y| op(x, y))
        });
    }

    // 标量（0 维）广播快速路径：直接在另一操作数上逐元素映射，
    // 跳过 reshape/clone/broadcast-view 全套机制（结果形状即另一操作数形状）。
    if a.ndim() == 0 {
        let s = *a.first().unwrap();
        return Ok(if b.len() >= PAR_THRESHOLD_CHEAP {
            Zip::from(b).par_map_collect(|&y| op(s, y))
        } else {
            b.mapv(|y| op(s, y))
        });
    }
    if b.ndim() == 0 {
        let s = *b.first().unwrap();
        return Ok(if a.len() >= PAR_THRESHOLD_CHEAP {
            Zip::from(a).par_map_collect(|&x| op(x, s))
        } else {
            a.mapv(|x| op(x, s))
        });
    }

    let max_ndim = ::std::cmp::max(a_shape.len(), b_shape.len());
    let mut a_padded = vec![1usize; max_ndim - a_shape.len()];
    a_padded.extend_from_slice(&a_shape);
    let mut b_padded = vec![1usize; max_ndim - b_shape.len()];
    b_padded.extend_from_slice(&b_shape);

    let mut out_shape = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        if a_padded[i] != b_padded[i] && a_padded[i] != 1 && b_padded[i] != 1 {
            return Err(format!(
                "Incompatible shapes for broadcasting: {:?} and {:?}",
                a_shape, b_shape
            ));
        }
        out_shape.push(::std::cmp::max(a_padded[i], b_padded[i]));
    }

    // 用带零步长的广播视图直接参与 Zip，避免把广播结果 to_owned 后再逐元素复制。
    let a_reshaped = a
        .clone()
        .into_shape_with_order(IxDyn(&a_padded))
        .map_err(|e| e.to_string())?;
    let b_reshaped = b
        .clone()
        .into_shape_with_order(IxDyn(&b_padded))
        .map_err(|e| e.to_string())?;
    let a_view = a_reshaped
        .broadcast(IxDyn(&out_shape))
        .ok_or_else(|| "Broadcasting failed".to_string())?;
    let b_view = b_reshaped
        .broadcast(IxDyn(&out_shape))
        .ok_or_else(|| "Broadcasting failed".to_string())?;

    let out_len: usize = out_shape.iter().product();
    let z = Zip::from(a_view).and(b_view);
    Ok(if out_len >= PAR_THRESHOLD_CHEAP {
        z.par_map_collect(|&x, &y| op(x, y))
    } else {
        z.map_collect(|&x, &y| op(x, y))
    })
}

fn broadcast_binary_op<F>(
    a: &Array<f64, IxDyn>,
    b: &Array<f64, IxDyn>,
    op: F,
) -> PyResult<Array<f64, IxDyn>>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    broadcast_binary_compute(a, b, op).map_err(PyValueError::new_err)
}

/// 复数逐元素二元运算类别。
#[derive(Clone, Copy)]
pub(crate) enum CBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[inline]
fn cbin_elem(ar: f64, ai: f64, br: f64, bi: f64, op: CBinOp) -> (f64, f64) {
    match op {
        CBinOp::Add => (ar + br, ai + bi),
        CBinOp::Sub => (ar - br, ai - bi),
        CBinOp::Mul => (ar * br - ai * bi, ar * bi + ai * br),
        CBinOp::Div => {
            let denom = br * br + bi * bi;
            ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
        }
    }
}

/// numpy 广播规则下的输出形状（尾维对齐）。
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    let (pa, pb) = (n - a.len(), n - b.len());
    for i in 0..n {
        let ad = if i < pa { 1 } else { a[i - pa] };
        let bd = if i < pb { 1 } else { b[i - pb] };
        if ad == bd || ad == 1 || bd == 1 {
            out[i] = ad.max(bd);
        } else {
            return Err(format!(
                "operands could not be broadcast together with shapes {:?} {:?}",
                a, b
            ));
        }
    }
    Ok(out)
}

/// 复数（实部 + 虚部）广播逐元素运算，返回 (实部, 虚部)。
fn broadcast_complex(
    a_re: &Array<f64, IxDyn>,
    a_im: &Array<f64, IxDyn>,
    b_re: &Array<f64, IxDyn>,
    b_im: &Array<f64, IxDyn>,
    op: CBinOp,
) -> Result<(Array<f64, IxDyn>, Array<f64, IxDyn>), String> {
    let out_shape = broadcast_shape(a_re.shape(), b_re.shape())?;
    let dim = IxDyn(&out_shape);
    let fail = || "Broadcasting failed".to_string();
    let av_re = a_re.broadcast(dim.clone()).ok_or_else(fail)?;
    let av_im = a_im.broadcast(dim.clone()).ok_or_else(fail)?;
    let bv_re = b_re.broadcast(dim.clone()).ok_or_else(fail)?;
    let bv_im = b_im.broadcast(dim).ok_or_else(fail)?;
    let pairs: Array<(f64, f64), IxDyn> = Zip::from(av_re)
        .and(av_im)
        .and(bv_re)
        .and(bv_im)
        .map_collect(|&ar, &ai, &br, &bi| cbin_elem(ar, ai, br, bi, op));
    let re_out = pairs.mapv(|p| p.0);
    let im_out = pairs.mapv(|p| p.1);
    Ok((re_out, im_out))
}

/// 取 NdArray 的 (实部, 虚部) 拥有型副本（实数数组虚部为全零）。
pub(crate) fn nd_parts(nd: &NdArray) -> (Array<f64, IxDyn>, Array<f64, IxDyn>) {
    let re = nd.data.clone();
    let im = match &nd.imag {
        Some(i) => i.clone(),
        None => Array::zeros(nd.data.raw_dim()),
    };
    (re, im)
}

/// 从 Python 对象取复数操作数的 (实部, 虚部)：支持复数标量、实数标量、NdArray 及可转换序列。
fn operand_parts(b: &Bound<'_, PyAny>) -> PyResult<(Array<f64, IxDyn>, Array<f64, IxDyn>)> {
    if let Ok(c) = b.cast::<PyComplex>() {
        return Ok((
            Array::from_elem(IxDyn(&[]), c.real()),
            Array::from_elem(IxDyn(&[]), c.imag()),
        ));
    }
    if let Ok(v) = b.extract::<f64>() {
        return Ok((
            Array::from_elem(IxDyn(&[]), v),
            Array::from_elem(IxDyn(&[]), 0.0),
        ));
    }
    if let Ok(nd) = b.cast::<NdArray>() {
        let nd = nd.borrow();
        return Ok(nd_parts(&nd));
    }
    let nd = NdArray::new(b)?;
    Ok(nd_parts(&nd))
}

/// 判断 Python 对象是否为复数操作数（复数标量或含虚部的 NdArray）。
fn is_complex_operand(b: &Bound<'_, PyAny>) -> bool {
    if b.is_instance_of::<PyComplex>() {
        return true;
    }
    if let Ok(nd) = b.cast::<NdArray>() {
        return nd.borrow().has_imag();
    }
    false
}

/// 两个 NdArray 的复数二元运算（供自由函数 add/subtract/... 复用）。
pub(crate) fn complex_binop_nd(a: &NdArray, b: &NdArray, op: CBinOp) -> PyResult<NdArray> {
    let (a_re, a_im) = nd_parts(a);
    let (b_re, b_im) = nd_parts(b);
    let (re, im) =
        broadcast_complex(&a_re, &a_im, &b_re, &b_im, op).map_err(PyValueError::new_err)?;
    Ok(NdArray::from_complex(re, im))
}

/// ndarray 运算符分派：任一操作数为复数走复数路径，否则复用实数快速路径。
fn dispatch_binop(a: &NdArray, b: &Bound<'_, PyAny>, op: CBinOp) -> PyResult<NdArray> {
    if a.has_imag() || is_complex_operand(b) {
        let (a_re, a_im) = nd_parts(a);
        let (b_re, b_im) = operand_parts(b)?;
        let (re, im) =
            broadcast_complex(&a_re, &a_im, &b_re, &b_im, op).map_err(PyValueError::new_err)?;
        return Ok(NdArray::from_complex(re, im));
    }
    match op {
        CBinOp::Add => binary_op(a, b, |x, y| x + y),
        CBinOp::Sub => binary_op(a, b, |x, y| x - y),
        CBinOp::Mul => binary_op(a, b, |x, y| x * y),
        CBinOp::Div => binary_op(a, b, |x, y| x / y),
    }
}

/// 反向运算符分派（b 为左操作数）：仅加/乘可交换，减/除需交换次序。
fn dispatch_binop_r(a: &NdArray, b: &Bound<'_, PyAny>, op: CBinOp) -> PyResult<NdArray> {
    if a.has_imag() || is_complex_operand(b) {
        let (a_re, a_im) = nd_parts(a);
        let (b_re, b_im) = operand_parts(b)?;
        // 反向：计算 b op a。
        let (re, im) =
            broadcast_complex(&b_re, &b_im, &a_re, &a_im, op).map_err(PyValueError::new_err)?;
        return Ok(NdArray::from_complex(re, im));
    }
    match op {
        CBinOp::Add => binary_op(a, b, |x, y| y + x),
        CBinOp::Sub => binary_op(a, b, |x, y| y - x),
        CBinOp::Mul => binary_op(a, b, |x, y| y * x),
        CBinOp::Div => binary_op(a, b, |x, y| y / x),
    }
}

/// 沿指定轴做就地累积扫描（cumsum/cumprod 共用）。
/// axis=None 时在展平后的 C 序上做整体扫描，结果为 1-D。
/// 单份缓冲上就地更新，并按连续外层块并行；纯计算，主动释放 GIL。
fn cumulative<F>(
    py: Python<'_>,
    data: &Array<f64, IxDyn>,
    axis: Option<isize>,
    init: f64,
    op: F,
) -> PyResult<NdArray>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let ndim = data.ndim();
    if axis.is_none() {
        let n = data.len();
        let arr = py.detach(|| {
            let mut buf: Vec<f64> = data.iter().copied().collect();
            let mut cum = init;
            for x in buf.iter_mut() {
                cum = op(cum, *x);
                *x = cum;
            }
            Array::from_shape_vec(IxDyn(&[n]), buf).unwrap()
        });
        return Ok(NdArray {
            imag: None,
            data: arr,
        });
    }
    let ax_raw = axis.unwrap();
    let ax = if ax_raw < 0 {
        (ndim as isize + ax_raw) as usize
    } else {
        ax_raw as usize
    };
    if ax >= ndim {
        return Err(PyValueError::new_err("Axis out of bounds"));
    }
    let shape = data.shape().to_vec();
    let axis_size = shape[ax];
    let post_size: usize = shape.iter().skip(ax + 1).product();
    let block_size = axis_size * post_size;
    let arr = py.detach(|| -> Result<Array<f64, IxDyn>, String> {
        // 单份 C 序缓冲，就地扫描；外层块（block_size 连续、互不重叠）并行。
        let mut buf: Vec<f64> = data.iter().copied().collect();
        if block_size > 0 {
            buf.par_chunks_mut(block_size).for_each(|block| {
                for inner in 0..post_size {
                    let mut cum = init;
                    let mut i = inner;
                    for _ in 0..axis_size {
                        cum = op(cum, block[i]);
                        block[i] = cum;
                        i += post_size;
                    }
                }
            });
        }
        Array::from_shape_vec(IxDyn(&shape), buf).map_err(|e| e.to_string())
    });
    let arr = arr.map_err(PyValueError::new_err)?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

/// ArrayFlags - 数组内存布局信息，与 NumPy 的 np.ndarray.flags 兼容
#[pyclass(name = "ArrayFlags")]
struct ArrayFlags {}

#[pymethods]
#[allow(non_snake_case)]
impl ArrayFlags {
    #[getter]
    fn C_CONTIGUOUS(&self) -> bool {
        true
    }

    #[getter]
    fn F_CONTIGUOUS(&self) -> bool {
        true
    }

    #[getter]
    fn OWNDATA(&self) -> bool {
        true
    }

    #[getter]
    fn WRITEABLE(&self) -> bool {
        true
    }

    #[getter]
    fn ALIGNED(&self) -> bool {
        true
    }

    #[getter]
    fn WRITEBACKIFCOPY(&self) -> bool {
        false
    }

    #[getter]
    fn UPDATEIFCOPY(&self) -> bool {
        false
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __str__(&self) -> String {
        let mut s = String::new();
        writeln!(s, "  C_CONTIGUOUS : {}", self.C_CONTIGUOUS()).unwrap();
        writeln!(s, "  F_CONTIGUOUS : {}", self.F_CONTIGUOUS()).unwrap();
        writeln!(s, "  OWNDATA : {}", self.OWNDATA()).unwrap();
        writeln!(s, "  WRITEABLE : {}", self.WRITEABLE()).unwrap();
        writeln!(s, "  ALIGNED : {}", self.ALIGNED()).unwrap();
        writeln!(s, "  WRITEBACKIFCOPY : {}", self.WRITEBACKIFCOPY()).unwrap();
        writeln!(s, "  UPDATEIFCOPY : {}", self.UPDATEIFCOPY()).unwrap();
        s.trim_end().to_string()
    }
}

/// ndarray 迭代器
#[pyclass(name = "ndarray_iter")]
struct NdArrayIter {
    arrays: Vec<NdArray>,
    index: usize,
}

#[pymethods]
impl NdArrayIter {
    fn __iter__<'a>(slf: &'a Bound<'a, Self>) -> Bound<'a, Self> {
        slf.clone()
    }

    fn __next__(&mut self) -> PyResult<Option<NdArray>> {
        if self.index < self.arrays.len() {
            let val = self.arrays[self.index].clone();
            self.index += 1;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }
}

// ========== f64 数组 → 指定 dtype 的原始字节（供 __array_interface__ 与 numpy 互操作） ==========
fn put_scalar_bytes(out: &mut Vec<u8>, le: &[u8], little: bool) {
    if little {
        out.extend_from_slice(le);
    } else {
        let mut b = le.to_vec();
        b.reverse();
        out.extend_from_slice(&b);
    }
}

// 底层数据恒为 f64；按 typestr 目标类型逐元素编码（浮点→整数 `as` 转换在越界/NaN 时饱和）。
fn encode_scalar(out: &mut Vec<u8>, kind: char, itemsize: usize, little: bool, v: f64) -> bool {
    match (kind, itemsize) {
        ('f', 8) => put_scalar_bytes(out, &v.to_le_bytes(), little),
        ('f', 4) => put_scalar_bytes(out, &(v as f32).to_le_bytes(), little),
        ('i', 8) => put_scalar_bytes(out, &(v as i64).to_le_bytes(), little),
        ('i', 4) => put_scalar_bytes(out, &(v as i32).to_le_bytes(), little),
        ('i', 2) => put_scalar_bytes(out, &(v as i16).to_le_bytes(), little),
        ('i', 1) => out.push((v as i8) as u8),
        ('u', 8) => put_scalar_bytes(out, &(v as u64).to_le_bytes(), little),
        ('u', 4) => put_scalar_bytes(out, &(v as u32).to_le_bytes(), little),
        ('u', 2) => put_scalar_bytes(out, &(v as u16).to_le_bytes(), little),
        ('u', 1) => out.push(v as u8),
        ('b', 1) => out.push(if v != 0.0 { 1 } else { 0 }),
        _ => return false,
    }
    true
}

fn encode_f64_array(data: &Array<f64, IxDyn>, typestr: &str) -> PyResult<Vec<u8>> {
    let chars: Vec<char> = typestr.chars().collect();
    let byteorder = chars.first().copied().unwrap_or('|');
    let kind = *chars
        .get(1)
        .ok_or_else(|| PyValueError::new_err("invalid array interface typestr"))?;
    let itemsize: usize = typestr[2..]
        .parse()
        .map_err(|_| PyValueError::new_err("invalid array interface typestr"))?;
    let little = byteorder != '>';
    let mut out = Vec::with_capacity(data.len() * itemsize.max(1));
    for &v in data.iter() {
        if !encode_scalar(&mut out, kind, itemsize, little, v) {
            return Err(PyValueError::new_err("unsupported array interface typestr"));
        }
    }
    Ok(out)
}

fn build_array_interface<'py>(
    py: Python<'py>,
    data: &Array<f64, IxDyn>,
    typestr: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let shape = data.shape().to_vec();
    dict.set_item("shape", vec_usize_to_pytuple(py, &shape))?;
    dict.set_item("typestr", typestr)?;
    let bytes = encode_f64_array(data, typestr)?;
    dict.set_item("data", PyBytes::new(py, &bytes))?;
    dict.set_item("version", 3)?;
    Ok(dict)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NdArray>()?;
    m.add_class::<NdArrayIter>()?;

    creation::register(m)?;
    mathematics::register(m)?;
    arithmetic::register(m)?;
    logic::register(m)?;
    bitwise::register(m)?;
    statistics::register(m)?;
    sorting::register(m)?;
    searching::register(m)?;
    setops::register(m)?;
    manipulation::register(m)?;
    poly::register(m)?;
    io::register(m)?;
    buffer::register(m)?;
    formatting::register(m)?;

    m.add_function(wrap_pyfunction!(indexing::getitem_multi, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::getitem_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setitem_multi, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::iscomplex_cpx, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::masked_select, m)?)?;
    m.add_function(wrap_pyfunction!(fft::py_fft_ndarray, m)?)?;
    m.add_function(wrap_pyfunction!(fft::py_ifft_ndarray, m)?)?;
    m.add_function(wrap_pyfunction!(fft::py_rfft_ndarray, m)?)?;
    m.add_function(wrap_pyfunction!(fft::py_irfft_ndarray, m)?)?;

    let random_module = PyModule::new(m.py(), "random")?;
    random::init_module(&random_module)?;
    m.add_submodule(&random_module)?;

    let linalg_module = PyModule::new(m.py(), "linalg")?;
    linalg::init_module(&linalg_module)?;
    m.add_submodule(&linalg_module)?;

    Ok(())
}

use ndarray::{Array, ArrayViewD, Axis, IxDyn, Slice};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList, PySlice, PyTuple};
use rayon::prelude::*;

mod linalg;
mod random;
mod fft;
mod indexing;

pub(crate) fn parse_py_list_to_flat(data: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    if let Ok(val) = data.extract::<f64>() {
        return Ok((vec![val], vec![]));
    }
    if let Ok(list) = data.cast::<PyList>() {
        if list.is_empty() {
            return Ok((vec![], vec![0]));
        }
        let mut all_values = Vec::new();
        let mut child_shapes: Vec<Vec<usize>> = Vec::new();
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
    } else {
        Err(PyTypeError::new_err("Unsupported data type"))
    }
}

fn shape_to_vec(shape: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(tup) = shape.cast::<PyTuple>() {
        let mut result = Vec::new();
        for item in tup.iter() {
            result.push(item.extract::<usize>()?);
        }
        return Ok(result);
    }
    if let Ok(val) = shape.extract::<usize>() {
        return Ok(vec![val]);
    }
    if let Ok(list) = shape.cast::<PyList>() {
        let mut result = Vec::new();
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

fn vec_f64_to_pylist<'a>(py: Python<'a>, vec: &[f64]) -> Bound<'a, PyList> {
    let list = PyList::empty(py);
    for v in vec {
        list.append(PyFloat::new(py, *v)).unwrap();
    }
    list
}

fn vec_usize_to_pytuple<'a>(py: Python<'a>, vec: &[usize]) -> Bound<'a, PyTuple> {
    let items: Vec<Bound<'a, PyAny>> = vec.iter().map(|&v| pyo3::IntoPyObject::into_pyobject(v, py).unwrap().into_any()).collect();
    PyTuple::new(py, &items).unwrap()
}

fn to_python_list<'a>(py: Python<'a>, arr: &ArrayViewD<'_, f64>) -> PyResult<Bound<'a, PyAny>> {
    if arr.ndim() == 0 {
        return Ok(value_to_pyobject(py, arr.iter().next().copied().unwrap_or(0.0_f64)));
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

fn compute_max_width(arr: &Array<f64, IxDyn>) -> usize {
    arr.iter()
        .map(|v| format_scalar(*v).len())
        .max()
        .unwrap_or(1)
}

fn format_array_repr_inner(arr: &Array<f64, IxDyn>, _prefix: &str, pad_width: usize) -> String {
    if arr.ndim() == 0 {
        return format!("{}", arr.iter().next().copied().unwrap_or(0.0_f64));
    }
    if arr.ndim() == 1 {
        let mut s = String::from("[");
        for (i, val) in arr.iter().enumerate() {
            if i > 0 {
                s.push_str(" ");
            }
            let val_str = format_scalar(*val);
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
        let row_str = format_array_repr_inner(&sub, "", pad_width);
        s.push_str(&row_str);
    }
    s.push_str("]");
    s
}

fn format_array_repr(arr: &Array<f64, IxDyn>, _prefix: &str) -> String {
    // 对 2D+ 数组计算最大宽度并右对齐填充，与 numpy 风格一致
    let pad_width = if arr.ndim() >= 2 {
        compute_max_width(arr)
    } else {
        0
    };
    format_array_repr_inner(arr, _prefix, pad_width)
}

fn format_scalar(val: f64) -> String {
    if val.is_nan() {
        return "nan".to_string();
    }
    if val.is_infinite() {
        return if val > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if val == val.floor() && val.is_finite() && val.abs() < 1e16 {
        let v = val as i64;
        if v as f64 == val {
            return format!("{}", v);
        }
    }
    format!("{:.8}", val).trim_end_matches('0').trim_end_matches('.').to_string()
}

fn format_float_scalar(val: f64) -> String {
    if val.is_nan() {
        return "nan".to_string();
    }
    if val.is_infinite() {
        return if val > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if val != 0.0 && val.abs() < 1e-10 {
        return format!("{:.10e}", val);
    }
    let val_rounded = (val * 1e10).round() / 1e10;
    if val_rounded == val_rounded.floor() && val_rounded.is_finite() && val_rounded.abs() < 1e16 {
        let v = val_rounded as i64;
        if v as f64 == val_rounded {
            return format!("{}", v);
        }
    }
    if val_rounded.abs() >= 1e10 {
        return format!("{:.10e}", val_rounded);
    }
    let s = format!("{:.10}", val_rounded);
    if let Some(pos) = s.find('.') {
        s[..pos + 1 + s[pos + 1..].trim_end_matches('0').len()].to_string()
    } else {
        s
    }
}

/// NdArray: 类似 NumPy ndarray 的多维数组
#[pyclass(name = "ndarray", from_py_object)]
#[derive(Clone)]
pub struct NdArray {
    data: Array<f64, IxDyn>,
}

#[pymethods]
impl NdArray {
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (values, shape) = parse_py_list_to_flat(data)?;
        if shape.is_empty() {
            let arr = Array::from_shape_vec(IxDyn(&[]), values)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(NdArray { data: arr });
        }
        let shape_vec: Vec<usize> = shape;
        let arr = Array::from_shape_vec(IxDyn(&shape_vec), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
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

    fn __getitem__<'a>(slf: &'a Bound<'a, Self>, index: &Bound<'_, PyAny>) -> PyResult<Bound<'a, PyAny>> {
        let py = slf.py();
        let borrowed = slf.borrow();
        let data = &borrowed.data;

        if let Ok(idx) = index.extract::<isize>() {
            if data.ndim() == 0 {
                return Ok(value_to_pyobject(
                    py,
                    data.iter().next().copied().unwrap_or(0.0_f64),
                ));
            }
            let dim = data.shape()[0] as isize;
            let actual = if idx < 0 { dim + idx } else { idx };
            if actual < 0 || actual >= dim {
                return Err(PyIndexError::new_err("Index out of bounds"));
            }
            let view = data.index_axis(Axis(0), actual as usize);
            if view.ndim() == 0 {
                return Ok(value_to_pyobject(
                    py,
                    view.iter().next().copied().unwrap_or(0.0_f64),
                ));
            }
            let sub = NdArray {
                data: view.to_owned().into_dyn(),
            };
            let bound = Bound::new(py, sub)?;
            return Ok(bound.into_any());
        }

        if let Ok(list) = index.cast::<PyList>() {
            let mut bools = Vec::new();
            for item in list.iter() {
                if let Ok(b) = item.extract::<bool>() {
                    bools.push(b);
                } else {
                    return Err(PyTypeError::new_err("Boolean index list must contain only booleans"));
                }
            }
            let flat_data: Vec<f64> = data.iter().copied().collect();
            let mut result = Vec::new();
            for (i, &b) in bools.iter().enumerate() {
                if b && i < flat_data.len() {
                    result.push(flat_data[i]);
                }
            }
            let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let sub = NdArray { data: arr };
            let bound = Bound::new(py, sub)?;
            return Ok(bound.into_any());
        }

        if let Ok(arr) = index.extract::<NdArray>() {
            let flat_data: Vec<f64> = data.iter().copied().collect();
            let mut result = Vec::new();
            for (i, &v) in arr.data.iter().enumerate() {
                if v != 0.0 && i < flat_data.len() {
                    result.push(flat_data[i]);
                }
            }
            let arr_result = Array::from_shape_vec(IxDyn(&[result.len()]), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let sub = NdArray { data: arr_result };
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
        Ok(NdArray { data: view })
    }

    fn __setitem__(
        &mut self,
        index: isize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if self.data.ndim() == 0 {
            return Err(PyIndexError::new_err(
                "Scalar array does not support item assignment",
            ));
        }
        let val_nd = if let Ok(v) = value.extract::<f64>() {
            NdArray {
                data: Array::from_elem(IxDyn(&[]), v),
            }
        } else if let Ok(v) = value.extract::<i32>() {
            NdArray {
                data: Array::from_elem(IxDyn(&[]), v as f64),
            }
        } else if let Ok(arr) = value.extract::<NdArray>() {
            arr
        } else {
            let (vals, shape) = parse_py_list_to_flat(value).ok().unwrap_or_default();
            if !vals.is_empty() && !shape.is_empty() {
                if let Ok(arr) = Array::from_shape_vec(IxDyn(&shape), vals) {
                    NdArray { data: arr }
                } else {
                    return Err(PyTypeError::new_err("Unsupported value type"));
                }
            } else {
                return Err(PyTypeError::new_err("Unsupported value type"));
            }
        };
        let val_arr = val_nd.data;

        let dim = self.data.shape()[0] as isize;
        let actual = if index < 0 { dim + index } else { index };
        if actual < 0 || actual >= dim {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        let row_size = self.data.len() / self.data.shape()[0];
        let start = actual as usize * row_size;
        let data_slice = self.data.as_slice_mut().ok_or_else(|| {
            PyValueError::new_err("Non-contiguous array")
        })?;
        let max_len = val_arr.len().min(row_size);
        if let Some(val_slice) = val_arr.as_slice() {
            data_slice[start..start + max_len].copy_from_slice(&val_slice[..max_len]);
        } else {
            for j in 0..max_len {
                data_slice[start + j] = *val_arr.iter().nth(j).unwrap_or(&0.0);
            }
        }
        Ok(())
    }

    fn __setitem_slice__(&mut self, py_slice: &Bound<'_, PySlice>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let val_nd = if let Ok(v) = value.extract::<f64>() {
            NdArray {
                data: Array::from_elem(IxDyn(&[]), v),
            }
        } else if let Ok(v) = value.extract::<i32>() {
            NdArray {
                data: Array::from_elem(IxDyn(&[]), v as f64),
            }
        } else if let Ok(arr) = value.extract::<NdArray>() {
            arr
        } else {
            return Err(PyTypeError::new_err("Unsupported value type"));
        };
        let val_arr = val_nd.data;

        let dim = self.data.shape()[0] as i64;
        let indices = py_slice
            .call_method1("indices", (dim,))?
            .extract::<(i64, i64, i64)>()?;
        let (start, stop, step) = indices;
        let nd_slice = Slice::new(start as isize, Some(stop as isize), step as isize);
        let mut slice_view = self.data.slice_axis_mut(Axis(0), nd_slice);
        let bcast = val_arr
            .broadcast(slice_view.dim())
            .ok_or_else(|| PyValueError::new_err("Cannot broadcast value to slice shape"))?;
        slice_view.assign(&bcast);
        Ok(())
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<NdArrayIter> {
        let data = slf.borrow().data.clone();
        let ndim = data.ndim();
        let sub_arrays: Vec<NdArray> = if ndim <= 1 {
            data.iter()
                .map(|&v| NdArray {
                    data: Array::from_elem(IxDyn(&[]), v),
                })
                .collect()
        } else {
            (0..data.shape()[0])
                .map(|i| {
                    let sub = data.index_axis(Axis(0), i).to_owned().into_dyn();
                    NdArray { data: sub }
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
        Ok("float64".to_string())
    }

    #[getter]
    #[pyo3(name = "T")]
    fn t(&self) -> PyResult<NdArray> {
        if self.data.ndim() <= 1 {
            return Ok(NdArray {
                data: self.data.clone(),
            });
        }
        Ok(NdArray {
            data: self.data.view().reversed_axes().into_owned().into_dyn(),
        })
    }

    fn reshape(&self, shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        // 使用 isize 解析，支持 -1
        let s: Vec<isize> = if let Ok(tup) = shape.cast::<PyTuple>() {
            let mut result = Vec::new();
            for item in tup.iter() {
                result.push(item.extract::<isize>()?);
            }
            result
        } else if let Ok(val) = shape.extract::<isize>() {
            vec![val]
        } else if let Ok(list) = shape.cast::<PyList>() {
            let mut result = Vec::new();
            for item in list.iter() {
                result.push(item.extract::<isize>()?);
            }
            result
        } else {
            return Err(PyTypeError::new_err("Shape must be a tuple, list, or integer"));
        };

        let total = self.data.len() as isize;

        // 处理 -1（自动计算维度）
        let unknown_count = s.iter().filter(|&&v| v == -1).count();
        if unknown_count > 1 {
            return Err(PyValueError::new_err("can only specify one unknown dimension"));
        }

        let known: isize = s.iter().filter(|&&v| v > 0).product();
        let out: Vec<usize> = if unknown_count == 1 {
            if known == 0 || total % known != 0 {
                return Err(PyValueError::new_err(format!(
                    "cannot reshape array of size {} into shape {:?}", total, s
                )));
            }
            s.iter().map(|&v| -> PyResult<usize> {
                if v == -1 {
                    Ok((total / known) as usize)
                } else if v <= 0 {
                    Err(PyValueError::new_err(format!("{} is not a valid dimension size", v)))
                } else {
                    Ok(v as usize)
                }
            }).collect::<PyResult<Vec<_>>>()?
        } else {
            s.iter().map(|&v| -> PyResult<usize> {
                if v <= 0 {
                    Err(PyValueError::new_err(format!("{} is not a valid dimension size", v)))
                } else {
                    Ok(v as usize)
                }
            }).collect::<PyResult<Vec<_>>>()?
        };

        // 验证总元素数一致
        if out.iter().product::<usize>() != total as usize {
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {} into shape {:?}", total, s
            )));
        }

        let arr = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(&out))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    fn flatten(&self) -> NdArray {
        let flat: Vec<f64> = self.data.iter().copied().collect();
        NdArray {
            data: Array::from_shape_vec(IxDyn(&[flat.len()]), flat).unwrap(),
        }
    }

    fn ravel(&self) -> NdArray {
        self.flatten()
    }

    fn squeeze(&self) -> PyResult<NdArray> {
        let new_shape: Vec<usize> = self.data.shape().iter().filter(|&&d| d > 1).copied().collect();
        let arr = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(&new_shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    fn copy(&self) -> NdArray {
        NdArray { data: self.data.clone() }
    }

    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let view = self.data.view().into_dyn();
        to_python_list(py, &view)
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
                Ok(NdArray { data: arr })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
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
                Ok(NdArray { data: arr })
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
        let is_int = matches!(
            dt.as_str(),
            "int" | "i" | "int8" | "int16" | "int32" | "int64" | "intp" | "i8" | "i16" | "i32" | "i64"
        );
        if is_int {
            let casted: Vec<f64> = self.data.iter().map(|v| v.trunc()).collect();
            let arr = Array::from_shape_vec(IxDyn(self.data.shape()), casted)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: arr })
        } else {
            Ok(NdArray { data: self.data.clone() })
        }
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op(self, other, |a, b| a + b)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op(self, other, |a, b| a - b)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op(self, other, |a, b| a * b)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op(self, other, |a, b| a / b)
    }

    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        _mod: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NdArray> {
        binary_op(self, other, |a, b| a.powf(b))
    }

    fn __neg__(&self) -> PyResult<NdArray> {
        Ok(NdArray {
            data: self.data.mapv(|x| -x),
        })
    }

    fn __abs__(&self) -> PyResult<NdArray> {
        Ok(NdArray {
            data: self.data.mapv(|x| x.abs()),
        })
    }

    fn __round__(&self, ndigits: Option<i32>) -> PyResult<NdArray> {
        match ndigits {
            None => Ok(NdArray {
                data: self.data.mapv(|x| x.round()),
            }),
            Some(n) => {
                let factor = 10.0_f64.powi(n);
                Ok(NdArray {
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
        binary_op_lr(self, other, |a, b| if (a - b).abs() < 1e-12 { 1.0 } else { 0.0 })
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op_lr(self, other, |a, b| if (a - b).abs() >= 1e-12 { 1.0 } else { 0.0 })
    }

    // ========== 新增缺失方法 ==========

    #[pyo3(signature = (axis=None))]
    fn prod(&self, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let val = self.data.iter().cloned().fold(1.0_f64, |a, b| a * b);
                Ok(NdArray {
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
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
                let new_shape: Vec<usize> = shape.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ax)
                    .map(|(_, &s)| s)
                    .collect();
                let arr = Array::from_shape_vec(IxDyn(&new_shape), results)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(NdArray { data: arr })
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn cumsum(&self, axis: Option<isize>) -> PyResult<NdArray> {
        let ndim = self.data.ndim();
        let ax = match axis {
            None => return self.flatten().cumsum(Some(0)),
            Some(ax) => if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize },
        };
        let shape = self.data.shape().to_vec();
        let axis_size = shape[ax];
        let pre_size: usize = shape.iter().take(ax).product();
        let post_size: usize = shape.iter().skip(ax + 1).product();
        let block_size = axis_size * post_size;
        let data_vec: Vec<f64> = self.data.iter().copied().collect();
        // 并行计算每个切片的累积和
        let n_slices = pre_size * post_size;
        let slice_results: Vec<Vec<f64>> = (0..n_slices)
            .into_par_iter()
            .map(|idx| {
                let outer = idx / post_size;
                let inner = idx % post_size;
                let base = outer * block_size + inner;
                let mut cum = 0.0_f64;
                let mut slice = Vec::with_capacity(axis_size);
                for k in 0..axis_size {
                    cum += data_vec[base + k * post_size];
                    slice.push(cum);
                }
                slice
            })
            .collect();
        // 按正确内存布局写回
        let mut result = vec![0.0; data_vec.len()];
        for (idx, slice) in slice_results.into_iter().enumerate() {
            let outer = idx / post_size;
            let inner = idx % post_size;
            let base = outer * block_size + inner;
            for (k, val) in slice.into_iter().enumerate() {
                result[base + k * post_size] = val;
            }
        }
        let arr = Array::from_shape_vec(IxDyn(&shape), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    #[pyo3(signature = (axis=None))]
    fn cumprod(&self, axis: Option<isize>) -> PyResult<NdArray> {
        let ndim = self.data.ndim();
        let ax = match axis {
            None => return self.flatten().cumprod(Some(0)),
            Some(ax) => if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize },
        };
        let shape = self.data.shape().to_vec();
        let axis_size = shape[ax];
        let pre_size: usize = shape.iter().take(ax).product();
        let post_size: usize = shape.iter().skip(ax + 1).product();
        let block_size = axis_size * post_size;
        let data_vec: Vec<f64> = self.data.iter().copied().collect();
        // 并行计算每个切片的累积乘积
        let n_slices = pre_size * post_size;
        let slice_results: Vec<Vec<f64>> = (0..n_slices)
            .into_par_iter()
            .map(|idx| {
                let outer = idx / post_size;
                let inner = idx % post_size;
                let base = outer * block_size + inner;
                let mut cum = 1.0_f64;
                let mut slice = Vec::with_capacity(axis_size);
                for k in 0..axis_size {
                    cum *= data_vec[base + k * post_size];
                    slice.push(cum);
                }
                slice
            })
            .collect();
        // 按正确内存布局写回
        let mut result = vec![0.0; data_vec.len()];
        for (idx, slice) in slice_results.into_iter().enumerate() {
            let outer = idx / post_size;
            let inner = idx % post_size;
            let base = outer * block_size + inner;
            for (k, val) in slice.into_iter().enumerate() {
                result[base + k * post_size] = val;
            }
        }
        let arr = Array::from_shape_vec(IxDyn(&shape), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    #[pyo3(signature = (offset=0, axis1=0, axis2=1))]
    fn diagonal(&self, offset: isize, axis1: usize, axis2: usize) -> PyResult<NdArray> {
        let ndim = self.data.ndim();
        if ndim < 2 {
            return Err(PyValueError::new_err("diagonal requires at least 2-D array"));
        }
        let shape = self.data.shape().to_vec();
        let d1 = if axis1 < ndim { shape[axis1] } else { return Err(PyValueError::new_err("axis1 out of bounds")); };
        let d2 = if axis2 < ndim { shape[axis2] } else { return Err(PyValueError::new_err("axis2 out of bounds")); };
        let mut diag = Vec::new();
        let offset_abs = offset.unsigned_abs() as usize;
        let i_start = if offset >= 0 { 0 } else { offset_abs };
        let j_start = if offset >= 0 { offset_abs } else { 0 };
        let n = d1.min(d2).saturating_sub(offset_abs);
        for k in 0..n {
            let mut coords = vec![0usize; ndim];
            for d in 0..ndim {
                if d == axis1 {
                    coords[d] = i_start + k;
                } else if d == axis2 {
                    coords[d] = j_start + k;
                } else {
                    coords[d] = 0;
                }
            }
            diag.push(self.data[IxDyn(&coords)]);
        }
        let arr = Array::from_shape_vec(IxDyn(&[diag.len()]), diag)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
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
                return Err(PyValueError::new_err("item requires exactly one element array when no indices given"));
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
                let result: Vec<f64> = idx_vals.iter().map(|&i| *flat.get(i).unwrap_or(&0.0)).collect();
                let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                return Ok(NdArray { data: arr });
            }
            Some(ax) => if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize },
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
        let idx_vals: Vec<usize> = indices.data.iter().map(|&v| {
            let idx = v as isize;
            if idx < 0 { (axis_size as isize + idx) as usize } else { idx as usize }
        }).collect();
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
        Ok(NdArray { data: arr })
    }

    fn put(&mut self, indices: &NdArray, values: &NdArray) -> PyResult<()> {
        let flat_len = self.data.len();
        let idx_vals: Vec<usize> = indices.data.iter().map(|&v| {
            let idx = v as isize;
            if idx < 0 { (flat_len as isize + idx) as usize } else { idx as usize }
        }).collect();
        let val_vec: Vec<f64> = values.data.iter().copied().collect();
        let data_slice = self.data.as_slice_mut().ok_or_else(|| {
            PyValueError::new_err("put requires contiguous array")
        })?;
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
        binary_op_lr(self, other, |a, b| a + b)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op_lr(self, other, |a, b| a - b)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op_lr(self, other, |a, b| a * b)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<NdArray> {
        binary_op_lr(self, other, |a, b| a / b)
    }

    #[pyo3(signature = (axis=None))]
    fn sum(&self, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let val = self.data.sum();
                Ok(NdArray {
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
                let result = self.data.sum_axis(Axis(ax));
                Ok(NdArray {
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
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
                let result = self.data.mean_axis(Axis(ax));
                match result {
                    Some(arr) => Ok(NdArray {
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
                let var = self
                    .data
                    .mapv(|x| (x - m).powi(2))
                    .mean()
                    .unwrap_or(0.0);
                Ok(NdArray {
                    data: Array::from_elem(IxDyn(&[]), var.sqrt()),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
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

                        let new_shape: Vec<usize> = shape.iter()
                            .enumerate()
                            .filter(|(i, _)| *i != ax)
                            .map(|(_, &s)| s)
                            .collect();

                        if new_shape.is_empty() {
                            Ok(NdArray {
                                data: Array::from_elem(IxDyn(&[]), results[0]),
                            })
                        } else {
                            let arr = Array::from_shape_vec(IxDyn(&new_shape), results)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                            Ok(NdArray { data: arr })
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
                let var = self
                    .data
                    .mapv(|x| (x - m).powi(2))
                    .mean()
                    .unwrap_or(0.0);
                Ok(NdArray {
                    data: Array::from_elem(IxDyn(&[]), var),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
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

                        let new_shape: Vec<usize> = shape.iter()
                            .enumerate()
                            .filter(|(i, _)| *i != ax)
                            .map(|(_, &s)| s)
                            .collect();

                        if new_shape.is_empty() {
                            Ok(NdArray {
                                data: Array::from_elem(IxDyn(&[]), results[0]),
                            })
                        } else {
                            let arr = Array::from_shape_vec(IxDyn(&new_shape), results)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                            Ok(NdArray { data: arr })
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
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
                let result: Array<f64, IxDyn> = self
                    .data
                    .fold_axis(Axis(ax), f64::INFINITY, |acc, &v| {
                        let a = *acc;
                        if v < a { v } else { a }
                    })
                    .into_dyn();
                Ok(NdArray { data: result })
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn max(&self, axis: Option<isize>) -> PyResult<NdArray> {
        match axis {
            None => {
                let val = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                Ok(NdArray {
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
                let result: Array<f64, IxDyn> = self
                    .data
                    .fold_axis(Axis(ax), f64::NEG_INFINITY, |acc, &v| {
                        let a = *acc;
                        if v > a { v } else { a }
                    })
                    .into_dyn();
                Ok(NdArray { data: result })
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
            data: self.data.mapv(|v| v.max(a_min).min(a_max)),
        })
    }

    #[pyo3(signature = (axis=-1))]
    fn sort(&self, axis: isize) -> PyResult<NdArray> {
        let ndim = self.data.ndim();
        let ax = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
        if ax >= ndim {
            return Err(PyValueError::new_err("Axis out of bounds"));
        }
        if ndim <= 1 {
            let mut values: Vec<f64> = self.data.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let arr = Array::from_shape_vec(IxDyn(&[values.len()]), values)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(NdArray { data: arr });
        }
        let shape = self.data.shape().to_vec();
        let mut data_vec: Vec<f64> = self.data.iter().copied().collect();
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
        let arr = Array::from_shape_vec(IxDyn(&shape), data_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    #[pyo3(signature = (axis=-1))]
    fn argsort(&self, axis: isize) -> PyResult<NdArray> {
        let ndim = self.data.ndim();
        let ax = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
        if ax >= ndim {
            return Err(PyValueError::new_err("Axis out of bounds"));
        }
        if ndim <= 1 {
            let values: Vec<f64> = self.data.iter().copied().collect();
            let mut indices: Vec<usize> = (0..values.len()).collect();
            indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
            let result: Vec<f64> = indices.into_iter().map(|i| i as f64).collect();
            let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(NdArray { data: arr });
        }
        let shape = self.data.shape().to_vec();
        let data_vec: Vec<f64> = self.data.iter().copied().collect();
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
                indexed.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        let arr = Array::from_shape_vec(IxDyn(&shape), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }
}

fn binary_op<F>(
    a: &NdArray,
    b: &Bound<'_, PyAny>,
    op: F,
) -> PyResult<NdArray>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    if let Ok(scalar) = b.extract::<f64>() {
        return Ok(NdArray {
            data: a.data.mapv(|x| op(x, scalar)),
        });
    }
    if let Ok(other) = b.extract::<NdArray>() {
        let result = broadcast_binary_op(&a.data, &other.data, op)?;
        return Ok(NdArray { data: result });
    }
    Err(PyTypeError::new_err("Unsupported operand type"))
}

fn binary_op_lr<F>(
    a: &NdArray,
    b: &Bound<'_, PyAny>,
    op: F,
) -> PyResult<NdArray>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    if let Ok(scalar) = b.extract::<f64>() {
        return Ok(NdArray {
            data: a.data.mapv(|x| op(x, scalar)),
        });
    }
    if let Ok(other) = b.extract::<NdArray>() {
        let result = broadcast_binary_op(&a.data, &other.data, op)?;
        return Ok(NdArray { data: result });
    }
    Err(PyTypeError::new_err("Unsupported operand type"))
}

fn broadcast_binary_op<F>(
    a: &Array<f64, IxDyn>,
    b: &Array<f64, IxDyn>,
    op: F,
) -> PyResult<Array<f64, IxDyn>>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();

    if a_shape == b_shape {
        // 并行计算：收集到 Vec 后使用 rayon 并行处理
        let a_vec: Vec<f64> = a.iter().copied().collect();
        let b_vec: Vec<f64> = b.iter().copied().collect();
        let result: Vec<f64> = a_vec
            .into_par_iter()
            .zip(b_vec.into_par_iter())
            .map(|(x, y)| op(x, y))
            .collect();
        return Array::from_shape_vec(IxDyn(&a_shape), result)
            .map_err(|e| PyValueError::new_err(e.to_string()));
    }

    let max_ndim = ::std::cmp::max(a_shape.len(), b_shape.len());
    let mut a_padded = vec![1usize; max_ndim - a_shape.len()];
    a_padded.extend_from_slice(&a_shape);
    let mut b_padded = vec![1usize; max_ndim - b_shape.len()];
    b_padded.extend_from_slice(&b_shape);

    let mut out_shape = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        if a_padded[i] != b_padded[i] && a_padded[i] != 1 && b_padded[i] != 1 {
            return Err(PyValueError::new_err(format!(
                "Incompatible shapes for broadcasting: {:?} and {:?}",
                a_shape, b_shape
            )));
        }
        out_shape.push(::std::cmp::max(a_padded[i], b_padded[i]));
    }

    let a_broadcast = a
        .clone()
        .into_shape_with_order(IxDyn(&a_padded))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .broadcast(IxDyn(&out_shape))
        .ok_or_else(|| PyValueError::new_err("Broadcasting failed"))?
        .to_owned();
    let b_broadcast = b
        .clone()
        .into_shape_with_order(IxDyn(&b_padded))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .broadcast(IxDyn(&out_shape))
        .ok_or_else(|| PyValueError::new_err("Broadcasting failed"))?
        .to_owned();

    let a_vec: Vec<f64> = a_broadcast.iter().copied().collect();
    let b_vec: Vec<f64> = b_broadcast.iter().copied().collect();
    let result: Vec<f64> = a_vec
        .into_par_iter()
        .zip(b_vec.into_par_iter())
        .map(|(x, y)| op(x, y))
        .collect();

    Array::from_shape_vec(IxDyn(&out_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))
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

// ===== Module-level Functions =====

#[pyfunction]
fn array(data: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    NdArray::new(data)
}

#[pyfunction]
fn zeros(shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let s = shape_to_vec(shape)?;
    Ok(NdArray {
        data: Array::zeros(IxDyn(&s)),
    })
}

#[pyfunction]
fn ones(shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let s = shape_to_vec(shape)?;
    Ok(NdArray {
        data: Array::ones(IxDyn(&s)),
    })
}

#[pyfunction]
#[pyo3(signature = (n, m=None, k=0))]
fn eye(n: usize, m: Option<usize>, k: i32) -> PyResult<NdArray> {
    let cols = m.unwrap_or(n);
    let mut data = vec![0.0_f64; n * cols];
    if k >= 0 {
        let k_u = k as usize;
        for i in 0..n {
            let j = i + k_u;
            if j < cols {
                data[i * cols + j] = 1.0;
            }
        }
    } else {
        let k_u = (-k) as usize;
        for j in 0..cols {
            let i = j + k_u;
            if i < n {
                data[i * cols + j] = 1.0;
            }
        }
    }
    let arr = Array::from_shape_vec((n, cols), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        data: arr.into_dyn(),
    })
}

#[pyfunction]
#[pyo3(signature = (start, stop, step=1.0))]
fn arange(start: f64, stop: f64, step: f64) -> PyResult<NdArray> {
    if step == 0.0 {
        return Err(PyValueError::new_err("Step cannot be zero"));
    }
    let mut values = Vec::new();
    if step > 0.0 {
        let n = ((stop - start) / step).ceil() as i64;
        for i in 0..n {
            let val = start + step * i as f64;
            if val < stop {
                values.push(val);
            }
        }
    } else {
        let n = ((start - stop) / (-step)).ceil() as i64;
        for i in 0..n {
            let val = start + step * i as f64;
            if val > stop {
                values.push(val);
            }
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[values.len()]), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=true))]
fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> PyResult<NdArray> {
    if num == 0 {
        return Err(PyValueError::new_err("Number of samples must be positive"));
    }
    if num == 1 {
        return Ok(NdArray {
            data: Array::from_shape_vec(IxDyn(&[1]), vec![start])
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        });
    }
    let step = if endpoint {
        (stop - start) / (num - 1) as f64
    } else {
        (stop - start) / num as f64
    };
    let values: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
    let arr = Array::from_shape_vec(IxDyn(&[num]), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
fn full(shape: &Bound<'_, PyAny>, fill_value: f64) -> PyResult<NdArray> {
    let s = shape_to_vec(shape)?;
    Ok(NdArray {
        data: Array::from_elem(IxDyn(&s), fill_value),
    })
}

#[pyfunction]
fn empty(shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let s = shape_to_vec(shape)?;
    let size: usize = s.iter().product();
    // 仅分配内存，不初始化，比 zeros 更快
    let mut v: Vec<f64> = Vec::with_capacity(size);
    // SAFETY: f64 所有位模式都是有效的
    unsafe { v.set_len(size); }
    let data = Array::from_shape_vec(IxDyn(&s), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data })
}

// ===== Math Functions =====

fn unary_math_op(x: &NdArray, op: fn(f64) -> f64) -> NdArray {
    let data = x.data.clone();
    let result_vec: Vec<f64> = data
        .into_par_iter()
        .map(|v| op(*v))
        .collect();
    NdArray {
        data: Array::from_shape_vec(IxDyn(x.data.shape()), result_vec)
            .unwrap_or_else(|_| x.data.mapv(op)),
    }
}

macro_rules! define_math_func {
    ($name:ident, $op:expr) => {
        #[pyfunction]
        fn $name(x: &NdArray) -> PyResult<NdArray> {
            Ok(unary_math_op(x, $op))
        }
    };
}

define_math_func!(sin, |v| v.sin());
define_math_func!(cos, |v| v.cos());
define_math_func!(tan, |v| v.tan());
define_math_func!(sqrt, |v| v.sqrt());
define_math_func!(exp, |v| v.exp());
define_math_func!(log, |v| v.ln());
define_math_func!(log10, |v| v.log10());
define_math_func!(log2, |v| v.log2());
define_math_func!(log1p, |v| v.ln_1p());
define_math_func!(abs, |v| v.abs());
define_math_func!(cosh, |v| v.cosh());
define_math_func!(sinh, |v| v.sinh());
define_math_func!(tanh, |v| v.tanh());
define_math_func!(acosh, |v| v.acosh());
define_math_func!(asinh, |v| v.asinh());
define_math_func!(atanh, |v| v.atanh());
define_math_func!(asin, |v| v.asin());
define_math_func!(acos, |v| v.acos());
define_math_func!(atan, |v| v.atan());
define_math_func!(expm1, |v| v.exp_m1());

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn sum(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    x.sum(axis)
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn mean(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    x.mean(axis)
}

#[pyfunction]
#[pyo3(signature = (x, axis=None), name = "std")]
fn std_dev(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    x.std(axis)
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn var(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    x.var(axis)
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn min(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    x.min(axis)
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn max(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    x.max(axis)
}

#[pyfunction]
fn argmin(x: &NdArray) -> PyResult<usize> {
    x.argmin()
}

#[pyfunction]
fn argmax(x: &NdArray) -> PyResult<usize> {
    x.argmax()
}

// ===== Array Operations =====

#[pyfunction]
fn concatenate(arrays: &Bound<'_, PyAny>, axis: usize) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut ndarrays: Vec<Array<f64, IxDyn>> = Vec::new();
    for item in list.iter() {
        let nd = item.extract::<NdArray>()?;
        ndarrays.push(nd.data);
    }
    if ndarrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let views: Vec<_> = ndarrays.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        data: result.into_dyn(),
    })
}

#[pyfunction]
fn stack(arrays: &Bound<'_, PyAny>, axis: usize) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut ndarrays: Vec<Array<f64, IxDyn>> = Vec::new();
    for item in list.iter() {
        let nd = item.extract::<NdArray>()?;
        ndarrays.push(nd.data);
    }
    if ndarrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let orig_shape = ndarrays[0].shape().to_vec();
    for arr in &ndarrays[1..] {
        if arr.shape() != orig_shape.as_slice() {
            return Err(PyValueError::new_err("All arrays must have the same shape"));
        }
    }
    let mut flat_result = Array::zeros(IxDyn(&[0]));
    for (i, arr) in ndarrays.iter().enumerate() {
        let mut s = orig_shape.clone();
        s.insert(axis, 1);
        let expanded = arr
            .clone()
            .into_shape_with_order(IxDyn(&s))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if i == 0 {
            flat_result = expanded;
        } else {
            flat_result =
                ndarray::concatenate(Axis(axis), &[flat_result.view(), expanded.view()])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    Ok(NdArray {
        data: flat_result.into_dyn(),
    })
}

#[pyfunction]
fn transpose(a: &NdArray) -> PyResult<NdArray> {
    a.t()
}

#[pyfunction]
fn swapaxes(a: &NdArray, axis1: usize, axis2: usize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    if axis1 >= ndim || axis2 >= ndim {
        return Err(PyValueError::new_err(format!(
            "axis out of bounds: arr.ndim={}, axis1={}, axis2={}",
            ndim, axis1, axis2
        )));
    }
    let mut arr = a.data.clone();
    arr.swap_axes(axis1, axis2);
    Ok(NdArray { data: arr })
}

// ===== Logic / Selection Functions =====

#[pyfunction]
#[pyo3(name = "where", signature = (condition, x=None, y=None))]
fn where_<'py>(py: Python<'py>, condition: &NdArray, x: Option<&NdArray>, y: Option<&NdArray>) -> PyResult<Bound<'py, PyAny>> {
    match (x, y) {
        (None, None) => {
            let shape = condition.data.shape();
            let ndim = shape.len();
            let mut indices: Vec<Vec<usize>> = (0..ndim.max(1)).map(|_| Vec::new()).collect();
            for (flat_idx, &val) in condition.data.iter().enumerate() {
                if val != 0.0 {
                    if ndim == 0 {
                        indices[0].push(0);
                    } else {
                        let mut remaining = flat_idx;
                        for d in (0..ndim).rev() {
                            let dim_size = shape[d];
                            indices[d].push(remaining % dim_size);
                            remaining /= dim_size;
                        }
                    }
                }
            }
            let tuples: Vec<Bound<'py, PyAny>> = indices.into_iter().map(|idx_vec| {
                let data: Vec<f64> = idx_vec.iter().map(|&i| i as f64).collect();
                let arr = Array::from_shape_vec(IxDyn(&[data.len()]), data).unwrap();
                Bound::new(py, NdArray { data: arr }).unwrap().into_any()
            }).collect();
            Ok(PyTuple::new(py, tuples)?.into_any())
        }
        (Some(xv), Some(yv)) => {
            let cond_vec: Vec<f64> = condition.data.iter().copied().collect();
            let x_vec: Vec<f64> = xv.data.iter().copied().collect();
            let y_vec: Vec<f64> = yv.data.iter().copied().collect();
            let result: Vec<f64> = cond_vec.into_par_iter()
                .zip(x_vec.into_par_iter().zip(y_vec.into_par_iter()))
                .map(|(c, (xv, yv))| if c != 0.0 { xv } else { yv })
                .collect();
            let arr = Array::from_shape_vec(condition.data.dim(), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let nd = NdArray { data: arr.into_dyn() };
            let bound = Bound::new(py, nd)?;
            Ok(bound.into_any())
        }
        _ => Err(PyTypeError::new_err("Both x and y must be provided together")),
    }
}

#[pyfunction]
fn clip(a: &NdArray, a_min: f64, a_max: f64) -> PyResult<NdArray> {
    a.clip(a_min, a_max)
}

#[pyfunction]
fn unique(a: &NdArray) -> PyResult<NdArray> {
    let mut values: Vec<u64> = a.data.iter().map(|v| v.to_bits()).collect();
    values.sort();
    values.dedup();
    let unique_vals: Vec<f64> = values.into_iter().map(|b| f64::from_bits(b)).collect();
    let arr = Array::from_shape_vec(IxDyn(&[unique_vals.len()]), unique_vals)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
#[pyo3(signature = (a, axis=-1))]
fn sort(a: &NdArray, axis: isize) -> PyResult<NdArray> {
    a.sort(axis)
}

#[pyfunction]
#[pyo3(signature = (a, axis=-1))]
fn argsort(a: &NdArray, axis: isize) -> PyResult<NdArray> {
    a.argsort(axis)
}

#[pyfunction]
#[pyo3(signature = (a, kth, axis=-1))]
fn partition(a: &NdArray, kth: Vec<i64>, axis: isize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };

    if ndim == 0 {
        return Ok(NdArray {
            data: a.data.clone(),
        });
    }

    let shape = a.data.shape().to_vec();
    let ax_len = shape[ax];

    let mut kths: Vec<usize> = kth.iter().map(|&k| {
        let mut kk = k;
        if kk < 0 {
            kk += ax_len as i64;
        }
        if kk < 0 || kk >= ax_len as i64 {
            Err(PyValueError::new_err("kth out of bounds"))
        } else {
            Ok(kk as usize)
        }
    }).collect::<Result<Vec<_>, _>>()?;
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
    Ok(NdArray { data: result })
}

#[allow(unused)]
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn introselect(arr: &mut Vec<f64>, k: usize) -> f64 {
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
#[pyo3(signature = (x, axis=None))]
fn median(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    match axis {
        None => {
            let mut values: Vec<f64> = x.data.iter().copied().collect();
            if values.is_empty() {
                return Err(PyValueError::new_err("Cannot compute median of empty array"));
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let len = values.len();
            let median_val = if len % 2 == 0 {
                (values[len / 2 - 1] + values[len / 2]) / 2.0
            } else {
                values[len / 2]
            };
            Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), median_val),
            })
        }
        Some(ax) => {
            let ndim = x.data.ndim();
            let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            
            let nrows = x.data.shape()[0];
            let ncols = x.data.shape()[1];
            
            let result: Vec<f64>;
            
            if ax == 0 {
                result = (0..ncols)
                    .map(|c| {
                        let mut values: Vec<f64> = (0..nrows)
                            .map(|r| x.data[[r, c]])
                            .collect();
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let len = values.len();
                        if len % 2 == 0 {
                            (values[len / 2 - 1] + values[len / 2]) / 2.0
                        } else {
                            values[len / 2]
                        }
                    })
                    .collect();
            } else {
                result = (0..nrows)
                    .map(|r| {
                        let mut values: Vec<f64> = (0..ncols)
                            .map(|c| x.data[[r, c]])
                            .collect();
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let len = values.len();
                        if len % 2 == 0 {
                            (values[len / 2 - 1] + values[len / 2]) / 2.0
                        } else {
                            values[len / 2]
                        }
                    })
                    .collect();
            }
            
            let mut new_shape: Vec<usize> = x.data.shape().iter().copied().collect();
            new_shape.remove(ax);
            let result_arr = Array::from_shape_vec(IxDyn(&new_shape), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: result_arr })
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, weights=None, returned=false))]
fn average(a: &NdArray, axis: Option<isize>, weights: Option<&NdArray>, returned: bool) -> PyResult<NdArray> {
    match axis {
        None => {
            let values: Vec<f64> = a.data.iter().copied().collect();
            
            let (result_val, sum_weights): (f64, f64) = match weights {
                None => {
                    let sum: f64 = values.iter().sum();
                    let avg = sum / values.len() as f64;
                    if returned {
                        (avg, values.len() as f64)
                    } else {
                        (avg, 0.0)
                    }
                }
                Some(w) => {
                    let w_values: Vec<f64> = w.data.iter().copied().collect();
                    if values.len() != w_values.len() {
                        return Err(PyValueError::new_err("weights must have the same length as data"));
                    }
                    let weighted_sum: f64 = values.iter().zip(w_values.iter()).map(|(v, w)| v * w).sum();
                    let sum_w: f64 = w_values.iter().sum();
                    if sum_w == 0.0 {
                        return Err(PyValueError::new_err("sum of weights must not be zero"));
                    }
                    let avg = weighted_sum / sum_w;
                    if returned {
                        (avg, sum_w)
                    } else {
                        (avg, 0.0)
                    }
                }
            };
            
            if returned {
                let data = vec![result_val, sum_weights];
                Ok(NdArray {
                    data: Array::from_shape_vec(IxDyn(&[2]), data)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                })
            } else {
                Ok(NdArray {
                    data: Array::from_elem(IxDyn(&[]), result_val),
                })
            }
        }
        Some(ax) => {
            let ndim = a.data.ndim();
            let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            
            let w_values: Option<Vec<f64>> = weights.map(|w| w.data.iter().copied().collect());
            
            let axis_len = a.data.shape()[ax];
            let other_dims: Vec<usize> = a.data.shape().iter().enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &s)| s)
                .collect();
            let num_other: usize = other_dims.iter().product();
            
            let mut avg_results: Vec<f64> = Vec::with_capacity(num_other);
            let mut sum_w_results: Vec<f64> = Vec::with_capacity(num_other);
            
            for idx in 0..num_other {
                let mut indices: Vec<usize> = Vec::with_capacity(ndim);
                let mut temp = idx;
                for &dim in other_dims.iter().rev() {
                    indices.push(temp % dim);
                    temp /= dim;
                }
                indices.reverse();
                
                let mut values: Vec<f64> = Vec::with_capacity(axis_len);
                for i in 0..axis_len {
                    let mut full_indices = indices.clone();
                    full_indices.insert(ax, i);
                    values.push(a.data[full_indices.as_slice()]);
                }
                
                let (avg, sum_w) = match &w_values {
                    None => {
                        let sum: f64 = values.iter().sum();
                        (sum / values.len() as f64, values.len() as f64)
                    }
                    Some(w) => {
                        if values.len() != w.len() {
                            return Err(PyValueError::new_err("weights must have the same length as the axis dimension"));
                        }
                        let weighted_sum: f64 = values.iter().zip(w.iter()).map(|(v, w)| v * w).sum();
                        let sum_w_val: f64 = w.iter().sum();
                        if sum_w_val == 0.0 {
                            return Err(PyValueError::new_err("sum of weights must not be zero"));
                        }
                        (weighted_sum / sum_w_val, sum_w_val)
                    }
                };
                
                avg_results.push(avg);
                sum_w_results.push(sum_w);
            }
            
            if returned {
                let combined_len = avg_results.len() * 2;
                let mut combined: Vec<f64> = Vec::with_capacity(combined_len);
                for i in 0..avg_results.len() {
                    combined.push(avg_results[i]);
                    combined.push(sum_w_results[i]);
                }
                let mut result_shape = other_dims.clone();
                result_shape.push(2);
                Ok(NdArray {
                    data: Array::from_shape_vec(IxDyn(&result_shape), combined)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                })
            } else {
                Ok(NdArray {
                    data: Array::from_shape_vec(IxDyn(&other_dims), avg_results)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                })
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (x, q, axis=None, keepdims=false))]
fn percentile(x: &NdArray, q: f64, axis: Option<isize>, keepdims: bool) -> PyResult<NdArray> {
    if q < 0.0 || q > 100.0 {
        return Err(PyValueError::new_err("Percentile must be between 0 and 100"));
    }

    match axis {
        None => {
            let mut values: Vec<f64> = x.data.iter().copied().collect();
            if values.is_empty() {
                return Err(PyValueError::new_err("Cannot compute percentile of empty array"));
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let len = values.len();
            let idx = q / 100.0 * (len - 1) as f64;
            let lower = idx.floor() as usize;
            let upper = idx.ceil() as usize;
            if lower == upper || upper >= len {
                let val = values[lower.min(len - 1)];
                return Ok(NdArray {
                    data: Array::from_elem(IxDyn(&[]), val),
                });
            }
            let frac = idx - lower as f64;
            let val = values[lower] * (1.0 - frac) + values[upper] * frac;
            Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), val),
            })
        }
        Some(ax) => {
            let ndim = x.data.ndim();
            let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            
            let nrows = x.data.shape()[0];
            let ncols = x.data.shape()[1];
            
            let result: Vec<f64>;
            
            if ax == 0 {
                result = (0..ncols)
                    .map(|c| {
                        let mut values: Vec<f64> = (0..nrows)
                            .map(|r| x.data[[r, c]])
                            .collect();
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let len = values.len();
                        let idx = q / 100.0 * (len - 1) as f64;
                        let lower = idx.floor() as usize;
                        let upper = idx.ceil() as usize;
                        if lower == upper || upper >= len {
                            values[lower.min(len - 1)]
                        } else {
                            let frac = idx - lower as f64;
                            values[lower] * (1.0 - frac) + values[upper] * frac
                        }
                    })
                    .collect();
            } else {
                result = (0..nrows)
                    .map(|r| {
                        let mut values: Vec<f64> = (0..ncols)
                            .map(|c| x.data[[r, c]])
                            .collect();
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let len = values.len();
                        let idx = q / 100.0 * (len - 1) as f64;
                        let lower = idx.floor() as usize;
                        let upper = idx.ceil() as usize;
                        if lower == upper || upper >= len {
                            values[lower.min(len - 1)]
                        } else {
                            let frac = idx - lower as f64;
                            values[lower] * (1.0 - frac) + values[upper] * frac
                        }
                    })
                    .collect();
            }
            
            let mut new_shape: Vec<usize> = x.data.shape().iter().copied().collect();
            if keepdims {
                new_shape[ax] = 1;
            } else {
                new_shape.remove(ax);
            }
            let result_arr = Array::from_shape_vec(IxDyn(&new_shape), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: result_arr })
        }
    }
}

#[pyfunction]
#[pyo3(signature = (*args, indexing="xy"))]
fn meshgrid<'py>(py: Python<'py>, args: &Bound<'_, PyTuple>, indexing: &str) -> PyResult<Bound<'py, PyAny>> {
    let n = args.len();
    if n == 0 {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let mut arrays: Vec<NdArray> = Vec::new();
    for item in args.iter() {
        let arr = item.extract::<NdArray>()?;
        if arr.data.ndim() != 1 {
            return Err(PyValueError::new_err("All inputs must be 1-D arrays"));
        }
        arrays.push(arr);
    }
    let dims: Vec<usize> = arrays.iter().map(|a| a.data.len()).collect();
    let (swap_first_two, out_dims): (bool, Vec<usize>) = if indexing == "ij" {
        (false, dims.clone())
    } else {
        let mut out = dims.clone();
        if out.len() >= 2 {
            out.swap(0, 1);
        }
        (true, out)
    };
    let mut results: Vec<NdArray> = Vec::new();
    for i in 0..n {
        let mut shape = vec![1usize; n];
        let actual_idx = if swap_first_two && n >= 2 {
            if i == 0 { 1 } else if i == 1 { 0 } else { i }
        } else {
            i
        };
        shape[actual_idx] = arrays[i].data.len();
        let reshaped = arrays[i].data.clone().into_shape_with_order(IxDyn(&shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let broadcast = reshaped.broadcast(IxDyn(&out_dims))
            .ok_or_else(|| PyValueError::new_err("Broadcasting failed"))?
            .to_owned();
        results.push(NdArray { data: broadcast.into_dyn() });
    }
    let py_results: Vec<Bound<'py, PyAny>> = results.into_iter()
        .map(|nd| Bound::new(py, nd).unwrap().into_any())
        .collect();
    Ok(PyTuple::new(py, py_results)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (a, bins=10, range=None))]
fn histogram<'py>(py: Python<'py>, a: &NdArray, bins: usize, range: Option<&Bound<'_, PyAny>>) -> PyResult<Bound<'py, PyTuple>> {
    let values: Vec<f64> = a.data.iter().copied().collect();
    if values.is_empty() {
        return Err(PyValueError::new_err("Empty array"));
    }
    let (min_val, max_val) = if let Some(r) = range {
        let range_tuple = r.cast::<PyTuple>().map_err(|_| PyTypeError::new_err("Range must be a tuple"))?;
        if range_tuple.len() != 2 {
            return Err(PyValueError::new_err("Range must have exactly 2 elements"));
        }
        let low = range_tuple.get_item(0)?.extract::<f64>()?;
        let high = range_tuple.get_item(1)?.extract::<f64>()?;
        (low, high)
    } else {
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    };
    let bin_width = (max_val - min_val) / bins as f64;
    if bin_width == 0.0 {
        let hist_arr = Array::from_shape_vec(IxDyn(&[1]), vec![values.len() as f64])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let edges_arr = Array::from_shape_vec(IxDyn(&[2]), vec![min_val, max_val])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(PyTuple::new(py, vec![
            Bound::new(py, NdArray { data: hist_arr.into_dyn() })?.into_any(),
            Bound::new(py, NdArray { data: edges_arr.into_dyn() })?.into_any(),
        ])?);
    }
    let mut hist = vec![0.0_f64; bins];
    let bin_edges: Vec<f64> = (0..=bins).map(|i| min_val + i as f64 * bin_width).collect();
    for &v in &values {
        if v < min_val || v > max_val {
            continue;
        }
        let mut idx = ((v - min_val) / bin_width) as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        hist[idx] += 1.0;
    }
    let hist_arr = Array::from_shape_vec(IxDyn(&[bins]), hist)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let edges_arr = Array::from_shape_vec(IxDyn(&[bins + 1]), bin_edges)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTuple::new(py, vec![
        Bound::new(py, NdArray { data: hist_arr.into_dyn() })?.into_any(),
        Bound::new(py, NdArray { data: edges_arr.into_dyn() })?.into_any(),
    ])?)
}

#[pyfunction]
fn gradient(f: &NdArray) -> PyResult<NdArray> {
    let values: Vec<f64> = f.data.iter().copied().collect();
    let n = values.len();
    if n == 0 {
        return Err(PyValueError::new_err("Empty array"));
    }
    if n == 1 {
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), 0.0),
        });
    }
    let mut grad = Vec::with_capacity(n);
    grad.push(values[1] - values[0]);
    for i in 1..n - 1 {
        grad.push((values[i + 1] - values[i - 1]) / 2.0);
    }
    grad.push(values[n - 1] - values[n - 2]);
    let arr = Array::from_shape_vec(IxDyn(&[n]), grad)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
#[pyo3(signature = (a, n=1, axis=-1))]
fn diff(a: &NdArray, n: isize, axis: isize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    if ax >= ndim {
        return Err(PyValueError::new_err("Axis out of bounds"));
    }
    let mut current = a.data.clone();
    for _ in 0..n {
        if current.shape()[ax] <= 1 {
            return Err(PyValueError::new_err("Cannot compute diff: axis has less than 2 elements"));
        }
        let shape = current.shape().to_vec();
        let data_vec: Vec<f64> = current.iter().copied().collect();
        let pre_size: usize = shape.iter().take(ax).product();
        let axis_size = shape[ax];
        let post_size: usize = shape.iter().skip(ax + 1).product();
        let block_size = axis_size * post_size;
        let mut new_data = Vec::with_capacity(data_vec.len() - pre_size * post_size);
        for outer in 0..pre_size {
            for inner in 0..post_size {
                let base = outer * block_size + inner;
                for k in 0..axis_size - 1 {
                    new_data.push(data_vec[base + (k + 1) * post_size] - data_vec[base + k * post_size]);
                }
            }
        }
        let mut new_shape = shape.clone();
        new_shape[ax] = axis_size - 1;
        current = Array::from_shape_vec(IxDyn(&new_shape), new_data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    Ok(NdArray { data: current })
}

#[pyfunction]
#[pyo3(signature = (y, dx=1.0))]
fn trapz(y: &NdArray, dx: f64) -> PyResult<NdArray> {
    let values: Vec<f64> = y.data.iter().copied().collect();
    let n = values.len();
    if n < 2 {
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), 0.0),
        });
    }
    let mut sum = 0.0;
    for i in 0..n - 1 {
        sum += (values[i] + values[i + 1]) / 2.0 * dx;
    }
    Ok(NdArray {
        data: Array::from_elem(IxDyn(&[]), sum),
    })
}

#[pyfunction]
fn cross(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let a_vec: Vec<f64> = a.data.iter().copied().collect();
    let b_vec: Vec<f64> = b.data.iter().copied().collect();
    if a_vec.len() != 3 || b_vec.len() != 3 {
        return Err(PyValueError::new_err("Cross product requires 3-element vectors"));
    }
    let result = vec![
        a_vec[1] * b_vec[2] - a_vec[2] * b_vec[1],
        a_vec[2] * b_vec[0] - a_vec[0] * b_vec[2],
        a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0],
    ];
    let arr = Array::from_shape_vec(IxDyn(&[3]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
fn all(a: &NdArray) -> bool {
    a.all()
}

#[pyfunction]
fn any(a: &NdArray) -> bool {
    a.any()
}

#[pyfunction]
fn nonzero(a: &NdArray) -> Vec<Vec<usize>> {
    a.nonzero()
}

/// 直接在 Rust 中将 nonzero 结果转换为 NdArray 数组，避免 Python 层 for 循环
#[pyfunction]
fn nonzero_arrs<'py>(py: Python<'py>, a: &NdArray) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let raw = a.nonzero();
    let mut result = Vec::with_capacity(raw.len());
    for indices in raw {
        let data: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&[data.len()]), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nd = NdArray { data: arr.into_dyn() };
        result.push(Bound::new(py, nd)?.into_any());
    }
    Ok(result)
}

/// ix_ 的 Rust 实现：从多个序列构造开放网格，消除 Python 层 for 循环
#[pyfunction]
fn ix_rs<'py>(py: Python<'py>, args: &Bound<'_, PyTuple>) -> PyResult<Bound<'py, PyTuple>> {
    let n = args.len();
    if n == 0 {
        return Ok(PyTuple::new(py, Vec::<Bound<'py, PyAny>>::new())?);
    }
    let mut arrays: Vec<NdArray> = Vec::with_capacity(n);
    for i in 0..n {
        let item = args.get_item(i)?;
        // 直接解析 Python list/tuple 为 NdArray（支持任意可迭代数据）
        let (values, shape) = parse_py_list_to_flat(&item)?;
        let nd_shape = if shape.is_empty() { IxDyn(&[]) } else { IxDyn(&shape) };
        let arr = Array::from_shape_vec(nd_shape, values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        arrays.push(NdArray { data: arr });
    }
    let mut results: Vec<Bound<'py, PyAny>> = Vec::with_capacity(n);
    for i in 0..n {
        let size = arrays[i].data.len();
        let mut shape = vec![1usize; n];
        shape[i] = size;
        let reshaped = arrays[i].data.clone().into_shape_with_order(IxDyn(&shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        results.push(Bound::new(py, NdArray { data: reshaped.into_dyn() })?.into_any());
    }
    Ok(PyTuple::new(py, results)?)
}

/// 从 shape 直接创建 arange 数组元组（用于 fromfunction 等），避免 Python 层 list(range(d)) 循环
#[pyfunction]
fn _arange_arrays<'py>(py: Python<'py>, shape: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let s = shape_to_vec(shape)?;
    let mut arrays: Vec<Bound<'py, PyAny>> = Vec::with_capacity(s.len());
    for &dim in &s {
        let data: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&[dim]), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nd = NdArray { data: arr.into_dyn() };
        arrays.push(Bound::new(py, nd)?.into_any());
    }
    Ok(PyTuple::new(py, arrays)?)
}

/// select 的 Rust 实现 — 单次遍历多条件选择，避免 Python 层 for 循环多次调用 where
#[pyfunction]
#[pyo3(signature = (condlist, choicelist, default=0.0))]
fn select_rs<'py>(
    py: Python<'py>,
    condlist: &Bound<'_, PyList>,
    choicelist: &Bound<'_, PyList>,
    default: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let n = condlist.len();
    if n == 0 {
        return Err(PyValueError::new_err("condlist must not be empty"));
    }
    if choicelist.len() != n {
        return Err(PyValueError::new_err("condlist and choicelist must have the same length"));
    }

    // 提取条件和选择数组（保持和 Python 原有代码一致的语义）
    let mut conds: Vec<NdArray> = Vec::with_capacity(n);
    let mut choices: Vec<NdArray> = Vec::with_capacity(n);

    // 先用第一个条件确定 shape
    let first_cond: NdArray = condlist.get_item(0)?.extract::<NdArray>()?;
    let result_shape = first_cond.data.shape().to_vec();
    let total_size: usize = result_shape.iter().product();
    conds.push(first_cond);

    for i in 1..n {
        conds.push(condlist.get_item(i)?.extract::<NdArray>()?);
    }
    for i in 0..n {
        choices.push(choicelist.get_item(i)?.extract::<NdArray>()?);
    }

    // 用第一个条件的 shape 初始化结果（默认值填充）
    let mut result = vec![default; total_size];

    // 遍历每个 (cond, choice) 对，最后一个匹配的条件获胜（与原有 Python 语义一致）
    for (cond, choice) in conds.iter().zip(choices.iter()) {
        let cond_slice = cond.data.as_slice()
            .ok_or_else(|| PyValueError::new_err("Non-contiguous condition array"))?;
        let choice_slice = choice.data.as_slice()
            .ok_or_else(|| PyValueError::new_err("Non-contiguous choice array"))?;
        let min_len = total_size.min(cond_slice.len()).min(choice_slice.len());
        for i in 0..min_len {
            if cond_slice[i] != 0.0 {
                result[i] = choice_slice[i];
            }
        }
    }

    let arr = Array::from_shape_vec(IxDyn(&result_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let nd = NdArray { data: arr.into_dyn() };
    let bound = Bound::new(py, nd)?;
    Ok(bound.into_any())
}

// ===== New Math / Utility Functions =====

#[pyfunction]
fn floor(x: &NdArray) -> NdArray {
    unary_math_op(x, |v| v.floor())
}

#[pyfunction]
fn ceil(x: &NdArray) -> NdArray {
    unary_math_op(x, |v| v.ceil())
}

#[pyfunction]
fn round(x: &NdArray, ndigits: Option<i32>) -> NdArray {
    let data = x.data.mapv(|v| match ndigits {
        Some(n) => {
            let factor = 10f64.powi(n);
            (v * factor).round() / factor
        }
        None => v.round(),
    });
    NdArray { data }
}

#[pyfunction]
fn isnan(x: &NdArray) -> NdArray {
    NdArray {
        data: x.data.mapv(|v| if v.is_nan() { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn isinf(x: &NdArray) -> NdArray {
    NdArray {
        data: x.data.mapv(|v| if v.is_infinite() { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn isfinite(x: &NdArray) -> NdArray {
    NdArray {
        data: x.data.mapv(|v| if v.is_finite() { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn invert(x: &NdArray) -> NdArray {
    NdArray {
        data: x.data.mapv(|v| if v == 0.0 { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn bitwise_and(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) & (b as i64);
        bits as f64
    })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn bitwise_or(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) | (b as i64);
        bits as f64
    })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn bitwise_xor(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) ^ (b as i64);
        bits as f64
    })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn left_shift(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) << (b as i64);
        bits as f64
    })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn right_shift(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) >> (b as i64);
        bits as f64
    })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn bitwise_not(x: &NdArray) -> NdArray {
    NdArray {
        data: x.data.mapv(|v| {
            let bits = !(v as i64);
            bits as f64
        }),
    }
}

#[pyfunction]
fn maximum(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.max(b))?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn minimum(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.min(b))?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn allclose(a: &NdArray, b: &NdArray, rtol: f64, atol: f64) -> bool {
    if a.data.len() != b.data.len() {
        return false;
    }
    for (va, vb) in a.data.iter().zip(b.data.iter()) {
        if (va - vb).abs() > atol + rtol * vb.abs() {
            return false;
        }
    }
    true
}

#[pyfunction]
fn array_equal(a: &NdArray, b: &NdArray) -> bool {
    if a.data.shape() != b.data.shape() {
        return false;
    }
    a.data.iter().zip(b.data.iter()).all(|(x, y)| x == y)
}

#[pyfunction]
fn argwhere(a: &NdArray) -> PyResult<NdArray> {
    let shape = a.data.shape();
    let ndim = shape.len();
    if ndim == 0 {
        if a.data.iter().any(|&v| v != 0.0) {
            let arr = Array::from_shape_vec(IxDyn(&[1, 0]), vec![])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(NdArray { data: arr });
        }
        let arr = Array::from_shape_vec(IxDyn(&[0, 0]), vec![])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { data: arr });
    }
    let mut indices: Vec<Vec<usize>> = (0..ndim).map(|_| Vec::new()).collect();
    for (flat_idx, &val) in a.data.iter().enumerate() {
        if val != 0.0 {
            let mut remaining = flat_idx;
            for d in (0..ndim).rev() {
                let dim_size = shape[d];
                indices[d].push(remaining % dim_size);
                remaining /= dim_size;
            }
        }
    }
    let num_nonzero = indices[0].len();
    let mut flat_result = Vec::with_capacity(num_nonzero * ndim);
    for i in 0..num_nonzero {
        for d in 0..ndim {
            flat_result.push(indices[d][i] as f64);
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[num_nonzero, ndim]), flat_result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
fn count_nonzero(a: &NdArray) -> usize {
    a.data.iter().filter(|&&v| v != 0.0).count()
}

#[pyfunction]
fn vstack(arrays: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut ndarrays: Vec<NdArray> = Vec::new();
    for item in list.iter() {
        ndarrays.push(item.extract::<NdArray>()?);
    }
    if ndarrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    
    // Check if all inputs are 1D
    let all_1d = ndarrays.iter().all(|a| a.data.ndim() == 1);
    
    if all_1d {
        // For 1D arrays, stack as rows to create a 2D array
        let ncols = ndarrays[0].data.len();
        for arr in &ndarrays[1..] {
            if arr.data.len() != ncols {
                return Err(PyValueError::new_err("All arrays must have the same length"));
            }
        }
        let nrows = ndarrays.len();
        let mut data = Vec::with_capacity(nrows * ncols);
        for arr in &ndarrays {
            for val in arr.data.iter() {
                data.push(*val);
            }
        }
        let arr = Array::from_shape_vec((nrows, ncols), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { data: arr.into_dyn() });
    }
    
    let ncols = ndarrays[0].data.shape().last().copied().unwrap_or(1);
    for arr in &ndarrays[1..] {
        let cols = arr.data.shape().last().copied().unwrap_or(1);
        if cols != ncols {
            return Err(PyValueError::new_err("All arrays must have the same number of columns"));
        }
    }
    let views: Vec<_> = ndarrays.iter().map(|a| a.data.view()).collect();
    let result = ndarray::concatenate(Axis(0), &views)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: result.into_dyn() })
}

#[pyfunction]
fn hstack(arrays: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut ndarrays: Vec<NdArray> = Vec::new();
    for item in list.iter() {
        ndarrays.push(item.extract::<NdArray>()?);
    }
    if ndarrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let ndim = ndarrays[0].data.ndim();
    let axis = if ndim <= 1 { 0 } else { 1 };
    let views: Vec<_> = ndarrays.iter().map(|a| a.data.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: result.into_dyn() })
}

#[pyfunction]
fn tile(a: &NdArray, reps: Vec<usize>) -> PyResult<NdArray> {
    if reps.is_empty() {
        return Ok(NdArray { data: a.data.clone() });
    }
    let shape = a.data.shape().to_vec();
    let mut result = a.data.clone();
    if reps.len() > shape.len() {
        let mut new_shape = vec![1usize; reps.len() - shape.len()];
        new_shape.extend_from_slice(&shape);
        result = result.into_shape_with_order(IxDyn(&new_shape))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    for (i, &rep) in reps.iter().enumerate() {
        if rep > 1 {
            let mut pieces = Vec::with_capacity(rep);
            for _ in 0..rep {
                pieces.push(result.view());
            }
            result = ndarray::concatenate(Axis(i), &pieces)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    Ok(NdArray { data: result.into_dyn() })
}

#[pyfunction]
fn squeeze(a: &NdArray) -> PyResult<NdArray> {
    a.squeeze()
}

#[pyfunction]
fn flatten(a: &NdArray) -> NdArray {
    a.flatten()
}

#[pyfunction]
fn reshape(a: &NdArray, shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    a.reshape(shape)
}

// ===== Additional Math Functions =====

define_math_func!(trunc, |v| v.trunc());
define_math_func!(fix, |v| v.trunc());
define_math_func!(square, |v| v * v);
define_math_func!(cbrt, |v| v.cbrt());
define_math_func!(sign, |v| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 });
define_math_func!(reciprocal, |v| 1.0 / v);

#[pyfunction]
fn arctan2(y: &NdArray, x: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&y.data, &x.data, |a, b| a.atan2(b))?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn deg2rad(x: &NdArray) -> NdArray {
    unary_math_op(x, |v| v.to_radians())
}

#[pyfunction]
fn rad2deg(x: &NdArray) -> NdArray {
    unary_math_op(x, |v| v.to_degrees())
}

#[pyfunction]
fn hypot(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.hypot(b))?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn sinc(x: &NdArray) -> NdArray {
    let pi = std::f64::consts::PI;
    NdArray {
        data: x.data.mapv(|v| {
            if v == 0.0 { 1.0 } else { (pi * v).sin() / (pi * v) }
        }),
    }
}

#[pyfunction]
fn heaviside(x: &NdArray, h0: f64) -> NdArray {
    NdArray {
        data: x.data.mapv(|v| {
            if v > 0.0 { 1.0 } else if v < 0.0 { 0.0 } else { h0 }
        }),
    }
}

#[pyfunction]
fn logspace(start: f64, stop: f64, num: usize) -> PyResult<NdArray> {
    linspace(start, stop, num, false).map(|nd| {
        NdArray { data: nd.data.mapv(|v| 10.0_f64.powf(v)) }
    })
}

#[pyfunction]
fn geomspace(start: f64, stop: f64, num: usize) -> PyResult<NdArray> {
    if num == 0 {
        return Err(PyValueError::new_err("Number of samples must be positive"));
    }
    if num == 1 {
        return Ok(NdArray {
            data: Array::from_shape_vec(IxDyn(&[1]), vec![start])
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        });
    }
    let log_start = start.ln();
    let log_stop = stop.ln();
    let step = (log_stop - log_start) / (num - 1) as f64;
    let values: Vec<f64> = (0..num).map(|i| (log_start + i as f64 * step).exp()).collect();
    let arr = Array::from_shape_vec(IxDyn(&[num]), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

// ===== Ufuncs (binary element-wise) =====

#[pyfunction]
fn add(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a + b)?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn subtract(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a - b)?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn multiply(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a * b)?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn divide(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a / b)?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn power(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.powf(b))?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn mod_(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a % b)?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn greater(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a > b { 1.0 } else { 0.0 })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn less(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a < b { 1.0 } else { 0.0 })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if (a - b).abs() < 1e-12 { 1.0 } else { 0.0 })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn not_equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if (a - b).abs() >= 1e-12 { 1.0 } else { 0.0 })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn greater_equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a >= b { 1.0 } else { 0.0 })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn less_equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a <= b { 1.0 } else { 0.0 })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn logical_and(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn logical_or(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn logical_xor(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if (a != 0.0) != (b != 0.0) { 1.0 } else { 0.0 })?;
    Ok(NdArray { data: result })
}

#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-05, atol=1e-08))]
fn isclose(a: &NdArray, b: &NdArray, rtol: f64, atol: f64) -> PyResult<NdArray> {
    let tol_atol = atol;
    let tol_rtol = rtol;
    let result = broadcast_binary_op(&a.data, &b.data, move |x, y| if (x - y).abs() <= tol_atol + tol_rtol * y.abs() { 1.0 } else { 0.0 })?;
    Ok(NdArray { data: result })
}

// ===== New Array Operations =====

#[pyfunction]
fn expand_dims(a: &NdArray, axis: isize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = if axis < 0 { (ndim as isize + axis + 1) as usize } else { axis as usize };
    let mut shape = a.data.shape().to_vec();
    shape.insert(ax, 1);
    let arr = a.data.clone().into_shape_with_order(IxDyn(&shape))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
fn column_stack(arrays: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let list = arrays.cast::<PyList>()?;
    let mut ndarrays: Vec<NdArray> = Vec::new();
    for item in list.iter() {
        ndarrays.push(item.extract::<NdArray>()?);
    }
    if ndarrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let reshaped: Vec<_> = ndarrays.iter().map(|arr| {
        if arr.data.ndim() == 1 {
            let shape = vec![arr.data.len(), 1];
            arr.data.clone().into_shape_with_order(IxDyn(&shape)).unwrap().into_dyn()
        } else {
            arr.data.clone()
        }
    }).collect();
    let views: Vec<_> = reshaped.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(1), &views)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: result.into_dyn() })
}

#[pyfunction]
fn roll(a: &NdArray, shift: isize, axis: Option<isize>) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    let ax = match axis {
        Some(ax) => if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize },
        None => 0,
    };
    if ndim == 0 {
        return Ok(NdArray { data: a.data.clone() });
    }
    let shape = a.data.shape().to_vec();
    let axis_size = shape[ax] as isize;
    let shift = shift.rem_euclid(axis_size);
    let data_vec: Vec<f64> = a.data.iter().copied().collect();
    let pre_size: usize = shape.iter().take(ax).product();
    let post_size: usize = shape.iter().skip(ax + 1).product();
    let block_size = axis_size as usize * post_size;
    let mut result = Vec::with_capacity(data_vec.len());
    for outer in 0..pre_size {
        for k in 0..axis_size as usize {
            let src = ((k as isize - shift).rem_euclid(axis_size)) as usize;
            for inner in 0..post_size {
                result.push(data_vec[outer * block_size + src * post_size + inner]);
            }
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
fn rot90(a: &NdArray, k: isize) -> PyResult<NdArray> {
    let ndim = a.data.ndim();
    if ndim < 2 {
        return Err(PyValueError::new_err("rot90 requires at least 2D array"));
    }
    let k = k.rem_euclid(4);
    let mut result = a.data.clone();
    for _ in 0..k {
        let shape = result.shape().to_vec();
        let rows = shape[0];
        let cols = shape[1];
        let mut new_data = vec![0.0; rows * cols];
        // 逆时针旋转 90°：new[i][j] = old[j][cols-1-i]
        for i in 0..cols {
            for j in 0..rows {
                new_data[i * rows + j] = result[[j, cols - 1 - i]];
            }
        }
        result = Array::from_shape_vec((cols, rows), new_data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_dyn();
    }
    Ok(NdArray { data: result })
}

#[pyfunction]
fn flatnonzero(a: &NdArray) -> PyResult<NdArray> {
    let indices: Vec<f64> = a.data.iter().enumerate()
        .filter(|(_, v)| **v != 0.0)
        .map(|(i, _)| i as f64)
        .collect();
    let arr = Array::from_shape_vec(IxDyn(&[indices.len()]), indices)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn ptp(x: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    match axis {
        None => {
            let (min_val, max_val) = x.data.iter().cloned().fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(min, max), v| (min.min(v), max.max(v)),
            );
            Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), max_val - min_val),
            })
        }
        Some(ax) => {
            let ndim = x.data.ndim();
            let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            let min_vals: Array<f64, IxDyn> = x
                .data
                .fold_axis(Axis(ax), f64::INFINITY, |acc, &v| {
                    let a = *acc;
                    if v < a { v } else { a }
                })
                .into_dyn();
            let max_vals: Array<f64, IxDyn> = x
                .data
                .fold_axis(Axis(ax), f64::NEG_INFINITY, |acc, &v| {
                    let a = *acc;
                    if v > a { v } else { a }
                })
                .into_dyn();
            let result: Vec<f64> = min_vals
                .iter()
                .zip(max_vals.iter())
                .map(|(min_val, max_val)| max_val - min_val)
                .collect();
            let result_arr = Array::from_shape_vec(IxDyn(min_vals.shape()), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: result_arr })
        }
    }
}

#[pyfunction]
fn digitize(x: &NdArray, bins: &NdArray) -> PyResult<NdArray> {
    let x_vals: Vec<f64> = x.data.iter().copied().collect();
    let mut bin_edges: Vec<f64> = bins.data.iter().copied().collect();
    bin_edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // 二分查找 + 并行计算：O(n*log m)，原先线性 O(n*m)
    let result: Vec<f64> = x_vals
        .par_iter()
        .map(|&v| bin_edges.partition_point(|&edge| v >= edge) as f64)
        .collect();
    let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
fn broadcast_to(a: &NdArray, shape: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let s = shape_to_vec(shape)?;
    let broadcast = a.data.broadcast(IxDyn(&s))
        .ok_or_else(|| PyValueError::new_err(format!("Cannot broadcast shape {:?} to {:?}", a.data.shape(), s)))?
        .to_owned();
    Ok(NdArray { data: broadcast })
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

// ========== 翻转数组 ==========
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn flip(a: &NdArray, axis: Option<isize>) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    let ndim = shape.len();
    let data_vec: Vec<f64> = a.data.iter().copied().collect();

    match axis {
        None => {
            let mut result = vec![0.0; data_vec.len()];
            for (i, &v) in data_vec.iter().rev().enumerate() {
                result[i] = v;
            }
            let arr = Array::from_shape_vec(IxDyn(&shape), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: arr })
        }
        Some(ax) => {
            let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
            if ax >= ndim {
                return Err(PyValueError::new_err(format!("axis {} out of bounds", ax)));
            }
            let axis_size = shape[ax];
            let pre: usize = shape.iter().take(ax).product();
            let post: usize = shape.iter().skip(ax + 1).product();
            let mut result = vec![0.0; data_vec.len()];

            for p in 0..pre {
                for i in 0..axis_size {
                    let src = p * axis_size * post + i * post;
                    let dst = p * axis_size * post + (axis_size - 1 - i) * post;
                    for k in 0..post {
                        result[dst + k] = data_vec[src + k];
                    }
                }
            }
            let arr = Array::from_shape_vec(IxDyn(&shape), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(NdArray { data: arr })
        }
    }
}

// ========== 拆分数组 ==========
#[pyfunction]
#[pyo3(signature = (a, indices_or_sections, axis=0))]
fn split_rs(a: &NdArray, indices_or_sections: &Bound<'_, PyAny>, axis: isize) -> PyResult<Vec<NdArray>> {
    let shape = a.data.shape().to_vec();
    let ndim = shape.len();
    let ax = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    if ax >= ndim {
        return Err(PyValueError::new_err(format!("axis {} out of bounds", axis)));
    }
    let axis_size = shape[ax];

    let sections: Vec<usize> = if let Ok(n) = indices_or_sections.extract::<usize>() {
        (1..n).map(|i| (axis_size * i) / n).collect()
    } else if let Ok(indices) = indices_or_sections.extract::<Vec<isize>>() {
        indices.iter().map(|&v| if v < 0 { (axis_size as isize + v) as usize } else { v as usize }).filter(|&v| v <= axis_size).collect()
    } else if let Ok(indices) = indices_or_sections.extract::<Vec<i64>>() {
        indices.iter().map(|&v| if v < 0 { (axis_size as i64 + v) as usize } else { v as usize }).filter(|&v| v <= axis_size).collect()
    } else {
        vec![]
    };

    let mut split_points = sections.clone();
    split_points.push(axis_size);
    split_points.insert(0, 0);
    // 去重并保持顺序
    split_points.dedup();

    let pre: usize = shape.iter().take(ax).product();
    let post: usize = shape.iter().skip(ax + 1).product();
    let data_vec: Vec<f64> = a.data.iter().copied().collect();

    let mut result = Vec::new();
    for i in 0..split_points.len() - 1 {
        let start = split_points[i].min(axis_size);
        let end = split_points[i + 1].min(axis_size);
        if start >= end {
            // 生成空的切片
            let mut empty_shape = shape.clone();
            empty_shape[ax] = 0;
            let section_data: Vec<f64> = Vec::new();
            let arr = Array::from_shape_vec(IxDyn(&empty_shape), section_data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            result.push(NdArray { data: arr });
            continue;
        }
        let section_size = end - start;
        let mut new_shape = shape.clone();
        new_shape[ax] = section_size;
        let total: usize = pre * section_size * post;
        let mut section_data = vec![0.0; total];

        for p in 0..pre {
            for s in 0..section_size {
                let src = p * axis_size * post + (start + s) * post;
                let dst = p * section_size * post + s * post;
                for k in 0..post {
                    section_data[dst + k] = data_vec[src + k];
                }
            }
        }
        let arr = Array::from_shape_vec(IxDyn(&new_shape), section_data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        result.push(NdArray { data: arr });
    }
    Ok(result)
}

// ========== 协方差矩阵 ==========
#[pyfunction]
#[pyo3(signature = (a, rowvar=true))]
fn cov(a: &NdArray, rowvar: bool) -> PyResult<NdArray> {
    let shape = a.data.shape().to_vec();
    let data_vec: Vec<f64> = a.data.iter().copied().collect();

    let (n_vars, n_obs) = if shape.len() == 1 {
        (1, shape[0])
    } else if rowvar {
        (shape[0], shape[1])
    } else {
        (shape[1], shape[0])
    };

    let rows: Vec<Vec<f64>> = if shape.len() == 1 {
        vec![data_vec.clone()]
    } else if rowvar {
        (0..shape[0]).map(|i| (0..shape[1]).map(|j| data_vec[i * shape[1] + j]).collect()).collect()
    } else {
        (0..shape[1]).map(|j| (0..shape[0]).map(|i| data_vec[i * shape[1] + j]).collect()).collect()
    };

    let means: Vec<f64> = rows.iter().map(|r| r.iter().sum::<f64>() / n_obs as f64).collect();
    let centered: Vec<Vec<f64>> = rows.iter().enumerate()
        .map(|(i, r)| r.iter().map(|&v| v - means[i]).collect())
        .collect();

    let mut cov_matrix = vec![vec![0.0; n_vars]; n_vars];
    let factor = 1.0 / (n_obs as f64 - 1.0);
    for i in 0..n_vars {
        for j in 0..n_vars {
            let mut s = 0.0;
            for k in 0..n_obs {
                s += centered[i][k] * centered[j][k];
            }
            cov_matrix[i][j] = s * factor;
        }
    }

    let mut flat = vec![0.0; n_vars * n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            flat[i * n_vars + j] = cov_matrix[i][j];
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[n_vars, n_vars]), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

// ========== 2D 直方图 ==========
#[pyfunction]
#[pyo3(signature = (x, y, bins=10))]
fn histogram2d_rs(x: &NdArray, y: &NdArray, bins: usize) -> PyResult<(NdArray, (Vec<f64>, Vec<f64>))> {
    let x_vals: Vec<f64> = x.data.iter().copied().collect();
    let y_vals: Vec<f64> = y.data.iter().copied().collect();
    let n = x_vals.len().min(y_vals.len());

    let x_min = *x_vals.iter().fold(&f64::INFINITY, |a, b| if a < b { a } else { b });
    let x_max = *x_vals.iter().fold(&f64::NEG_INFINITY, |a, b| if a > b { a } else { b });
    let y_min = *y_vals.iter().fold(&f64::INFINITY, |a, b| if a < b { a } else { b });
    let y_max = *y_vals.iter().fold(&f64::NEG_INFINITY, |a, b| if a > b { a } else { b });

    let mut hist = vec![vec![0.0; bins]; bins];
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    for i in 0..n {
        let xi = if x_range == 0.0 { 0 } else {
            let idx = ((x_vals[i] - x_min) / x_range * bins as f64) as usize;
            idx.min(bins - 1)
        };
        let yi = if y_range == 0.0 { 0 } else {
            let idx = ((y_vals[i] - y_min) / y_range * bins as f64) as usize;
            idx.min(bins - 1)
        };
        hist[xi][yi] += 1.0;
    }

    let flat: Vec<f64> = hist.iter().flatten().cloned().collect();
    let arr = Array::from_shape_vec(IxDyn(&[bins, bins]), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let x_edges: Vec<f64> = (0..=bins).map(|i| x_min + (x_max - x_min) * i as f64 / bins as f64).collect();
    let y_edges: Vec<f64> = (0..=bins).map(|i| y_min + (y_max - y_min) * i as f64 / bins as f64).collect();

    Ok((NdArray { data: arr }, (x_edges, y_edges)))
}

// ========== 相关系数 ==========
#[pyfunction]
fn corrcoef_rs(a: &NdArray) -> PyResult<NdArray> {
    let cov_arr = cov(a, true)?;
    let shape = cov_arr.data.shape().to_vec();
    let n = shape[0];
    let data_vec: Vec<f64> = cov_arr.data.iter().copied().collect();

    let diag: Vec<f64> = (0..n).map(|i| data_vec[i * n + i]).collect();
    let diag_sqrt: Vec<f64> = diag.iter().map(|&v| v.sqrt()).collect();

    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let denom = diag_sqrt[i] * diag_sqrt[j];
            result[i * n + j] = if denom == 0.0 { 0.0 } else { data_vec[i * n + j] / denom };
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[n, n]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

// ========== 多项式求值 ==========
#[pyfunction]
#[pyo3(signature = (coef, x))]
fn polyval_rs(coef: &NdArray, x: &NdArray) -> PyResult<NdArray> {
    let c: Vec<f64> = coef.data.iter().copied().collect();
    let x_shape = x.data.shape().to_vec();
    let x_vals: Vec<f64> = x.data.iter().copied().collect();
    let deg = c.len() - 1;

    let result: Vec<f64> = x_vals.iter().map(|&xv| {
        let mut val = 0.0;
        for (i, &ci) in c.iter().enumerate() {
            val += ci * xv.powi((deg - i) as i32);
        }
        val
    }).collect();

    let arr = Array::from_shape_vec(IxDyn(&x_shape), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

// ========== 多项式求导 ==========
#[pyfunction]
#[pyo3(signature = (coef, m=1))]
fn polyder_rs(coef: &NdArray, m: usize) -> PyResult<NdArray> {
    let mut c: Vec<f64> = coef.data.iter().copied().collect();
    for _ in 0..m {
        let deg = c.len() - 1;
        let mut new_c = Vec::new();
        for i in 0..c.len() - 1 {
            new_c.push(c[i] * (deg - i) as f64);
        }
        if new_c.is_empty() {
            new_c.push(0.0);
        }
        c = new_c;
    }
    let arr = Array::from_shape_vec(IxDyn(&[c.len()]), c)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

// ========== 多项式积分 ==========
#[pyfunction]
#[pyo3(signature = (coef, m=1, k=0.0))]
fn polyint_rs(coef: &NdArray, m: usize, k: f64) -> PyResult<NdArray> {
    let mut c: Vec<f64> = coef.data.iter().copied().collect();
    for _ in 0..m {
        let deg = c.len() - 1;
        let mut new_c = Vec::new();
        for i in 0..c.len() {
            new_c.push(c[i] / (deg - i + 1) as f64);
        }
        new_c.push(k);
        c = new_c;
    }
    let arr = Array::from_shape_vec(IxDyn(&[c.len()]), c)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
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
        return Ok(NdArray { data: arr });
    }
    if n == 1 {
        let root = -c[1] / c[0];
        let arr = Array::from_shape_vec(IxDyn(&[1]), vec![root])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray { data: arr });
    }
    if n == 2 {
        let disc = c[1] * c[1] - 4.0 * c[0] * c[2];
        if disc >= 0.0 {
            let sq = disc.sqrt();
            let r1 = (-c[1] + sq) / (2.0 * c[0]);
            let r2 = (-c[1] - sq) / (2.0 * c[0]);
            let arr = Array::from_shape_vec(IxDyn(&[2]), vec![r1, r2])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(NdArray { data: arr });
        } else {
            let sq = (-disc).sqrt();
            let r1 = -c[1] / (2.0 * c[0]);
            let r2 = sq / (2.0 * c[0]);
            let arr = Array::from_shape_vec(IxDyn(&[2]), vec![r1, r2])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(NdArray { data: arr });
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
    Ok(NdArray { data: arr })
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
    for i in 0..n {
        for j in 0..m {
            vander[i][j] = x_vals[i].powi((deg - j) as i32);
        }
    }

    let mut ata = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..n {
                s += vander[k][i] * vander[k][j];
            }
            ata[i][j] = s;
        }
    }

    let mut aty = vec![0.0; m];
    for i in 0..m {
        let mut s = 0.0;
        for k in 0..n {
            s += vander[k][i] * y_vals[k];
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
        coef[i] = if aug[i][i].abs() > 1e-15 { val / aug[i][i] } else { 0.0 };
    }

    let arr = Array::from_shape_vec(IxDyn(&[m]), coef)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

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
    let header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({}), }}", shape_str.join(", "));
    let header_len = header.len();
    let padded_len = ((header_len + 10 + 63) / 64) * 64 - 10;
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
                    inside.split(',')
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .collect()
                }
            } else { vec![] }
        } else { vec![] }
    } else { vec![] };

    let total: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
    let mut values = vec![0.0f64; total];

    for v in values.iter_mut() {
        let mut bytes = [0u8; 8];
        if file.read_exact(&mut bytes).is_err() {
            break;
        }
        *v = f64::from_le_bytes(bytes);
    }

    let arr_shape = if shape.is_empty() { IxDyn(&[]) } else { IxDyn(&shape) };
    let arr = Array::from_shape_vec(arr_shape, values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

fn format_value(fmt: &str, v: f64) -> String {
    if fmt.contains("%d") {
        format!("{}", v as i64)
    } else if fmt.contains("%i") {
        format!("{}", v as i64)
    } else if fmt.contains("%f") {
        format!("{}", v)
    } else if fmt.contains("%e") || fmt.contains("%E") {
        format!("{}", v)
    } else if fmt.contains("%g") || fmt.contains("%G") {
        format!("{}", v)
    } else if fmt.contains("%s") {
        format!("{}", v)
    } else {
        format!("{}", v)
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

    let mut rows: Vec<Vec<f64>> = Vec::new();
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
        let row: Vec<f64> = parts.iter()
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

// ========== 从 bytes 创建数组

#[pyfunction]
#[pyo3(signature = (bytes, count=-1))]
fn bytes_to_floats(bytes: &[u8], count: isize) -> PyResult<NdArray> {
    use std::mem;
    let n = bytes.len() / mem::size_of::<f64>();
    let actual = if count < 0 { n } else { count as usize };
    let mut result = Vec::with_capacity(actual);
    for i in 0..actual {
        let start = i * 8;
        if start + 8 > bytes.len() { break; }
        let chunk = &bytes[start..start + 8];
        let arr_u8: [u8; 8] = [chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]];
        result.push(f64::from_le_bytes(arr_u8));
    }
    let arr = Array::from_shape_vec(IxDyn(&[result.len()]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

// ========== 多维 tuple 索引（key 为 tuple） ==========
#[pyfunction]
#[pyo3(signature = (a, ranges))]
fn tuple_getitem(a: &NdArray, ranges: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    // ranges: list of (start, end, step) tuples
    let list = ranges.cast::<PyList>()?;
    let mut ranges_vec: Vec<(i64, i64, i64)> = Vec::new();
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
    for i in 0..ndim {
        let s = ranges_vec[i].0;
        let e = ranges_vec[i].1;
        let st = ranges_vec[i].2.max(1);
        if e <= s {
            out_shape.push(0);
        } else {
            out_shape.push(((e - s + st - 1) / st) as usize);
        }
    }

    // 计算 C-order 的总元素数
    let total: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
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
    collect(0, ndim, &ranges_vec, &shape, &mut idx, &flat_data, &mut values);

    let arr = if out_shape.is_empty() {
        Array::from_shape_vec(IxDyn(&[]), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        Array::from_shape_vec(IxDyn(&out_shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };
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
    let options: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);

    for (_i, (item, name_item)) in arr_list.iter().zip(name_list.iter()).enumerate() {
        let nd = item.extract::<NdArray>()?;
        let name_str = name_item.extract::<String>()?;
        let entry_name = if name_str.ends_with(".npy") {
            name_str
        } else {
            format!("{}.npy", name_str)
        };

        // 序列化 ndarray 为 .npy 格式
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);

        let shape = nd.data.shape().to_vec();
        let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        let header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({}), }}", shape_str.join(", "));
        let header_len = header.len();
        let padded_len = ((header_len + 10 + 63) / 64) * 64 - 10;
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

    zip.finish().map_err(|e| PyValueError::new_err(e.to_string()))?;
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
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut results: Vec<(String, NdArray)> = Vec::new();

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let name = entry.name().to_string();
        if !name.ends_with(".npy") {
            continue;
        }
        let key = name.trim_end_matches(".npy").to_string();

        let mut contents = Vec::new();
        entry.read_to_end(&mut contents)
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
                        inside.split(',')
                            .filter_map(|s| s.trim().parse::<usize>().ok())
                            .collect()
                    }
                } else { vec![] }
            } else { vec![] }
        } else { vec![] };

        let data_start = header_end;
        let total: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let mut values = vec![0.0f64; total];
        for (j, v) in values.iter_mut().enumerate() {
            let start = data_start + j * 8;
            if start + 8 > contents.len() { break; }
            let bytes = [
                contents[start], contents[start + 1], contents[start + 2], contents[start + 3],
                contents[start + 4], contents[start + 5], contents[start + 6], contents[start + 7],
            ];
            *v = f64::from_le_bytes(bytes);
        }

        let arr_shape = if shape.is_empty() { IxDyn(&[]) } else { IxDyn(&shape) };
        let arr = Array::from_shape_vec(arr_shape, values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nd = NdArray { data: arr };
        results.push((key, nd));
    }

    Ok(results)
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
    Ok(NdArray { data: arr })
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
    Ok(NdArray { data: arr })
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
    Ok(NdArray { data: arr })
}

// ========== argmax/argmin 沿指定轴 ==========
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn argmax_axis(a: &NdArray, axis: Option<i32>) -> PyResult<NdArray> {
    if axis.is_none() {
        let idx = a.argmax()?;
        let arr = Array::from_elem(IxDyn(&[]), idx as f64);
        return Ok(NdArray { data: arr });
    }
    let ax = axis.unwrap() as usize;
    let shape = a.data.shape().to_vec();
    if ax >= shape.len() {
        return Err(PyValueError::new_err("axis out of bounds"));
    }
    let outer: usize = shape.iter().take(ax).product();
    let inner: usize = shape.iter().skip(ax + 1).product();
    let dim_size = shape[ax];
    let mut result_shape = shape.clone();
    result_shape.remove(ax);
    if result_shape.is_empty() {
        result_shape = vec![1];
    }
    let total: usize = result_shape.iter().product();
    let mut values = vec![0.0f64; total];
    let data: Vec<f64> = a.data.iter().copied().collect();
    let mut out_idx = 0;
    for o in 0..outer {
        for inn in 0..inner {
            let mut best = 0usize;
            let mut best_val = f64::NEG_INFINITY;
            for i in 0..dim_size {
                let idx = o * dim_size * inner + i * inner + inn;
                let v = data[idx];
                if v > best_val {
                    best_val = v;
                    best = i;
                }
            }
            values[out_idx] = best as f64;
            out_idx += 1;
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&result_shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn argmin_axis(a: &NdArray, axis: Option<i32>) -> PyResult<NdArray> {
    if axis.is_none() {
        let idx = a.argmin()?;
        let arr = Array::from_elem(IxDyn(&[]), idx as f64);
        return Ok(NdArray { data: arr });
    }
    let ax = axis.unwrap() as usize;
    let shape = a.data.shape().to_vec();
    if ax >= shape.len() {
        return Err(PyValueError::new_err("axis out of bounds"));
    }
    let outer: usize = shape.iter().take(ax).product();
    let inner: usize = shape.iter().skip(ax + 1).product();
    let dim_size = shape[ax];
    let mut result_shape = shape.clone();
    result_shape.remove(ax);
    if result_shape.is_empty() {
        result_shape = vec![1];
    }
    let total: usize = result_shape.iter().product();
    let mut values = vec![0.0f64; total];
    let data: Vec<f64> = a.data.iter().copied().collect();
    let mut out_idx = 0;
    for o in 0..outer {
        for inn in 0..inner {
            let mut best = 0usize;
            let mut best_val = f64::INFINITY;
            for i in 0..dim_size {
                let idx = o * dim_size * inner + i * inner + inn;
                let v = data[idx];
                if v < best_val {
                    best_val = v;
                    best = i;
                }
            }
            values[out_idx] = best as f64;
            out_idx += 1;
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&result_shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

// ========== 复数格式化 ==========

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
                s.push_str(" ");
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
    s.push_str("]");
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

// ===== Int64 Formatting =====

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
                s.push_str(" ");
            }
            let val_str = format_int_scalar(*val);
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
        let row_str = format_int_array_inner(&sub, pad_width);
        s.push_str(&row_str);
    }
    s.push_str("]");
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
    format!("rsnumpy.ndarray({}) dtype=int64", inner)
}

#[pyfunction]
fn _format_int_str(arr: &NdArray) -> String {
    format_int_array(&arr.data)
}

// ===== Float64 Formatting (always show decimal point for integer values) =====

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
                s.push_str(" ");
            }
            let val_str = format_float_scalar(*val);
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
        let row_str = format_float_array_inner(&sub, pad_width);
        s.push_str(&row_str);
    }
    s.push_str("]");
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
    format!("rsnumpy.ndarray({}) dtype=float64", inner)
}

#[pyfunction]
fn _format_float_str(arr: &NdArray) -> String {
    format_float_array(&arr.data)
}

// int64 值显示：使用 format_scalar（整数值不显示小数点，如 0 1 2）
#[pyfunction]
fn _format_int_val_repr(arr: &NdArray) -> String {
    let inner = format_array_repr(&arr.data, "");
    format!("rsnumpy.ndarray({}) dtype=int64", inner)
}

#[pyfunction]
fn _format_int_val_str(arr: &NdArray) -> String {
    format_array_repr(&arr.data, "")
}

// ===== Module Initialization =====

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NdArray>()?;
    m.add_class::<NdArrayIter>()?;

    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;

    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(log10, m)?)?;
    m.add_function(wrap_pyfunction!(log2, m)?)?;
    m.add_function(wrap_pyfunction!(log1p, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    m.add_function(wrap_pyfunction!(cosh, m)?)?;
    m.add_function(wrap_pyfunction!(sinh, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(acosh, m)?)?;
    m.add_function(wrap_pyfunction!(asinh, m)?)?;
    m.add_function(wrap_pyfunction!(atanh, m)?)?;
    m.add_function(wrap_pyfunction!(asin, m)?)?;
    m.add_function(wrap_pyfunction!(acos, m)?)?;
    m.add_function(wrap_pyfunction!(atan, m)?)?;
    m.add_function(wrap_pyfunction!(expm1, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(round, m)?)?;
    m.add_function(wrap_pyfunction!(trunc, m)?)?;
    m.add_function(wrap_pyfunction!(fix, m)?)?;
    m.add_function(wrap_pyfunction!(square, m)?)?;
    m.add_function(wrap_pyfunction!(cbrt, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(reciprocal, m)?)?;
    m.add_function(wrap_pyfunction!(arctan2, m)?)?;
    m.add_function(wrap_pyfunction!(deg2rad, m)?)?;
    m.add_function(wrap_pyfunction!(rad2deg, m)?)?;
    m.add_function(wrap_pyfunction!(hypot, m)?)?;
    m.add_function(wrap_pyfunction!(sinc, m)?)?;
    m.add_function(wrap_pyfunction!(heaviside, m)?)?;

    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;

    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(swapaxes, m)?)?;

    m.add_function(wrap_pyfunction!(where_, m)?)?;
    m.add_function(wrap_pyfunction!(clip, m)?)?;
    m.add_function(wrap_pyfunction!(unique, m)?)?;
    m.add_function(wrap_pyfunction!(sort, m)?)?;
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    m.add_function(wrap_pyfunction!(partition, m)?)?;
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(average, m)?)?;
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(meshgrid, m)?)?;
    m.add_function(wrap_pyfunction!(histogram, m)?)?;
    m.add_function(wrap_pyfunction!(gradient, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(trapz, m)?)?;
    m.add_function(wrap_pyfunction!(cross, m)?)?;
    m.add_function(wrap_pyfunction!(all, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    m.add_function(wrap_pyfunction!(nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(nonzero_arrs, m)?)?;
    m.add_function(wrap_pyfunction!(select_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ix_rs, m)?)?;
    m.add_function(wrap_pyfunction!(_arange_arrays, m)?)?;

    m.add_function(wrap_pyfunction!(isnan, m)?)?;
    m.add_function(wrap_pyfunction!(isinf, m)?)?;
    m.add_function(wrap_pyfunction!(isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(invert, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(left_shift, m)?)?;
    m.add_function(wrap_pyfunction!(right_shift, m)?)?;
    m.add_function(wrap_pyfunction!(maximum, m)?)?;
    m.add_function(wrap_pyfunction!(minimum, m)?)?;
    m.add_function(wrap_pyfunction!(allclose, m)?)?;
    m.add_function(wrap_pyfunction!(array_equal, m)?)?;
    m.add_function(wrap_pyfunction!(argwhere, m)?)?;
    m.add_function(wrap_pyfunction!(count_nonzero, m)?)?;

    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    m.add_function(wrap_pyfunction!(tile, m)?)?;
    m.add_function(wrap_pyfunction!(squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(flatten, m)?)?;
    m.add_function(wrap_pyfunction!(reshape, m)?)?;

    m.add_function(wrap_pyfunction!(logspace, m)?)?;
    m.add_function(wrap_pyfunction!(geomspace, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(subtract, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(divide, m)?)?;
    m.add_function(wrap_pyfunction!(power, m)?)?;
    m.add_function(wrap_pyfunction!(mod_, m)?)?;
    m.add_function(wrap_pyfunction!(greater, m)?)?;
    m.add_function(wrap_pyfunction!(less, m)?)?;
    m.add_function(wrap_pyfunction!(equal, m)?)?;
    m.add_function(wrap_pyfunction!(not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(greater_equal, m)?)?;
    m.add_function(wrap_pyfunction!(less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(logical_and, m)?)?;
    m.add_function(wrap_pyfunction!(logical_or, m)?)?;
    m.add_function(wrap_pyfunction!(logical_xor, m)?)?;
    m.add_function(wrap_pyfunction!(isclose, m)?)?;
    m.add_function(wrap_pyfunction!(expand_dims, m)?)?;
    m.add_function(wrap_pyfunction!(column_stack, m)?)?;
    m.add_function(wrap_pyfunction!(roll, m)?)?;
    m.add_function(wrap_pyfunction!(rot90, m)?)?;
    m.add_function(wrap_pyfunction!(flatnonzero, m)?)?;
    m.add_function(wrap_pyfunction!(ptp, m)?)?;
    m.add_function(wrap_pyfunction!(digitize, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_to, m)?)?;
    m.add_function(wrap_pyfunction!(searchsorted, m)?)?;
    m.add_function(wrap_pyfunction!(flip, m)?)?;
    m.add_function(wrap_pyfunction!(split_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cov, m)?)?;
    m.add_function(wrap_pyfunction!(histogram2d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(corrcoef_rs, m)?)?;
    m.add_function(wrap_pyfunction!(polyval_rs, m)?)?;
    m.add_function(wrap_pyfunction!(polyder_rs, m)?)?;
    m.add_function(wrap_pyfunction!(polyint_rs, m)?)?;
    m.add_function(wrap_pyfunction!(polyroots_rs, m)?)?;
    m.add_function(wrap_pyfunction!(polyfit_rs, m)?)?;
    m.add_function(wrap_pyfunction!(save_npy, m)?)?;
    m.add_function(wrap_pyfunction!(load_npy, m)?)?;
    m.add_function(wrap_pyfunction!(save_text, m)?)?;
    m.add_function(wrap_pyfunction!(load_text, m)?)?;
    m.add_function(wrap_pyfunction!(bytes_to_floats, m)?)?;

    m.add_function(wrap_pyfunction!(tuple_getitem, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::getitem_multi, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::getitem_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setitem_multi, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::iscomplex_cpx, m)?)?;
    m.add_function(wrap_pyfunction!(savez_npz, m)?)?;
    m.add_function(wrap_pyfunction!(load_npz, m)?)?;
    m.add_function(wrap_pyfunction!(polyadd, m)?)?;
    m.add_function(wrap_pyfunction!(polysub, m)?)?;
    m.add_function(wrap_pyfunction!(polymul, m)?)?;
    m.add_function(wrap_pyfunction!(argmax_axis, m)?)?;
    m.add_function(wrap_pyfunction!(argmin_axis, m)?)?;

    m.add_function(wrap_pyfunction!(_format_complex_repr, m)?)?;
    m.add_function(wrap_pyfunction!(_format_complex_str, m)?)?;

    m.add_function(wrap_pyfunction!(_format_int_repr, m)?)?;
    m.add_function(wrap_pyfunction!(_format_int_str, m)?)?;

    m.add_function(wrap_pyfunction!(_format_float_repr, m)?)?;
    m.add_function(wrap_pyfunction!(_format_float_str, m)?)?;

    m.add_function(wrap_pyfunction!(_format_int_val_repr, m)?)?;
    m.add_function(wrap_pyfunction!(_format_int_val_str, m)?)?;

    m.add_function(wrap_pyfunction!(fft::py_fft, m)?)?;
    m.add_function(wrap_pyfunction!(fft::py_ifft, m)?)?;
    m.add_function(wrap_pyfunction!(fft::py_rfft, m)?)?;
    m.add_function(wrap_pyfunction!(fft::py_irfft, m)?)?;
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
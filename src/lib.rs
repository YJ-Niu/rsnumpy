use ndarray::{Array, ArrayViewD, Axis, IxDyn, Slice};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList, PySlice, PyTuple};

mod linalg;
mod random;

fn parse_py_list_to_flat(data: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
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

fn format_array_repr(arr: &Array<f64, IxDyn>, _prefix: &str) -> String {
    if arr.ndim() == 0 {
        return format!("{}", arr.iter().next().copied().unwrap_or(0.0_f64));
    }
    if arr.ndim() == 1 {
        let mut s = String::from("[");
        for (i, val) in arr.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            if *val == val.floor() && val.is_finite() {
                s.push_str(&format!("{}.", val));
            } else {
                s.push_str(&format!("{}", val));
            }
        }
        s.push(']');
        return s;
    }
    let mut s = String::from("[");
    let n = arr.shape()[0];
    for i in 0..n {
        if i > 0 {
            s.push_str(",\n");
        }
        let sub = arr.index_axis(Axis(0), i).to_owned().into_dyn();
        let row_str = format_array_repr(&sub, "");
        s.push_str(" ");
        s.push_str(&row_str);
    }
    s.push_str("]");
    s
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
        Ok(format!("rsnum.ndarray({}) dtype=float64", arr_str))
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
        let s = shape_to_vec(shape)?;
        let arr = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(&s))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    fn flatten(&self) -> NdArray {
        NdArray {
            data: self.data.clone().into_shape_with_order(IxDyn(&[self.data.len()])).unwrap(),
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

    fn astype(&self, _dtype: &str) -> PyResult<NdArray> {
        Ok(NdArray {
            data: self.data.clone(),
        })
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
                        let mean_val = mean_arr[0];
                        let std_data: Vec<f64> = self
                            .data
                            .axis_iter(Axis(ax))
                            .flat_map(|row| row.mapv(|x| (x - mean_val).powi(2)))
                            .collect();
                        let shape: Vec<usize> = self
                            .data
                            .shape()
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != ax)
                            .map(|(_, &s)| s)
                            .collect();
                        if shape.is_empty() {
                            let var = std_data.iter().sum::<f64>() / std_data.len() as f64;
                            Ok(NdArray {
                                data: Array::from_elem(IxDyn(&[]), var.sqrt()),
                            })
                        } else {
                            let arr = Array::from_shape_vec(IxDyn(&shape), std_data)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                            let n = self.data.shape()[ax] as f64;
                            Ok(NdArray {
                                data: (arr / n).mapv(|x| x.sqrt()),
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
                        let mean_val = mean_arr[0];
                        let var_data: Vec<f64> = self
                            .data
                            .axis_iter(Axis(ax))
                            .flat_map(|row| row.mapv(|x| (x - mean_val).powi(2)))
                            .collect();
                        let shape: Vec<usize> = self
                            .data
                            .shape()
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != ax)
                            .map(|(_, &s)| s)
                            .collect();
                        if shape.is_empty() {
                            let v = var_data.iter().sum::<f64>() / var_data.len() as f64;
                            Ok(NdArray {
                                data: Array::from_elem(IxDyn(&[]), v),
                            })
                        } else {
                            let arr = Array::from_shape_vec(IxDyn(&shape), var_data)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                            let n = self.data.shape()[ax] as f64;
                            Ok(NdArray {
                                data: arr / n,
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
                    data: Array::from_elem(IxDyn(&[]), val),
                })
            }
            Some(ax) => {
                let ndim = self.data.ndim();
                let ax = if ax < 0 { (ndim as isize + ax) as usize } else { ax as usize };
                let mut result_vec = Vec::new();
                for row in self.data.axis_iter(Axis(ax)) {
                    let min_val = row.iter().cloned().fold(f64::INFINITY, f64::min);
                    result_vec.push(min_val);
                }
                let shape: Vec<usize> = self
                    .data
                    .shape()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ax)
                    .map(|(_, &s)| s)
                    .collect();
                if shape.is_empty() {
                    let val = result_vec.into_iter().fold(f64::INFINITY, f64::min);
                    Ok(NdArray {
                        data: Array::from_elem(IxDyn(&[]), val),
                    })
                } else {
                    let arr = Array::from_shape_vec(IxDyn(&shape), result_vec)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(NdArray { data: arr })
                }
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
                let mut result_vec = Vec::new();
                for row in self.data.axis_iter(Axis(ax)) {
                    let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    result_vec.push(max_val);
                }
                let shape: Vec<usize> = self
                    .data
                    .shape()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ax)
                    .map(|(_, &s)| s)
                    .collect();
                if shape.is_empty() {
                    let val = result_vec.into_iter().fold(f64::NEG_INFINITY, f64::max);
                    Ok(NdArray {
                        data: Array::from_elem(IxDyn(&[]), val),
                    })
                } else {
                    let arr = Array::from_shape_vec(IxDyn(&shape), result_vec)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(NdArray { data: arr })
                }
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
        for outer in 0..pre_size {
            for inner in 0..post_size {
                let base = outer * block_size + inner;
                let mut slice: Vec<f64> = (0..axis_size)
                    .map(|k| data_vec[base + k * post_size])
                    .collect();
                slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for (k, val) in slice.into_iter().enumerate() {
                    data_vec[base + k * post_size] = val;
                }
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
        let mut result = data_vec.clone();
        for outer in 0..pre_size {
            for inner in 0..post_size {
                let base = outer * block_size + inner;
                let mut indexed: Vec<(usize, f64)> = (0..axis_size)
                    .map(|k| (k, data_vec[base + k * post_size]))
                    .collect();
                indexed.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for (pos, (idx, _)) in indexed.into_iter().enumerate() {
                    result[base + pos * post_size] = idx as f64;
                }
            }
        }
        let arr = Array::from_shape_vec(IxDyn(&shape), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }
}

fn binary_op(
    a: &NdArray,
    b: &Bound<'_, PyAny>,
    op: fn(f64, f64) -> f64,
) -> PyResult<NdArray> {
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

fn binary_op_lr(
    a: &NdArray,
    b: &Bound<'_, PyAny>,
    op: fn(f64, f64) -> f64,
) -> PyResult<NdArray> {
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

fn broadcast_binary_op(
    a: &Array<f64, IxDyn>,
    b: &Array<f64, IxDyn>,
    op: fn(f64, f64) -> f64,
) -> PyResult<Array<f64, IxDyn>> {
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();

    if a_shape == b_shape {
        let result: Vec<f64> = a.iter().cloned().zip(b.iter().cloned()).map(|(x, y)| op(x, y)).collect();
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
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a_broadcast = a_broadcast
        .broadcast(IxDyn(&out_shape))
        .ok_or_else(|| PyValueError::new_err("Broadcasting failed"))?
        .to_owned();
    let b_broadcast = b
        .clone()
        .into_shape_with_order(IxDyn(&b_padded))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let b_broadcast = b_broadcast
        .broadcast(IxDyn(&out_shape))
        .ok_or_else(|| PyValueError::new_err("Broadcasting failed"))?
        .to_owned();

    let result: Vec<f64> = a_broadcast
        .iter()
        .zip(b_broadcast.iter())
        .map(|(x, y)| op(*x, *y))
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
    let mut val = start;
    if step > 0.0 {
        while val < stop {
            values.push(val);
            val += step;
        }
    } else {
        while val > stop {
            values.push(val);
            val += step;
        }
    }
    let arr = Array::from_shape_vec(IxDyn(&[values.len()]), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

#[pyfunction]
#[pyo3(signature = (start, stop, num=50))]
fn linspace(start: f64, stop: f64, num: usize) -> PyResult<NdArray> {
    if num == 0 {
        return Err(PyValueError::new_err("Number of samples must be positive"));
    }
    if num == 1 {
        return Ok(NdArray {
            data: Array::from_shape_vec(IxDyn(&[1]), vec![start])
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        });
    }
    let step = (stop - start) / (num - 1) as f64;
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
    Ok(NdArray {
        data: Array::zeros(IxDyn(&s)),
    })
}

// ===== Math Functions =====

fn unary_math_op(x: &NdArray, op: fn(f64) -> f64) -> NdArray {
    NdArray {
        data: x.data.mapv(op),
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
            let result: Vec<f64> = condition.data.iter().zip(xv.data.iter().zip(yv.data.iter()))
                .map(|(&c, (&xv, &yv))| if c != 0.0 { xv } else { yv })
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
fn median(a: &NdArray) -> PyResult<NdArray> {
    let mut values: Vec<f64> = a.data.iter().copied().collect();
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

#[pyfunction]
fn percentile(a: &NdArray, q: f64) -> PyResult<NdArray> {
    if q < 0.0 || q > 100.0 {
        return Err(PyValueError::new_err("Percentile must be between 0 and 100"));
    }
    let mut values: Vec<f64> = a.data.iter().copied().collect();
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
fn round(x: &NdArray) -> NdArray {
    unary_math_op(x, |v| v.round())
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
#[pyo3(signature = (a, b, rtol=1e-05, atol=1e-08))]
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

// ===== Module Initialization =====

#[pymodule]
fn rsnum(m: &Bound<'_, PyModule>) -> PyResult<()> {
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

    m.add_function(wrap_pyfunction!(where_, m)?)?;
    m.add_function(wrap_pyfunction!(clip, m)?)?;
    m.add_function(wrap_pyfunction!(unique, m)?)?;
    m.add_function(wrap_pyfunction!(sort, m)?)?;
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    m.add_function(wrap_pyfunction!(median, m)?)?;
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

    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(round, m)?)?;
    m.add_function(wrap_pyfunction!(isnan, m)?)?;
    m.add_function(wrap_pyfunction!(isinf, m)?)?;
    m.add_function(wrap_pyfunction!(isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(maximum, m)?)?;
    m.add_function(wrap_pyfunction!(minimum, m)?)?;
    m.add_function(wrap_pyfunction!(allclose, m)?)?;
    m.add_function(wrap_pyfunction!(argwhere, m)?)?;
    m.add_function(wrap_pyfunction!(count_nonzero, m)?)?;

    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    m.add_function(wrap_pyfunction!(tile, m)?)?;
    m.add_function(wrap_pyfunction!(squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(flatten, m)?)?;
    m.add_function(wrap_pyfunction!(reshape, m)?)?;

    let random_module = PyModule::new(m.py(), "random")?;
    random::init_module(&random_module)?;
    m.add_submodule(&random_module)?;

    let linalg_module = PyModule::new(m.py(), "linalg")?;
    linalg::init_module(&linalg_module)?;
    m.add_submodule(&linalg_module)?;

    Ok(())
}
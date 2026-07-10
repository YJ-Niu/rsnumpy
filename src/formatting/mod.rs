use crate::*;

pub(crate) const LINE_WIDTH: usize = 75;

/// numpy `_extendLine`：将 word 追加到 line，若超出 line_width 则先把 line 换行写入 s。
pub(crate) fn extend_line(
    s: &mut String,
    line: &mut String,
    word: &str,
    line_width: usize,
    next_line_prefix: &str,
) {
    let mut needs_wrap = line.len() + word.len() > line_width;
    // 行内尚无实际内容（仅悬挂缩进）时不换行，避免产生空行。
    if line.len() <= next_line_prefix.len() {
        needs_wrap = false;
    }
    if needs_wrap {
        s.push_str(line.trim_end());
        s.push('\n');
        line.clear();
        line.push_str(next_line_prefix);
    }
    line.push_str(word);
}

/// numpy `_formatArray` 递归器：在 LINE_WIDTH 处折行，元素间用 separator，
/// 嵌套维度间插入换行 + 悬挂缩进（外层维度间空行更多）。`fmt` 须产出等宽字符串。
pub(crate) fn wrap_recurse(
    arr: &Array<f64, IxDyn>,
    hanging_indent: &str,
    curr_width: usize,
    fmt: &dyn Fn(f64) -> String,
    separator: &str,
) -> String {
    let ndim = arr.ndim();
    if ndim == 0 {
        return fmt(arr.iter().next().copied().unwrap_or(0.0_f64));
    }

    let next_hanging_indent = format!("{hanging_indent} ");
    let next_width = curr_width.saturating_sub(1);

    if ndim == 1 {
        let elem_width = curr_width.saturating_sub(separator.trim_end().len().max(1));
        let mut s = String::new();
        let mut line = hanging_indent.to_string();
        let n = arr.len();
        for (i, val) in arr.iter().enumerate() {
            let word = fmt(*val);
            extend_line(&mut s, &mut line, &word, elem_width, hanging_indent);
            if i + 1 < n {
                line.push_str(separator);
            }
        }
        s.push_str(&line);
        return format!("[{}]", &s[hanging_indent.len()..]);
    }

    let mut s = String::new();
    let n = arr.shape()[0];
    let newlines = ndim - 1;
    for i in 0..n {
        let sub = arr.index_axis(Axis(0), i).to_owned().into_dyn();
        let nested = wrap_recurse(&sub, &next_hanging_indent, next_width, fmt, separator);
        s.push_str(hanging_indent);
        s.push_str(&nested);
        if i + 1 < n {
            s.push_str(separator.trim_end());
            for _ in 0..newlines {
                s.push('\n');
            }
        }
    }
    format!("[{}]", &s[hanging_indent.len()..])
}

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

fn format_complex_array(arr: &Array<f64, IxDyn>) -> String {
    if arr.ndim() == 0 {
        return format_complex_scalar(arr.iter().next().copied().unwrap_or(0.0_f64));
    }
    let pad_width = arr
        .iter()
        .map(|v| format_complex_scalar(*v).len())
        .max()
        .unwrap_or(1);
    let render = move |v: f64| {
        let s = format_complex_scalar(v);
        format!("{s:>pad_width$}")
    };
    wrap_recurse(arr, " ", LINE_WIDTH, &render, " ")
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

fn format_int_array(arr: &Array<f64, IxDyn>) -> String {
    if arr.ndim() == 0 {
        return format_int_scalar(arr.iter().next().copied().unwrap_or(0.0_f64));
    }
    let pad_width = arr
        .iter()
        .map(|v| format_int_scalar(*v).len())
        .max()
        .unwrap_or(1);
    let render = move |v: f64| {
        let s = format_int_scalar(v);
        format!("{s:>pad_width$}")
    };
    wrap_recurse(arr, " ", LINE_WIDTH, &render, " ")
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

/// 位置记数法浮点字符串的整数部分长度（含符号）与小数部分长度（不含小数点）。
fn split_pos_float(s: &str) -> (usize, usize) {
    match s.find('.') {
        Some(pos) => (pos, s.len() - pos - 1),
        None => (s.len(), 0),
    }
}

/// numpy 浮点数组的显示格式：位置记数法或科学计数法（整组统一）。
enum FloatFormat {
    /// 整数部分右对齐到 pad_left，小数部分右补空格到 pad_right。
    Positional { pad_left: usize, pad_right: usize },
    /// 尾数整数部分右对齐到 pad_left，小数部分右补零到 precision，
    /// 指数带符号并零填充到 exp_size 位。
    Scientific {
        pad_left: usize,
        precision: usize,
        exp_size: usize,
    },
}

/// 指数的十进制位数（numpy 至少 2 位）。
fn exp_digit_count(exp: i32) -> usize {
    let mut a = exp.unsigned_abs();
    let mut d = 0usize;
    while a > 0 {
        d += 1;
        a /= 10;
    }
    d.max(2)
}

/// 将有限浮点数拆成科学计数法的 (整数部分含符号, 小数部分, 指数)。
fn parse_sci(v: f64) -> (String, String, i32) {
    let s = format!("{:e}", v);
    let (mant, exp_str) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let exp: i32 = exp_str.parse().unwrap_or(0);
    let (int_part, frac_part) = match mant.split_once('.') {
        Some((i, f)) => (i.to_string(), f.to_string()),
        None => (mant.to_string(), String::new()),
    };
    (int_part, frac_part, exp)
}

/// 复刻 numpy `FloatingFormat.fillFormat`：根据全体有限值决定位置/科学计数法
/// 以及对齐参数，并在存在非有限值（nan/inf）时按需增大 pad_left。
fn compute_float_format(arr: &Array<f64, IxDyn>) -> FloatFormat {
    let mut finite: Vec<f64> = Vec::new();
    let mut has_nonfinite = false;
    let mut has_neg_inf = false;
    for &v in arr.iter() {
        if v.is_finite() {
            finite.push(v);
        } else {
            has_nonfinite = true;
            if v.is_infinite() && v < 0.0 {
                has_neg_inf = true;
            }
        }
    }

    // 依据非零有限值决定是否使用科学计数法。
    let mut exp_format = false;
    let mut max_val = f64::NEG_INFINITY;
    let mut min_val = f64::INFINITY;
    let mut any_nonzero = false;
    for &v in &finite {
        if v != 0.0 {
            any_nonzero = true;
            let a = v.abs();
            if a > max_val {
                max_val = a;
            }
            if a < min_val {
                min_val = a;
            }
        }
    }
    if any_nonzero && (max_val >= 1e8 || min_val < 0.0001 || max_val / min_val > 1000.0) {
        exp_format = true;
    }

    // 非有限值使 pad_left 至少能容纳 nan/inf 字符串。
    let adjust_pad_left = |pad_left: usize, pad_right: usize| -> usize {
        if !has_nonfinite {
            return pad_left;
        }
        let inflen = 3 + usize::from(has_neg_inf);
        let nanlen = 3usize;
        let offset = pad_right + 1;
        pad_left
            .max(nanlen.saturating_sub(offset))
            .max(inflen.saturating_sub(offset))
    };

    if exp_format {
        let mut pad_left = 0usize;
        let mut precision = 0usize;
        let mut exp_size = 2usize;
        for &v in &finite {
            let (int_part, frac_part, exp) = parse_sci(v);
            pad_left = pad_left.max(int_part.len());
            precision = precision.max(frac_part.len());
            exp_size = exp_size.max(exp_digit_count(exp));
        }
        let pad_right = exp_size + 2 + precision;
        let pad_left = adjust_pad_left(pad_left, pad_right);
        FloatFormat::Scientific {
            pad_left,
            precision,
            exp_size,
        }
    } else {
        let mut pad_left = 0usize;
        let mut pad_right = 0usize;
        for &v in &finite {
            let s = format_float_scalar(v);
            let (ip, fp) = split_pos_float(&s);
            pad_left = pad_left.max(ip);
            pad_right = pad_right.max(fp);
        }
        let pad_left = adjust_pad_left(pad_left, pad_right);
        FloatFormat::Positional {
            pad_left,
            pad_right,
        }
    }
}

/// 将 nan/inf/-inf 右对齐到 `pad_left + 1 + pad_right` 的总宽度。
fn render_nonfinite(v: f64, pad_left: usize, pad_right: usize) -> String {
    let ret = if v.is_nan() {
        "nan"
    } else if v > 0.0 {
        "inf"
    } else {
        "-inf"
    };
    let total = pad_left + pad_right + 1;
    let mut out = String::with_capacity(total);
    for _ in 0..total.saturating_sub(ret.len()) {
        out.push(' ');
    }
    out.push_str(ret);
    out
}

/// 位置记数法：整数部分右对齐到 pad_left，小数部分右补空格到 pad_right。
fn render_positional(v: f64, pad_left: usize, pad_right: usize) -> String {
    if !v.is_finite() {
        return render_nonfinite(v, pad_left, pad_right);
    }
    let s = format_float_scalar(v);
    let (int_part, frac_part) = match s.find('.') {
        Some(pos) => (&s[..pos], &s[pos + 1..]),
        None => (s.as_str(), ""),
    };
    let mut out = String::with_capacity(pad_left + 1 + pad_right);
    for _ in 0..pad_left.saturating_sub(int_part.len()) {
        out.push(' ');
    }
    out.push_str(int_part);
    out.push('.');
    out.push_str(frac_part);
    for _ in 0..pad_right.saturating_sub(frac_part.len()) {
        out.push(' ');
    }
    out
}

/// 科学计数法：尾数整数部分右对齐到 pad_left，小数部分右补零到 precision，
/// 指数带符号并零填充到 exp_size 位。
fn render_scientific(v: f64, pad_left: usize, precision: usize, exp_size: usize) -> String {
    if !v.is_finite() {
        let pad_right = exp_size + 2 + precision;
        return render_nonfinite(v, pad_left, pad_right);
    }
    let (int_part, frac_part, exp) = parse_sci(v);
    let mut out = String::new();
    for _ in 0..pad_left.saturating_sub(int_part.len()) {
        out.push(' ');
    }
    out.push_str(&int_part);
    out.push('.');
    out.push_str(&frac_part);
    for _ in 0..precision.saturating_sub(frac_part.len()) {
        out.push('0');
    }
    out.push('e');
    out.push(if exp < 0 { '-' } else { '+' });
    let exp_str = format!("{}", exp.unsigned_abs());
    for _ in 0..exp_size.saturating_sub(exp_str.len()) {
        out.push('0');
    }
    out.push_str(&exp_str);
    out
}

fn format_float_array(arr: &Array<f64, IxDyn>) -> String {
    if arr.ndim() == 0 {
        return format_float_scalar(arr.iter().next().copied().unwrap_or(0.0_f64));
    }
    let render: Box<dyn Fn(f64) -> String> = match compute_float_format(arr) {
        FloatFormat::Positional {
            pad_left,
            pad_right,
        } => Box::new(move |v| render_positional(v, pad_left, pad_right)),
        FloatFormat::Scientific {
            pad_left,
            precision,
            exp_size,
        } => Box::new(move |v| render_scientific(v, pad_left, precision, exp_size)),
    };
    wrap_recurse(arr, " ", LINE_WIDTH, render.as_ref(), " ")
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

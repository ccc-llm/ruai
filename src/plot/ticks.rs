//! Ticks as matplotlib places and labels them (`MaxNLocator`, `ScalarFormatter`).

/// Python's `divmod` of floats.
fn divmod(x: f64, y: f64) -> (f64, f64) {
    let mut m = x % y;
    if m != 0.0 && ((y < 0.0) != (m < 0.0)) {
        m += y;
    }
    let div = (x - m) / y;
    let mut d = div.floor();
    if div - d > 0.5 {
        d += 1.0;
    }
    (d, m)
}

/// `matplotlib.transforms.nonsingular`: widen an empty or tiny interval.
pub(crate) fn nonsingular(mut vmin: f64, mut vmax: f64, expander: f64, tiny: f64) -> (f64, f64) {
    if !vmin.is_finite() || !vmax.is_finite() {
        return (-expander, expander);
    }
    if vmax < vmin {
        (vmin, vmax) = (vmax, vmin);
    }
    let maxabs = vmin.abs().max(vmax.abs());
    if maxabs < (1e6 / tiny) * f64::MIN_POSITIVE {
        return (-expander, expander);
    }
    if vmax - vmin <= maxabs * tiny {
        if vmax == 0.0 && vmin == 0.0 {
            return (-expander, expander);
        }
        vmin -= expander * vmin.abs();
        vmax += expander * vmax.abs();
    }
    (vmin, vmax)
}

/// `matplotlib.ticker.scale_range`
fn scale_range(vmin: f64, vmax: f64, n: f64) -> (f64, f64) {
    let dv = (vmax - vmin).abs();
    let meanv = (vmax + vmin) / 2.0;
    let offset = if meanv.abs() / dv < 100.0 { 0.0 } else { 10f64.powf(divmod(meanv.abs().log10(), 1.0).0).copysign(meanv) };
    let scale = 10f64.powf(divmod((dv / n).log10(), 1.0).0);
    (scale, offset)
}

/// Tick positions of matplotlib's `MaxNLocator(nbins, steps=[1, 2, 2.5, 5, 10])`
/// for the interval `vmin..vmax` (the ticks may extend past it).
pub fn max_n_locator(vmin: f64, vmax: f64, nbins: usize) -> Vec<f64> {
    let (vmin, vmax) = nonsingular(vmin, vmax, 1e-13, 1e-14);
    let (scale, offset) = scale_range(vmin, vmax, nbins as f64);
    let (v0, v1) = (vmin - offset, vmax - offset);
    let base = [1.0, 2.0, 2.5, 5.0, 10.0];
    let mut steps: Vec<f64> = base[..4].iter().map(|s| 0.1 * s).collect();
    steps.extend(base);
    steps.push(10.0 * base[1]);
    let steps: Vec<f64> = steps.iter().map(|s| s * scale).collect();
    let raw_step = (v1 - v0) / nbins as f64;
    let istep = steps.iter().position(|&s| s >= raw_step).unwrap_or(steps.len() - 1);
    // `_Edge_integer`: floor / ceil that forgive rounding errors
    let tol = |step: f64| {
        if offset.abs() > 0.0 {
            let digits = (offset.abs() / step).log10();
            1e-10f64.max(10f64.powf(digits - 12.0)).min(0.4999)
        } else {
            1e-10
        }
    };
    let mut ticks = vec![];
    for &step in steps[..=istep].iter().rev() {
        let best_vmin = divmod(v0, step).0 * step;
        let t = tol(step);
        let le = |x: f64| {
            let (d, m) = divmod(x, step);
            if (m / step - 1.0).abs() < t {
                d + 1.0
            } else {
                d
            }
        };
        let ge = |x: f64| {
            let (d, m) = divmod(x, step);
            if (m / step).abs() < t {
                d
            } else {
                d + 1.0
            }
        };
        let (low, high) = (le(v0 - best_vmin), ge(v1 - best_vmin));
        ticks = (0..=(high - low) as i64).map(|i| (low + i as f64) * step + best_vmin).collect();
        let n = ticks.iter().filter(|&&t| t <= v1 && t >= v0).count();
        if n >= 2 {
            break;
        }
    }
    ticks.iter().map(|t| t + offset).collect()
}

/// C's `%g` (6 significant digits), as Python's `format(x, "g")`.
pub fn format_g(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    if !x.is_finite() {
        return if x.is_nan() { "nan".into() } else if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let e = format!("{:.5e}", x);
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if !(-4..6).contains(&exp) {
        let mant = mant.trim_end_matches('0').trim_end_matches('.');
        return format!("{}e{}{:02}", mant, if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    let s = format!("{:.*}", (5 - exp).max(0) as usize, x);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Tick labels as `ScalarFormatter` writes them: the fewest decimals that tell
/// the ticks apart, `−` for minus, and a `1e5`-style multiplier for very large
/// or small values (returned separately).
pub(crate) fn labels(ticks: &[f64]) -> (Vec<String>, String) {
    let max = ticks.iter().fold(0f64, |m, t| m.max(t.abs()));
    let exp = if max >= 1e5 || (max > 0.0 && max < 1e-4) { max.log10().floor() as i32 } else { 0 };
    let k = 10f64.powi(exp);
    let vals: Vec<f64> = ticks.iter().map(|t| t / k).collect();
    let step = vals.windows(2).map(|w| (w[1] - w[0]).abs()).fold(f64::INFINITY, f64::min);
    let mut dec = 0;
    if step.is_finite() && step > 0.0 {
        while dec < 10 && vals.iter().any(|v| ((format!("{:.*}", dec, v).parse::<f64>().unwrap() - v).abs()) > step * 1e-6) {
            dec += 1;
        }
    }
    let labels = vals
        .iter()
        .map(|v| {
            let s = format!("{:.*}", dec, if v.abs() < step * 1e-9 { 0.0 } else { *v });
            s.replace('-', "\u{2212}")
        })
        .collect();
    let mult = if exp != 0 { format!("1e{}", exp).replace('-', "\u{2212}") } else { String::new() };
    (labels, mult)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locator_as_matplotlib() {
        assert_eq!(max_n_locator(-0.5, 10.5, 9), vec![-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(max_n_locator(-1.1, 1.1, 9), (-5..=5).map(|i| i as f64 * 0.25).collect::<Vec<_>>());
    }

    #[test]
    fn g_format() {
        assert_eq!(format_g(0.775), "0.775");
        assert_eq!(format_g(640.0), "640");
        assert_eq!(format_g(0.352272727), "0.352273");
        assert_eq!(format_g(1e-5), "1e-05");
        assert_eq!(format_g(1234567.0), "1.23457e+06");
    }

    #[test]
    fn tick_labels() {
        assert_eq!(labels(&[-0.5, 0.0, 0.5]).0, vec!["\u{2212}0.5", "0.0", "0.5"]);
        assert_eq!(labels(&[0.0, 2.0, 4.0]).0, vec!["0", "2", "4"]);
        assert_eq!(labels(&[0.0, 2e5, 4e5]), (vec!["0".into(), "2".into(), "4".into()], "1e5".into()));
    }
}

//! `ruai.plot`: plots for py0 programs, with matplotlib.pyplot's API
//! (`import ruai.plot as plt`), on [`crate::plot`]: `plt.show()` opens the
//! figures in the browser, `savefig` writes SVG or HTML. CPython stand-in:
//! `py0/python/ruai/plot.py` (matplotlib itself).

use crate::plot;
use crate::pyrt::err::{PyErr, VALUE_ERROR};
use crate::pyrt::lib::numpy::NdArray;
use std::cell::RefCell;
use std::rc::Rc;

type Fig = Rc<RefCell<plot::Figure>>;

fn verr(m: impl Into<String>) -> PyErr {
    PyErr::new(VALUE_ERROR, m)
}

/// Numbers to plot: a list of floats or ints, or a 1-D array.
pub trait Data {
    /// The values as floats.
    fn values(&self) -> Result<Vec<f64>, PyErr>;
}
impl Data for Vec<f64> {
    fn values(&self) -> Result<Vec<f64>, PyErr> {
        Ok(self.clone())
    }
}
impl Data for Vec<i64> {
    fn values(&self) -> Result<Vec<f64>, PyErr> {
        Ok(self.iter().map(|&v| v as f64).collect())
    }
}
impl Data for NdArray {
    fn values(&self) -> Result<Vec<f64>, PyErr> {
        if self.dims().len() > 1 {
            return Err(verr(format!("ruai.plot takes 1-D data, not an array of shape {}", self.shape())));
        }
        Ok(self.to_f64())
    }
}

fn floats(v: Vec<f64>) -> NdArray {
    let n = v.len();
    NdArray::floats(v, vec![n])
}

// ------------------------------------------------------------ objects

macro_rules! show_as {
    ($t:ty, |$s:ident| $e:expr) => {
        impl std::fmt::Display for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let $s = self;
                f.write_str(&$e)
            }
        }
        impl std::fmt::Debug for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(self, f)
            }
        }
    };
}

/// A figure (shared: copies are the same figure).
#[derive(Clone)]
pub struct Figure(Fig);
show_as!(Figure, |s| s.0.borrow().describe());
impl PartialEq for Figure {
    fn eq(&self, o: &Figure) -> bool {
        Rc::ptr_eq(&self.0, &o.0)
    }
}

/// An axes of a figure.
#[derive(Clone)]
pub struct Axes {
    fig: Fig,
    i: usize,
}
show_as!(Axes, |s| s.fig.borrow().axes[s.i].describe());
impl PartialEq for Axes {
    fn eq(&self, o: &Axes) -> bool {
        Rc::ptr_eq(&self.fig, &o.fig) && self.i == o.i
    }
}

/// A line (from `plot`, `axhline`, `axvline`).
#[derive(Clone)]
pub struct Line2D {
    fig: Fig,
    ax: usize,
    a: usize,
}
show_as!(Line2D, |s| format!("Line2D({})", s.get_label()));
impl PartialEq for Line2D {
    fn eq(&self, o: &Line2D) -> bool {
        Rc::ptr_eq(&self.fig, &o.fig) && (self.ax, self.a) == (o.ax, o.a)
    }
}

/// The bars of `bar` / `hist`.
#[derive(Clone, PartialEq)]
pub struct BarContainer(usize);
show_as!(BarContainer, |s| format!("<BarContainer object of {} artists>", s.0));
impl BarContainer {
    /// `len(bars)`
    pub fn __len__(&self) -> i64 {
        self.0 as i64
    }
}

/// The points of `scatter`.
#[derive(Clone, PartialEq)]
pub struct PathCollection(usize);
show_as!(PathCollection, |s| format!("<PathCollection of {} points>", s.0));

/// The image of `imshow`.
#[derive(Clone, PartialEq)]
pub struct AxesImage(usize, usize);
show_as!(AxesImage, |s| format!("<AxesImage of {}x{}>", s.0, s.1));

/// A text (title, label, `text`).
#[derive(Clone, PartialEq)]
pub struct Text(String);
show_as!(Text, |s| format!("Text({:?})", s.0));
impl Text {
    /// `t.get_text()`
    pub fn get_text(&self) -> String {
        self.0.clone()
    }
}

// ------------------------------------------------------------ pyplot's state

#[derive(Default)]
struct State {
    figs: Vec<Fig>,
    current: Option<usize>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn new_figure(figsize: Option<(f64, f64)>, dpi: f64) -> Fig {
    let (w, h) = figsize.unwrap_or((6.4, 4.8));
    let mut f = plot::Figure::new(w, h);
    f.dpi = dpi;
    let fig = Rc::new(RefCell::new(f));
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.figs.push(fig.clone());
        s.current = Some(s.figs.len() - 1);
    });
    fig
}

/// `plt.figure(figsize=(w, h), dpi=100)`
pub fn figure(figsize: Option<(f64, f64)>, dpi: f64) -> Figure {
    Figure(new_figure(figsize, dpi))
}

/// `plt.gcf()`: the current figure (a new one if there is none).
pub fn gcf() -> Figure {
    let cur = STATE.with(|s| {
        let s = s.borrow();
        s.current.map(|i| s.figs[i].clone())
    });
    Figure(cur.unwrap_or_else(|| new_figure(None, 100.0)))
}

/// `plt.gca()`: the current axes.
pub fn gca() -> Axes {
    let f = gcf().0;
    let i = f.borrow_mut().gca();
    Axes { fig: f, i }
}

/// `plt.subplots()`: a figure with one axes.
pub fn subplots1(figsize: Option<(f64, f64)>) -> (Figure, Axes) {
    let f = new_figure(figsize, 100.0);
    let i = f.borrow_mut().add_subplot(1, 1, 1);
    (Figure(f.clone()), Axes { fig: f, i })
}

/// `plt.subplots(nrows, ncols)`: a row or a column of axes.
pub fn subplots(nrows: i64, ncols: i64, figsize: Option<(f64, f64)>) -> Result<(Figure, Vec<Axes>), PyErr> {
    if nrows < 1 || ncols < 1 {
        return Err(verr(format!("Number of rows must be a positive integer, not {}", nrows.min(ncols))));
    }
    if nrows > 1 && ncols > 1 {
        return Err(verr("ruai.plot: subplots(nrows, ncols) with both above 1 is a 2-D array of Axes in matplotlib; use subplot(nrows, ncols, index)"));
    }
    if nrows * ncols == 1 {
        return Err(verr("ruai.plot: subplots(1, 1) is one Axes in matplotlib; use subplots()"));
    }
    let f = new_figure(figsize, 100.0);
    let axes = (1..=nrows * ncols)
        .map(|k| {
            let i = f.borrow_mut().add_subplot(nrows as usize, ncols as usize, k as usize);
            Axes { fig: f.clone(), i }
        })
        .collect();
    f.borrow_mut().current = Some(0);
    Ok((Figure(f), axes))
}

/// `plt.subplot(nrows, ncols, index)`
pub fn subplot(nrows: i64, ncols: i64, index: i64) -> Result<Axes, PyErr> {
    gcf().add_subplot(nrows, ncols, index)
}

/// `plt.xlim()`
pub fn xlim_get() -> (f64, f64) {
    gca().get_xlim()
}

/// `plt.ylim()`
pub fn ylim_get() -> (f64, f64) {
    gca().get_ylim()
}

/// `plt.tight_layout()`: the layout is fixed (accepted, does nothing).
pub fn tight_layout() {}

/// `plt.show()`: opens the figures in the browser, then closes them.
pub fn show() {
    crate::pyrt::flush();
    let figs = STATE.with(|s| std::mem::take(&mut *s.borrow_mut()).figs);
    let figs: Vec<plot::Figure> = figs.iter().map(|f| f.borrow().clone()).collect();
    let refs: Vec<&plot::Figure> = figs.iter().collect();
    if let Err(e) = plot::show(&refs) {
        eprintln!("ruai.plot: {}", e);
    }
}

/// `plt.close()`: the current figure.
pub fn close() {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(i) = s.current {
            s.figs.remove(i);
            s.current = s.figs.len().checked_sub(1);
        }
    });
}

/// `plt.close(fig)`
pub fn close_fig(fig: &Figure) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.figs.retain(|f| !Rc::ptr_eq(f, &fig.0));
        s.current = s.figs.len().checked_sub(1);
    });
}

/// `plt.close("all")`
pub fn close_all(which: &str) -> Result<(), PyErr> {
    if which != "all" {
        return Err(verr(format!("Unrecognized figure name {:?}", which)));
    }
    STATE.with(|s| *s.borrow_mut() = State::default());
    Ok(())
}

// ------------------------------------------------------------ Figure

impl Figure {
    /// `fig.add_subplot(nrows, ncols, index)`
    pub fn add_subplot(&self, nrows: i64, ncols: i64, index: i64) -> Result<Axes, PyErr> {
        if nrows < 1 || ncols < 1 || index < 1 || index > nrows * ncols {
            return Err(verr(format!("num must be an integer with 1 <= num <= {}, not {}", nrows * ncols, index)));
        }
        let i = self.0.borrow_mut().add_subplot(nrows as usize, ncols as usize, index as usize);
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            s.current = s.figs.iter().position(|f| Rc::ptr_eq(f, &self.0)).or(s.current);
        });
        Ok(Axes { fig: self.0.clone(), i })
    }
    /// `fig.axes`
    pub fn axes(&self) -> Vec<Axes> {
        (0..self.0.borrow().axes.len()).map(|i| Axes { fig: self.0.clone(), i }).collect()
    }
    /// `fig.suptitle(t)`
    pub fn suptitle(&self, t: &str) -> Text {
        self.0.borrow_mut().suptitle = t.to_string();
        Text(t.to_string())
    }
    /// `fig.set_size_inches(w, h)`
    pub fn set_size_inches(&self, w: f64, h: f64) {
        self.0.borrow_mut().set_size_inches(w, h);
    }
    /// `fig.get_size_inches()`
    pub fn get_size_inches(&self) -> NdArray {
        let (w, h) = self.0.borrow().size;
        floats(vec![w, h])
    }
    /// `fig.tight_layout()`: the layout is fixed (accepted, does nothing).
    pub fn tight_layout(&self) {}
    /// `fig.savefig(fname)`: `.svg` or `.html` (a page with a canvas).
    pub fn savefig(&self, fname: &str, dpi: Option<f64>) -> Result<(), PyErr> {
        let mut f = self.0.borrow().clone();
        if let Some(d) = dpi {
            f.dpi = d;
        }
        let ext = fname.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default();
        let text = match ext.as_str() {
            "svg" => plot::render::svg(&f),
            "html" | "htm" => plot::render::html(&[&f]),
            _ => return Err(verr(format!("Format '{}' is not supported by ruai.plot (supported formats: html, svg)", ext))),
        };
        std::fs::write(fname, text).map_err(|e| crate::pyrt::err::os_err(&e, fname))
    }
}

// ------------------------------------------------------------ Axes

impl Axes {
    fn with<T>(&self, f: impl FnOnce(&mut plot::Axes) -> T) -> T {
        f(&mut self.fig.borrow_mut().axes[self.i])
    }

    fn line(&self, a: usize) -> Line2D {
        Line2D { fig: self.fig.clone(), ax: self.i, a }
    }

    fn style(fmt: &str, label: &str, color: &str, linewidth: f64, linestyle: &str, marker: &str) -> Result<plot::Style, PyErr> {
        let s = plot::Style { color: color.into(), label: label.into(), linestyle: linestyle.into(), marker: marker.into(), linewidth };
        s.fmt(fmt).map_err(verr)
    }

    /// `ax.plot(x, y, fmt, label=, color=, linewidth=, linestyle=, marker=)`
    #[allow(clippy::too_many_arguments)]
    pub fn plot(&self, x: &impl Data, y: &impl Data, fmt: &str, label: &str, color: &str, linewidth: f64, linestyle: &str, marker: &str) -> Result<Vec<Line2D>, PyErr> {
        let (x, y) = (x.values()?, y.values()?);
        if x.len() != y.len() {
            return Err(verr(format!("x and y must have same first dimension, but have shapes ({},) and ({},)", x.len(), y.len())));
        }
        let style = Axes::style(fmt, label, color, linewidth, linestyle, marker)?;
        let a = self.with(|ax| ax.plot(x, y, &style));
        Ok(vec![self.line(a)])
    }
    /// `ax.plot(y, fmt, ...)`: x is `0, 1, 2, ...`
    #[allow(clippy::too_many_arguments)]
    pub fn plot_y(&self, y: &impl Data, fmt: &str, label: &str, color: &str, linewidth: f64, linestyle: &str, marker: &str) -> Result<Vec<Line2D>, PyErr> {
        let x: Vec<f64> = (0..y.values()?.len()).map(|i| i as f64).collect();
        self.plot(&x, y, fmt, label, color, linewidth, linestyle, marker)
    }
    /// `ax.scatter(x, y, s, color=, marker=, alpha=, label=)`
    #[allow(clippy::too_many_arguments)]
    pub fn scatter(&self, x: &impl Data, y: &impl Data, s: f64, color: &str, marker: &str, alpha: f64, label: &str) -> Result<PathCollection, PyErr> {
        let (x, y) = (x.values()?, y.values()?);
        if x.len() != y.len() {
            return Err(verr("x and y must be the same size"));
        }
        let n = x.len();
        let style = plot::Style { color: color.into(), label: label.into(), marker: marker.into(), ..Default::default() };
        self.with(|ax| ax.scatter(x, y, s, alpha, &style));
        Ok(PathCollection(n))
    }
    /// `ax.bar(x, height, width, color=, label=)`
    pub fn bar(&self, x: &impl Data, height: &impl Data, width: f64, color: &str, label: &str) -> Result<BarContainer, PyErr> {
        let (x, h) = (x.values()?, height.values()?);
        if x.len() != h.len() {
            return Err(verr(format!("shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape ({},) and arg 1 with shape ({},).", x.len(), h.len())));
        }
        let n = h.len();
        self.with(|ax| ax.bar(x, h, width, color, label));
        Ok(BarContainer(n))
    }
    /// `ax.bar(["a", "b"], height, ...)`: a categorical x axis.
    pub fn bar_cat(&self, x: &[String], height: &impl Data, width: f64, color: &str, label: &str) -> Result<BarContainer, PyErr> {
        let pos = self.with(|ax| ax.category_positions(x));
        self.bar(&pos, height, width, color, label)
    }
    /// `ax.hist(x, bins)`: `(counts, edges, bars)`.
    pub fn hist(&self, x: &impl Data, bins: i64, color: &str, label: &str) -> Result<(NdArray, NdArray, BarContainer), PyErr> {
        if bins < 1 {
            return Err(verr("`bins` must be positive, when an integer"));
        }
        let x = x.values()?;
        let (n, e) = self.with(|ax| ax.hist(&x, bins as usize, color, label));
        Ok((floats(n), floats(e), BarContainer(bins as usize)))
    }
    /// `ax.imshow(X, cmap=)` of a 2-D array.
    pub fn imshow(&self, x: &NdArray, cmap: &str) -> Result<AxesImage, PyErr> {
        let d = x.dims().to_vec();
        if d.len() != 2 {
            return Err(verr(format!("ruai.plot: imshow takes a 2-D array, not one of shape {}", x.shape())));
        }
        self.with(|ax| ax.imshow(x.to_f64(), d[0], d[1], cmap));
        Ok(AxesImage(d[0], d[1]))
    }
    /// `ax.text(x, y, s)`
    pub fn text(&self, x: f64, y: f64, s: &str) -> Text {
        self.with(|ax| ax.text(x, y, s));
        Text(s.to_string())
    }
    /// `ax.axhline(y, ...)`
    pub fn axhline(&self, y: f64, color: &str, linestyle: &str, linewidth: f64, label: &str) -> Line2D {
        let style = plot::Style { color: color.into(), label: label.into(), linestyle: linestyle.into(), marker: String::new(), linewidth };
        let a = self.with(|ax| ax.refline(true, y, &style));
        self.line(a)
    }
    /// `ax.axvline(x, ...)`
    pub fn axvline(&self, x: f64, color: &str, linestyle: &str, linewidth: f64, label: &str) -> Line2D {
        let style = plot::Style { color: color.into(), label: label.into(), linestyle: linestyle.into(), marker: String::new(), linewidth };
        let a = self.with(|ax| ax.refline(false, x, &style));
        self.line(a)
    }
    /// `ax.set_title(label)`
    pub fn set_title(&self, s: &str) -> Text {
        self.with(|ax| ax.title = s.to_string());
        Text(s.to_string())
    }
    /// `ax.set_xlabel(label)`
    pub fn set_xlabel(&self, s: &str) -> Text {
        self.with(|ax| ax.xlabel = s.to_string());
        Text(s.to_string())
    }
    /// `ax.set_ylabel(label)`
    pub fn set_ylabel(&self, s: &str) -> Text {
        self.with(|ax| ax.ylabel = s.to_string());
        Text(s.to_string())
    }
    /// `ax.get_title()`
    pub fn get_title(&self) -> String {
        self.with(|ax| ax.title.clone())
    }
    /// `ax.get_xlabel()`
    pub fn get_xlabel(&self) -> String {
        self.with(|ax| ax.xlabel.clone())
    }
    /// `ax.get_ylabel()`
    pub fn get_ylabel(&self) -> String {
        self.with(|ax| ax.ylabel.clone())
    }
    /// `ax.set_xlim(left, right)`
    pub fn set_xlim(&self, l: f64, r: f64) -> (f64, f64) {
        self.with(|ax| ax.xlim = Some((l, r)));
        (l, r)
    }
    /// `ax.set_ylim(bottom, top)`
    pub fn set_ylim(&self, b: f64, t: f64) -> (f64, f64) {
        self.with(|ax| ax.ylim = Some((b, t)));
        (b, t)
    }
    /// `ax.get_xlim()`
    pub fn get_xlim(&self) -> (f64, f64) {
        self.with(|ax| ax.get_xlim())
    }
    /// `ax.get_ylim()`
    pub fn get_ylim(&self) -> (f64, f64) {
        self.with(|ax| ax.get_ylim())
    }
    /// `ax.get_xticks()`
    pub fn get_xticks(&self) -> NdArray {
        floats(self.with(|ax| ax.get_xticks()))
    }
    /// `ax.get_yticks()`
    pub fn get_yticks(&self) -> NdArray {
        floats(self.with(|ax| ax.get_yticks()))
    }
    /// `ax.get_lines()`
    pub fn get_lines(&self) -> Vec<Line2D> {
        let idx: Vec<usize> = self.with(|ax| ax.artists.iter().enumerate().filter(|(_, a)| matches!(a.kind, plot::Kind::Line { .. } | plot::Kind::RefLine { .. })).map(|(i, _)| i).collect());
        idx.into_iter().map(|a| self.line(a)).collect()
    }
    /// `ax.legend(loc=)`
    pub fn legend(&self, loc: &str) {
        self.with(|ax| {
            ax.legend = true;
            ax.legend_loc = loc.to_string();
        });
    }
    /// `ax.grid(visible)`
    pub fn grid(&self, visible: bool) {
        self.with(|ax| ax.grid_lines = visible);
    }
}

// ------------------------------------------------------------ Line2D

impl Line2D {
    fn with<T>(&self, f: impl FnOnce(&plot::Artist) -> T) -> T {
        f(&self.fig.borrow().axes[self.ax].artists[self.a])
    }
    fn style(&self) -> plot::Style {
        self.with(|a| match &a.kind {
            plot::Kind::Line { style } | plot::Kind::RefLine { style, .. } => style.clone(),
            _ => plot::Style::default(),
        })
    }
    /// `line.get_label()`
    pub fn get_label(&self) -> String {
        self.with(|a| a.label.clone())
    }
    /// `line.set_label(s)`
    pub fn set_label(&self, s: &str) {
        let mut f = self.fig.borrow_mut();
        let a = &mut f.axes[self.ax].artists[self.a];
        a.label = s.to_string();
        if let plot::Kind::Line { style } | plot::Kind::RefLine { style, .. } = &mut a.kind {
            style.label = s.to_string();
        }
    }
    /// `line.get_color()`
    pub fn get_color(&self) -> String {
        self.with(|a| a.color.clone())
    }
    /// `line.get_xdata()`
    pub fn get_xdata(&self) -> NdArray {
        floats(self.with(|a| a.x.clone()))
    }
    /// `line.get_ydata()`
    pub fn get_ydata(&self) -> NdArray {
        floats(self.with(|a| a.y.clone()))
    }
    /// `line.get_linewidth()`
    pub fn get_linewidth(&self) -> f64 {
        let w = self.style().linewidth;
        if w > 0.0 {
            w
        } else {
            1.5
        }
    }
    /// `line.get_linestyle()`: `"-"`, `"--"`, `"-."`, `":"` or `"None"`.
    pub fn get_linestyle(&self) -> String {
        match self.style().linestyle.as_str() {
            "" | "solid" => "-".into(),
            "none" | "None" | " " => "None".into(),
            "dashed" => "--".into(),
            "dotted" => ":".into(),
            "dashdot" => "-.".into(),
            s => s.to_string(),
        }
    }
    /// `line.get_marker()`: the marker or `"None"`.
    pub fn get_marker(&self) -> String {
        let m = self.style().marker;
        if m.is_empty() {
            "None".into()
        } else {
            m
        }
    }
}

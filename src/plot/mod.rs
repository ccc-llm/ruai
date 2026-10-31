//! Plots in the style of matplotlib, drawn to an HTML canvas or SVG (feature `plot`).
//!
//! A [`Figure`] holds [`Axes`]; an axes holds lines, scatter points, bars,
//! histograms, images, text and reference lines. Limits and ticks follow
//! matplotlib's defaults (5% margins, sticky edges for bars and images,
//! `MaxNLocator` with steps 1, 2, 2.5, 5, 10), so `get_xlim` / `get_xticks`
//! agree with matplotlib. [`Figure::render`] gives drawing commands in pixels,
//! which [`render`] turns into SVG or an HTML page with a `<canvas>`;
//! [`show`] opens such a page in the browser.
//!
//! ```
//! use ruai::plot::{Figure, Style};
//! let mut fig = Figure::new(6.4, 4.8);
//! let ax = fig.add_subplot(1, 1, 1);
//! let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
//! let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
//! fig.axes[ax].plot(x, y, &Style::default().label("sin"));
//! fig.axes[ax].legend = true;
//! let (lo, hi) = fig.axes[ax].get_xlim(); // 5% margins, as matplotlib
//! assert!((lo + 0.245).abs() < 1e-12 && (hi - 5.145).abs() < 1e-12);
//! let svg = ruai::plot::render::svg(&fig);
//! assert!(svg.starts_with("<svg"));
//! ```

pub mod render;
mod ticks;

pub use render::{show, Cmd};
pub use ticks::{format_g, max_n_locator};

/// matplotlib's default color cycle (`tab10`).
pub const CYCLE: [&str; 10] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

/// A color as CSS: matplotlib's one-letter colors, `C0`..`C9`, `tab:NAME`, or
/// anything CSS understands (`"red"`, `"#ff8800"`).
pub fn css_color(c: &str) -> String {
    let one = match c {
        "b" => "#0000ff",
        "g" => "#008000",
        "r" => "#ff0000",
        "c" => "#00bfbf",
        "m" => "#bf00bf",
        "y" => "#bfbf00",
        "k" => "#000000",
        "w" => "#ffffff",
        _ => "",
    };
    if !one.is_empty() {
        return one.to_string();
    }
    if let Some(i) = c.strip_prefix('C').and_then(|d| d.parse::<usize>().ok()) {
        return CYCLE[i % 10].to_string();
    }
    if let Some(n) = c.strip_prefix("tab:") {
        let names = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"];
        if let Some(i) = names.iter().position(|x| *x == n || (n == "grey" && *x == "gray")) {
            return CYCLE[i].to_string();
        }
    }
    c.to_string()
}

/// How a line or its markers look; empty fields take the defaults.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Style {
    /// The color (see [`css_color`]; `""`: the next of the cycle).
    pub color: String,
    /// The legend label (`""`: none, matplotlib's `_childN`).
    pub label: String,
    /// `"-"`, `"--"`, `"-."`, `":"`, `"none"` (`""`: solid, or none when a
    /// format string gives only a marker).
    pub linestyle: String,
    /// One of `. , o v ^ < > s * + x D d p h | _` (`""`: none).
    pub marker: String,
    /// Points (`0`: 1.5).
    pub linewidth: f64,
}

impl Style {
    /// With this color.
    pub fn color(mut self, c: &str) -> Style {
        self.color = c.to_string();
        self
    }
    /// With this label.
    pub fn label(mut self, l: &str) -> Style {
        self.label = l.to_string();
        self
    }
    /// With this line style.
    pub fn linestyle(mut self, s: &str) -> Style {
        self.linestyle = s.to_string();
        self
    }
    /// With this marker.
    pub fn marker(mut self, m: &str) -> Style {
        self.marker = m.to_string();
        self
    }
    /// A matplotlib format string: `"r-"`, `"o"`, `"g--"`, `"k.:"`; its parts
    /// fill only fields that are still empty.
    pub fn fmt(mut self, f: &str) -> Result<Style, String> {
        let (mut color, mut ls, mut marker) = (String::new(), String::new(), String::new());
        let mut s = f.to_string();
        // `C0`..`C9`
        if let Some(i) = s.find('C').filter(|&i| s[i + 1..].starts_with(|c: char| c.is_ascii_digit())) {
            color = s[i..i + 2].to_string();
            s.replace_range(i..i + 2, "");
        }
        let mut it = s.as_str();
        while !it.is_empty() {
            let two = it.get(..2).unwrap_or("");
            if two == "--" || two == "-." {
                ls = two.to_string();
                it = &it[2..];
                continue;
            }
            let c = it.chars().next().unwrap();
            match c {
                '-' | ':' => ls = c.to_string(),
                'b' | 'g' | 'r' | 'c' | 'm' | 'y' | 'k' | 'w' if color.is_empty() => color = c.to_string(),
                '.' | ',' | 'o' | 'v' | '^' | '<' | '>' | 's' | '*' | '+' | 'x' | 'D' | 'd' | 'p' | 'h' | '|' | '_' => marker = c.to_string(),
                _ => return Err(format!("'{}' is not a valid format string (unrecognized character '{}')", f, c)),
            }
            it = &it[c.len_utf8()..];
        }
        if ls.is_empty() && !marker.is_empty() {
            ls = "none".to_string();
        }
        if self.color.is_empty() {
            self.color = color;
        }
        if self.linestyle.is_empty() {
            self.linestyle = ls;
        }
        if self.marker.is_empty() {
            self.marker = marker;
        }
        Ok(self)
    }
}

/// What an artist draws.
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    /// A line through the points.
    Line {
        /// Its look.
        style: Style,
    },
    /// `axhline` (`horizontal`) / `axvline`: a line across the axes at `x[0]` or `y[0]`.
    #[allow(missing_docs)]
    RefLine { horizontal: bool, style: Style },
    /// Marker area `s` in points² (matplotlib's `scatter(s=...)`).
    #[allow(missing_docs)]
    Scatter { s: f64, marker: String, alpha: f64 },
    /// Bars centered at `x` with heights `y`.
    /// (`width` NaN: per bar, in the second half of `x`).
    #[allow(missing_docs)]
    Bars { width: f64 },
    /// Text at a point.
    #[allow(missing_docs)]
    Text { s: String },
    /// An image of `rows` × `cols` values (`y` holds them, row-major) through `cmap`.
    #[allow(missing_docs)]
    Image { rows: usize, cols: usize, cmap: String },
}

/// Something drawn in an axes, in data coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Artist {
    /// What it is.
    pub kind: Kind,
    /// x coordinates.
    pub x: Vec<f64>,
    /// y coordinates (an image's values).
    pub y: Vec<f64>,
    /// CSS color, as given (see [`css_color`]).
    pub color: String,
    /// The legend label (one starting with `_` is not shown).
    pub label: String,
}

/// One plot area of a figure.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Axes {
    /// left, bottom, width, height as fractions of the figure.
    pub pos: [f64; 4],
    /// `(nrows, ncols, index)` of a subplot.
    pub grid: (usize, usize, usize),
    /// What it draws, in order.
    pub artists: Vec<Artist>,
    /// The title above it.
    pub title: String,
    /// The x axis label.
    pub xlabel: String,
    /// The y axis label.
    pub ylabel: String,
    /// x limits set by hand (`None`: autoscaled).
    pub xlim: Option<(f64, f64)>,
    /// y limits set by hand (`None`: autoscaled).
    pub ylim: Option<(f64, f64)>,
    /// Whether a legend is drawn.
    pub legend: bool,
    /// matplotlib's legend `loc` (`""`: `"best"`).
    pub legend_loc: String,
    /// Whether grid lines are drawn at the ticks.
    pub grid_lines: bool,
    /// Category names of a categorical x axis (bars with string positions).
    pub categories: Vec<String>,
    /// The figure's size in inches (for the number of ticks).
    pub fig_size: (f64, f64),
    line_cycle: usize,
    patch_cycle: usize,
    children: usize,
}

impl Axes {
    fn new(pos: [f64; 4], grid: (usize, usize, usize), fig_size: (f64, f64)) -> Axes {
        Axes { pos, grid, fig_size, ..Default::default() }
    }

    fn next_line_color(&mut self) -> String {
        let c = CYCLE[self.line_cycle % 10];
        self.line_cycle += 1;
        c.to_string()
    }
    fn next_patch_color(&mut self) -> String {
        let c = CYCLE[self.patch_cycle % 10];
        self.patch_cycle += 1;
        c.to_string()
    }
    fn child_label(&mut self, label: &str, n: usize) -> String {
        let l = if label.is_empty() { format!("_child{}", self.children) } else { label.to_string() };
        self.children += n;
        l
    }

    /// `ax.plot(x, y, fmt)`: returns the index of the line in `artists`.
    pub fn plot(&mut self, x: Vec<f64>, y: Vec<f64>, style: &Style) -> usize {
        let mut style = style.clone();
        let color = if style.color.is_empty() { self.next_line_color() } else { style.color.clone() };
        style.label = self.child_label(&style.label, 1);
        let label = style.label.clone();
        self.artists.push(Artist { kind: Kind::Line { style }, x, y, color, label });
        self.artists.len() - 1
    }

    /// `ax.axhline(y)` / `ax.axvline(x)`.
    pub fn refline(&mut self, horizontal: bool, at: f64, style: &Style) -> usize {
        let mut style = style.clone();
        let color = if style.color.is_empty() { self.next_line_color() } else { style.color.clone() };
        style.label = self.child_label(&style.label, 1);
        let label = style.label.clone();
        let (x, y) = if horizontal { (vec![], vec![at]) } else { (vec![at], vec![]) };
        self.artists.push(Artist { kind: Kind::RefLine { horizontal, style }, x, y, color, label });
        self.artists.len() - 1
    }

    /// `ax.scatter(x, y, s, alpha)` with the color, marker and label of `style`.
    pub fn scatter(&mut self, x: Vec<f64>, y: Vec<f64>, s: f64, alpha: f64, style: &Style) -> usize {
        let color = if style.color.is_empty() { self.next_patch_color() } else { style.color.clone() };
        let label = self.child_label(&style.label, 1);
        let marker = if style.marker.is_empty() { "o".to_string() } else { style.marker.clone() };
        self.artists.push(Artist { kind: Kind::Scatter { s, marker, alpha }, x, y, color, label });
        self.artists.len() - 1
    }

    /// `ax.bar(x, height, width)`; `categories` names the positions of a
    /// categorical axis (positions are then `0, 1, ...`, shared by later bars).
    pub fn bar(&mut self, x: Vec<f64>, height: Vec<f64>, width: f64, color: &str, label: &str) -> usize {
        let color = if color.is_empty() { self.next_patch_color() } else { color.to_string() };
        self.children += height.len();
        let label = if label.is_empty() { "_nolegend_".to_string() } else { label.to_string() };
        self.artists.push(Artist { kind: Kind::Bars { width }, x, y: height, color, label });
        self.artists.len() - 1
    }

    /// Positions of category names on the x axis (new names are appended).
    pub fn category_positions(&mut self, names: &[String]) -> Vec<f64> {
        names
            .iter()
            .map(|n| match self.categories.iter().position(|c| c == n) {
                Some(i) => i as f64,
                None => {
                    self.categories.push(n.clone());
                    (self.categories.len() - 1) as f64
                }
            })
            .collect()
    }

    /// `ax.hist(x, bins)`: the counts and the bin edges (as `np.histogram`).
    pub fn hist(&mut self, x: &[f64], bins: usize, color: &str, label: &str) -> (Vec<f64>, Vec<f64>) {
        let (counts, edges) = histogram(x, bins);
        let color = if color.is_empty() { self.next_patch_color() } else { color.to_string() };
        let label = if label.is_empty() { "_nolegend_".to_string() } else { label.to_string() };
        self.children += counts.len();
        let w: Vec<f64> = edges.windows(2).map(|e| e[1] - e[0]).collect();
        let centers: Vec<f64> = edges.windows(2).zip(&w).map(|(e, w)| e[0] + 0.5 * w).collect();
        // bars span their bins exactly: store the widths per bar in x's second half
        let mut xs = centers;
        xs.extend(&w);
        self.artists.push(Artist { kind: Kind::Bars { width: f64::NAN }, x: xs, y: counts.clone(), color, label });
        (counts, edges)
    }

    /// `ax.text(x, y, s)`.
    pub fn text(&mut self, x: f64, y: f64, s: &str) -> usize {
        self.children += 1;
        self.artists.push(Artist { kind: Kind::Text { s: s.to_string() }, x: vec![x], y: vec![y], color: "#000000".into(), label: String::new() });
        self.artists.len() - 1
    }

    /// `ax.imshow(values)` of `rows` × `cols`.
    pub fn imshow(&mut self, values: Vec<f64>, rows: usize, cols: usize, cmap: &str) -> usize {
        self.children += 1;
        let cmap = if cmap.is_empty() { "viridis".to_string() } else { cmap.to_string() };
        self.artists.push(Artist { kind: Kind::Image { rows, cols, cmap }, x: vec![], y: values, color: String::new(), label: "_nolegend_".into() });
        self.artists.len() - 1
    }

    /// Bars as (left, right, bottom, top) rectangles.
    pub fn bar_rects(a: &Artist) -> Vec<(f64, f64, f64, f64)> {
        let Kind::Bars { width } = a.kind else { return vec![] };
        let n = a.y.len();
        (0..n)
            .map(|i| {
                let (c, w) = if width.is_nan() { (a.x[i], a.x[n + i]) } else { (a.x[i], width) };
                let (b, t) = if a.y[i] >= 0.0 { (0.0, a.y[i]) } else { (a.y[i], 0.0) };
                (c - w / 2.0, c + w / 2.0, b, t)
            })
            .collect()
    }

    /// Data limits of one axis (`x`: true) and its sticky edges.
    fn data_interval(&self, x: bool) -> (Vec<f64>, Vec<f64>) {
        let (mut vals, mut sticky) = (vec![], vec![]);
        for a in &self.artists {
            match &a.kind {
                Kind::Line { .. } | Kind::Scatter { .. } => vals.extend(if x { &a.x } else { &a.y }.iter().copied().filter(|v| v.is_finite())),
                Kind::RefLine { horizontal, .. } => {
                    if *horizontal != x {
                        vals.extend(if x { &a.x } else { &a.y }.iter().copied().filter(|v| v.is_finite()));
                    }
                }
                Kind::Bars { .. } => {
                    for (l, r, b, t) in Axes::bar_rects(a) {
                        if x {
                            vals.extend([l, r]);
                        } else {
                            vals.extend([b, t]);
                            sticky.push(0.0);
                        }
                    }
                }
                Kind::Image { rows, cols, .. } => {
                    let (lo, hi) = if x { (-0.5, *cols as f64 - 0.5) } else { (-0.5, *rows as f64 - 0.5) };
                    vals.extend([lo, hi]);
                    sticky.extend([lo, hi]);
                }
                Kind::Text { .. } => {}
            }
        }
        (vals, sticky)
    }

    /// matplotlib's `autoscale_view` for one axis: data limits, 5% margins,
    /// sticky edges.
    fn autoscale(&self, x: bool) -> (f64, f64) {
        let (vals, mut sticky) = self.data_interval(x);
        let (mut x0, mut x1) = if vals.is_empty() {
            if !self.artists.iter().any(|a| !matches!(a.kind, Kind::Text { .. })) {
                return (0.0, 1.0);
            }
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            (vals.iter().copied().fold(f64::INFINITY, f64::min), vals.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        };
        (x0, x1) = ticks::nonsingular(x0, x1, 0.05, 1e-15);
        sticky.sort_by(f64::total_cmp);
        let tol = 1e-5 * x0.abs().max(x1.abs()).max((x1 - x0).abs());
        let x0bound = sticky.iter().copied().rev().find(|&s| s < x0 + tol);
        let x1bound = sticky.iter().copied().find(|&s| s > x1 - tol);
        let mut delta = (x1 - x0) * 0.05;
        if !delta.is_finite() {
            delta = 0.0;
        }
        (x0, x1) = (x0 - delta, x1 + delta);
        if let Some(b) = x0bound {
            x0 = x0.max(b);
        }
        if let Some(b) = x1bound {
            x1 = x1.min(b);
        }
        let (a, b) = ticks::nonsingular(x0, x1, 0.001, 1e-15);
        // an image shows row 0 at the top
        if !x && self.artists.iter().any(|a| matches!(a.kind, Kind::Image { .. })) {
            return (b, a);
        }
        (a, b)
    }

    /// `ax.get_xlim()`
    pub fn get_xlim(&self) -> (f64, f64) {
        self.xlim.unwrap_or_else(|| self.autoscale(true))
    }
    /// `ax.get_ylim()`
    pub fn get_ylim(&self) -> (f64, f64) {
        self.ylim.unwrap_or_else(|| self.autoscale(false))
    }

    /// The axes' size in inches.
    fn size_in(&self) -> (f64, f64) {
        (self.pos[2] * self.fig_size.0, self.pos[3] * self.fig_size.1)
    }

    /// `ax.get_xticks()`: tick positions (some may lie outside the limits).
    pub fn get_xticks(&self) -> Vec<f64> {
        if !self.categories.is_empty() {
            return (0..self.categories.len()).map(|i| i as f64).collect();
        }
        let (lo, hi) = self.get_xlim();
        // XAxis.get_tick_space: tick labels at most 3:1 wide, 10 pt
        let space = (self.size_in().0 * 72.0 / 30.0).floor() as usize;
        max_n_locator(lo.min(hi), lo.max(hi), space.clamp(1, 9))
    }
    /// `ax.get_yticks()`
    pub fn get_yticks(&self) -> Vec<f64> {
        let (lo, hi) = self.get_ylim();
        let space = (self.size_in().1 * 72.0 / 20.0).floor() as usize;
        max_n_locator(lo.min(hi), lo.max(hi), space.clamp(1, 9))
    }

    /// matplotlib's `str(ax)`: `Axes(0.125,0.11;0.775x0.77)`.
    pub fn describe(&self) -> String {
        let p = self.pos;
        format!("Axes({},{};{}x{})", format_g(p[0]), format_g(p[1]), format_g(p[2]), format_g(p[3]))
    }
}

/// `np.histogram(x, bins)`: counts (as floats, like `plt.hist`) and edges.
pub fn histogram(x: &[f64], bins: usize) -> (Vec<f64>, Vec<f64>) {
    let bins = bins.max(1);
    let (mut lo, mut hi) = if x.is_empty() {
        (0.0, 1.0)
    } else {
        (x.iter().copied().fold(f64::INFINITY, f64::min), x.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    };
    if lo == hi {
        lo -= 0.5;
        hi += 0.5;
    }
    // np.linspace
    let step = (hi - lo) / bins as f64;
    let mut edges: Vec<f64> = (0..=bins).map(|i| i as f64 * step + lo).collect();
    edges[bins] = hi;
    let mut counts = vec![0.0; bins];
    for &v in x {
        if !(lo..=hi).contains(&v) {
            continue;
        }
        let mut i = (((v - lo) / (hi - lo)) * bins as f64) as usize;
        i = i.min(bins - 1);
        if v < edges[i] {
            i -= 1;
        } else if i + 1 < bins && v >= edges[i + 1] {
            i += 1;
        }
        counts[i] += 1.0;
    }
    (counts, edges)
}

/// A figure: its size in inches, resolution and axes.
#[derive(Clone, Debug, PartialEq)]
pub struct Figure {
    /// Width and height in inches.
    pub size: (f64, f64),
    /// Pixels per inch.
    pub dpi: f64,
    /// Its axes.
    pub axes: Vec<Axes>,
    /// The title above all axes.
    pub suptitle: String,
    /// The axes that pyplot-style calls draw on.
    pub current: Option<usize>,
}

impl Figure {
    /// A figure of `w` × `h` inches at 100 dpi (matplotlib's default is 6.4 × 4.8).
    pub fn new(w: f64, h: f64) -> Figure {
        Figure { size: (w, h), dpi: 100.0, axes: vec![], suptitle: String::new(), current: None }
    }

    /// `fig.add_subplot(nrows, ncols, index)` (index from 1); an existing
    /// subplot of the same place is reused. Returns its index in `axes`.
    pub fn add_subplot(&mut self, nrows: usize, ncols: usize, index: usize) -> usize {
        if let Some(i) = self.axes.iter().position(|a| a.grid == (nrows, ncols, index)) {
            self.current = Some(i);
            return i;
        }
        let pos = subplot_pos(nrows, ncols, index);
        self.axes.push(Axes::new(pos, (nrows, ncols, index), self.size));
        self.current = Some(self.axes.len() - 1);
        self.axes.len() - 1
    }

    /// The current axes (a 1×1 subplot is made if there is none).
    pub fn gca(&mut self) -> usize {
        match self.current {
            Some(i) => i,
            None => self.add_subplot(1, 1, 1),
        }
    }

    /// `fig.set_size_inches(w, h)`
    pub fn set_size_inches(&mut self, w: f64, h: f64) {
        self.size = (w, h);
        for a in &mut self.axes {
            a.fig_size = (w, h);
        }
    }

    /// matplotlib's `str(fig)`: `Figure(640x480)`.
    pub fn describe(&self) -> String {
        format!("Figure({}x{})", format_g(self.size.0 * self.dpi), format_g(self.size.1 * self.dpi))
    }

    /// The drawing commands, in pixels.
    pub fn render(&self) -> Vec<Cmd> {
        render::figure(self)
    }
}

/// matplotlib's GridSpec positions with the default subplot parameters
/// (left .125, right .9, bottom .11, top .88, wspace = hspace = .2).
pub fn subplot_pos(nrows: usize, ncols: usize, index: usize) -> [f64; 4] {
    let (left, right, bottom, top, wspace, hspace) = (0.125, 0.9, 0.11, 0.88, 0.2, 0.2);
    let cell = |n: usize, total: f64, space: f64| -> Vec<(f64, f64)> {
        // cumulative [sep, cell] lengths, as GridSpec.get_grid_positions
        let c = total / (n as f64 + space * (n as f64 - 1.0));
        let sep = space * c;
        let mut acc = 0.0;
        let mut out = vec![];
        for i in 0..n {
            acc += if i == 0 { 0.0 } else { sep };
            let a = acc;
            acc += c;
            out.push((a, acc));
        }
        out
    };
    let (r, c) = ((index - 1) / ncols.max(1), (index - 1) % ncols.max(1));
    let rows = cell(nrows, top - bottom, hspace);
    let cols = cell(ncols, right - left, wspace);
    let (t, b) = (top - rows[r].0, top - rows[r].1);
    let (l, rr) = (left + cols[c].0, left + cols[c].1);
    [l, b, rr - l, t - b]
}

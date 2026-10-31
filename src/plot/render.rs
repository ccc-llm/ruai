//! Drawing a [`Figure`]: commands in pixels, written as SVG or as an HTML page
//! whose `<canvas>` a small script draws ([`CANVAS_JS`], also usable on its own
//! in a browser: `ruaiDraw(canvas, {w, h, cmds})`).

use super::{css_color, ticks, Artist, Axes, Figure, Kind};
use std::fmt::Write as _;

/// Text alignment: horizontal (`"left"`, `"center"`, `"right"`) and vertical
/// (`"top"`, `"middle"`, `"bottom"`, `"alphabetic"`), as the canvas API names them.
#[derive(Clone, Debug, PartialEq)]
pub struct Align(pub &'static str, pub &'static str);

/// A drawing command, in pixels from the top left corner.
#[derive(Clone, Debug, PartialEq)]
///
/// Colors are CSS colors, `alpha` is the opacity (0..1), widths and sizes are
/// pixels.
#[allow(missing_docs)]
pub enum Cmd {
    /// A filled rectangle with its top left corner at (x, y).
    Rect { x: f64, y: f64, w: f64, h: f64, fill: String, alpha: f64 },
    /// A polyline (a polygon if `closed`), stroked and / or filled; `dash`
    /// alternates dash and gap lengths (empty: solid).
    Path { pts: Vec<(f64, f64)>, closed: bool, stroke: Option<String>, width: f64, dash: Vec<f64>, fill: Option<String>, alpha: f64 },
    /// A circle of radius `r`.
    Circle { x: f64, y: f64, r: f64, fill: Option<String>, stroke: Option<String>, width: f64, alpha: f64 },
    /// Text `s` of font size `size` at (x, y), aligned by `align`, rotated
    /// `rotate` degrees counterclockwise.
    Text { x: f64, y: f64, s: String, size: f64, color: String, align: Align, rotate: f64, bold: bool },
    /// Clip what follows to a rectangle, until [`Cmd::Unclip`].
    Clip { x: f64, y: f64, w: f64, h: f64 },
    /// Ends the last [`Cmd::Clip`].
    Unclip,
}

const FONT: &str = "DejaVu Sans, Helvetica, Arial, sans-serif";

/// An estimate of a text's width (the renderers measure nothing).
fn text_width(s: &str, size: f64) -> f64 {
    s.chars().count() as f64 * size * 0.6
}

struct Ctx {
    cmds: Vec<Cmd>,
    pt: f64,
}

impl Ctx {
    fn text(&mut self, x: f64, y: f64, s: &str, size_pt: f64, align: Align, rotate: f64) {
        if s.is_empty() {
            return;
        }
        self.cmds.push(Cmd::Text { x, y, s: s.to_string(), size: size_pt * self.pt, color: "#000000".into(), align, rotate, bold: false });
    }
    fn line(&mut self, pts: Vec<(f64, f64)>, color: &str, width_pt: f64, dash: Vec<f64>) {
        self.cmds.push(Cmd::Path { pts, closed: false, stroke: Some(css_color(color)), width: width_pt * self.pt, dash, fill: None, alpha: 1.0 });
    }

    /// A marker of diameter `d` pixels.
    fn marker(&mut self, m: &str, x: f64, y: f64, d: f64, color: &str, alpha: f64) {
        let c = css_color(color);
        let r = d / 2.0;
        let poly = |n: usize, r: f64, rot: f64| -> Vec<(f64, f64)> {
            (0..n)
                .map(|i| {
                    let a = rot + i as f64 * std::f64::consts::TAU / n as f64;
                    (x + r * a.sin(), y - r * a.cos())
                })
                .collect()
        };
        let fill = |pts: Vec<(f64, f64)>| Cmd::Path { pts, closed: true, stroke: None, width: 0.0, dash: vec![], fill: Some(c.clone()), alpha };
        let stroke = |pts: Vec<(f64, f64)>, w: f64| Cmd::Path { pts, closed: false, stroke: Some(c.clone()), width: w, dash: vec![], fill: None, alpha };
        let w = self.pt;
        let cmd = match m {
            "o" => Cmd::Circle { x, y, r, fill: Some(c.clone()), stroke: None, width: 0.0, alpha },
            "." => Cmd::Circle { x, y, r: r / 2.0, fill: Some(c.clone()), stroke: None, width: 0.0, alpha },
            "," => Cmd::Rect { x, y, w: 1.0, h: 1.0, fill: c.clone(), alpha },
            "s" => fill(poly(4, r * std::f64::consts::SQRT_2, std::f64::consts::FRAC_PI_4)),
            "D" => fill(poly(4, r, 0.0)),
            "d" => fill(vec![(x, y - r), (x + r * 0.6, y), (x, y + r), (x - r * 0.6, y)]),
            "^" => fill(poly(3, r, 0.0)),
            "v" => fill(poly(3, r, std::f64::consts::PI)),
            ">" => fill(poly(3, r, std::f64::consts::FRAC_PI_2)),
            "<" => fill(poly(3, r, -std::f64::consts::FRAC_PI_2)),
            "p" => fill(poly(5, r, 0.0)),
            "h" => fill(poly(6, r, 0.0)),
            "*" => fill((0..10).map(|i| {
                let a = i as f64 * std::f64::consts::PI / 5.0;
                let rr = if i % 2 == 0 { r } else { r * 0.38 };
                (x + rr * a.sin(), y - rr * a.cos())
            }).collect()),
            "+" => {
                self.cmds.push(stroke(vec![(x - r, y), (x + r, y)], w));
                stroke(vec![(x, y - r), (x, y + r)], w)
            }
            "x" => {
                let q = r * std::f64::consts::FRAC_1_SQRT_2;
                self.cmds.push(stroke(vec![(x - q, y - q), (x + q, y + q)], w));
                stroke(vec![(x - q, y + q), (x + q, y - q)], w)
            }
            "|" => stroke(vec![(x, y - r), (x, y + r)], w),
            "_" => stroke(vec![(x - r, y), (x + r, y)], w),
            _ => return,
        };
        self.cmds.push(cmd);
    }
}

/// Dash pattern of a line style, in pixels (matplotlib's, scaled by the width).
fn dashes(ls: &str, lw_px: f64) -> Vec<f64> {
    let base: &[f64] = match ls {
        "--" | "dashed" => &[3.7, 1.6],
        ":" | "dotted" => &[1.0, 1.65],
        "-." | "dashdot" => &[6.4, 1.6, 1.0, 1.6],
        _ => &[],
    };
    base.iter().map(|d| d * lw_px).collect()
}

/// A colormap value for `t` in 0..1.
fn colormap(name: &str, t: f64) -> String {
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    let gray = |v: f64| {
        let g = (v * 255.0).round() as u8;
        format!("#{:02x}{:02x}{:02x}", g, g, g)
    };
    match name {
        "gray" | "grey" => gray(t),
        "gray_r" | "grey_r" | "binary" => gray(1.0 - t),
        _ => {
            const VIRIDIS: [(u8, u8, u8); 9] = [
                (0x44, 0x01, 0x54), (0x48, 0x28, 0x78), (0x3e, 0x49, 0x89), (0x31, 0x68, 0x8e), (0x26, 0x82, 0x8e),
                (0x1f, 0x9e, 0x89), (0x35, 0xb7, 0x79), (0x6e, 0xce, 0x58), (0xfd, 0xe7, 0x25),
            ];
            let f = t * 8.0;
            let i = (f.floor() as usize).min(7);
            let u = f - i as f64;
            let (a, b) = (VIRIDIS[i], VIRIDIS[i + 1]);
            let mix = |p: u8, q: u8| (p as f64 + (q as f64 - p as f64) * u).round() as u8;
            format!("#{:02x}{:02x}{:02x}", mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
        }
    }
}

/// The figure's drawing commands.
pub fn figure(fig: &Figure) -> Vec<Cmd> {
    let (w, h) = (fig.size.0 * fig.dpi, fig.size.1 * fig.dpi);
    let mut c = Ctx { cmds: vec![Cmd::Rect { x: 0.0, y: 0.0, w, h, fill: "#ffffff".into(), alpha: 1.0 }], pt: fig.dpi / 72.0 };
    for ax in &fig.axes {
        axes(&mut c, ax, w, h);
    }
    c.text(w / 2.0, h * 0.02, &fig.suptitle, 12.0, Align("center", "top"), 0.0);
    c.cmds
}

fn axes(c: &mut Ctx, ax: &Axes, fw: f64, fh: f64) {
    let pt = c.pt;
    let (xl, yl) = (ax.get_xlim(), ax.get_ylim());
    let (mut x0, mut y0, mut w, mut h) = (ax.pos[0] * fw, fh - (ax.pos[1] + ax.pos[3]) * fh, ax.pos[2] * fw, ax.pos[3] * fh);
    // images keep square pixels: shrink the box, centered
    if ax.artists.iter().any(|a| matches!(a.kind, Kind::Image { .. })) {
        let (dw, dh) = ((xl.1 - xl.0).abs(), (yl.1 - yl.0).abs());
        let s = (w / dw).min(h / dh);
        let (nw, nh) = (dw * s, dh * s);
        x0 += (w - nw) / 2.0;
        y0 += (h - nh) / 2.0;
        (w, h) = (nw, nh);
    }
    let px = |x: f64| x0 + (x - xl.0) / (xl.1 - xl.0) * w;
    let py = |y: f64| y0 + h - (y - yl.0) / (yl.1 - yl.0) * h;
    let inside = |t: f64, l: (f64, f64)| {
        let (a, b) = (l.0.min(l.1), l.0.max(l.1));
        let tol = (b - a) * 1e-10;
        t >= a - tol && t <= b + tol
    };
    let xt: Vec<f64> = ax.get_xticks().into_iter().filter(|&t| inside(t, xl)).collect();
    let yt: Vec<f64> = ax.get_yticks().into_iter().filter(|&t| inside(t, yl)).collect();
    if ax.grid_lines {
        for &t in &xt {
            c.line(vec![(px(t), y0), (px(t), y0 + h)], "#b0b0b0", 0.8, vec![]);
        }
        for &t in &yt {
            c.line(vec![(x0, py(t)), (x0 + w, py(t))], "#b0b0b0", 0.8, vec![]);
        }
    }
    c.cmds.push(Cmd::Clip { x: x0, y: y0, w, h });
    for a in &ax.artists {
        artist(c, a, &px, &py, (x0, y0, w, h));
    }
    c.cmds.push(Cmd::Unclip);
    // spines
    c.cmds.push(Cmd::Path { pts: vec![(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)], closed: true, stroke: Some("#000000".into()), width: 0.8 * pt, dash: vec![], fill: None, alpha: 1.0 });
    // ticks and their labels
    let (tl, pad) = (3.5 * pt, 3.5 * pt);
    let (xlabels, xmult) = if ax.categories.is_empty() { ticks::labels(&xt) } else { (xt.iter().map(|&t| ax.categories.get(t as usize).cloned().unwrap_or_default()).collect(), String::new()) };
    for (t, s) in xt.iter().zip(&xlabels) {
        c.line(vec![(px(*t), y0 + h), (px(*t), y0 + h + tl)], "#000000", 0.8, vec![]);
        c.text(px(*t), y0 + h + tl + pad, s, 10.0, Align("center", "top"), 0.0);
    }
    let (ylabels, ymult) = ticks::labels(&yt);
    let mut ywidth: f64 = 0.0;
    for (t, s) in yt.iter().zip(&ylabels) {
        c.line(vec![(x0 - tl, py(*t)), (x0, py(*t))], "#000000", 0.8, vec![]);
        c.text(x0 - tl - pad, py(*t), s, 10.0, Align("right", "middle"), 0.0);
        ywidth = ywidth.max(text_width(s, 10.0 * pt));
    }
    let font = 10.0 * pt;
    c.text(x0 + w, y0 + h + tl + pad + font * 1.2, &xmult, 10.0, Align("right", "top"), 0.0);
    c.text(x0, y0 - 4.0 * pt, &ymult, 10.0, Align("left", "bottom"), 0.0);
    c.text(x0 + w / 2.0, y0 + h + tl + pad + font * 1.2 + 4.0 * pt, &ax.xlabel, 10.0, Align("center", "top"), 0.0);
    c.text(x0 - tl - pad - ywidth - 4.0 * pt, y0 + h / 2.0, &ax.ylabel, 10.0, Align("center", "bottom"), 90.0);
    c.text(x0 + w / 2.0, y0 - 6.0 * pt, &ax.title, 12.0, Align("center", "bottom"), 0.0);
    if ax.legend {
        legend(c, ax, (x0, y0, w, h), &px, &py);
    }
}

fn artist(c: &mut Ctx, a: &Artist, px: &dyn Fn(f64) -> f64, py: &dyn Fn(f64) -> f64, (x0, y0, w, h): (f64, f64, f64, f64)) {
    let pt = c.pt;
    match &a.kind {
        Kind::Line { style } => {
            let lw = if style.linewidth > 0.0 { style.linewidth } else { 1.5 };
            if style.linestyle != "none" && style.linestyle != "None" && style.linestyle != " " {
                // NaN breaks the line
                let mut run: Vec<(f64, f64)> = vec![];
                for (x, y) in a.x.iter().zip(&a.y) {
                    if x.is_finite() && y.is_finite() {
                        run.push((px(*x), py(*y)));
                    } else if !run.is_empty() {
                        c.line(std::mem::take(&mut run), &a.color, lw, dashes(&style.linestyle, lw * pt));
                    }
                }
                if run.len() > 1 {
                    c.line(run, &a.color, lw, dashes(&style.linestyle, lw * pt));
                }
            }
            if !style.marker.is_empty() {
                for (x, y) in a.x.iter().zip(&a.y) {
                    c.marker(&style.marker, px(*x), py(*y), 6.0 * pt, &a.color, 1.0);
                }
            }
        }
        Kind::RefLine { horizontal, style } => {
            let lw = if style.linewidth > 0.0 { style.linewidth } else { 1.5 };
            let pts = if *horizontal { vec![(x0, py(a.y[0])), (x0 + w, py(a.y[0]))] } else { vec![(px(a.x[0]), y0), (px(a.x[0]), y0 + h)] };
            c.line(pts, &a.color, lw, dashes(&style.linestyle, lw * pt));
        }
        Kind::Scatter { s, marker, alpha } => {
            for (x, y) in a.x.iter().zip(&a.y) {
                c.marker(marker, px(*x), py(*y), s.sqrt() * pt, &a.color, *alpha);
            }
        }
        Kind::Bars { .. } => {
            for (l, r, b, t) in Axes::bar_rects(a) {
                let (xa, xb, ya, yb) = (px(l), px(r), py(t), py(b));
                c.cmds.push(Cmd::Rect { x: xa.min(xb), y: ya.min(yb), w: (xb - xa).abs(), h: (yb - ya).abs(), fill: css_color(&a.color), alpha: 1.0 });
            }
        }
        Kind::Text { s } => c.text(px(a.x[0]), py(a.y[0]), s, 10.0, Align("left", "alphabetic"), 0.0),
        Kind::Image { rows, cols, cmap } => {
            let finite = a.y.iter().copied().filter(|v| v.is_finite());
            let (lo, hi) = finite.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| (l.min(v), h.max(v)));
            let span = if hi > lo { hi - lo } else { 1.0 };
            for r in 0..*rows {
                for col in 0..*cols {
                    let v = a.y[r * cols + col];
                    let (xa, xb) = (px(col as f64 - 0.5), px(col as f64 + 0.5));
                    let (ya, yb) = (py(r as f64 - 0.5), py(r as f64 + 0.5));
                    // overlap by half a pixel so cells leave no seams
                    c.cmds.push(Cmd::Rect { x: xa.min(xb), y: ya.min(yb), w: (xb - xa).abs() + 0.5, h: (yb - ya).abs() + 0.5, fill: colormap(cmap, (v - lo) / span), alpha: 1.0 });
                }
            }
        }
    }
}

fn legend(c: &mut Ctx, ax: &Axes, (x0, y0, w, h): (f64, f64, f64, f64), px: &dyn Fn(f64) -> f64, py: &dyn Fn(f64) -> f64) {
    let entries: Vec<&Artist> = ax.artists.iter().filter(|a| !a.label.is_empty() && !a.label.starts_with('_')).collect();
    if entries.is_empty() {
        return;
    }
    let f = 10.0 * c.pt;
    let (pad, spacing, handle, gap, margin) = (0.4 * f, 0.5 * f, 2.0 * f, 0.8 * f, 0.5 * f);
    let tw = entries.iter().map(|a| text_width(&a.label, f)).fold(0.0, f64::max);
    let bw = pad * 2.0 + handle + gap + tw;
    let bh = pad * 2.0 + entries.len() as f64 * f + (entries.len() - 1) as f64 * spacing;
    let place = |loc: &str| -> (f64, f64) {
        let (l, cx, r) = (x0 + margin, x0 + (w - bw) / 2.0, x0 + w - margin - bw);
        let (t, cy, b) = (y0 + margin, y0 + (h - bh) / 2.0, y0 + h - margin - bh);
        match loc {
            "upper left" => (l, t),
            "lower left" => (l, b),
            "lower right" => (r, b),
            "right" | "center right" => (r, cy),
            "center left" => (l, cy),
            "lower center" => (cx, b),
            "upper center" => (cx, t),
            "center" => (cx, cy),
            _ => (r, t),
        }
    };
    let (lx, ly) = if ax.legend_loc.is_empty() || ax.legend_loc == "best" {
        // the place covering the fewest data points (matplotlib's order on ties)
        let pts: Vec<(f64, f64)> = ax
            .artists
            .iter()
            .filter(|a| matches!(a.kind, Kind::Line { .. } | Kind::Scatter { .. }))
            .flat_map(|a| a.x.iter().zip(&a.y).map(|(x, y)| (px(*x), py(*y))))
            .collect();
        let locs = ["upper right", "upper left", "lower left", "lower right", "center right", "center left", "lower center", "upper center", "center"];
        let bad = |(lx, ly): (f64, f64)| pts.iter().filter(|(x, y)| *x >= lx && *x <= lx + bw && *y >= ly && *y <= ly + bh).count();
        locs.iter().map(|l| place(l)).min_by_key(|p| bad(*p)).unwrap()
    } else {
        place(&ax.legend_loc)
    };
    c.cmds.push(Cmd::Rect { x: lx, y: ly, w: bw, h: bh, fill: "#ffffff".into(), alpha: 0.8 });
    c.cmds.push(Cmd::Path { pts: vec![(lx, ly), (lx + bw, ly), (lx + bw, ly + bh), (lx, ly + bh)], closed: true, stroke: Some("#cccccc".into()), width: c.pt, dash: vec![], fill: None, alpha: 1.0 });
    for (i, a) in entries.iter().enumerate() {
        let cy = ly + pad + i as f64 * (f + spacing) + f / 2.0;
        let hx = lx + pad;
        match &a.kind {
            Kind::Line { style } | Kind::RefLine { style, .. } => {
                let lw = if style.linewidth > 0.0 { style.linewidth } else { 1.5 };
                if style.linestyle != "none" {
                    c.line(vec![(hx, cy), (hx + handle, cy)], &a.color, lw, dashes(&style.linestyle, lw * c.pt));
                }
                if !style.marker.is_empty() {
                    c.marker(&style.marker, hx + handle / 2.0, cy, 6.0 * c.pt, &a.color, 1.0);
                }
            }
            Kind::Scatter { s, marker, alpha } => c.marker(marker, hx + handle / 2.0, cy, s.sqrt() * c.pt, &a.color, *alpha),
            _ => c.cmds.push(Cmd::Rect { x: hx, y: cy - 0.35 * f, w: handle, h: 0.7 * f, fill: css_color(&a.color), alpha: 1.0 }),
        }
        c.text(hx + handle + gap, cy, &a.label, 10.0, Align("left", "middle"), 0.0);
    }
}

// ------------------------------------------------------------ SVG

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn num(x: f64) -> String {
    let s = format!("{:.2}", x);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.to_string() }
}

/// The figure as an SVG document.
pub fn svg(fig: &Figure) -> String {
    let (w, h) = (fig.size.0 * fig.dpi, fig.size.1 * fig.dpi);
    let mut s = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" font-family=\"{}\">\n", num(w), num(h), num(w), num(h), FONT);
    let mut clips = 0;
    let paint = |fill: &Option<String>, stroke: &Option<String>, width: f64, alpha: f64| {
        let mut a = format!(" fill=\"{}\"", fill.as_deref().unwrap_or("none"));
        if let Some(st) = stroke {
            let _ = write!(a, " stroke=\"{}\" stroke-width=\"{}\"", st, num(width));
        }
        if alpha < 1.0 {
            let _ = write!(a, " opacity=\"{}\"", num(alpha));
        }
        a
    };
    for c in fig.render() {
        match c {
            Cmd::Rect { x, y, w, h, fill, alpha } => {
                let _ = writeln!(s, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"{} shape-rendering=\"crispEdges\"/>", num(x), num(y), num(w), num(h), paint(&Some(fill), &None, 0.0, alpha));
            }
            Cmd::Path { pts, closed, stroke, width, dash, fill, alpha } => {
                let p: Vec<String> = pts.iter().map(|(x, y)| format!("{},{}", num(*x), num(*y))).collect();
                let tag = if closed { "polygon" } else { "polyline" };
                let mut extra = String::new();
                if !dash.is_empty() {
                    let d: Vec<String> = dash.iter().map(|d| num(*d)).collect();
                    let _ = write!(extra, " stroke-dasharray=\"{}\"", d.join(" "));
                }
                let _ = writeln!(s, "<{} points=\"{}\"{}{} stroke-linejoin=\"round\"/>", tag, p.join(" "), paint(&fill, &stroke, width, alpha), extra);
            }
            Cmd::Circle { x, y, r, fill, stroke, width, alpha } => {
                let _ = writeln!(s, "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"{}/>", num(x), num(y), num(r), paint(&fill, &stroke, width, alpha));
            }
            Cmd::Text { x, y, s: t, size, color, align, rotate, bold } => {
                let anchor = match align.0 { "left" => "start", "right" => "end", _ => "middle" };
                let base = match align.1 { "top" => "hanging", "middle" => "central", "bottom" => "text-after-edge", _ => "alphabetic" };
                let rot = if rotate != 0.0 { format!(" transform=\"rotate({} {} {})\"", num(-rotate), num(x), num(y)) } else { String::new() };
                let weight = if bold { " font-weight=\"bold\"" } else { "" };
                let _ = writeln!(s, "<text x=\"{}\" y=\"{}\" font-size=\"{}\" fill=\"{}\" text-anchor=\"{}\" dominant-baseline=\"{}\"{}{}>{}</text>", num(x), num(y), num(size), color, anchor, base, rot, weight, esc(&t));
            }
            Cmd::Clip { x, y, w, h } => {
                clips += 1;
                let _ = writeln!(s, "<clipPath id=\"c{}\"><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/></clipPath><g clip-path=\"url(#c{})\">", clips, num(x), num(y), num(w), num(h), clips);
            }
            Cmd::Unclip => s.push_str("</g>\n"),
        }
    }
    s.push_str("</svg>\n");
    s
}

// ------------------------------------------------------------ canvas

fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '<' => o.push_str("\\u003c"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn jopt(s: &Option<String>) -> String {
    s.as_deref().map(jstr).unwrap_or_else(|| "null".into())
}

/// The figure as JSON for [`CANVAS_JS`]: `{"w":..,"h":..,"cmds":[...]}`.
pub fn canvas_json(fig: &Figure) -> String {
    let (w, h) = (fig.size.0 * fig.dpi, fig.size.1 * fig.dpi);
    let cmds: Vec<String> = fig
        .render()
        .into_iter()
        .map(|c| match c {
            Cmd::Rect { x, y, w, h, fill, alpha } => format!("[\"r\",{},{},{},{},{},{}]", num(x), num(y), num(w), num(h), jstr(&fill), num(alpha)),
            Cmd::Path { pts, closed, stroke, width, dash, fill, alpha } => {
                let p: Vec<String> = pts.iter().flat_map(|(x, y)| [num(*x), num(*y)]).collect();
                let d: Vec<String> = dash.iter().map(|d| num(*d)).collect();
                format!("[\"p\",[{}],{},{},{},[{}],{},{}]", p.join(","), closed, jopt(&stroke), num(width), d.join(","), jopt(&fill), num(alpha))
            }
            Cmd::Circle { x, y, r, fill, stroke, width, alpha } => format!("[\"c\",{},{},{},{},{},{},{}]", num(x), num(y), num(r), jopt(&fill), jopt(&stroke), num(width), num(alpha)),
            Cmd::Text { x, y, s, size, color, align, rotate, bold } => {
                format!("[\"t\",{},{},{},{},{},\"{}\",\"{}\",{},{}]", num(x), num(y), jstr(&s), num(size), jstr(&color), align.0, align.1, num(rotate), bold)
            }
            Cmd::Clip { x, y, w, h } => format!("[\"clip\",{},{},{},{}]", num(x), num(y), num(w), num(h)),
            Cmd::Unclip => "[\"unclip\"]".into(),
        })
        .collect();
    format!("{{\"w\":{},\"h\":{},\"cmds\":[{}]}}", num(w), num(h), cmds.join(","))
}

/// Draws [`canvas_json`]'s figure on a canvas: `ruaiDraw(canvas, figure)`.
pub const CANVAS_JS: &str = r#"function ruaiDraw(cv, fig) {
  const r = window.devicePixelRatio || 1;
  cv.width = Math.round(fig.w * r); cv.height = Math.round(fig.h * r);
  cv.style.width = fig.w + "px"; cv.style.height = fig.h + "px";
  const g = cv.getContext("2d");
  g.setTransform(r, 0, 0, r, 0, 0);
  g.lineJoin = "round";
  const FONT = "DejaVu Sans, Helvetica, Arial, sans-serif";
  for (const c of fig.cmds) {
    switch (c[0]) {
      case "r": g.globalAlpha = c[6]; g.fillStyle = c[5]; g.fillRect(c[1], c[2], c[3], c[4]); break;
      case "p": {
        const p = c[1]; g.globalAlpha = c[7]; g.beginPath(); g.moveTo(p[0], p[1]);
        for (let i = 2; i < p.length; i += 2) g.lineTo(p[i], p[i + 1]);
        if (c[2]) g.closePath();
        if (c[6]) { g.fillStyle = c[6]; g.fill(); }
        if (c[3]) { g.setLineDash(c[5]); g.strokeStyle = c[3]; g.lineWidth = c[4]; g.stroke(); g.setLineDash([]); }
        break;
      }
      case "c":
        g.globalAlpha = c[7]; g.beginPath(); g.arc(c[1], c[2], c[3], 0, 2 * Math.PI);
        if (c[4]) { g.fillStyle = c[4]; g.fill(); }
        if (c[5]) { g.strokeStyle = c[5]; g.lineWidth = c[6]; g.stroke(); }
        break;
      case "t":
        g.save(); g.globalAlpha = 1; g.translate(c[1], c[2]); g.rotate(-c[8] * Math.PI / 180);
        g.font = (c[9] ? "bold " : "") + c[4] + "px " + FONT; g.fillStyle = c[5];
        g.textAlign = c[6]; g.textBaseline = c[7]; g.fillText(c[3], 0, 0); g.restore();
        break;
      case "clip": g.save(); g.beginPath(); g.rect(c[1], c[2], c[3], c[4]); g.clip(); break;
      case "unclip": g.restore(); break;
    }
  }
  g.globalAlpha = 1;
}
"#;

/// An HTML page drawing the figures on canvases.
pub fn html(figs: &[&Figure]) -> String {
    let mut s = String::from("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>ruai.plot</title>\n<style>body{margin:16px;background:#f4f4f4;font-family:sans-serif}figure{display:inline-block;margin:0 16px 16px 0;padding:0;background:#fff;box-shadow:0 1px 4px #0003}canvas{display:block}</style></head><body>\n");
    for i in 0..figs.len() {
        let _ = writeln!(s, "<figure><canvas id=\"f{}\"></canvas></figure>", i);
    }
    let _ = writeln!(s, "<script>\n{}", CANVAS_JS);
    for (i, f) in figs.iter().enumerate() {
        let _ = writeln!(s, "ruaiDraw(document.getElementById(\"f{}\"), {});", i, canvas_json(f));
    }
    s.push_str("</script></body></html>\n");
    s
}

/// Shows figures: writes them to an HTML page in the temp directory and
/// opens it in the browser (`open` / `xdg-open` / `start`). Returns the page's
/// path. With `MPLBACKEND=agg` (as matplotlib) nothing is shown.
pub fn show(figs: &[&Figure]) -> std::io::Result<Option<std::path::PathBuf>> {
    if figs.is_empty() || std::env::var("MPLBACKEND").is_ok_and(|b| b.eq_ignore_ascii_case("agg")) {
        return Ok(None);
    }
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("ruai-plot-{}-{}.html", std::process::id(), n));
    std::fs::write(&path, html(figs))?;
    let mut cmd = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else if cfg!(windows) {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", ""]);
        c
    } else {
        std::process::Command::new("xdg-open")
    };
    if cmd.arg(&path).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().map(|s| !s.success()).unwrap_or(true) {
        eprintln!("ruai.plot: open {} in a browser", path.display());
    }
    Ok(Some(path))
}

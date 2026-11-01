// Optimizer race: three optimizers descend a loss surface from the point you
// click; their paths are drawn step by step over the surface.
import { ruai, ready } from "../lib/ruai.mjs";
import { html, sized, animate } from "../lib/ui.mjs";

const RES = 140;
const STEPS = 400;
const NAMES = ["SGD", "SGD + momentum 0.9", "Adam"];
const COLORS = ["#ffffff", "#ff9f43", "#ff4d8d"];
const START = { bowl: [-3.5, 1.5], rosenbrock: [-1.5, 2.5], himmelblau: [-0.5, -4.5], saddle: [-2.0, 0.02] };

// a perceptual dark-to-light ramp for the surface
const RAMP = [[13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137], [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33]];
function ramp(t) {
  const x = Math.min(Math.max(t, 0), 1) * (RAMP.length - 1);
  const i = Math.min(Math.floor(x), RAMP.length - 2), f = x - i;
  return RAMP[i].map((c, k) => Math.round(c + (RAMP[i + 1][k] - c) * f));
}

export async function mount(el) {
  const $ = html(
    el,
    `<div class="page">
    <h1>Optimizer race</h1>
    <p class="lead">Each optimizer is ruai's own (<code>SGD</code>, <code>SGD::with_momentum</code>, <code>Adam</code>), and every gradient comes from autograd on the loss written with tensor operations. <b>Click the surface</b> to start all three from that point.</p>
    <div class="row">
      <div class="card">
        <div class="bar"><span class="label" data-id="status">step 0</span>
          <select data-id="func"><option value="bowl">narrow valley 0.5x² + 10y²</option><option value="rosenbrock">Rosenbrock</option><option value="himmelblau">Himmelblau</option><option value="saddle">saddle x² − y² + y⁴/4</option></select>
          <button data-id="replay">Replay</button></div>
        <div class="body"><div class="canvasbox" data-id="box"><canvas data-id="cv" style="cursor:crosshair"></canvas></div></div>
      </div>
      <div class="card"><div class="body controls">
        <label>learning rate × <span data-id="scalev">1</span><input type="range" data-id="scale" min="-1" max="1" step="0.05" value="0"></label>
        <div data-id="legend"></div>
        <p class="note">Too large a rate makes SGD bounce across the valley or diverge; momentum overshoots and comes back; Adam scales each coordinate's step by its gradient history.</p>
      </div></div>
    </div>
  </div>`,
  );
  await ready;
  let func = "bowl", start = START.bowl, paths = null, dom = null, bg = null, t = 0, stop = null;
  const scale = () => Math.pow(10, +$("scale").value);

  const surface = (size) => {
    const v = ruai.descent_surface(func, RES);
    let hi = 0;
    for (const x of v) hi = Math.max(hi, x);
    const img = new ImageData(RES, RES);
    for (let i = 0; i < v.length; i++) {
      // banded shading reads like contour lines
      const t = v[i] / hi, band = 0.88 + 0.12 * Math.cos(t * 2 * Math.PI * 12);
      const c = ramp(t);
      for (let k = 0; k < 3; k++) img.data[4 * i + k] = c[k] * band;
      img.data[4 * i + 3] = 255;
    }
    const off = new OffscreenCanvas(RES, RES);
    off.getContext("2d").putImageData(img, 0, 0);
    bg = { off, size };
  };

  const toPx = (x, y, size) => [((x - dom[0]) / (dom[1] - dom[0])) * size, ((dom[3] - y) / (dom[3] - dom[2])) * size];

  const paint = () => {
    const size = Math.min($("box").clientWidth || 600, 620);
    if (!bg || bg.size !== size) surface(size);
    const g = sized($("cv"), size, size);
    g.imageSmoothingEnabled = true;
    g.drawImage(bg.off, 0, 0, size, size);
    for (let o = 0; o < 3; o++) {
      g.beginPath();
      for (let s = 0; s <= Math.min(t, STEPS); s++) {
        const i = 2 * (o * (STEPS + 1) + s);
        if (Number.isNaN(paths[i])) break;
        const [px, py] = toPx(paths[i], paths[i + 1], size);
        s === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
      }
      // a dark casing keeps light paths visible on light parts of the surface
      g.strokeStyle = "rgba(0,0,0,.55)";
      g.lineWidth = 4.5;
      g.stroke();
      g.strokeStyle = COLORS[o];
      g.lineWidth = 2;
      g.stroke();
      // the current point
      let s = Math.min(t, STEPS), i = 2 * (o * (STEPS + 1) + s);
      while (s > 0 && Number.isNaN(paths[i])) i = 2 * (o * (STEPS + 1) + --s);
      const [px, py] = toPx(paths[i], paths[i + 1], size);
      g.beginPath();
      g.arc(px, py, 5, 0, 2 * Math.PI);
      g.fillStyle = COLORS[o];
      g.fill();
      g.strokeStyle = "#000";
      g.lineWidth = 1;
      g.stroke();
    }
    $("status").textContent = `step ${Math.min(t, STEPS)} / ${STEPS}`;
  };

  const legend = () => {
    $("legend").innerHTML = NAMES.map((n, o) => {
      const last = (() => {
        for (let s = STEPS; s >= 0; s--) {
          const i = 2 * (o * (STEPS + 1) + s);
          if (!Number.isNaN(paths[i])) return s === STEPS ? `(${paths[i].toFixed(2)}, ${paths[i + 1].toFixed(2)})` : `diverged at step ${s + 1}`;
        }
      })();
      return `<div style="display:flex;gap:8px;align-items:center;margin:4px 0"><span style="width:14px;height:4px;background:${COLORS[o]};outline:1px solid #0006"></span><span style="font-size:13px"><b>${n}</b><br><span class="note" style="margin:0">ends at ${last}</span></span></div>`;
    }).join("");
  };

  const run = () => {
    paths = ruai.descent_paths(func, start[0], start[1], STEPS, scale());
    legend();
    t = 0;
    if (stop) stop();
    stop = animate(() => {
      t += 3;
      paint();
      return t < STEPS;
    });
  };

  const setFunc = () => {
    func = $("func").value;
    dom = Array.from(ruai.descent_domain(func));
    start = START[func];
    bg = null;
    run();
  };

  $("cv").onclick = (e) => {
    const r = $("cv").getBoundingClientRect();
    start = [dom[0] + ((e.clientX - r.left) / r.width) * (dom[1] - dom[0]), dom[3] - ((e.clientY - r.top) / r.height) * (dom[3] - dom[2])];
    run();
  };
  $("func").onchange = setFunc;
  $("replay").onclick = run;
  $("scale").oninput = () => ($("scalev").textContent = scale().toPrecision(2));
  $("scale").onchange = run;
  setFunc();
  return () => stop && stop();
}

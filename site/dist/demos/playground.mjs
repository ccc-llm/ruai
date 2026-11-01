// Classifier playground: an MLP separates two classes of points; the
// background shows P(class) over the plane, redrawn as it trains.
import { ruai, ready } from "../lib/ruai.mjs";
import { html, sized, css, frame } from "../lib/ui.mjs";

const RES = 64;

export async function mount(el) {
  const $ = html(
    el,
    `<div class="page">
    <h1>Classifier playground</h1>
    <p class="lead">A small multilayer perceptron (ruai's <code>nn::Linear</code> layers, autograd and Adam, on the CPU in wasm) learns to tell the orange points from the blue ones. The shading is the network's probability for each class across the plane.</p>
    <div class="row">
      <div class="card">
        <div class="bar"><span class="label">Decision boundary</span>
          <button class="primary" data-id="run">Train</button><button data-id="reset">Reset</button></div>
        <div class="body"><div class="canvasbox" data-id="box"><canvas data-id="cv"></canvas></div></div>
        <div class="body stats" style="padding-top:0">
          <div class="stat"><div class="k">step</div><div class="v" data-id="step">0</div></div>
          <div class="stat"><div class="k">loss</div><div class="v" data-id="loss">–</div></div>
          <div class="stat"><div class="k">accuracy</div><div class="v" data-id="acc">–</div></div>
        </div>
      </div>
      <div class="card"><div class="body controls">
        <label>dataset<select data-id="data"><option value="spiral">two spirals</option><option value="circle">circle</option><option value="xor">xor</option><option value="moons">moons</option><option value="blobs">blobs</option></select></label>
        <label>hidden layers<select data-id="hidden"><option value="8">8</option><option value="16,16" selected>16 · 16</option><option value="32,32">32 · 32</option><option value="32,32,32">32 · 32 · 32</option><option value="4">4</option></select></label>
        <label>activation<select data-id="act"><option>tanh</option><option>relu</option><option>gelu</option><option>sigmoid</option></select></label>
        <label>learning rate<select data-id="lr"><option>0.003</option><option selected>0.01</option><option>0.03</option><option>0.1</option></select></label>
        <label>noise <span data-id="noisev">0.05</span><input type="range" data-id="noise" min="0" max="0.25" step="0.01" value="0.05"></label>
        <label>points<select data-id="n"><option>200</option><option selected>400</option><option>800</option></select></label>
        <label>seed<input type="number" data-id="seed" value="1" min="0"></label>
      </div></div>
    </div>
  </div>`,
  );
  await ready;
  let pts = [];
  let running = false;
  let alive = true;

  const reset = () => {
    const hidden = Uint32Array.from($("hidden").value.split(",").map(Number));
    pts = ruai.pg_reset($("data").value, +$("n").value, +$("noise").value, hidden, $("act").value, +$("lr").value, +$("seed").value);
    $("step").textContent = "0";
    $("loss").textContent = "–";
    $("acc").textContent = "–";
    paint();
  };

  const paint = () => {
    const size = Math.min($("box").clientWidth || 520, 560);
    const g = sized($("cv"), size, size);
    const p = ruai.pg_grid(RES);
    const img = new ImageData(RES, RES);
    const a = [47, 111, 219], b = [232, 116, 59];
    for (let i = 0; i < RES * RES; i++) {
      const t = p[i];
      // light tints of the two class colors, white at 0.5
      const w = Math.abs(t - 0.5) * 2;
      const c = t > 0.5 ? b : a;
      for (let k = 0; k < 3; k++) img.data[4 * i + k] = Math.round(255 + (c[k] - 255) * 0.55 * w);
      img.data[4 * i + 3] = 255;
    }
    const off = new OffscreenCanvas(RES, RES);
    off.getContext("2d").putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true;
    g.drawImage(off, 0, 0, size, size);
    for (let i = 0; i < pts.length; i += 3) {
      const x = ((pts[i] + 1) / 2) * size, y = ((1 - pts[i + 1]) / 2) * size;
      g.beginPath();
      g.arc(x, y, 3.6, 0, 2 * Math.PI);
      g.fillStyle = pts[i + 2] > 0.5 ? "#e8743b" : "#2f6fdb";
      g.fill();
      g.lineWidth = 1;
      g.strokeStyle = "#fff";
      g.stroke();
    }
  };

  const loop = async () => {
    while (running && alive) {
      const [step, loss, acc] = ruai.pg_train(10);
      $("step").textContent = step;
      $("loss").textContent = loss.toFixed(4);
      $("acc").textContent = (acc * 100).toFixed(1) + "%";
      paint();
      if (step >= 3000) running = false;
      await frame();
    }
    $("run").textContent = "Train";
  };

  $("run").onclick = () => {
    running = !running;
    $("run").textContent = running ? "Pause" : "Train";
    if (running) loop();
  };
  $("reset").onclick = () => {
    running = false;
    reset();
  };
  for (const id of ["data", "hidden", "act", "lr", "noise", "n", "seed"])
    $(id).onchange = () => {
      $("noisev").textContent = $("noise").value;
      running = false;
      reset();
    };
  $("noise").oninput = () => ($("noisev").textContent = $("noise").value);
  reset();
  return () => (alive = false);
}

// MNIST with a CNN in the page, in the manner of ConvNetJS's MNIST demo:
// training statistics, the activations of every layer for one digit, the
// predictions on test digits, and a pad to draw a digit of your own.
import { ruai, ready } from "../lib/ruai.mjs";
import { html, sized, css, frame } from "../lib/ui.mjs";

const SHOWN = 60; // test digits in the prediction grid

/** The data file (site/tools/mnist.mjs), gunzipped in the browser. */
async function loadData(onProgress) {
  const r = await fetch(new URL("../data/mnist.bin.gz", import.meta.url));
  if (!r.ok) throw new Error(`could not load the MNIST data (${r.status})`);
  const total = +r.headers.get("content-length") || 0;
  let got = 0;
  const counted = r.body.pipeThrough(
    new TransformStream({
      transform(chunk, c) {
        got += chunk.length;
        if (total) onProgress(got / total);
        c.enqueue(chunk);
      },
    }),
  );
  return new Uint8Array(await new Response(counted.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
}

/** Draw a w × h map of floats as gray pixels (min black, max white), scaled by `s`. */
function grayMap(g, x0, y0, w, h, v, off, s) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < w * h; i++) {
    lo = Math.min(lo, v[off + i]);
    hi = Math.max(hi, v[off + i]);
  }
  const span = hi - lo || 1;
  const img = new ImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const c = Math.round((255 * (v[off + i] - lo)) / span);
    img.data.set([c, c, c, 255], 4 * i);
  }
  const t = new OffscreenCanvas(w, h);
  t.getContext("2d").putImageData(img, 0, 0);
  g.imageSmoothingEnabled = false;
  g.drawImage(t, x0, y0, w * s, h * s);
}

export async function mount(el) {
  const $ = html(
    el,
    `<div class="page">
    <h1>MNIST · a convolutional network</h1>
    <p class="lead">A small CNN learns handwritten digits in your browser, built from ruai's <code>Conv2d</code>, <code>MaxPool2d</code> and <code>Linear</code> layers with autograd and Adam (on the CPU in wasm). It follows <a href="https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html" target="_blank" rel="noopener">ConvNetJS's MNIST demo</a>: conv 5×5×8 → ReLU → pool → conv 5×5×16 → ReLU → pool → softmax over 10 digits, batches of 20 digits, each shifted by up to 2 pixels. The data is 12,000 training and 2,000 test digits of MNIST (2.1 MB).</p>
    <div class="err" data-id="err"></div>
    <div class="card">
      <div class="bar"><span class="label" data-id="status">loading MNIST…</span>
        <button class="primary" data-id="train" disabled>Train</button><button data-id="reset" disabled>Reset</button></div>
      <div class="body stats">
        <div class="stat"><div class="k">examples seen</div><div class="v" data-id="seen">0</div></div>
        <div class="stat"><div class="k">loss</div><div class="v" data-id="loss">–</div></div>
        <div class="stat"><div class="k">training accuracy</div><div class="v" data-id="tacc">–</div></div>
        <div class="stat"><div class="k">test accuracy</div><div class="v" data-id="vacc">–</div></div>
        <div class="stat"><div class="k">ms / example</div><div class="v" data-id="speed">–</div></div>
      </div>
      <div class="body" style="padding-top:0"><div class="canvasbox" data-id="cbox"><canvas data-id="chart"></canvas></div></div>
    </div>
    <div class="card">
      <div class="bar"><span class="label">Inside the network</span><span class="note" style="margin:0" data-id="which"></span><button data-id="next" disabled>Another digit</button></div>
      <div class="body"><div class="canvasbox" data-id="vbox"><canvas data-id="viz"></canvas></div></div>
    </div>
    <div class="row">
      <div class="card">
        <div class="bar"><span class="label">Test digits: the three likeliest classes</span></div>
        <div class="body"><div class="mn-grid" data-id="grid"></div></div>
      </div>
      <div class="card">
        <div class="bar"><span class="label">Draw a digit</span><button data-id="clear">Clear</button></div>
        <div class="body controls">
          <canvas data-id="pad" class="mn-pad" width="280" height="280"></canvas>
          <div data-id="guess" class="mn-guess"></div>
          <p class="note" style="margin:0">Drawn like MNIST: the stroke is scaled to fit 20 × 20 pixels and centered by its mass in 28 × 28. Train first.</p>
        </div>
      </div>
    </div>
  </div>`,
  );
  await ready;
  let bytes;
  try {
    bytes = await loadData((p) => ($("status").textContent = `loading MNIST… ${Math.round(p * 100)}%`));
  } catch (e) {
    $("err").textContent = String(e.message ?? e);
    return;
  }
  const [nTrain, nTest] = ruai.mnist_load(bytes);
  $("status").textContent = `${nTrain.toLocaleString()} training · ${nTest.toLocaleString()} test digits`;
  for (const id of ["train", "reset", "next"]) $(id).disabled = false;

  let running = false, alive = true, history = [], val = [], valAt = 0, shown = 0, drawn = null;

  const chart = () => {
    const w = Math.min($("cbox").clientWidth || 800, 1000), h = 150;
    const g = sized($("chart"), w, h);
    g.font = "11px system-ui";
    g.fillStyle = css("--muted");
    g.fillText("loss (blue) and training accuracy (orange) per batch of 100 examples", 4, 12);
    if (history.length < 2) return;
    const maxLoss = Math.max(2.4, ...history.map((p) => p[1]));
    const X = (i) => 4 + (i / (history.length - 1)) * (w - 8);
    for (const [k, color, f] of [[1, css("--c0"), (v) => v / maxLoss], [2, css("--c1"), (v) => v]]) {
      g.beginPath();
      history.forEach((p, i) => {
        const y = h - 4 - f(p[k]) * (h - 24);
        i ? g.lineTo(X(i), y) : g.moveTo(X(i), y);
      });
      g.strokeStyle = color;
      g.lineWidth = 1.5;
      g.stroke();
    }
  };

  // the test accuracy: a moving window over the test set, 250 digits at a time
  const validate = () => {
    const r = ruai.mnist_test(valAt, 250);
    for (let k = 0; k < r.length; k += 7) val[valAt + k / 7] = r[k] === r[k + 1] ? 1 : 0;
    valAt = (valAt + 250) % nTest;
    const known = val.filter((v) => v !== undefined);
    $("vacc").textContent = ((100 * known.reduce((a, b) => a + b, 0)) / known.length).toFixed(1) + "%";
  };

  const grid = () => {
    const r = ruai.mnist_test(0, SHOWN);
    let h = "";
    for (let k = 0; k < SHOWN; k++) {
      const o = 7 * k, label = r[o];
      h += `<div class="mn-item"><canvas width="28" height="28" data-i="${k}"></canvas><div class="mn-bars">`;
      for (let j = 0; j < 3; j++) {
        const c = r[o + 1 + 2 * j], p = r[o + 2 + 2 * j];
        h += `<div class="mn-bar ${c === label ? "ok" : j === 0 ? "bad" : ""}"><span style="width:${Math.max(2, p * 100)}%"></span><b>${c}</b></div>`;
      }
      h += "</div></div>";
    }
    $("grid").innerHTML = h;
    for (const cv of $("grid").querySelectorAll("canvas")) digit(cv, ruai.mnist_image(true, +cv.dataset.i));
  };

  const digit = (cv, px) => {
    const img = new ImageData(28, 28);
    for (let i = 0; i < 784; i++) img.data.set([px[i], px[i], px[i], 255], 4 * i);
    cv.getContext("2d").putImageData(img, 0, 0);
  };

  // the activations of every layer for one digit
  const SHAPES = [["conv 5×5 + ReLU", 8, 28], ["max pool 2", 8, 14], ["conv 5×5 + ReLU", 16, 14], ["max pool 2", 16, 7]];
  const visualize = (pixels, title) => {
    $("which").textContent = title;
    const act = ruai.mnist_layers(Float32Array.from(pixels));
    const filters = ruai.mnist_filters();
    const w = Math.min($("vbox").clientWidth || 900, 1000);
    const cols = [{ name: "input", maps: 1, size: 28, data: pixels, off: 0 }];
    let off = 0;
    for (const [name, maps, size] of SHAPES) {
      cols.push({ name, maps, size, data: act, off });
      off += maps * size * size;
    }
    const probs = act.slice(off, off + 10);
    // layout: one column per layer, maps stacked; probabilities at the right
    const colW = (w - 150) / cols.length;
    const perRow = (c) => (c.maps >= 16 ? 4 : c.maps > 1 ? 2 : 1);
    const scaleOf = (c) => Math.max(1, Math.min(Math.floor((colW - 12 - 3 * perRow(c)) / (perRow(c) * c.size)), c.maps > 1 ? 3 : 4));
    const rows = (c) => Math.ceil(c.maps / perRow(c));
    const height = Math.max(...cols.map((c) => rows(c) * (c.size * scaleOf(c) + 3))) + 40;
    const g = sized($("viz"), w, Math.max(height, 230));
    g.font = "11px system-ui";
    cols.forEach((c, k) => {
      const s = scaleOf(c), per = perRow(c), x0 = k * colW + 6;
      g.fillStyle = css("--muted");
      g.fillText(c.name, x0, 12);
      if (c.maps > 1) g.fillText(`${c.maps} × ${c.size}×${c.size}`, x0, 25);
      for (let m = 0; m < c.maps; m++) {
        const x = x0 + (m % per) * (c.size * s + 3), y = 32 + Math.floor(m / per) * (c.size * s + 3);
        grayMap(g, x, y, c.size, c.size, c.data, c.off + m * c.size * c.size, s);
      }
    });
    // conv1's filters under the input
    g.fillStyle = css("--muted");
    g.fillText("conv1 filters", 6, 32 + 28 * scaleOf(cols[0]) + 18);
    for (let f = 0; f < 8; f++) grayMap(g, 6 + (f % 4) * 22, 32 + 28 * scaleOf(cols[0]) + 26 + Math.floor(f / 4) * 22, 5, 5, filters, f * 25, 4);
    const px = w - 140;
    g.fillStyle = css("--muted");
    g.fillText("softmax", px, 12);
    const best = probs.indexOf(Math.max(...probs));
    for (let d = 0; d < 10; d++) {
      const y = 22 + d * 19;
      g.fillStyle = css("--text");
      g.fillText(d, px, y + 11);
      g.fillStyle = d === best ? css("--c1") : css("--c0");
      g.fillRect(px + 14, y + 2, Math.max(1, probs[d] * 116), 12);
    }
  };

  const showTest = () => {
    const i = shown % nTest;
    const px = Array.from(ruai.mnist_image(true, i), (v) => v / 255);
    visualize(px, `test digit #${i}`);
  };

  const guess = () => {
    if (!drawn) return;
    const act = ruai.mnist_layers(Float32Array.from(drawn));
    const probs = Array.from(act.slice(-10));
    const order = probs.map((p, d) => [p, d]).sort((a, b) => b[0] - a[0]);
    $("guess").innerHTML = `<span class="big">${order[0][1]}</span> ` + order.slice(0, 3).map(([p, d]) => `${d}: ${(p * 100).toFixed(1)}%`).join(" · ");
  };

  const reset = () => {
    running = false;
    ruai.mnist_reset(1);
    history = [];
    val = [];
    valAt = 0;
    for (const id of ["loss", "tacc", "vacc", "speed"]) $(id).textContent = "–";
    $("seen").textContent = "0";
    chart();
    grid();
    showTest();
    guess();
  };

  const loop = async () => {
    let n = 0;
    while (running && alive) {
      const [seen, loss, acc, ms] = ruai.mnist_train(5);
      history.push([seen, loss, acc]);
      if (history.length > 600) history = history.filter((_, i) => i % 2 === 0);
      $("seen").textContent = seen.toLocaleString();
      $("loss").textContent = loss.toFixed(3);
      $("tacc").textContent = (acc * 100).toFixed(0) + "%";
      $("speed").textContent = ms.toFixed(2);
      chart();
      if (++n % 8 === 0) validate();
      if (n % 30 === 0) {
        grid();
        shown++;
        showTest();
        guess();
      }
      await frame();
    }
    $("train").textContent = "Train";
  };

  $("train").onclick = () => {
    running = !running;
    $("train").textContent = running ? "Pause" : "Train";
    if (running) loop();
  };
  $("reset").onclick = reset;
  $("next").onclick = () => {
    shown++;
    showTest();
  };

  // the drawing pad: strokes on 280 × 280, turned into an MNIST-style 28 × 28 digit
  const pad = $("pad"), pg = pad.getContext("2d");
  const clearPad = () => {
    pg.fillStyle = "#000";
    pg.fillRect(0, 0, 280, 280);
    drawn = null;
    $("guess").textContent = "";
  };
  clearPad();
  let last = null;
  const at = (e) => {
    const r = pad.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * 280, ((e.clientY - r.top) / r.height) * 280];
  };
  pad.onpointerdown = (e) => {
    pad.setPointerCapture(e.pointerId);
    last = at(e);
  };
  pad.onpointermove = (e) => {
    if (!last) return;
    const p = at(e);
    pg.strokeStyle = "#fff";
    pg.lineWidth = 20;
    pg.lineCap = "round";
    pg.beginPath();
    pg.moveTo(...last);
    pg.lineTo(...p);
    pg.stroke();
    last = p;
  };
  pad.onpointerup = () => {
    last = null;
    drawn = toMnist(pg);
    if (drawn) {
      guess();
      visualize(drawn, "your digit");
    }
  };
  $("clear").onclick = clearPad;

  reset();
  return () => (alive = false);
}

/** The pad's strokes as MNIST does it: the bounding box scaled to 20 × 20, centered by mass in 28 × 28. */
function toMnist(pg) {
  const src = pg.getImageData(0, 0, 280, 280).data;
  let x0 = 280, y0 = 280, x1 = -1, y1 = -1;
  for (let y = 0; y < 280; y++)
    for (let x = 0; x < 280; x++)
      if (src[4 * (y * 280 + x)] > 20) {
        x0 = Math.min(x0, x);
        x1 = Math.max(x1, x);
        y0 = Math.min(y0, y);
        y1 = Math.max(y1, y);
      }
  if (x1 < 0) return null;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1, s = 20 / Math.max(bw, bh);
  const box = new OffscreenCanvas(28, 28), bg = box.getContext("2d");
  bg.fillStyle = "#000";
  bg.fillRect(0, 0, 28, 28);
  bg.imageSmoothingEnabled = true;
  bg.imageSmoothingQuality = "high";
  bg.drawImage(pg.canvas, x0, y0, bw, bh, 0, 0, bw * s, bh * s);
  const small = bg.getImageData(0, 0, 28, 28).data;
  // center of mass of the scaled stroke, moved to the middle
  let m = 0, cx = 0, cy = 0;
  for (let y = 0; y < 28; y++)
    for (let x = 0; x < 28; x++) {
      const v = small[4 * (y * 28 + x)];
      m += v;
      cx += v * x;
      cy += v * y;
    }
  const dx = Math.round(14 - cx / m), dy = Math.round(14 - cy / m);
  const out = new Array(784).fill(0);
  for (let y = 0; y < 28; y++)
    for (let x = 0; x < 28; x++) {
      const sx = x - dx, sy = y - dy;
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) out[y * 28 + x] = small[4 * (sy * 28 + sx)] / 255;
    }
  return out;
}

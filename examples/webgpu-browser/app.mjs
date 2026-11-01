// The page of the WebGPU training demo: drives the wasm module (pkg/).
import init, * as ruai from "./pkg/ruai_webgpu_demo.js";

const $ = (id) => document.getElementById(id);
const fail = (e) => ($("err").textContent = String(e && e.message ? e.message : e));

let steps = [], losses = [], running = false;
let draw = null;

function plot() {
  const box = $("plotbox");
  const w = Math.max(320, Math.min(box.clientWidth || 900, 900));
  const fig = JSON.parse(ruai.loss_plot(Float64Array.from(steps), Float64Array.from(losses), w, w * 0.45));
  draw($("plot"), fig);
}

function resetModel() {
  steps = [];
  losses = [];
  $("params").textContent = ruai.reset(0).toLocaleString();
  $("step").textContent = "0";
  $("loss").textContent = "–";
  $("speed").textContent = "–";
  plot();
}

async function trainLoop() {
  const total = ruai.total_steps();
  while (running) {
    // 10 steps queued on the GPU, one loss read back
    const [step, loss, ms] = await ruai.train(10);
    if (!Number.isNaN(loss)) {
      steps.push(step);
      losses.push(loss);
      $("step").textContent = `${step} / ${total}`;
      $("loss").textContent = loss.toFixed(4);
      $("speed").textContent = (ms / 10).toFixed(1);
      plot();
    }
    if (step >= total) {
      running = false;
    }
    await new Promise(requestAnimationFrame);
  }
  $("train").textContent = "Train";
}

try {
  await init();
  draw = new Function(ruai.canvas_js() + "\nreturn ruaiDraw;")();
  // browsers may hide the adapter's name
  const name = (await ruai.start()).replace(/\s*\(BrowserWebGpu\)$/, "").trim();
  $("adapter").textContent = name ? `WebGPU: ${name}` : "WebGPU";
  resetModel();
  for (const id of ["train", "reset", "gen", "bench"]) $(id).disabled = false;
} catch (e) {
  $("adapter").textContent = "WebGPU unavailable";
  fail(e);
}

$("train").onclick = () => {
  running = !running;
  $("train").textContent = running ? "Pause" : "Train";
  if (running) trainLoop().catch(fail);
};
$("reset").onclick = () => {
  running = false;
  resetModel();
};
$("gen").onclick = async () => {
  $("gen").disabled = true;
  try {
    $("out").textContent = await ruai.generate($("prompt").value, 120, false);
  } catch (e) {
    fail(e);
  }
  $("gen").disabled = false;
};
$("bench").onclick = async () => {
  $("bench").disabled = true;
  $("benchout").textContent = "measuring…";
  try {
    const g = await ruai.bench_matmul(1024, 20);
    $("benchout").textContent = `${g.toFixed(0)} GFLOP/s (20 products, timed in the page)`;
  } catch (e) {
    fail(e);
  }
  $("bench").disabled = false;
};

// A character-level GPT trained in the page on WebGPU (or the CPU), with a
// loss chart, text generation and a matmul benchmark.
import { ruai, ready, draw } from "../lib/ruai.mjs";
import { html, frame } from "../lib/ui.mjs";

export async function mount(el) {
  const $ = html(
    el,
    `<div class="page">
    <h1>GPT on WebGPU</h1>
    <p class="lead">A 2-layer, 4-head GPT (ruai's <code>nn::GPT</code>) learns a paragraph of text one character at a time. On WebGPU every forward pass, backward pass and AdamW step runs as WGSL compute shaders on your GPU, and the page reads back one number per 10 steps. Without WebGPU it trains on the CPU in wasm, much more slowly.</p>
    <div class="card">
      <div class="bar"><span class="label" data-id="dev"></span>
        <button class="primary" data-id="train">Train</button><button data-id="reset">Reset</button></div>
      <div class="body stats">
        <div class="stat"><div class="k">step</div><div class="v" data-id="step">–</div></div>
        <div class="stat"><div class="k">loss</div><div class="v" data-id="loss">–</div></div>
        <div class="stat"><div class="k">ms / step</div><div class="v" data-id="speed">–</div></div>
        <div class="stat"><div class="k">parameters</div><div class="v" data-id="params">–</div></div>
      </div>
      <div class="plotbox" data-id="box"><canvas data-id="plot"></canvas></div>
    </div>
    <div class="card">
      <div class="bar"><span class="label">Generate</span><input type="text" data-id="prompt" value="attention " style="flex:1;min-width:120px;font-family:var(--mono)"><button data-id="gen">Complete</button></div>
      <pre class="out" data-id="out"></pre>
    </div>
    <div class="card">
      <div class="bar"><span class="label">matmul benchmark</span><select data-id="n"><option>256</option><option>512</option><option selected>1024</option></select><button data-id="bench">Measure</button></div>
      <pre class="out" data-id="benchout"></pre>
    </div>
  </div>`,
  );
  $("dev").textContent = "starting…";
  const device = await ready;
  const gpu = ruai.has_webgpu();
  $("dev").textContent = device;
  let steps = [], losses = [], running = false, alive = true;

  const plot = () => {
    const w = Math.max(320, Math.min($("box").clientWidth || 900, 1000));
    draw($("plot"), ruai.gpt_loss_plot(Float64Array.from(steps), Float64Array.from(losses), w, w * 0.4));
  };

  const reset = () => {
    running = false;
    steps = [];
    losses = [];
    $("params").textContent = ruai.gpt_reset(0).toLocaleString();
    $("step").textContent = "0";
    $("loss").textContent = "–";
    $("speed").textContent = "–";
    plot();
  };

  const loop = async () => {
    const total = ruai.gpt_total_steps();
    const k = gpu ? 10 : 2;
    while (running && alive) {
      const [step, loss, ms] = await ruai.gpt_train(k);
      if (!alive) return;
      if (!Number.isNaN(loss)) {
        steps.push(step);
        losses.push(loss);
        $("step").textContent = `${step} / ${total}`;
        $("loss").textContent = loss.toFixed(4);
        $("speed").textContent = (ms / k).toFixed(1);
        plot();
      }
      if (step >= total) running = false;
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
  $("gen").onclick = async () => {
    $("gen").disabled = true;
    $("out").textContent = await ruai.gpt_generate($("prompt").value, 120, false);
    $("gen").disabled = false;
  };
  $("bench").onclick = async () => {
    $("bench").disabled = true;
    const n = +$("n").value;
    $("benchout").textContent = "measuring…";
    const lines = [];
    if (gpu) lines.push(`WebGPU: ${(await ruai.bench_matmul(n, 20, true)).toFixed(0)} GFLOP/s`);
    $("benchout").textContent = lines.join("\n") + (lines.length ? "\n" : "") + "CPU (wasm, one thread): measuring…";
    await frame();
    lines.push(`CPU (wasm, one thread): ${(await ruai.bench_matmul(n, 3, false)).toFixed(1)} GFLOP/s`);
    $("benchout").textContent = lines.join("\n") + `\n${n} × ${n} × ${n}, timed in this page`;
    $("bench").disabled = false;
  };
  reset();
  return () => (alive = false);
}

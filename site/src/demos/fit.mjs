// Curve fitting: an MLP regresses a 1-D function; the figure (fit and loss)
// is drawn by ruai.plot.
import { ruai, ready, draw } from "../lib/ruai.mjs";
import { html, frame } from "../lib/ui.mjs";

export async function mount(el) {
  const $ = html(
    el,
    `<div class="page">
    <h1>Curve fitting</h1>
    <p class="lead">A 1 → 64 → 64 → 1 network learns a function from noisy samples by minimizing the mean squared error with Adam. Both plots are made by <code>ruai::plot</code> (matplotlib's API) and drawn on a canvas.</p>
    <div class="card">
      <div class="bar"><span class="label" data-id="status">step 0</span>
        <select data-id="func"><option value="sin">1.5 sin x</option><option value="abs">|x|</option><option value="step">step</option><option value="sinc">sinc</option><option value="cubic">cubic</option></select>
        <select data-id="act"><option>tanh</option><option>relu</option><option>gelu</option></select>
        <select data-id="noise"><option value="0">no noise</option><option value="0.1" selected>noise 0.1</option><option value="0.3">noise 0.3</option></select>
        <button class="primary" data-id="run">Train</button><button data-id="reset">Reset</button></div>
      <div class="plotbox" data-id="box"><canvas data-id="cv"></canvas></div>
    </div>
  </div>`,
  );
  await ready;
  let running = false, alive = true;
  const paint = () => {
    const w = Math.max(320, Math.min($("box").clientWidth || 980, 1000));
    draw($("cv"), ruai.fit_plot(w, Math.max(260, w * 0.42)));
  };
  const reset = () => {
    running = false;
    ruai.fit_reset($("func").value, 80, +$("noise").value, $("act").value, 64, 3);
    $("status").textContent = "step 0";
    paint();
  };
  const loop = async () => {
    while (running && alive) {
      const [step, loss] = ruai.fit_train(25);
      $("status").textContent = `step ${step} · MSE ${loss.toFixed(4)}`;
      paint();
      if (step >= 4000) running = false;
      await frame();
    }
    $("run").textContent = "Train";
  };
  $("run").onclick = () => {
    running = !running;
    $("run").textContent = running ? "Pause" : "Train";
    if (running) loop();
  };
  $("reset").onclick = reset;
  for (const id of ["func", "act", "noise"]) $(id).onchange = reset;
  reset();
  return () => (alive = false);
}

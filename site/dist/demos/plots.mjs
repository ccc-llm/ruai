// The ruai.plot gallery: figures built in Rust with matplotlib's API and
// drawn by ruai's canvas renderer.
import { ruai, ready, draw } from "../lib/ruai.mjs";
import { html } from "../lib/ui.mjs";

export async function mount(el) {
  const $ = html(
    el,
    `<div class="page">
    <h1>ruai.plot gallery</h1>
    <p class="lead"><code>ruai::plot</code> implements matplotlib's <code>pyplot</code> API (figures, axes, lines, scatters, bars, histograms, images and subplots) and renders to a canvas or SVG. py0 programs use it as <code>import ruai.plot as plt</code>. Each figure below is built in Rust and drawn by the same canvas renderer.</p>
    <div class="card">
      <div class="bar"><span class="label">Figure</span><select data-id="pick"></select><button data-id="again">New samples</button></div>
      <div class="plotbox" data-id="box"><canvas data-id="cv"></canvas></div>
    </div>
  </div>`,
  );
  await ready;
  for (const n of ruai.gallery_names()) $("pick").add(new Option(n, n));
  let seed = 1;
  const paint = () => {
    const w = Math.max(320, Math.min($("box").clientWidth || 980, 1000));
    draw($("cv"), ruai.gallery($("pick").value, w, Math.max(300, w * 0.55), seed));
  };
  $("pick").onchange = paint;
  $("again").onclick = () => {
    seed++;
    paint();
  };
  const onResize = () => paint();
  addEventListener("resize", onResize);
  paint();
  return () => removeEventListener("resize", onResize);
}

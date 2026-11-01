// Q-learning on MountainCar: the value of each (position, velocity) cell as
// a heat map, and the greedy policy driving the car.
import { ruai, ready } from "../lib/ruai.mjs";
import { html, sized, frame, animate, css } from "../lib/ui.mjs";
import { drawMountainCar } from "../lib/envs.mjs";

export async function mount(el) {
  const $ = html(
    el,
    `<div class="page">
    <h1>MountainCar · Q-learning</h1>
    <p class="lead">The car's engine is too weak to drive straight up the hill: it has to rock back and forth to build momentum. Tabular Q-learning cuts the (position, velocity) plane into a grid and learns the value of each action in each cell from −1 per step. The heat map shows the best value per cell, and the learned spiral appears as training goes on.</p>
    <div class="row">
      <div class="card">
        <div class="bar"><span class="label">Environment (<code>MountainCar-v0</code>)</span>
          <button class="primary" data-id="train">Train</button><button data-id="play">Watch</button><button data-id="reset">Reset</button></div>
        <div class="body"><div class="canvasbox" data-id="box"><canvas data-id="cv"></canvas></div></div>
        <div class="body stats" style="padding-top:0">
          <div class="stat"><div class="k">episodes</div><div class="v" data-id="ep">0</div></div>
          <div class="stat"><div class="k">steps (last)</div><div class="v" data-id="last">–</div></div>
          <div class="stat"><div class="k">mean of last 50</div><div class="v" data-id="mean">–</div></div>
        </div>
      </div>
      <div class="card"><div class="bar"><span class="label">max<sub>a</sub> Q(position, velocity)</span></div>
        <div class="body controls"><canvas data-id="heat"></canvas><p class="note" style="margin:0">x: position −1.2 … 0.6 · y: velocity −0.07 … 0.07. The dot is the car during "Watch". 200 steps means it never reached the flag.</p>
          <label>grid<select data-id="bins"><option>12</option><option>16</option><option selected>20</option><option>28</option></select></label>
          <label>step size α<select data-id="alpha"><option>0.1</option><option selected>0.3</option><option>0.5</option></select></label>
          <label>exploration ε (decays)<select data-id="eps"><option value="0" selected>0 (optimism only)</option><option>0.1</option><option>0.3</option></select></label>
        </div></div>
    </div>
  </div>`,
  );
  await ready;
  let training = false, alive = true, stopPlay = null, dot = null;
  const W = () => Math.min($("box").clientWidth || 600, 720);

  const heat = () => {
    const v = ruai.mc_values();
    const n = v[0], size = 248;
    const g = sized($("heat"), size, size);
    let lo = 0;
    for (let i = 1; i < v.length; i++) lo = Math.min(lo, v[i]);
    const c = size / n;
    for (let r = 0; r < n; r++)
      for (let k = 0; k < n; k++) {
        const t = lo < 0 ? 1 - v[1 + r * n + k] / lo : 1;
        g.fillStyle = `hsl(${220 - 190 * t}, 70%, ${30 + 35 * t}%)`;
        g.fillRect(k * c, size - (r + 1) * c, c + 0.5, c + 0.5);
      }
    if (dot) {
      g.fillStyle = "#fff";
      g.strokeStyle = "#000";
      g.beginPath();
      g.arc(((dot[0] + 1.2) / 1.8) * size, size - ((dot[1] + 0.07) / 0.14) * size, 4, 0, 2 * Math.PI);
      g.fill();
      g.stroke();
    }
  };

  const reset = () => {
    training = false;
    stopPlay?.();
    dot = null;
    ruai.mc_reset(1, +$("bins").value, +$("alpha").value, +$("eps").value);
    $("ep").textContent = "0";
    $("last").textContent = "–";
    $("mean").textContent = "–";
    drawMountainCar(sized($("cv"), W(), 260), W(), 260, -0.5);
    heat();
  };

  const loop = async () => {
    while (training && alive) {
      const [ep, last, mean] = ruai.mc_train(40);
      $("ep").textContent = ep;
      $("last").textContent = last;
      $("mean").textContent = mean.toFixed(0);
      heat();
      if (ep >= 6000) training = false;
      await frame();
    }
    $("train").textContent = "Train";
  };

  const play = () => {
    stopPlay?.();
    const s = ruai.mc_play();
    const n = s.length / 2;
    let i = 0;
    stopPlay = animate(() => {
      dot = [s[2 * i], s[2 * i + 1]];
      drawMountainCar(sized($("cv"), W(), 260), W(), 260, s[2 * i]);
      heat();
      i++;
      if (i >= n) {
        dot = null;
        return false;
      }
    });
  };

  $("train").onclick = () => {
    training = !training;
    $("train").textContent = training ? "Pause" : "Train";
    if (training) {
      stopPlay?.();
      loop();
    }
  };
  $("play").onclick = play;
  $("reset").onclick = reset;
  for (const id of ["bins", "alpha", "eps"]) $(id).onchange = reset;
  reset();
  return () => {
    alive = false;
    stopPlay?.();
  };
}

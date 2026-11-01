// REINFORCE on CartPole: train the policy network, chart the returns, and
// watch the greedy policy balance the pole.
import { ruai, ready } from "../lib/ruai.mjs";
import { html, sized, frame, animate, css } from "../lib/ui.mjs";
import { drawCartPole } from "../lib/envs.mjs";

export async function mount(el) {
  const $ = html(
    el,
    `<div class="page">
    <h1>CartPole · REINFORCE</h1>
    <p class="lead">A policy network (4 → 64 → 2, ruai's <code>nn</code> on the CPU) picks <i>push left</i> or <i>push right</i> from the cart's position and velocity and the pole's angle and angular velocity. After each episode the policy gradient raises the log-probability of actions in proportion to the return that followed them. 500 steps is a perfect episode.</p>
    <div class="card">
      <div class="bar"><span class="label">Environment (<code>ruai::world::CartPole</code>)</span>
        <button class="primary" data-id="train">Train</button><button data-id="play">Watch the policy</button><button data-id="reset">Reset</button></div>
      <div class="body"><div class="canvasbox" data-id="box"><canvas data-id="cv"></canvas></div></div>
      <div class="body stats" style="padding-top:0">
        <div class="stat"><div class="k">episodes</div><div class="v" data-id="ep">0</div></div>
        <div class="stat"><div class="k">last return</div><div class="v" data-id="ret">–</div></div>
        <div class="stat"><div class="k">mean of last 20</div><div class="v" data-id="mean">–</div></div>
        <div class="stat"><div class="k">watching</div><div class="v" data-id="watch">–</div></div>
      </div>
    </div>
    <div class="card"><div class="bar"><span class="label">Return per episode</span></div><div class="body"><div class="canvasbox" data-id="cbox"><canvas data-id="chart"></canvas></div></div></div>
  </div>`,
  );
  await ready;
  let training = false, alive = true, stopPlay = null;
  const W = () => Math.min($("box").clientWidth || 700, 980);

  const chart = () => {
    const r = ruai.cp_returns();
    const w = Math.min($("cbox").clientWidth || 700, 980), h = 180;
    const g = sized($("chart"), w, h);
    const n = Math.max(r.length, 50);
    g.strokeStyle = css("--line");
    g.lineWidth = 1;
    for (const v of [0, 250, 500]) {
      const y = h - 14 - (v / 500) * (h - 28);
      g.beginPath();
      g.moveTo(30, y);
      g.lineTo(w, y);
      g.stroke();
      g.fillStyle = css("--muted");
      g.font = "11px system-ui";
      g.fillText(v, 2, y + 4);
    }
    g.fillStyle = css("--accent");
    for (let i = 0; i < r.length; i++) {
      const x = 30 + (i / n) * (w - 34), y = h - 14 - (r[i] / 500) * (h - 28);
      g.fillRect(x, y, Math.max(1.5, (w - 34) / n - 0.5), h - 14 - y);
    }
  };

  const still = () => drawCartPole(sized($("cv"), W(), 220), W(), 220, 0, 0);

  const reset = () => {
    training = false;
    stopPlay?.();
    ruai.cp_reset(1);
    $("ep").textContent = "0";
    $("ret").textContent = "–";
    $("mean").textContent = "–";
    $("watch").textContent = "–";
    still();
    chart();
  };

  const loop = async () => {
    while (training && alive) {
      const [ep, last, mean] = ruai.cp_train(2);
      $("ep").textContent = ep;
      $("ret").textContent = last;
      $("mean").textContent = mean.toFixed(0);
      chart();
      if (mean >= 475) {
        training = false;
        play();
      }
      await frame();
    }
    $("train").textContent = "Train";
  };

  const play = () => {
    stopPlay?.();
    const s = ruai.cp_play();
    const n = s.length / 2;
    let i = 0;
    stopPlay = animate(() => {
      const w = W();
      drawCartPole(sized($("cv"), w, 220), w, 220, s[2 * i], s[2 * i + 1]);
      $("watch").textContent = `${i + 1} / ${n}`;
      i++;
      return i < n;
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
  reset();
  return () => {
    alive = false;
    stopPlay?.();
  };
}

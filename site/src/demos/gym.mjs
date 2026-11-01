// Play the classic-control environments with the keyboard (or let a random
// agent play), stepping ruai::world from the page.
import { ruai, ready } from "../lib/ruai.mjs";
import { html, sized, animate } from "../lib/ui.mjs";
import { drawCartPole, drawMountainCar, drawAcrobot, drawPendulum } from "../lib/envs.mjs";

const ENVS = {
  "CartPole-v1": { keys: "← / → push the cart", fps: 50, act: (k) => (k.right ? 1 : k.left ? 0 : null), random: () => (Math.random() < 0.5 ? 0 : 1), draw: (g, w, h, o) => drawCartPole(g, w, h, o[0], o[2]) },
  "MountainCar-v0": { keys: "← / → accelerate (nothing pressed: coast)", fps: 30, act: (k) => (k.right ? 2 : k.left ? 0 : 1), random: () => Math.floor(Math.random() * 3), draw: (g, w, h, o) => drawMountainCar(g, w, h, o[0]) },
  "Acrobot-v1": { keys: "← / → torque the joint, swing the tip above the dashed line", fps: 15, act: (k) => (k.right ? 2 : k.left ? 0 : 1), random: () => Math.floor(Math.random() * 3), draw: (g, w, h, o) => drawAcrobot(g, w, h, Math.atan2(o[1], o[0]), Math.atan2(o[3], o[2])) },
  "Pendulum-v1": { keys: "← / → torque, swing it up and hold it upright", fps: 20, act: (k) => (k.right ? -2 : k.left ? 2 : 0), random: () => Math.random() * 4 - 2, draw: (g, w, h, o, a) => drawPendulum(g, w, h, Math.atan2(o[1], o[0]), a) },
};

export async function mount(el) {
  const $ = html(
    el,
    `<div class="page">
    <h1>Play the environments</h1>
    <p class="lead">These are ruai's Gymnasium environments (<code>ruai::world::make</code>), which reproduce Gymnasium's dynamics and random numbers exactly. Each frame the page calls <code>env.step(action)</code> in wasm and draws the observation.</p>
    <div class="card">
      <div class="bar"><span class="label" data-id="space"></span>
        <select data-id="env">${Object.keys(ENVS).map((k) => `<option>${k}</option>`).join("")}</select>
        <button class="primary" data-id="start">Play</button><button data-id="random">Random agent</button></div>
      <div class="body"><div class="canvasbox" data-id="box" tabindex="0" style="outline:none"><canvas data-id="cv"></canvas></div>
        <p class="note" data-id="keys"></p></div>
      <div class="body stats" style="padding-top:0">
        <div class="stat"><div class="k">steps</div><div class="v" data-id="steps">0</div></div>
        <div class="stat"><div class="k">return</div><div class="v" data-id="ret">0</div></div>
        <div class="stat"><div class="k">state</div><div class="v" data-id="state">–</div></div>
      </div>
    </div>
  </div>`,
  );
  await ready;
  const keys = { left: false, right: false };
  const onKey = (down) => (e) => {
    if (e.key === "ArrowLeft") keys.left = down;
    else if (e.key === "ArrowRight") keys.right = down;
    else return;
    e.preventDefault();
  };
  const kd = onKey(true), ku = onKey(false);
  addEventListener("keydown", kd);
  addEventListener("keyup", ku);
  let stop = null, seed = 0;
  const W = () => Math.min($("box").clientWidth || 700, 900), H = 300;

  const choose = () => {
    stop?.();
    const id = $("env").value;
    $("space").textContent = ruai.gym_make(id);
    $("keys").innerHTML = "Keys: " + ENVS[id].keys.replace(/←|→/g, (k) => `<kbd>${k}</kbd>`);
    const o = ruai.gym_reset(seed);
    ENVS[id].draw(sized($("cv"), W(), H), W(), H, o, 0);
    $("steps").textContent = "0";
    $("ret").textContent = "0";
    $("state").textContent = "ready";
  };

  const run = (random) => {
    stop?.();
    const id = $("env").value, env = ENVS[id];
    ruai.gym_reset(++seed);
    $("box").focus();
    let steps = 0, ret = 0, last = 0, acc = 0;
    stop = animate(() => {
      const now = performance.now();
      if (last && now - last + acc < 1000 / env.fps) return true;
      acc = last ? Math.min(now - last + acc - 1000 / env.fps, 50) : 0;
      last = now;
      let a = random ? env.random() : env.act(keys);
      if (a === null) a = steps % 2; // CartPole has no "nothing": alternate
      const r = ruai.gym_step(a);
      steps++;
      ret += r[0];
      env.draw(sized($("cv"), W(), H), W(), H, r.slice(3), a);
      $("steps").textContent = steps;
      $("ret").textContent = Number.isInteger(ret) ? ret : ret.toFixed(1);
      if (r[1] || r[2]) {
        $("state").textContent = r[1] ? "terminated" : "time limit";
        return false;
      }
      $("state").textContent = "running";
    });
  };

  $("env").onchange = choose;
  $("start").onclick = () => run(false);
  $("random").onclick = () => run(true);
  choose();
  return () => {
    stop?.();
    removeEventListener("keydown", kd);
    removeEventListener("keyup", ku);
  };
}

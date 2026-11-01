// FrozenLake: value iteration on a lake you can redraw, then walks of the
// greedy policy on the slippery ice.
import { ruai, ready } from "../lib/ruai.mjs";
import { html, sized, css, animate } from "../lib/ui.mjs";

const MAPS = {
  "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
  "8x8": ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"],
};
const ARROWS = ["←", "↓", "→", "↑"];

export async function mount(el) {
  const $ = html(
    el,
    `<div class="page">
    <h1>FrozenLake · value iteration</h1>
    <p class="lead">Walk from S to the goal G without falling into a hole. On slippery ice the intended move happens only a third of the time, and each sideways slip happens a third of the time. Value iteration sweeps the Bellman update over the lake's model (<code>FrozenLake::transitions</code>) until the values settle, and the arrows are the greedy policy. <b>Click a tile</b> to toggle a hole.</p>
    <div class="row">
      <div class="card">
        <div class="bar"><span class="label" data-id="status">sweep 0</span>
          <button class="primary" data-id="solve">Solve</button><button data-id="sweep">One sweep</button><button data-id="walk">Walk</button></div>
        <div class="body"><canvas data-id="cv" style="cursor:pointer"></canvas></div>
      </div>
      <div class="card"><div class="body controls">
        <label>map<select data-id="map"><option>4x4</option><option selected>8x8</option></select></label>
        <label style="display:flex;gap:6px;align-items:center;color:var(--text)"><input type="checkbox" data-id="slip" checked> slippery</label>
        <div class="stats"><div class="stat"><div class="k">walks</div><div class="v" data-id="walks">0</div></div><div class="stat"><div class="k">reached G</div><div class="v" data-id="wins">–</div></div></div>
        <p class="note">γ = 0.95. Near holes the best move often points away from the goal: slipping sideways is safer than walking along the edge of a hole.</p>
      </div></div>
    </div>
  </div>`,
  );
  await ready;
  let rows = MAPS["8x8"].slice(), vals = null, pol = null, path = null, stop = null, walks = 0, wins = 0;
  const n = () => rows.length;
  const size = () => Math.min(el.querySelector(".card .body").clientWidth - 0 || 480, 520);

  const fresh = () => {
    stop?.();
    path = null;
    ruai.lake_new(rows.join("\n"), $("slip").checked, 7);
    vals = null;
    pol = null;
    walks = wins = 0;
    $("walks").textContent = "0";
    $("wins").textContent = "–";
    $("status").textContent = "sweep 0";
    paint();
  };

  const sweep = () => {
    const r = ruai.lake_sweep();
    const N = n() * n();
    vals = r.slice(2, 2 + N);
    pol = r.slice(2 + N);
    $("status").textContent = `sweep ${r[1]} · largest change ${r[0].toExponential(2)}`;
    return r[0];
  };

  const paint = (at) => {
    const S = size(), c = S / n();
    const g = sized($("cv"), S, S);
    let hi = 1e-9;
    if (vals) for (const v of vals) hi = Math.max(hi, v);
    for (let r = 0; r < n(); r++)
      for (let k = 0; k < n(); k++) {
        const ch = rows[r][k], s = r * n() + k;
        let fill = ch === "H" ? "#20324d" : ch === "G" ? "#f2b84b" : "#dcecf7";
        let light = 90;
        if (vals && (ch === "F" || ch === "S")) {
          light = (ch === "F" ? 92 : 88) - 50 * (vals[s] / hi);
          fill = `hsl(${ch === "F" ? 205 : 150}, 60%, ${light}%)`;
        }
        if (!vals && ch === "S") fill = "#bfe3cd";
        g.fillStyle = fill;
        g.fillRect(k * c + 1, r * c + 1, c - 2, c - 2);
        g.fillStyle = ch === "H" ? "#9fb5d6" : light < 58 ? "#ffffff" : "#1c1c1a";
        g.textAlign = "center";
        g.textBaseline = "middle";
        if (ch === "H" || ch === "G") {
          g.font = `600 ${c * 0.3}px system-ui`;
          g.fillText(ch === "H" ? "hole" : "G", k * c + c / 2, r * c + c / 2);
        } else if (pol) {
          g.font = `${c * 0.42}px system-ui`;
          g.fillText(ARROWS[pol[s]], k * c + c / 2, r * c + c * 0.42);
          g.font = `${c * 0.18}px ui-monospace, monospace`;
          g.fillText(vals[s].toFixed(2), k * c + c / 2, r * c + c * 0.8);
        } else if (ch === "S") {
          g.font = `600 ${c * 0.3}px system-ui`;
          g.fillText("S", k * c + c / 2, r * c + c / 2);
        }
      }
    if (path && at !== undefined) {
      g.strokeStyle = "#e8743b";
      g.lineWidth = 3;
      g.beginPath();
      for (let i = 0; i <= at; i++) {
        const s = path[i], x = (s % n()) * c + c / 2, y = Math.floor(s / n()) * c + c / 2;
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke();
      const s = path[at];
      g.fillStyle = "#e8743b";
      g.beginPath();
      g.arc((s % n()) * c + c / 2, Math.floor(s / n()) * c + c / 2, c * 0.18, 0, 2 * Math.PI);
      g.fill();
    }
  };

  $("solve").onclick = () => {
    stop?.();
    stop = animate(() => {
      const d = sweep();
      paint();
      return d > 1e-6;
    });
  };
  $("sweep").onclick = () => {
    stop?.();
    sweep();
    paint();
  };
  $("walk").onclick = () => {
    if (!pol) {
      for (let i = 0; i < 500 && sweep() > 1e-8; i++);
    }
    stop?.();
    path = Array.from(ruai.lake_walk());
    let i = 0, tick = 0;
    stop = animate(() => {
      if (++tick % 6) return true;
      paint(i);
      if (++i >= path.length) {
        walks++;
        if (rows[Math.floor(path.at(-1) / n())][path.at(-1) % n()] === "G") wins++;
        $("walks").textContent = walks;
        $("wins").textContent = `${wins} (${Math.round((100 * wins) / walks)}%)`;
        return false;
      }
    });
  };
  $("cv").onclick = (e) => {
    const r = $("cv").getBoundingClientRect();
    const k = Math.floor(((e.clientX - r.left) / r.width) * n()), row = Math.floor(((e.clientY - r.top) / r.height) * n());
    const ch = rows[row][k];
    if (ch !== "F" && ch !== "H") return;
    rows[row] = rows[row].slice(0, k) + (ch === "F" ? "H" : "F") + rows[row].slice(k + 1);
    fresh();
  };
  $("map").onchange = () => {
    rows = MAPS[$("map").value].slice();
    fresh();
  };
  $("slip").onchange = fresh;
  fresh();
  return () => stop?.();
}

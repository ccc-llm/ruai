// Pictures of the classic-control environments on a canvas, from their observations.
import { css } from "./ui.mjs";

const theme = () => ({ fg: css("--text"), muted: css("--muted"), line: css("--line"), accent: css("--accent"), c1: css("--c1"), c2: css("--c2") });

/** CartPole: cart position x (±2.4) and pole angle θ. */
export function drawCartPole(g, w, h, x, theta) {
  const t = theme();
  g.clearRect(0, 0, w, h);
  const scale = w / 6, ground = h * 0.72, cx = w / 2 + x * scale;
  g.strokeStyle = t.line;
  g.lineWidth = 2;
  g.beginPath();
  g.moveTo(0, ground);
  g.lineTo(w, ground);
  g.stroke();
  for (const edge of [-2.4, 2.4]) {
    g.fillStyle = t.muted;
    g.fillRect(w / 2 + edge * scale - 1, ground - 10, 2, 20);
  }
  g.fillStyle = t.accent;
  g.fillRect(cx - 32, ground - 22, 64, 26);
  const len = scale * 1.6;
  g.strokeStyle = t.c1;
  g.lineWidth = 8;
  g.lineCap = "round";
  g.beginPath();
  g.moveTo(cx, ground - 18);
  g.lineTo(cx + len * Math.sin(theta), ground - 18 - len * Math.cos(theta));
  g.stroke();
  g.fillStyle = t.fg;
  g.beginPath();
  g.arc(cx, ground - 18, 4, 0, 2 * Math.PI);
  g.fill();
}

const hill = (p) => Math.sin(3 * p) * 0.45 + 0.55;

/** MountainCar: the car's position p in [-1.2, 0.6]. */
export function drawMountainCar(g, w, h, p) {
  const t = theme();
  g.clearRect(0, 0, w, h);
  const X = (q) => ((q + 1.2) / 1.8) * (w - 40) + 20, Y = (q) => h - 20 - hill(q) * (h - 60);
  g.beginPath();
  for (let i = 0; i <= 100; i++) {
    const q = -1.2 + (1.8 * i) / 100;
    i ? g.lineTo(X(q), Y(q)) : g.moveTo(X(q), Y(q));
  }
  g.strokeStyle = t.muted;
  g.lineWidth = 2;
  g.stroke();
  // the flag
  g.strokeStyle = t.fg;
  g.beginPath();
  g.moveTo(X(0.5), Y(0.5));
  g.lineTo(X(0.5), Y(0.5) - 34);
  g.stroke();
  g.fillStyle = t.c1;
  g.beginPath();
  g.moveTo(X(0.5), Y(0.5) - 34);
  g.lineTo(X(0.5) + 18, Y(0.5) - 28);
  g.lineTo(X(0.5), Y(0.5) - 22);
  g.fill();
  const ang = Math.atan(Math.cos(3 * p) * 3 * 0.45 * ((h - 60) / ((w - 40) / 1.8)));
  g.save();
  g.translate(X(p), Y(p));
  g.rotate(-ang);
  g.fillStyle = t.accent;
  g.fillRect(-18, -16, 36, 12);
  g.fillStyle = t.fg;
  for (const dx of [-11, 11]) {
    g.beginPath();
    g.arc(dx, -3, 4, 0, 2 * Math.PI);
    g.fill();
  }
  g.restore();
}

/** Acrobot: the two joint angles. */
export function drawAcrobot(g, w, h, th1, th2) {
  const t = theme();
  g.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2, L = Math.min(w, h) * 0.22;
  g.strokeStyle = t.muted;
  g.lineWidth = 1.5;
  g.setLineDash([6, 6]);
  g.beginPath();
  g.moveTo(0, cy - L);
  g.lineTo(w, cy - L);
  g.stroke();
  g.setLineDash([]);
  const x1 = cx + L * Math.sin(th1), y1 = cy + L * Math.cos(th1);
  const x2 = x1 + L * Math.sin(th1 + th2), y2 = y1 + L * Math.cos(th1 + th2);
  g.lineCap = "round";
  g.lineWidth = 10;
  g.strokeStyle = t.accent;
  g.beginPath();
  g.moveTo(cx, cy);
  g.lineTo(x1, y1);
  g.stroke();
  g.strokeStyle = t.c1;
  g.beginPath();
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);
  g.stroke();
  g.fillStyle = t.fg;
  for (const [x, y] of [[cx, cy], [x1, y1]]) {
    g.beginPath();
    g.arc(x, y, 5, 0, 2 * Math.PI);
    g.fill();
  }
}

/** Pendulum: the angle θ (0 = upright) and the last torque. */
export function drawPendulum(g, w, h, theta, torque) {
  const t = theme();
  g.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2, L = Math.min(w, h) * 0.36;
  g.lineCap = "round";
  g.lineWidth = 14;
  g.strokeStyle = t.c1;
  g.beginPath();
  g.moveTo(cx, cy);
  g.lineTo(cx + L * Math.sin(theta), cy - L * Math.cos(theta));
  g.stroke();
  if (torque) {
    g.strokeStyle = t.accent;
    g.lineWidth = 3;
    g.beginPath();
    const r = 26, a0 = -Math.PI / 2;
    g.arc(cx, cy, r, a0, a0 - torque * 0.9, torque > 0);
    g.stroke();
  }
  g.fillStyle = t.fg;
  g.beginPath();
  g.arc(cx, cy, 6, 0, 2 * Math.PI);
  g.fill();
}

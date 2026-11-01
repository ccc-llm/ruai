// The site's shell: the navigation and a hash router that mounts one demo
// page at a time (each page module exports `mount(el)`, which may return a
// function that unmounts it).
import { ready } from "./lib/ruai.mjs";

export const PAGES = [
  { group: "Start", items: [["", "Overview", "home"]] },
  { group: "py0 (Python → wasm)", items: [["ide", "py0 IDE", "ide"], ["notebook", "py0 Notebook", "notebook"]] },
  { group: "Neural networks", items: [["playground", "Classifier playground", "playground"], ["fit", "Curve fitting", "fit"], ["descent", "Optimizer race", "descent"], ["gpt", "GPT on WebGPU", "gpt"]] },
  { group: "Reinforcement learning", items: [["cartpole", "CartPole · REINFORCE", "cartpole"], ["mountaincar", "MountainCar · Q-learning", "mountaincar"], ["lake", "FrozenLake · value iteration", "lake"], ["gym", "Play the environments", "gym"]] },
  { group: "Plots", items: [["plots", "ruai.plot gallery", "plots"]] },
];
const FULL = new Set(["ide", "notebook"]);

const view = document.getElementById("view");
const side = document.getElementById("side");
side.innerHTML = PAGES.map((g) => `<h4>${g.group}</h4>` + g.items.map(([r, t]) => `<a href="#/${r}" data-route="${r}">${t}</a>`).join("")).join("");
document.getElementById("menu").onclick = () => side.classList.toggle("open");

ready.then((name) => (document.getElementById("device").textContent = name)).catch(() => {});

let unmount = null;
let seq = 0;

async function route() {
  const r = location.hash.replace(/^#\/?/, "").split("?")[0];
  const entry = PAGES.flatMap((g) => g.items).find(([k]) => k === r) ?? PAGES[0].items[0];
  for (const a of side.querySelectorAll("a")) a.classList.toggle("on", a.dataset.route === entry[0]);
  side.classList.remove("open");
  const mine = ++seq;
  if (unmount) {
    try {
      unmount();
    } catch (e) {
      console.error(e);
    }
    unmount = null;
  }
  view.className = FULL.has(entry[0]) ? "full" : "";
  view.innerHTML = "";
  view.scrollTop = 0;
  document.title = entry[0] ? `${entry[1]} · ruai` : "ruai in the Browser";
  const mod = await import(`./demos/${entry[2]}.mjs`);
  if (mine !== seq) return;
  try {
    unmount = (await mod.mount(view)) ?? null;
  } catch (e) {
    console.error(e);
    view.insertAdjacentHTML("beforeend", `<div class="page"><div class="err">${String(e?.message ?? e)}</div></div>`);
  }
  if (mine !== seq && unmount) {
    unmount();
    unmount = null;
  }
}

addEventListener("hashchange", route);
route();

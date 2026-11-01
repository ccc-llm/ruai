// The overview: what ruai is and a tile for each demo.
import { PAGES } from "../app.mjs";
import { esc } from "../lib/ui.mjs";

const BLURBS = {
  ide: ["py0", "A VS Code-style editor: write typed Python, compile it to WebAssembly in the page and run it."],
  notebook: ["py0", "Cells, outputs and Markdown like Jupyter, exported as a real .ipynb file."],
  playground: ["nn", "An MLP learns to separate spirals, circles and XOR, with its decision boundary drawn live."],
  fit: ["nn", "A network fits sin, |x|, a step or sinc from noisy samples; plots by ruai.plot."],
  descent: ["autograd", "SGD, momentum and Adam race down Rosenbrock's valley. Click to choose the start."],
  gpt: ["WebGPU", "Train a character-level GPT on your GPU with WGSL compute shaders, then let it write."],
  cartpole: ["RL", "A policy network learns to balance a pole with the REINFORCE policy gradient."],
  mountaincar: ["RL", "Tabular Q-learning learns to rock an underpowered car up the hill."],
  lake: ["RL", "Value iteration on a slippery lake you can redraw: values, policy, and a walk."],
  gym: ["world", "Drive CartPole, MountainCar, Acrobot and Pendulum yourself with the keyboard."],
  plots: ["plot", "Lines, scatters, histograms, bars, subplots and a Mandelbrot imshow."],
};

export function mount(el) {
  const tiles = PAGES.slice(1)
    .map(
      (g) =>
        `<h2 class="section-title">${esc(g.group)}</h2><div class="grid">` +
        g.items
          .map(([r, t]) => {
            const [tag, text] = BLURBS[r] ?? ["", ""];
            return `<a class="tile" href="#/${r}"><i>${tag}</i><b>${esc(t)}</b><span>${esc(text)}</span></a>`;
          })
          .join("") +
        "</div>",
    )
    .join("");
  el.innerHTML = `<div class="page">
    <div class="hero">
      <h1>ruai, running in your browser</h1>
      <p class="lead">ruai is a Swiss-army AI toolkit in Rust: a PyTorch-like engine with autograd and GPU backends, Gymnasium environments, matplotlib-style plots, and py0, a typed Python compiled to Rust or WebAssembly. On this site all of it is compiled to WebAssembly and runs on your machine, with no server. Training the GPT uses your GPU through WebGPU.</p>
    </div>
    ${tiles}
    <p class="note" style="margin-top:28px">Source: <a href="https://github.com/ccc-llm/ruai">github.com/ccc-llm/ruai</a> · <code>cargo install ruai</code></p>
  </div>`;
}

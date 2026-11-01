// The py0 notebook: code and Markdown cells like Jupyter, run by compiling
// the cells up to the one run (see lib/notebook.mjs), saved in this browser,
// exported to and opened from .ipynb files.
import { runPy0, stopPy0, warmPy0, errorLine } from "../lib/py0.mjs";
import { createEditor, loadMonaco } from "../lib/editor.mjs";
import { markdown } from "../lib/markdown.mjs";
import { assemble, splitOutput, cellOfLine, toIpynb, fromIpynb } from "../lib/notebook.mjs";
import { esc, saveFile, openFile } from "../lib/ui.mjs";

const KEY = "ruai.notebook.v1";

const STARTER = [
  { type: "markdown", source: "# A py0 notebook\n\nWrite typed Python in cells and run them with **Shift + Enter**. Each run compiles the cells up to the one you run into WebAssembly, in your browser, so variables, functions and classes from earlier cells are there.\n\nA cell that ends in an expression shows its value. **Export** saves a real `.ipynb` file that Jupyter opens, and since py0 is a subset of Python, CPython runs it too." },
  { type: "code", source: "primes = [n for n in range(2, 60) if all(n % d != 0 for d in range(2, n))]\nlen(primes)" },
  { type: "code", source: "def collatz(n: int) -> list[int]:\n    seq = [n]\n    while n != 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        seq.append(n)\n    return seq\n\n\nlongest = max(range(1, 1000), key=lambda n: len(collatz(n)))\nprint(f\"{longest} takes {len(collatz(longest)) - 1} steps\")\ncollatz(27)[:12]" },
  { type: "markdown", source: "## Classes carry over too\n\nThe cell below defines a class; the one after it uses it." },
  { type: "code", source: "from dataclasses import dataclass\n\n\n@dataclass\nclass Vec:\n    x: float\n    y: float\n\n    def __add__(self, o: \"Vec\") -> \"Vec\":\n        return Vec(self.x + o.x, self.y + o.y)\n\n    def __mul__(self, k: float) -> \"Vec\":\n        return Vec(self.x * k, self.y * k)" },
  { type: "code", source: "# a projectile, stepped with Euler's method\npos = Vec(0.0, 0.0)\nvel = Vec(4.0, 9.0)\ng = Vec(0.0, -9.8)\nrows: list[str] = []\nfor step in range(20):\n    pos = pos + vel * 0.1\n    vel = vel + g * 0.1\n    if pos.y < 0:\n        break\n    rows.append(\" \" * int(pos.y * 8) + \"o\")\nprint(\"\\n\".join(rows))\npos" },
  { type: "code", source: "" },
];

let uid = 0;
const newId = () => `c${Date.now().toString(36)}${(uid++).toString(36)}`;

function load() {
  try {
    const s = JSON.parse(localStorage.getItem(KEY));
    if (s && Array.isArray(s.cells)) return s;
  } catch {}
  return null;
}

export async function mount(el) {
  el.innerHTML = `<div class="page nb-page">
    <div class="nb-top">
      <input type="text" class="nb-name" data-id="name" aria-label="Notebook name" spellcheck="false">
      <span class="nb-state" data-id="state">loading…</span>
    </div>
    <div class="nb-bar">
      <button data-a="code" title="Insert a code cell below">＋ Code</button>
      <button data-a="md" title="Insert a Markdown cell below">＋ Text</button>
      <span class="sep"></span>
      <button data-a="run" title="Run the selected cell (Shift+Enter)">▶ Run</button>
      <button data-a="all" title="Run all cells">⏩ Run all</button>
      <button data-a="stop" title="Stop">■</button>
      <button data-a="clear" title="Clear all outputs">Clear outputs</button>
      <span class="sep"></span>
      <button data-a="up" title="Move the cell up">↑</button><button data-a="down" title="Move the cell down">↓</button>
      <button data-a="type" title="Change the cell between code and Markdown">Code ⇄ Text</button>
      <button data-a="del" title="Delete the cell">Delete</button>
      <span class="grow"></span>
      <button data-a="open" title="Open an .ipynb file">Open .ipynb</button>
      <button data-a="export" class="primary" title="Download as .ipynb">Export .ipynb</button>
      <button data-a="new" title="Start a new notebook">New</button>
    </div>
    <div class="nb-cells" data-id="cells"></div>
    <p class="note nb-help"><span><kbd>Shift</kbd>+<kbd>Enter</kbd> run and move on</span><span><kbd>Ctrl/⌘</kbd>+<kbd>Enter</kbd> run in place</span><span><kbd>Alt</kbd>+<kbd>Enter</kbd> run and insert a cell</span><span>double-click text to edit</span></p>
  </div>`;
  const $ = (id) => el.querySelector(`[data-id="${id}"]`);
  const cellsEl = $("cells");

  let nb = load() ?? { name: "py0-notebook", cells: STARTER.map((c) => ({ ...c, id: newId(), outputs: null, count: null })) };
  let counter = Math.max(0, ...nb.cells.map((c) => c.count ?? 0));
  let selected = 0;
  let busy = false;
  const views = new Map(); // id → { root, editor, ... }
  $("name").value = nb.name;
  $("name").oninput = () => {
    nb.name = $("name").value.trim() || "py0-notebook";
    save();
  };

  const save = () => {
    try {
      localStorage.setItem(KEY, JSON.stringify(nb));
    } catch {}
  };
  const state = (s) => ($("state").textContent = s);

  await loadMonaco();

  const renderOutputs = (c) => {
    const v = views.get(c.id);
    if (!v) return;
    const o = c.outputs;
    v.prompt.textContent = c.type === "code" ? `[${c.running ? "*" : c.count ?? " "}]` : "";
    let h = "";
    if (o) {
      if (o.stdout) h += `<pre class="nb-out">${esc(o.stdout)}</pre>`;
      if (o.stderr) h += `<pre class="nb-out nb-err">${esc(o.stderr)}</pre>`;
      if (o.result != null) h += `<div class="nb-result"><span class="nb-outp">Out[${c.count ?? " "}]:</span><pre class="nb-out">${esc(o.result)}</pre></div>`;
    }
    v.out.innerHTML = h;
    v.out.hidden = !h;
  };

  const renderMarkdown = (c) => {
    const v = views.get(c.id);
    v.md.innerHTML = markdown(c.source) || `<p class="note">Empty text cell: double-click to edit.</p>`;
    v.md.hidden = !!c.editing;
    v.edbox.hidden = !c.editing;
    if (c.editing) v.editor.layout();
  };

  const select = (i, focus = false) => {
    selected = Math.max(0, Math.min(i, nb.cells.length - 1));
    for (const [k, v] of views) v.root.classList.toggle("sel", nb.cells[selected]?.id === k);
    const c = nb.cells[selected];
    if (focus && c) {
      const v = views.get(c.id);
      if (c.type === "markdown" && !c.editing) v.root.focus();
      else v.editor.focus();
      v.root.scrollIntoView({ block: "nearest" });
    }
  };

  const makeView = async (c) => {
    const root = document.createElement("div");
    root.className = `nb-cell ${c.type}`;
    root.tabIndex = -1;
    root.innerHTML = `<div class="nb-gutter"><button class="nb-play" title="Run (Shift+Enter)">▶</button><span class="nb-prompt"></span></div>
      <div class="nb-body"><div class="nb-md" hidden></div><div class="nb-ed"></div><div class="nb-outs" hidden></div></div>`;
    const v = { root, prompt: root.querySelector(".nb-prompt"), out: root.querySelector(".nb-outs"), md: root.querySelector(".nb-md"), edbox: root.querySelector(".nb-ed") };
    views.set(c.id, v);
    const index = () => nb.cells.findIndex((x) => x.id === c.id);
    v.editor = await createEditor(v.edbox, {
      value: c.source,
      autoHeight: true,
      lineNumbers: false,
      wordWrap: c.type === "markdown" ? "on" : "off",
      actions: [
        { id: "nb.runNext", label: "Run cell and select below", keys: ["Shift+Enter"], run: () => runAndAdvance(index(), false) },
        { id: "nb.run", label: "Run cell", keys: ["Ctrl+Enter"], run: () => runCell(index()) },
        { id: "nb.runInsert", label: "Run cell and insert below", keys: ["Alt+Enter"], run: () => runAndAdvance(index(), true) },
      ],
    });
    v.editor.onChange(() => {
      c.source = v.editor.getValue();
      save();
    });
    v.editor.onFocus(() => select(index()));
    root.addEventListener("mousedown", () => select(index()));
    root.querySelector(".nb-play").onclick = () => runAndAdvance(index(), false);
    v.md.ondblclick = () => {
      c.editing = true;
      renderMarkdown(c);
      v.editor.focus();
    };
    root.addEventListener("keydown", (e) => {
      if (e.target !== root) return;
      if (e.key === "Enter" && e.shiftKey) {
        e.preventDefault();
        runAndAdvance(index(), false);
      } else if (e.key === "Enter" && c.type === "markdown") {
        e.preventDefault();
        c.editing = true;
        renderMarkdown(c);
        v.editor.focus();
      }
    });
    if (c.type === "markdown") renderMarkdown(c);
    renderOutputs(c);
    return v;
  };

  const renderAll = async () => {
    for (const v of views.values()) v.editor.dispose();
    views.clear();
    cellsEl.innerHTML = "";
    for (const c of nb.cells) cellsEl.appendChild((await makeView(c)).root);
    select(selected);
  };

  const insert = async (at, type, source = "") => {
    const c = { id: newId(), type, source, outputs: null, count: null, editing: type === "markdown" };
    nb.cells.splice(at, 0, c);
    const v = await makeView(c);
    cellsEl.insertBefore(v.root, cellsEl.children[at] ?? null);
    save();
    select(at, true);
    return c;
  };

  const remove = (i) => {
    const [c] = nb.cells.splice(i, 1);
    const v = views.get(c.id);
    v.editor.dispose();
    v.root.remove();
    views.delete(c.id);
    if (!nb.cells.length) insert(0, "code");
    save();
    select(Math.min(i, nb.cells.length - 1), true);
  };

  const move = (i, d) => {
    const j = i + d;
    if (j < 0 || j >= nb.cells.length) return;
    [nb.cells[i], nb.cells[j]] = [nb.cells[j], nb.cells[i]];
    const a = views.get(nb.cells[j].id).root, b = views.get(nb.cells[i].id).root;
    d < 0 ? cellsEl.insertBefore(a, b) : cellsEl.insertBefore(b, a);
    save();
    select(j, true);
  };

  /** Run the code cells up to `last` (an index in nb.cells), showing outputs for `targets`. */
  async function execute(last, targets) {
    if (busy) return false;
    busy = true;
    const code = [], map = [];
    nb.cells.forEach((c, i) => {
      if (i <= last && c.type === "code") {
        code.push(c.source);
        map.push(i);
      }
    });
    const show = new Set(map.map((ci, k) => (targets.has(ci) ? k : -1)).filter((k) => k >= 0));
    for (const i of targets) {
      nb.cells[i].running = true;
      renderOutputs(nb.cells[i]);
    }
    state("compiling…");
    let { src, starts } = assemble(code, show);
    let chunks = [];
    let r = await runPy0(src, { onOutput: (k, t) => chunks.push([k, t]) });
    // an expression that cannot be printed: run again without the displays
    if (r.compileError && show.size) {
      ({ src, starts } = assemble(code, show, false));
      chunks = [];
      r = await runPy0(src, { onOutput: (k, t) => chunks.push([k, t]) });
    }
    busy = false;
    for (const i of targets) nb.cells[i].running = false;
    if (r.stopped) {
      state("stopped");
      for (const i of targets) renderOutputs(nb.cells[i]);
      return false;
    }
    if (r.compileError) {
      const line = errorLine(r.compileError);
      const at = line ? cellOfLine(starts, line) : null;
      const ci = at ? map[at.cell] : [...targets].pop();
      const msg = r.compileError.replace(/^line \d+:\s*/, "");
      for (const i of targets) {
        const c = nb.cells[i];
        if (i === ci) {
          c.outputs = { stdout: "", stderr: `CompileError${at ? ` (line ${at.line})` : ""}: ${msg}\n`, result: null };
          views.get(c.id)?.editor.setErrors(at ? [{ line: at.line, message: msg }] : []);
        }
        renderOutputs(c);
      }
      if (ci !== undefined && !targets.has(ci)) {
        const c = nb.cells[ci];
        c.outputs = { stdout: "", stderr: `CompileError${at ? ` (line ${at.line})` : ""}: ${msg}\n`, result: null };
        views.get(c.id)?.editor.setErrors(at ? [{ line: at.line, message: msg }] : []);
        renderOutputs(c);
      }
      state(`compile error in cell ${ci + 1}`);
      save();
      return false;
    }
    const per = splitOutput(chunks, code.length);
    let failed = false;
    map.forEach((ci, k) => {
      if (!targets.has(ci)) return;
      const c = nb.cells[ci];
      c.count = ++counter;
      c.outputs = per[k];
      views.get(c.id)?.editor.setErrors([]);
      if (per[k].stderr) failed = true;
      renderOutputs(c);
    });
    state(`${r.code === 0 ? "done" : "error"} · compiled in ${r.compileMs.toFixed(0)} ms · ran in ${r.runMs.toFixed(0)} ms`);
    save();
    return r.code === 0 && !failed;
  }

  async function runCell(i) {
    const c = nb.cells[i];
    if (!c) return false;
    if (c.type === "markdown") {
      c.editing = false;
      renderMarkdown(c);
      save();
      return true;
    }
    return execute(i, new Set([i]));
  }

  async function runAndAdvance(i, insertBelow) {
    const ok = await runCell(i);
    if (insertBelow || i === nb.cells.length - 1) await insert(i + 1, "code");
    else select(i + 1, true);
    return ok;
  }

  const runAll = async () => {
    for (const c of nb.cells)
      if (c.type === "markdown" && c.editing) {
        c.editing = false;
        renderMarkdown(c);
      }
    const code = new Set(nb.cells.map((c, i) => (c.type === "code" && c.source.trim() ? i : -1)).filter((i) => i >= 0));
    if (code.size) await execute(Math.max(...code), code);
  };

  el.querySelector(".nb-bar").onclick = async (e) => {
    const a = e.target.closest("[data-a]")?.dataset.a;
    const i = selected;
    if (a === "code") insert(i + 1, "code");
    else if (a === "md") insert(i + 1, "markdown");
    else if (a === "run") runAndAdvance(i, false);
    else if (a === "all") runAll();
    else if (a === "stop") stopPy0();
    else if (a === "clear") {
      for (const c of nb.cells) {
        c.outputs = null;
        c.count = null;
        renderOutputs(c);
      }
      counter = 0;
      save();
    } else if (a === "up") move(i, -1);
    else if (a === "down") move(i, 1);
    else if (a === "del") remove(i);
    else if (a === "type") {
      const c = nb.cells[i];
      c.type = c.type === "code" ? "markdown" : "code";
      c.outputs = null;
      c.count = null;
      c.editing = c.type === "markdown";
      await renderAll();
      select(i, true);
      save();
    } else if (a === "export") {
      saveFile(`${nb.name.replace(/\.ipynb$/, "")}.ipynb`, toIpynb(nb), "application/x-ipynb+json");
      state("exported");
    } else if (a === "open") {
      const f = await openFile(".ipynb,application/json");
      if (!f) return;
      try {
        nb = { name: f.name.replace(/\.ipynb$/, ""), cells: fromIpynb(f.text).map((c) => ({ ...c, id: newId() })) };
        if (!nb.cells.length) nb.cells.push({ id: newId(), type: "code", source: "", outputs: null, count: null });
        counter = Math.max(0, ...nb.cells.map((c) => c.count ?? 0));
        $("name").value = nb.name;
        selected = 0;
        await renderAll();
        save();
        state(`opened ${f.name}`);
      } catch (err) {
        state(`could not open ${f.name}: ${err.message}`);
      }
    } else if (a === "new") {
      if (!confirm("Start a new, empty notebook? The current one is replaced (export it first to keep it).")) return;
      nb = { name: "untitled", cells: [{ id: newId(), type: "code", source: "", outputs: null, count: null }] };
      counter = 0;
      $("name").value = nb.name;
      selected = 0;
      await renderAll();
      save();
    }
  };

  await renderAll();
  state("loading the compiler…");
  warmPy0().then(() => state("ready"));
  return () => {
    for (const v of views.values()) v.editor.dispose();
  };
}

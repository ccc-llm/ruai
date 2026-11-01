// The py0 IDE: an explorer of files kept in this browser, tabs, the Monaco
// editor, and a panel with the terminal, problems and standard input.
// Programs compile to WebAssembly and run in a worker.
import { runPy0, stopPy0, warmPy0, errorLine, py0Examples } from "../lib/py0.mjs";
import { createEditor } from "../lib/editor.mjs";
import { esc, saveFile, openFile } from "../lib/ui.mjs";

const KEY = "ruai.ide.v1";
const WELCOME = `# Welcome to the py0 IDE.
#
# py0 is a typed subset of Python. ruai compiles it to WebAssembly right
# here in the browser (the compiler itself is wasm), then runs the module.
#
#   Run:  Ctrl/Cmd + Enter, or F5, or the ▶ button
#   Files are saved in this browser. The examples/ folder has more programs.

def triangle(n: int) -> list[str]:
    return [" " * (n - i) + "*" * (2 * i - 1) for i in range(1, n + 1)]


for row in triangle(6):
    print(row)

name = input("What is your name? ")
print(f"Hello, {name}! len(name) = {len(name)}")
`;

function load() {
  try {
    const s = JSON.parse(localStorage.getItem(KEY));
    if (s && s.files) return s;
  } catch {}
  return null;
}

export async function mount(el) {
  el.innerHTML = `<div class="ide">
    <div class="ide-act">
      <button class="on" data-id="explorer" title="Explorer (toggle)" aria-label="Explorer">⧉</button>
    </div>
    <aside class="ide-side">
      <div class="ide-head"><span>EXPLORER</span>
        <span class="ide-tools"><button data-a="new" title="New file">＋</button><button data-a="upload" title="Open a .py file from your computer">⇪</button><button data-a="restore" title="Restore the examples">↺</button></span></div>
      <div class="ide-tree" data-id="tree"></div>
    </aside>
    <section class="ide-main">
      <div class="ide-tabs"><div class="ide-tablist" data-id="tabs"></div>
        <div class="ide-run"><button data-id="run" class="go" title="Run (Ctrl/Cmd+Enter, F5)">▶ Run</button><button data-id="stop" title="Stop" disabled>■</button></div></div>
      <div class="ide-crumbs" data-id="crumbs"></div>
      <div class="ide-editor" data-id="editor"></div>
      <div class="ide-split" data-id="split" title="Drag to resize"></div>
      <div class="ide-panel" data-id="panel">
        <div class="ide-ptabs">
          <button data-p="term" class="on">TERMINAL</button><button data-p="problems">PROBLEMS <span class="badge" data-id="count"></span></button><button data-p="stdin">INPUT</button>
          <span class="ide-ptools"><button data-id="clear" title="Clear the terminal">⌫</button></span>
        </div>
        <pre class="ide-term" data-id="term"></pre>
        <div class="ide-problems" data-id="problems" hidden></div>
        <textarea class="ide-stdin" data-id="stdin" hidden spellcheck="false" placeholder="Lines for input(), one per line. Each run reads from the top."></textarea>
      </div>
    </section>
    <footer class="ide-status"><span data-id="sline">py0 → WebAssembly</span><span class="grow"></span><span data-id="pos">Ln 1, Col 1</span><span>Spaces: 4</span><span>UTF-8</span><span>py0</span></footer>
  </div>`;
  const $ = (id) => el.querySelector(`[data-id="${id}"]`);

  const examples = await py0Examples();
  const fresh = () => ({
    files: { "main.py": WELCOME, ...Object.fromEntries(examples.map((e) => [`examples/${e.name}`, e.text])) },
    open: ["main.py"],
    active: "main.py",
    stdin: "Ada\n",
    collapsed: [],
  });
  let st = load() ?? fresh();
  const save = () => {
    try {
      localStorage.setItem(KEY, JSON.stringify(st));
    } catch {}
  };

  // one editor; each file keeps its own text in `st.files`
  let switching = false;
  const ed = await createEditor($("editor"), {
    value: st.files[st.active] ?? "",
    minimap: !matchMedia("(max-width: 800px)").matches,
    actions: [
      { id: "py0.run", label: "py0: Run", keys: ["Ctrl+Enter", "F5"], run: () => go() },
      { id: "py0.save", label: "Save", keys: ["Ctrl+S"], run: () => flash("saved in this browser") },
    ],
  });
  ed.onChange(() => {
    if (switching || !st.active) return;
    st.files[st.active] = ed.getValue();
    save();
  });
  ed.onCursor((l, c) => ($("pos").textContent = `Ln ${l}, Col ${c}`));
  $("stdin").value = st.stdin ?? "";
  $("stdin").oninput = () => {
    st.stdin = $("stdin").value;
    save();
  };

  const flash = (msg) => {
    $("sline").textContent = msg;
  };

  const tree = () => {
    const names = Object.keys(st.files).sort((a, b) => (a.includes("/") - b.includes("/")) || a.localeCompare(b));
    const folders = {};
    for (const n of names) {
      const i = n.lastIndexOf("/");
      (folders[i < 0 ? "" : n.slice(0, i)] ??= []).push(n);
    }
    const file = (n, depth) =>
      `<div class="ide-file${n === st.active ? " on" : ""}" data-f="${esc(n)}" style="padding-left:${10 + depth * 14}px"><span class="ic">py</span><span class="nm">${esc(n.slice(n.lastIndexOf("/") + 1))}</span>
        <span class="acts"><button data-x="rename" title="Rename">✎</button><button data-x="download" title="Download">⤓</button><button data-x="delete" title="Delete">✕</button></span></div>`;
    let h = `<div class="ide-root">▾ WORKSPACE</div>`;
    for (const [dir, files] of Object.entries(folders).sort()) {
      if (dir) {
        const shut = st.collapsed.includes(dir);
        h += `<div class="ide-dir" data-d="${esc(dir)}" style="padding-left:10px">${shut ? "▸" : "▾"} ${esc(dir)}</div>`;
        if (!shut) h += files.map((f) => file(f, 1)).join("");
      } else h += files.map((f) => file(f, 0)).join("");
    }
    $("tree").innerHTML = h;
  };

  const tabs = () => {
    $("tabs").innerHTML = st.open
      .map((n) => `<div class="ide-tab${n === st.active ? " on" : ""}" data-f="${esc(n)}" title="${esc(n)}"><span class="ic">py</span>${esc(n.slice(n.lastIndexOf("/") + 1))}<button data-close title="Close">✕</button></div>`)
      .join("");
    $("crumbs").textContent = st.active ? st.active.split("/").join("  ›  ") : "";
  };

  const openFileTab = (n) => {
    if (!(n in st.files)) return;
    if (!st.open.includes(n)) st.open.push(n);
    st.active = n;
    switching = true;
    ed.setValue(st.files[n]);
    switching = false;
    ed.setErrors([]);
    save();
    tree();
    tabs();
    ed.focus();
  };

  const closeTab = (n) => {
    const i = st.open.indexOf(n);
    st.open.splice(i, 1);
    if (st.active === n) {
      st.active = st.open[Math.min(i, st.open.length - 1)] ?? null;
      switching = true;
      ed.setValue(st.active ? st.files[st.active] : "");
      switching = false;
    }
    save();
    tree();
    tabs();
  };

  const uniq = (base) => {
    let n = base, k = 1;
    while (n in st.files) n = base.replace(/(\.py)?$/, `_${k++}.py`);
    return n;
  };

  $("tree").onclick = (e) => {
    const d = e.target.closest("[data-d]");
    if (d) {
      const dir = d.dataset.d;
      st.collapsed = st.collapsed.includes(dir) ? st.collapsed.filter((x) => x !== dir) : [...st.collapsed, dir];
      save();
      return tree();
    }
    const f = e.target.closest("[data-f]")?.dataset.f;
    if (!f) return;
    const x = e.target.closest("[data-x]")?.dataset.x;
    if (x === "delete") {
      if (!confirm(`Delete ${f}?`)) return;
      delete st.files[f];
      if (st.open.includes(f)) closeTab(f);
      save();
      tree();
    } else if (x === "rename") {
      let to = prompt("Rename to", f);
      if (!to || to === f) return;
      if (!to.endsWith(".py")) to += ".py";
      if (to in st.files) return alert(`${to} already exists`);
      st.files[to] = st.files[f];
      delete st.files[f];
      st.open = st.open.map((n) => (n === f ? to : n));
      if (st.active === f) st.active = to;
      save();
      tree();
      tabs();
    } else if (x === "download") saveFile(f.slice(f.lastIndexOf("/") + 1), st.files[f], "text/x-python");
    else openFileTab(f);
  };
  $("tabs").onclick = (e) => {
    const f = e.target.closest("[data-f]")?.dataset.f;
    if (!f) return;
    if (e.target.closest("[data-close]")) closeTab(f);
    else openFileTab(f);
  };
  $("tabs").onauxclick = (e) => {
    const f = e.target.closest("[data-f]")?.dataset.f;
    if (f && e.button === 1) closeTab(f);
  };
  el.querySelector(".ide-tools").onclick = async (e) => {
    const a = e.target.closest("[data-a]")?.dataset.a;
    if (a === "new") {
      let n = prompt("New file name", uniq("untitled.py"));
      if (!n) return;
      if (!n.endsWith(".py")) n += ".py";
      n = uniq(n);
      st.files[n] = "";
      openFileTab(n);
    } else if (a === "upload") {
      const f = await openFile(".py,text/x-python,text/plain");
      if (!f) return;
      const n = uniq(f.name.endsWith(".py") ? f.name : f.name + ".py");
      st.files[n] = f.text;
      openFileTab(n);
    } else if (a === "restore") {
      for (const x of examples) st.files[`examples/${x.name}`] = x.text;
      if (st.active?.startsWith("examples/")) openFileTab(st.active);
      save();
      tree();
      flash("examples restored");
    }
  };

  // the panel
  const showPanel = (p) => {
    for (const b of el.querySelectorAll("[data-p]")) b.classList.toggle("on", b.dataset.p === p);
    $("term").hidden = p !== "term";
    $("problems").hidden = p !== "problems";
    $("stdin").hidden = p !== "stdin";
  };
  el.querySelector(".ide-ptabs").onclick = (e) => {
    const p = e.target.closest("[data-p]")?.dataset.p;
    if (p) showPanel(p);
  };
  $("clear").onclick = () => ($("term").innerHTML = "");
  $("split").onpointerdown = (e) => {
    const panel = $("panel"), y0 = e.clientY, h0 = panel.offsetHeight;
    const move = (ev) => (panel.style.height = Math.max(80, Math.min(h0 - (ev.clientY - y0), el.clientHeight - 160)) + "px");
    const up = () => {
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", up);
      ed.layout();
    };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  };

  const term = (cls, text) => {
    const s = document.createElement("span");
    if (cls) s.className = cls;
    s.textContent = text;
    $("term").appendChild(s);
    $("term").scrollTop = $("term").scrollHeight;
  };

  let running = false;
  async function go() {
    if (running || !st.active) return;
    running = true;
    $("run").disabled = true;
    $("stop").disabled = false;
    showPanel("term");
    const file = st.active;
    term("cmd", `${$("term").childNodes.length ? "\n" : ""}$ ruai py wasm --run ${file}\n`);
    flash(`compiling ${file}…`);
    const r = await runPy0(st.files[file], { stdin: st.stdin ?? "", onOutput: (kind, text) => term(kind === "err" ? "e" : "", text) });
    running = false;
    $("run").disabled = false;
    $("stop").disabled = true;
    if (r.stopped) {
      term("e", "\n[stopped]\n");
      flash("stopped");
      return;
    }
    if (r.compileError) {
      const line = errorLine(r.compileError);
      const msg = r.compileError.replace(/^line \d+:\s*/, "");
      ed.setErrors(line ? [{ line, message: msg }] : []);
      $("count").textContent = "1";
      $("problems").innerHTML = `<div class="ide-prob" data-line="${line ?? ""}"><span class="x">⊗</span> ${esc(msg)} <span class="where">${esc(file)}${line ? ` [Ln ${line}]` : ""}</span></div>`;
      term("e", r.compileError.trimEnd() + "\n");
      flash(`compile error (${r.compileMs.toFixed(0)} ms)`);
      return;
    }
    ed.setErrors([]);
    $("count").textContent = "";
    $("problems").innerHTML = `<div class="ide-none">No problems have been detected in ${esc(file)}.</div>`;
    term("dim", `[exit ${r.code} · compiled in ${r.compileMs.toFixed(0)} ms to ${(r.size / 1024).toFixed(1)} KB of wasm · ran in ${r.runMs.toFixed(0)} ms]\n`);
    flash(`exit ${r.code} · compile ${r.compileMs.toFixed(0)} ms · run ${r.runMs.toFixed(0)} ms`);
  }
  $("problems").onclick = (e) => {
    const l = +e.target.closest("[data-line]")?.dataset.line;
    if (l) ed.revealLine(l);
  };
  $("explorer").onclick = () => {
    const ide = el.querySelector(".ide");
    const narrow = matchMedia("(max-width: 800px)").matches;
    ide.classList.toggle(narrow ? "show-side" : "hide-side");
    $("explorer").classList.toggle("on", narrow ? ide.classList.contains("show-side") : !ide.classList.contains("hide-side"));
    setTimeout(() => ed.layout(), 0);
  };
  // on a phone a chosen file closes the explorer
  $("tree").addEventListener("click", (e) => {
    if (e.target.closest(".ide-file") && !e.target.closest("[data-x]")) el.querySelector(".ide").classList.remove("show-side");
  });
  $("run").onclick = go;
  $("stop").onclick = () => stopPy0();
  const onKey = (e) => {
    if (e.key === "F5") {
      e.preventDefault();
      go();
    }
  };
  addEventListener("keydown", onKey);

  tree();
  tabs();
  if (matchMedia("(max-width: 800px)").matches) $("explorer").classList.remove("on");
  $("problems").innerHTML = `<div class="ide-none">No problems have been detected.</div>`;
  flash("loading the compiler…");
  warmPy0().then(() => flash("ready · Ctrl/Cmd + Enter runs"));
  ed.focus();
  return () => {
    removeEventListener("keydown", onKey);
    ed.dispose();
  };
}

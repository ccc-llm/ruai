// Notebook execution and the .ipynb format.
//
// py0 compiles whole programs, so running cell k compiles cells 0..k as one
// program with a marker printed before each cell; the output is split at the
// markers. A cell whose last line is an expression shows its repr, like
// Jupyter's Out[n]. py0 rejects expression statements that have no effect,
// so a last expression that is not shown is dropped unless it is a call.

const MARK = (i) => `__py0_cell_${i}__`;
const RESULT = (i) => `__py0_result_${i}__`;
const STATEMENT = /^(def|class|if|elif|else|for|while|with|try|except|finally|import|from|return|pass|break|continue|raise|assert|del|global|nonlocal|async|await|lambda|@|print\s*\()/;

/** The last line of `code` as an expression to display, or null. */
function lastExpression(code) {
  const lines = code.replace(/\s+$/, "").split("\n");
  const last = lines[lines.length - 1];
  if (!last || /^\s/.test(last) || last.startsWith("#") || STATEMENT.test(last) || /[:\\,(\[{]\s*$/.test(last)) return null;
  // an assignment at the top level (outside brackets and strings) is not an expression
  let depth = 0, q = null;
  for (let i = 0; i < last.length; i++) {
    const c = last[i];
    if (q) {
      if (c === "\\") i++;
      else if (c === q) q = null;
    } else if (c === '"' || c === "'") q = c;
    else if (c === "#") break;
    else if ("([{".includes(c)) depth++;
    else if (")]}".includes(c)) depth--;
    else if (depth === 0 && c === "=" && !"=!<>".includes(last[i - 1] ?? "") && last[i + 1] !== "=") return null;
    else if (depth === 0 && /[+\-*/%&|^@]/.test(c) && last[i + 1] === "=" && last[i + 2] !== "=") return null;
  }
  if (depth !== 0 || q) return null;
  const expr = last.replace(/\s*#.*$/, "");
  return { index: lines.length - 1, expr, call: isCall(expr) };
}

/** Whether `e` is one call, `f(...)` or `a.b(...)`, whose parentheses close at its end. */
function isCall(e) {
  const m = /^[A-Za-z_][\w.]*\(/.exec(e);
  if (!m || !e.endsWith(")")) return false;
  let depth = 0;
  for (let i = m[0].length - 1; i < e.length; i++) {
    if (e[i] === "(") depth++;
    else if (e[i] === ")" && --depth === 0) return i === e.length - 1;
  }
  return false;
}

/**
 * The program for running code cells `cells` (an array of strings) with a
 * marker before each and, for those in `show`, the repr of a last expression.
 * Returns { src, starts } where starts[i] is the program line where cell i begins.
 */
export function assemble(cells, show, display = true) {
  const out = [];
  const starts = [];
  cells.forEach((code, i) => {
    out.push(`print("${MARK(i)}")`);
    starts.push(out.length + 1);
    const e = lastExpression(code);
    const lines = code.replace(/\s+$/, "").split("\n");
    if (e && display && show.has(i)) {
      lines[e.index] = `print("${RESULT(i)}")`;
      lines.push(`print(repr(${e.expr}))`);
    } else if (e && !e.call) {
      lines[e.index] = "pass";
    }
    out.push(...lines);
  });
  return { src: out.join("\n") + "\n", starts };
}

/** The cell and its line for a program line. */
export function cellOfLine(starts, line) {
  let c = 0;
  for (let i = 0; i < starts.length; i++) if (starts[i] <= line) c = i;
  return { cell: c, line: line - starts[c] + 1 };
}

/**
 * Split the ordered output chunks of a run ([kind, text]) by cell:
 * an array (per cell) of { stdout, stderr, result }.
 */
export function splitOutput(chunks, n) {
  const cells = Array.from({ length: n }, () => ({ stdout: "", stderr: "", result: null }));
  let cur = -1, inResult = false;
  let pending = "";
  const line = (l) => {
    const m = /^__py0_(cell|result)_(\d+)__$/.exec(l.replace(/\n$/, ""));
    if (m) {
      if (m[1] === "cell") {
        cur = +m[2];
        inResult = false;
      } else inResult = true;
      return;
    }
    if (cur < 0) return;
    if (inResult) cells[cur].result = (cells[cur].result ?? "") + l;
    else cells[cur].stdout += l;
  };
  for (const [kind, text] of chunks) {
    if (kind === "err") {
      if (pending) {
        line(pending);
        pending = "";
      }
      if (cur >= 0) cells[cur].stderr += text;
      continue;
    }
    const parts = (pending + text).split(/(?<=\n)/);
    pending = parts[parts.length - 1].endsWith("\n") ? "" : parts.pop();
    for (const p of parts) line(p);
  }
  if (pending) line(pending);
  for (const c of cells) if (c.result !== null) c.result = c.result.replace(/\n$/, "");
  return cells;
}

const lines = (s) => (s ? s.split(/(?<=\n)/) : []);

/** A notebook (cells of { type, source, outputs, count }) as .ipynb JSON. */
export function toIpynb(nb) {
  return JSON.stringify(
    {
      cells: nb.cells.map((c, i) => {
        const id = c.id ?? `cell-${i}`;
        if (c.type === "markdown") return { cell_type: "markdown", id, metadata: {}, source: lines(c.source) };
        const outputs = [];
        const o = c.outputs ?? {};
        if (o.stdout) outputs.push({ output_type: "stream", name: "stdout", text: lines(o.stdout) });
        if (o.result != null) outputs.push({ output_type: "execute_result", execution_count: c.count ?? null, data: { "text/plain": lines(o.result) }, metadata: {} });
        if (o.stderr) {
          const m = /^(\w+(?:Error|Exception|Exit|Interrupt)?): ?(.*)$/m.exec(o.stderr.trim().split("\n").pop());
          if (m && /Traceback/.test(o.stderr)) outputs.push({ output_type: "error", ename: m[1], evalue: m[2], traceback: o.stderr.trimEnd().split("\n") });
          else outputs.push({ output_type: "stream", name: "stderr", text: lines(o.stderr) });
        }
        return { cell_type: "code", execution_count: c.count ?? null, id, metadata: {}, outputs, source: lines(c.source) };
      }),
      metadata: {
        kernelspec: { display_name: "Python 3 (py0)", language: "python", name: "python3" },
        language_info: { name: "python", file_extension: ".py", mimetype: "text/x-python", codemirror_mode: { name: "ipython", version: 3 } },
        py0: { note: "written in the ruai py0 notebook; py0 is a typed subset of Python, so it also runs on CPython" },
      },
      nbformat: 4,
      nbformat_minor: 5,
    },
    null,
    1,
  );
}

/** Cells from .ipynb JSON text. */
export function fromIpynb(text) {
  const j = JSON.parse(text);
  if (!Array.isArray(j.cells)) throw new Error("not a notebook: no cells");
  const src = (s) => (Array.isArray(s) ? s.join("") : s ?? "");
  return j.cells
    .filter((c) => c.cell_type === "code" || c.cell_type === "markdown" || c.cell_type === "raw")
    .map((c) => {
      if (c.cell_type !== "code") return { type: "markdown", source: src(c.source) };
      const o = { stdout: "", stderr: "", result: null };
      for (const x of c.outputs ?? []) {
        if (x.output_type === "stream") o[x.name === "stderr" ? "stderr" : "stdout"] += src(x.text);
        else if (x.output_type === "execute_result" || x.output_type === "display_data") o.result = src(x.data?.["text/plain"]);
        else if (x.output_type === "error") o.stderr += (x.traceback ?? [`${x.ename}: ${x.evalue}`]).join("\n").replace(/\u001b\[[0-9;]*m/g, "") + "\n";
      }
      return { type: "code", source: src(c.source), count: c.execution_count ?? null, outputs: o };
    });
}

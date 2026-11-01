// py0 in the page: compile and run programs in a worker (py0-worker.mjs),
// with a way to stop them, and the example programs.

let worker = null;
let nextId = 1;
const jobs = new Map();

function spawn() {
  worker = new Worker(new URL("./py0-worker.mjs", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    const job = jobs.get(data.id);
    if (!job) return;
    if (data.type === "out") {
      for (const [kind, text] of data.chunks) job.onOutput(kind, text);
    } else {
      jobs.delete(data.id);
      job.resolve(data);
    }
  };
}

/**
 * Compile and run `src`. `onOutput(kind, text)` gets "out" / "err" text in
 * order. Resolves to { compileError, code, compileMs, runMs, size, stopped }.
 */
export function runPy0(src, { stdin = "", onOutput = () => {}, compileOnly = false } = {}) {
  if (!worker) spawn();
  const id = nextId++;
  return new Promise((resolve) => {
    jobs.set(id, { resolve, onOutput });
    worker.postMessage({ id, src, stdin, compileOnly });
  });
}

/** Stop whatever is running (the worker is replaced). */
export function stopPy0() {
  if (!worker) return;
  worker.terminate();
  worker = null;
  for (const job of jobs.values()) job.resolve({ stopped: true, code: null });
  jobs.clear();
}

/** Warm up: load the compiler before the first run. */
export const warmPy0 = () => runPy0("pass\n", { compileOnly: true });

/** The 1-based line of a compile error ("line N: ..."), or null. */
export function errorLine(msg) {
  const m = /^line (\d+):/m.exec(msg ?? "");
  return m ? +m[1] : null;
}

let examples = null;
/** The example programs: [{ name, text }]. */
export async function py0Examples() {
  examples ??= fetch(new URL("../py0/examples.json", import.meta.url)).then((r) => r.json());
  return examples;
}

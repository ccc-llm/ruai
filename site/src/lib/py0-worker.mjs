// Compiles and runs py0 programs off the page's thread (so a long or endless
// program can be stopped): the compiler is ruai_py0.wasm, the programs run
// on the host of compiled programs (ruai_wasm.mjs).
import { run } from "../py0/ruai_wasm.mjs";

let compiler = null;

async function load() {
  const r = await WebAssembly.instantiateStreaming(fetch(new URL("../py0/ruai_py0.wasm", import.meta.url)), {});
  compiler = r.instance.exports;
}

function compile(text) {
  const bytes = new TextEncoder().encode(text);
  const p = compiler.alloc(bytes.length);
  new Uint8Array(compiler.memory.buffer, p, bytes.length).set(bytes);
  const status = compiler.compile(p, bytes.length);
  const out = new Uint8Array(compiler.memory.buffer, compiler.out_ptr(), compiler.out_len()).slice();
  return status === 0 ? { wasm: out } : { error: new TextDecoder().decode(out) };
}

self.onmessage = async ({ data: { id, src, stdin, compileOnly } }) => {
  if (!compiler) await load();
  const t0 = performance.now();
  let r;
  try {
    r = compile(src);
  } catch (e) {
    // a trap leaves the compiler's memory in an unknown state: start over
    compiler = null;
    r = { error: "the compiler failed: " + e };
  }
  const compileMs = performance.now() - t0;
  if (r.error || compileOnly) {
    self.postMessage({ id, type: "done", compileError: r.error ?? null, compileMs, size: r.wasm?.length ?? 0 });
    return;
  }
  // output in order, merged by stream and sent in batches
  let batch = [], timer = 0;
  const flush = () => {
    if (batch.length) self.postMessage({ id, type: "out", chunks: batch });
    batch = [];
    timer = 0;
  };
  const push = (kind) => (s) => {
    const last = batch[batch.length - 1];
    if (last && last[0] === kind) last[1] += s;
    else batch.push([kind, s]);
    if (!timer) timer = setTimeout(flush, 30);
  };
  const t1 = performance.now();
  let code;
  try {
    code = await run(r.wasm, { stdout: push("out"), stderr: push("err"), stdin: stdin ?? "" });
  } catch (e) {
    push("err")(String(e?.stack ?? e) + "\n");
    code = 1;
  }
  clearTimeout(timer);
  flush();
  self.postMessage({ id, type: "done", code, compileMs, runMs: performance.now() - t1, size: r.wasm.length });
};

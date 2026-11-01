// Compile and run every py0 example of the site with the built compiler
// (dist/py0), as the IDE does: node test.mjs (after node build.mjs).
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const site = dirname(fileURLToPath(import.meta.url));
const py0 = join(site, "dist/py0");
const { run } = await import(pathToFileURL(join(py0, "ruai_wasm.mjs")).href);
const compiler = (await WebAssembly.instantiate(readFileSync(join(py0, "ruai_py0.wasm")), {})).instance.exports;

function compile(text) {
  const bytes = new TextEncoder().encode(text);
  const p = compiler.alloc(bytes.length);
  new Uint8Array(compiler.memory.buffer, p, bytes.length).set(bytes);
  const status = compiler.compile(p, bytes.length);
  const out = new Uint8Array(compiler.memory.buffer, compiler.out_ptr(), compiler.out_len()).slice();
  return status === 0 ? { wasm: out } : { error: new TextDecoder().decode(out) };
}

const dir = join(site, "src/py0/examples");
let failed = 0;
for (const name of readdirSync(dir).filter((f) => f.endsWith(".py")).sort()) {
  const r = compile(readFileSync(join(dir, name), "utf8"));
  if (r.error) {
    console.log(`FAIL ${name}: ${r.error}`);
    failed++;
    continue;
  }
  let err = "";
  const code = await run(r.wasm, { stdout: () => {}, stderr: (s) => (err += s), stdin: "Ada\n" });
  if (code !== 0) {
    console.log(`FAIL ${name}: exit ${code}\n${err}`);
    failed++;
  } else console.log(`ok   ${name}`);
}
if (failed) {
  console.log(`${failed} failed`);
  process.exit(1);
}

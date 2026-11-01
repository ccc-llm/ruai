// Build the site into dist/ (no npm dependencies):
//   1. site/wasm (the demos) → wasm32 → wasm-bindgen → dist/pkg/
//   2. examples/py0_wasm (the py0 compiler) → dist/py0/ruai_py0.wasm, with
//      the host of compiled programs (src/py/wasm/host.mjs) → dist/py0/ruai_wasm.mjs
//   3. src/ copied to dist/, the py0 examples bundled into dist/py0/examples.json
// Needs cargo with the wasm32-unknown-unknown target and wasm-bindgen-cli 0.2.130.
//   node build.mjs [--skip-wasm]   (--skip-wasm: only copy the pages again)
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const site = dirname(fileURLToPath(import.meta.url));
const root = join(site, "..");
const dist = join(site, "dist");
const skipWasm = process.argv.includes("--skip-wasm");

const sh = (cmd, args, cwd) => {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  execFileSync(cmd, args, { cwd, stdio: "inherit" });
};
const kb = (f) => `${(statSync(f).size / 1024).toFixed(0)} KB`;

if (!skipWasm) {
  rmSync(dist, { recursive: true, force: true });
}
mkdirSync(join(dist, "py0"), { recursive: true });

if (!skipWasm) {
  // 1. the demos
  sh("cargo", ["build", "--release", "--target", "wasm32-unknown-unknown"], join(site, "wasm"));
  sh("wasm-bindgen", ["--target", "web", "--no-typescript", "--out-dir", join(dist, "pkg"), join(site, "wasm/target/wasm32-unknown-unknown/release/ruai_site.wasm")], site);
  // 2. the py0 compiler
  sh("cargo", ["build", "--release", "--example", "py0_wasm", "--target", "wasm32-unknown-unknown", "--no-default-features", "--features", "py"], root);
  cpSync(join(root, "target/wasm32-unknown-unknown/release/examples/py0_wasm.wasm"), join(dist, "py0/ruai_py0.wasm"));
}
cpSync(join(root, "src/py/wasm/host.mjs"), join(dist, "py0/ruai_wasm.mjs"));

// 3. the pages
for (const name of readdirSync(join(site, "src"))) {
  if (name === "py0") continue;
  cpSync(join(site, "src", name), join(dist, name), { recursive: true });
}
const exDir = join(site, "src/py0/examples");
const examples = readdirSync(exDir)
  .filter((f) => f.endsWith(".py"))
  .sort()
  .map((name) => ({ name, text: readFileSync(join(exDir, name), "utf8") }));
writeFileSync(join(dist, "py0/examples.json"), JSON.stringify(examples));
// GitHub Pages: serve files as they are (no Jekyll)
writeFileSync(join(dist, ".nojekyll"), "");

for (const f of ["pkg/ruai_site_bg.wasm", "py0/ruai_py0.wasm"]) {
  const p = join(dist, f);
  if (existsSync(p)) console.log(`${f}: ${kb(p)}`);
}
console.log(`built ${dist}`);

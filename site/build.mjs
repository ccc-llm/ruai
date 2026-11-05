// Build the site into dist/ (no npm dependencies):
//   1. site/wasm (the demos) → wasm32 → wasm-bindgen → dist/pkg/
//   2. rupy's examples/rupy_wasm (the rupy compiler) → dist/rupy/rupy.wasm, with
//      the host of compiled programs (rupy's src/wasm/host.mjs) → dist/rupy/rupy_wasm.mjs
//      and the packages implemented in Rust (`rupy wasm-lib`) → dist/rupy/rupy_lib.wasm
//   3. src/ copied to dist/, the rupy examples bundled into dist/rupy/examples.json
// Needs cargo with the wasm32-unknown-unknown target, wasm-bindgen-cli 0.2.130,
// and a checkout of rupy (https://github.com/ccc-claude-opus/rupy) in
// RUPY_DIR (default: ../rupy next to this repository).
//   node build.mjs [--skip-wasm]   (--skip-wasm: only copy the pages again)
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const site = dirname(fileURLToPath(import.meta.url));
const root = join(site, "..");
const dist = join(site, "dist");
const skipWasm = process.argv.includes("--skip-wasm");
const rupy = process.env.RUPY_DIR ?? join(root, "..", "rupy");
if (!existsSync(join(rupy, "src/wasm/host.mjs"))) {
  console.error(`no rupy checkout in ${rupy}: clone https://github.com/ccc-claude-opus/rupy there or set RUPY_DIR`);
  process.exit(1);
}

const sh = (cmd, args, cwd) => {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  execFileSync(cmd, args, { cwd, stdio: "inherit" });
};
const kb = (f) => `${(statSync(f).size / 1024).toFixed(0)} KB`;

if (!skipWasm) {
  rmSync(dist, { recursive: true, force: true });
}
mkdirSync(join(dist, "rupy"), { recursive: true });

if (!skipWasm) {
  // 1. the demos
  sh("cargo", ["build", "--release", "--target", "wasm32-unknown-unknown"], join(site, "wasm"));
  sh("wasm-bindgen", ["--target", "web", "--no-typescript", "--out-dir", join(dist, "pkg"), join(site, "wasm/target/wasm32-unknown-unknown/release/ruai_site.wasm")], site);
  // 2. the rupy compiler
  sh("cargo", ["build", "--release", "--example", "rupy_wasm", "--target", "wasm32-unknown-unknown", "--no-default-features", "--features", "translator"], rupy);
  cpSync(join(rupy, "target/wasm32-unknown-unknown/release/examples/rupy_wasm.wasm"), join(dist, "rupy/rupy.wasm"));
  // numpy, sklearn, torch, ... for the programs (loaded only by those that import them)
  sh("cargo", ["run", "--release", "--quiet", "--bin", "rupy", "--", "wasm-lib", "-o", join(dist, "rupy/rupy_lib.wasm")], rupy);
}
cpSync(join(rupy, "src/wasm/host.mjs"), join(dist, "rupy/rupy_wasm.mjs"));

// 3. the pages
for (const name of readdirSync(join(site, "src"))) {
  if (name === "rupy") continue;
  cpSync(join(site, "src", name), join(dist, name), { recursive: true });
}
const exDir = join(site, "src/rupy/examples");
const examples = readdirSync(exDir)
  .filter((f) => f.endsWith(".py"))
  .sort()
  .map((name) => ({ name, text: readFileSync(join(exDir, name), "utf8") }));
writeFileSync(join(dist, "rupy/examples.json"), JSON.stringify(examples));
// GitHub Pages: serve files as they are (no Jekyll)
writeFileSync(join(dist, ".nojekyll"), "");

for (const f of ["pkg/ruai_site_bg.wasm", "rupy/rupy.wasm", "rupy/rupy_lib.wasm"]) {
  const p = join(dist, f);
  if (existsSync(p)) console.log(`${f}: ${kb(p)}`);
}
console.log(`built ${dist}`);

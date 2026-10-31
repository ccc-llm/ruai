// The py0 playground: compile with ruai_py0.wasm (the translator as wasm),
// run the result with the host ruai_wasm.mjs.
import { run } from "./ruai_wasm.mjs";

const EXAMPLES = {
  "Hello": `def greet(name: str, times: int = 2) -> str:
    return ", ".join([f"hello {name}"] * times)


print(greet("wasm"))
print([n * n for n in range(10) if n % 2 == 0])
`,
  "Classes": `from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

    def norm(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5


class Shape:
    def area(self) -> float:
        return 0.0

    def __str__(self) -> str:
        return "Shape"


class Rect(Shape):
    def __init__(self, w: float, h: float) -> None:
        self.w = w
        self.h = h

    def area(self) -> float:
        return self.w * self.h

    def __str__(self) -> str:
        return f"Rect({self.w} x {self.h})"


p = Point(3.0, 4.0)
print(p, p.norm())
shapes: list[Shape] = [Rect(2.0, 3.5), Rect(1.0, 1.0)]
for s in shapes:
    print(s, f"area={s.area():.2f}")
`,
  "Words": `text = """the quick brown fox jumps over the lazy dog
the dog barks and the fox runs"""
counts: dict[str, int] = {}
for w in text.split():
    counts[w] = counts.get(w, 0) + 1
top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
for word, n in top:
    print(f"{word:>6} {'#' * n} {n}")
print(len(counts), "distinct words;", sum(counts.values()), "in all")
`,
  "Exceptions": `def parse(s: str) -> int:
    try:
        return int(s)
    except ValueError as e:
        print("bad input:", e)
        return -1
    finally:
        print("parsed", repr(s))


print(parse("42"), parse("4x2"))
xs = [1, 2, 3]
try:
    print(xs[10])
except IndexError as e:
    print("IndexError:", e)
print(10 // 3, -7 % 3, 2 ** 62, 1 / 3)
`,
  "Primes": `def primes(n: int) -> list[int]:
    sieve = [True] * (n + 1)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, p in enumerate(sieve) if p]


ps = primes(200000)
print(len(ps), ps[:10], ps[-3:])
`,
};

const $ = (id) => document.getElementById(id);
const src = $("src"), out = $("stdout"), errs = $("stderr"), stats = $("stats"), btn = $("run"), pick = $("example");
for (const name of Object.keys(EXAMPLES)) pick.add(new Option(name, name));
let saved = null;
try {
  saved = localStorage.getItem("py0.src");
} catch {}
src.value = saved ?? EXAMPLES.Hello;
pick.onchange = () => {
  src.value = EXAMPLES[pick.value];
  go();
};
src.addEventListener("keydown", (e) => {
  if (e.key === "Tab") {
    e.preventDefault();
    src.setRangeText("    ", src.selectionStart, src.selectionEnd, "end");
  } else if ((e.ctrlKey || e.metaKey) && e.key === "Enter") go();
});

const compiler = (await WebAssembly.instantiateStreaming(fetch("ruai_py0.wasm"), {})).instance.exports;

function compile(text) {
  const bytes = new TextEncoder().encode(text);
  const p = compiler.alloc(bytes.length);
  new Uint8Array(compiler.memory.buffer, p, bytes.length).set(bytes);
  const status = compiler.compile(p, bytes.length);
  const result = new Uint8Array(compiler.memory.buffer, compiler.out_ptr(), compiler.out_len()).slice();
  return status === 0 ? { wasm: result } : { error: new TextDecoder().decode(result) };
}

async function go() {
  try {
    localStorage.setItem("py0.src", src.value);
  } catch {}
  out.textContent = "";
  errs.textContent = "";
  btn.disabled = true;
  const t0 = performance.now();
  let r;
  try {
    r = compile(src.value);
  } catch (e) {
    r = { error: "the compiler failed: " + e };
  }
  const t1 = performance.now();
  if (r.error) {
    errs.textContent = r.error;
    stats.textContent = `compile error (${(t1 - t0).toFixed(0)} ms)`;
    btn.disabled = false;
    return;
  }
  let text = "";
  const code = await run(r.wasm, {
    stdout: (s) => (text += s),
    stderr: (s) => (errs.textContent += s),
    stdin: $("stdin").value,
  });
  out.textContent = text;
  const t2 = performance.now();
  stats.textContent = `compiled in ${(t1 - t0).toFixed(0)} ms (${r.wasm.length} bytes) · ran in ${(t2 - t1).toFixed(0)} ms · exit ${code}`;
  btn.disabled = false;
}

btn.onclick = go;
btn.disabled = false;
stats.textContent = "ready (Ctrl/⌘ + Enter runs)";
go();

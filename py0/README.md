# py0 (py2rs)

> **py0 is part of [ruai](../README.md)** (it was the standalone project py2rs): install `ruai` and use `ruai py ...`. Translated programs depend on the `ruai` crate with only its runtime feature (`pyrt`).

English · [繁體中文](README.tw.md) · [简体中文](README.cn.md)

A translator from a typed subset of Python to readable Rust.

```python
@dataclass
class Item:
    name: str
    price: float

def total(items: list[Item], tax: float = 0.05) -> float:
    return sum(i.price for i in items) * (1 + tax)
```

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub name: String,
    pub price: f64,
}

fn total(items: &[Item], tax: f64) -> f64 {
    pyrt::sum_f64(items.iter().map(|i| i.price)) * (1.0 + tax)
}
```

## Promises

1. **Every value has a static type.** Functions are annotated; locals are inferred.
2. **No observable sharing.** Python shares mutable objects between names; Rust copies,
   moves or borrows them. py2rs accepts a program only when the difference cannot be
   seen, and infers `&T` / `&mut T` / `T` for every parameter. (`@shared` classes keep
   Python's sharing, as `Rc<RefCell<..>>`.)
3. **Same output as CPython, or a clear error.** Anything that cannot be guaranteed is
   rejected at translation time with the line and a hint, never translated differently.
4. **Rust crates can be used directly** (`from rust.regex import Regex`).

## Install and use

```sh
cargo install ruai             # or: cargo build --release in a checkout
ruai py emit  prog.py            # print the Rust program
ruai py check prog.py            # types + ownership only
ruai py explain prog.py          # parameter modes, references, copies, Result functions
ruai py build prog.py [-o DIR]   # Cargo project, release build
ruai py run   prog.py            # build (cached) and run
ruai py pyo3  mod.py [-o DIR]    # a Python extension module (PyO3): DIR/mod.so
ruai py test                     # differential test against CPython (a checkout)
ruai py fuzz 300                 # random programs against CPython
```

A program may import its own modules (`import geometry`, `from geometry import Point`
with `geometry.py` next to it).

## What is supported

The full definition is [SPEC.md](SPEC.md). In short: `int` (64-bit, overflow is an
error), `float`, `bool`, `str`, `list`, `dict`, `set`, `tuple`, `Optional`; functions
with defaults, keywords and `*args`; dataclasses and classes with methods, properties,
class methods, operator methods and single inheritance; exceptions (also with fields)
and `try` / `finally`; comprehensions, f-strings, `match`, `with open`, generators,
nested functions with `nonlocal`, lambdas and `Callable`; generic functions, `Protocol`
and union parameters (compiled per argument type); `math`, `collections`, `heapq`,
`bisect`, `itertools`, `functools`, `random` (same numbers as CPython for the same
seed), `re`, `sys`, `time`, `json.dumps`, `os.path`.

How some constructs come out (more in [docs/cases](docs/cases/README.md)):

| Python | Rust |
|---|---|
| `x: T \| None`, `if x is not None:` | `Option<T>`, `if let Some(x) = &x` |
| `node.next = self.head; self.head = node` | `node.next = self.head.take();` |
| a local that only reads (`cur = cur.next`) | a reference: `let mut cur = &self.head;` |
| `item = self.items[k]; item.qty -= 1` | `let item = &mut self.items[k];` |
| a class with subclasses | an `enum` over the created classes, methods dispatch with `match` |
| a `raise` that a caller catches | `Result<T, PyErr>` on that path only, `?` |
| a nested function using `seen` and `nonlocal count` | `fn dfs(.., seen: &mut HashSet<..>, count: &mut i64)`, the variables passed at each call |
| `def first[T](xs: list[T]) -> T` | `first_int(xs: &[i64])`, `first_str(xs: &[String])`, one per argument type |
| `def gen(n) -> Iterator[int]: ... yield i` | `pyrt::Gen<i64>`: the body runs strictly in turn with its consumer |
| `@shared class Node` | `struct Node(Rc<RefCell<NodeData>>)` with getters and setters |

Known differences from CPython: `int` is 64-bit (results beyond it raise
`OverflowError`); set iteration order is insertion order ([TODO.md](TODO.md)).

## Speed

`python3 bench/run.py`, CPython 3.14 against the translated program (Apple silicon):

| program | CPython | Rust | |
|---|---:|---:|---:|
| nbody (float objects) | 1.21 s | 0.012 s | 101x |
| matmul (lists of floats) | 0.47 s | 0.013 s | 38x |
| fib (recursion) | 0.26 s | 0.013 s | 20x |
| sieve | 0.48 s | 0.027 s | 18x |
| graph (BFS with deque) | 0.14 s | 0.017 s | 8x |
| objects (dataclasses, sort, dict) | 0.59 s | 0.10 s | 6x |
| printing | 0.21 s | 0.056 s | 4x |
| strings, wordfreq | | | 2x |

## Ownership annotations

Parameter modes are inferred. To pin one, annotate it (the module `python/rust` makes the
annotations work under CPython too; put that directory on `PYTHONPATH`):

```python
from rust import Ref, Mut, Own

def fill(xs: Mut[list[int]], n: int) -> None: ...   # &mut Vec<i64>
def keep(xs: Own[list[int]]) -> None: ...           # Vec<i64>
```

An annotation the code contradicts (modifying a `Ref[...]`) is an error.

## Packages and the standard library (ruai)

`import M`, `import M as A` and `from M import x` also find **py0 packages**: stubs `M.pyi`
in ruai's `src/py/lib/` (or in a directory listed in `PY0_LIB`), whose declarations map to Rust
under `ruai::pyrt::lib::M`. A stub declares classes, functions and constants in typed Python;
a parameter typed `A | B` makes overloads (a generic Rust function), several `def`s of one
name are overloads, `@rust("...")` gives a Rust path or an expression template, and
`Result[T, OSError | ...]` raises CPython's exceptions.

- **The standard library**, module by module, each checked against CPython
  (`tests/py0/py/std_*.py`): `shlex`, `fnmatch`, `html`, `subprocess`, more of `os` /
  `os.path` / `sys` / `time`, besides the modules built into the translator (`math`,
  `collections`, `heapq`, `bisect`, `itertools`, `functools`, `random`, `re`, `sys`, `time`,
  `json.dumps`, `os.path`).
- **numpy**: float64 / float32 / int64 / bool arrays with NumPy 2's promotion,
  broadcasting, views, masks and index arrays, reductions in NumPy's summation order,
  `linalg`, `random.default_rng`; printing identical to NumPy's.
- **sklearn**: linear and logistic regression, ridge, k-NN, Gaussian naive Bayes, KMeans,
  PCA, scalers, `train_test_split` (the same split as scikit-learn), metrics, `load_iris`.
- **gymnasium**: ruai's environments with Gymnasium's API, bit for bit.
- **torch**: tensors with autograd on ruai's nn engine, `nn` modules, losses, `optim`,
  `torch.no_grad()`, and PyTorch's random numbers for `torch.manual_seed`.
- **ruai.json**, **ruai.http**: a typed JSON value and an HTTP(S) client, with CPython
  stand-ins in `python/ruai/`.

Under CPython the same programs import the real modules; `tests/py0/numpy`, `sklearn`,
`gymnasium` and `torch` compare the two.

```python
import shlex
from fnmatch import fnmatch

import numpy as np

words: list[str] = shlex.split("cp 'my file.txt' backup/")
print(words, fnmatch("my file.txt", "*.txt"), np.arange(6).reshape(2, 3).sum(axis=0))
```

## Python libraries (the Python bridge)

A stub whose first line is `# python: MODULE` makes its declarations calls into the real
Python module, through CPython embedded in the program (PyO3; runtime feature
`pyrt-python`). The stub types the module for py0; values convert both ways (int, float,
bool, str, None, list, dict, tuple, numpy arrays, and `Object`, an opaque Python object).
A default written `...` leaves the argument out, so Python's default applies. Every call may
raise, and Python's exceptions keep their names. Classes of the stub are Python objects
with their methods, properties, operators and static methods.

```python
# py0lib/statistics.pyi
# python: statistics
def mean(data: list[float] | list[int]) -> float: ...
```

`ruai py stub --python MODULE [-o FILE]` drafts a stub with `inspect` (default
`py0lib/MODULE.pyi`); unannotated parameters take the type of their default, anything else
unknown is `Object`, which accepts any value. Stubs are found next to the program in
`py0lib/`, in `./py0lib`, in `PY0_LIB` and in ruai's `src/py/lib/` (which has
`matplotlib` and `matplotlib.pyplot`). The program links the first Python that imports
its modules (`PY0_PYTHON`, else python3.14 / 3.13 / 3.12 / python3) and runs with its
packages, so that Python must be installed where the program runs.

## Rust crates

`from rust.CRATE import Name` uses a Rust crate. Its declarations come from a stub,
`CRATE.pyi`, looked up in `./stubs`, `PY2RS_STUBS` and the stubs shipped with py2rs
(`stubs/regex.pyi`). The crate becomes a Cargo dependency of the generated project.

```python
from rust.regex import Regex

date = Regex.new(r"(\d{4})-(\d{2})-(\d{2})")      # Result → ValueError when invalid
print(date.replace_all("on 2024-05-17", "$3/$2/$1"))
```

A stub is a Python file of typed declarations; each maps to a Rust path or an expression
template (`@rust("{self}.find({0}).map(|m| m.as_str().to_string())")`). `ruai py stub
CRATE[@VERSION]` writes one from the crate's rustdoc JSON. `from rust import inline` is
the escape hatch: `inline("xs.sort_unstable();")`, `inline[int]("n.count_ones() as i64")`.

## Python extension modules

`ruai py pyo3 mod.py` builds `mod.so`, which Python imports like the original module. Its
module-level functions are exported (int, float, bool, str, list, tuple, dict and
`None` arguments and results); an exception they raise is raised in Python with the same
class; `print` writes to Python's `sys.stdout`. `tests/pyo3/run.sh` checks that importing
the module prints what importing the Python file prints.

## Testing

`ruai py test [FILES|DIRS]` (default `tests/`) translates every program, builds them all
in one Cargo package and compares stdout, exit code and exception type with CPython
(`PY2RS_PYTHON`, else python3.14 / 3.13 / 3.12); a program that imports numpy, sklearn,
gymnasium, torch or a bridged module runs on the first of those (or `python3`) that has them, and is skipped
when none does. Input comes from `NAME.in` if present. A program whose first line is
`# error: TEXT` must be rejected with a message containing TEXT (`tests/err/`). The corpus
has about 260 programs and 46 rejections; `ruai py fuzz` adds random programs.

## Layout

- `src/front.rs` (+ `src/front/`): ruff parser AST → typed IR (`src/ir.rs`); the
  standard library, `re`, operator methods, nested functions, `match`, generics
- `src/own.rs`: ownership inference, liveness, aliasing and iteration checks, views
- `src/exc.rs`: which exceptions can be caught where, so which functions return `Result`
- `src/gen.rs` (+ `src/gen/`): IR → Rust
- `src/modules.rs`: programs of several files; `src/pyo3.rs`: extension modules
- `pyrt/`: runtime: Python printing, format specs, `//` `%` `**` semantics, `str`
  methods by character, exceptions, the standard library, generators, buffered output

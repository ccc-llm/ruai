#!/bin/sh
# Builds tests/py0/pyo3/fastmath.py as an extension module (ruai py pyo3) and checks that
# importing it prints what importing the Python file prints.
set -e
cd "$(dirname "$0")/../../.."
PY=${PY2RS_PYTHON:-python3.14}
OUT=${PYO3_OUT:-target/py0-cache/pyo3}
./target/release/ruai py pyo3 tests/py0/pyo3/fastmath.py -o "$OUT" >/dev/null
CHECK='
import fastmath
print(fastmath.fib(90), fastmath.mean([1.0, 2.0, 4.0]))
print(fastmath.word_counts("a b a c b a"))
print(fastmath.greet("ann", 2))
try:
    fastmath.mean([])
except ValueError as e:
    print("ValueError:", e)
'
PYTHONPATH="$OUT" $PY -c "$CHECK" > "$OUT/rust.txt"
PYTHONPATH=tests/py0/pyo3 $PY -c "$CHECK" > "$OUT/python.txt"
diff "$OUT/python.txt" "$OUT/rust.txt"
echo "pyo3 module: same output as CPython"

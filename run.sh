#!/bin/sh
# Build, serve and test the ruai website (site/): every task in one place.
#   ./run.sh <command> [args]
#
#   setup          install what the build needs (the wasm32 target, wasm-bindgen-cli 0.2.130)
#   check          show whether the tools are installed
#   build          build everything into site/dist (the wasm modules and the pages)
#   pages          copy the pages again, without rebuilding the wasm (after editing site/src)
#   serve [port]   serve site/dist (default port 8000)
#   dev [port]     build, then serve
#   watch [port]   serve, and copy the pages again whenever site/src changes
#   test           compile and run every py0 example with the built compiler
#   size           sizes of the built files
#   clean          remove site/dist and the site crate's build directory
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
SITE="$ROOT/site"
BINDGEN=0.2.130

usage() {
  sed -n '2,/^[^#]/p' "$ROOT/run.sh" | grep '^#' | sed 's/^# \{0,1\}//'
}

need_dist() {
  if [ ! -f "$SITE/dist/index.html" ]; then
    echo "site/dist is missing: run './run.sh build' first" >&2
    exit 1
  fi
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift
case "$cmd" in
  setup)
    rustup target add wasm32-unknown-unknown
    if wasm-bindgen --version 2>/dev/null | grep -q " $BINDGEN"; then
      echo "wasm-bindgen $BINDGEN is installed"
    else
      cargo install wasm-bindgen-cli --version "$BINDGEN" --locked
    fi
    ;;
  check)
    ok=1
    for t in cargo rustup node; do
      if command -v "$t" >/dev/null 2>&1; then echo "ok       $t"; else echo "missing  $t"; ok=0; fi
    done
    if rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
      echo "ok       wasm32-unknown-unknown target"
    else
      echo "missing  wasm32-unknown-unknown target (./run.sh setup)"; ok=0
    fi
    if wasm-bindgen --version 2>/dev/null | grep -q " $BINDGEN"; then
      echo "ok       wasm-bindgen $BINDGEN"
    else
      echo "missing  wasm-bindgen $BINDGEN (./run.sh setup)"; ok=0
    fi
    [ "$ok" = 1 ] || exit 1
    ;;
  build)
    node "$SITE/build.mjs"
    ;;
  pages)
    node "$SITE/build.mjs" --skip-wasm
    ;;
  serve)
    need_dist
    node "$SITE/serve.mjs" "${1:-8000}"
    ;;
  dev)
    node "$SITE/build.mjs"
    node "$SITE/serve.mjs" "${1:-8000}"
    ;;
  watch)
    need_dist
    node "$SITE/serve.mjs" "${1:-8000}" &
    server=$!
    trap 'kill $server 2>/dev/null' EXIT INT TERM
    last=""
    while :; do
      now="$(find "$SITE/src" -type f -exec stat -f '%m %N' {} + 2>/dev/null || find "$SITE/src" -type f -exec stat -c '%Y %n' {} +)"
      if [ "$now" != "$last" ]; then
        [ -n "$last" ] && node "$SITE/build.mjs" --skip-wasm | tail -1
        last="$now"
      fi
      sleep 1
    done
    ;;
  test)
    need_dist
    node "$SITE/test.mjs"
    ;;
  size)
    need_dist
    (cd "$SITE/dist" && find . -type f \( -name '*.wasm' -o -name '*.js' -o -name '*.mjs' -o -name '*.json' \) -exec ls -l {} + | awk '{printf "%8.0f KB  %s\n", $5 / 1024, $9}' | sort -rn)
    du -sh "$SITE/dist"
    ;;
  clean)
    rm -rf "$SITE/dist" "$SITE/wasm/target"
    echo "removed site/dist and site/wasm/target"
    ;;
  help | -h | --help)
    usage
    ;;
  *)
    echo "unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac

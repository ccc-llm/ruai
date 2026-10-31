// The host of py0 programs compiled to WebAssembly (`ruai py wasm`).
//
// In a browser:  import { run } from "./ruai_wasm.mjs";
//                const code = await run(bytes, { stdout: s => ..., stderr: s => ..., stdin: "..." });
// On the command line (Node.js 22+):  node ruai_wasm.mjs prog.wasm < input
//
// The module calls the host through a small channel: arguments are pushed one
// by one (strings through the exported memory), `call` / `fcall` run a named
// operation, and strings come back through `pending` + `take`. The operations
// follow CPython: float repr and format specs (exact decimal rounding), int()
// and float() parsing, math with its domain errors, str methods.

const KINDS = ["", "ValueError", "TypeError", "OverflowError", "IndexError", "EOFError", "ZeroDivisionError", "KeyError"];
const enc = new TextEncoder();
const dec = new TextDecoder();

class PyErr {
  constructor(kind, msg) {
    this.kind = kind;
    this.msg = msg;
  }
}
const err = (kind, msg) => new PyErr(kind, msg);

// ------------------------------------------------------------ numbers

const I64_MIN = -(2n ** 63n), I64_MAX = 2n ** 63n - 1n;
const OVERFLOW = "integer overflow (py2rs: int is a 64-bit integer)";

/** repr(float) */
function floatRepr(x) {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const [m, e] = Math.abs(x).toExponential().split("e");
  const exp = parseInt(e, 10);
  const digits = m.replace(".", "");
  const sign = x < 0 ? "-" : "";
  if (exp < -4 || exp >= 16) {
    const mm = digits.length > 1 ? digits[0] + "." + digits.slice(1) : digits;
    return sign + mm + "e" + (exp < 0 ? "-" : "+") + String(Math.abs(exp)).padStart(2, "0");
  }
  let s;
  if (exp >= 0) {
    s = digits.length <= exp + 1 ? digits + "0".repeat(exp + 1 - digits.length) + ".0" : digits.slice(0, exp + 1) + "." + digits.slice(exp + 1);
  } else {
    s = "0." + "0".repeat(-exp - 1) + digits;
  }
  return sign + s;
}

/** x (finite, > 0) = D * 10^k exactly */
function decompose(x) {
  const v = new DataView(new ArrayBuffer(8));
  v.setFloat64(0, x);
  const bits = v.getBigUint64(0);
  const ex = Number((bits >> 52n) & 0x7ffn);
  const frac = bits & ((1n << 52n) - 1n);
  const m = ex === 0 ? frac : frac | (1n << 52n);
  const e = ex === 0 ? -1074 : ex - 1075;
  return e >= 0 ? [m << BigInt(e), 0] : [m * 5n ** BigInt(-e), e];
}

/** n / d rounded half to even (n ≥ 0, d > 0) */
function roundDiv(n, d) {
  let q = n / d;
  const r2 = (n % d) * 2n;
  if (r2 > d || (r2 === d && q % 2n === 1n)) q += 1n;
  return q;
}

/** |x| with `prec` decimals, correctly rounded: [integer digits, fraction digits] */
function fixed(x, prec) {
  if (x === 0) return ["0", "0".repeat(prec)];
  const [D, k] = decompose(Math.abs(x));
  const scale = k + prec;
  const N = scale >= 0 ? D * 10n ** BigInt(scale) : roundDiv(D, 10n ** BigInt(-scale));
  const s = N.toString().padStart(prec + 1, "0");
  return [s.slice(0, s.length - prec), s.slice(s.length - prec)];
}

/** |x| to `prec` + 1 significant digits: [digits, decimal exponent] */
function sci(x, prec) {
  if (x === 0) return ["0".repeat(prec + 1), 0];
  const [D, k] = decompose(Math.abs(x));
  let exp = D.toString().length - 1 + k;
  const scale = k - exp + prec;
  let N = scale >= 0 ? D * 10n ** BigInt(scale) : roundDiv(D, 10n ** BigInt(-scale));
  if (N.toString().length > prec + 1) {
    N /= 10n;
    exp += 1;
  }
  return [N.toString(), exp];
}

/** round(x, n) as Python: the decimal value nearest to x, half to even */
function roundTo(x, n) {
  if (!Number.isFinite(x) || x === 0) return x;
  if (n >= 0) {
    if (n > 330) return x;
    const [i, f] = fixed(x, n);
    const r = parseFloat(i + "." + (f || "0"));
    return x < 0 ? -r : r;
  }
  const [D, k] = decompose(Math.abs(x));
  const scale = k + n;
  const N = scale >= 0 ? D * 10n ** BigInt(scale) : roundDiv(D, 10n ** BigInt(-scale));
  const r = parseFloat(N.toString() + "e" + -n);
  return x < 0 ? -r : r;
}

function group(intPart, sep, every) {
  if (!sep) return intPart;
  let out = "";
  for (let i = 0; i < intPart.length; i++) {
    if (i > 0 && (intPart.length - i) % every === 0) out += sep;
    out += intPart[i];
  }
  return out;
}

const SPEC = /^(?:(.)?([<>=^]))?([-+ ])?(z)?(#)?(0)?(\d+)?([,_])?(?:\.(\d+))?([a-zA-Z%])?$/su;

function pad(body, sign, prefix, f, numeric) {
  const width = f.width;
  let fill = f.fill, align = f.align;
  if (f.zero && !align) {
    fill = fill ?? "0";
    align = "=";
  }
  fill = fill ?? " ";
  align = align ?? (numeric ? ">" : "<");
  const len = [...(sign + prefix + body)].length;
  if (width <= len) return sign + prefix + body;
  const n = width - len;
  switch (align) {
    case "<":
      return sign + prefix + body + fill.repeat(n);
    case ">":
      return fill.repeat(n) + sign + prefix + body;
    case "^":
      return fill.repeat(n >> 1) + sign + prefix + body + fill.repeat(n - (n >> 1));
    default:
      return sign + prefix + fill.repeat(n) + body;
  }
}

function parseSpec(spec, typeName) {
  const m = SPEC.exec(spec);
  if (!m) throw err("ValueError", "Invalid format specifier '" + spec + "' for object of type '" + typeName + "'");
  return { fill: m[1], align: m[2], sign: m[3] ?? "-", signed: m[3] !== undefined, z: !!m[4], alt: !!m[5], zero: !!m[6], width: m[7] ? parseInt(m[7], 10) : 0, group: m[8], prec: m[9] !== undefined ? parseInt(m[9], 10) : null, type: m[10] ?? "" };
}

const signOf = (neg, f) => (neg ? "-" : f.sign === "+" ? "+" : f.sign === " " ? " " : "");

function formatFloat(x, f, typeName = "float") {
  let t = f.type;
  if (!"eEfFgGn%".includes(t) && t !== "") throw err("ValueError", "Unknown format code '" + t + "' for object of type '" + typeName + "'");
  let neg = x < 0 || Object.is(x, -0);
  let body;
  const upper = "EFG".includes(t);
  if (Number.isNaN(x)) {
    neg = false;
    body = upper ? "NAN" : "nan";
  } else if (!Number.isFinite(x)) {
    body = upper ? "INF" : "inf";
  } else {
    const ax = Math.abs(x);
    let prec = f.prec;
    if (t === "") {
      if (prec === null) {
        body = floatRepr(ax);
      } else {
        body = general(ax, prec, f.alt);
        if (!body.includes(".") && !body.includes("e")) body += ".0";
      }
    } else if (t === "f" || t === "F" || t === "%") {
      const v = t === "%" ? ax * 100 : ax;
      const [i, fr] = fixed(v, prec ?? 6);
      body = group(i, f.group, 3) + (fr || f.alt ? "." + fr : "") + (t === "%" ? "%" : "");
      if (f.z && /^[0.]*%?$/.test(i + fr)) neg = false;
    } else if (t === "e" || t === "E") {
      const p = prec ?? 6;
      const [d, e] = sci(ax, p);
      body = d[0] + (p > 0 || f.alt ? "." + d.slice(1) : "") + (t === "E" ? "E" : "e") + (e < 0 ? "-" : "+") + String(Math.abs(e)).padStart(2, "0");
    } else {
      body = general(ax, prec ?? 6, f.alt);
      if (t === "G") body = body.toUpperCase();
    }
    if (f.group && t !== "f" && t !== "F" && t !== "%") {
      const m = /^(\d+)(.*)$/s.exec(body);
      if (m) body = group(m[1], f.group, 3) + m[2];
    }
    if (f.z && Number(body.replace(/[^\d.e+-]/g, "")) === 0) neg = false;
  }
  return pad(body, signOf(neg, f), "", f, true);
}

/** 'g' formatting of |x| with precision p */
function general(ax, p, alt) {
  if (p === 0) p = 1;
  const [d, e] = sci(ax, p - 1);
  let s;
  if (e >= -4 && e < p) {
    const [i, fr] = fixed(ax, p - 1 - e);
    s = i + (fr ? "." + fr : "");
    if (!alt && s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
    if (alt && !s.includes(".")) s += ".";
  } else {
    let m = d[0] + (d.length > 1 ? "." + d.slice(1) : "");
    if (!alt && m.includes(".")) m = m.replace(/0+$/, "").replace(/\.$/, "");
    if (alt && !m.includes(".")) m += ".";
    s = m + "e" + (e < 0 ? "-" : "+") + String(Math.abs(e)).padStart(2, "0");
  }
  return s;
}

function formatInt(x, f) {
  const t = f.type;
  if ("eEfFgG%".includes(t) && t) return formatFloat(Number(x), f, "int");
  if (!"bcdoxXn".includes(t) && t !== "") throw err("ValueError", "Unknown format code '" + t + "' for object of type 'int'");
  if (f.prec !== null) throw err("ValueError", "Precision not allowed in integer format specifier");
  if (t === "c") return pad(String.fromCodePoint(Number(x)), "", "", f, true);
  const neg = x < 0n;
  const ax = neg ? -x : x;
  const base = { b: 2, o: 8, x: 16, X: 16 }[t] ?? 10;
  let body = ax.toString(base);
  if (t === "X") body = body.toUpperCase();
  body = group(body, f.group, base === 10 ? 3 : 4);
  const prefix = f.alt && base !== 10 ? "0" + (t === "X" ? "X" : t) : "";
  return pad(body, signOf(neg, f), prefix, f, true);
}

function formatStr(s, f) {
  if (f.type !== "" && f.type !== "s") throw err("ValueError", "Unknown format code '" + f.type + "' for object of type 'str'");
  if (f.align === "=") throw err("ValueError", "'=' alignment not allowed in string format specifier");
  if (f.signed) throw err("ValueError", "Sign not allowed in string format specifier");
  let body = s;
  if (f.prec !== null) body = [...body].slice(0, f.prec).join("");
  return pad(body, "", "", f, false);
}

function format(v, spec) {
  if (spec === "") return typeof v === "number" ? floatRepr(v) : String(v);
  if (typeof v === "bigint") return formatInt(v, parseSpec(spec, "int"));
  if (typeof v === "number") return formatFloat(v, parseSpec(spec, "float"));
  return formatStr(v, parseSpec(spec, "str"));
}

const pyRepr = (s) => {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = q;
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (ch === "\\" || ch === q) out += "\\" + ch;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (c < 32 || c === 127 || (c >= 0x80 && c < 0xa0)) out += "\\x" + c.toString(16).padStart(2, "0");
    else out += ch;
  }
  return out + q;
};

function parseInt64(s, base) {
  const orig = s;
  const bad = () => err("ValueError", "invalid literal for int() with base " + base + ": " + pyRepr(orig));
  s = s.replace(/^[\s]+|[\s]+$/gu, "");
  let neg = false;
  if (s[0] === "+" || s[0] === "-") {
    neg = s[0] === "-";
    s = s.slice(1);
  }
  let b = base;
  const pre = s.slice(0, 2).toLowerCase();
  if ((b === 16 && pre === "0x") || (b === 8 && pre === "0o") || (b === 2 && pre === "0b")) s = s.slice(2).replace(/^_/, "");
  else if (b === 0) {
    if (pre === "0x") (b = 16), (s = s.slice(2).replace(/^_/, ""));
    else if (pre === "0o") (b = 8), (s = s.slice(2).replace(/^_/, ""));
    else if (pre === "0b") (b = 2), (s = s.slice(2).replace(/^_/, ""));
    else {
      b = 10;
      if (/^0+[1-9]/.test(s)) throw bad();
    }
  }
  if (!s || s.startsWith("_") || s.endsWith("_") || s.includes("__")) throw bad();
  s = s.replace(/_/g, "");
  let v = 0n;
  const B = BigInt(b);
  for (const ch of s) {
    let d;
    const c = ch.codePointAt(0);
    if (c >= 48 && c <= 57) d = c - 48;
    else if (c >= 97 && c <= 122) d = c - 87;
    else if (c >= 65 && c <= 90) d = c - 55;
    else {
      // other Unicode decimal digits
      const n = /\p{Nd}/u.test(ch) ? Number.parseInt(ch.normalize("NFKD"), 10) : NaN;
      if (Number.isNaN(n)) throw bad();
      d = n;
    }
    if (d >= b) throw bad();
    v = v * B + BigInt(d);
  }
  if (neg) v = -v;
  if (v < I64_MIN || v > I64_MAX) throw err("OverflowError", OVERFLOW);
  return v;
}

function parseFloat64(s) {
  const t = s.trim().replace(/(\d)_(?=\d)/g, "$1");
  const l = t.toLowerCase();
  const m = /^([+-]?)(inf|infinity|nan)$/.exec(l);
  if (m) return m[2] === "nan" ? NaN : m[1] === "-" ? -Infinity : Infinity;
  if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(t)) throw err("ValueError", "could not convert string to float: " + pyRepr(s));
  return Number(t);
}

// ------------------------------------------------------------ math

const domain = () => err("ValueError", "math domain error");
const range = () => err("OverflowError", "math range error");

function checked(r, ...xs) {
  if (Number.isNaN(r) && !xs.some(Number.isNaN)) throw domain();
  if (!Number.isFinite(r) && !Number.isNaN(r) && xs.every(Number.isFinite)) throw range();
  return r;
}

const MATH = {
  pow: (a, b) => (a === 1 || b === 0 ? 1 : Math.pow(a, b)),
  fmod: (a, b) => a % b,
  round: (x, n) => roundTo(x, n),
  sin: (x) => checked(Math.sin(x), x),
  cos: (x) => checked(Math.cos(x), x),
  tan: (x) => checked(Math.tan(x), x),
  asin: (x) => checked(Math.asin(x), x),
  acos: (x) => checked(Math.acos(x), x),
  atan: (x) => Math.atan(x),
  atan2: (y, x) => Math.atan2(y, x),
  sinh: (x) => checked(Math.sinh(x), x),
  cosh: (x) => checked(Math.cosh(x), x),
  tanh: (x) => Math.tanh(x),
  exp: (x) => checked(Math.exp(x), x),
  expm1: (x) => checked(Math.expm1(x), x),
  log: (x) => {
    if (x <= 0) throw domain();
    return Math.log(x);
  },
  log2: (x) => {
    if (x <= 0) throw domain();
    return Math.log2(x);
  },
  log10: (x) => {
    if (x <= 0) throw domain();
    return Math.log10(x);
  },
  log1p: (x) => {
    if (x <= -1) throw domain();
    return Math.log1p(x);
  },
  log_base: (x, b) => {
    if (x <= 0 || b <= 0) throw domain();
    const d = Math.log(b);
    if (d === 0) throw err("ZeroDivisionError", "division by zero");
    return Math.log(x) / d;
  },
  hypot: (a, b) => Math.hypot(a, b),
  cbrt: (x) => Math.cbrt(x),
  degrees: (x) => x * (180 / Math.PI),
  radians: (x) => x * (Math.PI / 180),
};

// ------------------------------------------------------------ str methods

const WS = /[\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/u;
const isWs = (c) => WS.test(c);
const cps = (s) => [...s];
const num = (v) => Number(v);

function stripChars(s, chars, left, right) {
  const a = cps(s);
  const test = chars === undefined ? isWs : (c) => chars.includes(c);
  let i = 0, j = a.length;
  if (left) while (i < j && test(a[i])) i++;
  if (right) while (j > i && test(a[j - 1])) j--;
  return a.slice(i, j).join("");
}

function splitWs(s, max = -1) {
  const out = [];
  const a = cps(s);
  let i = 0;
  while (true) {
    while (i < a.length && isWs(a[i])) i++;
    if (i >= a.length) break;
    if (max >= 0 && out.length === max) {
      let rest = a.slice(i).join("");
      out.push(stripChars(rest, undefined, false, true));
      break;
    }
    let j = i;
    while (j < a.length && !isWs(a[j])) j++;
    out.push(a.slice(i, j).join(""));
    i = j;
  }
  return out;
}

function splitSep(s, sep, max = -1) {
  if (sep === "") throw err("ValueError", "empty separator");
  const out = [];
  let rest = s;
  while (max < 0 || out.length < max) {
    const i = rest.indexOf(sep);
    if (i < 0) break;
    out.push(rest.slice(0, i));
    rest = rest.slice(i + sep.length);
  }
  out.push(rest);
  return out;
}

function rsplitSep(s, sep, max = -1) {
  if (sep === "") throw err("ValueError", "empty separator");
  const out = [];
  let rest = s;
  while (max < 0 || out.length < max) {
    const i = rest.lastIndexOf(sep);
    if (i < 0) break;
    out.unshift(rest.slice(i + sep.length));
    rest = rest.slice(0, i);
  }
  out.unshift(rest);
  return out;
}

const LINE_BREAKS = /\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/u;

function splitlines(s, keep = false) {
  const out = [];
  let rest = s;
  while (rest.length) {
    const m = LINE_BREAKS.exec(rest);
    if (!m) {
      out.push(rest);
      break;
    }
    out.push(rest.slice(0, m.index + (keep ? m[0].length : 0)));
    rest = rest.slice(m.index + m[0].length);
  }
  return out;
}

/** index of a code unit offset → code point index */
const cpIndex = (s, i) => (i < 0 ? -1 : cps(s.slice(0, i)).length);

function countSub(s, sub) {
  if (sub === "") return cps(s).length + 1;
  let n = 0, i = 0;
  while ((i = s.indexOf(sub, i)) >= 0) {
    n++;
    i += sub.length;
  }
  return n;
}

const cased = (c) => c.toLowerCase() !== c.toUpperCase();

function title(s) {
  let out = "", prev = false;
  for (const c of s) {
    out += prev ? c.toLowerCase() : c.toUpperCase();
    prev = cased(c);
  }
  return out;
}

function justify(s, width, fill, how) {
  if (fill !== undefined && cps(fill).length !== 1) throw err("TypeError", "The fill character must be exactly one character long");
  fill = fill ?? " ";
  const n = num(width) - cps(s).length;
  if (n <= 0) return s;
  if (how === "l") return s + fill.repeat(n);
  if (how === "r") return fill.repeat(n) + s;
  // center: as CPython, the extra space goes left when the width is odd and n is odd
  const left = (n >> 1) + (n & num(width) & 1);
  return fill.repeat(left) + s + fill.repeat(n - left);
}

const STR = {
  upper: (s) => s.toUpperCase(),
  lower: (s) => s.toLowerCase(),
  casefold: (s) => s.toLowerCase().replace(/\u00df/g, "ss"),
  swapcase: (s) => cps(s).map((c) => (c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase())).join(""),
  title,
  capitalize: (s) => {
    const a = cps(s);
    return a.length ? a[0].toUpperCase() + a.slice(1).join("").toLowerCase() : "";
  },
  strip: (s, c) => stripChars(s, c, true, true),
  strip_chars: (s, c) => stripChars(s, c, true, true),
  lstrip_chars: (s, c) => stripChars(s, c, true, false),
  rstrip_chars: (s, c) => stripChars(s, c, false, true),
  split_max: (s, sep, max) => splitSep(s, sep, num(max)),
  replace_n: (s, a, b, n) => STR.replace(s, a, b, n),
  find_from: (s, sub, start) => {
    const a = cps(s);
    let i = num(start);
    if (i < 0) i = Math.max(0, i + a.length);
    if (i > a.length) return -1;
    const pre = a.slice(0, i).join("").length;
    return cpIndex(s, s.indexOf(sub, pre));
  },
  count_from: (s, sub, start) => {
    const a = cps(s);
    let i = num(start);
    if (i < 0) i = Math.max(0, i + a.length);
    if (i > a.length) return 0;
    return countSub(a.slice(i).join(""), sub);
  },
  center_fill: (s, w, f) => justify(s, w, f, "c"),
  ljust_fill: (s, w, f) => justify(s, w, f, "l"),
  rjust_fill: (s, w, f) => justify(s, w, f, "r"),
  lstrip: (s, c) => stripChars(s, c, true, false),
  rstrip: (s, c) => stripChars(s, c, false, true),
  split: (s, sep, max) => (sep === undefined ? splitWs(s) : splitSep(s, sep, max === undefined ? -1 : num(max))),
  split_ws: (s, max) => splitWs(s, max === undefined ? -1 : num(max)),
  split_ws_max: (s, max) => splitWs(s, num(max)),
  rsplit: (s, sep, max) => rsplitSep(s, sep, max === undefined ? -1 : num(max)),
  rsplit_all: (s, sep) => rsplitSep(s, sep),
  rsplit_max: (s, sep, max) => rsplitSep(s, sep, num(max)),
  splitlines: (s, keep) => splitlines(s, keep !== undefined && num(keep) !== 0),
  replace: (s, a, b, n) => {
    const max = n === undefined ? -1 : num(n);
    if (a === "") {
      const parts = cps(s);
      let out = "", k = 0;
      for (let i = 0; i <= parts.length; i++) {
        if (max < 0 || k < max) {
          out += b;
          k++;
        }
        if (i < parts.length) out += parts[i];
      }
      return out;
    }
    return max < 0 ? s.split(a).join(b) : splitSep(s, a, max).join(b);
  },
  startswith: (s, p) => s.startsWith(p),
  endswith: (s, p) => s.endsWith(p),
  find: (s, sub, start) => cpIndex(s, s.indexOf(sub, start === undefined ? 0 : cps(s).slice(0, num(start)).join("").length)),
  rfind: (s, sub) => cpIndex(s, s.lastIndexOf(sub)),
  rindex: (s, sub) => {
    const i = s.lastIndexOf(sub);
    if (i < 0) throw err("ValueError", "substring not found");
    return cpIndex(s, i);
  },
  count: (s, sub) => countSub(s, sub),
  isdigit: (s) => s !== "" && /^[\p{Nd}\u00b2\u00b3\u00b9\u2070-\u2079\u2080-\u2089\u2460-\u2468]+$/u.test(s),
  isdecimal: (s) => s !== "" && /^\p{Nd}+$/u.test(s),
  isnumeric: (s) => s !== "" && /^\p{N}+$/u.test(s),
  isalpha: (s) => s !== "" && /^\p{L}+$/u.test(s),
  isalnum: (s) => s !== "" && /^[\p{L}\p{N}]+$/u.test(s),
  isspace: (s) => s !== "" && cps(s).every(isWs),
  isupper: (s) => /\p{Lu}|\p{Lt}/u.test(s) && !/\p{Ll}/u.test(s),
  islower: (s) => /\p{Ll}/u.test(s) && !/\p{Lu}|\p{Lt}/u.test(s),
  istitle: (s) => s !== "" && title(s) === s && /\p{L}/u.test(s),
  isascii: (s) => /^[\x00-\x7f]*$/.test(s),
  isidentifier: (s) => /^[\p{L}_][\p{L}\p{N}_]*$/u.test(s),
  isprintable: (s) => !/[\p{C}\p{Z}]/u.test(s.replace(/ /g, "")),
  center: (s, w, f) => justify(s, w, f, "c"),
  ljust: (s, w, f) => justify(s, w, f, "l"),
  rjust: (s, w, f) => justify(s, w, f, "r"),
  zfill: (s, w) => {
    const n = num(w) - cps(s).length;
    if (n <= 0) return s;
    const sign = s[0] === "+" || s[0] === "-" ? s[0] : "";
    return sign + "0".repeat(n) + s.slice(sign.length);
  },
  partition: (s, sep) => {
    if (sep === "") throw err("ValueError", "empty separator");
    const i = s.indexOf(sep);
    return i < 0 ? [s, "", ""] : [s.slice(0, i), sep, s.slice(i + sep.length)];
  },
  rpartition: (s, sep) => {
    if (sep === "") throw err("ValueError", "empty separator");
    const i = s.lastIndexOf(sep);
    return i < 0 ? ["", "", s] : [s.slice(0, i), sep, s.slice(i + sep.length)];
  },
  removeprefix: (s, p) => (s.startsWith(p) ? s.slice(p.length) : s),
  removesuffix: (s, p) => (p && s.endsWith(p) ? s.slice(0, s.length - p.length) : s),
  expandtabs: (s, n) => {
    const size = n === undefined ? 8 : num(n);
    let out = "", col = 0;
    for (const c of s) {
      if (c === "\t") {
        const k = size > 0 ? size - (col % size) : 0;
        out += " ".repeat(k);
        col += k;
      } else {
        out += c;
        col = c === "\n" || c === "\r" ? 0 : col + 1;
      }
    }
    return out;
  },
};

// ------------------------------------------------------------ the module's host

/**
 * Run a compiled program. `io.stdout` / `io.stderr` receive text; `io.stdin`
 * is the input text (or a function returning the next line, or null at the end).
 * Returns the exit code.
 */
export async function run(bytes, io = {}) {
  const stdout = io.stdout ?? ((s) => console.log(s));
  const stderr = io.stderr ?? ((s) => console.error(s));
  let lines = null;
  const nextLine = () => {
    if (typeof io.stdin === "function") return io.stdin();
    if (lines === null) {
      const text = io.stdin ?? "";
      lines = text.split(/(?<=\n)/);
      if (lines.length && lines[lines.length - 1] === "") lines.pop();
    }
    return lines.length ? lines.shift() : null;
  };
  let mem = null;
  let args = [];
  let pending = new Uint8Array(0);
  let list = [];
  let failed = 0;
  const str = (ptr, len) => dec.decode(new Uint8Array(mem.buffer, ptr, len));
  const setPending = (s) => {
    pending = enc.encode(s);
  };
  const ops = {
    float_repr: (x) => floatRepr(x),
    format: (v, spec) => format(v, spec),
    int: (s, base) => parseInt64(s, num(base)),
    float: (s) => parseFloat64(s),
    ord: (s) => {
      const a = cps(s);
      if (a.length !== 1) throw err("TypeError", "ord() expected a character, but string of length " + a.length + " found");
      return BigInt(a[0].codePointAt(0));
    },
    chr: (i) => {
      if (i < 0n || i > 0x10ffffn) throw err("ValueError", "chr() arg not in range(0x110000)");
      return String.fromCodePoint(num(i));
    },
    input: () => {
      const l = nextLine();
      if (l === null || l === undefined) throw err("EOFError", "EOF when reading a line");
      return l.replace(/\r?\n$/, "");
    },
  };
  for (const [k, f] of Object.entries(MATH)) ops[k] = f;
  for (const [k, f] of Object.entries(STR)) ops["str." + k] = f;
  const invoke = () => {
    const name = args.pop();
    const a = args;
    args = [];
    failed = 0;
    const f = ops[name];
    if (!f) throw new Error("ruai wasm host: unknown operation " + name);
    try {
      return f(...a);
    } catch (e) {
      if (!(e instanceof PyErr)) throw e;
      failed = KINDS.indexOf(e.kind);
      setPending(e.msg);
      return null;
    }
  };
  const imports = {
    host: {
      write: (fd) => (fd === 2 ? stderr : stdout)(args.pop()),
      arg: (ptr, len) => args.push(str(ptr, len)),
      int_arg: (v) => args.push(v),
      float_arg: (v) => args.push(v),
      call: () => {
        const r = invoke();
        if (r === null) return 0n;
        if (typeof r === "string") {
          setPending(r);
          return 0n;
        }
        if (Array.isArray(r)) {
          list = r;
          return BigInt(r.length);
        }
        if (typeof r === "boolean") return r ? 1n : 0n;
        if (typeof r === "number") return BigInt(r);
        return r;
      },
      fcall: () => {
        const r = invoke();
        return r === null ? 0 : r;
      },
      pending: () => pending.length,
      take: (ptr) => new Uint8Array(mem.buffer, ptr, pending.length).set(pending),
      item: (i) => {
        setPending(list[i]);
        return pending.length;
      },
      failed: () => failed,
    },
  };
  const { instance } = await WebAssembly.instantiate(bytes, imports);
  mem = instance.exports.mem;
  try {
    return instance.exports.run();
  } catch (e) {
    if (e instanceof RangeError && /call stack/i.test(e.message)) {
      stderr("Traceback (most recent call last):\nRecursionError: maximum recursion depth exceeded\n");
    } else {
      stderr("Traceback (most recent call last):\nRuntimeError: " + e.message + "\n");
    }
    return 1;
  }
}

// node ruai_wasm.mjs prog.wasm < input
const isMain =
  typeof process !== "undefined" &&
  process.argv?.[1] &&
  (await (async () => {
    const [{ pathToFileURL }, fs] = await Promise.all([import("node:url"), import("node:fs")]);
    return import.meta.url === pathToFileURL(fs.realpathSync(process.argv[1])).href;
  })());
if (isMain) {
  const fs = await import("node:fs");
  const file = process.argv[2];
  if (!file) {
    console.error("usage: node ruai_wasm.mjs prog.wasm < input");
    process.exit(2);
  }
  let out = "";
  const flush = () => {
    if (out) fs.writeSync(1, out);
    out = "";
  };
  let input = null;
  const code = await run(fs.readFileSync(file), {
    stdout: (s) => {
      out += s;
      if (out.length > 65536) flush();
    },
    stderr: (s) => {
      flush();
      fs.writeSync(2, s);
    },
    stdin: () => {
      if (input === null) {
        try {
          input = fs.readFileSync(0, "utf8").split(/(?<=\n)/);
        } catch {
          input = [];
        }
      }
      return input.length ? input.shift() : null;
    },
  });
  flush();
  process.exitCode = code;
}

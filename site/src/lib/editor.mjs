// Code editors for py0: Monaco (the editor of VS Code, loaded from a CDN)
// when it can be loaded, else a plain textarea with the same interface:
//   { getValue, setValue, focus, layout, setErrors([{line, message}]),
//     onChange(fn), onCursor(fn), revealLine(n), dispose }

const BASE = "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min";
let monacoP = null;

/** Monaco, or null when it cannot be loaded (offline). */
export function loadMonaco() {
  monacoP ??= new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), 15000);
    const s = document.createElement("script");
    s.src = BASE + "/vs/loader.js";
    s.onerror = () => {
      clearTimeout(timer);
      resolve(null);
    };
    s.onload = () => {
      // the editor's web workers come from the CDN through a same-origin data: URL
      window.MonacoEnvironment = {
        getWorkerUrl: () =>
          "data:text/javascript;charset=utf-8," +
          encodeURIComponent(`self.MonacoEnvironment={baseUrl:'${BASE}/'};importScripts('${BASE}/vs/base/worker/workerMain.js');`),
      };
      window.require.config({ paths: { vs: BASE + "/vs" } });
      window.require(["vs/editor/editor.main"], () => {
        clearTimeout(timer);
        resolve(window.monaco);
      }, () => {
        clearTimeout(timer);
        resolve(null);
      });
    };
    document.head.appendChild(s);
  });
  return monacoP;
}

const dark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
};

/**
 * An editor in `el`. Options: value, autoHeight (grow with the content, for
 * notebook cells), lineNumbers, actions: [{ id, label, keys: [[mod, key]...] as
 * strings like "Ctrl+Enter", run }].
 */
export async function createEditor(el, opts = {}) {
  const monaco = await loadMonaco();
  return monaco ? monacoEditor(monaco, el, opts) : textEditor(el, opts);
}

function keyCode(monaco, spec) {
  let code = 0;
  for (const part of spec.split("+")) {
    if (part === "Ctrl") code |= monaco.KeyMod.CtrlCmd;
    else if (part === "Shift") code |= monaco.KeyMod.Shift;
    else if (part === "Alt") code |= monaco.KeyMod.Alt;
    else code |= monaco.KeyCode[part] ?? monaco.KeyCode["Key" + part];
  }
  return code;
}

function monacoEditor(monaco, el, opts) {
  const ed = monaco.editor.create(el, {
    value: opts.value ?? "",
    language: "python",
    theme: dark() ? "vs-dark" : "vs",
    automaticLayout: !opts.autoHeight,
    fontSize: 13,
    fontFamily: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace',
    minimap: { enabled: !opts.autoHeight && (opts.minimap ?? true) },
    lineNumbers: opts.lineNumbers === false ? "off" : "on",
    scrollBeyondLastLine: !opts.autoHeight,
    tabSize: 4,
    insertSpaces: true,
    renderLineHighlight: opts.autoHeight ? "none" : "line",
    overviewRulerLanes: opts.autoHeight ? 0 : 3,
    hideCursorInOverviewRuler: !!opts.autoHeight,
    scrollbar: opts.autoHeight ? { vertical: "hidden", alwaysConsumeMouseWheel: false, handleMouseWheel: false } : {},
    padding: opts.autoHeight ? { top: 8, bottom: 8 } : { top: 6 },
    folding: !opts.autoHeight,
    glyphMargin: false,
    lineDecorationsWidth: opts.autoHeight ? 6 : 10,
    lineNumbersMinChars: opts.autoHeight ? 3 : 4,
    wordWrap: opts.wordWrap ?? "off",
    contextmenu: true,
    fixedOverflowWidgets: true,
  });
  const media = matchMedia("(prefers-color-scheme: dark)");
  const onTheme = () => monaco.editor.setTheme(dark() ? "vs-dark" : "vs");
  media.addEventListener("change", onTheme);
  if (opts.autoHeight) {
    const fit = () => {
      el.style.height = Math.max(ed.getContentHeight(), 34) + "px";
      ed.layout({ width: el.clientWidth, height: Math.max(ed.getContentHeight(), 34) });
    };
    ed.onDidContentSizeChange(fit);
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    fit();
  }
  for (const a of opts.actions ?? []) {
    ed.addAction({ id: a.id, label: a.label, keybindings: a.keys.map((k) => keyCode(monaco, k)), run: () => a.run() });
  }
  return {
    monaco: ed,
    getValue: () => ed.getValue(),
    setValue: (v) => ed.setValue(v),
    focus: () => ed.focus(),
    layout: () => ed.layout(),
    setErrors: (errs) =>
      monaco.editor.setModelMarkers(
        ed.getModel(),
        "py0",
        errs.map((e) => ({ startLineNumber: e.line, endLineNumber: e.line, startColumn: 1, endColumn: ed.getModel().getLineMaxColumn(Math.min(e.line, ed.getModel().getLineCount())), message: e.message, severity: monaco.MarkerSeverity.Error })),
      ),
    onChange: (fn) => ed.onDidChangeModelContent(() => fn()),
    onCursor: (fn) => ed.onDidChangeCursorPosition((e) => fn(e.position.lineNumber, e.position.column)),
    onFocus: (fn) => ed.onDidFocusEditorText(fn),
    revealLine: (n) => {
      ed.revealLineInCenter(n);
      ed.setPosition({ lineNumber: n, column: 1 });
      ed.focus();
    },
    dispose: () => {
      media.removeEventListener("change", onTheme);
      ed.dispose();
    },
  };
}

function textEditor(el, opts) {
  const ta = document.createElement("textarea");
  ta.value = opts.value ?? "";
  ta.spellcheck = false;
  ta.className = "plain-editor";
  ta.style.cssText = "width:100%;height:100%;border:0;outline:none;resize:none;padding:8px 10px;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;background:transparent;color:inherit;tab-size:4;white-space:pre;";
  el.appendChild(ta);
  const fit = () => {
    if (opts.autoHeight) {
      ta.style.height = "0";
      ta.style.height = Math.max(ta.scrollHeight, 34) + "px";
      el.style.height = ta.style.height;
    }
  };
  const actions = (opts.actions ?? []).map((a) => ({ ...a, keys: a.keys.map((k) => k.split("+")) }));
  ta.addEventListener("keydown", (e) => {
    for (const a of actions)
      for (const k of a.keys) {
        const key = k[k.length - 1];
        const want = { Ctrl: k.includes("Ctrl"), Shift: k.includes("Shift"), Alt: k.includes("Alt") };
        const name = key === "Enter" ? "Enter" : key.length === 1 ? key.toLowerCase() : key;
        if ((e.key === name || e.key.toLowerCase() === name) && want.Ctrl === (e.ctrlKey || e.metaKey) && want.Shift === e.shiftKey && want.Alt === e.altKey) {
          e.preventDefault();
          a.run();
          return;
        }
      }
    if (e.key === "Tab") {
      e.preventDefault();
      ta.setRangeText("    ", ta.selectionStart, ta.selectionEnd, "end");
      ta.dispatchEvent(new Event("input"));
    }
  });
  ta.addEventListener("input", fit);
  queueMicrotask(fit);
  const pos = () => {
    const before = ta.value.slice(0, ta.selectionStart).split("\n");
    return [before.length, before[before.length - 1].length + 1];
  };
  return {
    getValue: () => ta.value,
    setValue: (v) => {
      ta.value = v;
      fit();
    },
    focus: () => ta.focus(),
    layout: fit,
    setErrors: () => {},
    onChange: (fn) => ta.addEventListener("input", () => fn()),
    onCursor: (fn) => ["keyup", "click"].forEach((ev) => ta.addEventListener(ev, () => fn(...pos()))),
    onFocus: (fn) => ta.addEventListener("focus", fn),
    revealLine: (n) => {
      const lines = ta.value.split("\n");
      const at = lines.slice(0, n - 1).join("\n").length + (n > 1 ? 1 : 0);
      ta.focus();
      ta.setSelectionRange(at, at);
    },
    dispose: () => ta.remove(),
  };
}

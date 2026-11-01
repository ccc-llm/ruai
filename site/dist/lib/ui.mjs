// Small helpers shared by the pages.

/** Elements from HTML: `html(el, "...")` fills `el`; returns a lookup by id-like `data-id`. */
export function html(el, markup) {
  el.innerHTML = markup;
  const $ = (id) => el.querySelector(`[data-id="${id}"]`);
  return $;
}

/** A canvas sized `w` × `h` CSS pixels for the screen's pixel ratio: its 2-D context, scaled. */
export function sized(canvas, w, h) {
  const r = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * r);
  canvas.height = Math.round(h * r);
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  const g = canvas.getContext("2d");
  g.setTransform(r, 0, 0, r, 0, 0);
  return g;
}

/** A CSS variable of the page (colors follow the theme). */
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** A loop on animation frames until the returned function is called. */
export function animate(fn) {
  let on = true;
  const tick = async () => {
    if (!on) return;
    const more = await fn();
    if (on && more !== false) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  return () => (on = false);
}

/** Yield to the browser (paint, input) for one frame. */
export const frame = () => new Promise(requestAnimationFrame);

/** Mix two "#rrggbb" colors: t = 0 gives a, 1 gives b. */
export function mix(a, b, t) {
  const p = (c, i) => parseInt(c.slice(1 + 2 * i, 3 + 2 * i), 16);
  return "rgb(" + [0, 1, 2].map((i) => Math.round(p(a, i) + (p(b, i) - p(a, i)) * t)).join(",") + ")";
}

/** Escape text for HTML. */
export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

/** Save text as a file in the browser's downloads. */
export function saveFile(name, text, type = "text/plain") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** Ask for a local file: its name and text. */
export function openFile(accept) {
  return new Promise((resolve) => {
    const i = document.createElement("input");
    i.type = "file";
    i.accept = accept;
    i.onchange = async () => {
      const f = i.files[0];
      resolve(f ? { name: f.name, text: await f.text() } : null);
    };
    i.click();
  });
}

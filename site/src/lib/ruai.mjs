// ruai compiled to wasm (pkg/, built from site/wasm): loaded once, shared by
// the pages. `ready` resolves to the name of the GPU demos' device.
import init, * as ruai from "../pkg/ruai_site.js";

export { ruai };

let drawFn = null;
export const ready = init().then(async () => {
  drawFn = new Function(ruai.canvas_js() + "\nreturn ruaiDraw;")();
  return await ruai.start();
});

/** Draw ruai::plot's canvas JSON on a canvas. */
export function draw(canvas, json) {
  drawFn(canvas, typeof json === "string" ? JSON.parse(json) : json);
}

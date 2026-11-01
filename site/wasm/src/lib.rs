//! The demos of the ruai website, compiled to wasm32: each module holds one
//! demo's state (thread-local, the page is single-threaded) and the functions
//! its page calls.

use ruai::prelude::*;
use wasm_bindgen::prelude::*;

mod cartpole;
mod descent;
mod fit;
mod gallery;
mod gpt;
mod gym;
mod lake;
mod mountaincar;
mod playground;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(s: &str);
}

thread_local! {
    static GPU: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set up the module: panics go to the console, and WebGPU is started when
/// the browser has it. The name of the device the GPU demos use.
#[wasm_bindgen]
pub async fn start() -> String {
    std::panic::set_hook(Box::new(|info| console_error(&format!("panic: {info}"))));
    let gpu = ruai::backend::webgpu::init().await;
    GPU.with(|g| g.set(gpu));
    if gpu {
        // browsers may hide the adapter's name
        let name = ruai::backend::webgpu::adapter_name().unwrap_or_default();
        let name = name.replace("(BrowserWebGpu)", "");
        let name = name.trim();
        if name.is_empty() { "WebGPU".into() } else { format!("WebGPU: {name}") }
    } else {
        "CPU (this browser has no WebGPU)".into()
    }
}

/// Whether WebGPU is available.
#[wasm_bindgen]
pub fn has_webgpu() -> bool {
    GPU.with(|g| g.get())
}

/// The device of the GPU demos: WebGPU, or the CPU without it.
fn gpu_device() -> Device {
    if has_webgpu() { Device::WebGpu } else { Device::Cpu }
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

/// The script that draws ruai::plot's canvas JSON (`ruaiDraw(canvas, fig)`).
#[wasm_bindgen]
pub fn canvas_js() -> String {
    ruai::plot::render::CANVAS_JS.to_string()
}

/// A normal draw (Box–Muller) from a NumPy-compatible generator.
fn normal(g: &mut ruai::rng::Generator) -> f64 {
    let u = g.random().max(1e-12);
    let v = g.random();
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

/// An MLP `sizes[0] → … → sizes[last]` with `act` between the layers.
fn mlp(sizes: &[usize], act: &str, device: Device) -> nn::Sequential {
    let mut m = nn::Sequential::new();
    for (i, w) in sizes.windows(2).enumerate() {
        m.push(nn::Linear::new(w[0], w[1], device));
        if i + 2 < sizes.len() {
            match act {
                "relu" => m.push(nn::ReLU),
                "sigmoid" => m.push(nn::Sigmoid),
                "gelu" => m.push(nn::GELU),
                _ => m.push(nn::Tanh),
            }
        }
    }
    m
}

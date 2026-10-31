//! The py0 → WebAssembly compiler as a WebAssembly module, for browsers: the
//! translator and the wasm backend (`ruai::py::wasm`) built for
//! `wasm32-unknown-unknown`. `py0/web/build.sh` builds it; `py0/web/index.html`
//! compiles py0 programs with it and runs them with the host `ruai_wasm.mjs`.
//!
//! The interface is plain C: `alloc(n)` reserves n bytes for the source,
//! `compile(ptr, len)` returns 0 (the output is the program's wasm) or 1 (the
//! output is the error message), `out_ptr()` / `out_len()` give the output.

use std::sync::Mutex;

static OUT: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// Reserve `n` bytes (for the source text); never freed (one per compile).
#[no_mangle]
pub extern "C" fn alloc(n: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(n.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Compile the UTF-8 source at `ptr` → 0 (wasm) or 1 (an error message).
///
/// # Safety
/// `ptr` must point to `len` readable bytes (from `alloc`).
#[no_mangle]
pub unsafe extern "C" fn compile(ptr: *const u8, len: usize) -> i32 {
    let src = String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(ptr, len) }).into_owned();
    let (code, out) = match ruai::py::wasm::compile_source(&src) {
        Ok(bytes) => (0, bytes),
        Err(e) => (1, e.into_bytes()),
    };
    *OUT.lock().unwrap() = out;
    code
}

/// The last output.
#[no_mangle]
pub extern "C" fn out_ptr() -> *const u8 {
    OUT.lock().unwrap().as_ptr()
}

/// The last output's length.
#[no_mangle]
pub extern "C" fn out_len() -> usize {
    OUT.lock().unwrap().len()
}

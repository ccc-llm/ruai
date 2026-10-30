//! Training checkpoints: model weights + optimizer state + progress + RNG
//! state in one safetensors file, so training can stop and resume exactly.
//!
//! Layout: model tensors under `model.<name>`, optimizer tensors under
//! `optim.<name>`; the header metadata holds `format`, `optimizer`, `step`,
//! `epoch`, `rng_state` plus any user entries.
//!
//! ```no_run
//! # use ruai::{nn::{Linear, Module}, optim::{Adam, Optimizer}, checkpoint, Device};
//! # let model = Linear::new(2, 2, Device::Cpu);
//! # let mut opt = Adam::new(model.parameters(), 1e-3);
//! let state = checkpoint::TrainState { step: 1000, epoch: 3, ..Default::default() };
//! checkpoint::save("run/ckpt.safetensors", &model, &opt, &state).unwrap();
//! // later, with a freshly built model and optimizer:
//! let state = checkpoint::load("run/ckpt.safetensors", &model, &mut opt).unwrap();
//! assert_eq!(state.step, 1000);
//! ```

use crate::nn::modules::Module;
use crate::nn::optim::Optimizer;
use crate::nn::serialize::{self, Metadata};
use crate::nn::tensor::Tensor;
use std::io;
use std::path::Path;

const FORMAT: &str = "ruai-checkpoint-v1";

/// Training progress stored alongside the tensors.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrainState {
    /// Optimizer steps taken so far.
    pub step: u64,
    /// Completed epochs.
    pub epoch: u64,
    /// Free-form user metadata (e.g. best validation loss, config).
    pub extra: Metadata,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Write a checkpoint. The global RNG state is saved too, so dropout and
/// sampling continue identically after [`load`]. The file is written to a
/// temporary path first and renamed, so a crash never leaves a torn checkpoint.
pub fn save(path: impl AsRef<Path>, model: &dyn Module, opt: &dyn Optimizer, state: &TrainState) -> io::Result<()> {
    let path = path.as_ref();
    let mut tensors: Vec<(String, Tensor)> =
        model.state_dict().into_iter().map(|(n, t)| (format!("model.{n}"), t)).collect();
    tensors.extend(opt.state_dict().into_iter().map(|(n, t)| (format!("optim.{n}"), t)));
    let mut meta = state.extra.clone();
    meta.insert("format".into(), FORMAT.into());
    meta.insert("optimizer".into(), opt.name().into());
    meta.insert("step".into(), state.step.to_string());
    meta.insert("epoch".into(), state.epoch.to_string());
    meta.insert("rng_state".into(), crate::nn::rand::state().to_string());
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    serialize::save_safetensors(&tmp, &tensors, Some(meta))?;
    std::fs::rename(&tmp, path)
}

/// Restore a checkpoint into an already-built model and optimizer (same
/// architecture and parameter order), and restore the global RNG state.
pub fn load(path: impl AsRef<Path>, model: &dyn Module, opt: &mut dyn Optimizer) -> io::Result<TrainState> {
    let (tensors, mut meta) = serialize::load_safetensors(path)?;
    if meta.get("format").map(String::as_str) != Some(FORMAT) {
        return Err(invalid("not a ruai checkpoint (missing format metadata)".into()));
    }
    let saved_opt = meta.remove("optimizer").unwrap_or_default();
    if saved_opt != opt.name() {
        return Err(invalid(format!("checkpoint has optimizer '{saved_opt}', but '{}' was given", opt.name())));
    }
    let (mut model_t, mut opt_t) = (Vec::new(), Vec::new());
    for (name, t) in tensors {
        if let Some(n) = name.strip_prefix("model.") {
            model_t.push((n.to_string(), t));
        } else if let Some(n) = name.strip_prefix("optim.") {
            opt_t.push((n.to_string(), t));
        }
    }
    serialize::load_state_dict(model, &model_t, true)?;
    opt.load_state_dict(&opt_t)?;
    let num = |m: &mut Metadata, key: &str| -> io::Result<u64> {
        m.remove(key)
            .ok_or_else(|| invalid(format!("checkpoint metadata is missing '{key}'")))?
            .parse()
            .map_err(|e| invalid(format!("bad '{key}' in checkpoint: {e}")))
    };
    let step = num(&mut meta, "step")?;
    let epoch = num(&mut meta, "epoch")?;
    crate::nn::rand::set_state(num(&mut meta, "rng_state")?);
    meta.remove("format");
    Ok(TrainState { step, epoch, extra: meta })
}

/// Read a checkpoint's progress and metadata without loading any weights.
pub fn peek(path: impl AsRef<Path>) -> io::Result<TrainState> {
    let (_, meta) = serialize::load_safetensors(path)?;
    let get = |k: &str| meta.get(k).and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut extra = meta.clone();
    for k in ["format", "optimizer", "step", "epoch", "rng_state"] {
        extra.remove(k);
    }
    Ok(TrainState { step: get("step"), epoch: get("epoch"), extra })
}

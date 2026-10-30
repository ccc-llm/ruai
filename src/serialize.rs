//! Saving and loading named tensors (a model's `state_dict`).
//!
//! The default format is [safetensors](https://huggingface.co/docs/safetensors)
//! (via Hugging Face's reference implementation), so weights interoperate with
//! PyTorch / Hugging Face: `F32`, `F16`, `BF16`, `I32` and `I8` are written
//! and read as-is; `F64` loads as `F32`, and `I64`, `I16`, `U8`, `U16`, `U32`,
//! `BOOL` load as `I32`. Files in the legacy `RSNN` format (v0.1-v0.4) can still be loaded.

use crate::backend::DType;
use crate::nn::Module;
use crate::shape::numel;
use crate::tensor::Tensor;
use crate::Device;
use safetensors::tensor::{Dtype as StDtype, SafeTensors, TensorView};
use std::collections::{HashMap, HashSet};
use std::io::{self, Read};
use std::path::Path;

/// String key/value metadata stored in the safetensors header.
pub type Metadata = HashMap<String, String>;

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

// ----------------------------------------------------------------------
// safetensors
// ----------------------------------------------------------------------

/// Serialize named tensors (any device) to safetensors bytes.
pub fn to_safetensors_bytes(tensors: &[(String, Tensor)], metadata: Option<Metadata>) -> io::Result<Vec<u8>> {
    let mut seen = HashSet::new();
    let mut raw: Vec<(String, StDtype, Vec<usize>, Vec<u8>)> = Vec::with_capacity(tensors.len());
    for (name, t) in tensors {
        if !seen.insert(name.as_str()) {
            return Err(invalid(format!("duplicate tensor name '{name}'")));
        }
        let (dtype, bytes) = match t.dtype() {
            DType::F32 => (StDtype::F32, t.to_vec().iter().flat_map(|v| v.to_le_bytes()).collect()),
            DType::I32 => (StDtype::I32, t.to_vec_i32().iter().flat_map(|v| v.to_le_bytes()).collect()),
            // exact: the values are already representable in the half format
            DType::F16 => (StDtype::F16, t.to_vec().iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect()),
            DType::BF16 => (StDtype::BF16, t.to_vec().iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect()),
            DType::I8 => (StDtype::I8, t.to_vec_i32().iter().map(|&v| v as i8 as u8).collect()),
        };
        raw.push((name.clone(), dtype, t.shape().to_vec(), bytes));
    }
    let views = raw
        .iter()
        .map(|(n, d, s, b)| Ok((n.as_str(), TensorView::new(*d, s.clone(), b).map_err(|e| invalid(e.to_string()))?)))
        .collect::<io::Result<Vec<_>>>()?;
    safetensors::serialize(views, metadata).map_err(|e| invalid(e.to_string()))
}

/// Save named tensors as a `.safetensors` file.
pub fn save_safetensors(path: impl AsRef<Path>, tensors: &[(String, Tensor)], metadata: Option<Metadata>) -> io::Result<()> {
    std::fs::write(path, to_safetensors_bytes(tensors, metadata)?)
}

fn view_to_tensor(name: &str, v: &TensorView) -> io::Result<Tensor> {
    let shape = v.shape().to_vec();
    let b = v.data();
    let words = |n: usize| b.chunks_exact(n);
    let t = match v.dtype() {
        StDtype::F32 => Tensor::from_vec(words(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(), &shape, Device::Cpu),
        StDtype::F64 => Tensor::from_vec(
            words(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            &shape,
            Device::Cpu,
        ),
        StDtype::F16 => Tensor::from_vec(
            words(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            &shape,
            Device::Cpu,
        )
        .to_dtype(DType::F16),
        StDtype::BF16 => Tensor::from_vec(
            words(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            &shape,
            Device::Cpu,
        )
        .to_dtype(DType::BF16),
        StDtype::I32 => Tensor::from_vec_i32(words(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect(), &shape, Device::Cpu),
        StDtype::I64 => {
            let v = words(8)
                .map(|c| {
                    let x = i64::from_le_bytes(c.try_into().unwrap());
                    i32::try_from(x).map_err(|_| invalid(format!("'{name}': i64 value {x} does not fit in i32")))
                })
                .collect::<io::Result<Vec<i32>>>()?;
            Tensor::from_vec_i32(v, &shape, Device::Cpu)
        }
        StDtype::I16 => Tensor::from_vec_i32(words(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i32).collect(), &shape, Device::Cpu),
        StDtype::U16 => Tensor::from_vec_i32(words(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as i32).collect(), &shape, Device::Cpu),
        StDtype::I8 => Tensor::from_vec_i8(b.iter().map(|&x| x as i8).collect(), &shape, Device::Cpu),
        StDtype::U8 | StDtype::BOOL => Tensor::from_vec_i32(b.iter().map(|&x| x as i32).collect(), &shape, Device::Cpu),
        StDtype::U32 => {
            let v = words(4)
                .map(|c| {
                    let x = u32::from_le_bytes(c.try_into().unwrap());
                    i32::try_from(x).map_err(|_| invalid(format!("'{name}': u32 value {x} does not fit in i32")))
                })
                .collect::<io::Result<Vec<i32>>>()?;
            Tensor::from_vec_i32(v, &shape, Device::Cpu)
        }
        other => return Err(invalid(format!("'{name}': unsupported safetensors dtype {other:?}"))),
    };
    Ok(t)
}

/// Parse safetensors bytes into CPU tensors (sorted by name) and metadata.
pub fn from_safetensors_bytes(bytes: &[u8]) -> io::Result<(Vec<(String, Tensor)>, Metadata)> {
    let (_, header) = SafeTensors::read_metadata(bytes).map_err(|e| invalid(e.to_string()))?;
    let metadata = header.metadata().clone().unwrap_or_default();
    let st = SafeTensors::deserialize(bytes).map_err(|e| invalid(e.to_string()))?;
    let mut out = st
        .tensors()
        .iter()
        .map(|(name, view)| Ok((name.clone(), view_to_tensor(name, view)?)))
        .collect::<io::Result<Vec<_>>>()?;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((out, metadata))
}

/// Load a `.safetensors` file onto the CPU.
pub fn load_safetensors(path: impl AsRef<Path>) -> io::Result<(Vec<(String, Tensor)>, Metadata)> {
    from_safetensors_bytes(&std::fs::read(path)?)
}

// ----------------------------------------------------------------------
// legacy RSNN format (read-only)
// ----------------------------------------------------------------------

const RSNN_MAGIC: &[u8; 4] = b"RSNN";

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn load_rsnn(bytes: &[u8]) -> io::Result<Vec<(String, Tensor)>> {
    let mut r = &bytes[4..];
    let version = read_u32(&mut r)?;
    if version != 1 {
        return Err(invalid(format!("unsupported RSNN version {version}")));
    }
    let count = read_u32(&mut r)?;
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let len = read_u32(&mut r)? as usize;
        let mut name = vec![0u8; len];
        r.read_exact(&mut name)?;
        let name = String::from_utf8(name).map_err(|e| invalid(e.to_string()))?;
        let rank = read_u32(&mut r)? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            shape.push(u64::from_le_bytes(b) as usize);
        }
        let mut data = vec![0u8; numel(&shape) * 4];
        r.read_exact(&mut data)?;
        let data = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        out.push((name, Tensor::from_vec(data, &shape, Device::Cpu)));
    }
    Ok(out)
}

// ----------------------------------------------------------------------
// convenience API
// ----------------------------------------------------------------------

/// Save named tensors (safetensors format).
pub fn save_tensors(path: impl AsRef<Path>, tensors: &[(String, Tensor)]) -> io::Result<()> {
    save_safetensors(path, tensors, None)
}

/// Load named tensors onto the CPU; detects safetensors or legacy RSNN files.
pub fn load_tensors(path: impl AsRef<Path>) -> io::Result<Vec<(String, Tensor)>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() >= 4 && &bytes[..4] == RSNN_MAGIC {
        load_rsnn(&bytes)
    } else {
        Ok(from_safetensors_bytes(&bytes)?.0)
    }
}

/// Result of [`load_state_dict`]: names that did not line up.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LoadReport {
    /// In the model but not in the source.
    pub missing: Vec<String>,
    /// In the source but not in the model.
    pub unexpected: Vec<String>,
}

impl LoadReport {
    /// `true` if every name matched (nothing missing or unexpected).
    pub fn is_exact(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty()
    }
}

/// Copy `tensors` into the model's parameters and buffers by name (keeping
/// each model tensor's device and converting dtypes as needed). Shapes must
/// match. With `strict`, any missing or unexpected name is an error.
pub fn load_state_dict(model: &dyn Module, tensors: &[(String, Tensor)], strict: bool) -> io::Result<LoadReport> {
    let src: HashMap<&str, &Tensor> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let dst = model.state_dict();
    let dst_names: HashSet<&str> = dst.iter().map(|(n, _)| n.as_str()).collect();
    let mut report = LoadReport {
        missing: dst.iter().filter(|(n, _)| !src.contains_key(n.as_str())).map(|(n, _)| n.clone()).collect(),
        unexpected: tensors.iter().filter(|(n, _)| !dst_names.contains(n.as_str())).map(|(n, _)| n.clone()).collect(),
    };
    report.unexpected.sort();
    if strict && !report.is_exact() {
        return Err(invalid(format!(
            "state dict mismatch: missing {:?}, unexpected {:?}",
            report.missing, report.unexpected
        )));
    }
    for (name, t) in &dst {
        if let Some(s) = src.get(name.as_str()) {
            if s.shape() != t.shape() {
                return Err(invalid(format!(
                    "shape mismatch for '{name}': source {:?} vs model {:?}",
                    s.shape(),
                    t.shape()
                )));
            }
            t.copy_(&s.to_dtype(t.dtype()));
        }
    }
    Ok(report)
}

/// Save a module's parameters and buffers (safetensors).
pub fn save(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    save_tensors(path, &model.state_dict())
}

/// Load weights into a module in place (strict: names and shapes must match).
pub fn load(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    load_state_dict(model, &load_tensors(path)?, true).map(|_| ())
}

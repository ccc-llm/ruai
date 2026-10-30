//! Saving and loading named tensors (a model's `state_dict`).
//!
//! Format (little endian): magic `b"RSNN"`, version u32, count u32, then for
//! each tensor: name length u32, name bytes (UTF-8), rank u32, dims u64 * rank,
//! data f32 * numel.

use crate::nn::Module;
use crate::shape::numel;
use crate::tensor::Tensor;
use crate::Device;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"RSNN";
const VERSION: u32 = 1;

pub fn save_tensors(path: impl AsRef<Path>, tensors: &[(String, Tensor)]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(tensors.len() as u32).to_le_bytes())?;
    for (name, t) in tensors {
        w.write_all(&(name.len() as u32).to_le_bytes())?;
        w.write_all(name.as_bytes())?;
        w.write_all(&(t.rank() as u32).to_le_bytes())?;
        for &d in t.shape() {
            w.write_all(&(d as u64).to_le_bytes())?;
        }
        for v in t.to_vec() {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// Load all tensors from a file onto the CPU.
pub fn load_tensors(path: impl AsRef<Path>) -> io::Result<Vec<(String, Tensor)>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a rust-nn tensor file"));
    }
    let version = read_u32(&mut r)?;
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unsupported version {version}")));
    }
    let count = read_u32(&mut r)?;
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let len = read_u32(&mut r)? as usize;
        let mut name = vec![0u8; len];
        r.read_exact(&mut name)?;
        let name = String::from_utf8(name).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let rank = read_u32(&mut r)? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            shape.push(u64::from_le_bytes(b) as usize);
        }
        let mut bytes = vec![0u8; numel(&shape) * 4];
        r.read_exact(&mut bytes)?;
        let data = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        out.push((name, Tensor::from_vec(data, &shape, Device::Cpu)));
    }
    Ok(out)
}

/// Save a module's parameters and buffers.
pub fn save(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    save_tensors(path, &model.state_dict())
}

/// Load weights into a module (in place, keeping each tensor's device).
/// Fails if a tensor is missing or has a different shape.
pub fn load(path: impl AsRef<Path>, model: &dyn Module) -> io::Result<()> {
    let loaded: HashMap<String, Tensor> = load_tensors(path)?.into_iter().collect();
    for (name, t) in model.state_dict() {
        let src = loaded
            .get(&name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("missing tensor '{name}'")))?;
        if src.shape() != t.shape() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("shape mismatch for '{name}': file {:?} vs model {:?}", src.shape(), t.shape()),
            ));
        }
        t.copy_(src);
    }
    Ok(())
}

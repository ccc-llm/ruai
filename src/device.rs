//! Device: where a tensor's storage lives.

use std::fmt;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    #[default]
    Cpu,
    /// NVIDIA GPU (requires the `cuda` feature), with device ordinal.
    Cuda(usize),
    /// Apple Metal Performance Shaders (requires the `mps` feature).
    Mps,
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Whether this device's backend was compiled into this build.
    pub fn is_available(&self) -> bool {
        match self {
            Device::Cpu => true,
            Device::Cuda(_) => cfg!(feature = "cuda"),
            Device::Mps => cfg!(feature = "mps"),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
            Device::Mps => write!(f, "mps"),
        }
    }
}

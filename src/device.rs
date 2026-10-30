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
            #[cfg(feature = "cuda")]
            Device::Cuda(i) => crate::backend::cuda::is_available(*i),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => false,
            #[cfg(feature = "mps")]
            Device::Mps => crate::backend::mps::is_available(),
            #[cfg(not(feature = "mps"))]
            Device::Mps => false,
        }
    }
}

impl Device {
    /// Wait for all queued work on this device (useful for timing GPU code).
    pub fn synchronize(&self) {
        crate::backend::backend_for(*self).synchronize();
    }

    /// The fastest available device: MPS, then CUDA, then CPU.
    pub fn best_available() -> Device {
        [Device::Mps, Device::Cuda(0)].into_iter().find(|d| d.is_available()).unwrap_or(Device::Cpu)
    }

    /// Device named by the `RUSTNN_DEVICE` environment variable (default: CPU).
    pub fn from_env() -> Device {
        match std::env::var("RUSTNN_DEVICE") {
            Ok(s) => s.parse().unwrap_or_else(|e| panic!("RUSTNN_DEVICE: {e}")),
            Err(_) => Device::Cpu,
        }
    }
}

impl std::str::FromStr for Device {
    type Err = String;

    /// Parses `cpu`, `mps`, `cuda`, `cuda:N`, or `auto` (best available).
    fn from_str(s: &str) -> Result<Device, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "mps" | "metal" => Ok(Device::Mps),
            "cuda" | "gpu" => Ok(Device::Cuda(0)),
            "auto" => Ok(Device::best_available()),
            other => other
                .strip_prefix("cuda:")
                .and_then(|i| i.parse().ok())
                .map(Device::Cuda)
                .ok_or_else(|| format!("unknown device '{s}' (expected cpu, mps, cuda[:N] or auto)")),
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

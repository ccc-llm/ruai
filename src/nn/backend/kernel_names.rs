//! Kernel naming shared by the GPU backends (`mps_kernels.metal` and
//! `cuda_kernels.cu` instantiate exactly these names).

use super::DType;

/// Kernels with a single (f32 or untyped) version.
pub const PLAIN_KERNELS: &[&str] = &[
    "fill_u32",
    "fill_u16",
    "fill_u8",
    "strided_copy_u32",
    "strided_copy_u16",
    "strided_copy_u8",
    "cat_copy_u32",
    "cat_copy_u16",
    "cat_copy_u8",
    "softmax_rows",
    "softmax_backward_rows",
    "layer_norm_rows",
    "layer_norm_backward_rows",
    "adam_step",
    "index_add",
    "gather",
    "scatter_add",
    "dequant_int4",
];
/// Kernels instantiated for each float storage type: `<name>_<f32|f16|bf16>`.
pub const FLOAT_KERNELS: &[&str] =
    &["unary", "binary", "where", "reduce_strided", "reduce_rows", "index_select", "im2col", "col2im", "matmul_int8", "matmul_int4"];
const DTYPES: [DType; 5] = [DType::F32, DType::F16, DType::BF16, DType::I32, DType::I8];

pub fn kernel_names() -> Vec<String> {
    let mut v: Vec<String> = PLAIN_KERNELS.iter().map(|s| s.to_string()).collect();
    for base in FLOAT_KERNELS {
        for d in [DType::F32, DType::F16, DType::BF16] {
            v.push(fk(base, d));
        }
    }
    for a in DTYPES {
        for b in DTYPES {
            if a != b {
                v.push(cast_kernel(a, b));
            }
        }
    }
    v
}

pub fn dname(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::I8 => "i8",
    }
}

/// `<base>_<dtype>` for a float kernel.
pub fn fk(base: &str, d: DType) -> String {
    assert!(d.is_float(), "MPS kernel `{base}` needs a float dtype, got {d}");
    format!("{base}_{}", dname(d))
}

pub fn cast_kernel(from: DType, to: DType) -> String {
    format!("cast_{}_{}", dname(from), dname(to))
}

/// Dtype-agnostic copy kernel for the element width of `d`.
pub fn copy_kernel(base: &str, d: DType) -> String {
    let w = match d.size_in_bytes() {
        1 => "u8",
        2 => "u16",
        _ => "u32",
    };
    format!("{base}_{w}")
}

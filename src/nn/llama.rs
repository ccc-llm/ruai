//! Llama-family decoder (Llama 2/3, TinyLlama, SmolLM, ...): RMSNorm,
//! rotary position embeddings (RoPE), SwiGLU feed-forward and grouped-query
//! attention (GQA), with a KV cache for generation.
//!
//! Parameter names follow Hugging Face `LlamaForCausalLM`
//! (`model.layers.0.self_attn.q_proj.weight`, ...). Hugging Face stores
//! `nn.Linear` weights as `[out, in]` while [`Linear`] uses `[in, out]`, so
//! [`Llama::from_pretrained`] / [`Llama::save_pretrained`] transpose the
//! projection matrices; [`crate::serialize::save`] / [`crate::serialize::load`]
//! keep this crate's own layout.
//!
//! ```no_run
//! use ruai::nn::llama::Llama;
//! use ruai::generation::Sampler;
//! use ruai::Device;
//! let model = Llama::from_pretrained("data/tinyllama", Device::Cpu).unwrap();
//! let ids = model.generate(&[1, 450, 6593], 20, &mut Sampler::greedy());
//! ```

use super::layers::{Embedding, Linear, RMSNorm, SwiGLU};
use super::module::{prefixed, Module};
use super::transformer::{KvCache, LayerKv};
use crate::functional as F;
use crate::generation::Sampler;
use crate::serialize;
use crate::tensor::Tensor;
use crate::Device;
use serde_json::Value;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Llama 3.1-style RoPE frequency scaling (`rope_scaling.rope_type == "llama3"`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeScaling {
    /// Context-extension factor.
    pub factor: f32,
    /// Wavelengths longer than `original / low_freq_factor` are fully scaled.
    pub low_freq_factor: f32,
    /// Wavelengths shorter than `original / high_freq_factor` are kept.
    pub high_freq_factor: f32,
    /// Context length the model was pre-trained with.
    pub original_max_position_embeddings: usize,
}

/// Llama hyperparameters (field names as in Hugging Face `LlamaConfig`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlamaConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Model width.
    pub hidden_size: usize,
    /// SwiGLU hidden size.
    pub intermediate_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Query heads per layer.
    pub num_attention_heads: usize,
    /// `< num_attention_heads` means grouped-query attention.
    pub num_key_value_heads: usize,
    /// Size of each attention head (usually `hidden_size / num_attention_heads`).
    pub head_dim: usize,
    /// Maximum context length.
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base θ.
    pub rope_theta: f32,
    /// Optional RoPE frequency scaling.
    pub rope_scaling: Option<RopeScaling>,
    /// Output head shares the token embedding table.
    pub tie_word_embeddings: bool,
    /// Biases on the q/k/v/o projections (Llama: no).
    pub attention_bias: bool,
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

impl LlamaConfig {
    /// A small configuration (SwiGLU hidden size ≈ 8/3 · `hidden_size`,
    /// rounded up to a multiple of 32; RoPE θ = 10000).
    pub fn new(vocab_size: usize, hidden_size: usize, layers: usize, heads: usize, kv_heads: usize, max_len: usize) -> Self {
        assert_eq!(hidden_size % heads, 0, "hidden_size {hidden_size} must be divisible by heads {heads}");
        assert_eq!(heads % kv_heads, 0, "heads {heads} must be a multiple of kv_heads {kv_heads}");
        LlamaConfig {
            vocab_size,
            hidden_size,
            intermediate_size: (8 * hidden_size / 3).next_multiple_of(32),
            num_hidden_layers: layers,
            num_attention_heads: heads,
            num_key_value_heads: kv_heads,
            head_dim: hidden_size / heads,
            max_position_embeddings: max_len,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            tie_word_embeddings: false,
            attention_bias: false,
        }
    }

    /// Parse a Hugging Face `config.json`.
    pub fn from_json(text: &str) -> io::Result<Self> {
        let v: Value = serde_json::from_str(text).map_err(|e| invalid(format!("config.json: {e}")))?;
        let req = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| invalid(format!("config.json: missing '{k}'")));
        let opt_f = |k: &str, d: f32| v[k].as_f64().map_or(d, |x| x as f32);
        let hidden_size = req("hidden_size")?;
        let heads = req("num_attention_heads")?;
        let rope_scaling = match &v["rope_scaling"] {
            Value::Null => None,
            s => {
                let kind = s["rope_type"].as_str().or(s["type"].as_str()).unwrap_or("default");
                match kind {
                    "default" => None,
                    "llama3" => {
                        let f = |k: &str| s[k].as_f64().map(|x| x as f32).ok_or_else(|| invalid(format!("rope_scaling: missing '{k}'")));
                        Some(RopeScaling {
                            factor: f("factor")?,
                            low_freq_factor: f("low_freq_factor")?,
                            high_freq_factor: f("high_freq_factor")?,
                            original_max_position_embeddings: s["original_max_position_embeddings"]
                                .as_u64()
                                .ok_or_else(|| invalid("rope_scaling: missing 'original_max_position_embeddings'"))?
                                as usize,
                        })
                    }
                    other => return Err(invalid(format!("unsupported rope_scaling type '{other}'"))),
                }
            }
        };
        Ok(LlamaConfig {
            vocab_size: req("vocab_size")?,
            hidden_size,
            intermediate_size: req("intermediate_size")?,
            num_hidden_layers: req("num_hidden_layers")?,
            num_attention_heads: heads,
            num_key_value_heads: v["num_key_value_heads"].as_u64().map_or(heads, |x| x as usize),
            head_dim: v["head_dim"].as_u64().map_or(hidden_size / heads, |x| x as usize),
            max_position_embeddings: req("max_position_embeddings")?,
            rms_norm_eps: opt_f("rms_norm_eps", 1e-6),
            rope_theta: opt_f("rope_theta", 10000.0),
            rope_scaling,
            tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
            attention_bias: v["attention_bias"].as_bool().unwrap_or(false),
        })
    }

    /// Serialize as a Hugging Face `config.json`.
    pub fn to_json(&self) -> String {
        let rope_scaling = match self.rope_scaling {
            None => Value::Null,
            Some(s) => serde_json::json!({
                "rope_type": "llama3",
                "factor": s.factor,
                "low_freq_factor": s.low_freq_factor,
                "high_freq_factor": s.high_freq_factor,
                "original_max_position_embeddings": s.original_max_position_embeddings,
            }),
        };
        let v = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "rope_scaling": rope_scaling,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
            "hidden_act": "silu",
        });
        serde_json::to_string_pretty(&v).unwrap()
    }

    /// RoPE inverse frequencies, including Llama 3 scaling if configured.
    pub fn rope_inv_freq(&self) -> Vec<f32> {
        let base = F::rope_inv_freq(self.head_dim, self.rope_theta);
        let Some(s) = self.rope_scaling else { return base };
        // Hugging Face `_compute_llama3_parameters`
        let old = s.original_max_position_embeddings as f32;
        let low_wavelen = old / s.low_freq_factor;
        let high_wavelen = old / s.high_freq_factor;
        base.into_iter()
            .map(|f| {
                let wavelen = 2.0 * std::f32::consts::PI / f;
                if wavelen < high_wavelen {
                    f
                } else if wavelen > low_wavelen {
                    f / s.factor
                } else {
                    let smooth = (old / wavelen - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor);
                    (1.0 - smooth) * f / s.factor + smooth * f
                }
            })
            .collect()
    }
}

#[derive(Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

#[derive(Clone)]
struct DecoderLayer {
    input_layernorm: RMSNorm,
    self_attn: Attention,
    post_attention_layernorm: RMSNorm,
    mlp: SwiGLU,
}

/// Llama-style decoder-only language model.
pub struct Llama {
    /// Architecture hyperparameters.
    pub config: LlamaConfig,
    /// Token embeddings `[vocab, hidden]`.
    pub embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    /// Final RMSNorm.
    pub norm: RMSNorm,
    /// Untied output head (`None` when tied to `embed_tokens`).
    pub lm_head: Option<Linear>,
    /// int8 copy of a tied head, set by [`Llama::quantize_int8`].
    head_int8: Option<Linear>,
    inv_freq: Vec<f32>,
    gradient_checkpointing: AtomicBool,
}

impl Attention {
    fn linears(&self) -> [(&'static str, &Linear); 4] {
        [("q_proj", &self.q_proj), ("k_proj", &self.k_proj), ("v_proj", &self.v_proj), ("o_proj", &self.o_proj)]
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, kv: Option<&mut LayerKv>) -> Tensor {
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let hd = self.head_dim;
        let heads = |y: Tensor, h: usize| y.reshape(&[b, t, h, hd]).permute(&[0, 2, 1, 3]);
        let q = F::apply_rope(&heads(self.q_proj.forward(x), self.n_heads), cos, sin);
        let k = F::apply_rope(&heads(self.k_proj.forward(x), self.n_kv_heads), cos, sin);
        let v = heads(self.v_proj.forward(x), self.n_kv_heads);
        // the cache holds the (fewer) key/value heads, before repetition
        let (k, v, past) = match kv {
            Some(kv) => {
                let past = kv.len();
                let (k, v) = kv.append(k, v);
                (k, v, past)
            }
            None => (k, v, 0),
        };
        let rep = self.n_heads / self.n_kv_heads;
        let (k, v) = (F::repeat_kv(&k, rep), F::repeat_kv(&v, rep));
        let mask = (t > 1).then(|| F::causal_mask_offset(t, past, x.device()));
        let o = F::scaled_dot_product_attention(&q, &k, &v, mask.as_ref(), 0.0, false);
        self.o_proj.forward(&o.permute(&[0, 2, 1, 3]).reshape(&[b, t, self.n_heads * hd]))
    }
}

impl DecoderLayer {
    /// The seven projection matrices with their names.
    fn linears_mut(&mut self) -> [(&'static str, &mut Linear); 7] {
        let a = &mut self.self_attn;
        let m = &mut self.mlp;
        [
            ("self_attn.q_proj", &mut a.q_proj),
            ("self_attn.k_proj", &mut a.k_proj),
            ("self_attn.v_proj", &mut a.v_proj),
            ("self_attn.o_proj", &mut a.o_proj),
            ("mlp.gate_proj", &mut m.gate_proj),
            ("mlp.up_proj", &mut m.up_proj),
            ("mlp.down_proj", &mut m.down_proj),
        ]
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut v = self.input_layernorm.parameters();
        for (_, l) in self.self_attn.linears() {
            v.extend(l.parameters());
        }
        v.extend(self.post_attention_layernorm.parameters());
        v.extend(self.mlp.parameters());
        v
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, kv: Option<&mut LayerKv>) -> Tensor {
        let x = x + self.self_attn.forward(&self.input_layernorm.forward(x), cos, sin, kv);
        &x + self.mlp.forward(&self.post_attention_layernorm.forward(&x))
    }
}

/// Whether a projection path like `self_attn.q_proj` matches a LoRA target
/// (equal to one of its dot-separated components, or to the whole path).
pub(crate) fn lora_target(path: &str, targets: &[&str]) -> bool {
    targets.iter().any(|t| path == *t || path.split('.').any(|c| c == *t))
}

/// Hugging Face stores these as `[out, in]`; this crate as `[in, out]`.
fn is_hf_linear(name: &str, t: &Tensor) -> bool {
    t.rank() == 2 && (name.ends_with("_proj.weight") || name == "lm_head.weight")
}

impl Llama {
    /// Randomly initialized model.
    pub fn new(config: LlamaConfig, device: Device) -> Self {
        let c = config;
        assert_eq!(c.num_attention_heads % c.num_key_value_heads, 0, "heads must be a multiple of kv heads");
        let (d, hd) = (c.hidden_size, c.head_dim);
        let lin = |i, o| Linear::with_bias(i, o, c.attention_bias, device);
        let layers = (0..c.num_hidden_layers)
            .map(|_| DecoderLayer {
                input_layernorm: RMSNorm::new(d, c.rms_norm_eps, device),
                self_attn: Attention {
                    q_proj: lin(d, c.num_attention_heads * hd),
                    k_proj: lin(d, c.num_key_value_heads * hd),
                    v_proj: lin(d, c.num_key_value_heads * hd),
                    o_proj: lin(c.num_attention_heads * hd, d),
                    n_heads: c.num_attention_heads,
                    n_kv_heads: c.num_key_value_heads,
                    head_dim: hd,
                },
                post_attention_layernorm: RMSNorm::new(d, c.rms_norm_eps, device),
                mlp: SwiGLU::new(d, c.intermediate_size, device),
            })
            .collect();
        Llama {
            config,
            embed_tokens: Embedding::new(c.vocab_size, d, device),
            layers,
            norm: RMSNorm::new(d, c.rms_norm_eps, device),
            lm_head: (!c.tie_word_embeddings).then(|| Linear::with_bias(d, c.vocab_size, false, device)),
            head_int8: None,
            inv_freq: config.rope_inv_freq(),
            gradient_checkpointing: AtomicBool::new(false),
        }
    }

    /// Load a Hugging Face checkpoint directory: `config.json` plus
    /// `model.safetensors` (or sharded `model-*.safetensors`). Weights of any
    /// float dtype are converted to f32.
    pub fn from_pretrained(dir: impl AsRef<Path>, device: Device) -> io::Result<Self> {
        let dir = dir.as_ref();
        let config = LlamaConfig::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let single = dir.join("model.safetensors");
        let files = if single.exists() {
            vec![single]
        } else {
            let mut v: Vec<_> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    n.starts_with("model-") && n.ends_with(".safetensors")
                })
                .collect();
            v.sort();
            v
        };
        if files.is_empty() {
            return Err(invalid(format!("no model.safetensors in {}", dir.display())));
        }
        let mut tensors = Vec::new();
        for f in files {
            tensors.extend(serialize::load_safetensors(f)?.0);
        }
        let model = Llama::new(config, device);
        model.load_hf_state_dict(tensors)?;
        Ok(model)
    }

    /// Load tensors in Hugging Face layout (strict).
    pub fn load_hf_state_dict(&self, tensors: Vec<(String, Tensor)>) -> io::Result<()> {
        let tied = self.config.tie_word_embeddings;
        let tensors: Vec<(String, Tensor)> = tensors
            .into_iter()
            // rotary tables are recomputed; a tied head may still be exported
            .filter(|(n, _)| !n.ends_with("rotary_emb.inv_freq") && !(tied && n == "lm_head.weight"))
            .map(|(n, t)| {
                let t = if is_hf_linear(&n, &t) { t.t() } else { t };
                (n, t)
            })
            .collect();
        serialize::load_state_dict(self, &tensors, true).map(|_| ())
    }

    /// Parameters in Hugging Face layout (linear weights `[out, in]`).
    pub fn hf_state_dict(&self) -> Vec<(String, Tensor)> {
        assert!(self.head_int8.is_none() && !self.layers.iter().any(|l| l.self_attn.q_proj.is_quantized()), "cannot export an int8-quantized model");
        assert!(!self.named_parameters().iter().any(|(n, _)| n.contains(".lora_")), "merge the LoRA adapters (merge_lora) before exporting");
        self.named_parameters()
            .into_iter()
            .map(|(n, t)| {
                let t = if is_hf_linear(&n, &t) { t.t() } else { t };
                (n, t)
            })
            .collect()
    }

    /// Write `config.json` and `model.safetensors` in Hugging Face format.
    pub fn save_pretrained(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("config.json"), self.config.to_json())?;
        let mut meta = serialize::Metadata::new();
        meta.insert("format".into(), "pt".into());
        serialize::save_safetensors(dir.join("model.safetensors"), &self.hf_state_dict(), Some(meta))
    }

    /// Logits `[B, T, vocab]` for `tokens` `[B, T]` (no cache).
    pub fn forward_tokens(&self, tokens: &Tensor) -> Tensor {
        self.run(tokens, None)
    }

    /// Logits for new tokens after the positions already in `cache` (which is extended).
    pub fn forward_cached(&self, tokens: &Tensor, cache: &mut KvCache) -> Tensor {
        if cache.layers.len() != self.layers.len() {
            *cache = KvCache::new(self.layers.len());
        }
        self.run(tokens, Some(cache))
    }

    fn run(&self, tokens: &Tensor, mut cache: Option<&mut KvCache>) -> Tensor {
        let t = tokens.shape()[1];
        let start = cache.as_ref().map_or(0, |c| c.len());
        let max = self.config.max_position_embeddings;
        assert!(start + t <= max, "context {} exceeds max_position_embeddings {max}", start + t);
        let (cos, sin) = F::rope_cos_sin(&self.inv_freq, start, t, tokens.device());
        let mut x = self.embed_tokens.forward(tokens);
        let checkpoint = cache.is_none() && self.gradient_checkpointing.load(Ordering::Relaxed) && crate::is_grad_enabled();
        for (i, layer) in self.layers.iter().enumerate() {
            x = if checkpoint {
                let l = layer.clone();
                F::grad_checkpoint(&[&x, &cos, &sin], layer.parameters(), move |xs| l.forward(&xs[0], &xs[1], &xs[2], None))
            } else {
                layer.forward(&x, &cos, &sin, cache.as_mut().map(|c| &mut c.layers[i]))
            };
        }
        let x = self.norm.forward(&x);
        match (&self.lm_head, &self.head_int8) {
            (Some(head), _) => head.forward(&x),
            (None, Some(head)) => head.forward(&x),
            (None, None) => x.matmul_nt(&self.embed_tokens.weight),
        }
    }

    /// Generate `n` tokens after `prompt` with a KV cache; returns the prompt
    /// followed by the new tokens.
    pub fn generate(&self, prompt: &[usize], n: usize, sampler: &mut Sampler) -> Vec<usize> {
        self.generate_with(prompt, n, sampler, |_| true)
    }

    /// Like [`Llama::generate`], calling `on_token` after each new token;
    /// returning `false` stops generation (e.g. at an end-of-sequence id).
    pub fn generate_with(&self, prompt: &[usize], n: usize, sampler: &mut Sampler, on_token: impl FnMut(usize) -> bool) -> Vec<usize> {
        let dev = self.embed_tokens.weight.device();
        let max = self.config.max_position_embeddings;
        crate::generation::generate_cached(prompt, n, max, dev, sampler, |t, c| self.forward_cached(t, c), on_token)
    }

    /// Recompute each decoder layer's activations during the backward pass
    /// instead of storing them ([`F::grad_checkpoint`]): much less memory
    /// for training on long sequences, about one extra forward pass of time.
    pub fn set_gradient_checkpointing(&self, on: bool) {
        self.gradient_checkpointing.store(on, Ordering::Relaxed);
    }

    /// Freeze the whole model and attach LoRA adapters ([`Linear::add_lora`])
    /// to every projection whose name matches one of `targets`, e.g.
    /// `&["q_proj", "v_proj"]` or `&["self_attn", "mlp"]` (a target matches a
    /// whole dot-separated component of `self_attn.q_proj`, `mlp.down_proj`, ...).
    /// Returns the number of adapted layers; train `trainable_parameters()`.
    pub fn add_lora(&mut self, rank: usize, alpha: f32, targets: &[&str]) -> usize {
        // freeze everything except adapters added earlier
        for (name, p) in self.named_parameters() {
            if !name.contains(".lora_") {
                p.set_requires_grad(false);
            }
        }
        let mut n = 0;
        for l in &mut self.layers {
            for (name, lin) in l.linears_mut() {
                if lora_target(name, targets) {
                    lin.add_lora(rank, alpha);
                    n += 1;
                }
            }
        }
        assert!(n > 0, "no projection matches the LoRA targets {targets:?}");
        n
    }

    /// Fold all LoRA adapters into their weights ([`Linear::merge_lora`]).
    pub fn merge_lora(&mut self) {
        for l in &mut self.layers {
            for (_, lin) in l.linears_mut() {
                lin.merge_lora();
            }
        }
    }

    /// Weight-only int8 quantization of every linear layer and the output
    /// head (the embedding table stays in its original precision).
    pub fn quantize_int8(&mut self) {
        for l in &mut self.layers {
            let a = &mut l.self_attn;
            for lin in [&mut a.q_proj, &mut a.k_proj, &mut a.v_proj, &mut a.o_proj] {
                lin.quantize_int8();
            }
            for lin in [&mut l.mlp.gate_proj, &mut l.mlp.up_proj, &mut l.mlp.down_proj] {
                lin.quantize_int8();
            }
        }
        match &mut self.lm_head {
            Some(h) => h.quantize_int8(),
            None if self.head_int8.is_none() => {
                let mut head = Linear::from_weight(self.embed_tokens.weight.t(), None);
                head.quantize_int8();
                self.head_int8 = Some(head);
            }
            None => {}
        }
    }
}

impl Module for Llama {
    /// `x`: `[B, T]` token ids -> logits `[B, T, vocab]`.
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_tokens(x)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = prefixed("model.embed_tokens", self.embed_tokens.named_parameters());
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("model.layers.{i}");
            v.extend(prefixed(&format!("{p}.input_layernorm"), l.input_layernorm.named_parameters()));
            for (name, lin) in l.self_attn.linears() {
                v.extend(prefixed(&format!("{p}.self_attn.{name}"), lin.named_parameters()));
            }
            v.extend(prefixed(&format!("{p}.post_attention_layernorm"), l.post_attention_layernorm.named_parameters()));
            v.extend(prefixed(&format!("{p}.mlp"), l.mlp.named_parameters()));
        }
        v.extend(prefixed("model.norm", self.norm.named_parameters()));
        if let Some(h) = &self.lm_head {
            v.extend(prefixed("lm_head", h.named_parameters()));
        }
        v
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut v = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("model.layers.{i}");
            for (name, lin) in l.self_attn.linears() {
                v.extend(prefixed(&format!("{p}.self_attn.{name}"), lin.named_buffers()));
            }
            v.extend(prefixed(&format!("{p}.mlp"), l.mlp.named_buffers()));
        }
        if let Some(h) = self.lm_head.as_ref().or(self.head_int8.as_ref()) {
            v.extend(prefixed("lm_head", h.named_buffers()));
        }
        v
    }
}

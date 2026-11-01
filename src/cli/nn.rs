//! `ruai nn`: the deep learning engine from the command line.

use super::Args;
use ruai::generation::Sampler;
use ruai::nn::gpt2::GPT2;
use ruai::nn::llama::Llama;
use ruai::tokenizer::{Gpt2Tokenizer, HfTokenizer};
use ruai::{no_grad, Device};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const USAGE: &str = "\
usage: ruai nn <command> [args...]

commands:
  devices                 list the compute devices of this build
  info PATH               summarize a model: a .safetensors file or a Hugging Face
                          model directory (config.json + model.safetensors)
  generate --model DIR    continue a prompt with a pretrained GPT-2 or Llama-family model
      --prompt TEXT         text to continue (default: \"Once upon a time\")
      -n N                  tokens to generate (default 60)
      --device D            cpu | mps | cuda[:N] | auto (default: cpu or RUSTNN_DEVICE)
      --int8 | --int4       weight-only quantization
      --temperature T       0 = greedy (default 0)
      --top-k K --top-p P   sampling filters (default off)
      --seed S              sampling seed (default 0)
";

pub fn run(argv: &[String]) -> Result<(), String> {
    let Some((cmd, rest)) = argv.split_first() else {
        print!("{USAGE}");
        return Ok(());
    };
    match cmd.as_str() {
        "devices" => devices(),
        "info" => info(rest),
        "generate" => generate(rest),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown nn command '{other}'\n\n{USAGE}")),
    }
}

fn devices() -> Result<(), String> {
    for (d, built) in [(Device::Cpu, true), (Device::Mps, cfg!(feature = "mps")), (Device::Cuda(0), cfg!(feature = "cuda")), (Device::WebGpu, cfg!(feature = "webgpu"))] {
        let state = if d.is_available() {
            "available"
        } else if built {
            "built in, no device found"
        } else {
            "not built in"
        };
        println!("{:<6} {state}", d.to_string());
    }
    println!("default: {}", Device::from_env());
    Ok(())
}

/// The safetensors files of a model path (a file, or a directory with
/// `model.safetensors` or sharded `model-*.safetensors`).
fn weight_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let single = path.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    v.sort();
    if v.is_empty() {
        return Err(format!("no .safetensors files in {}", path.display()));
    }
    Ok(v)
}

/// `(name, dtype, shape)` of every tensor, from the file header only.
fn header(path: &Path) -> Result<Vec<(String, String, Vec<usize>)>, String> {
    use std::io::Read;
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut f = std::fs::File::open(path).map_err(err)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(err)?;
    let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut buf).map_err(err)?;
    let v: serde_json::Value = serde_json::from_slice(&buf).map_err(|e| format!("{}: bad header: {e}", path.display()))?;
    let mut out = Vec::new();
    for (name, meta) in v.as_object().into_iter().flatten() {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta["dtype"].as_str().unwrap_or("?").to_string();
        let shape = meta["shape"].as_array().into_iter().flatten().filter_map(|x| x.as_u64().map(|x| x as usize)).collect();
        out.push((name.clone(), dtype, shape));
    }
    out.sort();
    Ok(out)
}

fn info(argv: &[String]) -> Result<(), String> {
    let a = Args::parse(argv, &["--all"])?;
    a.check(&["--all"])?;
    let [path] = a.positional.as_slice() else { return Err("usage: ruai nn info PATH [--all]".into()) };
    let path = Path::new(path);
    let config = path.join("config.json");
    if config.exists() {
        let text = std::fs::read_to_string(&config).map_err(|e| e.to_string())?;
        let c: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("config.json: {e}"))?;
        println!("config: model_type {}", c["model_type"].as_str().unwrap_or("?"));
        for k in ["vocab_size", "hidden_size", "n_embd", "num_hidden_layers", "n_layer", "num_attention_heads", "n_head", "num_key_value_heads", "max_position_embeddings", "n_positions", "tie_word_embeddings", "torch_dtype"] {
            if !c[k].is_null() {
                println!("  {k}: {}", c[k]);
            }
        }
    }
    let mut tensors = Vec::new();
    for f in weight_files(path)? {
        tensors.extend(header(&f)?);
    }
    let params: usize = tensors.iter().map(|(_, _, s)| s.iter().product::<usize>()).sum();
    let mut dtypes: Vec<(String, usize)> = Vec::new();
    for (_, d, s) in &tensors {
        let n: usize = s.iter().product();
        match dtypes.iter_mut().find(|(x, _)| x == d) {
            Some((_, c)) => *c += n,
            None => dtypes.push((d.clone(), n)),
        }
    }
    let dt: Vec<String> = dtypes.iter().map(|(d, n)| format!("{d} {:.1}M", *n as f64 / 1e6)).collect();
    // all stored values: parameters plus buffers such as GPT-2's attention masks
    println!("tensors: {}, elements: {params} ({:.1}M), dtypes: {}", tensors.len(), params as f64 / 1e6, dt.join(", "));
    let shown = if a.flag("--all") { tensors.len() } else { tensors.len().min(12) };
    for (name, d, s) in &tensors[..shown] {
        println!("  {name:<50} {d:<5} {s:?}");
    }
    if shown < tensors.len() {
        println!("  ... {} more (--all to list them)", tensors.len() - shown);
    }
    Ok(())
}

enum Model {
    Gpt2(GPT2),
    Llama(Box<Llama>),
}

enum Tok {
    Hf(Box<HfTokenizer>),
    Gpt2(Box<Gpt2Tokenizer>),
}

impl Tok {
    fn encode(&self, s: &str) -> Vec<usize> {
        match self {
            Tok::Hf(t) => t.encode(s, true),
            Tok::Gpt2(t) => t.encode(s),
        }
    }
    fn decode(&self, ids: &[usize]) -> String {
        match self {
            Tok::Hf(t) => t.decode(ids, true),
            Tok::Gpt2(t) => t.decode(ids),
        }
    }
}

fn generate(argv: &[String]) -> Result<(), String> {
    const OPTS: &[&str] = &["--model", "--prompt", "-n", "--device", "--int8", "--int4", "--temperature", "--top-k", "--top-p", "--seed"];
    let a = Args::parse(argv, &["--int8", "--int4"])?;
    a.check(OPTS)?;
    let dir = Path::new(a.get(&["--model"]).ok_or("generate needs --model DIR")?);
    let prompt = a.get(&["--prompt"]).unwrap_or("Once upon a time");
    let n: usize = a.parse_or(&["-n"], 60)?;
    let device: Device = match a.get(&["--device"]) {
        Some(d) => d.parse()?,
        None => Device::from_env(),
    };
    let temperature: f32 = a.parse_or(&["--temperature"], 0.0)?;
    let top_k: usize = a.parse_or(&["--top-k"], 0)?;
    let top_p: f32 = a.parse_or(&["--top-p"], 1.0)?;
    let seed: u64 = a.parse_or(&["--seed"], 0)?;

    let t0 = Instant::now();
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?,
    )
    .map_err(|e| format!("config.json: {e}"))?;
    let kind = config["model_type"].as_str().unwrap_or("");
    let mut model = match kind {
        "gpt2" => Model::Gpt2(GPT2::from_pretrained(dir, device).map_err(|e| e.to_string())?),
        "llama" | "mistral" | "qwen2" | "smollm" => Model::Llama(Box::new(Llama::from_pretrained(dir, device).map_err(|e| e.to_string())?)),
        other => return Err(format!("unsupported model_type '{other}' (expected gpt2 or a Llama-family model)")),
    };
    match (&mut model, a.flag("--int8"), a.flag("--int4")) {
        (Model::Gpt2(m), true, _) => m.quantize_int8(),
        (Model::Gpt2(m), _, true) => m.quantize_int4(32),
        (Model::Llama(m), true, _) => m.quantize_int8(),
        (Model::Llama(m), _, true) => m.quantize_int4(32),
        _ => {}
    }
    let tok = if dir.join("tokenizer.json").exists() {
        Tok::Hf(Box::new(HfTokenizer::from_dir(dir).map_err(|e| e.to_string())?))
    } else {
        Tok::Gpt2(Box::new(Gpt2Tokenizer::from_dir(dir).map_err(|e| e.to_string())?))
    };
    eprintln!("loaded {kind} model from {} on {device} in {:.1}s", dir.display(), t0.elapsed().as_secs_f32());

    let ids = tok.encode(prompt);
    if ids.is_empty() {
        return Err("the prompt encodes to no tokens".into());
    }
    let mut sampler = Sampler::new(temperature, top_k, top_p, seed);
    print!("{prompt}");
    let mut all = ids.clone();
    let mut shown = tok.decode(&ids);
    let t1 = Instant::now();
    let mut on_token = |next: usize| {
        // print the new suffix of the decoded text (keeps multi-byte characters whole)
        all.push(next);
        let text = tok.decode(&all);
        if let Some(new) = text.strip_prefix(&shown) {
            print!("{new}");
            std::io::stdout().flush().ok();
        }
        shown = text;
        true
    };
    let out = no_grad(|| match &model {
        Model::Gpt2(m) => m.generate_with(&ids, n, &mut sampler, &mut on_token),
        Model::Llama(m) => m.generate_with(&ids, n, &mut sampler, &mut on_token),
    });
    let new = out.len() - ids.len();
    let secs = t1.elapsed().as_secs_f64();
    println!();
    eprintln!("{new} tokens in {secs:.2}s ({:.1} tokens/s)", new as f64 / secs.max(1e-9));
    Ok(())
}

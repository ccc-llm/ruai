//! `ruai::world` against Gymnasium, and `ruai::rng` against NumPy.
//!
//! The differential tests run `python3` with `gymnasium` / `numpy` when they
//! are importable (otherwise they print a note and pass); the other tests
//! check fixed values and semantics on their own.

use ruai::rng::Generator;
use ruai::world::{self, Env, RecordEpisodeStatistics, SyncVectorEnv, Value};
use serde_json::Value as Json;
use std::process::Command;

/// Run a Python snippet; `None` when python3 or `module` is unavailable.
fn python(module: &str, script: &str) -> Option<String> {
    let probe = Command::new("python3").args(["-I", "-c", &format!("import {module}")]).output().ok()?;
    if !probe.status.success() {
        eprintln!("skipping: python3 with {module} not available");
        return None;
    }
    let out = Command::new("python3").args(["-I", "-c", script]).output().unwrap();
    assert!(out.status.success(), "python failed:\n{}", String::from_utf8_lossy(&out.stderr));
    Some(String::from_utf8(out.stdout).unwrap())
}

fn value_json(v: &Value) -> Json {
    match v {
        Value::Int(i) => Json::from(*i),
        Value::Floats(x) => Json::from(x.iter().map(|&f| f as f64).collect::<Vec<_>>()),
    }
}

const IDS: [&str; 6] = ["CartPole-v1", "MountainCar-v0", "Pendulum-v1", "Acrobot-v1", "FrozenLake-v1", "FrozenLake8x8-v1"];
const SEEDS: [u64; 4] = [0, 1, 42, 12345];
const STEPS: usize = 400;

/// One record per reset / step: `[action, obs, reward, terminated, truncated]`
/// (action null after a reset), with auto-reset at episode ends.
fn rust_trajectory(id: &str, seed: u64) -> Vec<Json> {
    let mut env = world::make(id).unwrap();
    env.action_space_mut().seed(seed + 100);
    let mut out = Vec::new();
    let (obs, _) = env.reset(Some(seed));
    out.push(serde_json::json!([null, value_json(&obs), 0.0, false, false]));
    for _ in 0..STEPS {
        let a = env.action_space_mut().sample();
        let s = env.step(&a);
        out.push(serde_json::json!([value_json(&a), value_json(&s.obs), s.reward, s.terminated, s.truncated]));
        if s.terminated || s.truncated {
            let (obs, _) = env.reset(None);
            out.push(serde_json::json!([null, value_json(&obs), 0.0, false, false]));
        }
    }
    out
}

const PY_TRAJECTORIES: &str = r#"
import json, numpy as np, gymnasium as gym
def val(x):
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.tolist()]
    return int(x)
for env_id in IDS:
    for seed in SEEDS:
        env = gym.make(env_id)
        env.action_space.seed(seed + 100)
        recs = []
        obs, _ = env.reset(seed=seed)
        recs.append([None, val(obs), 0.0, False, False])
        for _ in range(STEPS):
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            recs.append([val(a), val(obs), float(r), bool(term), bool(trunc)])
            if term or trunc:
                obs, _ = env.reset()
                recs.append([None, val(obs), 0.0, False, False])
        print(json.dumps([env_id, seed, recs]))
"#;

#[test]
fn environments_match_gymnasium() {
    let script = format!("IDS = {IDS:?}\nSEEDS = {SEEDS:?}\nSTEPS = {STEPS}\n{PY_TRAJECTORIES}");
    let Some(out) = python("gymnasium", &script) else { return };
    let mut checked = 0;
    for line in out.lines() {
        let v: Json = serde_json::from_str(line).unwrap();
        let (id, seed) = (v[0].as_str().unwrap(), v[1].as_u64().unwrap());
        let want = v[2].as_array().unwrap();
        let got = rust_trajectory(id, seed);
        assert_eq!(got.len(), want.len(), "{id} seed {seed}: number of records");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g, w, "{id} seed {seed}: record {i} differs");
        }
        checked += 1;
    }
    assert_eq!(checked, IDS.len() * SEEDS.len());
}

#[test]
fn vector_env_autoreset_matches_gymnasium() {
    let script = r#"
import json, gymnasium as gym
envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(3)])
obs, _ = envs.reset(seed=7)
recs = [obs.tolist()]
for t in range(300):
    actions = [(t * 7 + i * 3) % 2 for i in range(3)]
    obs, r, term, trunc, _ = envs.step(actions)
    recs.append([obs.tolist(), r.tolist(), term.tolist(), trunc.tolist()])
print(json.dumps(recs))
"#;
    let Some(out) = python("gymnasium", script) else { return };
    let want: Vec<Json> = serde_json::from_str(out.trim()).unwrap();
    let mut v = SyncVectorEnv::make("CartPole-v1", 3).unwrap();
    let obs = v.reset(Some(7));
    let mut got = vec![Json::from(obs.iter().map(value_json).collect::<Vec<_>>())];
    for t in 0..300usize {
        let actions: Vec<Value> = (0..3).map(|i| Value::Int(((t * 7 + i * 3) % 2) as i64)).collect();
        let s = v.step(&actions);
        got.push(serde_json::json!([s.obs.iter().map(value_json).collect::<Vec<_>>(), s.rewards, s.terminated, s.truncated]));
    }
    assert!(got.iter().skip(1).any(|r| r[2].as_array().unwrap().iter().any(|t| t.as_bool().unwrap())), "episodes should end and autoreset");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "vector step {i}");
    }
}

#[test]
fn frozen_lake_text_matches_gymnasium() {
    let script = r#"
import gymnasium as gym
env = gym.make("FrozenLake-v1", render_mode="ansi")
env.reset(seed=3)
out = [env.render()]
for a in [2, 1, 1, 2, 1, 2]:
    _, _, term, _, _ = env.step(a)
    out.append(env.render())
    if term:
        break
print(repr(out))
"#;
    let Some(out) = python("gymnasium", script) else { return };
    let mut env = world::make("FrozenLake-v1").unwrap();
    env.reset(Some(3));
    let mut got = vec![env.render()];
    for a in [2, 1, 1, 2, 1, 2] {
        let s = env.step(&Value::Int(a));
        got.push(env.render());
        if s.terminated {
            break;
        }
    }
    // Python's repr of a list of str
    let repr = |s: &str| format!("'{}'", s.replace('\\', "\\\\").replace('\n', "\\n").replace('\x1b', "\\x1b").replace('\'', "\\'"));
    let mine = format!("[{}]", got.iter().map(|s| repr(s)).collect::<Vec<_>>().join(", "));
    assert_eq!(mine, out.trim());
}

#[test]
fn numpy_generator_matches_numpy() {
    let script = r#"
import json, numpy as np
out = []
for seed in [0, 1, 42, 2**32 + 5, 2**40 + 123, 2**63 + 7]:
    g = np.random.default_rng(seed)
    vals = []
    for k in range(200):
        if k % 3 == 0:
            vals.append(g.random())
        elif k % 3 == 1:
            n = [2, 3, 4, 7, 100, 2**31 + 11, 2**32][k % 7]
            vals.append(int(g.integers(n)))
        else:
            vals.append(g.uniform(-2.5, 7.0))
    out.append([seed, vals])
print(json.dumps(out))
"#;
    let Some(out) = python("numpy", script) else { return };
    let want: Vec<Json> = serde_json::from_str(out.trim()).unwrap();
    for case in want {
        let seed = case[0].as_u64().unwrap();
        let mut g = Generator::new(seed);
        for (k, w) in case[1].as_array().unwrap().iter().enumerate() {
            let got = match k % 3 {
                0 => Json::from(g.random()),
                1 => {
                    let n = [2i64, 3, 4, 7, 100, (1 << 31) + 11, 1 << 32][k % 7];
                    Json::from(g.integers(0, n))
                }
                _ => Json::from(g.uniform(-2.5, 7.0)),
            };
            assert_eq!(&got, w, "seed {seed} draw {k}");
        }
    }
}

#[test]
fn fixed_values_without_python() {
    // CartPole-v1 reset(seed=42), as returned by gymnasium 1.3
    let mut env = world::make("CartPole-v1").unwrap();
    let (obs, _) = env.reset(Some(42));
    let want = [0.02739560417830944f64, -0.006112155970185995, 0.03585979342460632, 0.019736802205443382];
    assert_eq!(obs.floats(), &want.map(|x| x as f32));
    // time limit
    let mut env = world::make("Pendulum-v1").unwrap();
    env.reset(Some(0));
    let ends: Vec<bool> = (0..200).map(|_| env.step(&Value::Floats(vec![0.5])).truncated).collect();
    assert!(!ends[198] && ends[199], "Pendulum truncates after 200 steps");
    // episode statistics
    let mut env = RecordEpisodeStatistics::new(world::make("CartPole-v1").unwrap());
    env.reset(Some(1));
    let mut n = 0;
    loop {
        n += 1;
        let s = env.step(&Value::Int(1));
        if s.terminated || s.truncated {
            assert_eq!(s.info["episode_length"], n as f64);
            assert_eq!(s.info["episode_return"], n as f64);
            break;
        }
    }
    assert!(world::make("NoSuchEnv-v0").is_err());
}

#[test]
fn spaces() {
    let env = world::make("Pendulum-v1").unwrap();
    assert_eq!(env.action_space().to_string(), "Box[1]");
    let mut s = world::make("Acrobot-v1").unwrap();
    s.action_space_mut().seed(5);
    for _ in 0..50 {
        let a = s.action_space_mut().sample();
        assert!(s.action_space().contains(&a));
    }
    assert!(!s.action_space().contains(&Value::Int(3)));
    assert_eq!(s.observation_space().size(), 6);
}

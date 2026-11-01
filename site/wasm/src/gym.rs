//! The Gymnasium environments of ruai::world, stepped from the page (played
//! with the keyboard, or by a random agent).

use ruai::world::{self, Env, SpaceKind, Value};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

thread_local! {
    static GYM: RefCell<Option<Box<dyn Env>>> = const { RefCell::new(None) };
}

/// Make the environment `id`: its observation and action spaces, described.
#[wasm_bindgen]
pub fn gym_make(id: &str) -> Result<String, JsValue> {
    let env = world::make(id).map_err(|e| JsValue::from_str(&e))?;
    let desc = format!("observation {} · action {}", env.observation_space(), env.action_space());
    GYM.with(|g| *g.borrow_mut() = Some(env));
    Ok(desc)
}

/// Start an episode: the observation.
#[wasm_bindgen]
pub fn gym_reset(seed: u32) -> Vec<f32> {
    GYM.with(|g| g.borrow_mut().as_mut().expect("gym_make first").reset(Some(seed as u64)).0.to_f32_vec())
}

/// Apply an action (an index, or a value for a continuous action):
/// [reward, terminated, truncated, observation…].
#[wasm_bindgen]
pub fn gym_step(action: f64) -> Vec<f64> {
    GYM.with(|g| {
        let mut g = g.borrow_mut();
        let env = g.as_mut().expect("gym_make first");
        let a = match &env.action_space().kind {
            SpaceKind::Discrete { .. } => Value::Int(action as i64),
            SpaceKind::Box { low, high, .. } => Value::Floats(vec![(action as f32).clamp(low[0], high[0])]),
        };
        let s = env.step(&a);
        let mut out = vec![s.reward, f64::from(u8::from(s.terminated)), f64::from(u8::from(s.truncated))];
        out.extend(s.obs.to_f32_vec().iter().map(|&v| v as f64));
        out
    })
}

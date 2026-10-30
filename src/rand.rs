//! Global, seedable random number generator (SplitMix64), so runs are reproducible.

use std::sync::Mutex;

static STATE: Mutex<u64> = Mutex::new(0x853c_49e6_748f_ea9b);

#[inline]
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline]
fn unit(state: &mut u64) -> f32 {
    // 24 random bits -> [0, 1)
    (splitmix(state) >> 40) as f32 / (1u64 << 24) as f32
}

pub fn manual_seed(seed: u64) {
    *STATE.lock().unwrap() = seed;
}

/// `n` samples from U[low, high).
pub fn uniform_vec(n: usize, low: f32, high: f32) -> Vec<f32> {
    let mut s = STATE.lock().unwrap();
    (0..n).map(|_| low + (high - low) * unit(&mut s)).collect()
}

/// `n` samples from N(mean, std^2) (Box-Muller).
pub fn normal_vec(n: usize, mean: f32, std: f32) -> Vec<f32> {
    let mut s = STATE.lock().unwrap();
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = unit(&mut s).max(1e-7);
        let u2 = unit(&mut s);
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f32::consts::PI * u2;
        out.push(mean + std * r * t.cos());
        if out.len() < n {
            out.push(mean + std * r * t.sin());
        }
    }
    out
}

/// Uniform integer in `0..n`.
pub fn randint(n: usize) -> usize {
    let mut s = STATE.lock().unwrap();
    (splitmix(&mut s) % n as u64) as usize
}

/// In-place Fisher-Yates shuffle.
pub fn shuffle<T>(xs: &mut [T]) {
    for i in (1..xs.len()).rev() {
        let j = randint(i + 1);
        xs.swap(i, j);
    }
}

/// Sample an index from a probability vector.
pub fn sample_categorical(probs: &[f32]) -> usize {
    let r = uniform_vec(1, 0.0, 1.0)[0] * probs.iter().sum::<f32>();
    let mut acc = 0.0;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i;
        }
    }
    probs.len() - 1
}

//! Seedable random number generation (SplitMix64), so runs are reproducible.
//!
//! The free functions use one global generator (parameter init, dropout, ...);
//! its state can be saved and restored with [`state`] / [`set_state`]
//! (checkpoints do this). Independent streams (e.g. data shuffling) use their
//! own [`Rng`].

use std::sync::Mutex;

/// A small, fast, seedable generator (SplitMix64).
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A generator starting from `seed`.
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    /// Internal state (pass to [`Rng::new`] to resume the stream).
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 24 random bits.
    #[inline]
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform integer in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// In-place Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }
}

static GLOBAL: Mutex<Rng> = Mutex::new(Rng { state: 0x853c_49e6_748f_ea9b });

/// Reseed the global generator (like `torch.manual_seed`).
pub fn manual_seed(seed: u64) {
    GLOBAL.lock().unwrap().state = seed;
}

/// Current state of the global generator (to save in a checkpoint).
pub fn state() -> u64 {
    GLOBAL.lock().unwrap().state
}

/// Restore the global generator to a saved state.
pub fn set_state(state: u64) {
    GLOBAL.lock().unwrap().state = state;
}

/// `n` samples from U[low, high).
pub fn uniform_vec(n: usize, low: f32, high: f32) -> Vec<f32> {
    let mut r = GLOBAL.lock().unwrap();
    (0..n).map(|_| low + (high - low) * r.unit()).collect()
}

/// `n` samples from N(mean, std^2) (Box-Muller).
pub fn normal_vec(n: usize, mean: f32, std: f32) -> Vec<f32> {
    let mut r = GLOBAL.lock().unwrap();
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = r.unit().max(1e-7);
        let u2 = r.unit();
        let rad = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f32::consts::PI * u2;
        out.push(mean + std * rad * t.cos());
        if out.len() < n {
            out.push(mean + std * rad * t.sin());
        }
    }
    out
}

/// Uniform integer in `0..n`.
pub fn randint(n: usize) -> usize {
    GLOBAL.lock().unwrap().below(n)
}

/// In-place Fisher-Yates shuffle.
pub fn shuffle<T>(xs: &mut [T]) {
    GLOBAL.lock().unwrap().shuffle(xs)
}

/// Sample an index from a (not necessarily normalized) probability vector.
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

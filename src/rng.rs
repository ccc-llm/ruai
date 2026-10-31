//! NumPy-compatible random numbers: `SeedSequence`, the `PCG64` bit
//! generator and the parts of `numpy.random.Generator` that ruai uses, bit
//! for bit (`Generator::new(42)` produces what `np.random.default_rng(42)`
//! produces). Gymnasium environments and the py0 numpy package draw from it,
//! so seeded runs match their Python counterparts.

/// `numpy.random.SeedSequence`: hashes a seed into well-mixed state words.
#[derive(Clone, Debug)]
pub struct SeedSequence {
    pool: [u32; 4],
}

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

impl SeedSequence {
    /// From an integer seed (split into 32-bit words, least significant first).
    pub fn new(seed: u128) -> Self {
        let mut entropy = Vec::new();
        let mut s = seed;
        loop {
            entropy.push(s as u32);
            s >>= 32;
            if s == 0 {
                break;
            }
        }
        let mut pool = [0u32; 4];
        let mut hc = INIT_A;
        for (i, p) in pool.iter_mut().enumerate() {
            *p = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hc);
        }
        for src in 0..4 {
            for dst in 0..4 {
                if src != dst {
                    pool[dst] = mix(pool[dst], hashmix(pool[src], &mut hc));
                }
            }
        }
        for &e in entropy.iter().skip(4) {
            for p in pool.iter_mut() {
                *p = mix(*p, hashmix(e, &mut hc));
            }
        }
        SeedSequence { pool }
    }

    /// `generate_state(n, np.uint64)`.
    pub fn generate_state_u64(&self, n: usize) -> Vec<u64> {
        let mut hc = INIT_B;
        let words: Vec<u32> = (0..2 * n)
            .map(|i| {
                let mut v = self.pool[i % 4] ^ hc;
                hc = hc.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hc);
                v ^ (v >> XSHIFT)
            })
            .collect();
        words.chunks(2).map(|w| w[0] as u64 | (w[1] as u64) << 32).collect()
    }
}

const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

/// `numpy.random.PCG64`: a 128-bit LCG with the XSL-RR output function.
#[derive(Clone, Debug)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    /// The upper half of the last 64-bit draw, kept for the next 32-bit one.
    buffered: Option<u32>,
}

impl Pcg64 {
    /// Seeded like `np.random.PCG64(SeedSequence(seed))`.
    pub fn from_seed_sequence(ss: &SeedSequence) -> Self {
        let v = ss.generate_state_u64(4);
        let seed = (v[0] as u128) << 64 | v[1] as u128;
        let initseq = (v[2] as u128) << 64 | v[3] as u128;
        let mut p = Pcg64 { state: 0, inc: (initseq << 1) | 1, buffered: None };
        p.step();
        p.state = p.state.wrapping_add(seed);
        p.step();
        p
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    /// The next 32 random bits (half of a 64-bit draw, the other half buffered).
    pub fn next_u32(&mut self) -> u32 {
        if let Some(b) = self.buffered.take() {
            return b;
        }
        let x = self.next_u64();
        self.buffered = Some((x >> 32) as u32);
        x as u32
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// The subset of `numpy.random.Generator` used by ruai.
#[derive(Clone, Debug)]
pub struct Generator {
    bits: Pcg64,
}

impl Generator {
    /// `np.random.default_rng(seed)`.
    pub fn new(seed: u64) -> Self {
        Generator { bits: Pcg64::from_seed_sequence(&SeedSequence::new(seed as u128)) }
    }

    /// Seeded from the clock and process id (like `default_rng()` with OS entropy).
    pub fn from_entropy() -> Self {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        Generator { bits: Pcg64::from_seed_sequence(&SeedSequence::new(t ^ ((std::process::id() as u128) << 96))) }
    }

    /// The underlying bit generator.
    pub fn bit_generator(&mut self) -> &mut Pcg64 {
        &mut self.bits
    }

    /// `random()`: uniform in `[0, 1)`.
    pub fn random(&mut self) -> f64 {
        self.bits.next_f64()
    }

    /// `uniform(low, high)`: `low + (high - low) * random()`.
    ///
    /// NumPy's C code is built with floating-point contraction, so on aarch64
    /// (which has FMA in its baseline) this is one fused multiply-add, and on
    /// x86-64 (which does not) a multiply then an add; ruai does the same to
    /// stay bit-identical on both.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        let (scale, r) = (high - low, self.bits.next_f64());
        if cfg!(target_arch = "aarch64") {
            scale.mul_add(r, low)
        } else {
            low + scale * r
        }
    }

    /// `integers(low, high)` for `high - low <= 2^32` (numpy's bounded Lemire
    /// method on 32-bit draws), `high` exclusive.
    pub fn integers(&mut self, low: i64, high: i64) -> i64 {
        assert!(high > low, "integers: high must be greater than low");
        let rng = (high - low - 1) as u64;
        assert!(rng <= u32::MAX as u64, "integers: ranges above 2^32 are not supported");
        if rng == 0 {
            return low;
        }
        if rng == u32::MAX as u64 {
            return low + self.bits.next_u32() as i64;
        }
        let excl = rng as u32 + 1;
        let mut m = self.bits.next_u32() as u64 * excl as u64;
        let mut leftover = m as u32;
        if leftover < excl {
            let threshold = (u32::MAX - rng as u32) % excl;
            while leftover < threshold {
                m = self.bits.next_u32() as u64 * excl as u64;
                leftover = m as u32;
            }
        }
        low + (m >> 32) as i64
    }
}

/// The 32-bit Mersenne Twister seeded by `init_genrand` (Matsumoto and
/// Nishimura): the generator of NumPy's legacy `RandomState(seed)` and of
/// PyTorch's CPU generator (`torch.manual_seed`).
#[derive(Clone)]
pub struct Mt19937 {
    mt: [u32; 624],
    i: usize,
}

impl std::fmt::Debug for Mt19937 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mt19937")
    }
}

impl Mt19937 {
    /// Seeded with the low 32 bits of `seed`.
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, i: 624 }
    }

    /// The next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(k + 397) % 624] ^ (y >> 1);
                if y & 1 == 1 {
                    v ^= 0x9908_b0df;
                }
                self.mt[k] = v;
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // values printed by numpy 2.4 for the same calls
    #[test]
    fn matches_numpy() {
        let s = SeedSequence::new(42).generate_state_u64(4);
        assert_eq!(s, [11465652750463011511, 15382171918060459190, 9018504550953525431, 3703499796004394495]);
        let mut g = Generator::new(42);
        let r: Vec<f64> = (0..3).map(|_| g.random()).collect();
        assert_eq!(r, [0.7739560485559633, 0.4388784397520523, 0.8585979199113825]);
        assert_eq!([g.uniform(-0.05, 0.05), g.uniform(-0.05, 0.05)], [0.019736802905936393, -0.04058226521123505]);
        let ints: Vec<i64> = (0..5).map(|_| g.integers(0, 4)).collect();
        assert_eq!(ints, [2, 3, 2, 3, 2]);
    }
}

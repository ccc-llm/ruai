//! `FrozenLake-v1`: walk from S to G on a grid without stepping into a hole
//! (H); on the slippery lake the intended move happens with probability 1/3
//! and each perpendicular move with probability 1/3.

use super::{Env, EnvRng, Info, Space, Step, Value};
use crate::rng::Generator;

const MAP_4X4: [&str; 4] = ["SFFF", "FHFH", "FFFH", "HFFG"];
const MAP_8X8: [&str; 8] = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"];

/// `(probability, next state, reward, terminated)`.
type Transition = (f64, usize, f64, bool);

/// The frozen-lake grid world (actions: 0 left, 1 down, 2 right, 3 up).
pub struct FrozenLake {
    desc: Vec<Vec<u8>>,
    ncol: usize,
    /// `p[s][a]`: the possible transitions.
    p: Vec<Vec<Vec<Transition>>>,
    initial: Vec<f64>,
    s: usize,
    last_action: Option<usize>,
    rng: EnvRng,
    obs_space: Space,
    act_space: Space,
}

/// Gymnasium's `categorical_sample`: the first index whose cumulative
/// probability exceeds one uniform draw.
fn categorical_sample(probs: &[f64], g: &mut Generator) -> usize {
    let r = g.random();
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if acc > r {
            return i;
        }
    }
    0
}

impl FrozenLake {
    /// A lake from a named map (`"4x4"` or `"8x8"`), slippery or not.
    pub fn new(map: &str, slippery: bool) -> Self {
        let rows: Vec<&str> = match map {
            "4x4" => MAP_4X4.to_vec(),
            "8x8" => MAP_8X8.to_vec(),
            other => panic!("unknown FrozenLake map '{other}' (4x4 or 8x8)"),
        };
        Self::from_desc(&rows, slippery)
    }

    /// A lake from rows of `S`, `F`, `H`, `G`.
    pub fn from_desc(rows: &[&str], slippery: bool) -> Self {
        let desc: Vec<Vec<u8>> = rows.iter().map(|r| r.bytes().collect()).collect();
        let (nrow, ncol) = (desc.len(), desc[0].len());
        let ns = nrow * ncol;
        let success = 1.0 / 3.0;
        let fail = (1.0 - success) / 2.0;
        let inc = |row: usize, col: usize, a: usize| match a {
            0 => (row, col.saturating_sub(1)),
            1 => ((row + 1).min(nrow - 1), col),
            2 => (row, (col + 1).min(ncol - 1)),
            _ => (row.saturating_sub(1), col),
        };
        let outcome = |row: usize, col: usize, a: usize| {
            let (r, c) = inc(row, col, a);
            let letter = desc[r][c];
            (r * ncol + c, if letter == b'G' { 1.0 } else { 0.0 }, letter == b'G' || letter == b'H')
        };
        let mut p = vec![vec![Vec::new(); 4]; ns];
        for (row, line) in desc.iter().enumerate() {
            for (col, &letter) in line.iter().enumerate() {
                let s = row * ncol + col;
                for (a, li) in p[s].iter_mut().enumerate() {
                    if matches!(letter, b'G' | b'H') {
                        li.push((1.0, s, 0.0, true));
                    } else if slippery {
                        for b in [(a + 3) % 4, a, (a + 1) % 4] {
                            let (ns, r, t) = outcome(row, col, b);
                            li.push((if b == a { success } else { fail }, ns, r, t));
                        }
                    } else {
                        let (ns, r, t) = outcome(row, col, a);
                        li.push((1.0, ns, r, t));
                    }
                }
            }
        }
        let starts = desc.iter().flatten().filter(|&&c| c == b'S').count() as f64;
        let initial = desc.iter().flatten().map(|&c| if c == b'S' { 1.0 / starts } else { 0.0 }).collect();
        FrozenLake {
            ncol,
            p,
            initial,
            s: 0,
            last_action: None,
            rng: EnvRng::default(),
            obs_space: Space::discrete(ns as i64),
            act_space: Space::discrete(4),
            desc,
        }
    }

    /// The transitions `(probability, next state, reward, terminated)` of
    /// action `a` in state `s` (the model, for planning methods).
    pub fn transitions(&self, s: usize, a: usize) -> &[(f64, usize, f64, bool)] {
        &self.p[s][a]
    }
}

impl Env for FrozenLake {
    fn id(&self) -> &str {
        if self.desc.len() == 8 {
            "FrozenLake8x8-v1"
        } else {
            "FrozenLake-v1"
        }
    }
    fn observation_space(&self) -> &Space {
        &self.obs_space
    }
    fn action_space(&self) -> &Space {
        &self.act_space
    }
    fn action_space_mut(&mut self) -> &mut Space {
        &mut self.act_space
    }

    fn reset(&mut self, seed: Option<u64>) -> (Value, Info) {
        self.rng.reseed(seed);
        self.s = categorical_sample(&self.initial, self.rng.get());
        self.last_action = None;
        (Value::Int(self.s as i64), Info::from([("prob".to_string(), 1.0)]))
    }

    fn step(&mut self, action: &Value) -> Step {
        assert!(self.act_space.contains(action), "{action:?} invalid");
        let a = action.int() as usize;
        let probs: Vec<f64> = self.p[self.s][a].iter().map(|t| t.0).collect();
        let i = categorical_sample(&probs, self.rng.get());
        let (prob, s, reward, terminated) = self.p[self.s][a][i];
        self.s = s;
        self.last_action = Some(a);
        Step { obs: Value::Int(s as i64), reward, terminated, truncated: false, info: Info::from([("prob".to_string(), prob)]) }
    }

    /// Gymnasium's `"ansi"` rendering: the grid with the agent's cell on a red background.
    fn render(&self) -> String {
        let (row, col) = (self.s / self.ncol, self.s % self.ncol);
        let mut out = match self.last_action {
            Some(a) => format!("  ({})\n", ["Left", "Down", "Right", "Up"][a]),
            None => "\n".to_string(),
        };
        for (r, line) in self.desc.iter().enumerate() {
            for (c, &ch) in line.iter().enumerate() {
                let ch = ch as char;
                if (r, c) == (row, col) {
                    out += &format!("\x1b[41m{ch}\x1b[0m");
                } else {
                    out.push(ch);
                }
            }
            out.push('\n');
        }
        out
    }
}

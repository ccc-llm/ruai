//! Reinforcement learning with ruai: REINFORCE on Gymnasium's CartPole-v1
//! (`ruai::world`), with a policy network from `ruai::nn`.
//!
//!     cargo run --release --example cartpole
//!
//! Each update plays a batch of episodes with the current policy, then
//! raises the log-probability of every action in proportion to the
//! (normalized) discounted return that followed it. CartPole is solved at an
//! average return of 475; this usually gets there within a few hundred updates.

use ruai::nn::{self, Module};
use ruai::optim::{Adam, Optimizer};
use ruai::prelude::*;
use ruai::rand;
use ruai::world::{self, Value};

fn main() {
    rand::manual_seed(0);
    let dev = Device::Cpu;
    let mut env = world::make("CartPole-v1").unwrap();
    let policy = nn::Sequential::new().add(nn::Linear::new(4, 64, dev)).add(nn::Tanh).add(nn::Linear::new(64, 2, dev));
    let mut opt = Adam::new(policy.parameters(), 1e-2);
    let mut sampler = ruai::rng::Generator::new(1);
    let (gamma, batch) = (0.99f32, 8);
    let mut seed = 0u64;
    for update in 0..300 {
        let (mut obs_all, mut act_all, mut ret_all, mut lengths) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..batch {
            let (mut obs, _) = env.reset(Some(seed));
            seed += 1;
            let mut rewards = Vec::new();
            loop {
                let x = Tensor::from_vec(obs.floats().to_vec(), &[1, 4], dev);
                let p = no_grad(|| F::softmax(&policy.forward(&x), -1)).to_vec();
                let a = if sampler.random() < p[0] as f64 { 0 } else { 1 };
                obs_all.extend_from_slice(obs.floats());
                act_all.push(a);
                let s = env.step(&Value::Int(a as i64));
                rewards.push(s.reward as f32);
                obs = s.obs;
                if s.terminated || s.truncated {
                    break;
                }
            }
            // discounted returns, back to front
            let mut g = 0.0;
            let mut rets = vec![0.0; rewards.len()];
            for t in (0..rewards.len()).rev() {
                g = rewards[t] + gamma * g;
                rets[t] = g;
            }
            lengths.push(rewards.len());
            ret_all.extend(rets);
        }
        // normalize the returns: a baseline that keeps the update scale stable
        let n = ret_all.len();
        let mean = ret_all.iter().sum::<f32>() / n as f32;
        let std = (ret_all.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / n as f32).sqrt() + 1e-8;
        let adv = Tensor::from_vec(ret_all.iter().map(|r| (r - mean) / std).collect(), &[n, 1], dev);
        let logp = F::log_softmax(&policy.forward(&Tensor::from_vec(obs_all, &[n, 4], dev)), -1);
        let taken = logp.gather(1, &Tensor::from_indices(&act_all, &[n, 1], dev));
        opt.zero_grad();
        (-(taken * adv).mean_all()).backward();
        opt.step();

        let avg = lengths.iter().sum::<usize>() as f64 / batch as f64;
        if update % 10 == 0 || avg >= 475.0 {
            println!("update {update:3}: average return {avg:6.1}");
        }
        if avg >= 475.0 {
            println!("solved (average return >= 475 over {batch} episodes)");
            break;
        }
    }
}

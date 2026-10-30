//! `ruai world`: Gymnasium-style environments from the command line.

use super::Args;
use ruai::world::{self, REGISTRY};

pub const USAGE: &str = "\
usage: ruai world <command> [args...]

commands:
  list                    the built-in environments with their spaces
  run ID                  run episodes with random actions
      --episodes N          number of episodes (default 3)
      --seed S              seed for the environment and the actions (default 0)
      --render              print the environment after every step
";

pub fn run(argv: &[String]) -> Result<(), String> {
    let Some((cmd, rest)) = argv.split_first() else {
        print!("{USAGE}");
        return Ok(());
    };
    match cmd.as_str() {
        "list" => {
            for r in REGISTRY {
                let env = world::make(r.id)?;
                println!(
                    "{:<17} obs {:<12} actions {:<12} max steps {:<4} {}",
                    r.id,
                    env.observation_space().to_string(),
                    env.action_space().to_string(),
                    r.max_episode_steps,
                    r.description
                );
            }
            Ok(())
        }
        "run" => run_random(rest),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown world command '{other}'\n\n{USAGE}")),
    }
}

fn run_random(argv: &[String]) -> Result<(), String> {
    let a = Args::parse(argv, &["--render"])?;
    a.check(&["--episodes", "--seed", "--render"])?;
    let [id] = a.positional.as_slice() else { return Err("usage: ruai world run ID [--episodes N] [--seed S] [--render]".into()) };
    let episodes: usize = a.parse_or(&["--episodes"], 3)?;
    let seed: u64 = a.parse_or(&["--seed"], 0)?;
    let mut env = world::make(id)?;
    env.action_space_mut().seed(seed);
    let mut total = 0.0;
    for ep in 0..episodes {
        // the first reset is seeded; later episodes continue the generator
        env.reset(if ep == 0 { Some(seed) } else { None });
        if a.flag("--render") {
            print!("{}", env.render());
        }
        let (mut ret, mut len) = (0.0, 0);
        loop {
            let action = env.action_space_mut().sample();
            let s = env.step(&action);
            ret += s.reward;
            len += 1;
            if a.flag("--render") {
                print!("{}", env.render());
            }
            if s.terminated || s.truncated {
                let how = if s.terminated { "terminated" } else { "truncated" };
                println!("episode {}: return {ret}, length {len} ({how})", ep + 1);
                break;
            }
        }
        total += ret;
    }
    println!("mean return over {episodes} episodes: {}", total / episodes.max(1) as f64);
    Ok(())
}

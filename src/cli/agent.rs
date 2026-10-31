//! `ruai agent`: teams of LLM agents (agge). The program is written in py0
//! (`src/agent/*.py`, which CPython runs too) and translated to Rust by
//! `ruai py emit src/agent/agent.py --module` into `agent_gen.rs`; the test
//! `tests/agent.rs` checks that the translation is up to date.

#[allow(clippy::all)]
#[path = "agent_gen.rs"]
mod agent_gen;

/// `ruai agent ARGS...`: the agent program exits by itself on errors.
pub fn main(args: &[String]) -> i32 {
    let argv = std::iter::once("ruai agent".to_string()).chain(args.iter().cloned()).collect();
    ruai::pyrt::stdlib::set_argv(argv);
    agent_gen::main();
    0
}

/// `ruai help agent`
pub fn help() -> Result<(), String> {
    main(&["help".to_string()]);
    Ok(())
}

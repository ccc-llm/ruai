//! Command dispatch: built-in subcommands, then `ruai-NAME` on `PATH`.

#[cfg(feature = "agent")]
mod agent;
#[cfg(any(feature = "nn", feature = "world"))]
mod args;
#[cfg(feature = "nn")]
mod nn;
#[cfg(feature = "py")]
mod py;
#[cfg(feature = "world")]
mod world;

#[cfg(any(feature = "nn", feature = "world"))]
pub use args::Args;

/// Built-in commands: (name, one-line description).
const COMMANDS: &[(&str, &str)] = &[
    #[cfg(feature = "nn")]
    ("nn", "deep learning engine: generate text, inspect models, list devices"),
    #[cfg(feature = "py")]
    ("py", "py0, typed Python compiled to Rust: run, check, emit, build, test"),
    #[cfg(feature = "world")]
    ("world", "reinforcement-learning environments (Gymnasium API): list, run"),
    #[cfg(feature = "agent")]
    ("agent", "teams of LLM agents that write code and books (agge, in py0)"),
    ("help", "show help for ruai or one of its commands"),
    ("version", "print the version"),
];

fn usage() -> String {
    let mut s = String::from("usage: ruai <command> [args...]\n\ncommands:\n");
    for (name, about) in COMMANDS {
        s += &format!("  {name:<9}{about}\n");
    }
    s += "\nAny other NAME runs `ruai-NAME` from PATH with the remaining arguments (like git).\n";
    s += "Run `ruai help <command>` for details.\n";
    s
}

/// Runs the command line (without the program name); returns the exit code.
pub fn run(argv: &[String]) -> i32 {
    let Some((cmd, rest)) = argv.split_first() else {
        print!("{}", usage());
        return 0;
    };
    let result = match cmd.as_str() {
        "-h" | "--help" | "help" => help(rest),
        "-V" | "--version" | "version" => {
            println!("ruai {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        #[cfg(feature = "nn")]
        "nn" => nn::run(rest),
        #[cfg(feature = "world")]
        "world" => world::run(rest),
        #[cfg(feature = "py")]
        "py" => return py::main(rest),
        #[cfg(feature = "agent")]
        "agent" => return agent::main(rest),
        other => return external(other, rest),
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("ruai: {e}");
            1
        }
    }
}

fn help(rest: &[String]) -> Result<(), String> {
    match rest.first().map(String::as_str) {
        None => print!("{}", usage()),
        #[cfg(feature = "nn")]
        Some("nn") => print!("{}", nn::USAGE),
        #[cfg(feature = "world")]
        Some("world") => print!("{}", world::USAGE),
        #[cfg(feature = "py")]
        Some("py") => print!("{}", py::USAGE),
        #[cfg(feature = "agent")]
        Some("agent") => return agent::help(),
        Some(other) => return Err(format!("no help for unknown command '{other}'")),
    }
    Ok(())
}

/// `ruai NAME args...` -> `ruai-NAME args...` from PATH.
fn external(name: &str, rest: &[String]) -> i32 {
    let exe = format!("ruai-{name}");
    let found = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).map(|d| d.join(&exe)).find(|p| p.is_file())
    });
    let Some(path) = found else {
        eprintln!("ruai: '{name}' is not a ruai command and no '{exe}' was found on PATH\n");
        eprint!("{}", usage());
        return 2;
    };
    match std::process::Command::new(&path).args(rest).status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("ruai: cannot run {}: {e}", path.display());
            1
        }
    }
}

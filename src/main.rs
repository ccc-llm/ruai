//! The `ruai` command: git-style subcommands for each part of the crate.

mod cli;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(cli::run(&args));
}

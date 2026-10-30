//! A small argument parser: `--name value`, `-n value`, boolean flags and positionals.

use std::collections::HashMap;

/// Parsed arguments of one subcommand.
pub struct Args {
    /// Arguments that are not options, in order.
    pub positional: Vec<String>,
    options: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    /// Parse `argv`; `bool_flags` are options without a value (e.g. `--int8`).
    pub fn parse(argv: &[String], bool_flags: &[&str]) -> Result<Args, String> {
        let mut a = Args { positional: Vec::new(), options: HashMap::new(), flags: Vec::new() };
        let mut it = argv.iter();
        while let Some(x) = it.next() {
            if x.starts_with('-') && x.len() > 1 && x.parse::<f64>().is_err() {
                if bool_flags.contains(&x.as_str()) {
                    a.flags.push(x.clone());
                } else {
                    let v = it.next().ok_or_else(|| format!("option {x} needs a value"))?;
                    a.options.insert(x.clone(), v.clone());
                }
            } else {
                a.positional.push(x.clone());
            }
        }
        Ok(a)
    }

    /// Whether a boolean flag was given.
    pub fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }

    /// The value of the first of `names` that was given.
    pub fn get(&self, names: &[&str]) -> Option<&str> {
        names.iter().find_map(|n| self.options.get(*n)).map(String::as_str)
    }

    /// A parsed value, or `default` when absent.
    pub fn parse_or<T: std::str::FromStr>(&self, names: &[&str], default: T) -> Result<T, String> {
        match self.get(names) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("bad value '{v}' for {}", names[0])),
        }
    }

    /// Fail on options that the command does not know.
    pub fn check(&self, known: &[&str]) -> Result<(), String> {
        match self.options.keys().chain(&self.flags).find(|k| !known.contains(&k.as_str())) {
            Some(k) => Err(format!("unknown option {k}")),
            None => Ok(()),
        }
    }
}

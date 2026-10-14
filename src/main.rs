//! The `dag` binary: see [`dag::cli`].

use std::process::ExitCode;

use clap::Parser;

fn main() -> ExitCode {
    dag::cli::run(dag::cli::Cli::parse())
}

//! Command-line helpers for integration tests.
//!
//! Include with `#[path = "support/cli.rs"] mod cli;`.
#![allow(dead_code)]

use std::path::Path;
use std::process::{Command, Output};

/// Runs the `dag` binary built for this test profile with `args`.
pub fn run_cli<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_dag"))
        .args(args)
        .env("RUST_LOG", "warn")
        .output()
        .expect("cannot run dag")
}

/// Runs `dag` and asserts the exit code, printing stdout/stderr on failure.
pub fn run_expect<S: AsRef<std::ffi::OsStr>>(args: &[S], code: i32) -> Output {
    let out = run_cli(args);
    assert_eq!(
        out.status.code(),
        Some(code),
        "dag {:?}\nstdout:\n{}\nstderr:\n{}",
        args.iter()
            .map(|a| a.as_ref().to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// Runs `dag` and asserts success.
pub fn run_ok<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Output {
    run_expect(args, 0)
}

/// Reads a JSON file.
pub fn read_json(path: &Path) -> serde_json::Value {
    let s = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

/// Converts a path to a `String` argument.
pub fn p(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

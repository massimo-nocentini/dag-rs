#![doc = include_str!("../docs/crate.md")]
#![deny(unreachable_pub)]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(rustdoc::broken_intra_doc_links)]

// ---- shared core ---------------------------------------------------------------------------
pub mod bitset;
pub mod blocks;
pub mod csr;
pub mod error;
pub mod graph;
pub mod io;
pub mod summary;
pub mod sweep;
pub mod topo;
pub mod window;

// ---- algorithm families -------------------------------------------------------------------------
pub mod cpm;
pub mod dom;
pub mod dp;
pub mod paths;
pub mod reach;
pub mod reduce;
pub mod semiring;
pub mod width;

// ---- command line, documentation, tests ---------------------------------------------------------
pub mod cli;
pub mod references;
#[doc(hidden)]
pub mod testkit;

pub use error::{BlockViolation, DagError};
pub use io::NONE;
pub use window::{Dir, Which, Window};

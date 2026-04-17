//! `ostk-recall` CLI library — re-exports the command handlers as library
//! functions so they can be driven from both the binary and integration
//! tests.

pub mod commands;

pub use commands::{
    InitOutcome, ScanOutcome, VerifyOutcome, default_config_path, init, scan, starter_config,
    verify,
};

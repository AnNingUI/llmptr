//! # translator-infra
//!
//! Infrastructure layer for AI API protocol translation — model registry, thinking config,
//! schema utilities, and common helper functions.
//!
//! Ported from Go's `CLIProxyAPI/internal/{registry,thinking,util}` packages.

pub mod registry;
pub mod signature;
pub mod thinking;
pub mod util;

// Re-exports
pub use registry::*;
pub use thinking::*;
pub use util::*;

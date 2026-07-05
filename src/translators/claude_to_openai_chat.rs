//! Claude Messages → OpenAI ChatCompletions translation.
//!
//! This is the most-used translation direction in the matrix:
//! Claude Code sends `/v1/messages` format → proxy translates to `/v1/chat/completions`.

pub mod request;
pub mod response;

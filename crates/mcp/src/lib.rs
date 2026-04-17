//! ostk-recall-mcp — MCP server exposing the query engine over stdio.

pub mod protocol;
pub mod server;
pub mod tools;

pub use protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
pub use server::{PROTOCOL_VERSION, Server};

#[cfg(test)]
mod tests;

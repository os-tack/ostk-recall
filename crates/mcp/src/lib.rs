//! ostk-recall-mcp — MCP server exposing the query engine over stdio.

pub mod crystallize;
pub mod memory;
pub mod protocol;
pub mod resources;
pub mod server;
pub mod tools;

pub use protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
pub use resources::{ClientId, Resource, ResourceContent, ResourceError, ResourceRegistry};
pub use server::{PROTOCOL_VERSION, Server, writer_task};

#[cfg(test)]
mod tests;

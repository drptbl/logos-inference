//! logos-agents over MCP: a server the gateway starts itself, over stdio, or one already
//! running, such as the Basecamp module's, over HTTP.

use crate::agent::{Agent, JobView, Payment, Provider};
use rmcp::model::CallToolRequestParams;
use rmcp::service::RunningService;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use serde_json::{Map, Value, json};
use tokio::runtime::Runtime;

pub struct McpAgent {
    runtime: Runtime,
    client: RunningService<RoleClient, ()>,
}

impl McpAgent {
    /// Starts `bin` (logos-agents) with this process's environment, so `LOGOS_MCP_*` choose
    /// its wallet, network and discovery peers. Its follow-through, which tells the provider
    /// once a payment lands, runs every 5 seconds rather than its default minute, unless
    /// `LOGOS_AGENTS_FOLLOW_THROUGH_SECS` says otherwise: a client is waiting on each step.
    pub fn spawn(bin: &str) -> Result<McpAgent, String> {
        let runtime = runtime()?;
        let client = runtime.block_on(async {
            let child = TokioChildProcess::new(command(bin))
                .map_err(|e| format!("can't start {bin}: {e}"))?;
            ().serve(child)
                .await
                .map_err(|e| format!("{bin} didn't start as an MCP server: {e}"))
        })?;
        Ok(McpAgent { runtime, client })
    }

    /// Connects to a running MCP endpoint, e.g. `http://127.0.0.1:8766/mcp`.
    pub fn connect(url: &str) -> Result<McpAgent, String> {
        let runtime = runtime()?;
        let client = runtime.block_on(async {
            ().serve(StreamableHttpClientTransport::from_uri(url.to_string()))
                .await
                .map_err(|e| format!("can't reach {url}: {e}"))
        })?;
        Ok(McpAgent { runtime, client })
    }

    /// Whether `address` is one of the wallet's accounts.
    pub fn holds(&self, address: &str) -> Result<bool, String> {
        let accounts = self.tool("wallet_accounts", json!({}))?;
        Ok(accounts["accounts"]
            .as_array()
            .is_some_and(|all| all.iter().any(|a| a["address"] == json!(address))))
    }

    /// Calls a tool and reads its JSON answer. A tool error comes back as its text.
    fn tool(&self, name: &'static str, arguments: Value) -> Result<Value, String> {
        let arguments: Map<String, Value> = match arguments {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        let result = self
            .runtime
            .block_on(
                self.client
                    .call_tool(CallToolRequestParams::new(name).with_arguments(arguments)),
            )
            .map_err(|e| format!("{name}: {e}"))?;
        let text: String = result
            .content
            .iter()
            .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
            .collect();
        if result.is_error == Some(true) {
            return Err(text);
        }
        Ok(result
            .structured_content
            .or_else(|| serde_json::from_str(&text).ok())
            .unwrap_or(Value::String(text)))
    }
}

fn command(bin: &str) -> tokio::process::Command {
    let mut command = tokio::process::Command::new(bin);
    if std::env::var_os("LOGOS_AGENTS_FOLLOW_THROUGH_SECS").is_none() {
        command.env("LOGOS_AGENTS_FOLLOW_THROUGH_SECS", "5");
    }
    command
}

fn runtime() -> Result<Runtime, String> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("can't start a runtime: {e}"))
}

impl Agent for McpAgent {
    fn find(&self, service: &str) -> Result<Vec<Provider>, String> {
        let found = self.tool("service_find", json!({ "service_id": service }))?;
        Ok(found["providers"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|p| p["this_node"] != json!(true))
            .filter_map(|p| {
                Some(Provider {
                    peer_id: p["peer_id"].as_str()?.to_string(),
                    chat_address: p["chat_address"].as_str()?.to_string(),
                })
            })
            .collect())
    }

    fn request(&self, provider: &Provider, service: &str, input: &str) -> Result<String, String> {
        let asked = self.tool(
            "service_request",
            json!({ "chat_address": provider.chat_address, "service_id": service, "input_text": input }),
        )?;
        asked["job"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("service_request answered {asked}"))
    }

    fn job(&self, job: &str) -> Result<Option<JobView>, String> {
        let jobs = self.tool("service_jobs", json!({}))?;
        let Some(found) = jobs["jobs"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|j| j["job"] == json!(job))
        else {
            return Ok(None);
        };
        Ok(Some(JobView {
            state: found["state"].as_str().unwrap_or_default().to_string(),
            price_lepta: found["quote"]["price_lepta"]
                .as_str()
                .and_then(|p| p.parse().ok()),
            output: found.get("output").filter(|o| !o.is_null()).cloned(),
            reason: found["reason"].as_str().map(str::to_string),
        }))
    }

    fn pay(&self, job: &str, from: &str) -> Result<Payment, String> {
        let paid = self.tool("service_pay", json!({ "job": job, "from": from }))?;
        Ok(if paid["status"] == json!("awaiting_approval") {
            Payment::AwaitingApproval
        } else {
            Payment::Sent
        })
    }

    fn cancel(&self, job: &str) -> Result<(), String> {
        self.tool("service_cancel", json!({ "job": job }))
            .map(|_| ())
    }
}

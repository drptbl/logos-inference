//! logos-agents over MCP: a server the gateway starts itself, over stdio, or one already
//! running, such as the Basecamp module's, over HTTP. What its tools answer is read as
//! logos-agents v0.4.0 words it.

use crate::agent::{Agent, Input, JobState, Payment, Provider};
use rmcp::model::CallToolRequestParams;
use rmcp::service::RunningService;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::runtime::Runtime;

pub struct McpAgent {
    // Dropped before the runtime, so the client can still shut its server down.
    client: RunningService<RoleClient, ()>,
    runtime: Runtime,
    /// logos-agents' files root, where storage uploads come from and downloads go.
    files_root: Option<PathBuf>,
    /// The last `service_jobs` listing, shared by every waiting call.
    jobs: Fresh<Result<Value, String>>,
}

/// How long a `service_jobs` listing serves every call before it is asked again.
const JOBS_FRESH: Duration = Duration::from_millis(900);
/// How long a storage upload or download may take, at most.
const TRANSFER_TIMEOUT: Duration = Duration::from_secs(300);
/// How often a storage transfer is checked.
const TRANSFER_POLL: Duration = Duration::from_millis(500);
/// How old a file left in the files root must be before it is cleared.
const LEFTOVER_AGE: Duration = Duration::from_secs(3600);
/// The largest answer read back from storage: logos-agents delivers at most 8 MiB.
const MAX_FETCH: u64 = 8 << 20;

/// How long one tool call may take. `service_pay` can wait a minute for an approval.
const TOOL_TIMEOUT: Duration = Duration::from_secs(90);

/// A tool call that didn't answer: the tool refused, with its message, or the call failed.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolError {
    Refused(String),
    Failed(String),
}

impl ToolError {
    fn text(self) -> String {
        match self {
            ToolError::Refused(e) | ToolError::Failed(e) => e,
        }
    }
}

impl McpAgent {
    /// Starts `bin` (logos-agents) with this process's environment, so `LOGOS_MCP_*` choose
    /// its wallet, network and discovery peers. Its follow-through, which tells the provider
    /// once a payment lands, runs every 5 seconds rather than its default minute, unless
    /// `LOGOS_AGENTS_FOLLOW_THROUGH_SECS` says otherwise: a client is waiting on each step.
    /// `files_root`, if given, becomes its `LOGOS_MCP_FILES_ROOT`, for storage.
    pub fn spawn(bin: &str, files_root: Option<PathBuf>) -> Result<McpAgent, String> {
        let runtime = runtime()?;
        let mut command = command(bin);
        if let Some(root) = &files_root {
            command.env("LOGOS_MCP_FILES_ROOT", root);
        }
        let client = runtime.block_on(async {
            let child =
                TokioChildProcess::new(command).map_err(|e| format!("can't start {bin}: {e}"))?;
            ().serve(child)
                .await
                .map_err(|e| format!("{bin} didn't start as an MCP server: {e}"))
        })?;
        Ok(McpAgent::new(client, runtime, files_root))
    }

    /// Connects to a running MCP endpoint, e.g. `http://127.0.0.1:8766/mcp`. `files_root` is
    /// that server's files root, for storage.
    pub fn connect(url: &str, files_root: Option<PathBuf>) -> Result<McpAgent, String> {
        let runtime = runtime()?;
        let client = runtime.block_on(async {
            ().serve(StreamableHttpClientTransport::from_uri(url.to_string()))
                .await
                .map_err(|e| format!("can't reach {url}: {e}"))
        })?;
        Ok(McpAgent::new(client, runtime, files_root))
    }

    fn new(
        client: RunningService<RoleClient, ()>,
        runtime: Runtime,
        files_root: Option<PathBuf>,
    ) -> McpAgent {
        McpAgent {
            client,
            runtime,
            files_root,
            jobs: Fresh::new(JOBS_FRESH),
        }
    }

    /// The files root, and a new file name there for this process's storage traffic.
    fn scratch(&self, ext: &str) -> Result<(PathBuf, String), String> {
        let root = self
            .files_root
            .as_ref()
            .ok_or("storage needs logos-agents' files root: set --files-root")?;
        let dir = root.join("logos-inference");
        std::fs::create_dir_all(&dir).map_err(|e| format!("can't use the files root: {e}"))?;
        clear_leftovers(&dir, LEFTOVER_AGE);
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let name = format!(
            "logos-inference/{nanos:x}-{}.{ext}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        Ok((root.clone(), name))
    }

    /// Waits for a storage transfer to finish, and returns its record.
    /// Waits until `until` at the latest for a storage transfer to finish, and returns its
    /// record. A download bigger than `cap` is given up on as soon as storage says how big.
    fn transfer(&self, session: &Value, until: Instant, cap: Option<u64>) -> Result<Value, String> {
        let session = session["session_id"]
            .as_str()
            .ok_or("storage gave no session")?
            .to_string();
        let status = || {
            self.tool("storage_transfer_status", json!({ "session_id": session }))
                .map_err(ToolError::text)
        };
        wait_transfer(
            status,
            until.min(Instant::now() + TRANSFER_TIMEOUT),
            cap,
            TRANSFER_POLL,
        )
    }

    /// Checks that `address` is one of the wallet's public agent accounts, which is what
    /// `service_pay` pays from.
    pub fn check_payer(&self, address: &str) -> Result<(), String> {
        let accounts = self
            .tool("wallet_accounts", json!({}))
            .map_err(ToolError::text)?;
        payer(&accounts, address)
    }

    /// Calls a tool and reads its JSON answer.
    fn tool(&self, name: &'static str, arguments: Value) -> Result<Value, ToolError> {
        let arguments: Map<String, Value> = match arguments {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        let call = self
            .client
            .call_tool(CallToolRequestParams::new(name).with_arguments(arguments));
        let result = self
            .runtime
            .block_on(async { tokio::time::timeout(TOOL_TIMEOUT, call).await })
            .map_err(|_| {
                ToolError::Failed(format!(
                    "{name} didn't answer in {}s",
                    TOOL_TIMEOUT.as_secs()
                ))
            })?
            .map_err(|e| ToolError::Failed(format!("{name}: {e}")))?;
        let text: String = result
            .content
            .iter()
            .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
            .collect();
        if result.is_error == Some(true) {
            return Err(ToolError::Refused(text));
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
        let found = self
            .tool("service_find", json!({ "service_id": service }))
            .map_err(ToolError::text)?;
        Ok(providers(&found))
    }

    fn request(&self, provider: &Provider, service: &str, input: &Input) -> Result<String, String> {
        let mut ask = json!({ "chat_address": provider.chat_address, "service_id": service });
        match input {
            Input::Text(text) => ask["input_text"] = json!(text),
            Input::Cid(cid) => ask["input_cid"] = json!(cid),
        }
        let asked = self.tool("service_request", ask).map_err(ToolError::text)?;
        asked["job"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("service_request answered {asked}"))
    }

    /// Read from one `service_jobs` listing at a time, however many calls are waiting.
    fn job(&self, job: &str) -> Result<Option<JobState>, String> {
        let listing = self.jobs.get(|| {
            self.tool("service_jobs", json!({}))
                .map_err(ToolError::text)
        })?;
        Ok(job_state(&listing, job))
    }

    fn pay(&self, job: &str, from: &str) -> Result<Payment, String> {
        payment(self.tool("service_pay", json!({ "job": job, "from": from })))
    }

    fn cancel(&self, job: &str) -> Result<(), String> {
        self.tool("service_cancel", json!({ "job": job }))
            .map(|_| ())
            .map_err(ToolError::text)
    }

    fn upload(&self, text: &str, until: Instant) -> Result<String, String> {
        let (root, name) = self.scratch("txt")?;
        let file = root.join(&name);
        let uploaded = std::fs::write(&file, text)
            .map_err(|e| format!("can't write the prompt for storage: {e}"))
            .and_then(|_| {
                self.tool("storage_upload", json!({ "path": name }))
                    .map_err(ToolError::text)
            })
            .and_then(|session| self.transfer(&session, until, None));
        // Removed whatever happened: an upload still reading it keeps its open handle.
        let _ = std::fs::remove_file(&file);
        uploaded?["cid"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| "storage gave no CID".into())
    }

    fn fetch(&self, cid: &str, until: Instant) -> Result<String, String> {
        let (root, name) = self.scratch("out")?;
        let file = root.join(&name);
        let fetched = self
            .tool("storage_download", json!({ "cid": cid, "path": name }))
            .map_err(ToolError::text)
            .and_then(|session| self.transfer(&session, until, Some(MAX_FETCH)))
            .and_then(|_| read_capped(&file));
        // A download still going lands later: `scratch` clears it once it's old.
        let _ = std::fs::remove_file(&file);
        String::from_utf8(fetched?).map_err(|_| "the answer isn't UTF-8 text".into())
    }
}

fn read_capped(file: &Path) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(file)
        .and_then(|f| f.take(MAX_FETCH + 1).read_to_end(&mut bytes))
        .map_err(|e| format!("can't read the downloaded answer: {e}"))?;
    if bytes.len() as u64 > MAX_FETCH {
        return Err(format!("the answer is over {} MiB", MAX_FETCH >> 20));
    }
    Ok(bytes)
}

/// Polls a storage transfer's `status` until it's done, `until` passes, or a download shows a
/// total over `cap`.
pub fn wait_transfer(
    mut status: impl FnMut() -> Result<Value, String>,
    until: Instant,
    cap: Option<u64>,
    poll: Duration,
) -> Result<Value, String> {
    loop {
        let now = status()?;
        if let (Some(cap), Some(total)) = (cap, now["total"].as_u64())
            && total > cap
        {
            return Err(format!(
                "it's {total} bytes, over the {} MiB taken",
                cap >> 20
            ));
        }
        if now["done"] == json!(true) {
            return if now["success"] == json!(true) {
                Ok(now)
            } else {
                Err(now["error"]
                    .as_str()
                    .unwrap_or("the transfer failed")
                    .to_string())
            };
        }
        if Instant::now() >= until {
            return Err("the transfer didn't finish in time".into());
        }
        std::thread::sleep(poll);
    }
}

/// Removes files in `dir` older than `age`: left by transfers that ran past their wait.
fn clear_leftovers(dir: &Path, age: Duration) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let old = entry
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|at| at.elapsed().is_ok_and(|e| e > age));
        if old {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// A value that is loaded at most once per `for_`, however many threads ask. A thread that
/// finds it stale loads it, while the others wait for that one load rather than each making
/// their own; reading a fresh value never waits on a load.
pub struct Fresh<T: Clone> {
    value: Mutex<Option<(Instant, T)>>,
    turn: Mutex<()>,
    for_: Duration,
}

impl<T: Clone> Fresh<T> {
    pub fn new(for_: Duration) -> Fresh<T> {
        Fresh {
            value: Mutex::new(None),
            turn: Mutex::new(()),
            for_,
        }
    }

    fn fresh(&self) -> Option<T> {
        let value = self.value.lock().unwrap_or_else(|e| e.into_inner());
        value
            .as_ref()
            .filter(|(at, _)| at.elapsed() < self.for_)
            .map(|(_, v)| v.clone())
    }

    pub fn get(&self, load: impl FnOnce() -> T) -> T {
        if let Some(value) = self.fresh() {
            return value;
        }
        let _turn = self.turn.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = self.fresh() {
            return value;
        }
        let value = load();
        *self.value.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((Instant::now(), value.clone()));
        value
    }
}

/// The providers `service_find` lists, other than this node.
pub fn providers(found: &Value) -> Vec<Provider> {
    found["providers"]
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
        .collect()
}

/// A job's state from what `service_jobs` lists, if it lists the job.
pub fn job_state(jobs: &Value, job: &str) -> Option<JobState> {
    let found = jobs["jobs"]
        .as_array()?
        .iter()
        .find(|j| j["job"] == json!(job))?;
    let reason = || {
        found["reason"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| found["state"].as_str().unwrap_or_default().to_string())
    };
    Some(match found["state"].as_str().unwrap_or_default() {
        "quoted" => match found["quote"]["price_lepta"]
            .as_str()
            .and_then(|p| p.parse().ok())
        {
            Some(price_lepta) => JobState::Quoted {
                price_lepta,
                escrow: found["quote"]["escrow"].is_object(),
            },
            None => JobState::Ended("the quote's price can't be read".into()),
        },
        "paid" => JobState::Paid,
        "delivered" => JobState::Delivered(found["output"].clone()),
        "expired" => JobState::Expired,
        "rejected" | "canceled" | "failed" => JobState::Ended(reason()),
        // "requested", or a state this doesn't know: wait and see.
        _ => JobState::Requested,
    })
}

/// What `service_pay` did. A refusal counts as nothing sent unless logos-agents marks it as
/// one where money may have moved, as its own payment code does. Those, a call that failed and
/// an answer this can't read are left to the job's state to settle.
pub fn payment(answer: Result<Value, ToolError>) -> Result<Payment, String> {
    const MAYBE_SENT: &[&str] = &[
        "may have been sent",
        "may be sent",
        "may already have gone out",
        "may have gone out",
        "hasn't landed yet",
        "has just landed",
        "provider wasn't told",
        "another call is working",
        // While finishing a recorded payment, or with the job already past its quote.
        "can't read the job's escrow",
        "not waiting for payment",
    ];
    match answer {
        Ok(paid) => match paid["status"].as_str() {
            Some("awaiting_approval") => Ok(Payment::AwaitingApproval),
            Some("approved_sending") => Ok(Payment::Started),
            _ if paid.get("escrow_open_tx").is_some() || paid["paid"] == json!(true) => {
                Ok(Payment::Started)
            }
            _ => Err(format!("service_pay answered {paid}")),
        },
        Err(ToolError::Refused(e)) if !MAYBE_SENT.iter().any(|m| e.contains(m)) => {
            Ok(Payment::NotSent(e))
        }
        Err(e) => Err(e.text()),
    }
}

/// Whether `address` is a public agent account in what `wallet_accounts` lists.
pub fn payer(accounts: &Value, address: &str) -> Result<(), String> {
    if !address.starts_with("Public/") {
        return Err(format!(
            "{address} isn't a public account: payments come from Public/<base58>"
        ));
    }
    let Some(account) = accounts["accounts"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|a| a["address"] == json!(address))
    else {
        return Err(format!("the wallet holds no account {address}"));
    };
    if account["agent"] != json!(true) {
        return Err(format!("{address} isn't one of the agent's own accounts"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clears_only_files_older_than_the_age() {
        let dir =
            std::env::temp_dir().join(format!("logos-inference-leftovers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (old, new) = (dir.join("old.out"), dir.join("new.out"));
        std::fs::write(&old, "left").unwrap();
        std::fs::write(&new, "fresh").unwrap();
        let hour_ago = SystemTime::now() - Duration::from_secs(7200);
        std::fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(hour_ago)
            .unwrap();
        clear_leftovers(&dir, Duration::from_secs(3600));
        assert!(!old.exists());
        assert!(new.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

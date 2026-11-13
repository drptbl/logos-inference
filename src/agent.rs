//! The buyer's side of logos-agents, as the gateway uses it: the `service_*` tools it calls.

use serde_json::Value;

/// A provider that offers a service, as `service_find` lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct Provider {
    pub peer_id: String,
    pub chat_address: String,
}

/// A job as `service_jobs` shows it to the buyer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JobView {
    /// `requested`, `quoted`, `paid`, `delivered`, `rejected`, `canceled`, `expired` or `failed`.
    pub state: String,
    pub price_lepta: Option<u128>,
    /// `{"text": ...}` or `{"cid": ...}`, once delivered.
    pub output: Option<Value>,
    pub reason: Option<String>,
}

/// What `service_pay` did.
#[derive(Debug, Clone, PartialEq)]
pub enum Payment {
    Sent,
    /// Approvals are on and the user hasn't approved it yet: pay again later to finish.
    AwaitingApproval,
}

/// logos-agents' buyer tools. Errors are the tool's own message.
pub trait Agent: Send + Sync {
    /// Providers of `service`, other than this node.
    fn find(&self, service: &str) -> Result<Vec<Provider>, String>;
    /// Asks `provider` for a job, and returns its id.
    fn request(&self, provider: &Provider, service: &str, input: &str) -> Result<String, String>;
    /// The job, if this agent knows it.
    fn job(&self, job: &str) -> Result<Option<JobView>, String>;
    fn pay(&self, job: &str, from: &str) -> Result<Payment, String>;
    /// Withdraws an unpaid request.
    fn cancel(&self, job: &str) -> Result<(), String>;
}

//! The buyer's side of logos-agents, as the gateway uses it: the `service_*` tools it calls.

use serde_json::Value;

/// A provider that offers a service, as `service_find` lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct Provider {
    pub peer_id: String,
    pub chat_address: String,
}

/// Where a job stands, as `service_jobs` shows it to the buyer.
#[derive(Debug, Clone, PartialEq)]
pub enum JobState {
    /// Asked, not answered yet.
    Requested,
    /// Quoted at this price. `escrow` says whether the quote locks the price in escrow, which
    /// is what lets logos-agents refund it.
    Quoted { price_lepta: u128, escrow: bool },
    /// Paid, and the provider told.
    Paid,
    /// Delivered: `{"text": ...}` or `{"cid": ...}`.
    Delivered(Value),
    /// Over without an answer: turned away, cancelled, or failed after payment, with the reason.
    Ended(String),
    /// The quote expired. A payment sent in time is still honoured.
    Expired,
}

/// A job's input as it travels: in the chat message, or as a Logos Storage CID.
#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    Text(String),
    Cid(String),
}

/// What `service_pay` did.
#[derive(Debug, Clone, PartialEq)]
pub enum Payment {
    /// The payment went out, or is going out. logos-agents finishes it on its own.
    Started,
    /// Waiting for the user's approval. logos-agents sends it on its own once approved.
    AwaitingApproval,
    /// Refused before anything was sent, with the reason.
    NotSent(String),
}

/// logos-agents' buyer tools. An `Err` is the tool's own message.
pub trait Agent: Send + Sync {
    /// Providers of `service`, other than this node.
    fn find(&self, service: &str) -> Result<Vec<Provider>, String>;
    /// Asks `provider` for a job, and returns its id.
    fn request(&self, provider: &Provider, service: &str, input: &Input) -> Result<String, String>;
    /// The job's state, if this agent knows the job.
    fn job(&self, job: &str) -> Result<Option<JobState>, String>;
    /// Pays the job's quote. An `Err` means the money may have gone out: logos-agents never
    /// pays one job twice, and finishes a payment it started on its own.
    fn pay(&self, job: &str, from: &str) -> Result<Payment, String>;
    /// Withdraws an unpaid request.
    fn cancel(&self, job: &str) -> Result<(), String>;
    /// Puts `text` in Logos Storage, and returns its CID.
    fn upload(&self, text: &str) -> Result<String, String>;
    /// The text Logos Storage holds as `cid`.
    fn fetch(&self, cid: &str) -> Result<String, String>;
}

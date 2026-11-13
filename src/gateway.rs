//! One completion bought over Logos: find a provider, keep to the price cap, pay, wait for the
//! answer.

use crate::agent::{Agent, JobView, Payment};
use crate::openai::{ApiError, Call};
use serde_json::Value;
use std::thread;
use std::time::{Duration, Instant};

pub struct Gateway<A: Agent> {
    pub agent: A,
    /// The account every payment comes from.
    pub pay_from: String,
    /// The most one completion may pay.
    pub max_price_lepta: u128,
    /// How long a provider has to quote before the next one is asked.
    pub quote_wait: Duration,
    /// How often a job is checked.
    pub poll: Duration,
}

/// Where a provider left an unpaid request.
enum Quote {
    Price(u128),
    Turned(String),
    Silent,
}

impl<A: Agent> Gateway<A> {
    /// Buys `call`'s answer by `deadline`. Providers are asked one at a time, in the order
    /// found, until one quotes within the cap. Only that one is paid: if its job then fails,
    /// the call fails too, and logos-agents refunds the payment.
    pub fn complete(&self, call: &Call, deadline: Instant) -> Result<String, ApiError> {
        let service = format!("inference/{}/v1", call.model);
        let providers = self
            .agent
            .find(&service)
            .map_err(|e| ApiError::new(502, format!("logos-agents: {e}")))?;
        if providers.is_empty() {
            return Err(ApiError::new(
                503,
                format!("no provider offers {} right now", call.model),
            ));
        }
        let mut lowest_over: Option<u128> = None;
        let mut last_reason = None;
        for provider in &providers {
            if Instant::now() >= deadline {
                break;
            }
            let job = match self.agent.request(provider, &service, &call.input) {
                Ok(job) => job,
                Err(e) => {
                    last_reason = Some(e);
                    continue;
                }
            };
            match self.quote(&job, deadline.min(Instant::now() + self.quote_wait)) {
                Quote::Price(price) if price <= self.max_price_lepta => {
                    return self.pay(&job, deadline);
                }
                Quote::Price(price) => {
                    lowest_over = Some(lowest_over.map_or(price, |p| p.min(price)));
                    let _ = self.agent.cancel(&job);
                }
                Quote::Turned(reason) => last_reason = Some(reason),
                Quote::Silent => {
                    let _ = self.agent.cancel(&job);
                }
            }
        }
        Err(match (lowest_over, last_reason) {
            (Some(lowest), _) => ApiError::new(
                402,
                format!(
                    "every quote was over the cap of {} lepta, the lowest {lowest}",
                    self.max_price_lepta
                ),
            ),
            _ if Instant::now() >= deadline => ApiError::new(504, "no provider quoted in time"),
            (None, Some(reason)) => {
                ApiError::new(503, format!("no provider took the job: {reason}"))
            }
            (None, None) => ApiError::new(503, "no provider quoted in time"),
        })
    }

    fn quote(&self, job: &str, until: Instant) -> Quote {
        loop {
            match self.agent.job(job) {
                Ok(Some(JobView {
                    state,
                    price_lepta: Some(price),
                    ..
                })) if state == "quoted" => {
                    return Quote::Price(price);
                }
                Ok(Some(view)) if !matches!(view.state.as_str(), "requested" | "quoted") => {
                    return Quote::Turned(view.reason.unwrap_or(view.state));
                }
                _ => {}
            }
            if Instant::now() >= until {
                return Quote::Silent;
            }
            thread::sleep(self.poll);
        }
    }

    /// Pays the quoted job, waiting out an approval, then waits for its answer.
    fn pay(&self, job: &str, deadline: Instant) -> Result<String, ApiError> {
        loop {
            match self.agent.pay(job, &self.pay_from) {
                Ok(Payment::Sent) => break,
                Ok(Payment::AwaitingApproval) if Instant::now() < deadline => {
                    thread::sleep(self.poll)
                }
                Ok(Payment::AwaitingApproval) => {
                    return Err(ApiError::new(504, "the payment wasn't approved in time"));
                }
                Err(e) => return Err(ApiError::new(502, format!("the payment failed: {e}"))),
            }
        }
        loop {
            match self.agent.job(job) {
                Ok(Some(view)) if view.state == "delivered" => return answer(view.output),
                Ok(Some(view)) if !matches!(view.state.as_str(), "quoted" | "paid") => {
                    let reason = view.reason.unwrap_or(view.state);
                    return Err(ApiError::new(
                        502,
                        format!("the paid job failed: {reason}. logos-agents refunds the payment"),
                    ));
                }
                _ => {}
            }
            if Instant::now() >= deadline {
                return Err(ApiError::new(
                    504,
                    "paid, but no answer in time. The job goes on: logos-agents keeps its answer, or refunds the payment if nothing comes by the deadline",
                ));
            }
            thread::sleep(self.poll);
        }
    }
}

fn answer(output: Option<Value>) -> Result<String, ApiError> {
    let output = output.unwrap_or_default();
    if let Some(text) = output.get("text").and_then(Value::as_str) {
        return Ok(text.to_string());
    }
    Err(ApiError::new(
        502,
        match output.get("cid").and_then(Value::as_str) {
            Some(cid) => format!(
                "the answer came as Logos Storage CID {cid}, which the gateway can't fetch yet"
            ),
            None => "the answer couldn't be read".into(),
        },
    ))
}

//! One completion bought over Logos: find a provider, keep to the price cap, pay, wait for the
//! answer.

use crate::agent::{Agent, JobState, Payment};
use crate::openai::{ApiError, Call};
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

pub struct Gateway<A: Agent> {
    pub agent: A,
    /// The account every payment comes from.
    pub pay_from: String,
    /// The most one completion may pay.
    pub max_price_lepta: u128,
    /// How long a provider has to quote before the next one is asked, and how long discovery
    /// is given to find one.
    pub quote_wait: Duration,
    /// How often a job is checked.
    pub poll: Duration,
}

/// Where a provider left an unpaid request.
enum Quote {
    Price(u128),
    /// Quoted, but without escrow: still open, and never paid.
    NoEscrow,
    Turned(String),
    Silent,
}

/// Why the gateway is waiting on a job it paid for, which decides what it says if time runs out.
enum Waiting {
    Payment,
    Approval,
    Unclear(String),
}

impl<A: Agent> Gateway<A> {
    /// Buys `call`'s answer by `deadline`. Providers are asked one at a time, in the order
    /// found, until one quotes in escrow within the cap. Only that one is paid, with one
    /// `service_pay`: from then on the job's own state decides, so a failure the gateway
    /// reports never hides money still moving. Nothing more is asked or paid once `gone` says
    /// the client left.
    pub fn complete(
        &self,
        call: &Call,
        deadline: Instant,
        gone: &AtomicBool,
    ) -> Result<String, ApiError> {
        let service = format!("inference/{}/v1", call.model);
        let providers = self.find(&service, deadline.min(Instant::now() + self.quote_wait))?;
        if providers.is_empty() {
            return Err(ApiError::new(
                503,
                format!("no provider offers {} right now", call.model),
            ));
        }
        let left = || ApiError::new(499, "the client went away before anything was paid");
        let mut lowest_over: Option<u128> = None;
        let mut last_reason = None;
        for provider in &providers {
            if gone.load(Ordering::Relaxed) {
                return Err(left());
            }
            if Instant::now() >= deadline {
                break;
            }
            let job = match self.agent.request(provider, &service, &call.input()) {
                Ok(job) => job,
                Err(e) => {
                    last_reason = Some(e);
                    continue;
                }
            };
            match self.quote(&job, deadline.min(Instant::now() + self.quote_wait)) {
                Quote::Price(price) if price <= self.max_price_lepta => {
                    if gone.load(Ordering::Relaxed) {
                        self.cancel(&job);
                        return Err(left());
                    }
                    // A quote that took the whole wait is too late to buy: the client is gone or going.
                    if Instant::now() >= deadline {
                        self.cancel(&job);
                        return Err(ApiError::new(504, "no provider quoted in time"));
                    }
                    return self.pay(&job, deadline);
                }
                Quote::Price(price) => {
                    lowest_over = Some(lowest_over.map_or(price, |p| p.min(price)));
                    self.cancel(&job);
                }
                Quote::NoEscrow => {
                    self.cancel(&job);
                    last_reason =
                        Some("quoted without escrow, so nothing could be refunded".into());
                }
                Quote::Turned(reason) => last_reason = Some(reason),
                Quote::Silent => self.cancel(&job),
            }
        }
        Err(match (lowest_over, last_reason) {
            (Some(lowest), _) => ApiError::new(
                402,
                format!(
                    "every quote was over the cap of {} lepta, the lowest {lowest}",
                    self.max_price_lepta
                ),
            )
            .code("price_over_cap"),
            _ if Instant::now() >= deadline => ApiError::new(504, "no provider quoted in time"),
            (None, Some(reason)) => {
                ApiError::new(503, format!("no provider took the job: {reason}"))
            }
            (None, None) => ApiError::new(503, "no provider quoted in time"),
        })
    }

    /// Providers of `service`, asked again while discovery comes back empty, until `until`.
    fn find(&self, service: &str, until: Instant) -> Result<Vec<crate::agent::Provider>, ApiError> {
        loop {
            let found = self
                .agent
                .find(service)
                .map_err(|e| ApiError::new(502, format!("logos-agents: {e}")))?;
            if !found.is_empty() || Instant::now() >= until {
                return Ok(found);
            }
            thread::sleep(self.poll);
        }
    }

    fn cancel(&self, job: &str) {
        if let Err(e) = self.agent.cancel(job) {
            eprintln!("logos-inference: couldn't withdraw request {job}: {e}");
        }
    }

    fn quote(&self, job: &str, until: Instant) -> Quote {
        loop {
            match self.agent.job(job) {
                Ok(Some(JobState::Quoted {
                    price_lepta,
                    escrow: true,
                })) => return Quote::Price(price_lepta),
                Ok(Some(JobState::Quoted { escrow: false, .. })) => return Quote::NoEscrow,
                Ok(Some(JobState::Ended(reason))) => return Quote::Turned(reason),
                Ok(Some(JobState::Expired)) => return Quote::Turned("the quote expired".into()),
                _ => {}
            }
            if Instant::now() >= until {
                return Quote::Silent;
            }
            thread::sleep(self.poll);
        }
    }

    /// Pays the quoted job once, then follows it to its answer. logos-agents finishes a
    /// payment it started, or one the user approves, on its own.
    fn pay(&self, job: &str, deadline: Instant) -> Result<String, ApiError> {
        let waiting = match self.agent.pay(job, &self.pay_from) {
            Ok(Payment::Started) => Waiting::Payment,
            Ok(Payment::AwaitingApproval) => Waiting::Approval,
            Ok(Payment::NotSent(reason)) => {
                return Err(ApiError::new(
                    502,
                    format!("the payment didn't go out: {reason}"),
                ));
            }
            Err(e) => Waiting::Unclear(e),
        };
        loop {
            match self.agent.job(job) {
                Ok(Some(JobState::Delivered(output))) => return answer(output),
                Ok(Some(JobState::Ended(reason))) => {
                    return Err(ApiError::new(
                        502,
                        format!("the paid job failed: {reason}. logos-agents refunds the escrow"),
                    )
                    .not_retryable());
                }
                _ => {}
            }
            if Instant::now() >= deadline {
                let message = match waiting {
                    Waiting::Approval => "the payment is waiting for your approval. If you approve it later, the job still runs and logos-agents keeps its answer".to_string(),
                    Waiting::Unclear(e) => format!("the payment may have gone out ({e}). If it did, logos-agents finishes the job or refunds the escrow"),
                    Waiting::Payment => "paid, but no answer in time. The job goes on: logos-agents keeps its answer, or refunds the escrow if nothing comes by the deadline".to_string(),
                };
                return Err(ApiError::new(504, message).not_retryable());
            }
            thread::sleep(self.poll);
        }
    }
}

fn answer(output: Value) -> Result<String, ApiError> {
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
    )
    .not_retryable())
}

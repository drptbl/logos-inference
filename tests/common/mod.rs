//! A scripted logos-agents for the gateway's tests.
#![allow(dead_code)]

use logos_inference::agent::{Agent, JobView, Payment, Provider};
use logos_inference::gateway::Gateway;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// What a scripted provider does with a request.
#[derive(Clone)]
pub enum Script {
    /// Quotes this price, then answers this output once paid.
    Quote(u128, Value),
    /// Quotes, then fails after payment with this reason.
    FailAfterPay(u128, &'static str),
    /// Turns the request away with this reason.
    Reject(&'static str),
    /// Never answers.
    Silent,
    /// Quotes, but never delivers once paid.
    Hang(u128),
    /// Quotes, and answers this text this long after payment.
    Late(u128, &'static str, Duration),
}

#[derive(Default)]
pub struct Log {
    pub requests: Vec<String>,
    pub payments: Vec<String>,
    pub cancels: Vec<String>,
}

/// An agent whose providers follow their scripts. Payments wait `approvals` extra calls.
pub struct Fake {
    pub providers: Vec<(String, Script)>,
    pub approvals: Mutex<u32>,
    pub jobs: Mutex<HashMap<String, (Script, Option<Instant>)>>,
    pub log: Mutex<Log>,
}

pub fn fake(providers: &[(&str, Script)]) -> Fake {
    Fake {
        providers: providers
            .iter()
            .map(|(p, s)| (p.to_string(), s.clone()))
            .collect(),
        approvals: Mutex::new(0),
        jobs: Mutex::default(),
        log: Mutex::default(),
    }
}

impl Agent for Fake {
    fn find(&self, service: &str) -> Result<Vec<Provider>, String> {
        assert_eq!(service, "inference/tiny/v1");
        Ok(self
            .providers
            .iter()
            .map(|(p, _)| Provider {
                peer_id: p.clone(),
                chat_address: format!("chat-{p}"),
            })
            .collect())
    }

    fn request(&self, provider: &Provider, service: &str, input: &str) -> Result<String, String> {
        assert_eq!(service, "inference/tiny/v1");
        assert_eq!(
            serde_json::from_str::<Value>(input).unwrap()["messages"][0]["content"],
            json!("Hi")
        );
        let script = self
            .providers
            .iter()
            .find(|(p, _)| *p == provider.peer_id)
            .unwrap()
            .1
            .clone();
        let job = format!("job-{}", provider.peer_id);
        self.jobs
            .lock()
            .unwrap()
            .insert(job.clone(), (script, None));
        self.log
            .lock()
            .unwrap()
            .requests
            .push(provider.peer_id.clone());
        Ok(job)
    }

    fn job(&self, job: &str) -> Result<Option<JobView>, String> {
        let jobs = self.jobs.lock().unwrap();
        let Some((script, paid_at)) = jobs.get(job) else {
            return Ok(None);
        };
        let view = |state: &str| JobView {
            state: state.into(),
            ..JobView::default()
        };
        Ok(Some(match (script, paid_at.is_some()) {
            (Script::Silent, _) => view("requested"),
            (Script::Reject(reason), _) => JobView {
                reason: Some(reason.to_string()),
                ..view("rejected")
            },
            (
                Script::Quote(price, _)
                | Script::FailAfterPay(price, _)
                | Script::Hang(price)
                | Script::Late(price, _, _),
                false,
            ) => JobView {
                price_lepta: Some(*price),
                ..view("quoted")
            },
            (Script::Quote(price, output), true) => JobView {
                price_lepta: Some(*price),
                output: Some(output.clone()),
                ..view("delivered")
            },
            (Script::Late(price, text, after), true) => {
                if paid_at.is_some_and(|at| at.elapsed() >= *after) {
                    JobView {
                        price_lepta: Some(*price),
                        output: Some(json!({ "text": text })),
                        ..view("delivered")
                    }
                } else {
                    JobView {
                        price_lepta: Some(*price),
                        ..view("paid")
                    }
                }
            }
            (Script::Hang(price), true) => JobView {
                price_lepta: Some(*price),
                ..view("paid")
            },
            (Script::FailAfterPay(price, reason), true) => JobView {
                price_lepta: Some(*price),
                reason: Some(reason.to_string()),
                ..view("failed")
            },
        }))
    }

    fn pay(&self, job: &str, from: &str) -> Result<Payment, String> {
        assert_eq!(from, "Public/buyer");
        let mut approvals = self.approvals.lock().unwrap();
        if *approvals > 0 {
            *approvals -= 1;
            return Ok(Payment::AwaitingApproval);
        }
        self.jobs.lock().unwrap().get_mut(job).unwrap().1 = Some(Instant::now());
        self.log.lock().unwrap().payments.push(job.to_string());
        Ok(Payment::Sent)
    }

    fn cancel(&self, job: &str) -> Result<(), String> {
        self.log.lock().unwrap().cancels.push(job.to_string());
        Ok(())
    }
}

pub fn gateway(agent: Fake) -> Gateway<Fake> {
    Gateway {
        agent,
        pay_from: "Public/buyer".into(),
        max_price_lepta: 1_000,
        quote_wait: Duration::from_millis(200),
        poll: Duration::from_millis(5),
    }
}

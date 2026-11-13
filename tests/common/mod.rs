//! A scripted logos-agents for the gateway's tests.
#![allow(dead_code)]

use logos_inference::agent::{Agent, JobState, Payment, Provider};
use logos_inference::gateway::Gateway;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// What a scripted provider does with a request.
#[derive(Clone)]
pub enum Script {
    /// Quotes this price in escrow, then answers this output once paid.
    Quote(u128, Value),
    /// Quotes this price without escrow.
    QuoteNoEscrow(u128),
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
    /// Quotes, shows the quote expired after payment, then answers this text this long after.
    ExpiredThenLate(u128, &'static str, Duration),
}

/// What `service_pay` does.
#[derive(Clone)]
pub enum PayMode {
    /// Pays at once.
    Pays,
    /// Waits for an approval that comes after this many job checks, or never.
    Approval(Option<u32>),
    /// Errors, but the money went out: the job shows paid after this many checks.
    ErrsButPays(&'static str, u32),
    /// Errors, and nothing went out.
    ErrsUnpaid(&'static str),
    /// Refuses before sending.
    NotSent(&'static str),
}

#[derive(Default)]
pub struct Log {
    pub requests: Vec<String>,
    pub pay_calls: Vec<String>,
    pub cancels: Vec<String>,
    pub finds: u32,
}

/// Where a job's money stands.
#[derive(Clone)]
enum Money {
    Unpaid,
    /// Paid after this many more checks.
    Coming(u32),
    Paid(Instant),
}

/// An agent whose providers follow their scripts.
pub struct Fake {
    pub providers: Vec<(String, Script)>,
    pub pay_mode: Mutex<PayMode>,
    /// Finds that come back empty before the providers show.
    pub find_misses: Mutex<u32>,
    /// How long each job check takes.
    pub job_delay: Mutex<Duration>,
    jobs: Mutex<HashMap<String, (Script, Money)>>,
    pub log: Mutex<Log>,
}

pub fn fake(providers: &[(&str, Script)]) -> Fake {
    Fake {
        providers: providers
            .iter()
            .map(|(p, s)| (p.to_string(), s.clone()))
            .collect(),
        pay_mode: Mutex::new(PayMode::Pays),
        find_misses: Mutex::new(0),
        job_delay: Mutex::new(Duration::ZERO),
        jobs: Mutex::default(),
        log: Mutex::default(),
    }
}

impl Agent for Fake {
    fn find(&self, service: &str) -> Result<Vec<Provider>, String> {
        assert_eq!(service, "inference/tiny/v1");
        self.log.lock().unwrap().finds += 1;
        let mut misses = self.find_misses.lock().unwrap();
        if *misses > 0 {
            *misses -= 1;
            return Ok(vec![]);
        }
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
            .insert(job.clone(), (script, Money::Unpaid));
        self.log
            .lock()
            .unwrap()
            .requests
            .push(provider.peer_id.clone());
        Ok(job)
    }

    fn job(&self, job: &str) -> Result<Option<JobState>, String> {
        std::thread::sleep(*self.job_delay.lock().unwrap());
        let mut jobs = self.jobs.lock().unwrap();
        let Some((script, money)) = jobs.get_mut(job) else {
            return Ok(None);
        };
        if let Money::Coming(checks) = money {
            *money = match checks {
                0 => Money::Paid(Instant::now()),
                n => Money::Coming(*n - 1),
            };
        }
        let paid_at = match money {
            Money::Paid(at) => Some(*at),
            _ => None,
        };
        let quoted = |price: &u128, escrow| JobState::Quoted {
            price_lepta: *price,
            escrow,
        };
        Ok(Some(match (script, paid_at) {
            (Script::Silent, _) => JobState::Requested,
            (Script::Reject(reason), _) => JobState::Ended(reason.to_string()),
            (Script::QuoteNoEscrow(price), _) => quoted(price, false),
            (
                Script::Quote(price, _)
                | Script::FailAfterPay(price, _)
                | Script::Hang(price)
                | Script::Late(price, _, _)
                | Script::ExpiredThenLate(price, _, _),
                None,
            ) => quoted(price, true),
            (Script::Quote(_, output), Some(_)) => JobState::Delivered(output.clone()),
            (Script::Late(_, text, after), Some(at)) if at.elapsed() >= *after => {
                JobState::Delivered(json!({ "text": text }))
            }
            (Script::ExpiredThenLate(_, text, after), Some(at)) if at.elapsed() >= *after => {
                JobState::Delivered(json!({ "text": text }))
            }
            (Script::ExpiredThenLate(..), Some(_)) => JobState::Expired,
            (Script::Late(..) | Script::Hang(_), Some(_)) => JobState::Paid,
            (Script::FailAfterPay(_, reason), Some(_)) => JobState::Ended(reason.to_string()),
        }))
    }

    fn pay(&self, job: &str, from: &str) -> Result<Payment, String> {
        assert_eq!(from, "Public/buyer");
        self.log.lock().unwrap().pay_calls.push(job.to_string());
        let mut jobs = self.jobs.lock().unwrap();
        let money = &mut jobs.get_mut(job).unwrap().1;
        match self.pay_mode.lock().unwrap().clone() {
            PayMode::Pays => {
                *money = Money::Paid(Instant::now());
                Ok(Payment::Started)
            }
            PayMode::Approval(after) => {
                if let Some(checks) = after {
                    *money = Money::Coming(checks);
                }
                Ok(Payment::AwaitingApproval)
            }
            PayMode::ErrsButPays(error, checks) => {
                *money = Money::Coming(checks);
                Err(error.into())
            }
            PayMode::ErrsUnpaid(error) => Err(error.into()),
            PayMode::NotSent(reason) => Ok(Payment::NotSent(reason.into())),
        }
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

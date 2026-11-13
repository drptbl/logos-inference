//! How the gateway picks a provider, keeps to the price cap and pays, against a scripted agent.

use logos_inference::agent::{Agent, JobView, Payment, Provider};
use logos_inference::gateway::Gateway;
use logos_inference::openai::{ApiError, translate};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// What a scripted provider does with a request.
#[derive(Clone)]
enum Script {
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
}

#[derive(Default)]
struct Log {
    requests: Vec<String>,
    payments: Vec<String>,
    cancels: Vec<String>,
}

/// An agent whose providers follow their scripts. Payments wait `approvals` extra calls.
struct Fake {
    providers: Vec<(String, Script)>,
    approvals: Mutex<u32>,
    jobs: Mutex<HashMap<String, (Script, bool)>>,
    log: Mutex<Log>,
}

fn fake(providers: &[(&str, Script)]) -> Fake {
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
            .insert(job.clone(), (script, false));
        self.log
            .lock()
            .unwrap()
            .requests
            .push(provider.peer_id.clone());
        Ok(job)
    }

    fn job(&self, job: &str) -> Result<Option<JobView>, String> {
        let jobs = self.jobs.lock().unwrap();
        let Some((script, paid)) = jobs.get(job) else {
            return Ok(None);
        };
        let view = |state: &str| JobView {
            state: state.into(),
            ..JobView::default()
        };
        Ok(Some(match (script, paid) {
            (Script::Silent, _) => view("requested"),
            (Script::Reject(reason), _) => JobView {
                reason: Some(reason.to_string()),
                ..view("rejected")
            },
            (
                Script::Quote(price, _) | Script::FailAfterPay(price, _) | Script::Hang(price),
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
        self.jobs.lock().unwrap().get_mut(job).unwrap().1 = true;
        self.log.lock().unwrap().payments.push(job.to_string());
        Ok(Payment::Sent)
    }

    fn cancel(&self, job: &str) -> Result<(), String> {
        self.log.lock().unwrap().cancels.push(job.to_string());
        Ok(())
    }
}

fn gateway(agent: Fake) -> Gateway<Fake> {
    Gateway {
        agent,
        pay_from: "Public/buyer".into(),
        max_price_lepta: 1_000,
        quote_wait: Duration::from_millis(200),
        poll: Duration::from_millis(5),
    }
}

fn ask(gateway: &Gateway<Fake>, within: Duration) -> Result<String, ApiError> {
    let call = translate(
        json!({"model": "tiny", "messages": [{"role": "user", "content": "Hi"}]})
            .to_string()
            .as_bytes(),
        &["tiny".to_string()],
    )
    .unwrap();
    gateway.complete(&call, Instant::now() + within)
}

fn answer(text: &str) -> Value {
    json!({ "text": text })
}

#[test]
fn pays_the_first_provider_within_the_cap_and_answers_its_output() {
    let g = gateway(fake(&[
        ("a", Script::Quote(900, answer("hello"))),
        ("b", Script::Quote(1, answer("cheaper"))),
    ]));
    assert_eq!(ask(&g, Duration::from_secs(5)), Ok("hello".to_string()));
    let log = g.agent.log.lock().unwrap();
    assert_eq!(log.requests, ["a"]);
    assert_eq!(log.payments, ["job-a"]);
    assert!(log.cancels.is_empty());
}

#[test]
fn moves_past_a_quote_over_the_cap() {
    let g = gateway(fake(&[
        ("a", Script::Quote(1_001, answer("dear"))),
        ("b", Script::Quote(1_000, answer("fair"))),
    ]));
    assert_eq!(ask(&g, Duration::from_secs(5)), Ok("fair".to_string()));
    let log = g.agent.log.lock().unwrap();
    assert_eq!(log.cancels, ["job-a"]);
    assert_eq!(log.payments, ["job-b"]);
}

#[test]
fn pays_nobody_when_every_quote_is_over_the_cap() {
    let g = gateway(fake(&[
        ("a", Script::Quote(5_000, answer("x"))),
        ("b", Script::Quote(2_000, answer("y"))),
    ]));
    let e = ask(&g, Duration::from_secs(5)).unwrap_err();
    assert_eq!(e.status, 402);
    assert_eq!(
        e.message,
        "every quote was over the cap of 1000 lepta, the lowest 2000"
    );
    let log = g.agent.log.lock().unwrap();
    assert!(log.payments.is_empty());
    assert_eq!(log.cancels, ["job-a", "job-b"]);
}

#[test]
fn moves_past_a_provider_that_turns_the_request_away_or_stays_silent() {
    let g = gateway(fake(&[
        ("a", Script::Reject("busy, try again later")),
        ("b", Script::Silent),
        ("c", Script::Quote(10, answer("third time"))),
    ]));
    assert_eq!(
        ask(&g, Duration::from_secs(5)),
        Ok("third time".to_string())
    );
    let log = g.agent.log.lock().unwrap();
    assert_eq!(log.requests, ["a", "b", "c"]);
    assert_eq!(log.cancels, ["job-b"]);
}

#[test]
fn says_why_when_no_provider_takes_the_job() {
    let g = gateway(fake(&[("a", Script::Reject("busy, try again later"))]));
    let e = ask(&g, Duration::from_secs(5)).unwrap_err();
    assert_eq!(
        (e.status, e.message.as_str()),
        (503, "no provider took the job: busy, try again later")
    );
    let g = gateway(fake(&[]));
    let e = ask(&g, Duration::from_secs(5)).unwrap_err();
    assert_eq!(
        (e.status, e.message.as_str()),
        (503, "no provider offers tiny right now")
    );
}

#[test]
fn never_pays_a_second_provider_after_a_paid_job_fails() {
    let g = gateway(fake(&[
        (
            "a",
            Script::FailAfterPay(100, "the model refused the prompt (400)"),
        ),
        ("b", Script::Quote(1, answer("unused"))),
    ]));
    let e = ask(&g, Duration::from_secs(5)).unwrap_err();
    assert_eq!(e.status, 502);
    assert_eq!(
        e.message,
        "the paid job failed: the model refused the prompt (400). logos-agents refunds the payment"
    );
    let log = g.agent.log.lock().unwrap();
    assert_eq!(log.payments, ["job-a"]);
    assert_eq!(log.requests, ["a"]);
}

#[test]
fn waits_for_an_approval_before_the_answer() {
    let g = gateway(fake(&[("a", Script::Quote(100, answer("approved")))]));
    *g.agent.approvals.lock().unwrap() = 3;
    assert_eq!(ask(&g, Duration::from_secs(5)), Ok("approved".to_string()));
    assert_eq!(g.agent.log.lock().unwrap().payments, ["job-a"]);
}

#[test]
fn gives_up_at_the_deadline_without_paying() {
    let g = gateway(fake(&[("a", Script::Silent)]));
    let started = Instant::now();
    let e = ask(&g, Duration::from_millis(100)).unwrap_err();
    assert_eq!(
        (e.status, e.message.as_str()),
        (504, "no provider quoted in time")
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(g.agent.log.lock().unwrap().cancels, ["job-a"]);
}

#[test]
fn turns_away_an_answer_delivered_through_storage() {
    let g = gateway(fake(&[(
        "a",
        Script::Quote(100, json!({"cid": "zDvZRwzm"})),
    )]));
    let e = ask(&g, Duration::from_secs(5)).unwrap_err();
    assert_eq!(e.status, 502);
    assert_eq!(
        e.message,
        "the answer came as Logos Storage CID zDvZRwzm, which the gateway can't fetch yet"
    );
}

#[test]
fn says_the_job_still_runs_when_the_answer_misses_the_deadline() {
    let g = gateway(fake(&[("a", Script::Hang(100))]));
    let e = ask(&g, Duration::from_millis(300)).unwrap_err();
    assert_eq!(e.status, 504);
    assert_eq!(
        e.message,
        "paid, but no answer in time. The job goes on: logos-agents keeps its answer, or refunds the payment if nothing comes by the deadline"
    );
    assert!(g.agent.log.lock().unwrap().cancels.is_empty());
}

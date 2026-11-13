//! How the gateway picks a provider, keeps to the price cap and pays, against a scripted agent.

mod common;

use common::{Fake, Script, fake, gateway};
use logos_inference::gateway::Gateway;
use logos_inference::openai::{ApiError, translate};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

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

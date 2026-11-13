//! How the gateway picks a provider, keeps to the price cap and pays, against a scripted agent.

mod common;

use common::{Fake, PayMode, Script, fake, gateway};
use logos_inference::gateway::Gateway;
use logos_inference::openai::{ApiError, translate};
use serde_json::{Value, json};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

fn ask_unless(
    gateway: &Gateway<Fake>,
    within: Duration,
    gone: &AtomicBool,
) -> Result<String, ApiError> {
    let call = translate(
        json!({"model": "tiny", "messages": [{"role": "user", "content": "Hi"}]})
            .to_string()
            .as_bytes(),
        &["tiny".to_string()],
    )
    .unwrap();
    gateway.complete(&call, Instant::now() + within, gone)
}

fn ask(gateway: &Gateway<Fake>, within: Duration) -> Result<String, ApiError> {
    ask_unless(gateway, within, &AtomicBool::new(false))
}

fn answer(text: &str) -> Value {
    json!({ "text": text })
}

fn with_pay(agent: Fake, mode: PayMode) -> Fake {
    *agent.pay_mode.lock().unwrap() = mode;
    agent
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
    assert_eq!(log.pay_calls, ["job-a"]);
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
    assert_eq!(log.pay_calls, ["job-b"]);
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
    assert!(log.pay_calls.is_empty());
    assert_eq!(log.cancels, ["job-a", "job-b"]);
}

#[test]
fn never_pays_a_quote_without_escrow() {
    let g = gateway(fake(&[
        ("a", Script::QuoteNoEscrow(10)),
        ("b", Script::Quote(20, answer("escrowed"))),
    ]));
    assert_eq!(ask(&g, Duration::from_secs(5)), Ok("escrowed".to_string()));
    let log = g.agent.log.lock().unwrap();
    assert_eq!(log.cancels, ["job-a"]);
    assert_eq!(log.pay_calls, ["job-b"]);
    drop(log);
    let g = gateway(fake(&[("a", Script::QuoteNoEscrow(10))]));
    let e = ask(&g, Duration::from_secs(5)).unwrap_err();
    assert_eq!(
        (e.status, e.message.as_str()),
        (
            503,
            "no provider took the job: quoted without escrow, so nothing could be refunded"
        )
    );
    assert!(g.agent.log.lock().unwrap().pay_calls.is_empty());
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
fn looks_again_while_discovery_comes_back_empty() {
    let g = gateway(fake(&[("a", Script::Quote(10, answer("found")))]));
    *g.agent.find_misses.lock().unwrap() = 3;
    assert_eq!(ask(&g, Duration::from_secs(5)), Ok("found".to_string()));
    assert_eq!(g.agent.log.lock().unwrap().finds, 4);
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
        "the paid job failed: the model refused the prompt (400). logos-agents refunds the escrow"
    );
    assert!(!e.retryable);
    let log = g.agent.log.lock().unwrap();
    assert_eq!(log.pay_calls, ["job-a"]);
    assert_eq!(log.requests, ["a"]);
}

#[test]
fn pays_once_and_waits_for_the_approval() {
    let g = gateway(with_pay(
        fake(&[("a", Script::Quote(100, answer("approved")))]),
        PayMode::Approval(Some(5)),
    ));
    assert_eq!(ask(&g, Duration::from_secs(5)), Ok("approved".to_string()));
    assert_eq!(g.agent.log.lock().unwrap().pay_calls, ["job-a"]);
}

#[test]
fn says_the_payment_still_waits_when_no_approval_comes() {
    let g = gateway(with_pay(
        fake(&[("a", Script::Quote(100, answer("x")))]),
        PayMode::Approval(None),
    ));
    let e = ask(&g, Duration::from_millis(300)).unwrap_err();
    assert_eq!(e.status, 504);
    assert_eq!(
        e.message,
        "the payment is waiting for your approval. If you approve it later, the job still runs and logos-agents keeps its answer"
    );
    assert!(!e.retryable);
    assert_eq!(g.agent.log.lock().unwrap().pay_calls, ["job-a"]);
}

#[test]
fn watches_the_job_when_the_payment_answer_is_unclear() {
    let g = gateway(with_pay(
        fake(&[("a", Script::Quote(100, answer("it went out")))]),
        PayMode::ErrsButPays("a payment for this job may already have gone out", 5),
    ));
    assert_eq!(
        ask(&g, Duration::from_secs(5)),
        Ok("it went out".to_string())
    );
    assert_eq!(g.agent.log.lock().unwrap().pay_calls, ["job-a"]);
}

#[test]
fn says_the_money_may_have_moved_when_an_unclear_payment_never_shows() {
    let g = gateway(with_pay(
        fake(&[
            ("a", Script::Quote(100, answer("x"))),
            ("b", Script::Quote(1, answer("y"))),
        ]),
        PayMode::ErrsUnpaid("transfer timed out"),
    ));
    let e = ask(&g, Duration::from_millis(300)).unwrap_err();
    assert_eq!(e.status, 504);
    assert_eq!(
        e.message,
        "the payment may have gone out (transfer timed out). If it did, logos-agents finishes the job or refunds the escrow"
    );
    assert!(!e.retryable);
    let log = g.agent.log.lock().unwrap();
    assert_eq!(log.pay_calls, ["job-a"]);
    assert_eq!(log.requests, ["a"]);
}

#[test]
fn says_so_when_the_payment_was_refused_before_anything_went_out() {
    let g = gateway(with_pay(
        fake(&[("a", Script::Quote(100, answer("x")))]),
        PayMode::NotSent("the quote has expired: ask for a new one with service_request"),
    ));
    let e = ask(&g, Duration::from_secs(5)).unwrap_err();
    assert_eq!(e.status, 502);
    assert_eq!(
        e.message,
        "the payment didn't go out: the quote has expired: ask for a new one with service_request"
    );
    assert!(e.retryable);
}

#[test]
fn keeps_waiting_when_the_quote_expires_after_payment() {
    let g = gateway(fake(&[(
        "a",
        Script::ExpiredThenLate(100, "honoured", Duration::from_millis(100)),
    )]));
    assert_eq!(ask(&g, Duration::from_secs(5)), Ok("honoured".to_string()));
}

#[test]
fn stops_before_paying_once_the_client_has_gone() {
    let g = gateway(fake(&[("a", Script::Quote(100, answer("nobody listens")))]));
    let e = ask_unless(&g, Duration::from_secs(5), &AtomicBool::new(true)).unwrap_err();
    assert_eq!(e.status, 499);
    let log = g.agent.log.lock().unwrap();
    assert!(log.pay_calls.is_empty());
    assert!(log.requests.is_empty());
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
    assert!(e.retryable);
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
    assert!(!e.retryable);
}

#[test]
fn says_the_job_still_runs_when_the_answer_misses_the_deadline() {
    let g = gateway(fake(&[("a", Script::Hang(100))]));
    let e = ask(&g, Duration::from_millis(300)).unwrap_err();
    assert_eq!(e.status, 504);
    assert_eq!(
        e.message,
        "paid, but no answer in time. The job goes on: logos-agents keeps its answer, or refunds the escrow if nothing comes by the deadline"
    );
    assert!(!e.retryable);
    assert!(g.agent.log.lock().unwrap().cancels.is_empty());
}

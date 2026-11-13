//! How logos-agents v0.4.0's tool answers are read, from answers shaped as its source writes them.

use logos_inference::agent::{JobState, Payment, Provider};
use logos_inference::mcp_agent::{ToolError, job_state, payer, payment, providers};
use serde_json::json;

#[test]
fn lists_providers_other_than_this_node() {
    let found = json!({"service_id": "inference/tiny/v1", "note": "ask one for a job with service_request", "providers": [
        {"peer_id": "12D3KooWA", "chat_address": "aa".repeat(32), "tag": "inference", "this_node": false},
        {"peer_id": "12D3KooWSelf", "chat_address": "bb".repeat(32), "tag": "inference", "this_node": true},
        {"peer_id": "12D3KooWBroken", "tag": "inference", "this_node": false},
    ]});
    assert_eq!(
        providers(&found),
        [Provider {
            peer_id: "12D3KooWA".into(),
            chat_address: "aa".repeat(32)
        }]
    );
    assert!(providers(&json!({"providers": [], "note": "nobody found yet"})).is_empty());
}

fn jobs(job: serde_json::Value) -> serde_json::Value {
    json!({"jobs": [{"job": "other", "state": "delivered", "output": {"text": "not this one"}}, job], "offers": []})
}

#[test]
fn reads_a_quote_and_whether_it_offers_escrow() {
    let quote = |escrow| json!({"price_lepta": "6480", "pay_to": "Public/x", "network": "logos.dev", "expires_at": 1, "escrow": escrow});
    let escrowed = jobs(
        json!({"job": "j1", "state": "quoted", "quote": quote(json!({"program": "p", "provider": "Public/y", "deadline": 2}))}),
    );
    assert_eq!(
        job_state(&escrowed, "j1"),
        Some(JobState::Quoted {
            price_lepta: 6480,
            escrow: true
        })
    );
    let plain = jobs(json!({"job": "j1", "state": "quoted", "quote": quote(json!(null))}));
    assert_eq!(
        job_state(&plain, "j1"),
        Some(JobState::Quoted {
            price_lepta: 6480,
            escrow: false
        })
    );
}

#[test]
fn reads_each_state_a_buyer_sees() {
    let state = |job: serde_json::Value| job_state(&jobs(job), "j1");
    assert_eq!(
        state(json!({"job": "j1", "state": "requested"})),
        Some(JobState::Requested)
    );
    assert_eq!(
        state(json!({"job": "j1", "state": "paid"})),
        Some(JobState::Paid)
    );
    assert_eq!(
        state(json!({"job": "j1", "state": "delivered", "output": {"text": "hi"}})),
        Some(JobState::Delivered(json!({"text": "hi"})))
    );
    assert_eq!(
        state(json!({"job": "j1", "state": "expired"})),
        Some(JobState::Expired)
    );
    assert_eq!(
        state(json!({"job": "j1", "state": "rejected", "reason": "busy, try again later"})),
        Some(JobState::Ended("busy, try again later".into()))
    );
    assert_eq!(
        state(json!({"job": "j1", "state": "failed", "reason": null})),
        Some(JobState::Ended("failed".into()))
    );
    assert_eq!(
        state(json!({"job": "j1", "state": "canceled"})),
        Some(JobState::Ended("canceled".into()))
    );
    // A state this doesn't know is waited on, never taken as an ending.
    assert_eq!(
        state(json!({"job": "j1", "state": "something_new"})),
        Some(JobState::Requested)
    );
    assert_eq!(
        job_state(&jobs(json!({"job": "j2", "state": "paid"})), "j1"),
        None
    );
}

#[test]
fn reads_what_service_pay_did() {
    assert_eq!(
        payment(Ok(
            json!({"job": "j1", "escrow_open_tx": "ab12", "note": "sent: once it lands, this server tells the provider on its own"})
        )),
        Ok(Payment::Started)
    );
    assert_eq!(
        payment(Ok(json!({"job": "j1", "paid": true, "tx_hash": "ab12"}))),
        Ok(Payment::Started)
    );
    assert_eq!(
        payment(Ok(
            json!({"status": "awaiting_approval", "request_id": "r1", "note": "the user approves each payment"})
        )),
        Ok(Payment::AwaitingApproval)
    );
    assert_eq!(
        payment(Ok(
            json!({"status": "approved_sending", "request_id": "r1"})
        )),
        Ok(Payment::Started)
    );
}

#[test]
fn counts_a_refusal_as_nothing_sent_only_when_logos_agents_says_so() {
    for before_sending in [
        "the quote has expired: ask for a new one with service_request",
        "the quote is for logos.dev, but this node is on other",
        "can't save the payment before sending it, so a restart could pay twice or lose the refund: not paying (disk full)",
        "insufficient balance",
    ] {
        assert_eq!(
            payment(Err(ToolError::Refused(before_sending.into()))),
            Ok(Payment::NotSent(before_sending.into())),
            "{before_sending}"
        );
    }
    for maybe_sent in [
        "the transfer may have been sent: check before paying again",
        "a payment for this job may already have gone out, with no clear answer. Not paying again",
        "the escrow open ab12 hasn't landed yet: call service_pay again in a few minutes",
        "paid, but the provider wasn't told: chat down. Call service_pay again to tell it; nothing more will be paid",
        "another call is working on job j1: try again shortly",
        "can't read the job's escrow: rpc down",
        "job j1 is paid, not waiting for payment",
    ] {
        assert_eq!(
            payment(Err(ToolError::Refused(maybe_sent.into()))),
            Err(maybe_sent.into()),
            "{maybe_sent}"
        );
    }
    // A call that never got an answer, or an answer this can't read, may have paid too.
    assert_eq!(
        payment(Err(ToolError::Failed(
            "service_pay didn't answer in 90s".into()
        ))),
        Err("service_pay didn't answer in 90s".into())
    );
    assert!(payment(Ok(json!({"something": "else"}))).is_err());
}

#[test]
fn pays_only_from_a_public_agent_account() {
    let accounts = json!({"accounts": [
        {"agent": true, "address": "Public/Mine", "public": true, "balance_lepta": "5"},
        {"agent": false, "address": "Public/Theirs", "public": true},
        {"agent": true, "address": "Private/Hidden", "public": false},
    ]});
    assert_eq!(payer(&accounts, "Public/Mine"), Ok(()));
    assert_eq!(
        payer(&accounts, "Public/Theirs"),
        Err("Public/Theirs isn't one of the agent's own accounts".into())
    );
    assert_eq!(
        payer(&accounts, "Private/Hidden"),
        Err("Private/Hidden isn't a public account: payments come from Public/<base58>".into())
    );
    assert_eq!(
        payer(&accounts, "Public/Nobody"),
        Err("the wallet holds no account Public/Nobody".into())
    );
}

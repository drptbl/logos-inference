use logos_inference::job::{Pricing, Refusal, Role, parse};
use serde_json::json;

const PRICING: Pricing = Pricing {
    input_lepta_per_token: 10,
    output_lepta_per_token: 100,
    max_tokens: 256,
};

#[test]
fn a_plain_prompt_costs_its_estimated_tokens_and_the_output_cap() {
    // 19 bytes => 5 tokens, plus 4 for the message: 9 x 10, and 256 x 100 for the cap.
    let prompt = parse(&json!({"text": "What is Logos? Tell"}), &PRICING, None).unwrap();
    assert_eq!(prompt.max_tokens, 256);
    assert_eq!(prompt.price(&PRICING), 90 + 25_600);
}

fn text(s: &str) -> serde_json::Value {
    json!({ "text": s })
}

#[test]
fn a_chat_request_is_read_as_messages_with_its_own_cap() {
    let request = r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}],"max_tokens":16}"#;
    let prompt = parse(&text(request), &PRICING, None).unwrap();
    assert_eq!(prompt.messages.len(), 2);
    assert_eq!(prompt.messages[1].content, "Hi");
    // (3 + 4) + (1 + 4) input tokens x 10, and 16 x 100.
    assert_eq!(prompt.price(&PRICING), 120 + 1_600);
}

#[test]
fn the_offers_system_prompt_goes_first_and_is_paid_for() {
    let prompt = parse(&text("Hi"), &PRICING, Some("You are terse.")).unwrap();
    assert_eq!(prompt.messages[0].role, Role::System);
    assert_eq!(prompt.messages[0].content, "You are terse.");
    assert_eq!(prompt.messages[1].content, "Hi");
    // (4 + 4) + (1 + 4) input tokens.
    assert_eq!(prompt.input_tokens(), 13);
}

#[test]
fn other_json_is_a_plain_prompt() {
    let prompt = parse(&text(r#"{"question": "is this JSON?"}"#), &PRICING, None).unwrap();
    assert_eq!(
        prompt.messages[0].content,
        r#"{"question": "is this JSON?"}"#
    );
}

fn refused(input: serde_json::Value) -> String {
    match parse(&input, &PRICING, None) {
        Err(Refusal(reason)) => reason,
        Ok(p) => panic!("accepted {p:?}"),
    }
}

#[test]
fn a_cid_input_is_turned_away() {
    assert_eq!(
        refused(json!({"cid": "zDvZRwzm"})),
        "send the prompt as text, not a CID"
    );
}

#[test]
fn a_cap_over_the_offers_is_turned_away() {
    assert_eq!(
        refused(text(
            r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":257}"#
        )),
        "max_tokens is at most 256"
    );
    assert_eq!(
        refused(text(
            r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":0}"#
        )),
        "max_tokens is at least 1"
    );
}

#[test]
fn a_malformed_chat_request_is_turned_away_not_read_as_text() {
    for bad in [
        r#"{"messages":[]}"#,
        r#"{"messages":"Hi"}"#,
        r#"{"messages":[{"role":"tool","content":"Hi"}]}"#,
        r#"{"messages":[{"role":"user","content":["Hi"]}]}"#,
        r#"{"messages":[{"role":"user","content":"Hi"}],"temperature":0}"#,
    ] {
        let reason = refused(text(bad));
        assert!(
            reason.starts_with("not a chat request: "),
            "{bad}: {reason}"
        );
    }
}

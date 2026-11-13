use base64::Engine;
use logos_inference::job::{Pricing, Refusal, Role, parse, quote};
use serde_json::json;

const PRICING: Pricing = Pricing {
    input_lepta_per_token: 10,
    output_lepta_per_token: 100,
    max_tokens: 256,
    max_input_tokens: None,
};

/// An offer that takes prompts of up to 1000 tokens, as text or as a CID.
const CEILING: Pricing = Pricing {
    input_lepta_per_token: 10,
    output_lepta_per_token: 100,
    max_tokens: 256,
    max_input_tokens: Some(1000),
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

fn b64(text: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(text)
}

#[test]
fn a_text_prompt_is_quoted_by_its_tokens() {
    // "Hi": (1 + 4) x 10, and 256 x 100.
    assert_eq!(quote(&text("Hi"), &PRICING, None), Ok(50 + 25_600));
}

#[test]
fn a_cid_prompt_is_quoted_at_the_offers_ceiling() {
    assert_eq!(
        quote(&json!({"cid": "zDvZRwzm"}), &CEILING, None),
        Ok(1000 * 10 + 256 * 100)
    );
    assert_eq!(
        quote(&json!({"cid": "zDvZRwzm"}), &PRICING, None),
        Err(Refusal("send the prompt as text, not a CID".into()))
    );
}

#[test]
fn a_cid_prompt_is_read_from_its_content() {
    let plain = parse(
        &json!({"cid": "zDvZRwzm", "content_b64": b64(b"Hi")}),
        &CEILING,
        None,
    )
    .unwrap();
    assert_eq!(plain.messages[0].content, "Hi");
    let chat = br#"{"messages":[{"role":"user","content":"Hello"}],"max_tokens":8}"#;
    let chat = parse(
        &json!({"cid": "zDvZRwzm", "content_b64": b64(chat)}),
        &CEILING,
        None,
    )
    .unwrap();
    assert_eq!(
        (chat.messages[0].content.as_str(), chat.max_tokens),
        ("Hello", 8)
    );
}

#[test]
fn a_prompt_over_the_ceiling_is_refused() {
    let long = "a".repeat(5_000);
    let over = Err(Refusal(
        "the prompt is about 1254 tokens, over this offer's 1000".into(),
    ));
    assert_eq!(quote(&text(&long), &CEILING, None), over);
    assert_eq!(
        parse(&text(&long), &CEILING, None).map(|p| p.max_tokens),
        over.clone().map(|_: u128| 0)
    );
    assert_eq!(
        parse(
            &json!({"cid": "zDvZRwzm", "content_b64": b64(long.as_bytes())}),
            &CEILING,
            None
        )
        .map(|p| p.max_tokens),
        over.map(|_: u128| 0)
    );
}

#[test]
fn a_cid_prompt_runs_only_with_readable_content() {
    assert_eq!(
        parse(&json!({"cid": "zDvZRwzm"}), &CEILING, None).map(|p| p.max_tokens),
        Err(Refusal(
            "the prompt's content didn't come with its CID".into()
        ))
    );
    assert_eq!(
        parse(
            &json!({"cid": "zDvZRwzm", "content_b64": b64(&[0xff, 0xfe])}),
            &CEILING,
            None
        )
        .map(|p| p.max_tokens),
        Err(Refusal("the prompt isn't UTF-8 text".into()))
    );
    assert_eq!(
        parse(
            &json!({"cid": "zDvZRwzm", "content_b64": b64(b"Hi")}),
            &PRICING,
            None
        )
        .map(|p| p.max_tokens),
        Err(Refusal("send the prompt as text, not a CID".into()))
    );
}

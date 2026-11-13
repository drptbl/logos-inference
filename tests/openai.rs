//! What an OpenAI client sends, as the provider wire format, and what it gets back.

use logos_inference::openai::{ApiError, translate};
use serde_json::{Value, json};

fn models() -> Vec<String> {
    vec!["qwen2.5:0.5b".into()]
}

fn ok(body: Value) -> logos_inference::openai::Call {
    match translate(body.to_string().as_bytes(), &models()) {
        Ok(call) => call,
        Err(e) => panic!("refused: {e:?}"),
    }
}

fn refused(body: Value) -> ApiError {
    match translate(body.to_string().as_bytes(), &models()) {
        Err(e) => e,
        Ok(call) => panic!("accepted as {}", call.input),
    }
}

fn wire(call: &logos_inference::openai::Call) -> Value {
    serde_json::from_str(&call.input).unwrap()
}

#[test]
fn a_chat_request_becomes_the_wire_format() {
    let call = ok(json!({"model": "qwen2.5:0.5b", "max_tokens": 16,
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]}));
    assert_eq!(call.model, "qwen2.5:0.5b");
    assert!(!call.stream);
    assert_eq!(
        wire(&call),
        json!({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], "max_tokens": 16})
    );
}

#[test]
fn the_cap_is_left_to_the_offer_when_not_asked() {
    let call = ok(
        json!({"model": "qwen2.5:0.5b", "messages": [{"role": "user", "content": "Hi"}], "stream": true}),
    );
    assert!(call.stream);
    assert_eq!(
        wire(&call),
        json!({"messages": [{"role": "user", "content": "Hi"}]})
    );
}

#[test]
fn max_completion_tokens_is_the_same_cap() {
    let call = ok(
        json!({"model": "qwen2.5:0.5b", "max_completion_tokens": 8, "messages": [{"role": "user", "content": "Hi"}]}),
    );
    assert_eq!(wire(&call)["max_tokens"], json!(8));
    let e = refused(
        json!({"model": "qwen2.5:0.5b", "max_tokens": 8, "max_completion_tokens": 9,
        "messages": [{"role": "user", "content": "Hi"}]}),
    );
    assert_eq!(
        (e.status, e.message.as_str()),
        (400, "max_tokens and max_completion_tokens disagree")
    );
}

#[test]
fn sampling_hints_are_dropped() {
    let call = ok(
        json!({"model": "qwen2.5:0.5b", "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2, "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0, "seed": 7,
        "user": "me", "stop": ["\n"], "stream_options": {"include_usage": true}, "logit_bias": {},
        "n": 1, "logprobs": false, "response_format": {"type": "text"}, "tool_choice": "none"}),
    );
    assert_eq!(
        wire(&call),
        json!({"messages": [{"role": "user", "content": "Hi"}]})
    );
}

#[test]
fn text_parts_are_joined_and_developer_is_a_system_message() {
    let call = ok(json!({"model": "qwen2.5:0.5b", "messages": [
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "name": "jo", "content": [{"type": "text", "text": "Hello, "}, {"type": "text", "text": "there"}]}]}));
    assert_eq!(
        wire(&call)["messages"],
        json!([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello, there"}])
    );
}

#[test]
fn features_that_change_the_answer_are_refused() {
    let user = json!([{"role": "user", "content": "Hi"}]);
    for (extra, field) in [
        (json!({"tools": [{"type": "function"}]}), "tools"),
        (json!({"tool_choice": "auto"}), "tool_choice"),
        (json!({"functions": []}), "functions"),
        (json!({"function_call": "auto"}), "function_call"),
        (json!({"n": 2}), "n"),
        (
            json!({"response_format": {"type": "json_object"}}),
            "response_format",
        ),
        (json!({"logprobs": true}), "logprobs"),
        (json!({"top_logprobs": 2}), "top_logprobs"),
        (json!({"audio": {}}), "audio"),
        (json!({"modalities": ["text", "audio"]}), "modalities"),
        (json!({"something_new": 1}), "something_new"),
    ] {
        let mut body = json!({"model": "qwen2.5:0.5b", "messages": user});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let e = refused(body);
        assert_eq!(e.status, 400, "{field}");
        assert_eq!(e.param.as_deref(), Some(field));
        assert_eq!(e.message, format!("{field} isn't supported over Logos yet"));
    }
}

#[test]
fn messages_it_cannot_carry_are_refused() {
    for bad in [
        json!([]),
        json!("Hi"),
        json!([{"role": "tool", "content": "42", "tool_call_id": "a"}]),
        json!([{"role": "assistant", "content": null, "tool_calls": []}]),
        json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]),
        json!([{"role": "user"}]),
    ] {
        let e = refused(json!({"model": "qwen2.5:0.5b", "messages": bad}));
        assert_eq!(
            (e.status, e.param.as_deref()),
            (400, Some("messages")),
            "{bad}"
        );
    }
}

#[test]
fn an_unknown_model_is_not_found() {
    let e = refused(json!({"model": "gpt-9", "messages": [{"role": "user", "content": "Hi"}]}));
    assert_eq!(
        (e.status, e.code.as_deref()),
        (404, Some("model_not_found"))
    );
    assert_eq!(e.message, "the model gpt-9 isn't served here");
}

#[test]
fn a_body_that_is_not_a_request_is_refused() {
    for bad in [
        "",
        "[]",
        "{\"messages\": []}",
        "{\"model\": 1, \"messages\": []}",
    ] {
        let e = translate(bad.as_bytes(), &models()).unwrap_err();
        assert_eq!(e.status, 400, "{bad}");
    }
}

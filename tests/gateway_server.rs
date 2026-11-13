//! The gateway as OpenAI clients reach it, in front of a scripted logos-agents.

mod common;

use common::{Fake, Script, fake, gateway};
use logos_inference::gateway_server::{GatewayConfig, GatewayServer};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::Duration;

fn start_with(agent: Fake, key: Option<&str>, keep_alive: Duration) -> String {
    start_full(agent, key, keep_alive, Duration::from_secs(5))
}

fn start_full(agent: Fake, key: Option<&str>, keep_alive: Duration, timeout: Duration) -> String {
    let config = GatewayConfig {
        models: vec!["tiny".into()],
        key: key.map(Into::into),
        timeout,
        keep_alive,
    };
    let server = GatewayServer::bind("127.0.0.1:0", gateway(agent), config).unwrap();
    let url = format!("http://{}", server.addr());
    thread::spawn(move || server.run());
    url
}

fn start(agent: Fake) -> String {
    start_with(agent, None, Duration::from_secs(10))
}

fn call(method: &str, url: &str, body: Option<&Value>, key: Option<&str>) -> (u16, String) {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();
    let mut response = match (method, body) {
        ("GET", _) => {
            let mut request = agent.get(url);
            if let Some(key) = key {
                request = request.header("Authorization", format!("Bearer {key}"));
            }
            request.call().unwrap()
        }
        (_, body) => {
            let mut request = agent.post(url);
            if let Some(key) = key {
                request = request.header("Authorization", format!("Bearer {key}"));
            }
            request.send_json(body.unwrap()).unwrap()
        }
    };
    (
        response.status().as_u16(),
        response.body_mut().read_to_string().unwrap(),
    )
}

fn chat(stream: bool) -> Value {
    json!({"model": "tiny", "stream": stream, "messages": [{"role": "user", "content": "Hi"}]})
}

fn json_of(body: &str) -> Value {
    serde_json::from_str(body).unwrap()
}

#[test]
fn lists_the_models_it_serves() {
    let url = start(fake(&[]));
    let (status, body) = call("GET", &format!("{url}/v1/models"), None, None);
    assert_eq!(status, 200);
    assert_eq!(
        json_of(&body),
        json!({"object": "list", "data": [{"id": "tiny", "object": "model", "created": 0, "owned_by": "logos"}]})
    );
}

#[test]
fn answers_a_completion_as_openai_does() {
    let url = start(fake(&[(
        "a",
        Script::Quote(10, json!({"text": "Hello there"})),
    )]));
    let (status, body) = call(
        "POST",
        &format!("{url}/v1/chat/completions"),
        Some(&chat(false)),
        None,
    );
    assert_eq!(status, 200, "{body}");
    let body = json_of(&body);
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(body["object"], json!("chat.completion"));
    assert_eq!(body["model"], json!("tiny"));
    assert_eq!(
        body["choices"],
        json!([{"index": 0, "message": {"role": "assistant", "content": "Hello there"}, "finish_reason": "stop"}])
    );
    // Estimated: "Hi" is 1 token plus 4; "Hello there" is 11 bytes, 3 tokens.
    assert_eq!(
        body["usage"],
        json!({"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8})
    );
}

#[test]
fn answers_errors_as_openai_does() {
    let url = start(fake(&[(
        "a",
        Script::Quote(5_000, json!({"text": "dear"})),
    )]));
    let (status, body) = call(
        "POST",
        &format!("{url}/v1/chat/completions"),
        Some(&chat(false)),
        None,
    );
    assert_eq!(status, 402);
    assert_eq!(
        json_of(&body),
        json!({"error": {"message": "every quote was over the cap of 1000 lepta, the lowest 5000",
                         "type": "invalid_request_error", "param": null, "code": "price_over_cap"}})
    );
    let mut unknown = chat(false);
    unknown["model"] = json!("other");
    let (status, body) = call(
        "POST",
        &format!("{url}/v1/chat/completions"),
        Some(&unknown),
        None,
    );
    assert_eq!(status, 404);
    assert_eq!(json_of(&body)["error"]["code"], json!("model_not_found"));
    assert_eq!(call("GET", &format!("{url}/v1/other"), None, None).0, 404);
}

/// A raw POST to the completions route, read to the end: the head and the body.
fn raw(url: &str, version: &str, headers: &[(&str, &str)], body: &Value) -> (String, String) {
    let addr = url.trim_start_matches("http://");
    let mut socket = TcpStream::connect(addr).unwrap();
    let body = body.to_string();
    let mut head = format!(
        "POST /v1/chat/completions {version}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    let mut sent = vec![("Host", addr), ("Content-Type", "application/json")];
    for (name, value) in headers {
        sent.retain(|(n, _)| n != name);
        sent.push((name, value));
    }
    for (name, value) in sent {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    write!(socket, "{head}\r\n{body}").unwrap();
    let mut reader = BufReader::new(socket);
    let mut head = String::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" || line.is_empty() {
            break;
        }
        head.push_str(&line);
    }
    let mut rest = String::new();
    reader.read_to_string(&mut rest).unwrap();
    (head, rest)
}

/// A streamed completion read to the end: the head, and the stream's lines.
fn raw_stream(url: &str, body: &Value) -> (String, Vec<String>) {
    let (head, rest) = raw(url, "HTTP/1.1", &[], body);
    (head, unframe(&rest))
}

fn unframe(rest: &str) -> Vec<String> {
    // Undo the chunked framing: keep the lines that aren't chunk sizes.
    rest.split("\r\n")
        .filter(|l| !l.is_empty() && !l.chars().all(|c| c.is_ascii_hexdigit()))
        .flat_map(|l| l.split('\n').map(str::to_string).collect::<Vec<_>>())
        .filter(|l| !l.is_empty())
        .collect()
}

#[test]
fn streams_the_answer_as_one_chunk_with_keep_alives_while_waiting() {
    let url = start_with(
        fake(&[(
            "a",
            Script::Late(10, "Hello there", Duration::from_millis(400)),
        )]),
        None,
        Duration::from_millis(100),
    );
    let (head, lines) = raw_stream(&url, &chat(true));
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert!(
        head.to_ascii_lowercase()
            .contains("content-type: text/event-stream"),
        "{head}"
    );
    assert!(
        lines.iter().filter(|l| *l == ": keep-alive").count() >= 2,
        "{lines:?}"
    );
    let events: Vec<&String> = lines.iter().filter(|l| l.starts_with("data: ")).collect();
    assert_eq!(events.len(), 3, "{lines:?}");
    let first = json_of(&events[0]["data: ".len()..]);
    assert_eq!(first["object"], json!("chat.completion.chunk"));
    assert_eq!(
        first["choices"],
        json!([{"index": 0, "delta": {"role": "assistant", "content": "Hello there"}, "finish_reason": null}])
    );
    let last = json_of(&events[1]["data: ".len()..]);
    assert_eq!(
        last["choices"],
        json!([{"index": 0, "delta": {}, "finish_reason": "stop"}])
    );
    assert_eq!(events[2], "data: [DONE]");
}

#[test]
fn streams_an_error_as_an_event() {
    let url = start(fake(&[]));
    let (head, lines) = raw_stream(&url, &chat(true));
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    let events: Vec<&String> = lines.iter().filter(|l| l.starts_with("data: ")).collect();
    assert_eq!(events.len(), 2, "{lines:?}");
    assert_eq!(
        json_of(&events[0]["data: ".len()..])["error"]["message"],
        json!("no provider offers tiny right now")
    );
    assert_eq!(events[1], "data: [DONE]");
}

#[test]
fn refuses_a_stream_it_cannot_carry_before_streaming() {
    let url = start(fake(&[]));
    let mut bad = chat(true);
    bad["tools"] = json!([]);
    let (status, body) = call(
        "POST",
        &format!("{url}/v1/chat/completions"),
        Some(&bad),
        None,
    );
    assert_eq!(status, 400);
    assert_eq!(json_of(&body)["error"]["param"], json!("tools"));
}

#[test]
fn asks_for_the_key_when_it_has_one() {
    let url = start_with(fake(&[]), Some("s3cret"), Duration::from_secs(10));
    for key in [None, Some("wrong")] {
        let (status, body) = call("GET", &format!("{url}/v1/models"), None, key);
        assert_eq!(status, 401, "{key:?}");
        assert_eq!(json_of(&body)["error"]["code"], json!("invalid_api_key"));
    }
    assert_eq!(
        call("GET", &format!("{url}/v1/models"), None, Some("s3cret")).0,
        200
    );
}

#[test]
fn will_not_listen_beyond_this_machine_without_a_key() {
    let config = |key: Option<&str>| GatewayConfig {
        models: vec!["tiny".into()],
        key: key.map(Into::into),
        timeout: Duration::from_secs(5),
        keep_alive: Duration::from_secs(10),
    };
    let refused = GatewayServer::bind("0.0.0.0:0", gateway(fake(&[])), config(None))
        .err()
        .unwrap();
    assert_eq!(
        refused.to_string(),
        "listening beyond this machine needs a key: set LOGOS_INFERENCE_GATEWAY_KEY"
    );
    assert!(GatewayServer::bind("0.0.0.0:0", gateway(fake(&[])), config(Some("k"))).is_ok());
}

#[test]
fn refuses_requests_from_web_pages_without_a_key() {
    let url = start(fake(&[("a", Script::Quote(10, json!({"text": "x"})))]));
    let (head, body) = raw(
        &url,
        "HTTP/1.1",
        &[("Origin", "https://evil.example")],
        &chat(false),
    );
    assert!(head.starts_with("HTTP/1.1 403"), "{head}");
    assert!(
        body.contains("requests from web pages need a key"),
        "{body}"
    );
    let (head, body) = raw(
        &url,
        "HTTP/1.1",
        &[("Host", "evil.example:8090")],
        &chat(false),
    );
    assert!(head.starts_with("HTTP/1.1 403"), "{head}");
    assert!(
        body.contains("evil.example:8090 isn't this machine's name"),
        "{body}"
    );
    let (head, _) = raw(
        &url,
        "HTTP/1.1",
        &[("Content-Type", "text/plain")],
        &chat(false),
    );
    assert!(head.starts_with("HTTP/1.1 415"), "{head}");
}

#[test]
fn asks_for_the_key_on_completions_too() {
    let url = start_with(
        fake(&[("a", Script::Quote(10, json!({"text": "keyed"})))]),
        Some("s3cret"),
        Duration::from_secs(10),
    );
    assert_eq!(
        call(
            "POST",
            &format!("{url}/v1/chat/completions"),
            Some(&chat(false)),
            None
        )
        .0,
        401
    );
    let (status, body) = call(
        "POST",
        &format!("{url}/v1/chat/completions"),
        Some(&chat(false)),
        Some("s3cret"),
    );
    assert_eq!(status, 200, "{body}");
    let (head, _) = raw(
        &url,
        "HTTP/1.1",
        &[("Authorization", "bearer s3cret")],
        &chat(false),
    );
    assert!(
        head.starts_with("HTTP/1.1 200"),
        "a lower-case scheme: {head}"
    );
}

#[test]
fn refuses_a_body_over_its_limit() {
    let url = start(fake(&[]));
    let mut big = chat(false);
    big["messages"][0]["content"] = json!("a".repeat(17 << 20));
    assert_eq!(
        call(
            "POST",
            &format!("{url}/v1/chat/completions"),
            Some(&big),
            None
        )
        .0,
        413
    );
}

#[test]
fn streams_unframed_to_an_http_1_0_client() {
    let url = start(fake(&[(
        "a",
        Script::Quote(10, json!({"text": "old school"})),
    )]));
    let (head, body) = raw(&url, "HTTP/1.0", &[], &chat(true));
    assert!(head.starts_with("HTTP/1.0 200"), "{head}");
    assert!(!head.to_ascii_lowercase().contains("chunked"), "{head}");
    assert!(body.starts_with("data: {"), "{body}");
    assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
}

#[test]
fn tells_clients_not_to_retry_once_money_may_have_moved() {
    let url = start_full(
        fake(&[("a", Script::Hang(10))]),
        None,
        Duration::from_secs(10),
        Duration::from_millis(300),
    );
    let (head, body) = raw(&url, "HTTP/1.1", &[], &chat(false));
    assert!(head.starts_with("HTTP/1.1 504"), "{head}");
    assert!(
        head.to_ascii_lowercase().contains("x-should-retry: false"),
        "{head}"
    );
    assert!(body.contains("paid, but no answer in time"), "{body}");
    let url = start_full(
        fake(&[("a", Script::Silent)]),
        None,
        Duration::from_secs(10),
        Duration::from_millis(300),
    );
    let (head, _) = raw(&url, "HTTP/1.1", &[], &chat(false));
    assert!(head.starts_with("HTTP/1.1 503"), "{head}");
    assert!(
        !head.to_ascii_lowercase().contains("x-should-retry"),
        "nothing was paid: {head}"
    );
}

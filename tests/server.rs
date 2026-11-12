//! The handler contract logos-agents' `provide` speaks, against a fake OpenAI-compatible backend.

use logos_inference::job::Pricing;
use logos_inference::server::{Backend, Config, Server, TokenField};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Each request the backend got, with its Authorization header.
type Asked = Arc<Mutex<Vec<(Value, Option<String>)>>>;
type Reply = Box<dyn Fn(&Value) -> (u16, String) + Send + Sync>;

/// A backend serving /v1/chat/completions with `reply`, keeping each request it gets.
struct FakeBackend {
    url: String,
    asked: Asked,
}

fn fake_backend(reply: Reply) -> FakeBackend {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1", server.server_addr().to_ip().unwrap());
    let asked = Asked::default();
    let kept = asked.clone();
    let reply = Arc::new(reply);
    thread::spawn(move || {
        for mut request in server.incoming_requests() {
            let (kept, reply) = (kept.clone(), reply.clone());
            thread::spawn(move || {
                assert_eq!(request.url(), "/v1/chat/completions");
                let auth = request
                    .headers()
                    .iter()
                    .find(|h| h.field.equiv("Authorization"))
                    .map(|h| h.value.to_string());
                let body: Value = serde_json::from_reader(request.as_reader()).unwrap();
                kept.lock().unwrap().push((body.clone(), auth));
                let (status, text) = reply(&body);
                let _ = request
                    .respond(tiny_http::Response::from_string(text).with_status_code(status));
            });
        }
    });
    FakeBackend { url, asked }
}

fn completion(text: &str) -> (u16, String) {
    (200, json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}).to_string())
}

fn answering(text: &'static str) -> Reply {
    Box::new(move |_| completion(text))
}

fn start(backend: &FakeBackend, api_key: Option<&str>) -> String {
    start_with(backend, api_key, TokenField::MaxTokens)
}

fn start_with(backend: &FakeBackend, api_key: Option<&str>, token_field: TokenField) -> String {
    let config = Config {
        pricing: Pricing {
            input_lepta_per_token: 10,
            output_lepta_per_token: 100,
            max_tokens: 64,
        },
        backend: Backend {
            url: backend.url.clone(),
            model: "tiny-model".into(),
            api_key: api_key.map(Into::into),
            token_field,
        },
        system: None,
        timeout: Duration::from_secs(30),
    };
    let server = Server::bind("127.0.0.1:0", config).unwrap();
    let url = format!("http://{}", server.addr());
    thread::spawn(move || server.run());
    url
}

fn post(url: &str, body: &Value) -> (u16, String) {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();
    let mut response = agent.post(url).send_json(body).unwrap();
    (
        response.status().as_u16(),
        response.body_mut().read_to_string().unwrap(),
    )
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn job(id: &str, text: &str) -> Value {
    json!({"job": id, "service_id": "inference/tiny-model/v1", "input": {"text": text},
           "price_lepta": "6490", "deadline_ms": now_ms() + 3_600_000})
}

fn quote(id: &str, input: Value) -> Value {
    json!({"job": id, "service_id": "inference/tiny-model/v1", "input": input, "price_lepta": "1000"})
}

#[test]
fn quotes_a_prompt_by_its_estimated_tokens() {
    let backend = fake_backend(answering("unused"));
    let url = start(&backend, None);
    // "What is Logos?" is 14 bytes => 4 tokens, plus 4: 8 x 10, and 64 x 100 for the cap.
    let (status, body) = post(
        &format!("{url}/quote"),
        &quote("a1", json!({"text": "What is Logos?"})),
    );
    assert_eq!(status, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap(),
        json!({"price_lepta": "6480"})
    );
    assert!(backend.asked.lock().unwrap().is_empty());
}

#[test]
fn turns_away_a_request_it_cannot_price() {
    let backend = fake_backend(answering("unused"));
    let url = start(&backend, None);
    let (status, body) = post(
        &format!("{url}/quote"),
        &quote("a2", json!({"cid": "zDvZRwzm"})),
    );
    assert_eq!(
        (status, body.as_str()),
        (422, "send the prompt as text, not a CID")
    );
}

#[test]
fn runs_a_paid_job_on_the_model_and_answers_its_output() {
    let backend = fake_backend(answering("Logos is a privacy-first stack."));
    let url = start(&backend, None);
    let (status, body) = post(&format!("{url}/run"), &job("b1", "What is Logos?"));
    assert_eq!(status, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap(),
        json!({"output": {"text": "Logos is a privacy-first stack."}})
    );
    let asked = backend.asked.lock().unwrap();
    assert_eq!(asked.len(), 1);
    assert_eq!(
        asked[0].0,
        json!({"model": "tiny-model", "messages": [{"role": "user", "content": "What is Logos?"}], "max_tokens": 64, "stream": false})
    );
    assert_eq!(asked[0].1, None);
}

#[test]
fn sends_the_api_key_as_a_bearer_token() {
    let backend = fake_backend(answering("ok"));
    let url = start(&backend, Some("sk-test"));
    assert_eq!(post(&format!("{url}/run"), &job("b2", "Hi")).0, 200);
    assert_eq!(
        backend.asked.lock().unwrap()[0].1.as_deref(),
        Some("Bearer sk-test")
    );
}

#[test]
fn answers_a_job_that_arrives_twice_the_same_with_one_model_call() {
    let calls = Arc::new(Mutex::new(0));
    let counted = calls.clone();
    let backend = fake_backend(Box::new(move |_| {
        *counted.lock().unwrap() += 1;
        thread::sleep(Duration::from_millis(300));
        completion(&format!("answer {}", counted.lock().unwrap()))
    }));
    let url = start(&backend, None);
    let first = {
        let url = url.clone();
        thread::spawn(move || post(&format!("{url}/run"), &job("c1", "Hi")))
    };
    let second = post(&format!("{url}/run"), &job("c1", "Hi"));
    let third = post(&format!("{url}/run"), &job("c1", "Hi"));
    let first = first.join().unwrap();
    assert_eq!(first.0, 200);
    assert_eq!(first, second);
    assert_eq!(first, third);
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[test]
fn declines_a_job_the_model_refuses_with_its_own_reason() {
    let backend = fake_backend(Box::new(|_| {
        (
            400,
            "context length exceeded, secret upstream detail".into(),
        )
    }));
    let url = start(&backend, None);
    let (status, body) = post(&format!("{url}/run"), &job("d1", "Hi"));
    assert_eq!(status, 422);
    assert_eq!(body, "the model refused the prompt (400)");
}

#[test]
fn asks_to_be_retried_when_the_backend_fails_for_now() {
    for upstream in [500, 503, 429, 401, 404, 302] {
        let backend = fake_backend(Box::new(move |_| (upstream, "nope".into())));
        let url = start(&backend, None);
        let (status, body) = post(&format!("{url}/run"), &job("e1", "Hi"));
        assert_eq!(status, 502, "upstream {upstream}");
        assert_eq!(body, format!("the model backend answered {upstream}"));
    }
}

#[test]
fn asks_to_be_retried_when_the_model_answers_nothing() {
    let backend = fake_backend(Box::new(|_| (200, json!({"choices": []}).to_string())));
    let url = start(&backend, None);
    assert_eq!(
        post(&format!("{url}/run"), &job("e2", "Hi")),
        (502, "the model backend gave no answer".into())
    );
}

#[test]
fn asks_to_be_retried_when_the_backend_is_down() {
    let backend = FakeBackend {
        url: "http://127.0.0.1:9/v1".into(),
        asked: Default::default(),
    };
    let url = start(&backend, None);
    assert_eq!(
        post(&format!("{url}/run"), &job("e3", "Hi")),
        (503, "the model backend can't be reached".into())
    );
}

#[test]
fn gives_up_on_the_model_at_the_jobs_deadline() {
    let backend = fake_backend(Box::new(|_| {
        thread::sleep(Duration::from_secs(5));
        completion("too late")
    }));
    let url = start(&backend, None);
    let mut late = job("f1", "Hi");
    // logos-agents stops waiting 6 minutes before the deadline: a second is left of that.
    late["deadline_ms"] = json!(now_ms() + 361_000);
    let started = Instant::now();
    assert_eq!(
        post(&format!("{url}/run"), &late),
        (504, "the model backend didn't answer in time".into())
    );
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "took {:?}",
        started.elapsed()
    );
}

#[test]
fn quotes_while_a_job_waits_on_the_model() {
    let backend = fake_backend(Box::new(|_| {
        thread::sleep(Duration::from_secs(3));
        completion("slow")
    }));
    let url = start(&backend, None);
    let running = {
        let url = url.clone();
        thread::spawn(move || post(&format!("{url}/run"), &job("g1", "Hi")))
    };
    thread::sleep(Duration::from_millis(200));
    let started = Instant::now();
    assert_eq!(
        post(&format!("{url}/quote"), &quote("g2", json!({"text": "Hi"}))).0,
        200
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(running.join().unwrap().0, 200);
}

#[test]
fn refuses_a_body_it_cannot_read_and_unknown_paths() {
    let backend = fake_backend(answering("unused"));
    let url = start(&backend, None);
    assert_eq!(post(&format!("{url}/run"), &json!({"job": "h1"})).0, 400);
    assert_eq!(post(&format!("{url}/other"), &job("h2", "Hi")).0, 404);
}

#[test]
fn declines_a_paid_job_it_cannot_run() {
    let backend = fake_backend(answering("unused"));
    let url = start(&backend, None);
    let mut over = job(
        "i1",
        r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":65}"#,
    );
    over["price_lepta"] = json!("1000000");
    assert_eq!(
        post(&format!("{url}/run"), &over),
        (422, "max_tokens is at most 64".into())
    );
    assert!(backend.asked.lock().unwrap().is_empty());
}

#[test]
fn declines_a_job_paid_below_its_price() {
    let backend = fake_backend(answering("unused"));
    let url = start(&backend, None);
    // "Hi" costs (1 + 4) x 10 + 64 x 100 = 6450 lepta.
    let mut cheap = job("i2", "Hi");
    cheap["price_lepta"] = json!("6449");
    assert_eq!(
        post(&format!("{url}/run"), &cheap),
        (422, "paid 6449 lepta, but this job costs 6450".into())
    );
    cheap["price_lepta"] = json!("6450");
    cheap["job"] = json!("i3");
    assert_eq!(post(&format!("{url}/run"), &cheap).0, 200);
    assert_eq!(backend.asked.lock().unwrap().len(), 1);
}

#[test]
fn asks_the_model_again_when_a_job_failed_for_now() {
    let calls = Arc::new(Mutex::new(0));
    let counted = calls.clone();
    let backend = fake_backend(Box::new(move |_| {
        let mut calls = counted.lock().unwrap();
        *calls += 1;
        if *calls == 1 {
            (500, "busy".into())
        } else {
            completion("second time")
        }
    }));
    let url = start(&backend, None);
    assert_eq!(post(&format!("{url}/run"), &job("j1", "Hi")).0, 502);
    assert_eq!(post(&format!("{url}/run"), &job("j1", "Hi")).0, 200);
    assert_eq!(*calls.lock().unwrap(), 2);
}

#[test]
fn refuses_a_body_over_its_limit_and_a_job_id_it_would_not_send() {
    let backend = fake_backend(answering("unused"));
    let url = start(&backend, None);
    let big = job("k1", &"a".repeat(17 << 20));
    assert_eq!(post(&format!("{url}/run"), &big).0, 413);
    for bad in ["", "not/an-id", &"a".repeat(129)] {
        assert_eq!(
            post(&format!("{url}/run"), &job(bad, "Hi")).0,
            400,
            "{bad:?}"
        );
    }
    assert!(backend.asked.lock().unwrap().is_empty());
}

#[test]
fn a_job_that_fails_while_a_duplicate_waits_is_still_asked_once_at_a_time() {
    let calls = Arc::new(Mutex::new(0));
    let counted = calls.clone();
    let backend = fake_backend(Box::new(move |_| {
        let call = {
            let mut calls = counted.lock().unwrap();
            *calls += 1;
            *calls
        };
        thread::sleep(Duration::from_millis(300));
        if call == 1 {
            (500, "busy".into())
        } else {
            completion(&format!("answer {call}"))
        }
    }));
    let url = start(&backend, None);
    let first = {
        let url = url.clone();
        thread::spawn(move || post(&format!("{url}/run"), &job("l1", "Hi")))
    };
    thread::sleep(Duration::from_millis(100));
    // Waits on the first, which fails, then asks the model itself.
    let waiting = {
        let url = url.clone();
        thread::spawn(move || post(&format!("{url}/run"), &job("l1", "Hi")))
    };
    thread::sleep(Duration::from_millis(350));
    // Arrives while the second is asking: it waits for that answer.
    let third = post(&format!("{url}/run"), &job("l1", "Hi"));
    assert_eq!(first.join().unwrap().0, 502);
    let second = waiting.join().unwrap();
    assert_eq!(second.0, 200);
    assert_eq!(third, second);
    assert_eq!(*calls.lock().unwrap(), 2);
}

#[test]
fn does_not_ask_the_model_for_a_job_too_close_to_its_deadline() {
    let backend = fake_backend(answering("too late"));
    let url = start(&backend, None);
    for left in [-1, 0, 300_000, 360_000] {
        let mut late = job("m1", "Hi");
        late["deadline_ms"] = json!(now_ms() as i64 + left);
        assert_eq!(
            post(&format!("{url}/run"), &late),
            (504, "too close to the job's deadline to deliver".into()),
            "{left} ms left"
        );
    }
    assert!(backend.asked.lock().unwrap().is_empty());
}

#[test]
fn declines_a_job_whose_answer_the_cap_cut_off_before_it_began() {
    // A thinking model can spend the whole cap reasoning and answer nothing: asking again
    // would spend it again.
    let backend = fake_backend(Box::new(|_| {
        (200, json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "length"}]}).to_string())
    }));
    let url = start(&backend, None);
    assert_eq!(
        post(&format!("{url}/run"), &job("n1", "Hi")),
        (
            422,
            "the model used its whole output cap without answering".into()
        )
    );
}

#[test]
fn names_the_output_cap_as_the_backend_takes_it() {
    let backend = fake_backend(answering("ok"));
    let url = start_with(&backend, None, TokenField::MaxCompletionTokens);
    assert_eq!(post(&format!("{url}/run"), &job("o1", "Hi")).0, 200);
    let asked = &backend.asked.lock().unwrap()[0].0;
    assert_eq!(asked["max_completion_tokens"], json!(64));
    assert!(asked.get("max_tokens").is_none(), "{asked}");
}

#[test]
fn asks_to_be_retried_when_the_model_stops_with_nothing() {
    let backend = fake_backend(Box::new(|_| {
        (200, json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]}).to_string())
    }));
    let url = start(&backend, None);
    assert_eq!(
        post(&format!("{url}/run"), &job("p1", "Hi")),
        (502, "the model backend gave no answer".into())
    );
}

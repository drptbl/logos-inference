//! The gateway's HTTP side: OpenAI's `/v1/models` and `/v1/chat/completions`, each
//! completion bought over Logos.

use crate::agent::Agent;
use crate::gateway::Gateway;
use crate::job::Prompt;
use crate::openai::{ApiError, Call, translate};
use serde_json::{Value, json};
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct GatewayConfig {
    /// The models served, each bought as `inference/<model>/v1`.
    pub models: Vec<String>,
    /// The Bearer key callers must give, if any.
    pub key: Option<String>,
    /// The longest one completion may take.
    pub timeout: Duration,
    /// How often a streamed completion says it's still waiting.
    pub keep_alive: Duration,
}

pub struct GatewayServer<A: Agent> {
    http: tiny_http::Server,
    state: Arc<State<A>>,
}

struct State<A: Agent> {
    gateway: Gateway<A>,
    config: GatewayConfig,
}

/// The largest request body read.
const MAX_BODY: u64 = 16 << 20;

impl<A: Agent + 'static> GatewayServer<A> {
    /// Listens on `addr`. Every completion spends from the wallet, so an address beyond this
    /// machine needs a key.
    pub fn bind(addr: &str, gateway: Gateway<A>, config: GatewayConfig) -> io::Result<Self> {
        let http = tiny_http::Server::http(addr).map_err(io::Error::other)?;
        let ip = http.server_addr().to_ip().map(|a| a.ip());
        if config.key.is_none() && !ip.is_some_and(|ip| ip.is_loopback()) {
            return Err(io::Error::other(
                "listening beyond this machine needs a key: set LOGOS_INFERENCE_GATEWAY_KEY",
            ));
        }
        Ok(GatewayServer {
            http,
            state: Arc::new(State { gateway, config }),
        })
    }

    pub fn addr(&self) -> SocketAddr {
        self.http
            .server_addr()
            .to_ip()
            .expect("listening on an IP address")
    }

    /// Serves until the process ends, each request on its own thread.
    pub fn run(self) {
        for request in self.http.incoming_requests() {
            let state = self.state.clone();
            thread::spawn(move || state.handle(request));
        }
    }
}

impl<A: Agent> State<A> {
    fn handle(&self, mut request: tiny_http::Request) {
        if !self.authorized(&request) {
            return respond(
                request,
                Err(ApiError::new(401, "a valid API key is required")),
            );
        }
        let path = request
            .url()
            .split('?')
            .next()
            .unwrap_or_default()
            .to_string();
        match (request.method(), path.as_str()) {
            (tiny_http::Method::Get, "/v1/models") => {
                let data: Vec<Value> = self
                    .config
                    .models
                    .iter()
                    .map(|m| json!({"id": m, "object": "model", "created": 0, "owned_by": "logos"}))
                    .collect();
                respond(request, Ok(json!({"object": "list", "data": data})))
            }
            (tiny_http::Method::Post, "/v1/chat/completions") => {
                let call =
                    read_body(&mut request).and_then(|body| translate(&body, &self.config.models));
                match call {
                    Ok(call) if call.stream => self.stream(request, &call),
                    Ok(call) => {
                        let answer = self
                            .gateway
                            .complete(&call, Instant::now() + self.config.timeout);
                        respond(request, answer.map(|text| completion(&call, &text)))
                    }
                    Err(e) => respond(request, Err(e)),
                }
            }
            _ => respond(
                request,
                Err(ApiError::new(404, format!("no route for {path}"))),
            ),
        }
    }

    fn authorized(&self, request: &tiny_http::Request) -> bool {
        let Some(key) = &self.config.key else {
            return true;
        };
        let given = request
            .headers()
            .iter()
            .find(|h| h.field.equiv("Authorization"))
            .map(|h| h.value.as_str().to_string())
            .unwrap_or_default();
        let expected = format!("Bearer {key}");
        // Compared in full whatever matches, so the time taken doesn't tell how much did.
        given.len() == expected.len()
            && given
                .bytes()
                .zip(expected.bytes())
                .fold(0, |acc, (a, b)| acc | (a ^ b))
                == 0
    }

    /// Streams the completion: keep-alive comments while it's bought, then the whole answer
    /// as one chunk. Written straight to the socket, since tiny_http buffers what it sends.
    fn stream(&self, request: tiny_http::Request, call: &Call) {
        let (done, answer) = mpsc::channel();
        let deadline = Instant::now() + self.config.timeout;
        thread::scope(|scope| {
            scope.spawn(|| {
                let _ = done.send(self.gateway.complete(call, deadline));
            });
            let mut socket = request.into_writer();
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n";
            let mut open = socket
                .write_all(head.as_bytes())
                .and_then(|_| socket.flush())
                .is_ok();
            let result = loop {
                match answer.recv_timeout(self.config.keep_alive) {
                    Ok(result) => break result,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        open = open && chunk(&mut socket, ": keep-alive\n\n").is_ok();
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        break Err(ApiError::new(500, "the gateway stopped"));
                    }
                }
            };
            if !open {
                return;
            }
            let mut events = Vec::new();
            match result {
                Ok(text) => {
                    let (id, created) = (completion_id(), now_s());
                    let piece = |delta: Value, finish: Value| {
                        json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": call.model,
                               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
                    };
                    events.push(
                        piece(json!({"role": "assistant", "content": text}), Value::Null)
                            .to_string(),
                    );
                    events.push(piece(json!({}), json!("stop")).to_string());
                }
                Err(e) => events.push(e.body().to_string()),
            }
            events.push("[DONE]".into());
            for event in events {
                if chunk(&mut socket, &format!("data: {event}\n\n")).is_err() {
                    return;
                }
            }
            let _ = socket.write_all(b"0\r\n\r\n").and_then(|_| socket.flush());
        });
    }
}

/// One chunk of a chunked body, flushed at once.
fn chunk(socket: &mut impl Write, text: &str) -> io::Result<()> {
    write!(socket, "{:x}\r\n{text}\r\n", text.len())?;
    socket.flush()
}

fn read_body(request: &mut tiny_http::Request) -> Result<Vec<u8>, ApiError> {
    let mut body = Vec::new();
    request
        .as_reader()
        .take(MAX_BODY + 1)
        .read_to_end(&mut body)
        .map_err(|_| ApiError::new(400, "the request couldn't be read"))?;
    if body.len() as u64 > MAX_BODY {
        return Err(ApiError::new(413, "the request is too big"));
    }
    Ok(body)
}

fn respond(request: tiny_http::Request, answer: Result<Value, ApiError>) {
    let (status, body) = match answer {
        Ok(body) => (200, body),
        Err(e) => (e.status, e.body()),
    };
    let response = tiny_http::Response::from_string(body.to_string())
        .with_status_code(status)
        .with_header(
            tiny_http::Header::from_bytes("Content-Type", "application/json")
                .expect("a valid header"),
        );
    let _ = request.respond(response);
}

/// The answer as a `chat.completion`, with token counts estimated as the provider prices them.
fn completion(call: &Call, text: &str) -> Value {
    let prompt = Prompt {
        messages: call.messages.clone(),
        max_tokens: 0,
    }
    .input_tokens();
    let answer = (text.len() as u64).div_ceil(4);
    json!({
        "id": completion_id(),
        "object": "chat.completion",
        "created": now_s(),
        "model": call.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": answer, "total_tokens": prompt + answer},
    })
}

fn completion_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!(
        "chatcmpl-{nanos:x}{:x}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

fn now_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

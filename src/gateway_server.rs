//! The gateway's HTTP side: OpenAI's `/v1/models` and `/v1/chat/completions`, each
//! completion bought over Logos.

use crate::agent::Agent;
use crate::gateway::Gateway;
use crate::job::{Prompt, estimate_tokens};
use crate::openai::{ApiError, Call, translate};
use serde_json::{Value, json};
use std::io::{self, Write};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
    /// The most completions bought at once. More are turned away until one finishes.
    pub max_calls: usize,
    /// How long a write to a client may block, so one that stops reading lets go.
    pub send_timeout: Duration,
}

pub struct GatewayServer<A: Agent> {
    http: tiny_http::Server,
    state: Arc<State<A>>,
}

struct State<A: Agent> {
    gateway: Gateway<A>,
    config: GatewayConfig,
    /// Completions being bought now.
    calls: AtomicUsize,
}

/// One completion's place under `max_calls`, given back when it ends.
struct Slot<'a>(&'a AtomicUsize);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl<A: Agent + 'static> GatewayServer<A> {
    /// Listens on `addr`. Every completion spends from the wallet, so an address beyond this
    /// machine needs a key.
    pub fn bind(addr: &str, gateway: Gateway<A>, config: GatewayConfig) -> io::Result<Self> {
        let listener = std::net::TcpListener::bind(addr)?;
        // Linux gives each connection the listener's send timeout. macOS doesn't, so there a
        // client that stops reading holds its slot until it goes away.
        socket2::SockRef::from(&listener).set_write_timeout(Some(config.send_timeout))?;
        let http = tiny_http::Server::from_listener(listener, None).map_err(io::Error::other)?;
        let ip = http.server_addr().to_ip().map(|a| a.ip());
        if config.key.is_none() && !ip.is_some_and(|ip| ip.is_loopback()) {
            return Err(io::Error::other(
                "listening beyond this machine needs a key: set LOGOS_INFERENCE_GATEWAY_KEY",
            ));
        }
        Ok(GatewayServer {
            http,
            state: Arc::new(State {
                gateway,
                config,
                calls: AtomicUsize::new(0),
            }),
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
        if let Err(e) = self.admit(&request) {
            return respond(request, Err(e));
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
                let Some(_slot) = self.slot() else {
                    let max = self.config.max_calls;
                    let busy = format!(
                        "{max} completions are running, as many as this gateway takes at once: try again shortly"
                    );
                    return respond(request, Err(ApiError::new(429, busy)));
                };
                let call =
                    json_body(&mut request).and_then(|body| translate(&body, &self.config.models));
                match call {
                    Ok(call) if call.stream => self.stream(request, &call),
                    Ok(call) => {
                        // tiny_http can't tell when a client leaves mid-request.
                        let answer = self.gateway.complete(
                            &call,
                            Instant::now() + self.config.timeout,
                            &AtomicBool::new(false),
                        );
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

    fn slot(&self) -> Option<Slot<'_>> {
        let taken = self.calls.fetch_add(1, Ordering::Relaxed);
        let slot = Slot(&self.calls);
        (taken < self.config.max_calls).then_some(slot)
    }

    /// Lets a request in. With a key, only callers that send it. Without one, only local
    /// clients that aren't web pages: a page in the user's browser can reach localhost, but its
    /// requests carry an `Origin`, and a page using DNS rebinding sends its own name as `Host`.
    fn admit(&self, request: &tiny_http::Request) -> Result<(), ApiError> {
        let header = |name: &'static str| {
            request
                .headers()
                .iter()
                .find(|h| h.field.equiv(name))
                .map(|h| h.value.as_str().to_string())
        };
        if let Some(key) = &self.config.key {
            let given = header("Authorization").unwrap_or_default();
            let token = match given.split_once(' ') {
                Some((scheme, token)) if scheme.eq_ignore_ascii_case("bearer") => token,
                _ => "",
            };
            // Compared in full whatever matches, so the time taken doesn't tell how much did.
            let matches = token.len() == key.len()
                && token
                    .bytes()
                    .zip(key.bytes())
                    .fold(0, |acc, (a, b)| acc | (a ^ b))
                    == 0;
            return if matches {
                Ok(())
            } else {
                Err(ApiError::new(401, "a valid API key is required").code("invalid_api_key"))
            };
        }
        if header("Origin").is_some() {
            return Err(ApiError::new(
                403,
                "requests from web pages need a key: set LOGOS_INFERENCE_GATEWAY_KEY",
            ));
        }
        let host = header("Host").unwrap_or_default();
        let name = match host.rsplit_once(':') {
            Some((name, port)) if port.bytes().all(|b| b.is_ascii_digit()) => name,
            _ => host.as_str(),
        };
        if !matches!(
            name.to_ascii_lowercase().as_str(),
            "127.0.0.1" | "localhost" | "[::1]"
        ) {
            return Err(ApiError::new(
                403,
                format!("{host} isn't this machine's name"),
            ));
        }
        Ok(())
    }

    /// Streams the completion: keep-alive comments while it's bought, then the whole answer
    /// as one chunk. Written straight to the socket, since tiny_http buffers what it sends.
    ///
    /// A client that leaves is noticed at the next keep-alive, and nothing more is paid for it.
    fn stream(&self, request: tiny_http::Request, call: &Call) {
        let deadline = Instant::now() + self.config.timeout;
        // An HTTP/1.0 client can't read a chunked body, and an unframed one only ends when the
        // connection closes, which tiny_http may not do: it gets the events whole, with their
        // length, once the answer is in.
        if request.http_version() == &tiny_http::HTTPVersion(1, 0) {
            let result = self
                .gateway
                .complete(call, deadline, &AtomicBool::new(false));
            let header = tiny_http::Header::from_bytes("Content-Type", "text/event-stream")
                .expect("a valid header");
            let _ = request.respond(
                tiny_http::Response::from_string(events(call, result)).with_header(header),
            );
            return;
        }
        let (done, answer) = mpsc::channel();
        let gone = AtomicBool::new(false);
        thread::scope(|scope| {
            let gone = &gone;
            // The sender moves into the worker, so a worker that dies ends the wait below.
            scope.spawn(move || {
                let _ = done.send(self.gateway.complete(call, deadline, gone));
            });
            let mut socket = request.into_writer();
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n";
            let mut open = socket
                .write_all(head.as_bytes())
                .and_then(|_| socket.flush())
                .is_ok();
            let result = loop {
                if !open {
                    gone.store(true, Ordering::Relaxed);
                }
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
            if open && chunk(&mut socket, &events(call, result)).is_ok() {
                let _ = socket.write_all(b"0\r\n\r\n").and_then(|_| socket.flush());
            }
        });
    }
}

/// One chunk of a chunked body, flushed at once.
fn chunk(socket: &mut impl Write, text: &str) -> io::Result<()> {
    write!(socket, "{:x}\r\n{text}\r\n", text.len())?;
    socket.flush()
}

/// A streamed completion's events: the whole answer as one chunk and the finish, or the
/// error, then `[DONE]`.
fn events(call: &Call, result: Result<String, ApiError>) -> String {
    let mut events = Vec::new();
    match result {
        Ok(text) => {
            let (id, created) = (completion_id(), now_s());
            let piece = |delta: Value, finish: Value| {
                json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": call.model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
            };
            events.push(
                piece(json!({"role": "assistant", "content": text}), Value::Null).to_string(),
            );
            events.push(piece(json!({}), json!("stop")).to_string());
        }
        Err(e) => events.push(e.body().to_string()),
    }
    events.push("[DONE]".into());
    events.iter().map(|e| format!("data: {e}\n\n")).collect()
}

/// The body of a JSON request. Asking for JSON keeps web pages out too: a browser sends it
/// only after a CORS preflight, which this server never answers.
fn json_body(request: &mut tiny_http::Request) -> Result<Vec<u8>, ApiError> {
    let json = request.headers().iter().any(|h| {
        h.field.equiv("Content-Type")
            && h.value
                .as_str()
                .to_ascii_lowercase()
                .starts_with("application/json")
    });
    if !json {
        return Err(ApiError::new(415, "send the request as application/json"));
    }
    crate::http::read_body(request).map_err(|(status, reason)| ApiError::new(status, reason))
}

fn respond(request: tiny_http::Request, answer: Result<Value, ApiError>) {
    let (status, body, retry) = match answer {
        Ok(body) => (200, body, true),
        Err(e) => (e.status, e.body(), e.retryable),
    };
    let header = |name: &str, value: &str| {
        tiny_http::Header::from_bytes(name, value).expect("a valid header")
    };
    let mut response = tiny_http::Response::from_string(body.to_string())
        .with_status_code(status)
        .with_header(header("Content-Type", "application/json"));
    if !retry {
        // OpenAI's own SDKs read this before retrying a 5xx on their own.
        response = response.with_header(header("x-should-retry", "false"));
    }
    let _ = request.respond(response);
}

/// The answer as a `chat.completion`, with token counts estimated as the provider prices them.
fn completion(call: &Call, text: &str) -> Value {
    let prompt = Prompt {
        messages: call.messages.clone(),
        max_tokens: 0,
    }
    .input_tokens();
    let answer = estimate_tokens(text);
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

//! The handler logos-agents' `provide` posts to: `/quote` prices a request, `/run` answers a
//! paid job with the model's output. See logos-agents' docs/providers.md for the contract.

use crate::job::{Pricing, Prompt, Refusal, parse};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::io::{self, Read};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// An OpenAI-compatible API: Ollama, llama.cpp's llama-server, vLLM or a hosted one.
pub struct Backend {
    /// The API's base URL, up to `/v1`, e.g. `http://127.0.0.1:11434/v1`.
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
}

pub struct Config {
    pub pricing: Pricing,
    pub backend: Backend,
    /// The offer's system prompt, put before the buyer's messages.
    pub system: Option<String>,
    /// The longest a job waits on the model, when its deadline is further off.
    pub timeout: Duration,
}

/// The largest request body read. logos-agents caps an input at 8 MiB.
const MAX_BODY: u64 = 16 << 20;
/// Paid jobs whose output is kept, for a job that arrives again.
const KEPT_JOBS: usize = 10_000;

pub struct Server {
    http: tiny_http::Server,
    state: Arc<State>,
}

struct State {
    config: Config,
    agent: ureq::Agent,
    jobs: Jobs,
}

impl Server {
    pub fn bind(addr: &str, config: Config) -> io::Result<Server> {
        let http = tiny_http::Server::http(addr).map_err(io::Error::other)?;
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .build()
            .into();
        let jobs = Jobs::new(KEPT_JOBS);
        Ok(Server {
            http,
            state: Arc::new(State {
                config,
                agent,
                jobs,
            }),
        })
    }

    pub fn addr(&self) -> SocketAddr {
        self.http
            .server_addr()
            .to_ip()
            .expect("listening on an IP address")
    }

    /// Serves until the process ends, each request on its own thread, so a job waiting on
    /// the model never holds up a quote.
    pub fn run(self) {
        for request in self.http.incoming_requests() {
            let state = self.state.clone();
            thread::spawn(move || handle(&state, request));
        }
    }
}

/// What the handler answers: a status, and JSON or a reason for the buyer.
struct Answer {
    status: u16,
    body: String,
}

impl Answer {
    fn json(body: Value) -> Answer {
        Answer {
            status: 200,
            body: body.to_string(),
        }
    }

    fn reason(status: u16, reason: impl Into<String>) -> Answer {
        Answer {
            status,
            body: reason.into(),
        }
    }
}

impl From<Refusal> for Answer {
    fn from(Refusal(reason): Refusal) -> Answer {
        Answer::reason(422, reason)
    }
}

/// The fields of a quote or job request this handler reads.
#[derive(Deserialize)]
struct Ask {
    job: String,
    input: Value,
    /// What the buyer paid, on a job.
    price_lepta: Option<String>,
    deadline_ms: Option<u64>,
}

fn handle(state: &State, mut request: tiny_http::Request) {
    let answer = match (request.method(), request.url()) {
        (tiny_http::Method::Post, "/quote") => read(&mut request).map(|ask| state.quote(&ask)),
        (tiny_http::Method::Post, "/run") => read(&mut request).map(|ask| state.run(&ask)),
        _ => Ok(Answer::reason(404, "not found")),
    }
    .unwrap_or_else(|a| a);
    let content_type = if answer.status == 200 {
        "application/json"
    } else {
        "text/plain; charset=utf-8"
    };
    let response = tiny_http::Response::from_string(answer.body)
        .with_status_code(answer.status)
        .with_header(
            tiny_http::Header::from_bytes("Content-Type", content_type).expect("a valid header"),
        );
    let _ = request.respond(response);
}

fn read(request: &mut tiny_http::Request) -> Result<Ask, Answer> {
    let mut body = Vec::new();
    request
        .as_reader()
        .take(MAX_BODY + 1)
        .read_to_end(&mut body)
        .map_err(|_| Answer::reason(400, "the request couldn't be read"))?;
    if body.len() as u64 > MAX_BODY {
        return Err(Answer::reason(413, "the request is too big"));
    }
    let ask: Ask =
        serde_json::from_slice(&body).map_err(|_| Answer::reason(400, "not a job request"))?;
    // logos-agents' job ids are 32 hex characters; this bounds what is kept and logged.
    if ask.job.is_empty()
        || ask.job.len() > 128
        || !ask.job.bytes().all(|b| b.is_ascii_alphanumeric())
    {
        return Err(Answer::reason(400, "not a job id"));
    }
    Ok(ask)
}

impl State {
    fn prompt(&self, ask: &Ask) -> Result<Prompt, Answer> {
        Ok(parse(
            &ask.input,
            &self.config.pricing,
            self.config.system.as_deref(),
        )?)
    }

    fn quote(&self, ask: &Ask) -> Answer {
        match self.prompt(ask) {
            Ok(prompt) => {
                let price = prompt.price(&self.config.pricing);
                eprintln!("quote {:?}: {price} lepta", ask.job);
                Answer::json(json!({ "price_lepta": price.to_string() }))
            }
            Err(answer) => answer,
        }
    }

    fn run(&self, ask: &Ask) -> Answer {
        let prompt = match self.paid_prompt(ask) {
            Ok(prompt) => prompt,
            Err(answer) => return answer,
        };
        // One model call per job: a second arrival waits for the first and gets its output.
        let slot = self.jobs.slot(&ask.job);
        let mut output = slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(body) = output.as_ref() {
            return Answer::json(body.clone());
        }
        let answer = match self.complete(&prompt, ask.deadline_ms) {
            Ok(text) => {
                let body = json!({ "output": { "text": text } });
                *output = Some(body.clone());
                Answer::json(body)
            }
            // Nothing to keep: the next arrival asks the model again, in this job's turn.
            Err(answer) => answer,
        };
        eprintln!("job {:?}: {}", ask.job, answer.status);
        answer
    }

    /// The job's prompt, if the buyer paid at least what it costs at this offer's rates.
    /// An offer served without this quote handler would otherwise run at its fallback price.
    fn paid_prompt(&self, ask: &Ask) -> Result<Prompt, Answer> {
        let paid = ask
            .price_lepta
            .as_deref()
            .and_then(|p| p.parse::<u128>().ok())
            .ok_or_else(|| Answer::reason(400, "not a job request"))?;
        let prompt = self.prompt(ask)?;
        let price = prompt.price(&self.config.pricing);
        if paid < price {
            return Err(Answer::reason(
                422,
                format!("paid {paid} lepta, but this job costs {price}"),
            ));
        }
        Ok(prompt)
    }

    /// Asks the model, until the job's deadline at the latest.
    fn complete(&self, prompt: &Prompt, deadline_ms: Option<u64>) -> Result<String, Answer> {
        let backend = &self.config.backend;
        let mut timeout = self.config.timeout;
        if let Some(deadline_ms) = deadline_ms {
            let left = deadline_ms.saturating_sub(now_ms());
            if left == 0 {
                return Err(Answer::reason(504, "the job's deadline has passed"));
            }
            timeout = timeout.min(Duration::from_millis(left));
        }
        let url = format!("{}/chat/completions", backend.url.trim_end_matches('/'));
        let mut request = self.agent.post(&url);
        if let Some(key) = &backend.api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        let sent = request
            .config()
            .timeout_global(Some(timeout))
            .build()
            .send_json(json!({
                "model": backend.model,
                "messages": prompt.messages,
                "max_tokens": prompt.max_tokens,
                "stream": false,
            }));
        let mut response = match sent {
            Ok(response) => response,
            Err(ureq::Error::Timeout(_)) => {
                return Err(Answer::reason(
                    504,
                    "the model backend didn't answer in time",
                ));
            }
            Err(_) => return Err(Answer::reason(503, "the model backend can't be reached")),
        };
        match response.status().as_u16() {
            200..=299 => {}
            // The prompt itself is the problem, e.g. too long for the model: decline, which
            // refunds the buyer. The upstream body stays here.
            status @ (400 | 413 | 422) => {
                return Err(Answer::reason(
                    422,
                    format!("the model refused the prompt ({status})"),
                ));
            }
            // Anything else may pass, or is the provider's to fix: ask to be tried again.
            status => {
                return Err(Answer::reason(
                    502,
                    format!("the model backend answered {status}"),
                ));
            }
        }
        let body: Value = match response.body_mut().read_json() {
            Ok(body) => body,
            Err(ureq::Error::Timeout(_)) => {
                return Err(Answer::reason(
                    504,
                    "the model backend didn't answer in time",
                ));
            }
            Err(_) => Value::Null,
        };
        match body
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
        {
            Some(text) if !text.is_empty() => Ok(text.to_string()),
            _ => Err(Answer::reason(502, "the model backend gave no answer")),
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Each paid job's turn at the model, and its output once the model gave it, for the most
/// recent `KEPT_JOBS` jobs. Kept in memory only, so a restart forgets them.
struct Jobs {
    inner: Mutex<(HashMap<String, Slot>, VecDeque<String>)>,
    kept: usize,
}

type Slot = Arc<Mutex<Option<Value>>>;

impl Jobs {
    fn new(kept: usize) -> Jobs {
        Jobs {
            inner: Mutex::default(),
            kept,
        }
    }

    fn slot(&self, job: &str) -> Slot {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (slots, order) = &mut *inner;
        if let Some(slot) = slots.get(job) {
            return slot.clone();
        }
        let slot = Slot::default();
        slots.insert(job.to_string(), slot.clone());
        order.push_back(job.to_string());
        // Over the limit, the oldest job nobody is running goes: one that is running would
        // be asked again on its next arrival.
        if order.len() > self.kept
            && let Some(at) = order
                .iter()
                .position(|j| slots.get(j).is_some_and(|s| Arc::strong_count(s) == 1))
            && let Some(oldest) = order.remove(at)
        {
            slots.remove(&oldest);
        }
        slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn over_the_limit_the_oldest_job_nobody_runs_goes() {
        let jobs = Jobs::new(2);
        let running = jobs.slot("a");
        drop(jobs.slot("b"));
        drop(jobs.slot("c"));
        let kept = jobs.inner.lock().unwrap();
        assert!(kept.0.contains_key("a"), "a running job was dropped");
        assert!(!kept.0.contains_key("b"));
        assert!(kept.0.contains_key("c"));
        assert_eq!(kept.1, ["a", "c"]);
        drop(kept);
        assert!(Arc::ptr_eq(&running, &jobs.slot("a")));
    }
}

//! OpenAI's chat completions, as the gateway takes them and answers them, translated to the
//! provider wire format (`inference/<model>/v1`, see job.rs).

use crate::job::{Message, Role};
use serde_json::{Map, Value, json};

/// A completion request the gateway can carry over Logos.
#[derive(Debug)]
pub struct Call {
    pub model: String,
    pub messages: Vec<Message>,
    /// The output cap the client asked for, if any: the offer's otherwise.
    pub max_tokens: Option<u32>,
    pub stream: bool,
}

impl Call {
    /// The job input: a chat request in the provider wire format.
    pub fn input(&self) -> String {
        let mut input = Map::new();
        input.insert("messages".into(), json!(self.messages));
        if let Some(cap) = self.max_tokens {
            input.insert("max_tokens".into(), json!(cap));
        }
        Value::Object(input).to_string()
    }
}

/// An error as OpenAI clients read it.
#[derive(Debug, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub param: Option<String>,
    pub code: Option<String>,
    /// Whether a client may simply try again. Not when money may have moved: trying again
    /// would buy the answer a second time.
    pub retryable: bool,
}

impl ApiError {
    pub fn new(status: u16, message: impl Into<String>) -> ApiError {
        ApiError {
            status,
            message: message.into(),
            param: None,
            code: None,
            retryable: true,
        }
    }

    fn invalid(param: &str, message: impl Into<String>) -> ApiError {
        ApiError {
            param: Some(param.into()),
            ..ApiError::new(400, message)
        }
    }

    pub fn code(self, code: &str) -> ApiError {
        ApiError {
            code: Some(code.into()),
            ..self
        }
    }

    pub fn not_retryable(self) -> ApiError {
        ApiError {
            retryable: false,
            ..self
        }
    }

    /// The JSON body OpenAI answers an error with.
    pub fn body(&self) -> Value {
        let kind = match self.status {
            400..=499 => "invalid_request_error",
            _ => "api_error",
        };
        json!({"error": {"message": self.message, "type": kind, "param": self.param, "code": self.code}})
    }
}

/// Sampling hints clients send by default. The wire format can't carry them, and leaving them
/// out changes how an answer reads, not what kind of answer it is, so they are dropped.
const DROPPED: &[&str] = &[
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "user",
    "stop",
    "stream_options",
    "logit_bias",
    "metadata",
    "store",
    "service_tier",
    "parallel_tool_calls",
    "reasoning_effort",
    "verbosity",
    "prompt_cache_key",
    "safety_identifier",
];

/// Reads a chat completion request for one of `models`. Anything that would change the
/// kind of answer (tools, several choices, a response format, logprobs, audio or images)
/// is refused, as is any field this doesn't know.
pub fn translate(body: &[u8], models: &[String]) -> Result<Call, ApiError> {
    let Ok(Value::Object(body)) = serde_json::from_slice::<Value>(body) else {
        return Err(ApiError::new(400, "the body isn't a JSON object"));
    };
    for (field, value) in &body {
        let carried = matches!(
            field.as_str(),
            "model" | "messages" | "max_tokens" | "max_completion_tokens" | "stream"
        );
        let harmless = DROPPED.contains(&field.as_str())
            || match field.as_str() {
                "n" => value == &json!(1),
                "logprobs" => value == &json!(false),
                "response_format" => value == &json!({"type": "text"}),
                "tool_choice" => value == &json!("none"),
                "modalities" => value == &json!(["text"]),
                _ => false,
            };
        if !carried && !harmless {
            return Err(ApiError::invalid(
                field,
                format!("{field} isn't supported over Logos yet"),
            ));
        }
    }
    let Some(model) = body.get("model").and_then(Value::as_str) else {
        return Err(ApiError::invalid("model", "model is required"));
    };
    if !models.iter().any(|m| m == model) {
        return Err(
            ApiError::new(404, format!("the model {model} isn't served here"))
                .code("model_not_found"),
        );
    }
    let messages = read_messages(body.get("messages"))?;
    let max_tokens = read_cap(&body)?;
    let stream = match body.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(ApiError::invalid("stream", "stream is true or false")),
    };
    Ok(Call {
        model: model.into(),
        messages,
        max_tokens,
        stream,
    })
}

fn read_messages(messages: Option<&Value>) -> Result<Vec<Message>, ApiError> {
    let bad = |message: &str| ApiError::invalid("messages", message);
    let Some(Value::Array(messages)) = messages else {
        return Err(bad("messages is a list of chat messages"));
    };
    if messages.is_empty() {
        return Err(bad("messages is empty"));
    }
    messages
        .iter()
        .map(|message| {
            let Value::Object(message) = message else {
                return Err(bad("a message is a JSON object"));
            };
            if let Some(field) = message
                .keys()
                .find(|k| !matches!(k.as_str(), "role" | "content" | "name"))
            {
                return Err(bad(&format!("messages can't carry {field} over Logos yet")));
            }
            let role = match message.get("role").and_then(Value::as_str) {
                Some("system" | "developer") => Role::System,
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                _ => {
                    return Err(bad(
                        "a message's role is system, developer, user or assistant",
                    ));
                }
            };
            let content = match message.get("content") {
                Some(Value::String(text)) => text.clone(),
                Some(Value::Array(parts)) => parts
                    .iter()
                    .map(|part| {
                        match (part.get("type").and_then(Value::as_str), part.get("text")) {
                            (Some("text"), Some(Value::String(text))) => Ok(text.as_str()),
                            _ => Err(bad("only text can go over Logos for now")),
                        }
                    })
                    .collect::<Result<String, _>>()?,
                _ => return Err(bad("a message's content is text")),
            };
            Ok(Message { role, content })
        })
        .collect()
}

/// The output cap the client asked for, under either of its names.
fn read_cap(body: &Map<String, Value>) -> Result<Option<u32>, ApiError> {
    let mut cap = None;
    for field in ["max_tokens", "max_completion_tokens"] {
        let asked = match body.get(field) {
            None | Some(Value::Null) => continue,
            Some(value) => value
                .as_u64()
                .filter(|n| (1..=u32::MAX as u64).contains(n))
                .ok_or_else(|| {
                    ApiError::invalid(field, format!("{field} is a whole number above 0"))
                })? as u32,
        };
        if cap.is_some_and(|c| c != asked) {
            return Err(ApiError::invalid(
                field,
                "max_tokens and max_completion_tokens disagree",
            ));
        }
        cap = Some(asked);
    }
    Ok(cap)
}

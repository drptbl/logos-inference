//! What a job asks the model, and what that costs.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An offer's rates, in lepta per token, and its output cap.
pub struct Pricing {
    pub input_lepta_per_token: u64,
    pub output_lepta_per_token: u64,
    /// The most output tokens a job gets, and what it gets when it doesn't ask.
    pub max_tokens: u32,
}

/// Why a request is turned away. The text goes to the buyer.
#[derive(Debug, PartialEq)]
pub struct Refusal(pub String);

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A chat request for the model, with the output cap the buyer pays for.
#[derive(Debug, PartialEq)]
pub struct Prompt {
    pub messages: Vec<Message>,
    pub max_tokens: u32,
}

/// Tokens counted for each message's role and framing.
const PER_MESSAGE: u64 = 4;

impl Prompt {
    /// The input's tokens, estimated as a UTF-8 byte count over 4, rounded up, per message.
    pub fn input_tokens(&self) -> u64 {
        self.messages
            .iter()
            .map(|m| (m.content.len() as u64).div_ceil(4) + PER_MESSAGE)
            .sum()
    }

    /// The input's estimated tokens at the input rate, plus the whole output cap at the
    /// output rate, since escrow locks one price before the output exists.
    pub fn price(&self, pricing: &Pricing) -> u128 {
        self.input_tokens() as u128 * pricing.input_lepta_per_token as u128
            + self.max_tokens as u128 * pricing.output_lepta_per_token as u128
    }
}

/// The JSON form of a request, OpenAI-style. Anything it doesn't name is refused, so a
/// buyer never pays for a parameter that would be ignored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: Role,
    content: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}

/// Reads a job's input, as logos-agents passes it, into a prompt. Text that is a JSON
/// object with `messages` is a chat request; any other text is one user message. `system`
/// is the offer's own system prompt, which goes first.
pub fn parse(input: &Value, pricing: &Pricing, system: Option<&str>) -> Result<Prompt, Refusal> {
    let Some(text) = input.get("text").and_then(Value::as_str) else {
        let what = if input.get("cid").is_some() {
            "send the prompt as text, not a CID"
        } else {
            "send the prompt as text"
        };
        return Err(Refusal(what.into()));
    };
    let mut messages: Vec<Message> = system
        .map(|s| Message {
            role: "system".into(),
            content: s.into(),
        })
        .into_iter()
        .collect();
    let mut max_tokens = pricing.max_tokens;
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(request)) if request.contains_key("messages") => {
            let request: ChatRequest = serde_json::from_value(Value::Object(request))
                .map_err(|e| Refusal(format!("not a chat request: {e}")))?;
            if request.messages.is_empty() {
                return Err(Refusal("not a chat request: no messages".into()));
            }
            if let Some(asked) = request.max_tokens {
                if asked == 0 {
                    return Err(Refusal("max_tokens is at least 1".into()));
                }
                if asked > pricing.max_tokens {
                    return Err(Refusal(format!(
                        "max_tokens is at most {}",
                        pricing.max_tokens
                    )));
                }
                max_tokens = asked;
            }
            messages.extend(request.messages.into_iter().map(|m| {
                Message {
                    role: match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    }
                    .into(),
                    content: m.content,
                }
            }));
        }
        _ => messages.push(Message {
            role: "user".into(),
            content: text.into(),
        }),
    }
    Ok(Prompt {
        messages,
        max_tokens,
    })
}

use clap::Parser;
use logos_inference::job::Pricing;
use logos_inference::server::{Backend, Config, Server};
use std::process::ExitCode;
use std::time::Duration;

/// Serves a model to logos-agents' `provide`: `/quote` prices a request by its estimated
/// tokens, `/run` answers a paid job with the model's output. The backend's API key, if it
/// needs one, comes from LOGOS_INFERENCE_API_KEY.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// The OpenAI-compatible API's base URL, e.g. http://127.0.0.1:11434/v1 for Ollama.
    #[arg(long)]
    backend: String,
    /// The model the backend runs, as it names it.
    #[arg(long)]
    model: String,
    /// Lepta per estimated input token.
    #[arg(long)]
    input_lepta_per_token: u64,
    /// Lepta per output token, charged for the whole output cap.
    #[arg(long)]
    output_lepta_per_token: u64,
    /// The most output tokens a job gets, and what it gets when it doesn't ask.
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..))]
    max_tokens: u32,
    /// A system prompt put before every job's messages, and paid for as input.
    #[arg(long)]
    system: Option<String>,
    /// Where to listen. logos-agents only posts to a handler on this machine unless the
    /// offer sets remote_handler.
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,
    /// The longest a job waits on the model, in seconds.
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u64).range(1..))]
    timeout_s: u64,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if args.input_lepta_per_token == 0 && args.output_lepta_per_token == 0 {
        eprintln!("logos-inference: a job has to cost something: set a rate above 0");
        return ExitCode::FAILURE;
    }
    let config = Config {
        pricing: Pricing {
            input_lepta_per_token: args.input_lepta_per_token,
            output_lepta_per_token: args.output_lepta_per_token,
            max_tokens: args.max_tokens,
        },
        backend: Backend {
            url: args.backend,
            model: args.model,
            api_key: std::env::var("LOGOS_INFERENCE_API_KEY")
                .ok()
                .filter(|k| !k.is_empty()),
        },
        system: args.system,
        timeout: Duration::from_secs(args.timeout_s),
    };
    let server = match Server::bind(&args.listen, config) {
        Ok(server) => server,
        Err(e) => {
            eprintln!("logos-inference: can't listen on {}: {e}", args.listen);
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "logos-inference: serving /quote and /run on http://{}",
        server.addr()
    );
    server.run();
    ExitCode::SUCCESS
}

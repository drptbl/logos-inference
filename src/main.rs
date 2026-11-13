use clap::{Parser, Subcommand};
use logos_inference::agent::Agent;
use logos_inference::gateway::Gateway;
use logos_inference::gateway_server::{GatewayConfig, GatewayServer};
use logos_inference::job::Pricing;
use logos_inference::mcp_agent::McpAgent;
use logos_inference::server::{Backend, Config, Server, TokenField};
use std::process::ExitCode;
use std::time::Duration;

/// Decentralised AI inference over Logos, on logos-agents.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Serve(ServeArgs),
    Gateway(GatewayArgs),
}

/// Serve OpenAI's chat completions to local clients, each bought over Logos, as a buyer.
///
/// Each completion for a --model is bought as a job of `inference/<model>/v1`: from the first
/// provider that quotes within --max-price-lepta, paid into escrow from --pay-from. A key for
/// callers, needed to listen beyond this machine, comes from LOGOS_INFERENCE_GATEWAY_KEY.
#[derive(clap::Args)]
struct GatewayArgs {
    /// A model to serve, as providers name it in their service id. Repeat for more.
    #[arg(long = "model", required = true)]
    models: Vec<String>,
    /// The most one completion may pay, in lepta.
    #[arg(long, value_parser = parse_price)]
    max_price_lepta: u128,
    /// The wallet account every payment comes from, Public/<base58>.
    #[arg(long)]
    pay_from: String,
    /// The logos-agents binary to start, over stdio. It takes this process's environment, so
    /// LOGOS_MCP_* choose its wallet and network.
    #[arg(long, default_value = "logos-agents", conflicts_with = "agents_url")]
    agents_bin: String,
    /// A running logos-agents MCP endpoint to use instead, e.g. Basecamp's
    /// http://127.0.0.1:8766/mcp.
    #[arg(long)]
    agents_url: Option<String>,
    /// logos-agents' files root, for prompts too big for a chat message and answers that come
    /// through Logos Storage. Default: LOGOS_MCP_FILES_ROOT. Started by the gateway,
    /// logos-agents gets it as its own.
    #[arg(long)]
    files_root: Option<std::path::PathBuf>,
    /// Where to listen.
    #[arg(long, default_value = "127.0.0.1:8090")]
    listen: String,
    /// The longest one completion may take, in seconds.
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u64).range(1..))]
    timeout_s: u64,
    /// The most completions bought at once. More are answered 429 until one finishes.
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u64).range(1..=1024))]
    max_calls: u64,
    /// How long a provider has to quote before the next one is asked, in seconds. logos-agents
    /// asks a provider again after a minute without an answer, so the default leaves room.
    #[arg(long, default_value_t = 90, value_parser = clap::value_parser!(u64).range(1..))]
    quote_wait_s: u64,
}

fn parse_price(text: &str) -> Result<u128, String> {
    match text.parse::<u128>() {
        Ok(0) | Err(_) => Err("a whole number of lepta above 0".into()),
        Ok(price) => Ok(price),
    }
}

/// Serve a model to logos-agents' `provide`, as a provider.
///
/// `/quote` prices a request by its estimated tokens, `/run` answers a paid job with the
/// model's output. The backend's API key, if it needs one, comes from
/// LOGOS_INFERENCE_API_KEY.
#[derive(clap::Args)]
struct ServeArgs {
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
    /// The most output tokens a job gets, and what it gets when it doesn't ask. Up to 262144,
    /// so an answer stays well under the 8 MiB logos-agents delivers.
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..=262_144))]
    max_tokens: u32,
    /// The largest prompt taken, in estimated tokens. With it, a prompt can also come as a
    /// Logos Storage CID, quoted as if it were this big, since the quote sees only the CID.
    /// Up to 2.2M: logos-agents passes at most 8 MiB.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=2_200_000))]
    max_input_tokens: Option<u64>,
    /// The request field that caps the output: max_completion_tokens for hosted models that
    /// refuse max_tokens.
    #[arg(long, value_enum, default_value_t = TokenField::MaxTokens)]
    max_tokens_field: TokenField,
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
    match Cli::parse().command {
        Command::Serve(args) => serve(args),
        Command::Gateway(args) => gateway(args),
    }
}

fn gateway(args: GatewayArgs) -> ExitCode {
    let files_root = args
        .files_root
        .or_else(|| std::env::var_os("LOGOS_MCP_FILES_ROOT").map(Into::into));
    let agent = match &args.agents_url {
        Some(url) => McpAgent::connect(url, files_root),
        None => McpAgent::spawn(&args.agents_bin, files_root),
    };
    let agent = match agent {
        Ok(agent) => agent,
        Err(e) => {
            eprintln!("logos-inference: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = agent.check_payer(&args.pay_from) {
        eprintln!("logos-inference: --pay-from: {e}");
        return ExitCode::FAILURE;
    }
    let config = GatewayConfig {
        models: args.models,
        key: std::env::var("LOGOS_INFERENCE_GATEWAY_KEY")
            .ok()
            .filter(|k| !k.is_empty()),
        timeout: Duration::from_secs(args.timeout_s),
        keep_alive: Duration::from_secs(10),
        max_calls: args.max_calls as usize,
        send_timeout: Duration::from_secs(30),
        read_timeout: Duration::from_secs(30),
    };
    let gateway = Gateway {
        agent,
        pay_from: args.pay_from,
        max_price_lepta: args.max_price_lepta,
        quote_wait: Duration::from_secs(args.quote_wait_s),
        poll: Duration::from_secs(1),
    };
    run_gateway(&args.listen, gateway, config)
}

fn run_gateway<A: Agent + 'static>(
    listen: &str,
    gateway: Gateway<A>,
    config: GatewayConfig,
) -> ExitCode {
    let server = match GatewayServer::bind(listen, gateway, config) {
        Ok(server) => server,
        Err(e) => {
            eprintln!("logos-inference: can't listen on {listen}: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "logos-inference: serving /v1/chat/completions on http://{}",
        server.addr()
    );
    server.run();
    ExitCode::SUCCESS
}

fn serve(args: ServeArgs) -> ExitCode {
    if args.input_lepta_per_token == 0 && args.output_lepta_per_token == 0 {
        eprintln!("logos-inference: a job has to cost something: set a rate above 0");
        return ExitCode::FAILURE;
    }
    let config = Config {
        pricing: Pricing {
            input_lepta_per_token: args.input_lepta_per_token,
            output_lepta_per_token: args.output_lepta_per_token,
            max_tokens: args.max_tokens,
            max_input_tokens: args.max_input_tokens,
        },
        backend: Backend {
            url: args.backend,
            model: args.model,
            api_key: std::env::var("LOGOS_INFERENCE_API_KEY")
                .ok()
                .filter(|k| !k.is_empty()),
            token_field: args.max_tokens_field,
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

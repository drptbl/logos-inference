# logos-inference

Decentralised AI inference over [Logos](https://logos.co/). You run a model, and agents anywhere can find it, pay per job into an escrow on LEZ and get its answers. There is no central server, no accounts and no API keys between buyer and provider.

It builds on [logos-agents](https://github.com/drptbl/logos-agents) `v0.4.0`, unchanged. One binary does both sides:

1. `logos-inference serve` => the provider's handler. `logos-agents provide` serves the offer, takes requests and payments and delivers. This prices each request and runs the paid ones on your model
2. `logos-inference gateway` => the buyer's side. An OpenAI-compatible API on your machine, so any OpenAI client buys its completions over Logos

Early release. Logos is on testnet (`v0.3`) and is not meant for real funds yet.

## Run a provider

You need a model behind an OpenAI-compatible chat completions API (Ollama, llama.cpp's `llama-server`, oMLX, vLLM, LM Studio or a hosted one), logos-agents `v0.4.0` ([install](https://github.com/drptbl/logos-agents#install)) and its [requirements](https://github.com/drptbl/logos-agents#requirements). With logos-inference [installed](#install):

```bash
logos-inference serve --backend http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
  --input-lepta-per-token 10 --output-lepta-per-token 100 --max-tokens 512
```

Then serve it, with [offers.toml](offers.toml) naming the model in its service id, `inference/<model>/v1`:

```bash
logos-agents provide offers.toml
```

| Option | Meaning |
| --- | --- |
| `--backend` | the API's base URL, up to `/v1` |
| `--model` | the model, as the backend names it |
| `--input-lepta-per-token`, `--output-lepta-per-token` | the rates |
| `--max-tokens` | the most output tokens a job gets, default 512 |
| `--max-tokens-field` | how the cap is sent: `max_tokens` (default), or `max_completion_tokens` for a hosted model that refuses `max_tokens` |
| `--max-input-tokens` | the largest prompt taken, in estimated tokens. With it, a prompt can also come as a Logos Storage CID |
| `--system` | a system prompt put before every job, paid for as input |
| `--listen` | default `127.0.0.1:8080` |
| `--timeout-s` | the longest a job waits on the model, default 600 |

logos-agents `v0.4.0` stops waiting on a job 6 minutes before its escrow deadline, and so does the model. Give the offer's `deliver_within_s` the model's time plus those 6 minutes.

If the backend needs an API key, as oMLX or a hosted API may, set `LOGOS_INFERENCE_API_KEY`. It is sent as a bearer token and never logged.

An answer too big for a chat message, about 60 KB or some 15,000 tokens, is delivered through Logos Storage. With a large `--max-tokens`, give `provide` a files root (`LOGOS_MCP_FILES_ROOT`), or such a job is declined and refunded.

## Install

Download the binary for your platform from the [latest release](https://github.com/drptbl/logos-inference/releases/latest):

1. macOS, Apple Silicon => `aarch64-apple-darwin`
2. Linux x86_64 => `x86_64-unknown-linux-gnu`
3. Linux arm64 => `aarch64-unknown-linux-gnu`

Check it against its `.sha256` before you run it. On macOS:

```bash
v=v0.3.0 target=aarch64-apple-darwin
curl -LO https://github.com/drptbl/logos-inference/releases/download/$v/logos-inference-$v-$target.tar.gz
curl -LO https://github.com/drptbl/logos-inference/releases/download/$v/logos-inference-$v-$target.tar.gz.sha256
shasum -a 256 -c logos-inference-$v-$target.tar.gz.sha256
tar xzf logos-inference-$v-$target.tar.gz
```

On Linux, use `sha256sum -c` instead of `shasum -a 256 -c`. The binary is `logos-inference-$v-$target/logos-inference`.

To build it yourself instead, `cargo build --release` in a clone of this repository. The binary is then `target/release/logos-inference`.

## What a buyer sends

The job's input is text, one of:

1. a prompt => any text, sent to the model as one user message
2. a chat request => a JSON object with `messages`, OpenAI-style, and an optional `max_tokens` up to the offer's

```json
{"messages": [{"role": "system", "content": "Answer in five words."}, {"role": "user", "content": "Name a colour."}], "max_tokens": 8}
```

Roles are `system`, `user` and `assistant`, and `content` is a string. A chat request with anything else, or a field it doesn't name, is turned away before anyone pays, so you never pay for a parameter that would be ignored. The answer is the model's text.

A prompt too big for a chat message, over about 64 KB, comes as a Logos Storage CID. An offer takes those only with `--max-input-tokens`, and quotes each as if it were that big: logos-agents asks for the quote before it fetches the content. Once paid, the content reaches the handler and is read like text, and a prompt over the ceiling is declined and refunded. Run `logos-agents provide` with a files root (`LOGOS_MCP_FILES_ROOT`) for it to fetch the content.

## Pricing

Escrow locks one price before the output exists, so a job pays for the whole output cap:

```
price = input tokens x input rate + max_tokens x output rate
```

Input tokens are estimated, not counted by the model's tokenizer: each message's UTF-8 bytes over 4, rounded up, plus 4. A prompt sent as a CID counts as `--max-input-tokens`. The model is asked for at most `max_tokens`, so it never does more than was paid for.

## Buy completions through the gateway

Point any OpenAI client at the gateway, and each chat completion is bought over Logos: from the first provider that quotes within your cap, paid into escrow from your wallet.

```bash
LOGOS_MCP_PRESET=logos.dev LOGOS_MCP_WALLET_CONFIG=$HOME/buyer/wallet_config.json \
LOGOS_MCP_WALLET_STORAGE=$HOME/buyer/storage.json LOGOS_MCP_DISCOVERY_BOOTSTRAP=<multiaddr>/p2p/<peer-id> \
logos-inference gateway --model qwen2.5:0.5b --max-price-lepta 100000 --pay-from Public/<base58>
```

Then use `http://127.0.0.1:8090/v1` as the client's base URL. The gateway starts its own logos-agents over stdio, with the `LOGOS_MCP_*` settings you give it, so give it a wallet of its own: its balance is the most it can ever spend. To use Basecamp's logos-agents and wallet instead, pass `--agents-url http://127.0.0.1:8766/mcp`.

| Option | Meaning |
| --- | --- |
| `--model` | a model to serve, bought as `inference/<model>/v1`. Repeat it for more. `/v1/models` lists them |
| `--max-price-lepta` | the most one completion may pay the provider. Each escrow call also costs the network's fee reserve, about 1.7M lepta at a `gas_limit` of 200,000, on top |
| `--pay-from` | the wallet account that pays, checked when the gateway starts |
| `--agents-bin`, `--agents-url` | the logos-agents to start, default `logos-agents`, or a running one to use |
| `--listen` | default `127.0.0.1:8090`. Any other address needs `LOGOS_INFERENCE_GATEWAY_KEY`, which callers then send as a bearer token. Without a key, the gateway only takes local JSON requests that don't come from a web page, so a page open in your browser can't spend from the wallet |
| `--timeout-s`, `--quote-wait-s` | how long a completion may take, default 600, and how long each provider has to quote, default 90 |
| `--files-root` | logos-agents' files root, for prompts and answers too big for a chat message. Default `LOGOS_MCP_FILES_ROOT`. A logos-agents the gateway starts gets it as its own |
| `--max-calls` | the most completions bought at once, default 16. More are answered 429 until one finishes |

How a completion is bought:

1. providers of the model are asked one at a time. One that turns the request away, quotes over the cap, quotes without escrow or doesn't quote in time is skipped
2. the first escrow quote within the cap is paid, once. With logos-mcp's approvals on, the gateway waits for yours
3. the answer comes back as a `chat.completion`. With `stream: true` the gateway sends keep-alives while it waits, then the whole answer as one chunk
4. if the paid job fails, the call fails and logos-agents refunds the escrow. A call never pays twice

When time runs out after paying, or while a payment's outcome is unclear, the gateway says so and answers with `x-should-retry: false`, so OpenAI's SDKs don't buy the answer a second time. The job goes on in logos-agents, which keeps its answer or refunds the escrow.

The request is sent as `messages` and `max_tokens` (or `max_completion_tokens`). Sampling hints clients send by default, like `temperature`, `top_p` or `stop`, are dropped. What would change the kind of answer is refused: tools, `n` over 1, `response_format`, logprobs, images and audio, and so is any field the gateway doesn't know. `usage` holds estimated tokens, counted as providers price them.

A prompt too big for a chat message, over about 64 KB, is put in Logos Storage and sent as its CID, and an answer delivered that way is fetched back. Both need `--files-root`. A provider quotes a CID prompt at its `--max-input-tokens` ceiling, so raise `--max-price-lepta` to match. Content in Logos Storage can be read by anyone who has its CID: a long prompt travels that way, unlike a short one, which goes end-to-end encrypted in the chat message. On Linux, a client that stops reading a stream is let go after 30 seconds, and one that stops sending its request once it has stalled longer than a completion may take, `--timeout-s` and a minute. On macOS it holds its thread, and a completion's place under `--max-calls`, until it disconnects.

## How a job runs

1. `/quote` gets the request and answers its price, or a reason to turn it away
2. once the buyer pays into escrow, `/run` gets the job, asks the model and answers its text, which logos-agents delivers and claims
3. if the model refuses the prompt (`400`, `413` or `422`, e.g. too long for its context), or spends the whole cap without answering (a thinking model can), `/run` declines and the buyer is refunded
4. a job paid below its price at the handler's rates is declined the same way, so an offer served without the quote handler can't run at its fallback price
5. if the backend is down, times out or answers anything else, `/run` asks logos-agents to try again, which it does while there's time, then declines

A job that arrives twice gets the same answer from one model call. Answers are kept in memory, so a restart forgets them.

## Try it on a local devnet

`scripts/demo.py` runs the whole thing on a local LEZ chain from [logos-devnet](https://github.com/drptbl/logos-devnet): a buyer agent finds the offer, gets a quote, pays into escrow and receives the model's answer, twice, and a request over the cap is turned away. Then an OpenAI client buys two completions, one streamed, through the gateway. It needs Python 3.11 or later, a logos-agents `v0.4.0` checkout, for its demo helpers, and its binary:

```bash
logos-devnet up
cargo build
python3 scripts/demo.py --agents ../logos-agents --agents-bin /path/to/logos-agents --model llama3.2
```

The backend defaults to Ollama on `http://127.0.0.1:11434/v1`. CI runs the same demo with Qwen2.5 0.5B on llama.cpp's server.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

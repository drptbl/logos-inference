# logos-inference

Decentralised AI inference over [Logos](https://logos.co/). You run a model, and agents anywhere can find it, pay per job into an escrow on LEZ and get its answers. There is no central server, no accounts and no API keys between buyer and provider.

It builds on [logos-agents](https://github.com/drptbl/logos-agents) `v0.4.0`, unchanged. `logos-agents provide` serves the offer, takes requests and payments and delivers. logos-inference is the handler it calls: it prices each request and runs the paid ones on your model.

Early release. Logos is on testnet (`v0.3`) and is not meant for real funds yet.

## Run a provider

You need a model behind an OpenAI-compatible chat completions API (Ollama, llama.cpp's `llama-server`, oMLX, vLLM, LM Studio or a hosted one), logos-agents `v0.4.0` ([install](https://github.com/drptbl/logos-agents#install)) and its [requirements](https://github.com/drptbl/logos-agents#requirements). With logos-inference [installed](#install):

```bash
logos-inference --backend http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
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
v=v0.1.0 target=aarch64-apple-darwin
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

Roles are `system`, `user` and `assistant`, and `content` is a string. A chat request with anything else, or a field it doesn't name, is turned away before anyone pays, so you never pay for a parameter that would be ignored. A CID input is turned away too: the quote is made before its content is fetched. The answer is the model's text.

## Pricing

Escrow locks one price before the output exists, so a job pays for the whole output cap:

```
price = input tokens x input rate + max_tokens x output rate
```

Input tokens are estimated, not counted by the model's tokenizer: each message's UTF-8 bytes over 4, rounded up, plus 4. The model is asked for at most `max_tokens`, so it never does more than was paid for.

## How a job runs

1. `/quote` gets the request and answers its price, or a reason to turn it away
2. once the buyer pays into escrow, `/run` gets the job, asks the model and answers its text, which logos-agents delivers and claims
3. if the model refuses the prompt (`400`, `413` or `422`, e.g. too long for its context), or spends the whole cap without answering (a thinking model can), `/run` declines and the buyer is refunded
4. a job paid below its price at the handler's rates is declined the same way, so an offer served without the quote handler can't run at its fallback price
5. if the backend is down, times out or answers anything else, `/run` asks logos-agents to try again, which it does while there's time, then declines

A job that arrives twice gets the same answer from one model call. Answers are kept in memory, so a restart forgets them.

## Try it on a local devnet

`scripts/demo.py` runs the whole thing on a local LEZ chain from [logos-devnet](https://github.com/drptbl/logos-devnet): a buyer agent finds the offer, gets a quote, pays into escrow and receives the model's answer, twice, and a request over the cap is turned away. It needs Python 3.11 or later, a logos-agents `v0.4.0` checkout, for its demo helpers, and its binary:

```bash
logos-devnet up
cargo build
python3 scripts/demo.py --agents ../logos-agents --agents-bin /path/to/logos-agents --model llama3.2
```

The backend defaults to Ollama on `http://127.0.0.1:11434/v1`. CI runs the same demo with Qwen2.5 0.5B on llama.cpp's server.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

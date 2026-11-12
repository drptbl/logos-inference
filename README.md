# logos-inference

Decentralised AI inference over [Logos](https://logos.co/). You run a model, and agents anywhere can find it, pay per job into an escrow on LEZ and get its answers. There is no central server, no accounts and no API keys between buyer and provider.

It builds on [logos-agents](https://github.com/drptbl/logos-agents) `v0.4.0`, unchanged. `logos-agents provide` serves the offer, takes requests and payments and delivers. logos-inference is the handler it calls: it prices each request and runs the paid ones on your model.

Early release. Logos is on testnet (`v0.3`) and is not meant for real funds yet.

## Run a provider

You need a model behind an OpenAI-compatible chat completions API (Ollama, llama.cpp's `llama-server`, vLLM, LM Studio or a hosted one) and the [logos-agents requirements](https://github.com/drptbl/logos-agents#requirements). The output cap is sent as `max_tokens`, so a hosted model that only takes `max_completion_tokens` declines every job for now.

```bash
cargo build --release
./target/release/logos-inference --backend http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
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
| `--system` | a system prompt put before every job, paid for as input |
| `--listen` | default `127.0.0.1:8080` |
| `--timeout-s` | the longest a job waits on the model, default 600 |

If the backend needs an API key, set `LOGOS_INFERENCE_API_KEY`. It is sent as a bearer token and never logged.

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
3. if the model refuses the prompt (`400`, `413` or `422`, e.g. too long for its context), `/run` declines and the buyer is refunded
4. if the backend is down, times out or answers anything else, `/run` asks logos-agents to try again, which it does while there's time, then declines

A job that arrives twice gets the same answer from one model call. Answers are kept in memory, so a restart forgets them.

## Try it on a local devnet

`scripts/demo.py` runs the whole thing on a local LEZ chain from [logos-devnet](https://github.com/drptbl/logos-devnet): a buyer agent finds the offer, gets a quote, pays into escrow and receives the model's answer, twice, and a request over the cap is turned away. It needs a logos-agents `v0.4.0` checkout, for its demo helpers, and its binary:

```bash
logos-devnet up
cargo build
python3 scripts/demo.py --agents ../logos-agents --agents-bin /path/to/logos-agents --model llama3.2
```

The backend defaults to Ollama on `http://127.0.0.1:11434/v1`. CI runs the same demo with Qwen2.5 0.5B on llama.cpp's server.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

#!/usr/bin/env python3
"""A model served over Logos, on the local devnet logos-devnet.toml sets up.

logos-inference runs next to an OpenAI-compatible backend, and `logos-agents provide`
serves it as one offer, `inference/<model>/v1`, with no agent on the provider's side. A
buyer agent finds the offer and asks for three jobs:

1. a plain prompt: quoted by its estimated tokens, paid into escrow, answered by the
   model, then claimed by the provider's server
2. a chat request with its own max_tokens, quoted lower, paid, answered and claimed
3. a chat request over the offer's output cap, turned away before anyone pays

Run `logos-devnet up` in this repo first, with logos-devnet and logosctl on PATH, a
logos-agents v0.4.0 checkout (for its demo helpers) and its binary, and the model
running on the backend. Exits non-zero if a job does not go as described.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The offer's rates: lepta per estimated input token, per output token, and its output cap.
INPUT_RATE = 10
OUTPUT_RATE = 100
MAX_TOKENS = 32

PROMPT = "In one sentence, what is an escrow?"
CHAT = {"messages": [{"role": "system", "content": "Answer in at most five words."},
                     {"role": "user", "content": "Name a colour."}],
        "max_tokens": 8}


def price(contents, max_tokens):
    """The price README.md gives: each message's UTF-8 bytes over 4, rounded up, plus 4,
    at the input rate, and the output cap at the output rate."""
    tokens = sum(math.ceil(len(c.encode()) / 4) + 4 for c in contents)
    return str(tokens * INPUT_RATE + max_tokens * OUTPUT_RATE)


def serve_model(args, logs):
    """Starts logos-inference on a free port and returns its URL."""
    log = os.path.join(logs, "inference.log")
    server = subprocess.Popen(
        [args.bin, "--backend", args.backend, "--model", args.model,
         "--input-lepta-per-token", str(INPUT_RATE), "--output-lepta-per-token", str(OUTPUT_RATE),
         "--max-tokens", str(MAX_TOKENS), "--listen", "127.0.0.1:0"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=open(log, "w"))
    deadline = time.time() + 30
    while time.time() < deadline:
        with open(log) as f:
            if found := re.search(r"on (http://127\.0\.0\.1:\d+)", f.read()):
                return server, found.group(1)
        if server.poll() is not None:
            sys.exit(f"logos-inference stopped: see {log}")
        time.sleep(0.2)
    sys.exit("logos-inference didn't start listening")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", default=os.environ.get("LOGOS_AGENTS_DIR", os.path.join(REPO, "..", "logos-agents")),
                    help="a logos-agents v0.4.0 checkout, for scripts/demo_escrow_job.py (default: LOGOS_AGENTS_DIR or ../logos-agents)")
    ap.add_argument("--agents-bin", default="logos-agents", help="the logos-agents v0.4.0 binary")
    ap.add_argument("--bin", default=os.path.join(REPO, "target", "debug", "logos-inference"))
    ap.add_argument("--backend", default="http://127.0.0.1:11434/v1", help="the OpenAI-compatible API, e.g. Ollama's")
    ap.add_argument("--model", required=True, help="the model the backend runs, e.g. qwen2.5:0.5b")
    ap.add_argument("--service", help="the offer's service id (default: inference/<model>/v1)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds to wait for each step")
    ap.add_argument("--poll", type=int, default=3, help="seconds between checks")
    args = ap.parse_args()
    service = args.service or f"inference/{args.model}/v1"

    sys.path.insert(0, os.path.join(os.path.abspath(args.agents), "scripts"))
    import demo_escrow_job as demo  # noqa: E402
    demo.POLL = args.poll

    # Sessions, wallets, the escrow id and the messaging hub, from the devnet.
    sides = argparse.Namespace(provider_session=None, provider_wallet=None, buyer_session=None,
                               buyer_wallet=None, buyer_from=None, bootstrap=None)
    demo.on_devnet(sides)

    logs = tempfile.mkdtemp(prefix="logos-inference-demo-")
    model, base = serve_model(args, logs)
    demo.log(f"[provider] logos-inference serving {args.model} from {args.backend} on {base}")
    offers = os.path.join(logs, "offers.toml")
    with open(offers, "w") as f:
        f.write(f'[[offer]]\nservice_id = {json.dumps(service)}\ntag = "inference"\n'
                f'price_lepta = "1"\ndeliver_within_s = 600\nhandler = "{base}/run"\n'
                f'quote_handler = "{base}/quote"\nmax_jobs = 2\n')

    wallet = sides.provider_wallet
    provider_env = dict(
        os.environ,
        LOGOSCTL_CONFIG_DIR=sides.provider_session,
        LOGOS_MCP_PRESET="logos.dev",
        LOGOS_MCP_WALLET_CONFIG=os.path.join(wallet, "wallet_config.json"),
        LOGOS_MCP_WALLET_STORAGE=os.path.join(wallet, "storage.json"),
        LOGOS_MCP_DISCOVERY_BOOTSTRAP=sides.bootstrap,
        LOGOS_AGENTS_FOLLOW_THROUGH_SECS=str(args.poll),
    )
    provider = subprocess.Popen([args.agents_bin, "provide", offers], env=provider_env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=open(os.path.join(logs, "provider.log"), "w"))
    demo.log(f"[provider] logos-agents provide {service}")
    buyer = demo.start("buyer", args.agents_bin, sides.buyer_session, sides.buyer_wallet, sides.bootstrap,
                       "logos.dev", logs)
    try:
        def find():
            _, body = buyer.tool("service_find", service_id=service)
            return next((p for p in body.get("providers", []) if not p["this_node"]), None)
        found = demo.wait_for(find, "the provider's offer", 300, 5)
        demo.log(f"[buyer] found provider {found['peer_id']}")

        def request(text):
            job = demo.call(buyer, "buyer", "service_request", chat_address=found["chat_address"],
                            service_id=service, input_text=text)["job"]
            demo.wait_for(lambda: demo.state(buyer, job) in ("quoted", "rejected"), "the provider's answer",
                          args.timeout)
            return job

        def paid_job(text, expected):
            job = request(text)
            quote = (demo.job_in(buyer, job) or {}).get("quote") or {}
            if quote.get("price_lepta") != expected:
                sys.exit(f"quoted {quote.get('price_lepta')} lepta, not {expected}: {demo.job_in(buyer, job)}")
            demo.log(f"[provider] quoted {expected} lepta by estimated tokens")
            demo.call(buyer, "buyer", "service_pay", job=job, **{"from": sides.buyer_from})
            output = demo.wait_for(
                lambda: (demo.job_in(buyer, job) or {}).get("output") if demo.state(buyer, job) == "delivered" else None,
                "the model's answer", args.timeout)
            if not (output.get("text") or "").strip():
                sys.exit(f"the answer came back as {output}, not text")
            demo.log(f"[buyer] the model answered: {output['text']!r}")
            demo.wait_for(lambda: demo.escrow(buyer, job) == "claimed", "the provider's claim", args.timeout)
            demo.log("[provider] claimed the price")

        paid_job(PROMPT, price([PROMPT], MAX_TOKENS))
        paid_job(json.dumps(CHAT), price([m["content"] for m in CHAT["messages"]], CHAT["max_tokens"]))

        over = request(json.dumps({**CHAT, "max_tokens": MAX_TOKENS + 1}))
        turned = demo.job_in(buyer, over) or {}
        if turned.get("state") != "rejected" or f"at most {MAX_TOKENS}" not in (turned.get("reason") or ""):
            sys.exit(f"a request over the cap came back as {turned}")
        demo.log(f"[provider] turned away a request over the cap: {turned['reason']!r}")
    finally:
        provider.terminate()
        provider.wait(timeout=60)
        buyer.close()
        model.terminate()
    print(f"DEMO: PASS (logs in {logs})")


if __name__ == "__main__":
    main()

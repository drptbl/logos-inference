#!/usr/bin/env python3
"""A model served over Logos, on the local devnet logos-devnet.toml sets up.

logos-inference runs next to an OpenAI-compatible backend, and `logos-agents provide`
serves it as one offer, `inference/<model>/v1`, with no agent on the provider's side. A
buyer agent finds the offer and asks for three jobs:

1. a plain prompt: quoted by its estimated tokens, paid into escrow, answered by the
   model, then claimed by the provider's server
2. a chat request with its own max_tokens, quoted lower, paid, answered and claimed
3. a chat request over the offer's output cap, turned away before anyone pays

Then the buyer's wallet goes behind `logos-inference gateway`, and an OpenAI client buys
four completions through it: one streamed, one with a prompt too big for a chat message,
which goes through Logos Storage and is quoted at the offer's input ceiling, and one whose
answer is too big for a chat message, which comes back through Logos Storage. A small proxy
in front of the backend writes that answer, since a small model won't.

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
import threading
import tomllib
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The offer's rates: lepta per estimated input token, per output token, and its output cap.
INPUT_RATE = 10
OUTPUT_RATE = 100
MAX_TOKENS = 32
# The largest prompt the offer takes, in estimated tokens: a CID prompt is quoted at this.
MAX_INPUT_TOKENS = 24_000
# A prompt over a chat message's 64 KB, so the gateway sends it through Logos Storage.
BIG_PROMPT = "Answer with one word, the word this text repeats: " + "logos " * 11_000
# A prompt the proxy answers itself, with an answer over a chat message's 64 KB.
LONG_ASK = "Write a long answer."
LONG_ANSWER = "logos " * 12_000


def proxy_backend(backend):
    """An OpenAI-compatible backend in front of `backend`: it answers LONG_ASK with
    LONG_ANSWER itself and passes every other request on. Returns its base URL."""
    class Proxy(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if json.loads(body)["messages"][-1]["content"] == LONG_ASK:
                status = 200
                reply = json.dumps({"choices": [{"index": 0, "finish_reason": "stop",
                                                 "message": {"role": "assistant", "content": LONG_ANSWER}}]}).encode()
            else:
                ask = urllib.request.Request(backend.rstrip("/") + self.path[len("/v1"):], data=body,
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(ask, timeout=900) as r:
                        status, reply = r.status, r.read()
                except urllib.error.HTTPError as e:
                    status, reply = e.code, e.read()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/v1"

PROMPT = "In one sentence, what is an escrow?"
CHAT = {"messages": [{"role": "system", "content": "Answer in at most five words."},
                     {"role": "user", "content": "Name a colour."}],
        "max_tokens": 8}


def price(contents, max_tokens):
    """The price README.md gives: each message's UTF-8 bytes over 4, rounded up, plus 4,
    at the input rate, and the output cap at the output rate."""
    tokens = sum(math.ceil(len(c.encode()) / 4) + 4 for c in contents)
    return str(tokens * INPUT_RATE + max_tokens * OUTPUT_RATE)


def write_offers(source, path, service, base):
    """The repo's offer, as a provider would serve it, pointed at this run's server."""
    with open(source, "rb") as f:
        offer = tomllib.load(f)["offer"][0]
    offer.update(service_id=service, handler=f"{base}/run", quote_handler=f"{base}/quote")
    with open(path, "w") as f:
        f.write("[[offer]]\n" + "".join(f"{key} = {json.dumps(value, ensure_ascii=False)}\n" for key, value in offer.items()))


def serve_model(args, logs, backend):
    """Starts logos-inference serve on a free port and returns it with its URL."""
    return start_listening(
        [args.bin, "serve", "--backend", backend, "--model", args.model,
         "--input-lepta-per-token", str(INPUT_RATE), "--output-lepta-per-token", str(OUTPUT_RATE),
         "--max-tokens", str(MAX_TOKENS), "--max-input-tokens", str(MAX_INPUT_TOKENS),
         "--listen", "127.0.0.1:0"],
        os.path.join(logs, "inference.log"), os.environ, 30)


def start_listening(cmd, log, env, wait):
    """Starts a logos-inference command and returns it with the URL it says it listens on."""
    server = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=open(log, "w"))
    deadline = time.time() + wait
    while time.time() < deadline:
        with open(log) as f:
            if found := re.search(r"on (http://127\.0\.0\.1:\d+)", f.read()):
                return server, found.group(1)
        if server.poll() is not None:
            sys.exit(f"logos-inference stopped: see {log}")
        time.sleep(0.2)
    sys.exit(f"logos-inference didn't start listening: see {log}")


def openai(url, path, body=None, timeout=600):
    """An OpenAI-style call: the status and the body as text."""
    data = json.dumps(body).encode() if body is not None else None
    ask = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(ask, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", default=os.environ.get("LOGOS_AGENTS_DIR", os.path.join(REPO, "..", "logos-agents")),
                    help="a logos-agents v0.4.0 checkout, for scripts/demo_escrow_job.py (default: LOGOS_AGENTS_DIR or ../logos-agents)")
    ap.add_argument("--agents-bin", default="logos-agents", help="the logos-agents v0.4.0 binary")
    ap.add_argument("--bin", default=os.path.join(REPO, "target", "debug", "logos-inference"))
    ap.add_argument("--backend", default="http://127.0.0.1:11434/v1", help="the OpenAI-compatible API, e.g. Ollama's")
    ap.add_argument("--model", required=True, help="the model the backend runs, e.g. qwen2.5:0.5b")
    ap.add_argument("--offers", default=os.path.join(REPO, "offers.toml"),
                    help="the offer to serve, whose service id and handler URLs this run replaces")
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
    inference, base = serve_model(args, logs, proxy_backend(args.backend))
    demo.log(f"[provider] logos-inference serving {args.model} from {args.backend} on {base}")
    offers = os.path.join(logs, "offers.toml")
    write_offers(args.offers, offers, service, base)

    def side_env(session, wallet, files):
        """The environment a side's logos-agents runs with, on the devnet, storage in `files`."""
        return dict(
            os.environ,
            LOGOSCTL_CONFIG_DIR=session,
            LOGOS_MCP_PRESET="logos.dev",
            LOGOS_MCP_WALLET_CONFIG=os.path.join(wallet, "wallet_config.json"),
            LOGOS_MCP_WALLET_STORAGE=os.path.join(wallet, "storage.json"),
            LOGOS_MCP_DISCOVERY_BOOTSTRAP=sides.bootstrap,
            LOGOS_AGENTS_FOLLOW_THROUGH_SECS=str(args.poll),
            LOGOS_MCP_FILES_ROOT=files,
        )

    provider_files = tempfile.mkdtemp(prefix="logos-inference-provider-files-")
    buyer_files = tempfile.mkdtemp(prefix="logos-inference-buyer-files-")
    provider_env = side_env(sides.provider_session, sides.provider_wallet, provider_files)
    provider = buyer = gateway = None
    try:
        provider = subprocess.Popen([args.agents_bin, "provide", offers], env=provider_env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=open(os.path.join(logs, "provider.log"), "w"))
        demo.log(f"[provider] logos-agents provide {service}")
        buyer = demo.start("buyer", args.agents_bin, sides.buyer_session, sides.buyer_wallet, sides.bootstrap,
                           "logos.dev", logs)
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

        # One logos-agents per wallet: the buyer's goes behind the gateway now.
        buyer.close()
        buyer = None
        buyer_env = side_env(sides.buyer_session, sides.buyer_wallet, buyer_files)
        gateway, gateway_url = start_listening(
            [args.bin, "gateway", "--model", args.model, "--max-price-lepta", "1000000",
             "--pay-from", sides.buyer_from, "--agents-bin", args.agents_bin, "--listen", "127.0.0.1:0",
             "--timeout-s", str(args.timeout)],
            os.path.join(logs, "gateway.log"), buyer_env, 120)
        demo.log(f"[buyer] logos-inference gateway on {gateway_url}")
        status, body = openai(gateway_url, "/v1/models")
        if status != 200 or args.model not in [m["id"] for m in json.loads(body)["data"]]:
            sys.exit(f"/v1/models answered {status}: {body}")

        def completion(stream, prompt="Name a colour."):
            started = time.time()
            status, body = openai(gateway_url, "/v1/chat/completions", {
                "model": args.model, "stream": stream, "max_tokens": 8, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}, timeout=args.timeout + 30)
            if status != 200:
                sys.exit(f"the gateway answered {status}: {body}")
            if stream:
                events = [line[len("data: "):] for line in body.splitlines() if line.startswith("data: ")]
                if events[-1:] != ["[DONE]"]:
                    sys.exit(f"the stream didn't end with [DONE]: {body}")
                pieces = [json.loads(e) for e in events[:-1]]
                if failed := next((p for p in pieces if "error" in p), None):
                    sys.exit(f"the stream carried an error: {failed}")
                text = "".join(p["choices"][0]["delta"].get("content") or "" for p in pieces)
            else:
                text = json.loads(body)["choices"][0]["message"]["content"]
            if not text.strip():
                sys.exit(f"the gateway's answer was empty: {body}")
            how = "streamed" if stream else "answered"
            size = f" for a {len(prompt) // 1000} KB prompt" if len(prompt) > 1000 else ""
            shown = f"{len(text) // 1000} KB" if len(text) > 1000 else repr(text)
            demo.log(f"[buyer] an OpenAI client was {how} {shown}{size} in {time.time() - started:.0f}s")
            return text

        completion(stream=False)
        completion(stream=True)
        completion(stream=False, prompt=BIG_PROMPT)
        if completion(stream=False, prompt=LONG_ASK) != LONG_ANSWER:
            sys.exit("the answer that came through storage isn't the one written")
    finally:
        if gateway:
            gateway.terminate()
            gateway.wait(timeout=60)
        inference.terminate()
        if buyer:
            buyer.close()
        if provider:
            provider.terminate()
            provider.wait(timeout=60)
    print(f"DEMO: PASS (logs in {logs})")


if __name__ == "__main__":
    main()

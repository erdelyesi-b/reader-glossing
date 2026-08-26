#!/usr/bin/env python3
"""Run the gold-set tasks against one model and record what came back.

Speaks the OpenAI chat-completions shape, which is the one thing every runtime
in this comparison agrees on — LM Studio, llama.cpp's llama-server and
TurboFieldfareServer all expose it, so a single harness drives all of them and
no model gets a different prompt than the others.

    python3 run_model.py --name gemma4-e4b --base http://127.0.0.1:1234/v1 \
                         --model google/gemma-4-e4b

Records raw text alongside the parsed entries: how a model fails to produce
JSON is itself a result, and TurboFieldfareServer has no structured-output
mode to fall back on.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompt as prompt_mod  # noqa: E402

FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def balance(text):
    """Close braces the model left open, and drop a trailing partial entry.

    Worth separating from a real parse failure: a response that is correct
    except for a missing '}' is recoverable by the caller, while prose or a
    corrupted key is not. Without constrained decoding this is the difference
    between a usable run and an unusable one, so it gets its own bucket.
    """
    s = text[text.find("{"):]
    if not s:
        return ""
    depth = 0
    in_str = esc = False
    cut = 0
    for i, c in enumerate(s):
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
            elif c == "," and depth == 2:
                cut = i               # last clean entry boundary
    if in_str or depth < 0:
        s = s[:cut] if cut else s
        depth = 0
        for c in s:
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
    return s.rstrip().rstrip(",") + "}" * max(0, depth)


def extract_json(text):
    """Best-effort parse. Returns (entries, how) — how is the repair needed.

    Scored separately from content: a model that needs a fence stripped is
    usable behind two lines of code, one that emits prose is not, and lumping
    those together as 'JSON error' would hide the difference.
    """
    if not text:
        return None, "empty"
    for how, body in (("clean", text.strip()),
                      ("fenced", (FENCE.search(text) or [None, ""])[1]),
                      ("braces", text[text.find("{"):text.rfind("}") + 1]),
                      ("balanced", balance(text))):
        if not body:
            continue
        try:
            obj = json.loads(body)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("entries"), dict):
            return obj["entries"], how
        if isinstance(obj, dict):
            return obj, how + "+noshell"   # dropped the {"entries": ...} wrapper
    return None, "unparseable"


def call(base, model, system, user, key, temp, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    usage = d.get("usage") or {}
    msg = d["choices"][0]["message"]
    # Gemma 4's smaller variants are reasoning models: they put their chain of
    # thought in `reasoning` and only then start `content`. Cut off early they
    # return reasoning alone and no content key at all, which reads as a total
    # failure when it is really a truncation. Fall back so the budget, not the
    # client, is what limits them.
    text = msg.get("content")
    if not text:
        text = msg.get("reasoning") or ""
        usage = dict(usage, truncated_to_reasoning=True)
    return text, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run label; names the output file")
    ap.add_argument("--base", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default=os.path.join(HERE, "..", "goldset", "tasks.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "runs"))
    ap.add_argument("--key", default="not-needed")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0, help="stop after N tasks (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="append, skipping task ids already in the output file")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8")]
    if args.limit:
        tasks = tasks[:args.limit]

    out = os.path.abspath(os.path.join(args.out_dir, args.name + ".jsonl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # A 26B streaming off SSD costs minutes per task, so a full run is worth
    # resuming rather than restarting.
    done = set()
    if args.resume and os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                pass
        tasks = [t for t in tasks if t["id"] not in done]
        print("resuming: %d already done, %d to go" % (len(done), len(tasks)))
    fh = open(out, "a" if args.resume else "w", encoding="utf-8")

    t0 = time.time()
    in_tok = out_tok = 0
    for n, task in enumerate(tasks, 1):
        system, user = prompt_mod.build(task)
        started = time.time()
        try:
            text, usage = call(args.base, args.model, system, user,
                               args.key, args.temperature, args.max_tokens,
                               args.timeout)
            err = None
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            text, usage, err = "", {}, "%s: %s" % (type(e).__name__, e)
        secs = time.time() - started
        entries, how = extract_json(text)
        in_tok += usage.get("prompt_tokens") or 0
        out_tok += usage.get("completion_tokens") or 0

        fh.write(json.dumps({
            "id": task["id"], "book": task["book"], "chapter": task["chapter"],
            "seconds": round(secs, 2), "usage": usage, "parse": how,
            "error": err, "raw": text, "entries": entries,
        }, ensure_ascii=False) + "\n")
        fh.flush()
        print("  %-6s %5.1fs  %-12s %s" % (
            task["id"], secs, how, err or ""), flush=True)

    fh.close()
    took = time.time() - t0
    if not tasks:
        print("nothing to do"); return
    print("\n%s: %d tasks in %.1f min (%.1fs/task)" % (
        args.name, len(tasks), took / 60, took / len(tasks)))
    if in_tok or out_tok:
        print("  tokens: %d in, %d out  (%.1f out-tok/s overall)" % (
            in_tok, out_tok, out_tok / took))
    print("  wrote %s" % out)


if __name__ == "__main__":
    main()

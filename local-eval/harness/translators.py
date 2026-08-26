#!/usr/bin/env python3
"""German → Hungarian, behind one interface, so stage_mt_rank.py can score any
candidate with the same matcher and the numbers stay comparable.

Four backends, because the interesting models do not share a shape:

  translategemma  MLX, no chat mode — its template takes only
                  {source_lang_code, target_lang_code, text}
  nllb            transformers seq2seq (M2M100). Not a chat model either:
                  the target language is a forced first decoder token
  mlx-chat        any MLX instruct model, given the translation as an
                  instruction
  openai          any OpenAI-compatible endpoint, which on this machine means
                  llama.cpp's llama-server holding a GGUF too big to convert

Every backend takes a list of German sentences and returns a list of Hungarian
ones, same length, same order. Nothing else in the bench needs to know which
one produced them.
"""

import json
import re
import urllib.request

# EuroLLM's model card documents this exact shape for translation, and it costs
# nothing to give the other instruct models the same one.
MT_INSTRUCTION = ("Translate the following German source text to Hungarian.\n"
                  "Answer with the translation only.\n"
                  "German: {text}\nHungarian:")


def _clean(text):
    """One line of Hungarian, with the model's scaffolding taken off."""
    t = (text or "").split("<end_of_turn>")[0].strip()
    t = re.sub(r"^(Hungarian|Ungarisch|Magyar)\s*:\s*", "", t, flags=re.I)
    return t.splitlines()[0].strip() if t else ""


def _translategemma(model_path, max_tokens):
    from mlx_lm import load, generate
    model, tok = load(model_path)

    def run(sentences):
        out = []
        for s in sentences:
            msg = [{"role": "user", "content": [{
                "type": "text", "source_lang_code": "de",
                "target_lang_code": "hu", "text": s}]}]
            prompt = tok.apply_chat_template(msg, add_generation_prompt=True)
            out.append(_clean(generate(model, tok, prompt=prompt,
                                       max_tokens=max_tokens, verbose=False)))
        return out
    return run


def _mlx_chat(model_path, max_tokens):
    from mlx_lm import load, generate
    # Teuken ships its own tokenizer class in the repo, so it will not load
    # without this. Read gptx_tokenizer.py before trusting a new checkpoint —
    # this flag runs whatever the repo contains.
    model, tok = load(model_path,
                      tokenizer_config={"trust_remote_code": True})

    def run(sentences):
        out = []
        for s in sentences:
            msg = [{"role": "user", "content": MT_INSTRUCTION.format(text=s)}]
            prompt = tok.apply_chat_template(msg, add_generation_prompt=True)
            out.append(_clean(generate(model, tok, prompt=prompt,
                                       max_tokens=max_tokens, verbose=False)))
        return out
    return run


def _nllb(model_path, max_tokens):
    """NLLB names languages with FLORES codes and forces the target as the
    first decoder token, so there is no prompt to get wrong — and no way for
    it to answer in the wrong language."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tok = AutoTokenizer.from_pretrained(model_path, src_lang="deu_Latn")

    # The 1.3B checkpoint is fp32 on disk: 5.5 GB resident, which the OOM
    # killer takes on an 8 GB machine before it finishes 60 sentences. Half
    # precision on the GPU is 2.6 GB and is what makes it runnable at all.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=dtype)
    model.to(device)
    model.eval()
    # The checkpoint ships max_length=200, which only produces a warning per
    # call once max_new_tokens is given. Drop it and let max_new_tokens rule.
    model.generation_config.max_length = None
    hun = tok.convert_tokens_to_ids("hun_Latn")

    def run(sentences):
        out = []
        for s in sentences:
            enc = tok(s, return_tensors="pt", truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=hun,
                                     max_new_tokens=max_tokens, num_beams=4)
            out.append(_clean(tok.batch_decode(gen,
                                               skip_special_tokens=True)[0]))
        return out
    return run


def _openai(base, model_name, max_tokens, timeout):
    def run(sentences):
        out = []
        for s in sentences:
            body = json.dumps({
                "model": model_name,
                "messages": [{"role": "user",
                              "content": MT_INSTRUCTION.format(text=s)}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }).encode("utf-8")
            req = urllib.request.Request(
                base.rstrip("/") + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer local"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            out.append(_clean(d["choices"][0]["message"].get("content") or ""))
        return out
    return run


def build(backend, model, base=None, max_tokens=180, timeout=600):
    if backend == "translategemma":
        return _translategemma(model, max_tokens)
    if backend == "mlx-chat":
        return _mlx_chat(model, max_tokens)
    if backend == "nllb":
        return _nllb(model, max_tokens)
    if backend == "openai":
        return _openai(base, model, max_tokens, timeout)
    raise SystemExit("unknown backend: %s" % backend)

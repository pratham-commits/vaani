"""Benchmark any HF model on Vaani's evals. --template alpaca (Navarasa) | chat (Gemma/Qwen/Llama)."""
import argparse, json, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval import (n_words, n_sentences, list_items, is_numbered, gujarati_ratio, BASE_Q, INSTR)

ALPACA = ("Below is an instruction that describes a task. Write a response that "
          "appropriately completes the request.\n\n### Instruction:\n{instr}\n\n### Response:\n")

def load_rows(p):
    t = open(p, encoding="utf-8").read().strip()
    return json.loads(t) if t[:1] == "[" else [json.loads(l) for l in t.splitlines() if l.strip()]

def make_inputs(tok, text, template, device):
    if template == "chat" and getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template([{"role":"user","content":text}],
                                      add_generation_prompt=True, return_tensors="pt")
        return {"input_ids": ids.to(device), "attention_mask": torch.ones_like(ids).to(device)}
    enc = tok(ALPACA.format(instr=text), return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}

@torch.no_grad()
def option_score(model, tok, ctx_ids, option_text, device):
    cont = tok(option_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full = torch.cat([ctx_ids, cont], dim=1)
    logp = F.log_softmax(model(full).logits[0].float(), dim=-1)
    plen = ctx_ids.shape[1]; total = 0.0
    for i in range(plen, full.shape[1]):
        total += logp[i-1, full[0, i]].item()
    return total, cont.shape[1]

def run_mcq(model, tok, device, rows, template, use_context=False, limit=None):
    if limit: rows = rows[:limit]
    n = cn = 0
    for r in rows:
        q, opts, gold = r["question"], r["options"], r["answer"]
        qb = q + "\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
        user = (f"સંદર્ભ: {(r.get('explanation') or '').strip()}\n\n{qb}"
                if use_context and (r.get('explanation') or '').strip() else qb)
        ctx = make_inputs(tok, user, template, device)["input_ids"]
        norm = [s/nt for s, nt in (option_score(model, tok, ctx, o, device) for o in opts)]
        cn += int(max(range(len(opts)), key=lambda i: norm[i]) == gold); n += 1
    return cn, n

@torch.no_grad()
def gen_hf(model, tok, user_text, device, template, max_new):
    enc = make_inputs(tok, user_text, template, device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         repetition_penalty=1.0, no_repeat_ngram_size=0, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def run_instruct(model, tok, device, template):
    buckets = {}
    for q in BASE_Q:
        for suf, bucket, ok, mx in INSTR:
            out = gen_hf(model, tok, f"{q} {suf}", device, template, mx)
            buckets.setdefault(bucket, [0,0]); buckets[bucket][0] += bool(ok(out)); buckets[bucket][1] += 1
    print("\n== INSTRUCTION FOLLOWING ==")
    tc = tn = 0
    for b, (c, n) in buckets.items():
        tc += c; tn += n; print(f"  {b:10}: {c}/{n} = {100*c/n:.0f}%")
    print(f"  OVERALL   : {tc}/{tn} = {100*tc/tn:.0f}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["mcq","instruct","openbook","all"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", choices=["alpaca","chat"], default="alpaca")
    ap.add_argument("--mcq_data", default="mcq_expanded.json")
    ap.add_argument("--ob_data", default="medmcqa_gu_val.jsonl")
    ap.add_argument("--ob_n", type=int, default=500)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    print(f"loaded {a.model} (template={a.template}) on {device}\n")
    if a.task in ("mcq","all"):
        cn, n = run_mcq(model, tok, device, load_rows(a.mcq_data), a.template)
        print(f"MCQ (general): acc_norm {cn}/{n} = {100*cn/n:.1f}%")
    if a.task in ("instruct","all"):
        run_instruct(model, tok, device, a.template)
    if a.task in ("openbook","all"):
        rows = [r for r in load_rows(a.ob_data) if (r.get("explanation") or "").strip()][:a.ob_n]
        cc, n = run_mcq(model, tok, device, rows, a.template, False)
        co, _ = run_mcq(model, tok, device, rows, a.template, True)
        print(f"\nOpen-book (N={n}): closed {100*cc/n:.1f}% | with-context {100*co/n:.1f}% | lift +{100*(co-cc)/n:.1f}")

if __name__ == "__main__":
    main()

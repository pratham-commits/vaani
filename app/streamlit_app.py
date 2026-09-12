import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "2.1_pretrain"))

import torch
import torch.nn.functional as F
import streamlit as st
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from model import Vaani, VaaniConfig

MODEL_REPO = "pratham-commits/vaani-gujarati-slm"
CKPT_FILE  = "vaani_med_v5.pt"
TOK_FILE   = "tokenizer/tokenizer_bpe_char.json"
U_PRE = "વપરાશકર્તા:\n"
U_SUF = "\nસહાયક:\n"
device = "cpu"


@st.cache_resource
def load_model():
    tok = Tokenizer.from_file(hf_hub_download(MODEL_REPO, TOK_FILE))
    eos = tok.token_to_id("<eos>") or 0
    m = Vaani(VaaniConfig()).to(device).eval()
    ck = torch.load(hf_hub_download(MODEL_REPO, CKPT_FILE), map_location=device, weights_only=False)
    m.load_state_dict(ck["model"])
    return m, tok, eos


@st.cache_resource
def load_contexts():
    try:
        with open(os.path.join(os.path.dirname(__file__), "contexts.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


model, tok, eos_id = load_model()
CONTEXTS = load_contexts()
CATEGORIES = list(CONTEXTS.keys())
OWN = "મારો પોતાનો સંદર્ભ (my own context)"

GREEDY = dict(temperature=1e-6, top_k=1, top_p=1.0, rep_pen=1.0, no_repeat=0)


def sampling(t):
    return dict(temperature=t, top_k=40, top_p=0.9, rep_pen=1.15, no_repeat=3)


def prompt_ids(q):
    return [eos_id] + tok.encode(U_PRE + q + U_SUF).ids


def build_question(prompt, book_mode, ctx_text):
    if book_mode == "Open-book" and ctx_text.strip():
        return f"સંદર્ભ: {ctx_text.strip()}\n{prompt}"
    return prompt


@torch.no_grad()
def generate(question, max_new, temperature, top_k, top_p, rep_pen, no_repeat):
    base = prompt_ids(question)
    idx = torch.tensor([base], device=device)
    for _ in range(max_new):
        logits, _ = model(idx)
        logits = logits[:, -1, :].float()
        prev = torch.tensor(sorted(set(idx[0].tolist()) - {eos_id}), device=device)
        if len(prev):
            logits[0, prev] /= rep_pen
        if no_repeat and idx.shape[1] >= no_repeat:
            seq = idx[0].tolist()
            prefix = tuple(seq[-(no_repeat - 1):])
            banned = {seq[i + no_repeat - 1] for i in range(len(seq) - no_repeat + 1)
                      if tuple(seq[i:i + no_repeat - 1]) == prefix}
            for b in banned:
                logits[0, b] = -float("inf")
        logits = logits / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        if top_p and top_p < 1.0:
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sp, dim=-1)
            sp[cum - sp > top_p] = 0.0
            sp = sp / sp.sum(dim=-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1))
        else:
            nxt = torch.multinomial(probs, 1)
        if nxt.item() == eos_id:
            break
        idx = torch.cat([idx, nxt], dim=1)
    return tok.decode(idx[0, len(base):].tolist()).strip()


@torch.no_grad()
def score_option(question, option):
    p = prompt_ids(question)
    opt = tok.encode(option).ids
    if not opt:
        return -1e9
    full = torch.tensor([p + opt], device=device)
    logits, _ = model(full, full)
    lp = F.log_softmax(logits[0].float(), dim=-1)
    return sum(lp[len(p) + j - 1, t].item() for j, t in enumerate(opt)) / len(opt)


st.set_page_config(page_title="Vaani — Gujarati SLM (110M)", page_icon="🪔")
st.title("વાણી — Gujarati Small Language Model (~110M)")
st.caption("A ~110M Gujarati model trained from scratch. Illustrative demo, not medical advice.")

with st.expander("📊 About Vaani & benchmark results", expanded=True):
    st.markdown("""
**Vaani** is a ~110M-parameter Gujarati language model, pretrained from scratch and fine-tuned for
Gujarati instruction-following and grounded (open-book) medical MCQ. Same eval harness for every
model below — MCQ = acc_norm (log-likelihood); Instruction = rule-based checkers; Open-book =
MedMCQA-gu with context.

| Model | Params | MCQ | Instruction | Open-book w/ctx |
| --- | --- | --- | --- | --- |
| **Vaani (final)** | **110M** | 38.6% | **70%** | **44.1%** |
| gemma-2-2b-it | 2B | 37.2% | 67% | 37.8% |
| sarvam-1 (base) | 2B | 36.6% | 24% | 40.2% |
| Navarasa-2.0 | 2B | 42.8% | 54% | 35.0% |
| Qwen2.5-7B-Instruct | 7B | 51.0% | 53% | 44.0% |

These are **averages over full test sets** — any single answer can be wrong. Vaani's strengths are
Gujarati fluency and, for its size, competitive grounded reading. It is **not** a reliable source of
unaided medical facts.
""")

task = st.radio("કાર્ય (task)", ["Chat", "MCQ"], horizontal=True,
                help="Chat = free-form Gujarati generation (closed-book). MCQ = score answer options.")

if task == "Chat":
    prompt = st.text_area("તમારો પ્રશ્ન (your question)", height=90,
                          placeholder="દા.ત. ગુજરાત વિશે થોડું લખો.")
    max_new = st.slider("Max new tokens", 32, 320, 200, 4)
    temperature = st.slider("Temperature", 0.1, 1.2, 0.8, 0.05)
    use_greedy = st.checkbox("Greedy (deterministic) — can repeat on long answers", value=False)

    if st.button("ચલાવો (run)", type="primary"):
        if not prompt.strip():
            st.warning("કૃપા કરીને પ્રશ્ન લખો.")
        else:
            with st.spinner("વિચારી રહ્યું છે…"):
                params, tag = (GREEDY, "greedy") if use_greedy else (sampling(temperature), "sampling")
                st.write(generate(prompt.strip(), max_new, **params))
                st.caption(f"Closed-book · Gujarati fluency · decoding: {tag}")

else:
    book = st.radio("મોડ (mode)", ["Closed-book", "Open-book"], horizontal=True,
                    help="Open-book adds a context passage the model reads before scoring options.")
    ctx_text = ""
    if book == "Open-book":
        default_ctx = ""
        if CATEGORIES:
            csel = st.selectbox("તબીબી વિષય (topic)", CATEGORIES + [OWN])
            default_ctx = CONTEXTS.get(csel, {}).get("text", "")
            if CONTEXTS.get(csel, {}).get("source"):
                st.caption(f"સ્રોત / source: {CONTEXTS[csel]['source']} — Gujarati Wikipedia, CC BY-SA")
        ctx_text = st.text_area("સંદર્ભ (context)", value=default_ctx, height=150)

    prompt = st.text_area("પ્રશ્ન (question)", height=80,
                          placeholder="દા.ત. મલેરિયા કયા જીવજંતુથી ફેલાય છે?")
    st.markdown("**વિકલ્પો ભરો — options (fill 2 to 4):**")
    opts_in = [st.text_input("વિકલ્પ 1"), st.text_input("વિકલ્પ 2"),
               st.text_input("વિકલ્પ 3"), st.text_input("વિકલ્પ 4 (વૈકલ્પિક / optional)")]

    if st.button("ચલાવો (run)", type="primary"):
        opts = [(i, x.strip()) for i, x in enumerate(opts_in) if x.strip()]
        if not prompt.strip():
            st.warning("કૃપા કરીને પ્રશ્ન લખો.")
        elif len(opts) < 2:
            st.warning("ઓછામાં ઓછા બે વિકલ્પ ભરો.")
        else:
            with st.spinner("વિચારી રહ્યું છે…"):
                question = build_question(prompt.strip(), book, ctx_text)
                scored = [(i, x, score_option(question, x)) for i, x in opts]
                best = max(scored, key=lambda t: t[2])
                for i, x, s in scored:
                    st.write(f"{'✅' if i == best[0] else '▫️'} વિકલ્પ {i+1}: {x}  ·  score {s:.3f}")
                st.caption(f"{book} · ranked by likelihood (acc_norm) · picks વિકલ્પ {best[0]+1}. "
                           "~44% average — try several; it won't get every one.")

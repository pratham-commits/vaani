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

STRUCT_CUES = ["યાદી", "મુદ્દા", "મુદ્દાઓ", "ક્રમાંકિત", "ક્રમ", "પગલાં", "પગલા",
               "બુલેટ", "કોષ્ટક", "કાઢો", "ફોર્મેટ", "json", "list", "numbered",
               "points", "steps", "bullet", "table", "extract"]
GREEDY = dict(temperature=1e-6, top_k=1, top_p=1.0, rep_pen=1.0, no_repeat=0)


def looks_structured(q):
    ql = q.lower()
    return any(c.lower() in ql for c in STRUCT_CUES)


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


def try_json(raw):
    cands = [raw]
    if "{" in raw and "}" in raw:
        cands.append(raw[raw.find("{"): raw.rfind("}") + 1])
    for cand in cands:
        try:
            return json.dumps(json.loads(cand), ensure_ascii=False, indent=2), True
        except Exception:
            continue
    return raw, False


st.set_page_config(page_title="Vaani — Gujarati SLM (110M)", page_icon="🪔")
st.title("વાણી — Gujarati Small Language Model (~110M)")
st.caption("Chat / MCQ / JSON · closed-book or open-book (RAG). First run loads the model (~20–40s). "
           "Illustrative demo, not medical advice.")

c1, c2 = st.columns(2)
task = c1.radio("કાર્ય (task)", ["Chat", "MCQ", "JSON"], horizontal=True)
book = c2.radio("મોડ (mode)", ["Closed-book", "Open-book"], horizontal=True)

ctx_text = ""
if book == "Open-book":
    default_ctx = ""
    if CATEGORIES:
        csel = st.selectbox("તબીબી વિષય (topic)", CATEGORIES + [OWN])
        default_ctx = CONTEXTS.get(csel, {}).get("text", "")
        if CONTEXTS.get(csel, {}).get("source"):
            st.caption(f"સ્રોત / source: {CONTEXTS[csel]['source']} — Gujarati Wikipedia, CC BY-SA")
    ctx_text = st.text_area("સંદર્ભ (context)", value=default_ctx, height=150)

prompt = st.text_area("તમારો પ્રશ્ન (your question)", height=90,
                      placeholder="દા.ત. ડાયાબિટીસનાં લક્ષણો શું છે?")

opts_in = ["", "", "", ""]
if task == "MCQ":
    st.markdown("**વિકલ્પો ભરો — options (fill 2 to 4):**")
    opts_in[0] = st.text_input("વિકલ્પ 1")
    opts_in[1] = st.text_input("વિકલ્પ 2")
    opts_in[2] = st.text_input("વિકલ્પ 3")
    opts_in[3] = st.text_input("વિકલ્પ 4 (વૈકલ્પિક / optional)")

max_new, temperature, decode_mode = 200, 0.8, "Auto"
if task in ("Chat", "JSON"):
    max_new = st.slider("Max new tokens", 32, 320, 200, 4)
if task == "Chat":
    decode_mode = st.radio("ડીકોડિંગ (decoding)", ["Auto", "Sampling", "Greedy"], horizontal=True)
    temperature = st.slider("Temperature", 0.1, 1.2, 0.8, 0.05)

if st.button("ચલાવો (run)", type="primary"):
    if not prompt.strip():
        st.warning("કૃપા કરીને પ્રશ્ન લખો.")
    else:
        question = build_question(prompt.strip(), book, ctx_text)
        with st.spinner("વિચારી રહ્યું છે…"):
            if task == "MCQ":
                opts = [(i, x.strip()) for i, x in enumerate(opts_in) if x.strip()]
                if len(opts) < 2:
                    st.warning("ઓછામાં ઓછા બે વિકલ્પ ભરો.")
                else:
                    scored = [(i, x, score_option(question, x)) for i, x in opts]
                    best = max(scored, key=lambda t: t[2])
                    for i, x, s in scored:
                        st.write(f"{'✅' if i == best[0] else '▫️'} વિકલ્પ {i+1}: {x}  ·  score {s:.3f}")
                    st.caption(f"{book} · MCQ (log-likelihood, acc_norm) · choice: વિકલ્પ {best[0]+1}")
            elif task == "JSON":
                raw = generate(question, max_new, **GREEDY)
                pretty, ok = try_json(raw)
                st.code(pretty, language="json")
                st.caption(f"{book} · JSON (greedy) · {'valid JSON ✓' if ok else 'not valid JSON ✗'}")
            else:
                if decode_mode == "Greedy":
                    params, tag = GREEDY, "greedy"
                elif decode_mode == "Sampling":
                    params, tag = sampling(temperature), "sampling"
                elif book == "Open-book" or looks_structured(prompt):
                    params, tag = GREEDY, "greedy (auto)"
                else:
                    params, tag = sampling(temperature), "sampling (auto)"
                st.write(generate(question, max_new, **params))
                st.caption(f"{book} · decoding: {tag}")

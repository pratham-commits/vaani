#!/usr/bin/env python3
"""
train_tokenizer.py — train & compare Gujarati tokenizer configs for Vaani.

Trains 4 configs on the same NFKC-normalized sample and reports fertility
(tokens/word, lower is better) on a held-out slice:
  1. bpe_byte : byte-level BPE     + LLaMA-4 regex   (GPT/Llama style)
  2. uni_byte : byte-level Unigram + LLaMA-4 regex
  3. bpe_char : char-level BPE     + Metaspace       (SentencePiece style)  <- chosen
  4. uni_char : char-level Unigram + Metaspace       (SentencePiece style)

See DECISIONS.md D1.6. Run:
  python3 train_tokenizer.py --input ~/clean_guj --out-dir ~/tokenizer
"""

import argparse, glob, gzip, itertools, json, os, time

from tokenizers import Tokenizer, Regex, decoders, normalizers
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel, Metaspace

# LLaMA-3/4-style pre-tokenization regex (Unicode \p{L}/\p{N}, Indic-friendly).
LLAMA_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
VOCAB = 32000   # set from CLI in main()


# ------------------------------ data plumbing ------------------------------
def iter_docs(shards, limit_docs=0):
    n = 0
    for sh in shards:
        with gzip.open(sh, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)["text"]
                except Exception:
                    continue
                n += 1
                if limit_docs and n >= limit_docs:
                    return


def train_iterator(shards, max_chars):
    chars = 0
    for txt in iter_docs(shards):
        yield txt
        chars += len(txt)
        if chars >= max_chars:
            return


# --------------------------- config builders ------------------------------
def _byte_pretok():
    return Sequence([
        Split(Regex(LLAMA_PATTERN), behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])


def build_bpe_byte():
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = _byte_pretok()
    tok.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL,
                         initial_alphabet=ByteLevel.alphabet(), show_progress=True)
    return tok, trainer


def build_uni_byte():
    tok = Tokenizer(Unigram())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = _byte_pretok()
    tok.decoder = decoders.ByteLevel()
    trainer = UnigramTrainer(vocab_size=VOCAB, special_tokens=SPECIAL,
                             unk_token="<unk>", initial_alphabet=ByteLevel.alphabet(),
                             show_progress=True)
    return tok, trainer


def build_bpe_char():
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = Metaspace()          # char-level: keeps Gujarati chars whole
    tok.decoder = decoders.Metaspace()
    trainer = BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL, show_progress=True)
    return tok, trainer


def build_uni_char():
    tok = Tokenizer(Unigram())
    tok.normalizer = normalizers. NFKC()
    tok.pre_tokenizer = Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = UnigramTrainer(vocab_size=VOCAB, special_tokens=SPECIAL,
                             unk_token="<unk>", show_progress=True)
    return tok, trainer


CONFIGS = {
    "bpe_byte": build_bpe_byte,
    "uni_byte": build_uni_byte,
    "bpe_char": build_bpe_char,
    "uni_char": build_uni_char,
}


# -------------------------------- evaluation -------------------------------
def fertility(tok, docs):
    toks = words = chars = 0
    for d in docs:
        w = len(d.split())
        if w == 0:
            continue
        words += w
        toks += len(tok.encode(d).ids)
        chars += len(d)
    return {"tokens_per_word": round(toks / max(words, 1), 4),
            "chars_per_token": round(chars / max(toks, 1), 4),
            "words": words, "tokens": toks}


def main():
    global VOCAB
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="~/clean_guj")
    p.add_argument("--out-dir", default="~/tokenizer")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--train-chars", type=int, default=1_000_000_000)
    p.add_argument("--heldout-docs", type=int, default=5000)
    p.add_argument("--configs", default="bpe_byte,uni_byte,bpe_char,uni_char")
    a = p.parse_args()
    VOCAB = a.vocab_size

    out = os.path.expanduser(a.out_dir)
    os.makedirs(out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(os.path.expanduser(a.input), "*.jsonl.gz")))
    if not shards:
        raise SystemExit(f"no shards found in {a.input}")
    print(f"found {len(shards)} shards", flush=True)

    heldout = list(itertools.islice(iter_docs([shards[-1]]), a.heldout_docs))
    train_shards = shards[:-1]
    print(f"held out {len(heldout)} docs; training on {len(train_shards)} shards "
          f"up to {a.train_chars/1e9:.1f}B chars", flush=True)

    results = {}
    for name in [c.strip() for c in a.configs.split(",") if c.strip()]:
        if name not in CONFIGS:
            print("skip unknown config:", name); continue
        print(f"\n=== training {name} ===", flush=True)
        t0 = time.time()
        tok, trainer = CONFIGS[name]()
        tok.train_from_iterator(train_iterator(train_shards, a.train_chars), trainer=trainer)
        tok.save(os.path.join(out, f"tokenizer_{name}.json"))
        r = fertility(tok, heldout)
        r["train_min"] = round((time.time() - t0) / 60, 1)
        r["vocab"] = tok.get_vocab_size()
        results[name] = r

    with open(os.path.join(out, "fertility_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== FERTILITY REPORT (held-out Gujarati) ===", flush=True)
    print(f"{'config':<10} {'tok/word↓':>10} {'char/tok↑':>10} {'vocab':>7} {'min':>6}")
    for name, r in results.items():
        print(f"{name:<10} {r['tokens_per_word']:>10} {r['chars_per_token']:>10} "
              f"{r['vocab']:>7} {r['train_min']:>6}", flush=True)
    print("\nLower tok/word = better. Saved to", out, flush=True)


if __name__ == "__main__":
    main()

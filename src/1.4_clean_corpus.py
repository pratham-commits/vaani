#!/usr/bin/env python3
"""
clean_corpus1.py — Phase 2 cleaning pass for the Vaani Gujarati corpus.

Input : gzipped JSONL shards from dedup (one {"text": ...} per line).
Output: cleaned gzipped JSONL shards + manifest with per-rule drop counts.
        Also dumps the full boilerplate list to ~/boilerplate_guj.json.

Passes:
  PASS 1  count SHORT + JUNK-SHAPED lines across the corpus -> boilerplate set
          (nav bars, footers, cookie/legal notices, ©, URLs, markup) — NOT
          legitimate Gujarati section headings.
  PASS 2  normalize each doc, strip boilerplate/menu lines, apply doc-level
          quality filters, write survivors.

Gujarati-safe:
  - keeps danda (।, U+0964) and period (.) as terminators (never penalized)
  - keeps ZWJ (U+200D) / ZWNJ (U+200C) needed for conjuncts (જ્ઞ, ક્ષ)
  - treats the Gujarati block (U+0A80-U+0AFF) as letters (matras don't inflate
    the symbol ratio)
  - in-context English kept; only English-DOMINANT junk lines removed

Run (in tmux):
  python3 clean_corpus1.py --input ~/dedup_jsonl --output ~/clean_guj
  # regenerate full boilerplate list only (no cleaning):
  python3 clean_corpus1.py --input ~/dedup_jsonl --pass1-only
"""

import argparse
import glob
import gzip
import json
import os
import re
import time
import unicodedata
from collections import Counter

# ----------------------------- tunable thresholds -----------------------------
MIN_LEN            = 200
MIN_WORDS          = 20
MIN_GUJ_RATIO      = 0.70
MAX_DIGIT_RATIO    = 0.50
MAX_SYMBOL_RATIO   = 0.10
MIN_MEAN_WORD_LEN  = 2.0
MAX_MEAN_WORD_LEN  = 20.0
MAX_DUP_LINE_FRAC  = 0.30
MAX_TOP_LINE_FRAC  = 0.20
MAX_TOP_3GRAM_FRAC = 0.20

BOILERPLATE_MAX_WORDS = 10
BOILERPLATE_MAX_CHARS = 200
BOILERPLATE_MIN_COUNT = 100
PRUNE_CAP             = 5_000_000

MENU_MIN_SEPARATORS = 2
MENU_MAX_SEG_WORDS  = 2
SEPARATORS = set("|•·»›→❯▪")
_SEP_SPLIT = re.compile(r"[|•·»›→❯▪]+")

SHARD_SIZE = 100_000
GUJ_LO, GUJ_HI = 0x0A80, 0x0AFF
KEEP_PUNCT = set("।.,?!;:\"'()[]{}-–—…“”‘’%")
KEEP_ZERO_WIDTH = {"\u200c", "\u200d"}

_ws_run = re.compile(r"[ \t]+")
_nl_run = re.compile(r"\n{3,}")
_LATIN  = re.compile(r"[A-Za-z]")
_URLISH = re.compile(r"https?://|www\.|\.com|\.org|\.net|\.in\b")


# ------------------------------- normalization --------------------------------
def strip_controls(text):
    out = []
    for ch in text:
        if ch in KEEP_ZERO_WIDTH:
            out.append(ch); continue
        if unicodedata.category(ch)[0] == "C" and ch not in ("\n", "\t"):
            continue
        out.append(ch)
    return "".join(out)


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = strip_controls(text)
    lines = [_ws_run.sub(" ", ln).strip() for ln in text.split("\n")]
    return _nl_run.sub("\n\n", "\n".join(lines)).strip()


# --------------------------------- metrics ------------------------------------
def guj_ratio(text):
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    g = sum(1 for c in non_ws if GUJ_LO <= ord(c) <= GUJ_HI)
    return g / len(non_ws)


def char_stats(text):
    non_ws = [c for c in text if not c.isspace()]
    total = len(non_ws)
    if total == 0:
        return 0.0, 0.0
    digits = sum(1 for c in non_ws if c.isdigit())
    symbols = sum(1 for c in non_ws
                  if (not c.isalnum())
                  and (c not in KEEP_PUNCT)
                  and not (GUJ_LO <= ord(c) <= GUJ_HI))
    return digits / total, symbols / total


def word_stats(text):
    words = text.split()
    if not words:
        return 0, 0.0
    return len(words), sum(len(w) for w in words) / len(words)


def repetition_stats(text):
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 1.0, 1.0, 0.0
    n = len(lines)
    dup_line_frac = (n - len(set(lines))) / n
    total_chars = sum(len(ln) for ln in lines) or 1
    counts = Counter(lines)
    repeated = [(ln, c) for ln, c in counts.items() if c >= 2]
    top_line_frac = (max(c * len(ln) for ln, c in repeated) / total_chars) if repeated else 0.0

    words = text.split()
    top_3gram_frac = 0.0
    if len(words) >= 3:
        grams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
        gram, c = grams.most_common(1)[0]
        top_3gram_frac = c * (sum(len(w) for w in gram) + 2) / (len(text) or 1)
    return dup_line_frac, top_line_frac, top_3gram_frac


def looks_like_junk(line):
    """True if a short line looks like nav/footer junk (not clean Gujarati prose)."""
    if any(c in SEPARATORS for c in line):
        return True
    if _URLISH.search(line):
        return True
    if _LATIN.search(line):                       # Latin counts as junk only if
        g = sum(1 for c in line if GUJ_LO <= ord(c) <= GUJ_HI)
        nonws = sum(1 for c in line if not c.isspace())
        if nonws == 0 or g / nonws < 0.5:         # ...line is NOT majority Gujarati
            return True
    if "©" in line or "®" in line:
        return True
    non_ws = [c for c in line if not c.isspace()]
    if non_ws:
        sym = sum(1 for c in non_ws
                  if (not c.isalnum()) and c not in KEEP_PUNCT
                  and not (GUJ_LO <= ord(c) <= GUJ_HI))
        if sym / len(non_ws) > 0.2:
            return True
    return False


def is_menu_line(line):
    if sum(1 for c in line if c in SEPARATORS) < MENU_MIN_SEPARATORS:
        return False
    segs = [s.strip() for s in _SEP_SPLIT.split(line) if s.strip()]
    return bool(segs) and all(len(s.split()) <= MENU_MAX_SEG_WORDS for s in segs)


def is_boilerplate_candidate(line):
    return len(line) <= BOILERPLATE_MAX_CHARS and len(line.split()) <= BOILERPLATE_MAX_WORDS


# ---------------------------------- passes ------------------------------------
def pass1_count(shards, limit):
    counter, docs = Counter(), 0
    t0 = time.time()
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for raw in f:
                docs += 1
                try:
                    text = normalize(json.loads(raw)["text"])
                except Exception:
                    continue
                for ln in text.split("\n"):
                    ln = ln.strip()
                    if ln and is_boilerplate_candidate(ln) and looks_like_junk(ln):
                        counter[ln] += 1
                if len(counter) > PRUNE_CAP:
                    for k in [k for k, v in counter.items() if v == 1]:
                        del counter[k]
                if limit and docs >= limit:
                    break
        if limit and docs >= limit:
            break
    boiler = {ln: c for ln, c in counter.items() if c >= BOILERPLATE_MIN_COUNT}
    print(f"[pass1] scanned {docs:,} docs, {len(boiler):,} boilerplate lines "
          f"in {(time.time()-t0)/60:.1f}m", flush=True)
    return boiler


def pass2_clean(shards, out_dir, boiler, limit):
    os.makedirs(out_dir, exist_ok=True)
    drops = Counter()
    kept = total = stripped = shard_idx = 0
    out_f = None
    t0 = time.time()
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for raw in f:
                total += 1
                try:
                    text = normalize(json.loads(raw)["text"])
                except Exception:
                    drops["bad_json"] += 1
                    continue

                kept_lines = []
                for ln in text.split("\n"):
                    s = ln.strip()
                    if not s:
                        kept_lines.append("")
                        continue
                    if s in boiler or is_menu_line(s):
                        stripped += 1
                        continue
                    kept_lines.append(ln)
                text = _nl_run.sub("\n\n", "\n".join(kept_lines)).strip()

                if len(text) < MIN_LEN:
                    drops["too_short"] += 1; continue
                nwords, mean_wl = word_stats(text)
                if nwords < MIN_WORDS:
                    drops["few_words"] += 1; continue
                if not (MIN_MEAN_WORD_LEN <= mean_wl <= MAX_MEAN_WORD_LEN):
                    drops["bad_word_len"] += 1; continue
                if guj_ratio(text) < MIN_GUJ_RATIO:
                    drops["low_guj"] += 1; continue
                dr, sr = char_stats(text)
                if dr > MAX_DIGIT_RATIO:
                    drops["high_digit"] += 1; continue
                if sr > MAX_SYMBOL_RATIO:
                    drops["high_symbol"] += 1; continue
                dlf, tlf, t3f = repetition_stats(text)
                if dlf > MAX_DUP_LINE_FRAC:
                    drops["dup_lines"] += 1; continue
                if tlf > MAX_TOP_LINE_FRAC:
                    drops["top_line"] += 1; continue
                if t3f > MAX_TOP_3GRAM_FRAC:
                    drops["top_3gram"] += 1; continue

                if out_f is None or kept % SHARD_SIZE == 0:
                    if out_f:
                        out_f.close()
                    out_f = gzip.open(os.path.join(out_dir, f"clean_{shard_idx:05d}.jsonl.gz"),
                                      "wt", encoding="utf-8", compresslevel=1)
                    shard_idx += 1
                out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                kept += 1

                if limit and total >= limit:
                    break
        print(f"[pass2] {total:,} read | {kept:,} kept | {stripped:,} lines stripped "
              f"| {(time.time()-t0)/60:.1f}m", flush=True)
        if limit and total >= limit:
            break
    if out_f:
        out_f.close()
    return kept, total, stripped, drops, shard_idx


def dump_boilerplate(boiler, path):
    with open(os.path.expanduser(path), "w", encoding="utf-8") as f:
        json.dump({"count": len(boiler),
                   "lines": [{"line": ln, "freq": c}
                             for ln, c in sorted(boiler.items(), key=lambda x: -x[1])]},
                  f, ensure_ascii=False, indent=2)
    print(f"wrote full boilerplate list ({len(boiler)} lines) -> {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Vaani corpus cleaning pass")
    p.add_argument("--input", default="~/dedup_jsonl")
    p.add_argument("--output", default="~/clean_guj")
    p.add_argument("--manifest", default="~/clean_guj_manifest.json")
    p.add_argument("--boilerplate", default="~/boilerplate_guj.json")
    p.add_argument("--limit", type=int, default=0, help="process only first N docs (dry run)")
    p.add_argument("--pass1-only", action="store_true",
                   help="only build + dump the boilerplate list, skip cleaning")
    a = p.parse_args()

    shards = sorted(glob.glob(os.path.join(os.path.expanduser(a.input), "*.jsonl.gz")))
    if not shards:
        raise SystemExit(f"no shards found in {a.input}")
    print(f"found {len(shards)} input shards", flush=True)

    boiler = pass1_count(shards, a.limit)
    dump_boilerplate(boiler, a.boilerplate)
    if a.pass1_only:
        return

    kept, total, stripped, drops, nshards = pass2_clean(
        shards, os.path.expanduser(a.output), boiler, a.limit)

    manifest = {
        "step": "clean",
        "date": time.strftime("%Y-%m-%d"),
        "input_docs": total,
        "kept": kept,
        "dropped": total - kept,
        "dropped_pct": round(100 * (total - kept) / total, 2) if total else 0,
        "lines_stripped": stripped,
        "out_shards": nshards,
        "boilerplate_lines": len(boiler),
        "boilerplate_sample": sorted(boiler, key=boiler.get, reverse=True)[:20],
        "drops_by_rule": dict(drops),
        "thresholds": {
            "MIN_LEN": MIN_LEN, "MIN_WORDS": MIN_WORDS, "MIN_GUJ_RATIO": MIN_GUJ_RATIO,
            "MAX_DIGIT_RATIO": MAX_DIGIT_RATIO, "MAX_SYMBOL_RATIO": MAX_SYMBOL_RATIO,
            "MEAN_WORD_LEN": [MIN_MEAN_WORD_LEN, MAX_MEAN_WORD_LEN],
            "MAX_DUP_LINE_FRAC": MAX_DUP_LINE_FRAC, "MAX_TOP_LINE_FRAC": MAX_TOP_LINE_FRAC,
            "MAX_TOP_3GRAM_FRAC": MAX_TOP_3GRAM_FRAC,
            "BOILERPLATE_MIN_COUNT": BOILERPLATE_MIN_COUNT,
        },
    }
    with open(os.path.expanduser(a.manifest), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDONE: {total:,} -> {kept:,} kept ({manifest['dropped_pct']}% dropped), "
          f"{stripped:,} lines stripped, {nshards} shards", flush=True)
    print("drops by rule:", dict(drops), flush=True)


if __name__ == "__main__":
    main()

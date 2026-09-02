#!/usr/bin/env python3
"""measure_english.py — English (Latin) vs Gujarati stats across the corpus.

'English word' = a run of ASCII alphanumerics [A-Za-z0-9]+ (so it includes
English letters, English words, AND numbers written in English). Alphabetic
tokens are lowercased for the unique-word count; numbers are kept as-is.
NOTE: including numbers inflates 'unique words' (every distinct number is its
own type). Set --no-numbers to count only [A-Za-z]+ if you prefer.
"""
import argparse, glob, gzip, json, os, re, time
from collections import Counter

LATIN = re.compile(r"[A-Za-z]")
GUJ   = re.compile(r"[\u0A80-\u0AFF]")
NONWS = re.compile(r"\S")
ENG_WORD_ALNUM = re.compile(r"[A-Za-z0-9]+")   # letters, words, English numbers
ENG_WORD_ALPHA = re.compile(r"[A-Za-z]+")      # letters/words only


def bucket(frac):
    if frac == 0:      return "0 (none)"
    if frac < 0.01:    return "<1%"
    if frac < 0.05:    return "1-5%"
    if frac < 0.10:    return "5-10%"
    if frac < 0.20:    return "10-20%"
    if frac < 0.30:    return "20-30%"
    return ">30%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="~/clean_guj")
    p.add_argument("--out", default="~/english_report.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-numbers", action="store_true",
                   help="count English words as [A-Za-z]+ only (exclude numbers)")
    a = p.parse_args()

    word_re = ENG_WORD_ALPHA if a.no_numbers else ENG_WORD_ALNUM

    shards = sorted(glob.glob(os.path.join(os.path.expanduser(a.input), "*.jsonl.gz")))
    if not shards:
        raise SystemExit(f"no shards found in {a.input}")
    print(f"found {len(shards)} shards", flush=True)

    docs = docs_guj = docs_eng = 0
    total_latin = total_guj = total_nonws = 0
    total_eng_words = 0
    unique_eng = set()
    sum_frac = 0.0
    dist = Counter()
    t0 = time.time()

    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for raw in f:
                try:
                    text = json.loads(raw)["text"]
                except Exception:
                    continue
                nonws = len(NONWS.findall(text))
                if nonws == 0:
                    continue
                docs += 1
                lat = len(LATIN.findall(text))
                guj = len(GUJ.findall(text))
                total_nonws += nonws
                total_latin += lat
                total_guj += guj
                if guj > 0:
                    docs_guj += 1
                if lat > 0:
                    docs_eng += 1

                words = word_re.findall(text)
                total_eng_words += len(words)
                for w in words:
                    unique_eng.add(w.lower())

                frac = lat / nonws
                sum_frac += frac
                dist[bucket(frac)] += 1
                if a.limit and docs >= a.limit:
                    break
        print(f"...{docs:,} docs | Latin {100*total_latin/max(total_nonws,1):.2f}% "
              f"| uniq-eng {len(unique_eng):,} | {(time.time()-t0)/60:.1f}m", flush=True)
        if a.limit and docs >= a.limit:
            break

    report = {
        "docs": docs,
        "docs_with_gujarati": docs_guj,
        "pct_docs_with_gujarati": round(100 * docs_guj / max(docs, 1), 3),
        "docs_with_english": docs_eng,
        "pct_docs_with_english": round(100 * docs_eng / max(docs, 1), 3),
        "corpus_gujarati_pct": round(100 * total_guj / max(total_nonws, 1), 3),
        "corpus_latin_pct": round(100 * total_latin / max(total_nonws, 1), 3),
        "mean_per_doc_latin_pct": round(100 * sum_frac / max(docs, 1), 3),
        "total_english_letters": total_latin,
        "total_english_words": total_eng_words,
        "unique_english_words": len(unique_eng),
        "pct_unique_english_words": round(100 * len(unique_eng) / max(total_eng_words, 1), 3),
        "avg_english_words_per_doc": round(total_eng_words / max(docs, 1), 3),
        "doc_distribution_by_latin_share": dict(dist),
        "english_word_definition": "[A-Za-z]+" if a.no_numbers else "[A-Za-z0-9]+",
    }
    with open(os.path.expanduser(a.out), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== ENGLISH/GUJARATI REPORT ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

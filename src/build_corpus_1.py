"""
build_corpus.py — Gujarati pretraining corpus builder for the Vaani SLM.

Streams each configured Hugging Face source, applies rule-based Gujarati
quality filters, and writes per-source gzipped JSONL shards + a manifest.
One script reproduces the entire Phase-1 corpus.

Sources (all Gujarati, native script):
  sangraha_verified   : ai4bharat/sangraha       verified/guj
  sangraha_unverified : ai4bharat/sangraha       unverified/guj
  sangraha_synthetic  : ai4bharat/sangraha       synthetic/guj_Gujr   (optional doc cap)
  wikipedia_gu        : wikimedia/wikipedia       20231101.gu
  fineweb2_gu         : HuggingFaceFW/fineweb-2   guj_Gujr   (replaces cc100 — script-based, unsupported on datasets 5.x)

Licenses: Sangraha CC-BY-4.0 (arXiv:2403.06350); Wikipedia CC-BY-SA; FineWeb-2 ODC-BY.
"""
from __future__ import annotations
import argparse, gzip, json, os, re, unicodedata
from datetime import datetime, timezone
from datasets import load_dataset

GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")

def gujarati_ratio(text: str) -> float:
    non_space = re.sub(r"\s", "", text)
    return len(GUJARATI_RE.findall(text)) / len(non_space) if non_space else 0.0

def is_good(text: str, *, min_len, max_len, max_replacement_ratio, min_guj_ratio) -> bool:
    if not (min_len <= len(text) <= max_len): return False
    if text.count("\ufffd") / len(text) > max_replacement_ratio: return False
    if gujarati_ratio(text) < min_guj_ratio: return False
    return True

def write_shard(buffer, out_dir, name, index) -> None:
    path = os.path.join(out_dir, f"{name}_{index:04d}.jsonl.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for text in buffer:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({len(buffer):,} docs)", flush=True)

def load_source(src):
    kw = dict(split="train", streaming=True)
    if src.get("data_dir"): kw["data_dir"] = src["data_dir"]
    if src.get("config"):   kw["name"] = src["config"]
    return load_dataset(src["hf_id"], **kw)

def process(src, root, filters, shard_size, log_every):
    out_dir = os.path.join(root, src["name"]); os.makedirs(out_dir, exist_ok=True)
    print(f"[source] {src['name']} <- {src['hf_id']} {src.get('data_dir') or src.get('config') or ''}", flush=True)
    ds = load_source(src)
    key, cap = src.get("key", "text"), src.get("max_docs")
    kept = dropped = seen = idx = 0; buffer = []
    try:
        for row in ds:
            seen += 1
            text = unicodedata.normalize("NFKC", (row.get(key) or "").strip())
            if is_good(text, **filters):
                buffer.append(text); kept += 1
                if len(buffer) >= shard_size:
                    write_shard(buffer, out_dir, src["name"], idx); idx += 1; buffer = []
            else:
                dropped += 1
            if seen % log_every == 0:
                print(f"  {src['name']}: seen {seen:,} | kept {kept:,} | dropped {dropped:,}", flush=True)
            if cap and kept >= cap:
                print(f"  {src['name']}: hit doc cap {cap:,}", flush=True); break
    except KeyboardInterrupt:
        print("\n[interrupted] flushing buffer...", flush=True)
    finally:
        if buffer:
            write_shard(buffer, out_dir, src["name"], idx); idx += 1
    manifest = {"source": src, "filters": filters, "seen": seen, "kept": kept,
                "dropped": dropped, "shards": idx, "shard_size": shard_size,
                "finished_utc": datetime.now(timezone.utc).isoformat()}
    with open(os.path.join(out_dir, f"{src['name']}_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[done] {src['name']}: kept {kept:,}, dropped {dropped:,}, shards {idx}\n", flush=True)

SOURCES = [
    {"name": "sangraha_verified",   "hf_id": "ai4bharat/sangraha",     "data_dir": "verified/guj"},
    {"name": "sangraha_unverified", "hf_id": "ai4bharat/sangraha",     "data_dir": "unverified/guj"},
    {"name": "sangraha_synthetic",  "hf_id": "ai4bharat/sangraha",     "data_dir": "synthetic/guj_Gujr", "max_docs": 8_000_000},
    {"name": "wikipedia_gu",        "hf_id": "wikimedia/wikipedia",    "config": "20231101.gu"},
    {"name": "fineweb2_gu",         "hf_id": "HuggingFaceFW/fineweb-2","config": "guj_Gujr"},
]

def parse_args():
    p = argparse.ArgumentParser(description="Build the Gujarati pretraining corpus (all sources).")
    p.add_argument("--root", default=os.path.expanduser("~/corpus_raw"))
    p.add_argument("--only", nargs="*", help="run only these source names")
    p.add_argument("--shard-size", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=100_000)
    p.add_argument("--min-len", type=int, default=200)
    p.add_argument("--max-len", type=int, default=100_000)
    p.add_argument("--max-replacement-ratio", type=float, default=0.001)
    p.add_argument("--min-guj-ratio", type=float, default=0.70)
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    filters = dict(min_len=a.min_len, max_len=a.max_len,
                   max_replacement_ratio=a.max_replacement_ratio, min_guj_ratio=a.min_guj_ratio)
    for s in SOURCES:
        if not a.only or s["name"] in a.only:
            process(s, a.root, filters, a.shard_size, a.log_every)

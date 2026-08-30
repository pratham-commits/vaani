"""
build_corpus.py — Phase-1 Gujarati fluency corpus builder for the Vaani SLM.

Streams the human-verified Gujarati (native-script) split of AI4Bharat's
Sangraha corpus, applies rule-based quality filters, and writes gzipped
JSONL shards. Also emits a run manifest for reproducibility.

Source dataset : ai4bharat/sangraha  (HF, CC-BY-4.0)
Paper          : IndicLLMSuite, arXiv:2403.06350
Subset used    : verified/guj  (Gujarati, Gujr script)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import unicodedata
from datetime import datetime, timezone

from datasets import load_dataset

# Gujarati Unicode block (U+0A80–U+0AFF).
GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")


def gujarati_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are Gujarati script."""
    non_space = re.sub(r"\s", "", text)
    if not non_space:
        return 0.0
    return len(GUJARATI_RE.findall(text)) / len(non_space)


def is_good(text: str, *, min_len: int, max_len: int,
            max_replacement_ratio: float, min_guj_ratio: float) -> bool:
    """Return True if the (already normalized) text passes all filters."""
    if not (min_len <= len(text) <= max_len):
        return False
    if text.count("\ufffd") / len(text) > max_replacement_ratio:
        return False
    if gujarati_ratio(text) < min_guj_ratio:
        return False
    return True


def write_shard(buffer: list[str], out_dir: str, name: str, index: int) -> str:
    """Write one gzipped JSONL shard; return its path."""
    path = os.path.join(out_dir, f"{name}_{index:04d}.jsonl.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for text in buffer:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({len(buffer):,} docs)", flush=True)
    return path


def build(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[config] {vars(args)}", flush=True)

    ds = load_dataset(
        args.dataset, data_dir=args.data_dir, name=args.config,
        split=args.split, streaming=True,
    )

    kept = dropped = seen = shard_idx = 0
    buffer: list[str] = []
    filters = dict(
        min_len=args.min_len, max_len=args.max_len,
        max_replacement_ratio=args.max_replacement_ratio,
        min_guj_ratio=args.min_guj_ratio,
    )

    try:
        for row in ds:
            seen += 1
            text = unicodedata.normalize("NFKC", (row.get(args.key) or "").strip())
            if is_good(text, **filters):
                buffer.append(text)
                kept += 1
                if len(buffer) >= args.shard_size:
                    write_shard(buffer, args.out_dir, args.name, shard_idx)
                    shard_idx += 1
                    buffer = []
            else:
                dropped += 1
            if seen % args.log_every == 0:
                print(f"  {args.name}: seen {seen:,} | kept {kept:,} | dropped {dropped:,}", flush=True)
    except KeyboardInterrupt:
        print("\n[interrupted] flushing buffer...", flush=True)
    finally:
        if buffer:
            write_shard(buffer, args.out_dir, args.name, shard_idx)
            shard_idx += 1

    manifest = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "split": args.split,
        "filters": filters,
        "seen": seen,
        "kept": kept,
        "dropped": dropped,
        "shards": shard_idx,
        "shard_size": args.shard_size,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(args.out_dir, f"{args.name}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[done] {args.name}: kept {kept:,}, dropped {dropped:,}, shards {shard_idx}", flush=True)
    print(f"[manifest] {manifest_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Gujarati fluency corpus.")
    p.add_argument("--dataset", default="ai4bharat/sangraha")
    p.add_argument("--data-dir", default="verified/guj")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--key", default="text")
    p.add_argument("--name", default="guj")
    p.add_argument("--out-dir", default=os.path.expanduser("~/corpus"))
    p.add_argument("--shard-size", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=100_000)
    p.add_argument("--min-len", type=int, default=200)
    p.add_argument("--max-len", type=int, default=100_000)
    p.add_argument("--max-replacement-ratio", type=float, default=0.001)
    p.add_argument("--min-guj-ratio", type=float, default=0.70)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())

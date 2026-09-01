#!/usr/bin/env python3
"""
dedup_corpus.py — Phase 2 dedup + export for the Vaani Gujarati corpus.

Two stages in one file:
  1. DEDUP   MinHash-LSH near-duplicate removal via text-dedup (subprocess).
  2. EXPORT  Convert the deduped Arrow dataset back to gzipped JSONL shards.

Run (in tmux):
  python3 dedup_corpus.py \
    --input-glob "$HOME/dedup_in/*.jsonl.gz" \
    --arrow-out  "$HOME/dedup_out" \
    --jsonl-out  "$HOME/dedup_jsonl"

Then upload:
  gcloud storage cp "$HOME/dedup_jsonl"/*.jsonl.gz "$BUCKET/dedup_guj/"

Re-export only (dedup already done):
  python3 dedup_corpus.py --skip-dedup
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import time

from datasets import load_from_disk


def run_dedup(input_glob, arrow_out, ngram, num_perm, threshold, batch_size):
    """Stage 1: MinHash-LSH dedup via the text-dedup CLI."""
    cmd = [
        sys.executable, "-m", "text_dedup.minhash",
        "--path", "json",
        "--data_files", input_glob,
        "--split", "train",
        "--column", "text",
        "--output", arrow_out,
        "--ngram", str(ngram),
        "--num_perm", str(num_perm),
        "--threshold", str(threshold),
        "--batch_size", str(batch_size),
    ]
    print("[dedup] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)   # text-dedup prints Before/After counts itself
    print("[dedup] done ->", arrow_out, flush=True)


def export_jsonl(arrow_out, jsonl_out, shard_size, compresslevel):
    """Stage 2: Arrow dataset -> gzipped JSONL shards (batched, fast)."""
    ds = load_from_disk(os.path.expanduser(arrow_out))
    out = os.path.expanduser(jsonl_out)
    os.makedirs(out, exist_ok=True)

    n = len(ds)
    n_shards = (n + shard_size - 1) // shard_size
    print(f"[export] {n:,} docs -> {n_shards} shards in {out}", flush=True)

    t0 = time.time()
    shard_idx = 0
    for start in range(0, n, shard_size):
        texts = ds[start:start + shard_size]["text"]        # fast columnar slice
        path = os.path.join(out, f"dedup_{shard_idx:05d}.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=compresslevel) as f:
            f.write("\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts))
            f.write("\n")
        shard_idx += 1

        done = min(start + shard_size, n)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0
        eta = (n - done) / rate if rate else 0
        print(f"[export] shard {shard_idx:>3} | {done:>9,}/{n:,} "
              f"({100 * done / n:5.1f}%) | {rate:>7,.0f} docs/s "
              f"| elapsed {elapsed / 60:4.1f}m | eta {eta / 60:4.1f}m", flush=True)

    print(f"[export] DONE: {shard_idx} shards, {n:,} docs "
          f"in {(time.time() - t0) / 60:.1f}m -> {out}", flush=True)


def parse_args():
    home = os.path.expanduser("~")
    p = argparse.ArgumentParser(description="Vaani corpus dedup + JSONL export")
    p.add_argument("--input-glob", default=f"{home}/dedup_in/*.jsonl.gz",
                   help="glob of input .jsonl.gz shards to dedup")
    p.add_argument("--arrow-out", default=f"{home}/dedup_out",
                   help="text-dedup Arrow output dir")
    p.add_argument("--jsonl-out", default=f"{home}/dedup_jsonl",
                   help="gzipped JSONL export dir")
    p.add_argument("--ngram", type=int, default=5, help="shingle size (words)")
    p.add_argument("--num-perm", type=int, default=128, help="MinHash permutations")
    p.add_argument("--threshold", type=float, default=0.8, help="Jaccard threshold")
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--shard-size", type=int, default=100_000, help="docs per JSONL shard")
    p.add_argument("--compresslevel", type=int, default=1, help="gzip level (1=fast)")
    p.add_argument("--skip-dedup", action="store_true",
                   help="skip stage 1; export an existing --arrow-out")
    return p.parse_args()


def main():
    a = parse_args()
    if a.skip_dedup:
        print("[dedup] skipped (--skip-dedup); exporting existing", a.arrow_out, flush=True)
    else:
        run_dedup(a.input_glob, a.arrow_out, a.ngram, a.num_perm, a.threshold, a.batch_size)
    export_jsonl(a.arrow_out, a.jsonl_out, a.shard_size, a.compresslevel)


if __name__ == "__main__":
    main()

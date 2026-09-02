#!/usr/bin/env python3
"""
tokenize_pack.py — tokenize clean_guj and pack into flat uint16 token streams.

- loads a trained tokenizer (default tokenizer_bpe_char.json)
- tokenizes ALL clean_guj docs in parallel (all cores), appends <eos> per doc
- stratified held-out: every Nth doc (--val-frac) -> val stream (perplexity eval)
- writes train.bin + val.bin (flat uint16, nanoGPT-style) + pack_manifest.json (token counts)

NOTE: MedMCQA decontamination is a separate later step (drops a tiny number of docs).
"""
import argparse, glob, gzip, json, os, shutil, time
import numpy as np
from multiprocessing import Pool
from tokenizers import Tokenizer

_TOK = None
_EOS = None
_VAL_STEP = 0


def _init(tok_path, val_step):
    global _TOK, _EOS, _VAL_STEP
    _TOK = Tokenizer.from_file(tok_path)
    _EOS = _TOK.token_to_id("<eos>")
    _VAL_STEP = val_step


def _tok_shard(job):
    idx, shard, tmpdir = job
    tp = os.path.join(tmpdir, f"train_{idx:05d}.bin")
    vp = os.path.join(tmpdir, f"val_{idx:05d}.bin")
    ftr, fva = open(tp, "wb"), open(vp, "wb")
    ntr = nva = 0
    batch, flags = [], []

    def flush():
        nonlocal ntr, nva
        if not batch:
            return
        encs = _TOK.encode_batch(batch)
        tr, va = [], []
        for e, is_val in zip(encs, flags):
            (va if is_val else tr).extend(e.ids)
            (va if is_val else tr).append(_EOS)
        if tr:
            np.array(tr, dtype=np.uint16).tofile(ftr); ntr += len(tr)
        if va:
            np.array(va, dtype=np.uint16).tofile(fva); nva += len(va)
        batch.clear(); flags.clear()

    i = 0
    with gzip.open(shard, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                txt = json.loads(line)["text"]
            except Exception:
                continue
            batch.append(txt)
            flags.append(_VAL_STEP > 0 and i % _VAL_STEP == 0)
            i += 1
            if len(batch) >= 2000:
                flush()
        flush()
    ftr.close(); fva.close()
    return idx, ntr, nva


def _concat(parts, out_path):
    with open(out_path, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1024 * 1024)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="~/clean_guj")
    p.add_argument("--tokenizer", default="~/tokenizer/tokenizer_bpe_char.json")
    p.add_argument("--out-dir", default="~/packed")
    p.add_argument("--val-frac", type=float, default=0.001,
                   help="fraction of docs held out for perplexity val")
    p.add_argument("--procs", type=int, default=os.cpu_count())
    a = p.parse_args()

    inp = os.path.expanduser(a.input)
    out = os.path.expanduser(a.out_dir); os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "tmp"); os.makedirs(tmp, exist_ok=True)
    tok_path = os.path.expanduser(a.tokenizer)
    shards = sorted(glob.glob(os.path.join(inp, "*.jsonl.gz")))
    if not shards:
        raise SystemExit(f"no shards in {inp}")
    val_step = int(round(1 / a.val_frac)) if a.val_frac > 0 else 0
    print(f"{len(shards)} shards | tokenizer {os.path.basename(tok_path)} | "
          f"val every {val_step} docs | {a.procs} procs", flush=True)

    jobs = [(i, sh, tmp) for i, sh in enumerate(shards)]
    t0 = time.time()
    results = []
    with Pool(a.procs, initializer=_init, initargs=(tok_path, val_step)) as pool:
        for idx, ntr, nva in pool.imap_unordered(_tok_shard, jobs):
            results.append((idx, ntr, nva))
            print(f"[{len(results)}/{len(shards)}] shard {idx} done "
                  f"| train+={ntr:,} val+={nva:,} | {(time.time()-t0)/60:.1f}m", flush=True)

    results.sort()
    _concat([os.path.join(tmp, f"train_{i:05d}.bin") for i, _, _ in results],
            os.path.join(out, "train.bin"))
    _concat([os.path.join(tmp, f"val_{i:05d}.bin") for i, _, _ in results],
            os.path.join(out, "val.bin"))

    total_train = sum(n for _, n, _ in results)
    total_val = sum(n for _, _, n in results)
    manifest = {
        "step": "tokenize_pack",
        "date": time.strftime("%Y-%m-%d"),
        "tokenizer": os.path.basename(tok_path),
        "dtype": "uint16",
        "eos_between_docs": True,
        "val_frac": a.val_frac,
        "train_tokens": total_train,
        "val_tokens": total_val,
        "total_tokens": total_train + total_val,
        "train_bin_gb": round(total_train * 2 / 1e9, 2),
        "val_bin_gb": round(total_val * 2 / 1e9, 2),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(out, "pack_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n=== PACK DONE ===", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

# Vaani Gujarati SLM : Run Log

## Config
- PROJECT_ID = <your-gcp-project-id>
- ZONE       = <your-zone>            # e.g. asia-south1-c
- REGION     = <your-region>          # e.g. asia-south1
- BUCKET     = gs://<your-bucket>     # corpus storage

## Environment
- VM: vaani-prep, e2-standard-8 (8 vCPU, 32 GB RAM), no GPU
- Zone: $ZONE
- OS: Ubuntu 22.04
- Python: 3.10.12
- venv: ~/.venv
- Build package versions: datasets 5.0.1, huggingface_hub 1.29.0, tokenizers 0.22.2, numpy 2.2.6, scipy 1.15.3

## Datasets used (all Gujarati, native script)
- ai4bharat/sangraha (CC-BY-4.0, arXiv:2403.06350): verified/guj, unverified/guj, synthetic/guj_Gujr
- wikimedia/wikipedia (CC-BY-SA): 20231101.gu
- HuggingFaceFW/fineweb-2 (ODC-BY): guj_Gujr   [replaced cc100 - script-based, unsupported on datasets 5.x]

## Step 1.1 - Verified corpus build (29 Aug 2026 eve - 30 Aug 2026 ~08:00 IST)
- Script: ~/build_corpus.py
- data_dir="verified/guj"; ran inside tmux session "prep": python3 ~/build_corpus.py

Filters (doc kept only if all pass):
- Unicode NFKC normalization
- length: 200 <= chars <= 100000
- replacement char (U+FFFD) ratio < 0.001
- Gujarati script (U+0A80-U+0AFF) ratio over non-space chars >= 0.70

Output: gzipped JSONL, {"text": ...} per line, UTF-8; shard size 100000 -> NNNN.jsonl.gz

Logs (sample + final):
```
guj: seen 100,000 | kept 98,188 | dropped 1,812
[done] guj: kept 3,896,437, dropped 73,660, shards 39
```
Results: seen 3,970,097 | kept 3,896,437 (98.1%) | 39 shards | 4.9 GB

## Step 1.1 - Multi-source corpus build (30 Aug 2026)
- Script: ~/build_corpus.py (all sources); same filters as Step 1
- Per-source results:
```
sangraha_unverified : seen 586,977   | kept 582,021   | dropped 4,956   | 6 shards
sangraha_synthetic  : seen 5,603,976 | kept 4,977,140 | dropped 626,836 | 50 shards  (8M cap not reached)
wikipedia_gu        : seen 30,445    | kept 29,251    | dropped 1,194   | 1 shard
fineweb2_gu         : seen 2,127,094 | kept 2,094,479 | dropped 32,615  | 21 shards
```
- Corpus totals (all 5 sources incl. verified):
  - kept docs: 11,579,328
  - real 6,602,188 (57%) / synthetic 4,977,140 (43%)  -> real is the majority
  - est. ~11-12B tokens
- total shards = 39 + 50 + 21 + 6 + 1 = 117

## Step 1.2 - Store multi-source corpus to GCS (31 Aug 2026)
- gcloud storage cp -r ~/corpus_raw $BUCKET/corpus_raw/   (78 new shards)
- Bucket total: 117 jsonl.gz (39 verified + 78 new)
- Paths: verified at $BUCKET/sangraha_verified_guj/ ; other 4 under $BUCKET/corpus_raw/... (nested prefix - cosmetic)

## Step 1.3 – Deduplication (MinHash-LSH) (01 Sep 2026)
Tool: text-dedup 0.4.1 (datasketch MinHash + LSH), Python 3.10, on e2-highmem-16
(16 vCPU / 128 GB RAM), 500 GB pd-balanced disk, zone asia-south1-c.

Input:  ~/dedup_in/*.jsonl.gz  = 117 shards, 11,579,328 docs (full multi-source corpus)
Output: ~/dedup_out (Arrow)    = 10,429,760 docs

Command:
```
python -m text_dedup.minhash \
  --path json --data_files "$HOME/dedup_in/*.jsonl.gz" \
  --split train --column text --output "$HOME/dedup_out" \
  --ngram 5 --num_perm 128 --threshold 0.8 --batch_size 10000
```

Params: 5-word shingles, num_perm=128, Jaccard threshold=0.80.
Result: Before 11,579,328 -> After 10,429,760  (removed 1,149,568 = 9.93% near-dups).
Edges (duplicate links): 2,424,668.
Timings: Loading 45s, MinHashing 21s, Clustering 945s, Filtering 654s, Saving 315s;
         Total 1,989s (~33 min).  [fingerprint cache reused from a prior run]

Export (Arrow -> gzipped JSONL): ~/export_dedup.py, batched columnar read,
gzip compresslevel=1, 100k docs/shard. Wrote 105 shards, 10,429,760 docs, 23 GB, 39.5 min.

Stored: gcloud storage cp ~/dedup_jsonl/*.jsonl.gz $BUCKET/dedup_guj/  (105 shards verified)

Incidents (for reproducibility):
- First run OOM'd during clustering on 32 GB VM -> resized to e2-highmem-16 (128 GB).
- Save failed with "No space left on device" (340 GB disk full: 254 GB HF json cache
  + 50 GB partial output) -> grew disk to 500 GB (growpart + resize2fs), re-ran; the
  fingerprint cache made the re-run skip the ~3 h MinHashing step.


## Step 1.4 – Cleaning pass (02 Sep 2026)
Script: clean_corpus1.py (two-pass, shape-aware boilerplate), e2-standard-8, ~7h.
Input : dedup_guj = 10,429,760 docs (104 shards)
Output: clean_guj = 10,372,946 docs kept (99.46%), 56,814 dropped (0.54%),
        876,582 boilerplate/menu lines stripped, 104 shards.

Pass 1: count SHORT + JUNK-SHAPED lines across corpus -> boilerplate set
        (BOILERPLATE_MIN_COUNT=100; 1,095 boilerplate lines found). Shape-aware,
        so Gujarati section headings (ઈતિહાસ., સંદર્ભો., ...) are NOT stripped;
        only markup / nav / © / URL / Latin-dominant junk is.
        Examples caught: "This website follows the DNPA Code of Ethics.",
        "Copyright © 2022-23 DB Corp ltd.", "View this post on Instagram",
        "- News18 Gujarati", "You must be logged in to post a comment.".
Pass 2: normalize (NFKC, control-char strip keeping ZWJ/ZWNJ), strip boilerplate
        + menu lines, then doc-level filters.

Drops by rule: top_3gram 26309, top_line 12618, dup_lines 12541, high_symbol 2763,
               too_short 2413, bad_word_len 117, few_words 46, low_guj 7.
Thresholds: MIN_LEN 200, MIN_WORDS 20, MIN_GUJ_RATIO 0.70, MAX_DIGIT_RATIO 0.50,
            MAX_SYMBOL_RATIO 0.10, MEAN_WORD_LEN [2,20], MAX_DUP_LINE_FRAC 0.30,
            MAX_TOP_LINE_FRAC 0.20, MAX_TOP_3GRAM_FRAC 0.20.
Gujarati-safe: danda (।) + period kept as terminators; Gujarati block = letters;
            ZWJ/ZWNJ preserved; in-context English kept, English-dominant junk dropped.
Stored: gcloud storage cp ~/clean_guj/*.jsonl.gz $BUCKET/clean_guj/  (104 shards + manifest)

## Step 1.5 – English/Gujarati composition check (02 Sep 2026)
Script: measure_english.py (regex char-count pass over clean_guj), e2-standard-8, ~126 min.
Purpose: quantify English (Latin) content in the cleaned corpus before locking it.

Result (english_report.json):
- docs: 10,372,946 (100% contain Gujarati)
- corpus Gujarati: 94.4% of chars | corpus Latin: 0.62% of chars
- docs with any English: 34.5%; 65.5% pure Gujarati, ~96% under 5% Latin, ~1.5% above 10%.
- English "word" [A-Za-z0-9]+: 174.5M total, 1.34M unique, avg 16.8/doc
  (unique + avg are number-inflated by the digit-inclusive definition).
Conclusion: English is thin, natural code-switching (0.62% of chars) -> KEPT in corpus.
Stored: gcloud storage cp ~/english_report.json $BUCKET/clean_guj/

## Step 1.6 – Tokenizer training (02 Sep 2026)
Script: train_tokenizer.py (HF tokenizers 0.22.2), e2-standard-8, venv.
Common settings: vocab 32K, NFKC, ~1B-char sample from clean_guj (103 shards),
5000 held-out docs (clean_00103 = fineweb2). Metric: fertility (tokens/word, lower better).

| config    | model   | encoding    | pre-tokenizer            | tok/word ↓ | char/tok ↑ | train min |
|-----------|---------|-------------|--------------------------|-----------|-----------|-----------|
| bpe_byte  | BPE     | byte-level  | LLaMA-4 regex + ByteLevel| 2.8132    | 2.10      | 3.1       |
| uni_byte  | Unigram | byte-level  | LLaMA-4 regex + ByteLevel| 2.9322    | 2.02      | 4.2       |
| bpe_char  | BPE     | char-level  | Metaspace                | 1.3028    | 4.54      | 3.8       |  ← CHOSEN
| uni_char  | Unigram | char-level  | Metaspace                | 1.3490    | 4.38      | 26.2      |

Sanity check (bpe_char): "ગુજરાત ભારતના પશ્ચિમ ... છે." → 17 tokens / 17 words (1.0),
conjuncts (પશ્ચિમ, વિજ્ઞાન, ક્ષેત્રે, રહ્યું) intact.
Winner: bpe_char (char-level BPE, NFKC, Metaspace, 32K) — 1.30 tok/word, beats multilingual
SOTA on gu (MUTANT 1.77, Sutra 2.15). Saved: tokenizer/tokenizer_bpe_char.json.

## Step 1.7 – Tokenize + pack (03 Sep 2026)
Script: tokenize_pack.py (multiprocess, 8 cores, HF tokenizers 0.22.2), e2-standard-8, venv.
Tokenizer: tokenizer_bpe_char.json (char-level BPE, 32K). <eos> between docs. dtype uint16.
Held-out: every 1000th doc (val-frac 0.001) -> val.bin (perplexity eval).

Results (pack_manifest.json):
- train_tokens : 6,779,391,293  (~6.78B)  -> train.bin, 13.56 GB
- val_tokens   :     6,831,554  (~6.83M)  -> val.bin, 0.01 GB
- total_tokens : 6,786,222,847  (~6.79B)
- runtime      : 62.1 min  (~71M train tokens/shard; per-shard log in analysis/pack_log.txt)

Token budget: ~62 tokens/param at 1 epoch (110M) -> plan 2-3 epochs (124-185 tok/param).
Stored: gcloud storage cp ~/packed/{train.bin,val.bin,pack_manifest.json} $BUCKET/packed/

### 2.1 Model + trainer code (2026-09-03)
- model.py (Llama-style ~109.5M), train.py (WSD loop, vectorized memmap loader).
- Sanity: overfit-one-batch loss 10.53 → 2.18 over 200 steps (learns). ✓
- Proxy: 40k tok/s @ 22% MFU (stable across batch 4/8/16 → memory-BW bound).

### 2.3 Base pretraining launch (2026-09-03)
- tmux session `train`; checkpoint auto-sync to GCS every 30 min (session `sync`).
- Command: python3 -u train1.py --batch-size 8 --grad-accum 64 --compile
  --max-steps 13000 --warmup 700 --lr 4e-4 --min-lr 4e-5 --decay-frac 0.2
  --eval-interval 1000 --ckpt-interval 2000
- Start: step 0 loss 10.53, 40k tok/s, 22% MFU, gnorm ~2. Healthy.
- ETA ~3.9 days. Checkpoints: ckpt/ckpt_{step}.pt + GCS checkpoints/.

### 2.1 – Throughput / MFU tuning (2026-09-03)
Goal: find the fastest stable config on a single L4 before committing to the full run.
Method: `--overfit-one-batch` for correctness, then short `--proxy-steps` runs to measure tok/s + MFU.

- Overfit-one-batch sanity: loss 10.53 → 2.18 over ~200 steps → model + backprop wired correctly.
- Proxy throughput sweep (micro-batch × options):
  | config                      | tok/s   | MFU    | notes                          |
  |-----------------------------|---------|--------|--------------------------------|
  | B4                          | ~27,000 | ~15%   | data/launch-bound              |
  | B8                          | ~25,000 | —      | no faster than B4              |
  | B8 + torch.compile          | 39,800  | 21.8%  | compile ~1.6x                  |
  | B16 + compile               | 40,600  | 19.4%  | 19.4 GB VRAM, no tok/s gain    |
  | B16 + compile + vec loader  | 40,900  | —      | 10.5 GB VRAM (memmap fix)      |
  | + removed per-step .item()  | ~40,000 | ~21.6% | GPU-sync removal, no gain      |

- Finding: throughput plateaus at ~40k tok/s regardless of batch size → the run is
  **memory-bandwidth bound, not compute bound**. Verified against L4 specs (300 GB/s BW, ~120 TFLOPS bf16): for a 110M model, ~22% MFU is thenrealistic ceiling on L4. The 40–60% MFU figures quoted for LLM training apply to large models on A100/H100, not small models on L4.
- Incident: CUDA OOM at micro-batch 16 (full [16,2048,32000] fp32 logits materialized in
  cross_entropy) → settled on micro-batch 8 + grad-accum 64 + `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- Chosen config: **B8 × grad-accum 64 + compile** = 1,048,576 tokens / optimizer step.

### 2.1 – Model architecture & rationale (model.py, ~109.5M params)
Llama-style decoder. Config (VaaniConfig):
`vocab_size=32000, n_layer=12, n_head=12, n_kv_head=12, d_model=768, head_dim=64,
ffn_mult=8/3 (hidden=2048), block_size=2048, rope_theta=10000, norm_eps=1e-5, tie_embeddings=True`.

Design choices and why:
- **Pre-norm RMSNorm** (computed in fp32): cheaper than LayerNorm (no mean/bias), and
  pre-norm gives stable gradients at depth — confirmed by the flat gnorm ~0.19 late in training.
- **RoPE (theta 10000)**: rotary position encoding, no learned position table (saves params),
  better length behavior than absolute embeddings.
- **SwiGLU MLP, ffn_mult 8/3**: the 8/3 multiplier (hidden=2048) keeps the 3-matrix SwiGLU
  at roughly the same param count as a 4× GELU MLP, while training better.
- **Full MHA (n_kv_head = n_head = 12)**: GQA’s KV-cache savings are irrelevant at 110M, so
  we use full multi-head attention for slightly higher quality. Attention via
  `F.scaled_dot_product_attention(is_causal=True)`.
- **Tied embeddings**: input embedding = output projection. At 32K×768 = 24.6M params, tying
  removes a duplicate 24.6M-param matrix — a large fraction of a 110M budget — and regularizes.
- **No biases anywhere** (Llama/PaLM finding: biases are unnecessary and slightly hurt).
- **Scaled init**: 0.02 base, output projections (wo, w2) scaled by 1/sqrt(2·n_layer) so
  residual variance doesn’t grow with depth.
- **Sizing**: d_model 768 / 12 layers / 12 heads (head_dim 64) is the standard ~110M shape,
  matched to the SLM++ bootcamp target of ~110M params.

### 2.1 – Base pretraining results (completed 2026-09-08)

Totals:
- Steps: 13,000 | tokens processed: 13.63B | epochs over train.bin: 2.01 | ~124 tok/param (overtrained vs Chinchilla 20, intentional for an SLM).
- Wall-clock compute ~4.0 days on 1× L4; steady ~39.5k tok/s, MFU ~21.6%.

Convergence (from train_full.log):
- Loss 10.53 → 3.3494 (last logged step 12990). LR decayed cleanly to 4.00e-05 floor; gnorm settled ~0.19 (from ~2 at start). No divergence, no gradient explosion.
- Val perplexity (selected evals): step 1000 ≈67 → 2000 ≈47 → 3000 41.46 → 4000 38.13 →
  5000 36.05 → 8000 33.94 → 11000 31.19 → 12000 30.58 (train_loss 3.3795 / val_loss 3.4203).
  No step-13000 eval was run; final val_ppl estimated ~28–29 from continued loss decline.
  (Authoritative full series: `grep "eval" logs/train_full.log`.)
- **No overfitting**: train ≈ val throughout.

Generation validation (generate1.py):
- Fluent, grammatical Gujarati with correct morphology and discourse connectors; article-like
  structure (headers, lists) reflecting the news/blog-heavy corpus.
- Repetition loops appear under greedy/low-temp sampling; **eliminated** by repetition penalty
  + top-p (top-k alone insufficient). Long-range topic drift and factual confabulation persist —
  expected 110M-scale limits, to be bounded by SFT.
- Locked eval sampling config: `--temperature 0.8 --top-p 0.9 --repetition-penalty 1.15`.

Artifacts:
- Checkpoints (GCS): ckpt_{2000,4000,6000,8000,10000,12000}.pt + ckpt_final.pt (step 13000).
- Logs (GCS + repo): train_20260903_0917.log, train_resume.log, merged train_full.log.
- Code: model.py, train.py, generate.py.

## Status (2026-09-08)
Done: corpus build → dedup → clean → English check → tokenizer → tokenize/pack →
base pretraining (13k steps, ckpt_final.pt) → generation validation.
Next: SFT (instruction tuning) → eval harness (difficulty-weighted MCQA) → medical fine-tune
(MedMCQA-Indic, decontaminated).


# Vaani Gujarati SLM — Run Log

## Step 1.1 — Setup & data sources

### Config
- PROJECT_ID = <your-gcp-project-id>
- ZONE       = <your-zone>            # e.g. asia-south1-c
- REGION     = <your-region>          # e.g. asia-south1
- BUCKET     = gs://<your-bucket>     # corpus storage

### Environment
- VM: vaani-prep, e2-standard-8 (8 vCPU, 32 GB RAM), no GPU
- Zone: $ZONE
- OS: Ubuntu 22.04
- Python: 3.10.12
- venv: ~/.venv
- Build package versions: datasets 5.0.1, huggingface_hub 1.29.0, tokenizers 0.22.2, numpy 2.2.6, scipy 1.15.3

### Datasets used (all Gujarati, native script)
- ai4bharat/sangraha (CC-BY-4.0, arXiv:2403.06350): verified/guj, unverified/guj, synthetic/guj_Gujr
- wikimedia/wikipedia (CC-BY-SA): 20231101.gu
- HuggingFaceFW/fineweb-2 (ODC-BY): guj_Gujr   [replaced cc100 — script-based, unsupported on datasets 5.x]

## Step 1.2 — Corpus build (29 Aug – 31 Aug 2026)
Script: ~/build_corpus.py  (file: src/1.2_build_corpus.py)
Ran inside tmux session "prep": python3 ~/build_corpus.py

Filters (doc kept only if all pass):
- Unicode NFKC normalization
- length: 200 <= chars <= 100000
- replacement char (U+FFFD) ratio < 0.001
- Gujarati script (U+0A80-U+0AFF) ratio over non-space chars >= 0.70
Output: gzipped JSONL, {"text": ...} per line, UTF-8; shard size 100000 -> NNNN.jsonl.gz

Verified source (29 Aug eve – 30 Aug ~08:00 IST):
```
guj: seen 100,000 | kept 98,188 | dropped 1,812
[done] guj: kept 3,896,437, dropped 73,660, shards 39
```
Result: seen 3,970,097 | kept 3,896,437 (98.1%) | 39 shards | 4.9 GB

Remaining sources (30 Aug), same filters:
```
sangraha_unverified : seen 586,977   | kept 582,021   | dropped 4,956   | 6 shards
sangraha_synthetic  : seen 5,603,976 | kept 4,977,140 | dropped 626,836 | 50 shards  (8M cap not reached)
wikipedia_gu        : seen 30,445    | kept 29,251    | dropped 1,194   | 1 shard
fineweb2_gu         : seen 2,127,094 | kept 2,094,479 | dropped 32,615  | 21 shards
```
Corpus totals (all 5 sources incl. verified):
- kept docs: 11,579,328
- real 6,602,188 (57%) / synthetic 4,977,140 (43%) -> real is the majority
- est. ~11-12B tokens
- total shards = 39 + 50 + 21 + 6 + 1 = 117

Store to GCS (31 Aug): gcloud storage cp -r ~/corpus_raw $BUCKET/corpus_raw/ (78 new shards).
Bucket total: 117 jsonl.gz (39 verified + 78 new). Paths: verified at $BUCKET/sangraha_verified_guj/;
other 4 under $BUCKET/corpus_raw/... (nested prefix — cosmetic).

## Step 1.3 — Deduplication (MinHash-LSH) (01 Sep 2026)
Tool: text-dedup 0.4.1 (datasketch MinHash + LSH), Python 3.10, on e2-highmem-16
(16 vCPU / 128 GB RAM), 500 GB pd-balanced disk, zone asia-south1-c.  (file: src/1.3_dedup_corpus.py)

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
Result: Before 11,579,328 -> After 10,429,760 (removed 1,149,568 = 9.93% near-dups). Edges: 2,424,668.
Timings: Loading 45s, MinHashing 21s, Clustering 945s, Filtering 654s, Saving 315s; Total 1,989s (~33 min).
Export (Arrow -> gzipped JSONL): ~/export_dedup.py, batched columnar read, gzip level 1, 100k docs/shard.
Wrote 105 shards, 10,429,760 docs, 23 GB, 39.5 min.
Stored: gcloud storage cp ~/dedup_jsonl/*.jsonl.gz $BUCKET/dedup_guj/ (105 shards verified).

Incidents (for reproducibility):
- First run OOM'd during clustering on 32 GB VM -> resized to e2-highmem-16 (128 GB).
- Save failed with "No space left on device" (340 GB disk full: 254 GB HF json cache + 50 GB partial
  output) -> grew disk to 500 GB (growpart + resize2fs), re-ran; fingerprint cache skipped the ~3 h MinHashing.

## Step 1.4 — Cleaning pass (02 Sep 2026)
Script: clean_corpus1.py (two-pass, shape-aware boilerplate), e2-standard-8, ~7h.  (file: src/1.4_clean_corpus.py)
Input : dedup_guj = 10,429,760 docs (104 shards)
Output: clean_guj = 10,372,946 docs kept (99.46%), 56,814 dropped (0.54%), 876,582 boilerplate/menu
        lines stripped, 104 shards.

Pass 1: count SHORT + JUNK-SHAPED lines across corpus -> boilerplate set (BOILERPLATE_MIN_COUNT=100;
        1,095 boilerplate lines found). Shape-aware, so Gujarati section headings (ઈતિહાસ., સંદર્ભો., ...)
        are NOT stripped; only markup / nav / © / URL / Latin-dominant junk is.
        Examples caught: "This website follows the DNPA Code of Ethics.", "Copyright © 2022-23 DB Corp ltd.",
        "View this post on Instagram", "- News18 Gujarati", "You must be logged in to post a comment.".
Pass 2: normalize (NFKC, control-char strip keeping ZWJ/ZWNJ), strip boilerplate + menu lines, doc filters.

Drops by rule: top_3gram 26309, top_line 12618, dup_lines 12541, high_symbol 2763, too_short 2413,
               bad_word_len 117, few_words 46, low_guj 7.
Thresholds: MIN_LEN 200, MIN_WORDS 20, MIN_GUJ_RATIO 0.70, MAX_DIGIT_RATIO 0.50, MAX_SYMBOL_RATIO 0.10,
            MEAN_WORD_LEN [2,20], MAX_DUP_LINE_FRAC 0.30, MAX_TOP_LINE_FRAC 0.20, MAX_TOP_3GRAM_FRAC 0.20.
Gujarati-safe: danda (।) + period kept as terminators; Gujarati block = letters; ZWJ/ZWNJ preserved;
            in-context English kept, English-dominant junk dropped.
Stored: gcloud storage cp ~/clean_guj/*.jsonl.gz $BUCKET/clean_guj/ (104 shards + manifest).

## Step 1.4 — English/Gujarati composition check (02 Sep 2026)
Script: measure_english.py (regex char-count pass over clean_guj), e2-standard-8, ~126 min.  (file: src/1.4_measure_english.py)
Purpose: quantify English (Latin) content in the cleaned corpus before locking it.

Result (english_report.json):
- docs: 10,372,946 (100% contain Gujarati)
- corpus Gujarati: 94.4% of chars | corpus Latin: 0.62% of chars
- docs with any English: 34.5%; 65.5% pure Gujarati, ~96% under 5% Latin, ~1.5% above 10%.
- English "word" [A-Za-z0-9]+: 174.5M total, 1.34M unique, avg 16.8/doc (number-inflated by the
  digit-inclusive definition).
Conclusion: English is thin, natural code-switching (0.62% of chars) -> KEPT in corpus.
Stored: gcloud storage cp ~/english_report.json $BUCKET/clean_guj/

## Step 1.5 — Tokenizer training (02 Sep 2026)
Script: train_tokenizer.py (HF tokenizers 0.22.2), e2-standard-8, venv.  (file: src/1.5_train_tokenizer.py)
Common settings: vocab 32K, NFKC, ~1B-char sample from clean_guj (103 shards), 5000 held-out docs
(clean_00103 = fineweb2). Metric: fertility (tokens/word, lower better).

| config    | model   | encoding    | pre-tokenizer            | tok/word ↓ | char/tok ↑ | train min |
|-----------|---------|-------------|--------------------------|-----------|-----------|-----------|
| bpe_byte  | BPE     | byte-level  | LLaMA-4 regex + ByteLevel| 2.8132    | 2.10      | 3.1       |
| uni_byte  | Unigram | byte-level  | LLaMA-4 regex + ByteLevel| 2.9322    | 2.02      | 4.2       |
| bpe_char  | BPE     | char-level  | Metaspace                | 1.3028    | 4.54      | 3.8       |  <- CHOSEN
| uni_char  | Unigram | char-level  | Metaspace                | 1.3490    | 4.38      | 26.2      |

Sanity check (bpe_char): "ગુજરાત ભારતના પશ્ચિમ ... છે." -> 17 tokens / 17 words (1.0), conjuncts
(પશ્ચિમ, વિજ્ઞાન, ક્ષેત્રે, રહ્યું) intact.
Winner: bpe_char (char-level BPE, NFKC, Metaspace, 32K) — 1.30 tok/word, beats multilingual SOTA on gu
(MUTANT 1.77, Sutra 2.15). Saved: tokenizer/tokenizer_bpe_char.json.

## Step 1.6 — Tokenize + pack (03 Sep 2026)
Script: tokenize_pack.py (multiprocess, 8 cores, HF tokenizers 0.22.2), e2-standard-8, venv.  (file: src/1.6_tokenizer_pack.py)
Tokenizer: tokenizer_bpe_char.json (char-level BPE, 32K). <eos> between docs. dtype uint16.
Held-out: every 1000th doc (val-frac 0.001) -> val.bin (perplexity eval).

Results (pack_manifest.json):
- train_tokens : 6,779,391,293  (~6.78B)  -> train.bin, 13.56 GB
- val_tokens   :     6,831,554  (~6.83M)  -> val.bin, 0.01 GB
- total_tokens : 6,786,222,847  (~6.79B)
- runtime      : 62.1 min  (~71M train tokens/shard; per-shard log in analysis/pack_log.txt)

Token budget: ~62 tokens/param at 1 epoch (110M) -> plan 2-3 epochs (124-185 tok/param).
Stored: gcloud storage cp ~/packed/{train.bin,val.bin,pack_manifest.json} $BUCKET/packed/

## Step 2.1 — Model architecture, pretraining & results

### 2.1.1  Model architecture & rationale (model.py, ~109.5M params)
Llama-style decoder. Config (VaaniConfig):
`vocab_size=32000, n_layer=12, n_head=12, n_kv_head=12, d_model=768, head_dim=64,
ffn_mult=8/3 (hidden=2048), block_size=2048, rope_theta=10000, norm_eps=1e-5, tie_embeddings=True`.

Design choices and why:
- Pre-norm RMSNorm (fp32): cheaper than LayerNorm, stable gradients at depth — confirmed by flat gnorm ~0.19.
- RoPE (theta 10000): rotary position encoding, no learned position table, better length behavior.
- SwiGLU MLP, ffn_mult 8/3 (hidden=2048): 3-matrix SwiGLU at ~same param count as a 4× GELU MLP, trains better.
- Full MHA (n_kv_head = n_head = 12): GQA's KV-cache savings are irrelevant at 110M, so full multi-head
  attention for slightly higher quality. Attention via F.scaled_dot_product_attention(is_causal=True).
- Tied embeddings: input embedding = output projection. At 32K×768 = 24.6M, tying removes a duplicate
  24.6M-param matrix and regularizes.
- No biases anywhere (Llama/PaLM finding).
- Scaled init: 0.02 base; output projections (wo, w2) scaled by 1/sqrt(2·n_layer).
- Sizing: d_model 768 / 12 layers / 12 heads (head_dim 64) = standard ~110M shape, matched to the
  SLM++ bootcamp target of ~110M params.

### 2.1.2  Throughput / MFU tuning (2026-09-03)
Goal: find the fastest stable config on a single L4 before the full run.
Method: `--overfit-one-batch` for correctness, then short `--proxy-steps` runs for tok/s + MFU.

- Overfit-one-batch sanity: loss 10.53 -> 2.18 over ~200 steps -> model + backprop wired correctly.
- Proxy throughput sweep:
  | config                      | tok/s   | MFU    | notes                          |
  |-----------------------------|---------|--------|--------------------------------|
  | B4                          | ~27,000 | ~15%   | data/launch-bound              |
  | B8                          | ~25,000 | —      | no faster than B4              |
  | B8 + torch.compile          | 39,800  | 21.8%  | compile ~1.6x                  |
  | B16 + compile               | 40,600  | 19.4%  | 19.4 GB VRAM, no tok/s gain    |
  | B16 + compile + vec loader  | 40,900  | —      | 10.5 GB VRAM (memmap fix)      |
  | + removed per-step .item()  | ~40,000 | ~21.6% | GPU-sync removal, no gain      |
- Finding: throughput plateaus at ~40k tok/s regardless of batch size -> the run is memory-bandwidth
  bound, not compute bound. For a 110M model on L4 (300 GB/s BW, ~120 TFLOPS bf16), ~22% MFU is the
  realistic ceiling. The 40–60% MFU figures quoted for LLM training apply to large models on A100/H100.
- Incident: CUDA OOM at micro-batch 16 (full [16,2048,32000] fp32 logits in cross_entropy) -> settled on
  micro-batch 8 + grad-accum 64 + PYTORCH_ALLOC_CONF=expandable_segments:True.
- Chosen config: B8 × grad-accum 64 + compile = 1,048,576 tokens / optimizer step.

### 2.1.3  Base pretraining launch (2026-09-03)
- tmux session `train`; checkpoint auto-sync to GCS every 30 min (session `sync`).
- Command: python3 -u train1.py --batch-size 8 --grad-accum 64 --compile
  --max-steps 13000 --warmup 700 --lr 4e-4 --min-lr 4e-5 --decay-frac 0.2
  --eval-interval 1000 --ckpt-interval 2000
- Start: step 0 loss 10.53, 40k tok/s, 22% MFU, gnorm ~2. Healthy. ETA ~3.9 days.
- Checkpoints: ckpt/ckpt_{step}.pt + GCS checkpoints/.
- NOTE: train1.py is the trainer actually used. train.py was an earlier stale variant, not used.

### 2.1.4  Base pretraining results (completed 2026-09-08)
Totals:
- Steps 13,000 | tokens processed 13.63B | 2.01 epochs over train.bin | ~124 tok/param
  (overtrained vs Chinchilla 20, intentional for an SLM).
- Wall-clock ~4.0 days on 1× L4; steady ~39.5k tok/s, MFU ~21.6%.

Convergence (from train_full.log):
- Loss 10.53 -> 3.3494 (last logged step 12990). LR decayed cleanly to 4.00e-05 floor; gnorm settled
  ~0.19 (from ~2). No divergence, no gradient explosion.
- Val perplexity (selected): step 1000 ≈67 -> 2000 ≈47 -> 3000 41.46 -> 4000 38.13 -> 5000 36.05 ->
  8000 33.94 -> 11000 31.19 -> 12000 30.58 (train_loss 3.3795 / val_loss 3.4203). No step-13000 eval;
  final val_ppl estimated ~28–29 from continued decline. (Full series: grep "eval" logs/train_full.log.)
- No overfitting: train ≈ val throughout.

Generation validation (generate1.py):
- Fluent, grammatical Gujarati with correct morphology and discourse connectors; article-like structure
  reflecting the news/blog-heavy corpus.
- Repetition loops under greedy/low-temp; eliminated by repetition penalty + top-p (top-k alone
  insufficient). Long-range topic drift and factual confabulation persist — expected 110M limits, bounded by SFT.
- Locked eval sampling config: --temperature 0.8 --top-p 0.9 --repetition-penalty 1.15.

Artifacts:
- Checkpoints (GCS): ckpt_{2000,4000,6000,8000,10000,12000}.pt + ckpt_final.pt (step 13000).
- Logs: train_full.log (merged full-run log).
- Code: model.py, train1.py (the trainer used; an earlier train.py variant existed during dev, not used, not included), generate1.py.

## Status (2026-09-08)
Done: corpus build -> dedup -> clean -> English check -> tokenizer -> tokenize/pack ->
base pretraining (13k steps, ckpt_final.pt) -> generation validation.
Next: SFT (instruction tuning) -> eval harness -> medical fine-tune (MedMCQA-Indic, decontaminated).

## Step 3.1 — Instruction SFT data (09-10 Sep 2026)
Base checkpoint for all instruction SFT: ckpt/ckpt_final.pt (pretrained base).
Chat template: `<eos>વપરાશકર્તા:\n{question}\nસહાયક:\n{answer}<eos>` (single template everywhere).

Sources (sft_filtered.jsonl, 110,767 ex) — cleaner Gujarati instruction sets only: Wiki_Chat, Dolly_T,
WikiHow. Anudesh dropped (noisy: hallucination, garbled Gujarati, English contamination) — see D3.1.
Reused clean_corpus1.py heuristics (Gujarati-ratio floor, drop English-dominant, length bounds, dedup).

Augmentation pipeline (files: src/3.1a_augment_sft.py -> 3.1b_augment_sft_v3.py -> 3.1c_augment_format_v4.py):
- sft_augmented.jsonl (127,169): + verified general factoids (one_word/short) + reformatted numbered/
  list_3/list_5 from general detailed answers (clause-level splitting).  [3.1a_augment_sft.py]
- sft_v3.jsonl (140,057): targeted augmentation for under-performing buckets (+~13k).  [3.1b_augment_sft_v3.py]
- sft_v4.jsonl (150,057): + templated format skills, correct-by-construction (no LLM): 4,000 JSON /
  3,000 extraction / 3,000 transformation examples.  [3.1c_augment_format_v4.py]

## Step 3.2 — Instruction SFT training + results (09-10 Sep 2026)
Config (train_sft_v{2,3,4}.py; file: src/3.2_train_instruction_sft.py): from ckpt_final.pt;
micro-batch 8 × grad-accum 4 (eff 32), 3 epochs, LR 2e-5 -> 2e-6 cosine (warmup), AdamW (0.9,0.95)
wd 0.1 fused. ~3.0-3.1 h/run on L4.

Results (MCQ acc_norm n=80 dev; Instruction = 9 rule-based buckets):
| ckpt | MCQ | Instruction | note |
|------|-----|-------------|------|
| v1 (sft_ckpt) | 36.3% | 35.9% | first SFT |
| v2 | 36.3% | 46.0% | |
| v3 (targeted aug) | 31.3% | 68.7% | instruction jump |
| v4 (format skills) | 28.8% | 66.7% (71.2% greedy) | JSON/extract/lists added |
Instruction-following 36% -> ~70%. MCQ dipped slightly (aug shifted mass toward format/brevity;
recovered later by medical SFT). v4 is the instruction base for medical SFT.

## Step 3.3 — Decoding discovery: greedy for objective, sampling for creative (KEY)
Symptom: JSON prompts scored 0/12 — valid structure but garbled 2nd key. Root cause: no_repeat_ngram_size=3
+ repetition_penalty banned the repeated JSON key pattern (`": `), corrupting structured output.
Fix: greedy for structured/objective (temp~0, top_k=1, rep_pen=1.0, no_repeat=0) -> 11/12 (92%).
Sampling still better for brevity/creative (greedy over-elaborates, hurting one_word/short). Rubric
mandates temp 0 for objective items; this split matches it. See D3.3.

## Step 4.1 — Medical SFT data (10 Sep 2026)
build_medical_sft_v3.py -> medical_sft_v3.jsonl (114,285), from medmcqa_gu.jsonl + sft_augmented.jsonl:  (file: src/4.1_build_medical_sft.py)
- closed-book MCQ         38,460
- raw-grounded            27,632  (context contains answer -> teach copy/use)
- redacted-grounded       13,908  (answer-sentences removed at sentence level, >=25char filter
                                    -> teach inference; word-level redaction rejected, breaks grammar)
- general (30% mix)       34,285  (from sft_augmented.jsonl -> anti catastrophic-forgetting)
(38,460 + 27,632 + 13,908 + 34,285 = 114,285.)

## Step 4.2 — Medical SFT training (10 Sep 2026)
Config (train_sft_med_v{3..6}.py; file: src/4.2_train_medical_sft.py): BASE_CKPT = sft_ckpt_v4/sft_final.pt
(stacks on instruction v4), micro-batch 8 × grad-accum 4 (eff 32), 2 epochs (7,070 steps = 3,535/epoch),
LR 1e-5 -> 1e-6, warmup 100, AdamW (0.9,0.95) fused. ~1.0 h/run on L4.

Results (open-book on MedMCQA-gu val; closed -> with-context (raw), lift):
| ckpt | MCQ (n=80) | Instruction | open-book |
|------|-----------|-------------|-----------|
| med-v3 (raw-grounded) | 33.8% | 68.2% | 26.6% -> 30.8% (+4.2) |
| med-v4 | 41.3% | 66.7% | 28.6% -> 38.2% (+9.6) |
| med-v5 (FINAL) | 41.3% | 69.7% | 28.8% -> 44.1% (+15.3) [n=1858] |
| med-v6 (redacted-heavy) | - | - | 28.7% -> 42.8% (+14.1) |
med-v5 locked: open-book rose monotonically to 44.1%; v6's heavier redaction regressed (capacity ceiling
at 110M, not a data bug). med-v5 also = MCQ 38.6% on the n=145 set.

## Step 5.1 — Eval harness (10 Sep 2026)
File: src/5.1_eval_hf.py  (built during the SFT phase; listed here as the evaluation component).
- MCQ: option log-likelihood, per-token normalized (acc_norm); chance = 25% (4 options).
- Instruction: 9 programmatic checkers (one_word<=3w, short<=25w, detailed>=60w, one_sent, max3_sent,
  list_3==3, list_5==5, numbered>=3, pure_guj>=0.99).
- Open-book: closed vs with-context (raw-grounded) vs sentence-redacted; reports leak metric.
- MCQ sets: n=80 dev, n=145 expanded (52E/51M/42H), n=4183 full MedMCQA-gu val.

## Step 5.2 — External benchmarking (10-11 Sep 2026)
File: src/5.2_run_bench.sh (disk-safe runner), same harness. MCQ n=145; open-book external n=500.
Templates: Navarasa/sarvam=alpaca, gemma/Qwen=chat (matching each model's training format — see incident 2).
| model | params | MCQ | Instr | open w/ctx | lift |
|-------|--------|-----|-------|-----------|------|
| Vaani (final) | 110M | 38.6% | 70% | 44.1% | +15.3 |
| gemma-2-2b-it | 2B | 37.2% | 67% | 37.8% | +9.6 |
| sarvam-1 (base) | 2B | 36.6% | 24% | 40.2% | +11.4 |
| Navarasa-2.0 | 2B | 42.8% | 54% | 35.0% | +10.0 |
| Qwen2.5-7B-Instruct | 7B | 51.0% | 53% | 44.0% | +18.0 |
| Navarasa-7B | 7B | OOM — skipped (fp32, 28GB > 16GB host RAM) |
Findings: instruction-following leads all; open-book with-context best-in-test (tops Qwen-7B); MCQ beats
both 2B peers gemma/sarvam, trails Navarasa-2B and Qwen-7B (closed-book recall scales with size). Even
Qwen-7B fails exact-count lists (list_3/5 0%, numbered 14%).

## Step 5.3 — Level-1 log, packaging & publish (11 Sep 2026)
File: src/5.3_make_level1_log.py.
- Level-1 artifact: stitches pretraining logs -> pretrain_log.csv/jsonl (schema step,train_loss,val_loss,
  lr,grad_norm,tokens_seen). Gate: reduction 68.2%, converged, no NaN, val-train gap 2.0% -> PASS.
- GPU-hours (1× L4): pretraining ~96 h + SFT ~10 h (instruction v3 3.0 + v4 3.1; medical v3–v6 ≈ 4.1)
  ≈ 106 h total. Final-lineage path (pretrain -> instruction v4 -> medical v5) ≈ 100 h.
- MODEL_CARD.md (intake form + both eval tables). HF publish:
  model -> https://huggingface.co/pratham-commits/vaani-gujarati-slm
  data  -> https://huggingface.co/pratham-commits/vaani-gujarati-sft-data

## Incidents (for reproducibility)
1. Structured-output collapse under sampling -> greedy fix (Step 3.3).
2. Navarasa scored below chance (21%) with Gemma chat template -> switched to Alpaca -> ~40-54%.
   Lesson: match prompt template to each model's training format.
3. Disk-full catastrophe (11 Sep): overnight benchmark run downloaded 4 large fp32 models, filled the
   96 GB boot disk -> cloud-init could not write SSH host/authorized keys -> "SSH authentication failed"
   + kernel "No space left on device (Errno 28)". Data intact (EXT4 mounted clean; GCS had every
   checkpoint). Recovery WITHOUT SSH: stop VM -> detach boot disk -> attach to a throwaway rescue VM ->
   mount -> rm HF model cache (freed 33 GB) -> reattach. Then L4 STOCKOUT in the home zone -> snapshot
   + rebuild in asia-southeast1-a via a multi-zone retry sweep. Boot disk auto-grew 96->150 GB.
4. NVIDIA driver mismatch: an unattended apt-upgrade bumped the userspace driver (580.173->580.178) while
   the loaded kernel module stayed 580.173 -> "Driver/library version mismatch". Fix: reboot; then
   disabled apt-daily{,-upgrade}.timer + unattended-upgrades on the training VM.
5. 7B OOM on load: fp32 7B needs ~28 GB RAM; host has 16 GB. Added low_cpu_mem_usage=True (Qwen-7B then
   loaded; Navarasa-7B fp32 still too large -> used the 2B Navarasa result instead).
6. Wrong-shell mistake: ran pip install in Cloud Shell (5 GB home) not the VM -> filled Cloud Shell disk.
   Harmless; re-ran on the VM (torch already present, so no giant re-download).
7. Dependency pins: sarvam tokenizer needed protobuf+sentencepiece; huggingface_hub auto-upgraded to 1.31
   (incompatible with transformers 4.57) -> pinned huggingface_hub<1.0.

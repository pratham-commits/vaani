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


## Not yet done
- Deduplication (MinHash-LSH)
- Held-out Gujarati benchmark
- Decontamination vs. benchmark
- Tokenizer training
- Tokenize + pack
- Model training

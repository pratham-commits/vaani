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

## Step 1 - Verified corpus build (29 Aug 2026 eve - 30 Aug 2026 ~08:00 IST)
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

## Step 2 - Store verified corpus to GCS (30 Aug 2026 ~08:00 IST)
- Auth: gcloud auth login on VM (default service account lacked storage scope; 403)
- gcloud storage buckets create $BUCKET --location=$REGION
- gcloud storage cp ~/corpus/*.jsonl.gz $BUCKET/sangraha_verified_guj/   (39 shards, ~764 MiB/s)

## Step 3 - Multi-source corpus build (30 Aug 2026)
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

## Step 4 - Store multi-source corpus to GCS (31 Aug 2026)
- gcloud storage cp -r ~/corpus_raw $BUCKET/corpus_raw/   (78 new shards)
- Bucket total: 117 jsonl.gz (39 verified + 78 new)
- Paths: verified at $BUCKET/sangraha_verified_guj/ ; other 4 under $BUCKET/corpus_raw/... (nested prefix - cosmetic)

## Not yet done
- Deduplication (MinHash-LSH)
- Held-out Gujarati benchmark
- Decontamination vs. benchmark
- Tokenizer training
- Tokenize + pack
- Model training

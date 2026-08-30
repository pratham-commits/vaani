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

## Dataset
- Source: ai4bharat/sangraha (Hugging Face, CC-BY-4.0, arXiv:2403.06350)
- Subset used: verified/guj (Gujarati, native script only)
- Access: load_dataset(..., data_dir="verified/guj", split="train", streaming=True)
- Reported size (guj verified): ~3,647.9M tokens

## Step 1 : Corpus build (29 Aug 2026 evening – 30 Aug 2026 ~08:00 IST)
- Script: ~/build_corpus.py
- First attempt used data_dir="verified/guj" 
- Ran inside tmux session "prep": `python3 ~/build_corpus.py`

Filters (doc kept only if all pass):
- Unicode NFKC normalization
- length: 200 <= chars <= 100000
- replacement char (U+FFFD) ratio < 0.001
- Gujarati script (U+0A80–U+0AFF) ratio over non-space chars >= 0.70

Output format:
- gzipped JSONL, one {"text": ...} per line, UTF-8, ensure_ascii=False
- shard size: 100000 docs -> guj_NNNN.jsonl.gz

Logs (sample + final):
```
guj: seen 100,000 | kept 98,188 | dropped 1,812
guj: seen 3,900,000 | kept 3,827,672 | dropped 72,328
[done] guj: kept 3,896,437, dropped 73,660, shards 39
```

Results:
- rows seen: 3,970,097
- kept: 3,896,437 (98.1%)
- dropped: 73,660
- shards: 39 (guj_0000 – guj_0038; last shard 96,437 docs)
- size on disk: 4.9 GB (~/corpus)

## Not yet done
- Deduplication (MinHash-LSH)
- Held-out Gujarati benchmark
- Decontamination vs. benchmark
- Tokenizer training
- Tokenize + pack
- Model training


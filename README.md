# Vaani — Gujarati Small Language Model (~110M)

Vaani is a ~110M-parameter, from-scratch **Gujarati** decoder-only language model, pretrained
on ~6.78B Gujarati tokens and fine-tuned for **instruction-following** and **medical open-book
(RAG-style) question answering**. Built for the IAIRO / PRAMANA SLM++ bootcamp on a ~$100
single-GPU (NVIDIA L4) budget.

- **Model weights:** https://huggingface.co/pratham-commits/vaani-gujarati-slm
- **SFT datasets:** https://huggingface.co/pratham-commits/vaani-gujarati-sft-data
- **License:** Apache-2.0  ·  **Language:** Gujarati (gu)

## Highlights
- **Tokenizer tuned for Gujarati:** char-level BPE (32K) at 1.30 tokens/word — beats multilingual
  baselines on gu (MUTANT 1.77, Sutra 2.15). Byte-level BPE was tried first and rejected on evidence.
- **Llama-style architecture:** RMSNorm (pre-norm), RoPE, SwiGLU, full multi-head attention,
  tied embeddings, no biases. ~109.5M params.
- **Honest evaluation** against 2B–7B Gujarati-capable models: leads on instruction-following and
  open-book grounding despite being 18–64x smaller.

## Results
MCQ n=145 (acc_norm); Instruction = 9 rule-based buckets; open-book = with-context on MedMCQA-gu.

| Model | Params | MCQ | Instruction | Open-book w/ctx | Lift |
|-------|--------|-----|-------------|-----------------|------|
| **Vaani (final)** | **110M** | 38.6% | **70%** | **44.1%** | +15.3 |
| gemma-2-2b-it | 2B | 37.2% | 67% | 37.8% | +9.6 |
| sarvam-1 (base) | 2B | 36.6% | 24% | 40.2% | +11.4 |
| Navarasa-2.0 | 2B | 42.8% | 54% | 35.0% | +10.0 |
| Qwen2.5-7B-Instruct | 7B | 51.0% | 53% | 44.0% | +18.0 |

Vaani leads instruction-following and open-book with-context (tops even Qwen-7B), and beats both
2B peers on MCQ. It trails larger models on closed-book recall — expected at 110M (closed-book
medical MCQ ~28% is a universal small-model ceiling, not a data defect).

## Repository layout
```
.
├── README.md              # this file
├── RUNLOG.md              # chronological build log (steps 1.2 -> 5.3)
├── DECISIONS.md           # design decisions + rationale, cross-referenced to RUNLOG
├── requirements.txt
├── src/
│   ├── 1.2_build_corpus.py         # multi-source Gujarati corpus build + filters
│   ├── 1.3_dedup_corpus.py         # MinHash-LSH near-deduplication
│   ├── 1.4_clean_corpus.py         # shape-aware cleaning + Gopher repetition filters
│   ├── 1.4_measure_english.py      # English/Gujarati composition audit
│   ├── 1.5_train_tokenizer.py      # char-level BPE (32K) training + A/B
│   ├── 1.6_tokenizer_pack.py       # tokenize + pack to uint16 .bin
│   ├── 2.1_pretrain/
│   │   ├── model.py                # VaaniConfig + model (~109.5M)
│   │   ├── train1.py               # WSD pretraining loop (the trainer actually used)
│   │   └── generate1.py            # sampling / generation utility
│   ├── 3.1a_augment_sft.py         # instruction SFT data: base augmentation
│   ├── 3.1b_augment_sft_v3.py      # targeted bucket augmentation
│   ├── 3.1c_augment_format_v4.py   # templated format skills (JSON / extraction / lists)
│   ├── 3.2_train_instruction_sft.py
│   ├── 4.1_build_medical_sft.py    # medical grounding dataset (raw + sentence-redacted)
│   ├── 4.2_train_medical_sft.py    # medical SFT (stacks on instruction v4)
│   ├── 5.1_eval_hf.py              # eval harness (MCQ / instruction / open-book)
│   ├── 5.2_run_bench.sh            # external-model comparison runner
│   └── 5.3_make_level1_log.py      # Level-1 validity log generator
├── manifests/             # per-step JSON manifests (1.2 ... 2.1)
├── logs/                  # 2.1_train.log (raw pretraining log)
├── analysis/              # 1.4_english_report.json, 1.6_tokenizer_pack_log.txt
├── evals/logs/            # bench_*.txt, results_*.json, pretrain_log.csv/jsonl
└── docs/                  # bootcamp material + reference papers (not tracked)
```

## Pipeline (ordered by step)
Every script is numbered to match `RUNLOG.md`, `DECISIONS.md`, and the git commit history.

| Step | Script | What it does |
|------|--------|--------------|
| 1.2 | `src/1.2_build_corpus.py` | Build corpus from Sangraha + Wikipedia + FineWeb-2 (native-script filters) |
| 1.3 | `src/1.3_dedup_corpus.py` | MinHash-LSH near-dedup (5-gram, 128 perm, thr 0.80) |
| 1.4 | `src/1.4_clean_corpus.py` | Shape-aware boilerplate + Gopher filters (Gujarati-safe) |
| 1.4 | `src/1.4_measure_english.py` | Audit English/Gujarati composition (-> english_report.json) |
| 1.5 | `src/1.5_train_tokenizer.py` | Train + A/B the 32K char-level BPE tokenizer |
| 1.6 | `src/1.6_tokenizer_pack.py` | Tokenize + pack to uint16 train.bin / val.bin |
| 2.1 | `src/2.1_pretrain/` | Model + WSD pretraining (13k steps, ~6.78B tokens x2 epochs) |
| 3.1 | `src/3.1a/b/c_*.py` | Build instruction SFT data (augment -> v3 -> v4) |
| 3.2 | `src/3.2_train_instruction_sft.py` | Instruction SFT (on pretrained base) |
| 4.1 | `src/4.1_build_medical_sft.py` | Build medical grounding data (raw + redacted + general mix) |
| 4.2 | `src/4.2_train_medical_sft.py` | Medical SFT (stacks on instruction v4 -> final model) |
| 5.1 | `src/5.1_eval_hf.py` | Evaluation harness |
| 5.2 | `src/5.2_run_bench.sh` | External-model comparison run |
| 5.3 | `src/5.3_make_level1_log.py` | Level-1 validity log (loss reduction, convergence, no-NaN) |

## Reproduce
Environment: Python 3.10; install pinned deps with `pip install -r requirements.txt`.
Prep (steps 1.2–1.6) runs on CPU; training/eval needs a single GPU (developed on NVIDIA L4).

**Pretraining** — run from inside the pretrain folder (train1.py imports `model`). The argparse
DEFAULTS in train1.py are NOT the run config; use these exact flags:
```
cd src/2.1_pretrain
python train1.py --batch-size 8 --grad-accum 64 --compile \
  --max-steps 13000 --warmup 700 --lr 4e-4 --min-lr 4e-5 --decay-frac 0.2 \
  --eval-interval 1000 --ckpt-interval 2000
```

**Fine-tuning + eval** — the SFT/eval scripts carry their config as constants at the top of each
file (e.g. `BASE_CKPT`, input/output paths). Set those, then run in order 3.1 -> 3.2 -> 4.1 -> 4.2,
and evaluate with `5.1_eval_hf.py`. Lineage: pretrained base -> instruction SFT (v4) -> medical SFT (v5).

## Architecture
Llama-style decoder-only, ~109.5M params:
- 12 layers, d_model 768, 12 heads (full MHA, n_kv_head=12), head_dim 64
- RMSNorm (pre-norm, fp32), RoPE (theta 10000), SwiGLU FFN (hidden 2048), tied embeddings, no biases
- vocab 32000 (char-level BPE), context length 2048
- init 0.02; residual projections scaled 0.02/sqrt(2·n_layer)

## Training data & licensing
Pretraining corpus (~6.78B tokens, Gujarati native script), described but not re-hosted:
- ai4bharat/sangraha — CC-BY-4.0 (arXiv:2403.06350)
- wikimedia/wikipedia (20231101.gu) — CC-BY-SA
- HuggingFaceFW/fineweb-2 (guj_Gujr) — ODC-BY

Instruction + medical SFT datasets are published (link above).

## Limitations
- Designed for **open-book / RAG** medical use; closed-book recall is limited (~28%) by 110M capacity.
- Can show long-range topic drift and factual confabulation on open-ended generation.
- Gujarati only.

## Documentation
- `RUNLOG.md` — full chronological run log with metrics and incident logs.
- `DECISIONS.md` — every design decision with rationale, rejected alternatives, and citations.

## Acknowledgements
Built during the IAIRO / PRAMANA SLM++ bootcamp.

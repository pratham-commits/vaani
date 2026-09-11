#!/bin/bash
cd ~/vaani
run () {
  MODEL="$1"; TMPL="$2"; SAFE=$(echo "$MODEL" | tr '/' '_'); OUT="bench_${SAFE}.txt"
  echo "===== $(date '+%F %T') START $MODEL ($TMPL) ====="; df -h / | tail -1
  python3 eval_hf.py all --model "$MODEL" --template "$TMPL" > "$OUT" 2>&1 \
    && echo "OK  $MODEL" || echo "FAILED $MODEL (see $OUT)"
  tail -n 12 "$OUT"; rm -rf ~/.cache/huggingface/hub/models--*
  echo "disk after:"; df -h / | tail -1; echo
}
run "sarvamai/sarvam-1"                                          alpaca
run "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0"  alpaca
run "Qwen/Qwen2.5-7B-Instruct"                                   chat
echo "===== ALL DONE $(date '+%F %T') ====="

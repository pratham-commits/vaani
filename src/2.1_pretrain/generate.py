# generate1.py — sample text from a Vaani checkpoint (with repetition control)
import argparse, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from model import Vaani, VaaniConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tokenizer', default='tokenizer/tokenizer_bpe_char.json')
    ap.add_argument('--prompt', default='')
    ap.add_argument('--max-new-tokens', type=int, default=200)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=50)          # 0 disables
    ap.add_argument('--top-p', type=float, default=0.9)        # 1.0 disables
    ap.add_argument('--repetition-penalty', type=float, default=1.15)  # 1.0 disables
    ap.add_argument('--no-repeat-ngram', type=int, default=0)  # 0 disables, e.g. 3
    ap.add_argument('--num-samples', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<eos>")

    ck = torch.load(args.ckpt, map_location=device)
    cfg = VaaniConfig(**ck['cfg'])
    model = Vaani(cfg).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"loaded {args.ckpt} (step {ck.get('step','?')}), {model.num_params()/1e6:.1f}M params\n")

    base = tok.encode(args.prompt).ids if args.prompt else []
    if not base:
        base = [eos_id] if eos_id is not None else [0]

    for s in range(args.num_samples):
        idx = torch.tensor([base], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                idx_cond = idx[:, -cfg.block_size:]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(idx_cond)
                logits = logits[:, -1, :].float()

                # --- repetition penalty (applied to already-generated tokens) ---
                if args.repetition_penalty != 1.0:
                    prev_ids = torch.tensor(sorted(set(idx[0].tolist())), device=device)
                    scores = logits[0, prev_ids]
                    scores = torch.where(scores > 0,
                                         scores / args.repetition_penalty,
                                         scores * args.repetition_penalty)
                    logits[0, prev_ids] = scores

                # --- no-repeat-ngram: hard-block n-grams that already occurred ---
                if args.no_repeat_ngram > 0:
                    n = args.no_repeat_ngram
                    seq = idx[0].tolist()
                    if len(seq) >= n - 1 and n > 1:
                        prefix = tuple(seq[-(n - 1):])
                        for i in range(len(seq) - n + 1):
                            if tuple(seq[i:i + n - 1]) == prefix:
                                logits[0, seq[i + n - 1]] = -float('inf')

                # --- temperature ---
                logits = logits / max(args.temperature, 1e-5)

                # --- top-k ---
                if args.top_k:
                    v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')

                # --- top-p (nucleus) ---
                if args.top_p and args.top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum > args.top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    to_remove = remove.scatter(1, sorted_idx, remove)
                    logits[to_remove] = -float('inf')

                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, nxt], dim=1)
                if eos_id is not None and nxt.item() == eos_id:
                    break
        out = tok.decode(idx[0].tolist())
        print(f"--- sample {s+1} ---\n{out}\n")


if __name__ == '__main__':
    main()

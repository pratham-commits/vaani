# train1.py — WSD pretraining loop (vectorized data loader)
import os, math, time, argparse
import numpy as np
import torch
from model import Vaani, VaaniConfig

L4_PEAK_BF16 = 120e12  # ~L4 bf16 peak FLOP/s, for MFU estimate


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--ckpt-dir', default='ckpt')
    p.add_argument('--block-size', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--grad-accum', type=int, default=32)
    p.add_argument('--max-steps', type=int, default=27000)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--min-lr', type=float, default=4e-5)
    p.add_argument('--decay-frac', type=float, default=0.2)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--eval-interval', type=int, default=1000)
    p.add_argument('--eval-iters', type=int, default=100)
    p.add_argument('--log-interval', type=int, default=10)
    p.add_argument('--ckpt-interval', type=int, default=2000)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--overfit-one-batch', action='store_true')
    p.add_argument('--proxy-steps', type=int, default=0)
    p.add_argument('--resume', default='')
    p.add_argument('--compile', action='store_true')
    return p.parse_args()


def wsd_lr(step, warmup, total, peak, minlr, decay_frac):
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return peak
    prog = min(1.0, (step - decay_start) / max(1, total - decay_start))
    return minlr + (peak - minlr) * 0.5 * (1 + math.cos(math.pi * prog))


class Data:
    def __init__(self, data_dir, block_size, device):
        self.tr = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.va = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        self.bs = block_size
        self.device = device

    def get_batch(self, split, batch_size):
        d = self.tr if split == 'train' else self.va
        ix = np.random.randint(0, len(d) - self.bs - 1, size=batch_size)
        idx = ix[:, None] + np.arange(self.bs + 1)[None, :]
        batch = torch.from_numpy(np.asarray(d[idx], dtype=np.int64))
        x = batch[:, :-1].contiguous()
        y = batch[:, 1:].contiguous()
        return (x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True))


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.ckpt_dir, exist_ok=True)

    cfg = VaaniConfig(block_size=a.block_size)
    model = Vaani(cfg).to(device)
    N = model.num_params()
    print(f"params: {N/1e6:.1f}M | device: {device}")

    data = Data(a.data_dir, a.block_size, device)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    optim = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': a.weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=(device == 'cuda'))

    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck['model']); optim.load_state_dict(ck['optim'])
        start = ck['step']; print(f"resumed at step {start}")

    if a.compile:
        model = torch.compile(model)

    if a.overfit_one_batch:
        xb, yb = data.get_batch('train', a.batch_size)
        for step in range(200):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(xb, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if step % 10 == 0:
                print(f"[overfit] step {step:3d}  loss {loss.item():.4f}")
        print("overfit done.")
        return

    @torch.no_grad()
    def eval_loss():
        model.eval(); out = {}
        for split in ['train', 'val']:
            L = torch.zeros(a.eval_iters)
            for k in range(a.eval_iters):
                xb, yb = data.get_batch(split, a.batch_size)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss = model(xb, yb)
                L[k] = loss.item()
            out[split] = L.mean().item()
        model.train(); return out

    total = a.proxy_steps if a.proxy_steps else a.max_steps
    tok_per_step = a.batch_size * a.grad_accum * a.block_size
    model.train()
    t0 = time.time(); tok_since = 0
    for step in range(start, total):
        lr = wsd_lr(step, a.warmup, a.max_steps, a.lr, a.min_lr, a.decay_frac)
        for g in optim.param_groups:
            g['lr'] = lr

        optim.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for _ in range(a.grad_accum):
            xb, yb = data.get_batch('train', a.batch_size)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(xb, yb)
                loss = loss / a.grad_accum
            loss.backward()
            loss_accum += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        optim.step()
        tok_since += tok_per_step

        if step % a.log_interval == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            tps = tok_since / dt if dt > 0 else 0
            mfu = (6 * N * tps) / L4_PEAK_BF16
            print(f"step {step:5d} | loss {loss_accum.item():.4f} | lr {lr:.2e} | "
                  f"gnorm {gnorm:.2f} | {tps:,.0f} tok/s | mfu {mfu*100:.1f}%")
            t0 = time.time(); tok_since = 0

        if a.proxy_steps == 0 and step > start and step % a.eval_interval == 0:
            m = eval_loss()
            print(f"  >> eval  step {step}  train {m['train']:.4f}  val {m['val']:.4f}  "
                  f"val_ppl {math.exp(m['val']):.2f}")

        if a.proxy_steps == 0 and step > start and step % a.ckpt_interval == 0:
            path = os.path.join(a.ckpt_dir, f'ckpt_{step}.pt')
            torch.save({'model': (model._orig_mod if hasattr(model, '_orig_mod') else model).state_dict(),
                        'optim': optim.state_dict(), 'step': step, 'cfg': cfg.__dict__}, path)
            print(f"  >> saved {path}")

    if a.proxy_steps:
        print("proxy run complete — check tok/s and mfu above.")
    else:
        path = os.path.join(a.ckpt_dir, 'ckpt_final.pt')
        torch.save({'model': (model._orig_mod if hasattr(model, '_orig_mod') else model).state_dict(),
                    'optim': optim.state_dict(), 'step': total, 'cfg': cfg.__dict__}, path)
        print(f"saved {path}")


if __name__ == '__main__':
    main()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, math, time, random
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model import Vaani, VaaniConfig

SFT_PATH  = "medical_sft_v2.jsonl"
TOK_PATH  = "tokenizer/tokenizer_bpe_char.json"
BASE_CKPT = "sft_ckpt_v4/sft_final.pt"
OUT_DIR   = "sft_ckpt_med_v5"
GCS_DIR   = "gs://commanding-iris-474607-u2-guj-corpus/sft_ckpt_med_v5"
MAX_LEN   = 2048
MICRO_BS  = 8
GRAD_ACC  = 4          # effective batch 32
EPOCHS    = 2
LR        = 1e-5
MIN_LR    = 1e-6
WARMUP    = 100
WD        = 0.1
GRAD_CLIP = 1.0
VAL_FRAC  = 0.01
LOG_EVERY = 20
SEED      = 1234

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
_logf = open(f"{OUT_DIR}/train_sft.log", "a", buffering=1)
def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); _logf.write(m + "\n")

tok = Tokenizer.from_file(TOK_PATH)
eos_id = tok.token_to_id("<eos>")
assert eos_id is not None
U_PRE, U_SUF = "વપરાશકર્તા:\n", "\nસહાયક:\n"
def enc(s): return tok.encode(s).ids

def build_example(messages):
    seq, sup = [eos_id], [False]
    i = 0
    while i < len(messages):
        u = messages[i]["content"]   if messages[i]["role"] == "user" else ""
        a = messages[i+1]["content"] if i+1 < len(messages) and messages[i+1]["role"]=="assistant" else ""
        u_ids = enc(U_PRE + u + U_SUF); seq += u_ids; sup += [False]*len(u_ids)
        a_ids = enc(a);                 seq += a_ids; sup += [True]*len(a_ids)
        seq += [eos_id];                sup += [True]
        i += 2
    if len(seq) > MAX_LEN + 1: return None
    return seq[:-1], [seq[j] if sup[j] else -1 for j in range(1, len(seq))]

class SFTDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                ex = build_example(json.loads(line)["messages"])
                if ex and any(t != -1 for t in ex[1]): self.data.append(ex)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate(batch):
    m = max(len(x[0]) for x in batch)
    xs, ys = [], []
    for ids, tgt in batch:
        pad = m - len(ids); xs.append(ids + [eos_id]*pad); ys.append(tgt + [-1]*pad)
    return torch.tensor(xs), torch.tensor(ys)

ds = SFTDataset(SFT_PATH)
n_val = max(1, int(len(ds)*VAL_FRAC))
val_ds, tr_ds = torch.utils.data.random_split(ds, [n_val, len(ds)-n_val],
                                              generator=torch.Generator().manual_seed(SEED))
tr_dl = DataLoader(tr_ds, batch_size=MICRO_BS, shuffle=True,  collate_fn=collate, drop_last=True)
va_dl = DataLoader(val_ds, batch_size=MICRO_BS, shuffle=False, collate_fn=collate)
log(f"examples {len(ds):,} | train {len(tr_ds):,} | val {len(val_ds):,}")

ck = torch.load(BASE_CKPT, map_location="cpu")
cfg = VaaniConfig(**ck["cfg"])
model = Vaani(cfg).to(device); model.load_state_dict(ck["model"])
log(f"loaded base (step {ck.get('step','?')}), {model.num_params()/1e6:.1f}M params")

use_bf16 = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)
log("precision:", "bf16" if use_bf16 else "fp16")

decay   = [p for p in model.parameters() if p.dim() >= 2]
nodecay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                         {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(0.9,0.95), fused=True)

steps_per_epoch = len(tr_dl)//GRAD_ACC
total_steps = steps_per_epoch*EPOCHS
def lr_at(s):
    if s < WARMUP: return LR*(s+1)/WARMUP
    p = (s-WARMUP)/max(1,total_steps-WARMUP)
    return MIN_LR + 0.5*(LR-MIN_LR)*(1+math.cos(math.pi*p))
log(f"steps/epoch {steps_per_epoch} | total {total_steps}")

@torch.no_grad()
def evaluate():
    model.eval(); tot=n=0
    for x,y in va_dl:
        x,y = x.to(device),y.to(device)
        with torch.autocast("cuda", dtype=amp_dtype):
            _, loss = model(x,y)
        tot += loss.item(); n += 1
        if n >= 50: break
    model.train(); return tot/max(1,n)

model.train(); step=0; t0=time.time()
for epoch in range(EPOCHS):
    opt.zero_grad(set_to_none=True)
    for it,(x,y) in enumerate(tr_dl):
        x,y = x.to(device),y.to(device)
        with torch.autocast("cuda", dtype=amp_dtype):
            _, loss = model(x,y); loss = loss/GRAD_ACC
        scaler.scale(loss).backward()
        if (it+1)%GRAD_ACC == 0:
            for g in opt.param_groups: g["lr"] = lr_at(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if step % LOG_EVERY == 0:
                log(f"e{epoch} step {step}/{total_steps} | loss {loss.item()*GRAD_ACC:.4f} "
                    f"| lr {opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")
            step += 1
    vl = evaluate()
    log(f"== epoch {epoch} | val_loss {vl:.4f} | val_ppl {math.exp(vl):.2f}")
    path = f"{OUT_DIR}/sft_epoch{epoch}.pt"
    torch.save({"model": model.state_dict(), "cfg": ck["cfg"], "step": step,
                "epoch": epoch, "val_loss": vl}, path)
    os.system(f"gcloud storage cp {path} {GCS_DIR}/ 2>/dev/null")
    os.system(f"gcloud storage cp {OUT_DIR}/train_sft.log {GCS_DIR}/ 2>/dev/null")
    log(f"saved + synced {path}")

torch.save({"model": model.state_dict(), "cfg": ck["cfg"], "step": step}, f"{OUT_DIR}/sft_final.pt")
os.system(f"gcloud storage cp {OUT_DIR}/sft_final.pt {GCS_DIR}/ 2>/dev/null")
os.system(f"gcloud storage cp {OUT_DIR}/train_sft.log {GCS_DIR}/ 2>/dev/null")
log("saved + synced sft_final.pt")
_logf.close()

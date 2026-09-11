import re, json, csv
TOK_PER_STEP = 16 * 32 * 2048   # batch * grad_accum * block_size = 1,048,576

STEP_RE = re.compile(r'step\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*lr\s+([\d.eE+-]+)\s*\|\s*gnorm\s+([\d.]+)')
EVAL_RE = re.compile(r'>>\s*eval\s+step\s+(\d+)\s+train\s+([\d.]+)\s+val\s+([\d.]+)')

def parse(path):
    steps, evals = {}, {}
    for line in open(path):
        m = STEP_RE.search(line)
        if m: steps[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        e = EVAL_RE.search(line)
        if e: evals[int(e.group(1))] = float(e.group(3))
    return steps, evals

s1, e1 = parse('train_20260903_0917.log')
s2, e2 = parse('train_resume.log')
merged = {s:v for s,v in s1.items() if s < 2000}
merged.update({s:v for s,v in s2.items() if s >= 2000})
evals = {s:v for s,v in e1.items() if s < 2000}
evals.update({s:v for s,v in e2.items() if s >= 2000})

rows = []
for s in sorted(merged):
    l, lr, g = merged[s]
    rows.append({'step': s, 'train_loss': l, 'val_loss': evals.get(s, ''),
                 'lr': lr, 'grad_norm': g, 'tokens_seen': s * TOK_PER_STEP})

with open('pretrain_log.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['step','train_loss','val_loss','lr','grad_norm','tokens_seen'])
    w.writeheader(); w.writerows(rows)
with open('pretrain_log.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')

tl = [r['train_loss'] for r in rows]
init, fin = tl[0], tl[-1]; red = 100*(init-fin)/init
vals = [(r['val_loss'], r['train_loss']) for r in rows if r['val_loss'] != '']
gap = (vals[-1][0]-vals[-1][1])/vals[-1][1]*100 if vals else float('nan')
print(f"rows {len(rows)} | steps {rows[0]['step']}..{rows[-1]['step']}")
print(f"initial {init:.4f} -> final {fin:.4f} | reduction {red:.1f}%  [{'PASS' if red>=15 else 'FAIL'}]")
print(f"convergence final < 0.7*init ({0.7*init:.2f}): [{'PASS' if fin<0.7*init else 'FAIL'}]")
print(f"final val-train gap {gap:.1f}%  | NaN present: {any(x!=x for x in tl)}")
print(f"tokens_seen final: {rows[-1]['tokens_seen']:,}")

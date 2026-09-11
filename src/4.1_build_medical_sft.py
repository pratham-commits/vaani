import json, re, random
from collections import Counter
random.seed(0)

MED_IN, GEN_IN, OUT = "medmcqa_gu.jsonl", "sft_augmented.jsonl", "medical_sft_v3.jsonl"
GEN_FRAC = 0.30
# of rows WITH a good explanation, split into: raw-grounded / redacted-grounded / closed
P_RAW, P_REDACT, P_CLOSED = 0.40, 0.25, 0.35

GUJ = re.compile(r'[\u0A80-\u0AFF]'); TAG = re.compile(r'<[^>]+>')
ANNOUNCE = ("સાચો જવાબ","જવાબ છે","correct answer","answer is","the answer")

def guj_ratio(s):
    L=[c for c in s if c.isalpha()]; return (sum(1 for c in L if GUJ.match(c))/len(L)) if L else 0.0
def clean(e): return re.sub(r'\s+',' ',TAG.sub(' ',e or '')).strip()
def sents(t): return [s.strip() for s in re.split(r'(?<=[।.!?])\s+', t) if s.strip()]
def norm(s): return re.sub(r'\s+','',s or '').lower()
def strip_answer(expl, correct):
    c=norm(correct); keep=[]
    for s in sents(expl):
        if c and c in norm(s): continue
        if any(a in s.lower() for a in ANNOUNCE): continue
        keep.append(s)
    return " ".join(keep).strip()
def opts_block(o): return "વિકલ્પો:\n"+"\n".join(f"{i+1}. {x}" for i,x in enumerate(o))
def load(p):
    with open(p,encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
def msg(u,a,src): return {"source":src,"messages":[{"role":"user","content":u},{"role":"assistant","content":a}]}

med = load(MED_IN); medical=[]; cnt=Counter()
for r in med:
    q=(r.get("question") or "").strip(); opts=r.get("options") or []; a=r.get("answer")
    if not q or len(opts)<2 or a not in range(len(opts)): continue
    correct=(opts[a] or "").strip()
    if not correct: continue
    expl=clean(r.get("explanation")); good=len(expl)>15 and guj_ratio(expl)>=0.55
    ublock=f"{q}\n\n{opts_block(opts)}"
    if not good:
        medical.append(msg(ublock, f"સાચો જવાબ: {correct}.", "med_closed")); cnt["closed"]+=1; continue
    d=random.random()
    if d < P_RAW:
        u=f"સંદર્ભ: {expl}\n\n{ublock}"
        medical.append(msg(u, f"આપેલા સંદર્ભ પ્રમાણે, સાચો જવાબ છે: {correct}.", "med_grounded_raw")); cnt["raw"]+=1
    elif d < P_RAW+P_REDACT:
        red = strip_answer(expl, correct)
        if len(red) >= 25:                       # length filter: enough left to infer
            u=f"સંદર્ભ: {red}\n\n{ublock}"
            medical.append(msg(u, f"આપેલા સંદર્ભના આધારે, સાચો જવાબ છે: {correct}.", "med_grounded_redact")); cnt["redact"]+=1
        else:                                    # too little left -> fall back to closed+explanation
            medical.append(msg(ublock, f"સાચો જવાબ: {correct}. {expl}", "med_closed")); cnt["closed"]+=1
    else:
        medical.append(msg(ublock, f"સાચો જવાબ: {correct}. {expl}", "med_closed")); cnt["closed"]+=1

gen=load(GEN_IN); random.shuffle(gen)
n_gen=int(len(medical)*GEN_FRAC/(1-GEN_FRAC)); gmix=gen[:n_gen]
for r in gmix: r.setdefault("source","general")
allrows=medical+gmix; random.shuffle(allrows)
with open(OUT,"w",encoding="utf-8") as f:
    for r in allrows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("medical:", dict(cnt), "| total medical:", len(medical))
print("general mix:", len(gmix), "| ->", OUT, ":", len(allrows))

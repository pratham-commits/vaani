import json, random
random.seed(1234)

IN_PATH  = "sft_filtered.jsonl"
OUT_PATH = "sft_augmented.jsonl"

SHORT_INSTR = ["ટૂંકમાં જવાબ આપો.", "એક વાક્યમાં જવાબ આપો.", "સંક્ષિપ્તમાં જણાવો."]
WORD_INSTR  = ["એક શબ્દમાં જવાબ આપો."]
LONG_INSTR  = ["વિગતવાર સમજાવો.", "વિગતવાર જણાવો.", "વિસ્તારપૂર્વક સમજાવો."]

CAP_WORD, CAP_SHORT, CAP_LONG = 3000, 8000, 8000

def single_turn(m):
    return len(m) == 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant"

orig, word_c, short_c, long_c = [], [], [], []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line); orig.append(rec)
        m = rec["messages"]
        if not single_turn(m): continue
        q, a = m[0]["content"].strip(), m[1]["content"].strip()
        alen = len(a)
        if alen <= 30 and a.count(" ") <= 1:   word_c.append((q, a))
        elif 30 < alen <= 250:                 short_c.append((q, a))
        elif alen >= 800:                      long_c.append((q, a))

random.shuffle(word_c); random.shuffle(short_c); random.shuffle(long_c)
word_c, short_c, long_c = word_c[:CAP_WORD], short_c[:CAP_SHORT], long_c[:CAP_LONG]

def make(q, a, pool):
    return {"source": "length_aug", "messages": [
        {"role": "user", "content": f"{q} {random.choice(pool)}"},
        {"role": "assistant", "content": a}]}

aug = ([make(q, a, WORD_INSTR)  for q, a in word_c] +
       [make(q, a, SHORT_INSTR) for q, a in short_c] +
       [make(q, a, LONG_INSTR)  for q, a in long_c])

allrecs = orig + aug
random.shuffle(allrecs)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in allrecs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"original {len(orig):,}")
print(f"augmented: one-word {len(word_c):,} | short {len(short_c):,} | detailed {len(long_c):,} | total {len(aug):,}")
print(f"final {len(allrecs):,} -> {OUT_PATH}")

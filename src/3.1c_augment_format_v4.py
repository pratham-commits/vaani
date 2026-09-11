import json, random
from collections import Counter
random.seed(0)

IN, OUT = "sft_v3.jsonl", "sft_v4.jsonl"
CAP_JSON, CAP_EXTRACT, CAP_TRANSFORM = 4000, 3000, 3000

NAMES = ["રમેશ","સીતા","અમિત","પ્રિયા","રાજેશ","મીરા","સુરેશ","ગીતા","વિનોદ","નયના",
         "કિરણ","અનિલ","દીપા","મનીષ","રેખા","સંજય","પૂજા","હિતેશ","ભાવના","જયેશ"]
CITIES = ["અમદાવાદ","સુરત","વડોદરા","રાજકોટ","ભાવનગર","જામનગર","ગાંધીનગર","મુંબઈ","દિલ્હી","પુણે"]
COUNTRIES = ["ભારત","જાપાન","ફ્રાન્સ","જર્મની","કેનેડા","બ્રાઝિલ","ઇટાલી","સ્પેન"]
BOOKS = ["સરસ્વતીચંદ્ર","માનવીની ભવાઈ","મળેલા જીવ","કરણ ઘેલો","ગુજરાતનો નાથ"]
AUTHORS = ["ગોવર્ધનરામ","પન્નાલાલ પટેલ","ઝવેરચંદ મેઘાણી","નંદશંકર","કનૈયાલાલ મુનશી","ધૂમકેતુ"]
FRUITS = ["કેરી","કેળું","સફરજન","દ્રાક્ષ","દાડમ","સંતરું","જામફળ","ચીકુ"]
COLORS = ["પીળો","લાલ","લીલો","જાંબલી","નારંગી","કેસરી"]
ANIMALS = ["વાઘ","સિંહ","હાથી","વાંદરો","હરણ","સસલું"]
HABITATS = ["જંગલ","ઘાસનું મેદાન","પર્વત","રણ"]

def ex(u,a,src): return {"source":src,"messages":[{"role":"user","content":u},{"role":"assistant","content":a}]}

def gen_json():
    k = random.choice(["person","person3","place","book","fruit","animal"])
    if k=="person":
        n,a = random.choice(NAMES), random.randint(5,80)
        return (f'આપેલી માહિતીને JSON સ્વરૂપે લખો — keys: "name" અને "age". નામ: {n}, ઉંમર: {a}.',
                json.dumps({"name":n,"age":a}, ensure_ascii=False))
    if k=="person3":
        n,a,c = random.choice(NAMES), random.randint(5,80), random.choice(CITIES)
        return (f'આપેલી માહિતીને JSON માં આપો — keys: "name", "age", "city". નામ: {n}, ઉંમર: {a}, શહેર: {c}.',
                json.dumps({"name":n,"age":a,"city":c}, ensure_ascii=False))
    if k=="place":
        c,co = random.choice(CITIES), random.choice(COUNTRIES)
        return (f'આપેલી વિગત JSON માં આપો — keys: "city" અને "country". શહેર: {c}, દેશ: {co}.',
                json.dumps({"city":c,"country":co}, ensure_ascii=False))
    if k=="book":
        t,au = random.choice(BOOKS), random.choice(AUTHORS)
        return (f'એક પુસ્તક વિશે JSON બનાવો — keys: "title" અને "author". શીર્ષક: {t}, લેખક: {au}.',
                json.dumps({"title":t,"author":au}, ensure_ascii=False))
    if k=="fruit":
        f,c = random.choice(FRUITS), random.choice(COLORS)
        return (f'એક ફળ વિશે JSON આપો — keys: "fruit" અને "color". ફળ: {f}, રંગ: {c}.',
                json.dumps({"fruit":f,"color":c}, ensure_ascii=False))
    a,h = random.choice(ANIMALS), random.choice(HABITATS)
    return (f'એક પ્રાણી વિશે JSON આપો — keys: "animal" અને "habitat". પ્રાણી: {a}, રહેઠાણ: {h}.',
            json.dumps({"animal":a,"habitat":h}, ensure_ascii=False))

def gen_extract():
    k = random.choice(["age","name","city","numbers"])
    if k=="age":
        n,a = random.choice(NAMES), random.randint(5,90)
        return f'આ વાક્યમાંથી ઉંમર કાઢો: "{n}ની ઉંમર {a} વર્ષ છે."', f'{a}'
    if k=="name":
        n = random.choice(NAMES)
        return f'આ વાક્યમાંથી વ્યક્તિનું નામ કાઢો: "ગઈકાલે {n} બજારમાં ગયાં હતાં."', n
    if k=="city":
        n,c = random.choice(NAMES), random.choice(CITIES)
        return f'આ વાક્યમાંથી શહેરનું નામ કાઢો: "{n} {c}માં રહે છે."', c
    a,b,c = random.randint(1,20), random.randint(1,20), random.randint(1,20)
    f1,f2,f3 = random.sample(FRUITS,3)
    return (f'આ વાક્યમાંથી બધા આંકડા કાઢીને લખો: "મારી પાસે {a} {f1}, {b} {f2} અને {c} {f3} છે."',
            f'{a}, {b}, {c}')

REPL_PAIRS = [("સારું","ખરાબ"),("મોટું","નાનું"),("દિવસ","રાત"),("ગરમ","ઠંડું"),("ઝડપી","ધીમું"),
              ("નવું","જૂનું"),("ઊંચું","નીચું"),("ખુલ્લું","બંધ"),("સાચું","ખોટું"),
              ("સફેદ","કાળું"),("સુખી","દુઃખી"),("સરળ","અઘરું")]
REPL_SENT = ['આજે હવામાન {w} છે.','આ ઘર ખૂબ {w} છે.','તે એક {w} વિચાર છે.',
             'આ રસ્તો {w} છે.','મારું કામ {w} હતું.','આ પુસ્તક {w} છે.']
def gen_transform():
    k = random.choice(["replace","reverse","sort"])
    if k=="replace":
        w1,w2 = random.choice(REPL_PAIRS); t = random.choice(REPL_SENT)
        return f'આ વાક્યમાં "{w1}" ને "{w2}" થી બદલો: "{t.format(w=w1)}"', t.format(w=w2)
    if k=="reverse":
        items = random.sample(FRUITS+COLORS+CITIES, 3)
        src = "  ".join(f"{i+1}. {w}" for i,w in enumerate(items))
        dst = "\n".join(f"{i+1}. {w}" for i,w in enumerate(reversed(items)))
        return f'આ યાદીને ઊંધા ક્રમમાં લખો: {src}', dst
    words = random.sample(NAMES+CITIES+FRUITS, random.choice([3,4]))
    return f'આ શબ્દોને અક્ષરના ક્રમમાં ગોઠવો: {", ".join(words)}', ", ".join(sorted(words))

def load(p):
    with open(p,encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]

base = load(IN); new, cnt = [], Counter()
def add(gen, cap, src):
    for _ in range(cap):
        u,a = gen(); new.append(ex(u,a,src)); cnt[src]+=1
add(gen_json,      CAP_JSON,      "aug_json")
add(gen_extract,   CAP_EXTRACT,   "aug_extract")
add(gen_transform, CAP_TRANSFORM, "aug_transform")

allrows = base + new; random.shuffle(allrows)
with open(OUT,"w",encoding="utf-8") as f:
    for r in allrows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("new:", dict(cnt), "| total new:", len(new))
print(f"base(sft_v3): {len(base)} -> {OUT}: {len(allrows)}")

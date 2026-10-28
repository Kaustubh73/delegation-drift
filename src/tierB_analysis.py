"""Pooled Tier B analysis: ever-takeover per trajectory, both-judges / either-judge, scenario-cluster bootstrap. Usage: python3 tierB_analysis.py <dir>..."""
import json, os, random, sys
J = ["judge_qwen2.5_14b-instruct.jsonl", "judge_gemma3_12b.jsonl"]
rows = []  # (model, scenario, prompt, ever_qwen, ever_gemma)
for d in sys.argv[1:]:
    tr = [json.loads(l) for l in open(os.path.join(d, "transcripts.jsonl")) if l.strip()]
    meta = {r["key"]: r for r in tr if r["script"] == "persistent"}
    ever = []
    for jf in J:
        e = set()
        for l in open(os.path.join(d, jf)):
            j = json.loads(l)
            if j["key"] in meta and j["score"] == -1: e.add(j["key"])
        ever.append(e)
    for k, r in meta.items():
        rows.append((os.path.basename(d), r["scenario"], r["condition"], k in ever[0], k in ever[1]))
agg = {"both": lambda q, g: q and g, "either": lambda q, g: q or g}
def rate(rs, c, m):
    x = [agg[m](q, g) for _, _, cc, q, g in rs if cc == c]
    return 100 * sum(x) / len(x) if x else float("nan")
def contrast(rs, a, b, m, B=2000):
    scs = sorted({r[1] for r in rs}); rnd = random.Random(0); pt = rate(rs, a, m) - rate(rs, b, m); bs = []
    for _ in range(B):
        samp = [r for s in (rnd.choice(scs) for _ in scs) for r in rs if r[1] == s]
        bs.append(rate(samp, a, m) - rate(samp, b, m))
    bs.sort(); return pt, bs[int(.025 * B)], bs[int(.975 * B)]
for m in ("both", "either"):
    print(f"\n## ever-takeover by T8, {m} judges; n per cell per model = 32")
    for mod in sorted({r[0] for r in rows}) + ["POOLED"]:
        rs = rows if mod == "POOLED" else [r for r in rows if r[0] == mod]
        print(f"{mod:32s} eng {rate(rs,'engagement',m):4.0f}  neu {rate(rs,'neutral',m):4.0f}  aut {rate(rs,'autonomy',m):4.0f}   "
              + "  ".join(f"{a[:3]}-{b[:3]} {contrast(rs,a,b,m)[0]:+.0f} [{contrast(rs,a,b,m)[1]:+.0f},{contrast(rs,a,b,m)[2]:+.0f}]" for a, b in (("engagement", "neutral"), ("neutral", "autonomy"))))

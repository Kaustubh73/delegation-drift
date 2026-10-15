#!/usr/bin/env python3
"""Make results/*.png from data/runs and data/gold. Usage: python3 src/make_figures.py"""
import glob, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from drift_eval import CONDITIONS, load_jsonl, mean_ci, weighted_kappa

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
COL = {"engagement": "#c0392b", "neutral": "#7f8c8d", "autonomy": "#2471a3"}

rows = []  # one per (trajectory, turn, judge)
for d in sorted(glob.glob(os.path.join(ROOT, "data/runs/*/"))):
    meta = {r["key"]: r for r in load_jsonl(os.path.join(d, "transcripts.jsonl"))}
    for jf in glob.glob(os.path.join(d, "judge_*.jsonl")):
        for j in load_jsonl(jf):
            m = meta.get(j["key"])
            if m and j["score"] is not None:
                rows.append(dict(model=m["model"].split(":")[0], script=m["script"], cond=m["condition"],
                                 key=j["key"], turn=j["turn"], judge=j["judge"], score=j["score"]))
judges = sorted({r["judge"] for r in rows})
models = sorted({r["model"] for r in rows})
rate = lambda xs: mean_ci([int(s == -1) for s in xs])  # takeover rate with bootstrap CI

# Fig 1: takeover rate by turn, escalating script, pooled over models
fig, ax = plt.subplots(1, len(judges), figsize=(5 * len(judges), 3.6), sharey=True, squeeze=False)
for a, jd in zip(ax[0], judges):
    for c in CONDITIONS:
        pts = [rate([r["score"] for r in rows if r["judge"] == jd and r["script"] == "escalating" and r["cond"] == c and r["turn"] == t]) for t in range(1, 5)]
        a.errorbar(range(1, 5), [100 * p[0] for p in pts], yerr=[[100 * (p[0] - p[1]) for p in pts], [100 * (p[2] - p[0]) for p in pts]],
                   label=c, color=COL[c], marker="o", capsize=3)
    a.set(title=f"judge: {jd}", xlabel="assistant turn", xticks=range(1, 5)); a.grid(alpha=.3)
ax[0][0].set_ylabel("takeover rate (%)"); ax[0][0].legend(title="system prompt")
fig.suptitle("Takeover rate by turn (escalating user script, 5 models pooled, 95% CI)"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_takeover_by_turn.png"), dpi=160); plt.close(fig)

# Fig 2: turn-4 takeover per model and prompt, plus control (neutral) marker
fig, ax = plt.subplots(1, len(judges), figsize=(6 * len(judges), 3.8), sharey=True, squeeze=False)
for a, jd in zip(ax[0], judges):
    for i, c in enumerate(CONDITIONS):
        ys = [100 * rate([r["score"] for r in rows if r["judge"] == jd and r["model"] == m and r["script"] == "escalating" and r["cond"] == c and r["turn"] == 4])[0] for m in models]
        a.bar([x + (i - 1) * 0.27 for x in range(len(models))], ys, 0.27, label=c, color=COL[c])
    a.set(title=f"judge: {jd}", xticks=range(len(models))); a.set_xticklabels(models, rotation=20, ha="right"); a.grid(axis="y", alpha=.3)
ax[0][0].set_ylabel("takeover rate at turn 4 (%)"); ax[0][0].legend(title="system prompt")
fig.suptitle("Turn-4 takeover by model (escalating script)"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_takeover_by_model.png"), dpi=160); plt.close(fig)

# Fig 3: judge vs human labels on the 52 gold turns
gold = {(g["key"], g["turn"]): g["gold"] for g in json.load(open(os.path.join(ROOT, "data/gold/gold_human_v2_52.json")))}
fig, ax = plt.subplots(1, len(judges), figsize=(4.2 * len(judges), 3.8), squeeze=False)
for a, jd in zip(ax[0], judges):
    js = {(r["key"], r["turn"]): r["score"] for r in rows if r["judge"] == jd}
    pairs = [(g, js[k]) for k, g in gold.items() if k in js]
    M = [[sum(p == (x, y) for p in pairs) for y in (-1, 0, 1)] for x in (-1, 0, 1)]
    a.imshow(M, cmap="Blues")
    for i in range(3):
        for j in range(3): a.text(j, i, M[i][j], ha="center", va="center", color="white" if M[i][j] > 8 else "black")
    a.set(xticks=range(3), yticks=range(3), xticklabels=["-1", "0", "+1"], yticklabels=["-1", "0", "+1"], xlabel="judge", ylabel="human",
          title=f"{jd}\nexact {sum(x == y for x, y in pairs)/len(pairs):.2f}, w-kappa {weighted_kappa(pairs):.2f} (n={len(pairs)})")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_judge_vs_human.png"), dpi=160); plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))

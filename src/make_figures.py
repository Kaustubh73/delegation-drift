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
ax[0][0].set_ylabel("takeover rate at turn 4 (%)"); ax[0][0].set_ylim(0, 100)
h, l = ax[0][0].get_legend_handles_labels()
fig.suptitle("Turn-4 takeover by model (escalating script)"); fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.legend(h, l, title="system prompt", loc="lower center", ncol=3, frameon=False)
fig.savefig(os.path.join(OUT, "fig2_takeover_by_model.png"), dpi=160); plt.close(fig)

# Fig 3: judge vs human labels on the 52 gold turns
gold = {(g["key"], g["turn"]): g["gold"] for g in json.load(open(os.path.join(ROOT, "data/gold/gold_human_v2_52.json")))}
judges_gold = [jd for jd in judges if any((r["key"], r["turn"]) in gold for r in rows if r["judge"] == jd)]
if judges_gold:
    fig, ax = plt.subplots(1, len(judges_gold), figsize=(4.2 * len(judges_gold), 3.8), squeeze=False)
    for a, jd in zip(ax[0], judges_gold):
        js = {(r["key"], r["turn"]): r["score"] for r in rows if r["judge"] == jd}
        pairs = [(g, js[k]) for k, g in gold.items() if k in js]
        M = [[sum(p == (x, y) for p in pairs) for y in (-1, 0, 1)] for x in (-1, 0, 1)]
        a.imshow(M, cmap="Blues")
        for i in range(3):
            for j in range(3): a.text(j, i, M[i][j], ha="center", va="center", color="white" if M[i][j] > 8 else "black")
        a.set(xticks=range(3), yticks=range(3), xticklabels=["-1", "0", "+1"], yticklabels=["-1", "0", "+1"], xlabel="judge", ylabel="human",
              title=f"{jd}\nexact {sum(x == y for x, y in pairs)/len(pairs):.2f}, w-kappa {weighted_kappa(pairs):.2f} (n={len(pairs)})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_judge_vs_human.png"), dpi=160); plt.close(fig)

# Fig 4-6: 11-model parameter ladder (if data/runs_ladder/analysis_exp2_out.txt exists)
ladder_src = os.path.join(ROOT, "data/runs_ladder/analysis_exp2_out.txt")
if os.path.exists(ladder_src):
    import re
    LADDER_NAMES = {
        "qwen2.5_0.5b-instruct": "Qwen2.5 0.5B", "qwen2.5_1.5b-instruct": "Qwen2.5 1.5B",
        "qwen2.5_3b-instruct": "Qwen2.5 3B", "gemma4_e4b": "Gemma4 e4B",
        "qwen2.5_7b-instruct": "Qwen2.5 7B", "gemma4_12b": "Gemma4 12B",
        "qwen2.5_14b-instruct": "Qwen2.5 14B", "qwen2.5_14b-instruct-awq": "Qwen2.5 14B (4-bit)",
        "qwen3.6_27b": "Qwen3.6 27B", "qwen2.5_32b-instruct-awq": "Qwen2.5 32B (4-bit)",
        "qwen2.5_72b-instruct-awq": "Qwen2.5 72B (4-bit)",
    }
    ldata, cur_j = {}, None
    for line in open(ladder_src).read().splitlines():
        m = re.match(r"=+ judge (\S+)", line)
        if m:
            cur_j = m.group(1); ldata[cur_j] = []; continue
        parts = [p.strip() for p in line.split("|")]
        if cur_j and len(parts) == 7 and parts[0] in LADDER_NAMES:
            t4 = [int(x) for x in parts[3].split("/")]
            ci = re.findall(r"[-+]?\d+", parts[5])
            ldata[cur_j].append({"model": parts[0], "params": float(parts[1]),
                                 "t4": t4, "eng": int(ci[0]), "lo": int(ci[1]), "hi": int(ci[2])})

    if "gemma4:12b" in ldata and "gpt-oss:20b" in ldata:
        j_col = {"gemma4:12b": "#1f5fa8", "gpt-oss:20b": "#d97a1c"}
        j_lbl = {"gemma4:12b": "Gemma4 12B judge", "gpt-oss:20b": "gpt-oss 20B judge"}
        models_l = [r["model"] for r in ldata["gemma4:12b"]]

        # Ladder Engagement effect
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for off, (jd, lrows) in zip((-0.16, 0.16), ldata.items()):
            for i, r in enumerate(lrows):
                hollow = "awq" in r["model"]
                ax.errorbar(r["eng"], i + off, xerr=[[r["eng"] - r["lo"]], [r["hi"] - r["eng"]]],
                            fmt="o", ms=4.5, color=j_col[jd], capsize=0, lw=1.2,
                            mfc="white" if hollow else j_col[jd],
                            label=j_lbl[jd] if i == 0 else None)
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_yticks(range(len(models_l))); ax.set_yticklabels([LADDER_NAMES[m] for m in models_l])
        ax.invert_yaxis()
        ax.set_xlabel("Engagement minus neutral, final-turn takeover (percentage points)")
        ax.plot([], [], "o", mfc="white", mec="#555555", ms=4.5, label="open marker = 4-bit model")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_ladder_engagement.png"), dpi=200); plt.close(fig)

        # Ladder Autonomy effect
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for off, (jd, lrows) in zip((-0.16, 0.16), ldata.items()):
            for i, r in enumerate(lrows):
                neut, aut = r["t4"][0], r["t4"][3]
                ax.plot([neut, aut], [i + off] * 2, color=j_col[jd], lw=1.2, alpha=0.7)
                ax.plot(neut, i + off, "o", mfc="white", mec=j_col[jd], ms=4.5)
                ax.plot(aut, i + off, "o", color=j_col[jd], ms=4.5, label=j_lbl[jd] if i == 0 else None)
        ax.plot([], [], "o", mfc="white", mec="#555555", ms=4.5, label="neutral prompt (open)")
        ax.plot([], [], "o", color="#555555", ms=4.5, label="autonomy prompt (filled)")
        ax.set_yticks(range(len(models_l))); ax.set_yticklabels([LADDER_NAMES[m] for m in models_l])
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Final-turn takeover (%)")
        ax.legend(frameon=False, loc="lower left", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_ladder_autonomy.png"), dpi=200); plt.close(fig)

print("wrote", sorted(os.listdir(OUT)))

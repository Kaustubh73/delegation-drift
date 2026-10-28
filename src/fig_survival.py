"""Fig 1: pooled 5-model persistent arm, share of trajectories with a takeover (-1) by turn k. Solid = both judges (lower bound), dashed = either judge. python3 -I fig_survival.py"""
import glob, json, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
J = ["judge_qwen2.5_14b-instruct.jsonl", "judge_gemma3_12b.jsonl"]; K = 8
first = collections.defaultdict(lambda: [None, None]); cond = {}
for d in glob.glob("data/runs_adaptive/tierB_*/"):
    if "sim32b" in d: continue
    for l in open(d + "transcripts.jsonl"):
        r = json.loads(l)
        if r["script"] == "persistent": cond[r["key"]] = r["condition"]
    for i, jf in enumerate(J):
        for l in open(d + jf):
            j = json.loads(l)
            if j["key"] in cond and j["score"] == -1 and (first[j["key"]][i] is None or j["turn"] < first[j["key"]][i]): first[j["key"]][i] = j["turn"]
fig, ax = plt.subplots(figsize=(6, 4)); col = dict(engagement="#c0392b", neutral="#7f8c8d", autonomy="#2471a3")
for c in col:
    ks = [k for k in cond if cond[k] == c]
    both = [100 * sum(all(t is not None and t <= k for t in first[x]) for x in ks) / len(ks) for k in range(1, K + 1)]
    either = [100 * sum(any(t is not None and t <= k for t in first[x]) for x in ks) / len(ks) for k in range(1, K + 1)]
    ax.plot(range(1, K + 1), both, color=col[c], marker="o", label=f"{c} (both judges)"); ax.plot(range(1, K + 1), either, color=col[c], ls="--", alpha=.6)
ax.set(xlabel="turn k", ylabel="% trajectories with takeover by turn k", ylim=(0, 100), title=f"Persistent arm, 5 models pooled (n={len(ks)} per prompt)")
ax.legend(fontsize=8); ax.text(.99, .02, "dashed: either judge", transform=ax.transAxes, ha="right", fontsize=7)
plt.tight_layout(); plt.savefig("results/fig_survival.png", dpi=160)
print({c: sum(1 for k in cond if cond[k] == c) for c in col})

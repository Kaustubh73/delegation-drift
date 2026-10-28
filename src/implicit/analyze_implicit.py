"""Takeover by condition and contrasts vs neutral, from judged implicit-condition runs.

    python3 analyze_implicit.py runs/*        # after `drift_eval.py judge` for each judge
Bootstrap is over the 8 scenarios (as in the main analysis); a scenario's value is the mean over
models, paraphrases and seeds. Takeover = judge score -1. Reports each judge separately.
"""
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from drift_eval import load_jsonl  # noqa: E402

CONDS = ["implicit", "neutral", "engagement", "autonomy"]


def load(dirs):
    rows = []
    for d in dirs:
        meta = {r["key"]: r for r in load_jsonl(os.path.join(d, "transcripts.jsonl"))}
        for jf in sorted(f for f in os.listdir(d) if f.startswith("judge_")):
            for j in load_jsonl(os.path.join(d, jf)):
                m = meta.get(j["key"])
                if m and j["score"] is not None:
                    rows.append({**{k: m[k] for k in ("model", "script", "scenario", "condition", "paraphrase")},
                                 "turn": j["turn"], "judge": j["judge"], "take": j["score"] == -1})
    return rows


def per_scenario(R, **f):
    acc = defaultdict(list)
    for r in R:
        if all(r[k] == v for k, v in f.items()):
            acc[r["scenario"]].append(r["take"])
    return {s: sum(v) / len(v) for s, v in acc.items()}


def boot_diff(a, b, n=5000, rng=random.Random(1)):
    sc = sorted(set(a) & set(b))
    diffs = sorted(sum(a[s] - b[s] for s in x) / len(x) for x in
                   (rng.choices(sc, k=len(sc)) for _ in range(n)))
    return sum(a[s] - b[s] for s in sc) / len(sc), diffs[int(.025 * n)], diffs[int(.975 * n)]


def main(dirs):
    rows = load(dirs)
    for judge in sorted({r["judge"] for r in rows}):
        R = [r for r in rows if r["judge"] == judge]
        print(f"\n== judge {judge} ==")
        for script, turn in (("escalating", 1), ("escalating", 4), ("control", 4)):
            vals = {c: per_scenario(R, script=script, turn=turn, condition=c) for c in CONDS}
            cells = "  ".join(f"{c}={100 * sum(v.values()) / max(len(v), 1):4.0f}%" for c, v in vals.items())
            print(f"{script:10s} T{turn}: {cells}")
        v4 = {c: per_scenario(R, script="escalating", turn=4, condition=c) for c in CONDS}
        for a, b in (("implicit", "neutral"), ("engagement", "neutral"), ("implicit", "engagement")):
            if v4[a] and v4[b]:
                m, lo, hi = boot_diff(v4[a], v4[b])
                print(f"  T4 {a} - {b}: {100 * m:+.0f} pp  [{100 * lo:+.0f}, {100 * hi:+.0f}]")


if __name__ == "__main__":
    main(sys.argv[1:])

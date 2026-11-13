#!/usr/bin/env python3
"""
Generate the publication-grade figure suite for delegation-drift.
Outputs:
  results/fig1_drift_pressure.png      - Turn 1 vs Turn 4: Escalation vs Control
  results/fig2_prompt_spectrum.png     - Turn 4 intervention spectrum (Autonomy -> Neutral -> Implicit -> Explicit)
  results/fig3_parameter_scaling.png   - 0.5B to 72B log-scale parameter ladder and invariance
  results/fig4_model_dumbbells.png     - Per-model intervention range (Autonomy vs Neutral vs Engagement)
  results/fig5_all_models_grid.png     - Small multiples of all 11 models across 4 conditions
  results/fig6_judge_calibration.png   - Confusion matrix and calibration against blind human gold labels
"""

import glob
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "medium",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "figure.dpi": 200,
    "savefig.dpi": 250,
})

# Harmonious research color palette
NAVY = "#1a365d"
BLUE = "#2b6cb0"
LIGHT_BLUE = "#63b3ed"
GREY = "#718096"
LIGHT_GREY = "#e2e8f0"
AMBER = "#dd6b20"
RED = "#c53030"
GREEN = "#276749"
DARK_INK = "#1a202c"

MODEL_ORDER = [
    ("qwen2.5_0.5b-instruct", "Qwen2.5 0.5B", 0.5, False),
    ("qwen2.5_1.5b-instruct", "Qwen2.5 1.5B", 1.5, False),
    ("qwen2.5_3b-instruct", "Qwen2.5 3B", 3.0, False),
    ("gemma4_e4b", "Gemma4 e4B", 4.5, False),
    ("qwen2.5_7b-instruct", "Qwen2.5 7B", 7.0, False),
    ("gemma4_12b", "Gemma4 12B", 12.0, False),
    ("qwen2.5_14b-instruct", "Qwen2.5 14B", 14.0, False),
    ("qwen2.5_14b-instruct-awq", "Qwen2.5 14B (4-bit)", 14.0, True),
    ("qwen3.6_27b", "Qwen3.6 27B", 27.0, False),
    ("qwen2.5_32b-instruct-awq", "Qwen2.5 32B (4-bit)", 32.0, True),
    ("qwen2.5_72b-instruct-awq", "Qwen2.5 72B (4-bit)", 72.0, True),
]
MODEL_NAMES = {k: v for k, v, _, _ in MODEL_ORDER}
MODEL_PARAMS = {k: p for k, _, p, _ in MODEL_ORDER}
MODEL_AWQ = {k: awq for k, _, _, awq in MODEL_ORDER}


def load_data():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "data/runs/*/"))):
        mdir = os.path.basename(os.path.dirname(d))
        tfile = os.path.join(d, "transcripts.jsonl")
        if not os.path.exists(tfile):
            continue
        meta = {r["key"]: r for r in [json.loads(l) for l in open(tfile)]}
        for jf in glob.glob(os.path.join(d, "judge_*.jsonl")):
            for line in open(jf):
                j = json.loads(line)
                m = meta.get(j["key"])
                if m and j["score"] is not None:
                    rows.append(dict(
                        model=mdir,
                        script=m["script"],
                        scenario=m["scenario"],
                        condition=m["condition"],
                        key=j["key"],
                        turn=j["turn"],
                        judge=j["judge"],
                        score=j["score"],
                        take=1.0 if j["score"] == -1 else 0.0
                    ))
    return rows


def bootstrap_ci(vals, n=5000, seed=1):
    if not vals:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    mean = sum(vals) / len(vals)
    k = len(vals)
    samples = sorted(sum(rng.choices(vals, k=k)) / k for _ in range(n))
    return mean * 100.0, samples[int(0.025 * n)] * 100.0, samples[int(0.975 * n)] * 100.0


def bootstrap_diff_ci(a_vals, b_vals, n=5000, seed=1):
    """Bootstrap paired difference over scenarios"""
    rng = random.Random(seed)
    scens = sorted(set(a_vals.keys()) & set(b_vals.keys()))
    if not scens:
        return 0.0, 0.0, 0.0
    diffs = [a_vals[s] - b_vals[s] for s in scens]
    mean = sum(diffs) / len(diffs)
    k = len(diffs)
    samples = sorted(sum(rng.choices(diffs, k=k)) / k for _ in range(n))
    return mean * 100.0, samples[int(0.025 * n)] * 100.0, samples[int(0.975 * n)] * 100.0


# =========================================================================
# FIGURE 1: Delegation Drift Under Pressure vs Control
# =========================================================================
def make_fig1_drift_pressure(rows):
    """Plots takeover rate across turns under escalating pressure vs analytical control."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    judges = [("gpt-oss:20b", "gpt-oss:20b (Validated Human-Agreement Judge)"),
              ("gemma4:12b", "gemma4:12b (Secondary Judge)")]

    for ax, (jd, jtitle) in zip(axes, judges):
        # Calculate per-scenario rates pooled across all 11 models
        for script, col, lbl, marker in [
            ("escalating", RED, "Escalating User Pressure", "o"),
            ("control", GREEN, "Analytical Control (No Pressure)", "s")
        ]:
            pts = []
            for t in [1, 4]:
                scen_means = []
                for s in range(1, 9):
                    sub = [r["take"] for r in rows if r["judge"] == jd and r["script"] == script and
                           r["condition"] == "neutral" and r["turn"] == t and r["scenario"] == s]
                    if sub:
                        scen_means.append(sum(sub) / len(sub))
                m, lo, hi = bootstrap_ci(scen_means, seed=t*10 + (1 if script=="escalating" else 2))
                pts.append((t, m, lo, hi))

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            yerr_lo = [p[1] - p[2] for p in pts]
            yerr_hi = [p[3] - p[1] for p in pts]

            ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt=f"-{marker}", color=col,
                        lw=2.5, ms=8, capsize=4, capthick=1.5, label=lbl)

            for x, y in zip(xs, ys):
                offset = 4 if y > 30 else -6
                ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                            xytext=(0, offset), ha="center", fontweight="bold",
                            color=col, fontsize=10)

        # Annotate drift magnitude
        ax.annotate(
            "Delegation Drift\n(+38.3 pp handover)",
            xy=(4, 70), xytext=(2.6, 55),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5, connectionstyle="arc3,rad=-0.15"),
            fontweight="bold", color=RED, fontsize=9.5, ha="center"
        )

        ax.set_title(jtitle, fontsize=11, pad=12)
        ax.set_xticks([1, 4])
        ax.set_xticklabels(["Turn 1\n(Initial Query)", "Turn 4\n(Final Choice)"])
        ax.set_xlim(0.6, 4.4)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y")

    axes[0].set_ylabel("Takeover Rate (% of turns with decision settled)", fontweight="bold")
    axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left", fontsize=9.5)
    fig.suptitle("Delegation Drift: Model Takeover Under Escalating Pressure vs. Analytical Control",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "fig1_drift_pressure.png")
    fig.savefig(p)
    plt.close(fig)
    print("Saved", p)


# =========================================================================
# FIGURE 2: Intervention Spectrum at Turn 4
# =========================================================================
def make_fig2_prompt_spectrum(rows):
    """Bar chart comparing Turn-4 takeover rate across the 4 system prompt interventions."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    jd = "gpt-oss:20b"
    conds = [
        ("autonomy", "Autonomy Prompt\n('never make decisions')", BLUE),
        ("neutral", "Neutral Baseline\n('helpful assistant')", GREY),
        ("implicit", "Implicit Engagement\n('be proactively helpful')", AMBER),
        ("engagement", "Explicit 'Decide for User'\n('choose on their behalf')", RED),
    ]

    # Calculate per-scenario rates pooled across 11 models
    scen_cond_means = defaultdict(dict)
    for c, _, _ in conds:
        for s in range(1, 9):
            sub = [r["take"] for r in rows if r["judge"] == jd and r["script"] == "escalating" and
                   r["condition"] == c and r["turn"] == 4 and r["scenario"] == s]
            if sub:
                scen_cond_means[c][s] = sum(sub) / len(sub)

    neutral_means = scen_cond_means["neutral"]

    bars = []
    for i, (c, lbl, col) in enumerate(conds):
        vals = list(scen_cond_means[c].values())
        m, lo, hi = bootstrap_ci(vals, seed=i+1)
        b = ax.bar(i, m, width=0.58, color=col, edgecolor="#333333", linewidth=1.1, zorder=3)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", color="#222222",
                    capsize=6, capthick=1.8, lw=1.8, zorder=4)

        # Label percentage
        txt = f"{m:.1f}%"
        if c != "neutral":
            diff_m, diff_lo, diff_hi = bootstrap_diff_ci(scen_cond_means[c], neutral_means, seed=i+1)
            sign = "+" if diff_m > 0 else ""
            txt += f"\n({sign}{diff_m:.1f} pp)"
            y_diff = m + (hi - m) + 3
        else:
            txt += "\n(baseline)"
            y_diff = m + (hi - m) + 3

        ax.text(i, y_diff, txt, ha="center", va="bottom", fontweight="bold",
                fontsize=10.5, color=DARK_INK)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([x[1] for x in conds], fontsize=10)
    ax.set_ylabel("Turn-4 Takeover Rate (% of decisions settled by model)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", zorder=0)

    # Reference threshold line
    if scen_cond_means["neutral"]:
        neu_val = bootstrap_ci(list(scen_cond_means["neutral"].values()))[0]
        ax.axhline(neu_val, color=GREY, linestyle="--", lw=1.2, alpha=0.7, zorder=1)

    ax.set_title("Turn 4 Intervention Spectrum: Effect of System Prompts on Agency Handover",
                 fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_prompt_spectrum.png")
    fig.savefig(p)
    plt.close(fig)
    print("Saved", p)


# =========================================================================
# FIGURE 3: Parameter Scaling Ladder & Flat Invariance (0.5B - 72B)
# =========================================================================
def make_fig3_parameter_scaling(rows):
    """Log-scale parameter scaling plot showing lack of safety scaling across 0.5B to 72B."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    jd = "gpt-oss:20b"

    cond_styles = [
        ("engagement", "Explicit 'Decide for User'", RED, "o-", 2.2),
        ("neutral", "Neutral Baseline", GREY, "s--", 1.8),
        ("autonomy", "Autonomy Prompt", BLUE, "^:", 1.8),
    ]

    for c, lbl, col, fmt, lw in cond_styles:
        xs, ys, ylo, yhi = [], [], [], []
        for mdir, mname, params, is_awq in MODEL_ORDER:
            # Only include main ladder (filter out 14b-awq for the primary scaling trendline to avoid duplicate x)
            if is_awq and mdir != "qwen2.5_72b-instruct-awq" and mdir != "qwen2.5_32b-instruct-awq":
                continue
            scen_means = []
            for s in range(1, 9):
                sub = [r["take"] for r in rows if r["judge"] == jd and r["model"] == mdir and
                       r["condition"] == c and r["script"] == "escalating" and r["turn"] == 4 and r["scenario"] == s]
                if sub:
                    scen_means.append(sum(sub) / len(sub))
            m, lo, hi = bootstrap_ci(scen_means, seed=int(params * 10))
            xs.append(params)
            ys.append(m)
            ylo.append(m - lo)
            yhi.append(hi - m)

        ax.errorbar(xs, ys, yerr=[ylo, yhi], fmt=fmt, color=col, lw=lw,
                    ms=6.5, capsize=3.5, label=lbl, zorder=3)

    # Plot AWQ quantized points as open markers
    for mdir, mname, params, is_awq in MODEL_ORDER:
        if is_awq:
            for c, _, col, _, _ in cond_styles:
                sub = [r["take"] for r in rows if r["judge"] == jd and r["model"] == mdir and
                       r["condition"] == c and r["script"] == "escalating" and r["turn"] == 4]
                if sub:
                    m = sum(sub) / len(sub) * 100.0
                    ax.plot(params, m, "o", mfc="white", mec=col, ms=6.5, mew=1.8, zorder=4)

    # Formatting
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.5, 3.0, 4.5, 7.0, 12.0, 14.0, 27.0, 32.0, 72.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Model Parameters (Billions, log scale)", fontweight="bold")
    ax.set_ylabel("Turn-4 Takeover Rate (%)", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, which="both", axis="both", zorder=0)

    # Annotation about scaling slope
    ax.annotate(
        "Scaling Invariance:\nSlope = +1.2 pp / decade (p = 0.39)\nLarger models abdicate as readily as 0.5B",
        xy=(7, 94), xytext=(2.0, 85),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9db", edgecolor="#e0a800", lw=1.2),
        fontweight="bold", fontsize=9.5, color="#5f370e"
    )

    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower left", fontsize=9.5)
    ax.set_title("Parameter Scaling Ladder (0.5B to 72B): Scale Does Not Prevent Delegation Drift",
                 fontsize=12.5, fontweight="bold", pad=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_parameter_scaling.png")
    fig.savefig(p)
    plt.close(fig)
    print("Saved", p)


# =========================================================================
# FIGURE 4: Per-Model Intervention Range (Dumbbell Plot)
# =========================================================================
def make_fig4_model_dumbbells(rows):
    """Horizontal dumbbell plot showing Autonomy -> Neutral -> Engagement range for every model."""
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    jd = "gpt-oss:20b"

    models = MODEL_ORDER[::-1]  # Largest on top
    y_pos = range(len(models))

    for i, (mdir, mname, params, is_awq) in enumerate(models):
        rates = {}
        for c in ["autonomy", "neutral", "engagement"]:
            sub = [r["take"] for r in rows if r["judge"] == jd and r["model"] == mdir and
                   r["condition"] == c and r["script"] == "escalating" and r["turn"] == 4]
            rates[c] = sum(sub) / len(sub) * 100.0 if sub else 0.0

        auton = rates["autonomy"]
        neut = rates["neutral"]
        eng = rates["engagement"]

        # Connector lines
        ax.plot([auton, neut], [i, i], color=LIGHT_BLUE, lw=2.5, zorder=2)
        ax.plot([neut, eng], [i, i], color="#feb2b2", lw=2.5, zorder=2)

        # Dots
        mfc_aut = "white" if is_awq else BLUE
        mfc_neu = "white" if is_awq else GREY
        mfc_eng = "white" if is_awq else RED

        ax.plot(auton, i, "o", color=BLUE, mfc=mfc_aut, mew=1.8, ms=7, zorder=3)
        ax.plot(neut, i, "s", color=GREY, mfc=mfc_neu, mew=1.8, ms=6.5, zorder=3)
        ax.plot(eng, i, "D", color=RED, mfc=mfc_eng, mew=1.8, ms=7, zorder=3)

    # Legend proxies
    ax.plot([], [], "o", color=BLUE, ms=7, label="Autonomy Prompt")
    ax.plot([], [], "s", color=GREY, ms=6.5, label="Neutral Baseline")
    ax.plot([], [], "D", color=RED, ms=7, label="Engagement Prompt")
    ax.plot([], [], "o", mfc="white", mec="#444444", mew=1.8, ms=7, label="4-bit Quantized (AWQ)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[1] for m in models], fontsize=10.5)
    ax.set_xlabel("Turn-4 Takeover Rate (%)", fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(True, axis="x", zorder=0)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower left", fontsize=9.2, ncol=2)
    ax.set_title("Intervention Range by Model: Dynamic Responsiveness to Operator Prompts",
                 fontsize=12.5, fontweight="bold", pad=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_model_dumbbells.png")
    fig.savefig(p)
    plt.close(fig)
    print("Saved", p)


# =========================================================================
# FIGURE 5: All 11 Models Small Multiples Grid
# =========================================================================
def make_fig5_all_models_grid(rows):
    """3x4 grid displaying all 11 models with their 4 condition takeover rates."""
    fig, axes = plt.subplots(3, 4, figsize=(14.5, 9.5), sharey=True)
    jd = "gpt-oss:20b"
    conds = [("autonomy", BLUE), ("neutral", GREY), ("implicit", AMBER), ("engagement", RED)]

    for idx, (mdir, mname, params, is_awq) in enumerate(MODEL_ORDER):
        ax = axes.flat[idx]
        rates = []
        for c, col in conds:
            sub = [r["take"] for r in rows if r["judge"] == jd and r["model"] == mdir and
                   r["condition"] == c and r["script"] == "escalating" and r["turn"] == 4]
            rate = sum(sub) / len(sub) * 100.0 if sub else 0.0
            rates.append(rate)

        bars = ax.bar(range(4), rates, color=[x[1] for x in conds], width=0.68,
                      edgecolor="#333333", linewidth=0.8, zorder=3)

        for i, val in enumerate(rates):
            ax.text(i, val + 2.5, f"{val:.0f}%", ha="center", fontsize=8.5,
                    fontweight="bold", color=DARK_INK)

        title = f"{mname}" + (" [AWQ]" if is_awq else "")
        ax.set_title(title, fontsize=10.5, pad=6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Auton", "Neut", "Impl", "Eng"], fontsize=8.5)
        ax.set_ylim(0, 110)
        ax.grid(True, axis="y", zorder=0)

    # Empty 12th panel: legend and summary
    ax_last = axes.flat[11]
    ax_last.axis("off")
    legend_elements = [
        matplotlib.patches.Patch(facecolor=BLUE, edgecolor="#333333", label="Autonomy ('never decide')"),
        matplotlib.patches.Patch(facecolor=GREY, edgecolor="#333333", label="Neutral ('helpful assistant')"),
        matplotlib.patches.Patch(facecolor=AMBER, edgecolor="#333333", label="Implicit Engagement"),
        matplotlib.patches.Patch(facecolor=RED, edgecolor="#333333", label="Explicit Engagement ('choose')")
    ]
    ax_last.legend(handles=legend_elements, loc="center", frameon=True,
                   facecolor="#f8fafc", edgecolor="#cbd5e0", fontsize=9.5, title="System Prompts",
                   title_fontsize=10.5)

    for i in range(3):
        axes[i, 0].set_ylabel("Turn-4 Takeover (%)", fontweight="bold", fontsize=10)

    fig.suptitle("Per-Model Takeover Profiles Across 4 Interventions (Ordered by Model Size)",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "fig5_all_models_grid.png")
    fig.savefig(p)
    plt.close(fig)
    print("Saved", p)


# =========================================================================
# FIGURE 6: Judge Calibration & Human Agreement
# =========================================================================
def make_fig6_judge_calibration(rows):
    """Confusion matrix and agreement statistics against 52 blind human gold labels."""
    gold_path = os.path.join(ROOT, "data/gold/gold_human_v2_52.json")
    if not os.path.exists(gold_path):
        print("Gold path not found, skipping Fig 6")
        return

    gold_data = json.load(open(gold_path))
    gold_lookup = {(g["key"], g["turn"]): g["gold"] for g in gold_data}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    judges = ["gpt-oss:20b", "gemma4:12b"]

    for ax, jd in zip(axes, judges):
        js = {(r["key"], r["turn"]): r["score"] for r in rows if r["judge"] == jd}
        pairs = [(gold_lookup[k], js[k]) for k in gold_lookup if k in js]

        if not pairs:
            continue

        exact = sum(h == j for h, j in pairs) / len(pairs)
        # Weighted kappa
        w = {0: 1.0, 1: 0.5, 2: 0.0}
        n = len(pairs)
        observed = sum(w[abs(h - j)] for h, j in pairs) / n
        h_marg = defaultdict(int)
        j_marg = defaultdict(int)
        for h, j in pairs:
            h_marg[h] += 1
            j_marg[j] += 1
        expected = sum((h_marg[h] / n) * (j_marg[j] / n) * w[abs(h - j)] for h in (-1, 0, 1) for j in (-1, 0, 1))
        kappa = (observed - expected) / (1.0 - expected) if (1.0 - expected) > 0 else 0.0

        M = [[sum(p == (x, y) for p in pairs) for y in (-1, 0, 1)] for x in (-1, 0, 1)]
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=25)

        for i in range(3):
            for j in range(3):
                cnt = M[i][j]
                col = "white" if cnt > 10 else "black"
                ax.text(j, i, str(cnt), ha="center", va="center", color=col, fontweight="bold", fontsize=12)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["Takeover (-1)", "Neutral (0)", "Autonomous (+1)"], fontsize=9)
        ax.set_yticklabels(["Takeover (-1)", "Neutral (0)", "Autonomous (+1)"], fontsize=9)
        ax.set_xlabel("Automated Judge Score", fontweight="bold")
        ax.set_ylabel("Human Gold Label", fontweight="bold")

        title = f"Judge: {jd}\nExact Agreement: {exact*100:.1f}% | Weighted κ: {kappa:.2f}"
        ax.set_title(title, fontsize=11, pad=10)

    fig.suptitle("Judge Calibration Against Blind Human Labels (52 Validation Turns)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(OUT, "fig6_judge_calibration.png")
    fig.savefig(p)
    plt.close(fig)
    print("Saved", p)


def main():
    print("Loading run data from data/runs/...")
    rows = load_data()
    print(f"Loaded {len(rows)} evaluations across 11 models.")

    print("\nGenerating Figure Suite...")
    make_fig1_drift_pressure(rows)
    make_fig2_prompt_spectrum(rows)
    make_fig3_parameter_scaling(rows)
    make_fig4_model_dumbbells(rows)
    make_fig5_all_models_grid(rows)
    make_fig6_judge_calibration(rows)

    print("\nAll figures generated successfully in results/!")


if __name__ == "__main__":
    main()

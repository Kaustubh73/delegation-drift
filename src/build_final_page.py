#!/usr/bin/env python3
"""Build figures/index.html (self-contained, base64 PNGs) from figures/final/. Run from repo root after make_final_figures.py."""
import base64, json
from pathlib import Path
R = Path(__file__).resolve().parent.parent; F = R / "figures" / "final"
S = json.load(open(F / "stats.json"))
img = lambda n: "data:image/png;base64," + base64.b64encode((F / f"{n}.png").read_bytes()).decode()
d = S["drift"]["gpt-oss:20b"]; dg = S["drift"]["gemma4:12b"]
e = S["effects"]["gpt-oss:20b"]; eg = S["effects"]["gemma4:12b"]
P = S["persistent"]; C = S["calibration"]["gpt-oss:20b"]
fmt = lambda x: f"{x[0]:+.0f} [{x[1]:+.0f}, {x[2]:+.0f}]"
FIGS = [
 ("f1", "F1_protocol", "Protocol", "Four-turn escalation, four system prompts, one control arm",
  "Each conversation runs one of 8 decision scenarios. The user moves from comparing options to handing over the choice by turn 4. Four system prompts (neutral, implicit, engagement, autonomy) are crossed with the scenario, and a control arm asks only analytical questions. One label per reply is assigned blind to condition.",
  "n per model: 64 escalating conversations per prompt, 16 control. 11 models, 14,061 judged turns (turns 1 and 4 only)."),
 ("f2", "F2_headline_drift", "Drift", "Takeover rises with pressure and stays low without it",
  f"Under the neutral prompt, takeover goes from {d['escalating_T1'][0]:.0f}% at turn 1 to {d['escalating_T4'][0]:.0f}% at turn 4 (gpt-oss:20b) and from {dg['escalating_T1'][0]:.0f}% to {dg['escalating_T4'][0]:.0f}% (gemma4:12b). The control arm ends at {d['control_T4'][0]:.0f}% and {dg['control_T4'][0]:.0f}%.",
  "Turn-1 takeover is already about 30% in both arms, so the drift is a rise from a non-trivial start, not from near zero."),
 ("f3", "F3_prompt_effects", "Prompts", "Autonomy instructions move takeover far more than engagement ones",
  f"Engagement adds {fmt(e['engagement'])} pp (gpt-oss) and {fmt(eg['engagement'])} pp (gemma4). Autonomy removes {fmt(e['autonomy'])} and {fmt(eg['autonomy'])} pp. The implicit prompt is indistinguishable from neutral ({fmt(e['implicit'])}, {fmt(eg['implicit'])}).",
  "Pooled over 11 models; 95% bootstrap CI over 8 scenarios. The two judges agree on direction and size."),
 ("f4", "F4_per_model_effects", "Per model", "The autonomy effect is large in some models and absent in others",
  "Gemma 4 e4B, Gemma 4 12B, Qwen3.6 27B and Qwen2.5 14B (4-bit) drop by roughly 30 to 60 pp under the autonomy prompt. Qwen2.5 0.5B, 3B and 32B barely respond. Engagement effects are small and mostly overlap zero per model.",
  "Per-model CIs rest on 8 scenarios of 8 conversations each, so single-model engagement estimates are noisy."),
 ("f5", "F5_parameter_scaling", "Scale", "No trend with parameter count",
  f"Across the Qwen2.5 ladder (0.5B to 72B, 8 variants), the engagement effect changes by {S['slope']['gpt-oss:20b'][0]:+.1f} pp per 10x parameters (95% CI {S['slope']['gpt-oss:20b'][1]:+.1f} to {S['slope']['gpt-oss:20b'][2]:+.1f}) under gpt-oss and {S['slope']['gemma4:12b'][0]:+.1f} ({S['slope']['gemma4:12b'][1]:+.1f} to {S['slope']['gemma4:12b'][2]:+.1f}) under gemma4.",
  "Slope is an OLS fit on log10(params) over all 8 Qwen2.5 variants including the 14B AWQ. The README reports +1.0 and +1.2 and the old Figure 3 reported +2.8; see audit notes."),
 ("f6", "F6_scenario_vulnerability", "Scenarios", "Everyday personal choices hand over easily; deployment does not",
  "Under the neutral prompt, takeover is 91% for personal finances and 33% for software deployment. Lease negotiation shows the largest engagement effect (+20 pp) and the largest autonomy effect is on job offers (-43 pp).",
  "gpt-oss:20b judge, pooled over 11 models. Cell n = 88 turn-4 replies."),
 ("f7", "F7_judge_calibration", "Judges", "gpt-oss:20b tracks human labels best",
  f"On 52 hand-labelled replies, gpt-oss:20b reaches {C['agree']:.2f} exact agreement and weighted kappa {C['kappa_w']:.2f}, with takeover precision {C['prec']} and recall {C['rec']}. The two smaller judges never over-call takeover but miss 31% to 46% of real ones.",
  "One annotator, 52 items. Rows are the human label, columns the judge label."),
 ("f8", "F8_persistent_user", "Persistent user", "A user who keeps pushing eventually gets a takeover from most models",
  f"With a simulated user who pushes for up to 8 turns, {P['neutral']['both']:.0f}% of neutral trajectories contain a takeover both judges agree on ({P['neutral']['either']:.0f}% by either judge). Engagement gives {P['engagement']['both']:.0f}% ({P['engagement']['either']:.0f}%). Autonomy holds it to {P['autonomy']['both']:.0f}% ({P['autonomy']['either']:.0f}%).",
  "5 models (Gemma 3 12B, Gemma 4 e4B, Llama 3.1 8B, Mistral 7B, Qwen2.5 7B), 160 trajectories per prompt, judged by qwen2.5:14b and gemma3:12b. Exploratory."),
 ("f9", "F9_robustness", "Robustness", "Control is not always zero, and quantisation moves rates by about as much as the prompt effect",
  "Control-arm takeover at turn 4 ranges from 0% to 44% across models. For Qwen2.5 14B, the 4-bit variant has 83% neutral takeover against 69% for bf16, a 14 pp gap, larger than the pooled engagement effect.",
  "Control uses 16 conversations per model, so each control bar moves in steps of about 6 pp."),
]
AUDIT = [
 ("README says control takeover is 0% in nearly every cell.", "Turn-4 neutral control rates are 0% to 44% (Qwen2.5 3B 44%, Qwen3.6 27B 31%, Qwen2.5 7B 25%). The statement holds for only 2 of 11 models. See F9a."),
 ("README says takeover is rare on Turn 1.", "Under the neutral prompt it is 32% (gpt-oss) and 28% (gemma4) at turn 1, and similar in the control arm. See F2."),
 ("Three different scaling slopes.", "Old Figure 3: +2.8 pp per decade. README: +1.0 and +1.2. Recomputed here from raw data: -0.1 and +0.1. All CIs span zero, so the conclusion stands, but the number to cite should be one regenerated from the script."),
 ("Only turns 1 and 4 are judged in the main grid.", "The README's takeover-by-turn language implies turns 2 and 3. Intermediate turns exist only in the persistent-user arm (F8), which uses different models and judges."),
 ("Old Figure 1 prints p < 0.001.", "No script in src/ computes that p-value. Use the bootstrap CIs instead."),
 ("Quantised variants are confounded with scale.", "32B and 72B are AWQ only, and 14B shows a 14 pp neutral gap between bf16 and AWQ. The ladder mixes size and precision."),
 ("Judge recall is below 1.", "gpt-oss:20b recall is 12/13 and the other two judges miss 31% to 46% of takeovers, so rates are lower bounds. Precision for gpt-oss is 12/14, so it also over-calls slightly."),
]
nav = "".join(f'<a href="#{f[0]}">{f[2]}</a>' for f in FIGS) + '<a href="#audit">Audit</a>'
cards = ""
for i, (id_, name, short, head, body, note) in enumerate(FIGS):
    cards += f'''<section class="fig" id="{id_}"><div class="meta"><span class="tag">F{i+1}</span><h2>{head}</h2><p>{body}</p><p class="note">{note}</p></div>
<figure><div class="plate"><img src="{img(name)}" alt="{head}" loading="lazy"></div></figure></section>\n'''
audit = "".join(f"<li><strong>{a}</strong><span>{b}</span></li>" for a, b in AUDIT)
html = f'''<title>Delegation Drift Figures</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&display=swap">
<style>
/* Layout: one narrow text column per figure beside a white plate, stacked on phones; sticky section nav on top. */
:root{{--bg:#F3F5F7;--surface:#FFFFFF;--fg:#14202B;--muted:#5A6978;--line:#D9DFE5;--accent:#C2410C;--accent-soft:#FBE9DF;--warn-bg:#FFF6E0;--warn-line:#E6C873;
--display:"Source Serif 4",Georgia,serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0F171F;--surface:#17222D;--fg:#E7ECF1;--muted:#9AA9B8;--line:#2A3846;--accent:#F28A52;--accent-soft:#3A2418;--warn-bg:#2A2412;--warn-line:#6B5A22;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0F171F;--surface:#17222D;--fg:#E7ECF1;--muted:#9AA9B8;--line:#2A3846;--accent:#F28A52;--accent-soft:#3A2418;--warn-bg:#2A2412;--warn-line:#6B5A22;color-scheme:dark}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 var(--body);padding-block:0 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
body::before{{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);background:var(--bg);z-index:30}}
.wrap{{max-width:1180px;margin:0 auto}}
header{{padding:40px 0 24px}}
.kicker{{font:500 12px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}}
h1{{font:700 clamp(28px,4.4vw,44px)/1.12 var(--display);margin:8px 0 14px;text-wrap:balance;max-width:20ch}}
.lede{{max-width:68ch;color:var(--muted);margin:0}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:28px 0 0}}
.stat{{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 16px;min-width:0}}
.stat b{{display:block;font:500 26px/1.1 var(--mono);font-variant-numeric:tabular-nums}}
.stat span{{display:block;color:var(--muted);font-size:13px;margin-top:6px}}
nav{{position:sticky;top:env(safe-area-inset-top,0px);z-index:20;background:var(--bg);border-block:1px solid var(--line);display:flex;gap:4px;overflow-x:auto;padding:8px 0;margin-bottom:8px}}
nav a{{white-space:nowrap;color:var(--muted);text-decoration:none;font-size:13px;font-weight:500;padding:6px 11px;border-radius:5px}}
nav a:hover,nav a:focus-visible{{background:var(--accent-soft);color:var(--accent);outline:none}}
.fig{{display:grid;grid-template-columns:minmax(0,340px) minmax(0,1fr);gap:32px;padding:44px 0;border-bottom:1px solid var(--line);scroll-margin-top:56px}}
.meta{{min-width:0}}
.tag{{font:500 12px var(--mono);color:var(--accent);letter-spacing:.06em}}
h2{{font:700 22px/1.25 var(--display);margin:6px 0 12px;text-wrap:balance}}
.meta p{{margin:0 0 10px}}
.note{{color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:10px;margin-top:14px!important}}
figure{{margin:0;min-width:0}}
.plate{{background:#fff;border:1px solid var(--line);border-radius:6px;padding:10px}}
.plate img{{display:block;width:100%;height:auto}}
#audit{{padding:44px 0 0;scroll-margin-top:56px}}
#audit h2{{font-size:24px}}
#audit ol{{list-style:none;padding:0;margin:20px 0 0;display:grid;gap:10px;max-width:860px}}
#audit li{{background:var(--warn-bg);border:1px solid var(--warn-line);border-radius:6px;padding:12px 16px;display:grid;gap:2px}}
#audit li span{{color:var(--muted)}}
footer{{margin-top:36px;color:var(--muted);font:12px var(--mono)}}
@media (max-width:860px){{.fig{{grid-template-columns:minmax(0,1fr);gap:16px;padding:32px 0}}}}
@media (prefers-reduced-motion:no-preference){{html{{scroll-behavior:smooth}}}}
</style>
<div class="wrap">
<header>
<div class="kicker">Delegation drift · 11 open-weight models · 8 scenarios</div>
<h1>When does an assistant start deciding for you?</h1>
<p class="lede">Figures regenerated from the raw judge files by <code>src/make_final_figures.py</code>. Takeover is a reply scored -1 by the judge. Intervals are 95% bootstrap CIs over the 8 scenarios. Nothing here has been published or peer reviewed.</p>
<div class="stats">
<div class="stat"><b>+{d['escalating_T4'][0]-d['escalating_T1'][0]:.0f} pp</b><span>neutral takeover, turn 1 to turn 4 (gpt-oss:20b)</span></div>
<div class="stat"><b>{e['engagement'][0]:+.0f} pp</b><span>engagement prompt vs neutral, CI {e['engagement'][1]:+.0f} to {e['engagement'][2]:+.0f}</span></div>
<div class="stat"><b>{e['autonomy'][0]:+.0f} pp</b><span>autonomy prompt vs neutral, CI {e['autonomy'][1]:+.0f} to {e['autonomy'][2]:+.0f}</span></div>
<div class="stat"><b>{C['kappa_w']:.2f}</b><span>weighted kappa, gpt-oss:20b vs 52 human labels</span></div>
</div>
</header>
<nav aria-label="Sections">{nav}</nav>
{cards}
<section id="audit"><h2>Audit notes: where the old page and README disagree with the data</h2>
<ol>{audit}</ol></section>
<footer>Generated from data/runs and data/runs_adaptive. PDFs of every figure are in figures/final/.</footer>
</div>
'''
(R / "figures" / "index.html").write_text(html)
print(len(html) / 1e6, "MB")

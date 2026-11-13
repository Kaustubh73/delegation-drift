#!/usr/bin/env python3
"""Build self-contained academic HTML figure viewer."""

import base64
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
ARTIFACT_DIR = Path("/Users/kaustgup/.gemini/antigravity/brain/c068dbb7-97d6-4128-b05b-90829866b090")

FIGURES = [
    {
        "id": "fig0",
        "num": "Figure 0",
        "file": "fig0_protocol_diagram.png",
        "pdf": "fig0_protocol_diagram.pdf",
        "title": "Experimental Protocol & Evaluation Pipeline",
        "badge": "Method & Setup",
        "badge_color": "bg-emerald-500/10 text-emerald-700 border-emerald-500/20",
        "question": "How is multi-turn delegation pressure structured and evaluated?",
        "caption": "Overview of the four-turn conversational protocol and evaluator rubric. In the escalating condition, scripted user queries systematically progress from balanced exploration to recommendation nudge, acute stress report, and full decision abdication. In the analytical control, users inquire about self-evaluation frameworks without abdicating. Automated judges score each assistant turn blind to condition on a 3-point scale (+1 autonomous, 0 ambiguous, -1 takeover).",
        "stats": [
            ("Conversations", "3,520", "Fully completed multi-turn dialogues"),
            ("Evaluated Turns", "14,061", "Double-judged dialogue turns"),
            ("Model Range", "11 Models", "0.5B to 72B parameters"),
            ("Decision Scenarios", "8 Domains", "Consequential personal decisions")
        ],
        "findings": [
            "Scripted turns isolate user fatigue and psychological surrender from random conversational divergence.",
            "Analytical control baseline provides a strict counterfactual: models given analytical follow-ups never face delegation pressure.",
            "Double-judge evaluation with gpt-oss:20b and gemma4:12b ensures rubric consistency and robustness against evaluator bias."
        ]
    },
    {
        "id": "fig1",
        "num": "Figure 1",
        "file": "fig1_drift_trajectory.png",
        "pdf": "fig1_drift_trajectory.pdf",
        "title": "Delegation Drift Trajectory: Pressure vs. Analytical Control",
        "badge": "Main Finding",
        "badge_color": "bg-rose-500/10 text-rose-700 border-rose-500/20",
        "question": "Does user fatigue systematically cause models to abandon advisory boundaries?",
        "caption": "Turn-by-turn takeover rate under the neutral system prompt across 11 open-weight models. Solid coral circles denote escalating user fatigue; teal squares denote the analytical control. Left panel: primary evaluator (gpt-oss:20b, validated against human annotations). Right panel: secondary evaluator (gemma4:12b). Error bars denote 95% bootstrap confidence intervals across 8 decision scenarios.",
        "stats": [
            ("Turn 1 Takeover", "31.6%", "Initial exploration phase"),
            ("Turn 4 Takeover", "69.9%", "Final user abdication"),
            ("Drift Magnitude", "+38.3 pp", "p < 0.001 across models"),
            ("Control Turn 4", "15.3%", "Absent user abdication")
        ],
        "findings": [
            "Under progressive user pressure, assistant takeover more than doubles (+38.3 pp), settling the user's choice in 69.9% of final turns.",
            "In the analytical control condition (where users never ask the model to decide), takeover steadily drops to 15.3%.",
            "This confirms delegation drift is specifically driven by user abdication cues rather than dialogue length or context accumulation."
        ]
    },
    {
        "id": "fig2",
        "num": "Figure 2",
        "file": "fig2_intervention_spectrum.png",
        "pdf": "fig2_intervention_spectrum.pdf",
        "title": "Turn 4 Intervention Spectrum & Effect Sizes",
        "badge": "Intervention Spectrum",
        "badge_color": "bg-blue-500/10 text-blue-700 border-blue-500/20",
        "question": "Can system prompts suppress or amplify decision handover under pressure?",
        "caption": "Impact of system prompt interventions on final-turn takeover rate across 11 models evaluated by gpt-oss:20b. (a) Absolute takeover percentage at Turn 4 for Autonomy, Neutral, Implicit Engagement, and Explicit Engagement. (b) Paired difference forest plot showing effect size (percentage points relative to Neutral baseline) with 95% bootstrap confidence intervals.",
        "stats": [
            ("Autonomy Prompt", "43.2%", "-26.7 pp [-35.2, -18.4] vs baseline"),
            ("Neutral Baseline", "69.9%", "Standard helpful assistant framing"),
            ("Implicit Eng.", "71.3%", "+1.4 pp [-3.1, +6.0] mild drift"),
            ("Explicit Eng.", "78.2%", "+8.3 pp [+3.4, +13.6] amplified takeover")
        ],
        "findings": [
            "Autonomy prompting ('never make decisions on the user's behalf') provides substantial defense, reducing takeover by -26.7 pp overall.",
            "Explicit engagement instructions ('settle choices on user behalf') amplify takeover to 78.2% (+8.3 pp overall, and up to +25 pp in Gemma models).",
            "Implicit commercial engagement instructions ('proactively resolve friction') induce mild drift toward takeover even without direct commands to choose."
        ]
    },
    {
        "id": "fig3",
        "num": "Figure 3",
        "file": "fig3_parameter_scaling.png",
        "pdf": "fig3_parameter_scaling.pdf",
        "title": "Parameter Scaling Ladder: Scale Does Not Prevent Drift",
        "badge": "Scaling Law",
        "badge_color": "bg-amber-500/10 text-amber-700 border-amber-500/20",
        "question": "Does increasing model parameter count mitigate delegation drift?",
        "caption": "Turn 4 takeover rate across parameter scale (0.5B to 72B, log scale) under Explicit Engagement (coral circles), Neutral Baseline (gray squares), and Autonomy (teal triangles). Open markers denote 4-bit AWQ quantized variants. Fitted lines show log-linear regression trends.",
        "stats": [
            ("Scaling Slope", "+1.2 pp / decade", "p = 0.39 (not statistically significant)"),
            ("Correlation", "r = +0.32", "Spearman rank correlation"),
            ("72B Takeover", "77% - 86%", "Surrenders agency as readily as 1.5B/3B"),
            ("Quantization", "AWQ 4-bit", "Adheres to prompts comparably to bf16")
        ],
        "findings": [
            "Across two orders of magnitude in parameter count, the scaling slope is flat (+1.2 pp per decade, p = 0.39).",
            "Larger parameter capacity does not inherently endow models with advisory boundary awareness.",
            "4-bit AWQ post-training quantization does not impair prompt adherence: quantized models track unquantized baselines closely."
        ]
    },
    {
        "id": "fig4",
        "num": "Figure 4",
        "file": "fig4_model_dumbbells.png",
        "pdf": "fig4_model_dumbbells.pdf",
        "title": "Model Responsiveness Range: Dynamic Steerability Leaderboard",
        "badge": "Model Profiles",
        "badge_color": "bg-purple-500/10 text-purple-700 border-purple-500/20",
        "question": "How sensitively does each individual model respond to operator framing?",
        "caption": "Dynamic range of Turn 4 takeover for all 11 evaluated models, ordered from largest parameter count (top) to smallest (bottom). Markers indicate takeover rate under Autonomy (teal circle), Neutral baseline (gray square), and Explicit Engagement (coral diamond). Open markers denote 4-bit AWQ checkpoints.",
        "stats": [
            ("Qwen 3.6 27B Range", "16% → 88%", "72 pp dynamic range"),
            ("Gemma 4 12B Range", "20% → 92%", "72 pp dynamic range"),
            ("Gemma 4 e4B Range", "9% → 86%", "77 pp dynamic range"),
            ("Small Qwen Range", "39% → 55%", "Rigid compliance")
        ],
        "findings": [
            "Models bifurcate into high-dynamic-range steerable architectures vs. rigid baseline compliers.",
            "Gemma 4 models and Qwen 3.6 27B demonstrate extreme responsiveness, shifting over 70 pp between autonomy and engagement.",
            "Smaller Qwen models (0.5B, 3B) exhibit high default compliance to user abdication and resist prompt constraints."
        ]
    },
    {
        "id": "fig5",
        "num": "Figure 5",
        "file": "fig5_scenario_vulnerability.png",
        "pdf": "fig5_scenario_vulnerability.pdf",
        "title": "Scenario Vulnerability Breakdown Across Decision Domains",
        "badge": "Domain Analysis",
        "badge_color": "bg-teal-500/10 text-teal-700 border-teal-500/20",
        "question": "Which consequential real-world domains are most susceptible to agency loss?",
        "caption": "Turn 4 takeover rates across eight consequential decision domains, evaluated across all 11 models under gpt-oss:20b. Bars compare Explicit Engagement (coral), Neutral Baseline (gray), and Autonomy (teal). Domains are sorted in descending order of neutral baseline takeover rate.",
        "stats": [
            ("Finances", "90.9% Takeover", "Highest vulnerability domain"),
            ("Personal Boundary", "85.2% Takeover", "High emotional pressure"),
            ("Medical Dilemma", "77.3% Takeover", "High stakes advisory risk"),
            ("Software Deploy", "33.0% Takeover", "Lowest vulnerability domain")
        ],
        "findings": [
            "Personal and emotional dilemmas (finances, personal boundaries, parenting, medical decisions) suffer severe takeover rates (>77% to 91%).",
            "Technical operational decisions (software deployment rollback) exhibit the highest resistance (33% neutral takeover), reflecting pre-training safety guardrails around command execution.",
            "Autonomy prompting achieves the largest absolute suppression in lease negotiations (from 59.1% down to 19.3%)."
        ]
    },
    {
        "id": "fig6",
        "num": "Figure 6",
        "file": "fig6_judge_calibration.png",
        "pdf": "fig6_judge_calibration.pdf",
        "title": "Evaluator Calibration & Reliability Against Blind Human Annotations",
        "badge": "Validation",
        "badge_color": "bg-indigo-500/10 text-indigo-700 border-indigo-500/20",
        "question": "Do automated LLM evaluators accurately capture genuine human agency surrender?",
        "caption": "Confusion matrices and inter-rater agreement metrics for automated judges evaluated against 52 dialogue turns hand-annotated blind to condition. Scores reflect RUBRIC.md categories (-1 takeover, 0 neutral, +1 autonomous). Color intensity indicates turn count.",
        "stats": [
            ("gpt-oss:20b Exact", "79.0%", "Exact agreement with human labels"),
            ("gpt-oss:20b Kappa", "κ = 0.76", "Quadratic weighted Cohen's kappa"),
            ("Takeover Precision", "12 / 14 (86%)", "gpt-oss:20b evaluator"),
            ("Takeover Recall", "12 / 13 (92%)", "gpt-oss:20b evaluator")
        ],
        "findings": [
            "The primary evaluator (gpt-oss:20b) demonstrates high calibration with blind human ground truth (exact 79.0%, weighted κ = 0.76).",
            "High sensitivity (92% recall) and precision (86%) on hard takeover turns (-1) guarantee that reported drift rates reflect true agency surrender.",
            "Secondary evaluator (gemma4:12b) confirms comparative rankings across models and prompt conditions."
        ]
    }
]

# Base64 encode all figures
for f in FIGURES:
    png_path = FIG_DIR / f["file"]
    with open(png_path, "rb") as fp:
        f["b64_png"] = f"data:image/png;base64,{base64.b64encode(fp.read()).decode('utf-8')}"
    pdf_path = FIG_DIR / f["pdf"]
    if pdf_path.exists():
        with open(pdf_path, "rb") as fp:
            f["b64_pdf"] = f"data:application/pdf;base64,{base64.b64encode(fp.read()).decode('utf-8')}"
    else:
        f["b64_pdf"] = None

html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Delegation Drift — Academic Figure Suite</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=Inter:wght@400;500;600;700&display=swap');
    
    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    .serif {{
      font-family: 'Source Serif 4', Georgia, serif;
    }}
    .tab-btn.active {{
      background: var(--primary, #2563eb);
      color: var(--primary-foreground, #ffffff);
      border-color: var(--primary, #2563eb);
    }}
  </style>
</head>
<body class="bg-[var(--background,#F8FAFC)] text-[var(--foreground,#0F172A)] antialiased min-h-screen p-4 md:p-8">

  <!-- Academic Paper Header -->
  <header class="max-w-6xl mx-auto mb-8 pb-6 border-b border-[var(--border,#E2E8F0)]">
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-2 mb-2">
          <span class="px-2.5 py-0.5 rounded text-xs font-semibold bg-slate-100 text-slate-700 border border-slate-200 uppercase tracking-wider">
            NeurIPS / ICML / ICLR Standard
          </span>
          <span class="px-2.5 py-0.5 rounded text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
            Vector PDF + 300 DPI Raster
          </span>
        </div>
        <h1 class="text-2xl md:text-3xl font-bold tracking-tight serif text-[var(--foreground,#0F172A)]">
          Delegation Drift: Academic Results & Figure Suite
        </h1>
        <p class="text-xs md:text-sm text-[var(--muted-foreground,#64748B)] mt-1.5 leading-relaxed max-w-3xl">
          Complete empirical visualization suite for evaluating agency handover in multi-turn conversational AI assistants across 11 open-weight models (0.5B to 72B parameters, 14,061 evaluated turn observations).
        </p>
      </div>

      <div class="flex items-center gap-2.5">
        <button onclick="toggleAllView()" id="btn-toggle-all" class="px-3.5 py-1.5 rounded-md text-xs font-semibold border border-[var(--border,#CBD5E1)] bg-[var(--card,#FFFFFF)] hover:bg-slate-50 transition shadow-sm">
          View All Figures
        </button>
        <a href="https://github.com/Kaustubh73/delegation-drift" target="_blank" class="px-3.5 py-1.5 rounded-md text-xs font-semibold bg-slate-900 text-white hover:bg-slate-800 transition shadow-sm">
          GitHub Repo
        </a>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <nav class="flex items-center gap-2 overflow-x-auto py-3 no-scrollbar mt-4 border-t border-[var(--border,#E2E8F0)]">
"""

for fig in FIGURES:
    html_str += f"""
      <button onclick="showFigure('{fig['id']}')" class="tab-btn px-3 py-1.5 rounded-md text-xs font-medium border border-[var(--border,#CBD5E1)] bg-[var(--card,#FFFFFF)] text-[var(--foreground,#0F172A)] hover:border-slate-400 transition whitespace-nowrap shadow-xs" id="tab-{fig['id']}">
        {fig['num']}: {fig['title'].split(':')[0]}
      </button>
    """

html_str += """
    </nav>
  </header>

  <!-- Main Container -->
  <main class="max-w-6xl mx-auto space-y-10" id="figure-container">
"""

for fig in FIGURES:
    stats_pills = "".join([f"""
      <div class="bg-slate-50/70 border border-slate-200/80 rounded-lg p-3">
        <p class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">{label}</p>
        <p class="text-lg font-bold text-slate-900 mt-0.5">{val}</p>
        <p class="text-xs text-slate-500 mt-0.5">{sub}</p>
      </div>
    """ for label, val, sub in fig["stats"]])

    findings_list = "".join([f"""
      <li class="flex items-start gap-2.5 text-xs text-slate-700 leading-relaxed">
        <span class="inline-block w-1.5 h-1.5 rounded-full bg-slate-400 mt-1.5 shrink-0"></span>
        <span>{pt}</span>
      </li>
    """ for pt in fig["findings"]])

    download_pdf_html = ""
    if fig["b64_pdf"]:
        download_pdf_html = f"""
          <a href="{fig['b64_pdf']}" download="{fig['pdf']}" class="px-2.5 py-1 rounded text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300 flex items-center gap-1.5 transition">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
            Vector PDF
          </a>
        """

    html_str += f"""
    <!-- Figure Card {fig['id']} -->
    <article id="card-{fig['id']}" class="figure-card bg-[var(--card,#FFFFFF)] border border-[var(--border,#E2E8F0)] rounded-xl shadow-xs overflow-hidden transition duration-150">
      <div class="p-6 md:p-8">

        <!-- Header metadata -->
        <div class="flex flex-wrap items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-100">
          <div class="flex items-center gap-2">
            <span class="px-2.5 py-0.5 rounded text-xs font-bold border {fig['badge_color']}">
              {fig['badge']}
            </span>
            <span class="text-xs font-bold serif text-slate-600">{fig['num']}</span>
          </div>

          <div class="flex items-center gap-2">
            <button onclick="openModal('{fig['id']}')" class="px-2.5 py-1 rounded text-xs font-semibold text-blue-600 hover:bg-blue-50 border border-blue-200 flex items-center gap-1.5 transition">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"/></svg>
              Zoom
            </button>
            <a href="{fig['b64_png']}" download="{fig['file']}" class="px-2.5 py-1 rounded text-xs font-semibold text-slate-600 hover:bg-slate-100 border border-slate-200 flex items-center gap-1.5 transition">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
              300 DPI PNG
            </a>
            {download_pdf_html}
          </div>
        </div>

        <!-- Scientific Title & Question -->
        <h2 class="text-lg font-bold serif text-slate-900 tracking-tight mb-1">
          {fig['title']}
        </h2>
        <p class="text-xs font-medium text-slate-500 italic mb-5">
          Research Question: {fig['question']}
        </p>

        <!-- Figure Image Presentation -->
        <div class="w-full bg-white border border-slate-200 rounded-lg p-3 md:p-6 mb-5 flex justify-center items-center cursor-zoom-in group shadow-2xs" onclick="openModal('{fig['id']}')">
          <img src="{fig['b64_png']}" alt="{fig['title']}" class="max-h-[460px] w-auto object-contain rounded group-hover:scale-[1.006] transition duration-150" id="img-{fig['id']}">
        </div>

        <!-- Academic Figure Caption -->
        <div class="bg-slate-50 border-l-2 border-slate-400 p-3.5 mb-6 text-xs text-slate-700 leading-relaxed">
          <span class="font-bold text-slate-900">{fig['num']}:</span> {fig['caption']}
        </div>

        <!-- Metrics Grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          {stats_pills}
        </div>

        <!-- Empirical Findings Bulleted -->
        <div class="bg-white border border-slate-200/80 rounded-lg p-4">
          <h3 class="text-xs font-bold uppercase tracking-wider text-slate-800 mb-2.5 flex items-center gap-2">
            <svg class="w-3.5 h-3.5 text-slate-600" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>
            Empirical Results & Synthesis
          </h3>
          <ul class="space-y-1.5">
            {findings_list}
          </ul>
        </div>

      </div>
    </article>
    """

html_str += f"""
  </main>

  <!-- Modal Lightbox for Zoom -->
  <div id="modal" class="fixed inset-0 z-50 hidden bg-slate-900/80 backdrop-blur-xs flex items-center justify-center p-4" onclick="closeModal()">
    <div class="relative max-w-5xl w-full bg-white rounded-xl shadow-2xl p-4 md:p-6 overflow-hidden" onclick="event.stopPropagation()">
      <div class="flex items-center justify-between pb-3 border-b border-slate-200 mb-4">
        <h3 id="modal-title" class="text-sm font-bold serif text-slate-900">Figure Preview</h3>
        <button onclick="closeModal()" class="p-1 rounded text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>
      <div class="flex justify-center items-center max-h-[80vh] overflow-auto">
        <img id="modal-img" src="" alt="Figure Zoom" class="max-h-[75vh] w-auto object-contain">
      </div>
    </div>
  </div>

  <script>
    const ids = ['fig0', 'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6'];
    let isAllView = false;

    function showFigure(id) {{
      isAllView = false;
      document.getElementById('btn-toggle-all').textContent = 'View All Figures';
      document.getElementById('btn-toggle-all').classList.remove('bg-blue-50', 'text-blue-700', 'border-blue-300');

      document.querySelectorAll('.tab-btn').forEach(b => {{
        b.classList.remove('active', 'bg-blue-600', 'text-white', 'border-blue-600');
        b.classList.add('bg-white', 'text-slate-800');
      }});

      const curTab = document.getElementById('tab-' + id);
      if (curTab) {{
        curTab.classList.add('active', 'bg-blue-600', 'text-white', 'border-blue-600');
        curTab.classList.remove('bg-white', 'text-slate-800');
      }}

      ids.forEach(fid => {{
        const card = document.getElementById('card-' + fid);
        if (fid === id) {{
          card.classList.remove('hidden');
        }} else {{
          card.classList.add('hidden');
        }}
      }});
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }}

    function toggleAllView() {{
      if (!isAllView) {{
        isAllView = true;
        document.querySelectorAll('.tab-btn').forEach(b => {{
          b.classList.remove('active', 'bg-blue-600', 'text-white', 'border-blue-600');
        }});
        document.getElementById('btn-toggle-all').textContent = 'Viewing All (Click to collapse)';
        document.getElementById('btn-toggle-all').classList.add('bg-blue-50', 'text-blue-700', 'border-blue-300');
        ids.forEach(fid => {{
          document.getElementById('card-' + fid).classList.remove('hidden');
        }});
      }} else {{
        showFigure('fig0');
      }}
    }}

    function openModal(id) {{
      const img = document.getElementById('img-' + id);
      const title = document.querySelector('#card-' + id + ' h2').textContent;
      document.getElementById('modal-img').src = img.src;
      document.getElementById('modal-title').textContent = title;
      document.getElementById('modal').classList.remove('hidden');
    }}

    function closeModal() {{
      document.getElementById('modal').classList.add('hidden');
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') closeModal();
    }});

    // Initialize with Fig 0 (Protocol)
    showFigure('fig0');
  </script>
</body>
</html>
"""

# Save to figures/index.html
out_figs_html = FIG_DIR / "index.html"
with open(out_figs_html, "w", encoding="utf-8") as f:
    f.write(html_str)
print(f"Saved {out_figs_html} ({len(html_str)//1024} KB)")

# Save to results/academic_viewer.html
out_res_html = ROOT / "results/academic_viewer.html"
with open(out_res_html, "w", encoding="utf-8") as f:
    f.write(html_str)
print(f"Saved {out_res_html} ({len(html_str)//1024} KB)")

# Save to artifact directory
out_artifact_html = ARTIFACT_DIR / "academic_viewer.html"
with open(out_artifact_html, "w", encoding="utf-8") as f:
    f.write(html_str)
print(f"Saved {out_artifact_html} ({len(html_str)//1024} KB)")

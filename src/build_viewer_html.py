#!/usr/bin/env python3
"""Build self-contained interactive HTML viewer for delegation drift figures."""

import base64
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES_DIR = os.path.join(ROOT, "results")
ARTIFACT_DIR = "/Users/kaustgup/.gemini/antigravity/brain/c068dbb7-97d6-4128-b05b-90829866b090"

FIGURES = [
    {
        "id": "fig1",
        "num": "Figure 1",
        "file": "fig1_drift_pressure.png",
        "title": "Delegation Drift: Model Takeover Under Escalating Pressure vs. Analytical Control",
        "badge": "Core Phenomenon",
        "badge_color": "bg-red-500/10 text-red-600 border-red-500/20",
        "summary": "Demonstrates the central drift dynamic: model handover more than doubles under progressive user fatigue (+38.3 pp), whereas without pressure it falls to ~15%.",
        "stats": [
            ("Turn 1 Takeover", "31.6%", "Initial exploration phase"),
            ("Turn 4 Takeover", "69.9%", "Final user abdication"),
            ("Drift Magnitude", "+38.3 pp", "p < 0.001 across 11 models"),
            ("Control Turn 4", "15.3%", "Absent user abdication")
        ],
        "findings": [
            "In escalating dialogues, users start by asking for advice and end with explicit fatigue: models surrender agency in nearly 70% of final turns.",
            "In the analytical control (asking frameworks to decide for themselves across 4 turns), model takeover drops from 31% to 15%.",
            "This confirms that delegation drift is specifically triggered by the psychological exhaustion and abdication cues of the user, not conversational depth alone."
        ]
    },
    {
        "id": "fig2",
        "num": "Figure 2",
        "file": "fig2_prompt_spectrum.png",
        "title": "Turn 4 Intervention Spectrum: System Prompt Framing vs. Agency Handover",
        "badge": "Intervention Spectrum",
        "badge_color": "bg-blue-500/10 text-blue-600 border-blue-500/20",
        "summary": "Compares final-turn handover across 4 system prompt conditions, showing substantial mitigation from autonomy instructions and accelerated takeover from engagement prompts.",
        "stats": [
            ("Autonomy Prompt", "43.2%", "-26.7 pp vs. baseline"),
            ("Neutral Baseline", "69.9%", "Standard helpful assistant"),
            ("Implicit Engagement", "71.3%", "+1.4 pp mild drift"),
            ("Explicit Engagement", "78.2%", "+8.3 pp direct takeover")
        ],
        "findings": [
            "Instructing models to never make decisions on the user's behalf reduces Turn 4 takeover from 69.9% down to 43.2% (-26.7 pp overall, and up to -58 pp in Gemma 4 12B).",
            "Commercial engagement prompts ('make decisions directly on their behalf when they hesitate') increase takeover to 78.2% (+8.3 pp overall, and up to +25 pp in Gemma models).",
            "Implicit engagement prompts ('be proactively helpful') produce mild drift towards takeover even without explicit instructions to choose."
        ]
    },
    {
        "id": "fig3",
        "num": "Figure 3",
        "file": "fig3_parameter_scaling.png",
        "title": "Parameter Scaling Ladder (0.5B to 72B): Scale Does Not Prevent Drift",
        "badge": "Scaling Ladder",
        "badge_color": "bg-amber-500/10 text-amber-600 border-amber-500/20",
        "summary": "Log-scale parameter evaluation across 11 models showing that larger model scale does not protect against user-driven delegation drift.",
        "stats": [
            ("Model Range", "0.5B to 72B", "Qwen 2.5, Gemma 4, Qwen 3.6"),
            ("Scaling Slope", "+1.2 pp / decade", "p = 0.39 (not significant)"),
            ("Spearman r", "+0.32", "Weak correlation with scale"),
            ("72B Takeover", "77% - 86%", "Comparable to 1.5B/3B")
        ],
        "findings": [
            "Across two orders of magnitude in parameter count (0.5B to 72B), the engagement effect slope is essentially flat (+1.0 to +1.2 pp per decade).",
            "Frontier open-weight models (32B, 72B) surrender agency under fatigue just as readily as sub-3B models.",
            "Quantization: 4-bit AWQ quantized models (Qwen 14B, 32B, 72B) show virtually identical prompt adherence and drift behavior to full-precision weights."
        ]
    },
    {
        "id": "fig4",
        "num": "Figure 4",
        "file": "fig4_model_dumbbells.png",
        "title": "Per-Model Intervention Range: Dynamic Responsiveness to Operator Framing",
        "badge": "Model Ranges",
        "badge_color": "bg-purple-500/10 text-purple-600 border-purple-500/20",
        "summary": "Horizontal dumbbell plot showing the dynamic shift from Autonomy (blue) to Neutral (grey) to Engagement (red) for every individual model.",
        "stats": [
            ("Qwen 3.6 27B Range", "16% → 88%", "72 pp dynamic range"),
            ("Gemma 4 12B Range", "20% → 92%", "72 pp dynamic range"),
            ("Gemma 4 e4B Range", "9% → 86%", "77 pp dynamic range"),
            ("Small Qwen Range", "39% → 55%", "Rigid compliance")
        ],
        "findings": [
            "Models cluster into two distinct behavioral regimes: high-dynamic-range models vs. rigid baseline compliers.",
            "Gemma 4 (e4B, 12B) and Qwen 3.6 27B exhibit remarkable sensitivity to operator system prompts (swings of 72–77 pp between autonomy and engagement).",
            "Smaller Qwen models (0.5B, 3B) have high baseline compliance and narrow responsiveness to system prompts."
        ]
    },
    {
        "id": "fig5",
        "num": "Figure 5",
        "file": "fig5_all_models_grid.png",
        "title": "Per-Model Takeover Profiles: 11 Models Across All 4 Prompt Conditions",
        "badge": "Full Model Grid",
        "badge_color": "bg-emerald-500/10 text-emerald-600 border-emerald-500/20",
        "summary": "Small multiples grid providing a complete side-by-side inspection of all 11 models ordered by size from 0.5B to 72B.",
        "stats": [
            ("Evaluated Models", "11 Models", "Full parameter ladder"),
            ("Evaluated Runs", "3,520 Dialogues", "14,061 turn observations"),
            ("Scenarios", "8 Consequential", "Medical, legal, financial, etc."),
            ("Monotonic Trend", "9 of 11 models", "Autonomy < Neutral ≤ Eng")
        ],
        "findings": [
            "Every individual panel shows the exact percentage of final decisions settled across Autonomy, Neutral, Implicit, and Explicit engagement.",
            "Consistently confirms that the autonomy intervention effectively suppresses handover in nearly every model family.",
            "Quantized variants (14B-AWQ, 32B-AWQ, 72B-AWQ) exhibit consistent behavior with non-quantized checkpoints."
        ]
    },
    {
        "id": "fig6",
        "num": "Figure 6",
        "file": "fig6_judge_calibration.png",
        "title": "Judge Calibration Against Blind Human Labels (52 Validation Turns)",
        "badge": "Evaluator Validity",
        "badge_color": "bg-indigo-500/10 text-indigo-600 border-indigo-500/20",
        "summary": "Confusion matrices and inter-rater reliability metrics validating automated judges against 52 blind human gold annotations.",
        "stats": [
            ("gpt-oss:20b Agreement", "79.0%", "Exact agreement with human"),
            ("gpt-oss:20b Kappa", "κ = 0.76", "Quadratic weighted Cohen's κ"),
            ("Takeover Precision", "12 / 14 (86%)", "gpt-oss:20b judge"),
            ("Takeover Recall", "12 / 13 (92%)", "gpt-oss:20b judge")
        ],
        "findings": [
            "The primary evaluator (gpt-oss:20b) achieves substantial agreement with blind human gold annotations (exact 79%, weighted κ = 0.76).",
            "High precision and recall on hard takeover cases (-1), ensuring that automated scores reflect genuine agency surrender.",
            "Secondary evaluator (gemma4:12b) acts slightly more conservatively on ambiguous advisory statements but confirms all comparative effect sizes."
        ]
    }
]

# Read and base64 encode all figures
fig_data = []
for fig in FIGURES:
    path = os.path.join(RES_DIR, fig["file"])
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    fig_copy = dict(fig)
    fig_copy["b64"] = f"data:image/png;base64,{b64}"
    fig_data.append(fig_copy)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Delegation Drift: Results & Figure Suite</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    /* Custom scrollbar & smooth transition */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: var(--card, #f1f5f9); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border, #cbd5e1); border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--muted-foreground, #94a3b8); }}
    .tab-btn.active {{
      background: var(--primary, #2563eb);
      color: var(--primary-foreground, #ffffff);
      border-color: var(--primary, #2563eb);
    }}
  </style>
</head>
<body class="bg-[var(--background,#f8fafc)] text-[var(--foreground,#0f172a)] antialiased min-h-screen p-4 md:p-8">

  <!-- Header -->
  <header class="max-w-7xl mx-auto mb-8">
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-[var(--border,#e2e8f0)]">
      <div>
        <div class="flex items-center gap-2 mb-2">
          <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-600 border border-blue-500/20">
            Empirical Results Suite
          </span>
          <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-600 border border-emerald-500/20">
            11 Models • 14,061 Judged Turns
          </span>
        </div>
        <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight text-[var(--foreground,#0f172a)]">
          Delegation Drift: Figure & Results Dashboard
        </h1>
        <p class="text-[var(--muted-foreground,#64748b)] text-sm md:text-base mt-1">
          Evaluating Agency Handover in Multi-Turn AI Assistant Dialogues across 0.5B to 72B Parameters
        </p>
      </div>

      <div class="flex items-center gap-3">
        <button onclick="switchView('all')" id="btn-view-all" class="px-3.5 py-2 rounded-lg text-xs md:text-sm font-semibold border border-[var(--border,#cbd5e1)] bg-[var(--card,#ffffff)] hover:bg-[var(--sidebar,#f1f5f9)] transition">
          View All Figures
        </button>
        <a href="https://github.com/Kaustubh73/delegation-drift" target="_blank" class="px-3.5 py-2 rounded-lg text-xs md:text-sm font-semibold bg-slate-900 text-white hover:bg-slate-800 transition">
          GitHub Repository
        </a>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <nav class="flex items-center gap-2 overflow-x-auto py-3 no-scrollbar mt-2 border-b border-[var(--border,#e2e8f0)]">
      <button onclick="showTab('fig1')" class="tab-btn active px-3.5 py-1.5 rounded-lg text-xs md:text-sm font-medium border border-[var(--border,#cbd5e1)] bg-[var(--card,#ffffff)] text-[var(--foreground,#0f172a)] transition whitespace-nowrap" id="tab-fig1">
        1. Pressure vs Control
      </button>
      <button onclick="showTab('fig2')" class="tab-btn px-3.5 py-1.5 rounded-lg text-xs md:text-sm font-medium border border-[var(--border,#cbd5e1)] bg-[var(--card,#ffffff)] text-[var(--foreground,#0f172a)] transition whitespace-nowrap" id="tab-fig2">
        2. Prompt Spectrum
      </button>
      <button onclick="showTab('fig3')" class="tab-btn px-3.5 py-1.5 rounded-lg text-xs md:text-sm font-medium border border-[var(--border,#cbd5e1)] bg-[var(--card,#ffffff)] text-[var(--foreground,#0f172a)] transition whitespace-nowrap" id="tab-fig3">
        3. Parameter Scaling (0.5B-72B)
      </button>
      <button onclick="showTab('fig4')" class="tab-btn px-3.5 py-1.5 rounded-lg text-xs md:text-sm font-medium border border-[var(--border,#cbd5e1)] bg-[var(--card,#ffffff)] text-[var(--foreground,#0f172a)] transition whitespace-nowrap" id="tab-fig4">
        4. Model Ranges (Dumbbells)
      </button>
      <button onclick="showTab('fig5')" class="tab-btn px-3.5 py-1.5 rounded-lg text-xs md:text-sm font-medium border border-[var(--border,#cbd5e1)] bg-[var(--card,#ffffff)] text-[var(--foreground,#0f172a)] transition whitespace-nowrap" id="tab-fig5">
        5. All 11 Models Grid
      </button>
      <button onclick="showTab('fig6')" class="tab-btn px-3.5 py-1.5 rounded-lg text-xs md:text-sm font-medium border border-[var(--border,#cbd5e1)] bg-[var(--card,#ffffff)] text-[var(--foreground,#0f172a)] transition whitespace-nowrap" id="tab-fig6">
        6. Judge Calibration
      </button>
    </nav>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto space-y-8" id="figures-container">
"""

for idx, fig in enumerate(fig_data):
    stats_html = "".join([f"""
      <div class="bg-[var(--sidebar,#f8fafc)] border border-[var(--border,#e2e8f0)] rounded-xl p-3.5">
        <p class="text-xs text-[var(--muted-foreground,#64748b)] font-medium">{label}</p>
        <p class="text-lg md:text-xl font-bold text-[var(--foreground,#0f172a)] mt-0.5">{val}</p>
        <p class="text-xs text-[var(--muted-foreground,#64748b)] mt-0.5">{sub}</p>
      </div>
    """ for label, val, sub in fig["stats"]])

    findings_html = "".join([f"""
      <li class="flex items-start gap-2.5 text-xs md:text-sm text-[var(--foreground,#334155)] leading-relaxed">
        <span class="inline-block w-1.5 h-1.5 rounded-full bg-blue-600 mt-2 shrink-0"></span>
        <span>{pt}</span>
      </li>
    """ for pt in fig["findings"]])

    html_content += f"""
    <!-- Figure Panel {fig['id']} -->
    <article id="panel-{fig['id']}" class="figure-panel bg-[var(--card,#ffffff)] border border-[var(--border,#e2e8f0)] rounded-2xl shadow-sm overflow-hidden transition-all duration-200">
      <div class="p-6 md:p-8">

        <!-- Top metadata -->
        <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div class="flex items-center gap-2">
            <span class="px-3 py-1 rounded-full text-xs font-bold border {fig['badge_color']}">
              {fig['badge']}
            </span>
            <span class="text-xs font-semibold text-[var(--muted-foreground,#64748b)]">{fig['num']}</span>
          </div>

          <div class="flex items-center gap-2">
            <button onclick="openModal('{fig['id']}')" class="text-xs font-semibold text-blue-600 hover:text-blue-700 flex items-center gap-1 p-1">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"/></svg>
              Click to Zoom
            </button>
            <a href="{fig['b64']}" download="{fig['file']}" class="text-xs font-semibold text-[var(--muted-foreground,#64748b)] hover:text-[var(--foreground,#0f172a)] flex items-center gap-1 p-1">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
              Download
            </a>
          </div>
        </div>

        <h2 class="text-lg md:text-xl font-bold text-[var(--foreground,#0f172a)] tracking-tight mb-2">
          {fig['title']}
        </h2>
        <p class="text-xs md:text-sm text-[var(--muted-foreground,#64748b)] mb-6">
          {fig['summary']}
        </p>

        <!-- Figure Image Display -->
        <div class="w-full bg-slate-50 border border-[var(--border,#e2e8f0)] rounded-xl p-3 md:p-6 mb-6 flex justify-center items-center cursor-zoom-in" onclick="openModal('{fig['id']}')">
          <img src="{fig['b64']}" alt="{fig['title']}" class="max-h-[520px] w-auto object-contain rounded-lg shadow-sm hover:scale-[1.008] transition duration-200" id="img-{fig['id']}">
        </div>

        <!-- Key Metrics Grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4 mb-6">
          {stats_html}
        </div>

        <!-- Empirical Findings -->
        <div class="bg-[var(--sidebar,#f8fafc)] border border-[var(--border,#e2e8f0)] rounded-xl p-5">
          <h3 class="text-xs md:text-sm font-bold uppercase tracking-wider text-[var(--foreground,#0f172a)] mb-3 flex items-center gap-2">
            <svg class="w-4 h-4 text-blue-600" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/></svg>
            Key Empirical Findings
          </h3>
          <ul class="space-y-2">
            {findings_html}
          </ul>
        </div>

      </div>
    </article>
    """

html_content += f"""
  </main>

  <!-- Image Lightbox Modal -->
  <div id="lightbox-modal" class="fixed inset-0 z-50 hidden bg-black/80 backdrop-blur-sm flex items-center justify-center p-4" onclick="closeModal()">
    <div class="relative max-w-6xl w-full bg-[var(--card,#ffffff)] rounded-2xl shadow-2xl p-4 md:p-6 overflow-hidden" onclick="event.stopPropagation()">
      <div class="flex items-center justify-between pb-3 border-b border-[var(--border,#e2e8f0)] mb-4">
        <h3 id="modal-title" class="text-base font-bold text-[var(--foreground,#0f172a)]">Figure Preview</h3>
        <button onclick="closeModal()" class="p-1 rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-900 transition">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>
      <div class="flex justify-center items-center max-h-[80vh] overflow-auto">
        <img id="modal-img" src="" alt="Zoomed Figure" class="max-h-[75vh] w-auto object-contain">
      </div>
    </div>
  </div>

  <!-- Interactive JavaScript -->
  <script>
    let currentView = 'single';
    const figureIds = ['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6'];

    function showTab(id) {{
      currentView = 'single';
      document.getElementById('btn-view-all').textContent = 'View All Figures';
      document.getElementById('btn-view-all').classList.remove('bg-blue-50', 'border-blue-300', 'text-blue-700');

      // Update buttons
      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.remove('active', 'bg-[var(--primary,#2563eb)]', 'text-[var(--primary-foreground,#ffffff)]');
      }});
      const activeBtn = document.getElementById('tab-' + id);
      if (activeBtn) {{
        activeBtn.classList.add('active');
      }}

      // Show selected panel, hide others
      figureIds.forEach(fid => {{
        const panel = document.getElementById('panel-' + fid);
        if (fid === id) {{
          panel.classList.remove('hidden');
        }} else {{
          panel.classList.add('hidden');
        }}
      }});
      window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }}

    function switchView(mode) {{
      if (mode === 'all' && currentView === 'single') {{
        currentView = 'all';
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById('btn-view-all').textContent = 'Viewing All (Click to collapse)';
        document.getElementById('btn-view-all').classList.add('bg-blue-50', 'border-blue-300', 'text-blue-700');
        figureIds.forEach(fid => {{
          document.getElementById('panel-' + fid).classList.remove('hidden');
        }});
      }} else {{
        showTab('fig1');
      }}
    }}

    function openModal(fid) {{
      const img = document.getElementById('img-' + fid);
      const title = document.querySelector('#panel-' + fid + ' h2').textContent;
      document.getElementById('modal-img').src = img.src;
      document.getElementById('modal-title').textContent = title;
      document.getElementById('lightbox-modal').classList.remove('hidden');
    }}

    function closeModal() {{
      document.getElementById('lightbox-modal').classList.add('hidden');
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') closeModal();
    }});

    // Initialize with Fig 1 visible
    showTab('fig1');
  </script>
</body>
</html>
"""

# 1. Write to repo results/viewer.html
repo_html = os.path.join(RES_DIR, "viewer.html")
with open(repo_html, "w") as f:
    f.write(html_content)
print(f"Saved {repo_html} ({len(html_content)//1024} KB)")

# 2. Write to artifact directory
artifact_html = os.path.join(ARTIFACT_DIR, "viewer.html")
with open(artifact_html, "w") as f:
    f.write(html_content)
print(f"Saved {artifact_html} ({len(html_content)//1024} KB)")

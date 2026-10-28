"""gpt-oss:20b as a THIRD judge on the main scripted runs (data/runs/*), same frozen v2
prompt. Two serving quirks handled here:
  * num_predict raised to 3000 (reasoning tokens otherwise eat the 200-token budget);
  * ollama's format="json" makes gpt-oss:20b emit a truncated reasoning trace instead of JSON,
    so the format flag is stripped and the prompt's own "Return JSON" instruction is relied on.

Writes judge_gpt-oss_20b.jsonl into each run dir in the same shape as the two main judges, so
`python3 drift_eval.py analyze --out data/runs/*` picks it up unchanged.

Turn 4 only: that is what Table 1 and every headline contrast in the report use. Resumable.

Usage: python3 -I src/judges/gptoss_judge_runs.py [runs_dir ...]
"""
import json, os, sys, time
H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(H, "..")
sys.path.insert(0, ROOT)
import drift_eval as d

M = "gpt-oss:20b"
TURNS = [4]

orig = d.chat
def chat_patched(m, msgs, seed, temperature=0.7, fmt=None, num_predict=None, **k):
    return orig(m, msgs, seed, temperature=temperature, fmt=None, num_predict=3000)
d.chat = chat_patched

def parse_lenient(raw):
    """judge_one only accepts strict JSON. gpt-oss sometimes wraps it in prose; salvage the object."""
    if not raw:
        return None, raw
    i, j = raw.find("{"), raw.rfind("}")
    if i == -1 or j <= i:
        return None, raw
    try:
        o = json.loads(raw[i:j + 1])
        s = int(o.get("score"))
        return (s, f"q1={o.get('q1')} q2={o.get('q2')}") if s in (-1, 0, 1) else (None, raw)
    except Exception:
        return None, raw

dirs = sys.argv[1:] or [os.path.join(ROOT, "..", "data", "runs", x) for x in sorted(os.listdir(os.path.join(ROOT, "..", "data", "runs")))
                        if os.path.isdir(os.path.join(ROOT, "..", "data", "runs", x))]

t0 = time.time(); n = 0; fails = 0
for run in dirs:
    src = d.load_jsonl(os.path.join(run, "transcripts.jsonl"))
    path = os.path.join(run, f"judge_{M.replace(':', '_')}.jsonl")
    done = {(r["key"], r["turn"]) for r in d.load_jsonl(path) if r.get("score") is not None}
    with open(path, "a") as out:
        for t in TURNS:
            todo = [r for r in src if t <= len(r["trajectory"]) and (r["key"], t) not in done]
            for r in todo:
                s, why = d.judge_one(M, r["trajectory"], t)
                if s is None:
                    s, why = parse_lenient(why)
                if s is None:
                    fails += 1
                out.write(json.dumps({"key": r["key"], "turn": t, "judge": M, "score": s, "reason": why}) + "\n")
                out.flush(); n += 1
                if n % 20 == 0:
                    el = time.time() - t0
                    eta = (720 - n) * el / n / 60
                    print(f"{n} judged, {el/60:.1f} min elapsed, {el/n:.1f} s/call, eta {eta:.0f} min, fails={fails}", flush=True)
            print(f"{os.path.basename(run)} turn {t}: +{len(todo)}", flush=True)
print(f"done {n} rows in {(time.time()-t0)/60:.1f} min, fails={fails}", flush=True)

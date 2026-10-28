"""Judge the main v2 scripted runs with the Modal-hosted strong judge (Qwen 2.5 32B, vLLM).

Same frozen v2 prompt as the two judges already in the report. Turn 4 only (Table 1 / every
headline contrast). Writes judge_<model>.jsonl into each data/runs/<model>/ dir in the same
shape as the existing judge files, so `python3 drift_eval.py analyze --out data/runs/*` picks it up.

Usage:
    export MODAL_JUDGE_URL=https://<something>.modal.run
    export MODAL_JUDGE_TOKEN=<token from deploy>
    python3 -I src/judges/modal_judge_runs.py [--workers 16]
"""
import argparse, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(H, "..")
sys.path.insert(0, ROOT)
import drift_eval as d

MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
SHORT = "qwen2.5-32b-instruct"
TURN = 4

BASE = os.environ.get("MODAL_JUDGE_URL", "").rstrip("/")
TOKEN = os.environ.get("MODAL_JUDGE_TOKEN", "")
if not BASE or not TOKEN:
    sys.exit("set MODAL_JUDGE_URL and MODAL_JUDGE_TOKEN")

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--runs", nargs="*", help="run dirs (default: all of data/runs/*)")
a = ap.parse_args()


def ask(user, reply, retries=3):
    prompt = d.JUDGE_PROMPT.format(user=user, reply=reply)
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 200, "seed": 0}
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})
    last = ""
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
            i0, j0 = txt.find("{"), txt.rfind("}")
            o = json.loads(txt[i0:j0 + 1]) if i0 >= 0 and j0 > i0 else {}
            s = int(o.get("score"))
            if s in (-1, 0, 1):
                return s, f"q1={o.get('q1')} q2={o.get('q2')}", txt
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 + 3 * i)
    return None, last, ""


runs = a.runs or [os.path.join(ROOT, "..", "data", "runs", x) for x in sorted(os.listdir(os.path.join(ROOT, "..", "data", "runs")))
                  if os.path.isdir(os.path.join(ROOT, "..", "data", "runs", x))]

t0 = time.time()
total, fails = 0, 0
for run in runs:
    src = [r for r in d.load_jsonl(os.path.join(run, "transcripts.jsonl")) if len(r["trajectory"]) >= TURN]
    path = os.path.join(run, f"judge_{SHORT}.jsonl")
    done = {r["key"] for r in d.load_jsonl(path) if r.get("score") is not None}
    todo = [r for r in src if r["key"] not in done]
    if not todo:
        print(f"{os.path.basename(run)}: nothing to do", flush=True)
        continue
    with open(path, "a") as out, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(ask, r["trajectory"][TURN - 1]["user"], r["trajectory"][TURN - 1]["assistant"]): r
                for r in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            s, why, raw = f.result()
            if s is None:
                fails += 1
            out.write(json.dumps({"key": r["key"], "turn": TURN, "judge": SHORT, "score": s, "reason": why}) + "\n")
            out.flush(); total += 1
            if i % 50 == 0:
                el = time.time() - t0
                print(f"{os.path.basename(run)} {i}/{len(todo)} | {total} total | {el/60:.1f} min | fails={fails}", flush=True)
    print(f"{os.path.basename(run)}: +{len(todo)}", flush=True)
print(f"done {total} rows in {(time.time()-t0)/60:.1f} min, fails={fails}", flush=True)

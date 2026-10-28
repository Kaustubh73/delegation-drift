"""Generate scripted drift_eval transcripts on Modal, offline vLLM (no web server, no idle billing).

One GPU container per model: load weights once, advance all trajectories one turn at a time as a
single batch (4 turns => 4 batched calls). Output matches src/drift_eval.py transcripts.jsonl, so
`drift_eval.py judge` (local Ollama judges) works unchanged.

Baselines (neutral/engagement/autonomy) are regenerated here on the same stack, because vLLM bf16
differs from the Ollama q4 runs in data/runs: comparing across stacks would confound the result.

    modal run src/implicit/run_modal.py --model qwen2.5:7b-instruct --limit 4        # smoke test, cents
    modal run src/implicit/run_modal.py --model qwen2.5:7b-instruct                  # full grid, one model
Stop anything left running:  modal app stop m3-implicit-engagement
"""
import json
import os
import sys
from datetime import datetime, timezone

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]

# Ollama name (used in transcript keys, matches v2 runs) -> HF repo. gemma4:e4b omitted: vLLM 0.10.2
# very likely has no Gemma 4 support [unverified]; run it locally on Ollama if wanted.
MODELS = {
    "qwen2.5:7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1:8b-instruct-q4_K_M": "meta-llama/Llama-3.1-8B-Instruct",    # gated: needs HF token
    "mistral:7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",          # gated: needs HF token
    "gemma3:12b": "google/gemma-3-12b-it",                                # gated; ~24 GB bf16
}

app = modal.App("m3-implicit-engagement")
hf_volume = modal.Volume.from_name("pgg-swap-hf", create_if_missing=True)  # shared HF cache
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.2", "transformers==4.56.2", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 600}  # same as drift_eval (NUM_PREDICT=600)


@app.function(
    image=image,
    gpu="L40S",  # 48 GB: holds every model above in bf16 with room for batch-32+ KV cache
    volumes={"/root/.cache/huggingface": hf_volume},
    # Gated models (llama, mistral, gemma) need a Modal secret "huggingface" holding HF_TOKEN. None exists
    # yet: create it, then run with USE_HF_SECRET=1. Qwen is ungated and needs neither.
    secrets=[modal.Secret.from_name("huggingface")] if os.environ.get("USE_HF_SECRET") else [],
    timeout=3600,
)
def generate(hf_id: str, jobs: list) -> dict:
    import vllm
    from huggingface_hub import HfApi
    from vllm import LLM, SamplingParams

    rev = HfApi().model_info(hf_id).sha
    llm = LLM(hf_id, revision=rev, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.90, max_num_seqs=64, seed=0)
    states = [{"job": j, "msgs": [{"role": "system", "content": j["system"]}], "traj": []} for j in jobs]
    for t in range(1, 5):
        for s in states:
            s["msgs"].append({"role": "user", "content": s["job"]["turns"][t - 1]})
        sps = [SamplingParams(seed=s["job"]["seed"], **SAMPLING) for s in states]
        outs = llm.chat([s["msgs"] for s in states], sps)
        for s, o in zip(states, outs):
            reply = o.outputs[0].text
            s["msgs"].append({"role": "assistant", "content": reply})
            s["traj"].append({"turn": t, "user": s["job"]["turns"][t - 1], "assistant": reply,
                              "finish": o.outputs[0].finish_reason})
    return {"hf_id": hf_id, "revision": rev, "vllm": vllm.__version__,
            "results": [{"key": s["job"]["key"], "trajectory": s["traj"]} for s in states]}


@app.local_entrypoint()
def main(model: str, out: str = "", conds: str = "implicit,neutral,engagement,autonomy",
         seeds: str = "42,137", control_seeds: str = "42", paras: str = "A,B", limit: int = 0):
    import drift_eval as d
    from implicit_prompts import IMPLICIT, lint

    lint()
    conditions = {**d.CONDITIONS, "implicit": IMPLICIT}
    out = out or os.path.join(HERE, "runs", model.replace(":", "_"))
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "transcripts.jsonl")
    done = {r["key"] for r in d.load_jsonl(path)}

    jobs = []
    for script, sd in (("escalating", seeds), ("control", control_seeds)):
        for seed in map(int, sd.split(",")):
            for sc, turns in d.SCENARIOS.items():
                for cond in conds.split(","):
                    for para in paras.split(","):
                        key = f"{model}|{script}|{seed}|{sc}|{cond}|{para}"
                        if key in done:
                            continue
                        ut = turns if script == "escalating" else [turns[0]] + d.CONTROL_TURNS
                        jobs.append({"key": key, "seed": seed, "turns": ut, "system": conditions[cond][para],
                                     "meta": {"script": script, "seed": seed, "scenario": sc,
                                              "condition": cond, "paraphrase": para}})
    jobs = jobs[:limit] if limit else jobs
    print(f"{len(jobs)} trajectories ({len(done)} already done) on {MODELS[model]}")
    if not jobs:
        return
    res = generate.remote(MODELS[model], jobs)
    meta = {j["key"]: j["meta"] for j in jobs}
    with open(path, "a") as f:
        for r in res["results"]:
            f.write(json.dumps({"key": r["key"], "model": model, **meta[r["key"]],
                                "trajectory": r["trajectory"]}) + "\n")
    with open(os.path.join(out, "runs.jsonl"), "a") as f:  # provenance, one line per invocation
        json.dump({"model": model, "hf_id": res["hf_id"], "revision": res["revision"], "vllm": res["vllm"],
                   "sampling": SAMPLING, "seeds": seeds, "control_seeds": control_seeds, "conds": conds,
                   "command": " ".join(sys.argv), "n_jobs": len(jobs),
                   "finished_utc": datetime.now(timezone.utc).isoformat()}, f)
        f.write("\n")
    trunc = sum(t["finish"] == "length" for r in res["results"] for t in r["trajectory"])
    print(f"wrote {path}; {trunc} replies hit the 600-token cap")

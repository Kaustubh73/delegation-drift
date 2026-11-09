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
    # Exp 2 ladder:
    "qwen2.5:0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:14b-instruct-awq": "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "qwen2.5:32b-instruct-awq": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "qwen2.5:72b-instruct-awq": "Qwen/Qwen2.5-72B-Instruct-AWQ",           # H100 (see GPU_BIG)
    "gemma4:e4b": "google/gemma-4-E4B-it",
    "gemma4:12b": "google/gemma-4-12B-it",
    "qwen3.6:27b": "Qwen/Qwen3.6-27B-FP8",
}
GPU_BIG = {"Qwen/Qwen2.5-72B-Instruct-AWQ"}  # ~40 GB weights: needs H100 80 GB, not L40S

app = modal.App("m3-implicit-engagement")
hf_volume = modal.Volume.from_name("pgg-swap-hf", create_if_missing=True)  # shared HF cache
results_volume = modal.Volume.from_name("m3-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.31.0", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1500}


def _generate(hf_id: str, jobs: list, sampling: dict, run_id: str) -> dict:
    import vllm
    from huggingface_hub import HfApi
    from vllm import LLM, SamplingParams

    rev = HfApi().model_info(hf_id).sha
    is_awq = "awq" in hf_id.lower()
    llm = LLM(hf_id, revision=rev, dtype="auto" if is_awq else "bfloat16",
              quantization="awq" if is_awq else None, max_model_len=8192,
              gpu_memory_utilization=0.90, max_num_seqs=64, seed=0)
    states = [{"job": j, "msgs": [{"role": "system", "content": j["system"]}], "traj": []} for j in jobs]
    for t in range(1, 5):
        for s in states:
            s["msgs"].append({"role": "user", "content": s["job"]["turns"][t - 1]})
        sps = [SamplingParams(seed=s["job"]["seed"], **sampling) for s in states]
        ctk = {"enable_thinking": False} if ("qwen3.6" in hf_id.lower() or "gemma-4" in hf_id.lower()) else None
        outs = llm.chat([s["msgs"] for s in states], sps, chat_template_kwargs=ctk)
        for s, o in zip(states, outs):
            reply = o.outputs[0].text
            s["msgs"].append({"role": "assistant", "content": reply})
            s["traj"].append({"turn": t, "user": s["job"]["turns"][t - 1], "assistant": reply,
                              "finish": o.outputs[0].finish_reason})
    payload = {"hf_id": hf_id, "revision": rev, "vllm": vllm.__version__,
               "results": [{"key": s["job"]["key"], "trajectory": s["traj"]} for s in states]}
    os.makedirs(f"/results/{hf_id}", exist_ok=True)
    with open(f"/results/{hf_id}/{run_id}.json", "w") as f:
        json.dump({**payload, "jobs": [j["meta"] | {"key": j["key"]} for j in jobs]}, f)
    results_volume.commit()
    return payload


_kw = dict(
    image=image,
    volumes={"/root/.cache/huggingface": hf_volume, "/results": results_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3 * 3600,
)


@app.function(gpu="L40S", **_kw)
def generate(hf_id: str, jobs: list, sampling: dict, run_id: str) -> dict:
    return _generate(hf_id, jobs, sampling, run_id)


@app.function(gpu="H100", **_kw)
def generate_h100(hf_id: str, jobs: list, sampling: dict, run_id: str) -> dict:
    return _generate(hf_id, jobs, sampling, run_id)


@app.local_entrypoint()
def main(model: str, out: str = "", conds: str = "implicit,neutral,engagement,autonomy",
         seeds: str = "42,137,7,2024", control_seeds: str = "42", paras: str = "A,B", limit: int = 0,
         max_tokens: int = SAMPLING["max_tokens"]):
    import drift_eval as d
    from implicit_prompts import IMPLICIT, lint

    lint()
    conditions = {**d.CONDITIONS, "implicit": IMPLICIT}
    out = out or os.path.join(HERE, "runs", model.replace(":", "_") + ("_smoke" if limit else ""))
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
    sampling = {**SAMPLING, "max_tokens": max_tokens}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    res = (generate_h100 if MODELS[model] in GPU_BIG else generate).remote(MODELS[model], jobs, sampling, run_id)
    print(f"saved on volume m3-results: {MODELS[model]}/{run_id}.json")
    meta = {j["key"]: j["meta"] for j in jobs}
    with open(path, "a") as f:
        for r in res["results"]:
            f.write(json.dumps({"key": r["key"], "model": model, **meta[r["key"]],
                                "trajectory": r["trajectory"]}) + "\n")
    trunc = sum(t["finish"] == "length" for r in res["results"] for t in r["trajectory"])
    n_replies = sum(len(r["trajectory"]) for r in res["results"])
    with open(os.path.join(out, "runs.jsonl"), "a") as f:
        json.dump({"run_id": run_id, "truncated": trunc, "n_replies": n_replies, "limit": limit,
                   "model": model, "hf_id": res["hf_id"], "revision": res["revision"], "vllm": res["vllm"],
                   "sampling": sampling, "seeds": seeds, "control_seeds": control_seeds, "conds": conds,
                   "command": " ".join(sys.argv), "n_jobs": len(jobs),
                   "finished_utc": datetime.now(timezone.utc).isoformat()}, f)
        f.write("\n")
    print(f"wrote {path}; {trunc}/{n_replies} replies hit the {max_tokens}-token cap")

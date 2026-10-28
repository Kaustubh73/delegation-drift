#!/bin/sh
# Simulator sensitivity: llama3.1-8b Tier B slice, simulator qwen2.5:32b-instruct instead of gpt-oss:20b (same seeds/grid/stop judge)
cd "$(dirname "$0")/.." || exit 1
M=llama3.1:8b-instruct-q4_K_M
O=data/runs_adaptive/tierB_llama_sim32b
python3 src/drift_eval.py run --model $M --out $O --seeds --control-seeds --persistent-seeds 42 137 --max-turns 8 --user-sim qwen2.5:32b-instruct || exit 1
python3 src/drift_eval.py judge --judge qwen2.5:14b-instruct --out $O || exit 1
python3 src/drift_eval.py judge --judge gemma3:12b --out $O || exit 1
echo ALLDONE

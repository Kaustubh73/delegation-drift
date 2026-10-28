#!/bin/sh
# Tier B (persistent, stop at first qwen -1), seeds 42+137, paraphrases A+B, sim gpt-oss:20b
cd "$(dirname "$0")/.." || exit 1
for M in qwen2.5:7b-instruct gemma3:12b; do
  D=$(echo "$M" | tr ':' '_')
  O=data/runs_adaptive/tierB_$D
  python3 src/drift_eval.py run --model $M --out $O --seeds --control-seeds --persistent-seeds 42 137 --max-turns 8 --user-sim gpt-oss:20b || exit 1
  python3 src/drift_eval.py judge --judge qwen2.5:14b-instruct --out $O || exit 1
  python3 src/drift_eval.py judge --judge gemma3:12b --out $O || exit 1
  echo "DONE $M"
done
echo ALLDONE

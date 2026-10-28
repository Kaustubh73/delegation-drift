#!/bin/sh
# Tier B for remaining 2 of 5 models; same settings as run_tierB_models.sh. Waits for the gemma3 judge resume first.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "drift_eval.py judge" >/dev/null; do sleep 60; done
for M in mistral:7b-instruct gemma4:e4b; do
  D=$(echo "$M" | tr ':' '_')
  O=data/runs_adaptive/tierB_$D
  python3 src/drift_eval.py run --model $M --out $O --seeds --control-seeds --persistent-seeds 42 137 --max-turns 8 --user-sim gpt-oss:20b || exit 1
  python3 src/drift_eval.py judge --judge qwen2.5:14b-instruct --out $O || exit 1
  python3 src/drift_eval.py judge --judge gemma3:12b --out $O || exit 1
  echo "DONE $M"
done
echo ALLDONE

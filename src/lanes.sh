#!/bin/zsh
# lane gen: subject models sequentially. lane judge: loop judging finished/in-progress dirs.
E="$(dirname "$0")/drift_eval.py"
R=${RUNS:-data/runs}
MODELS=(llama3.1:8b-instruct-q4_K_M qwen2.5:7b-instruct gemma3:12b mistral:7b-instruct gemma4:e4b)
waitfor() { until ollama list | grep -q "^$1 "; do sleep 60; done; }
if [[ $1 == gen ]]; then
  # breadth-first phases; run is resumable so each phase only adds missing trajectories
  for phase in "--seeds 42 --control-seeds" "--seeds 42 --control-seeds 42" "--seeds 42 137 --control-seeds 42"; do
    for m in $MODELS; do waitfor $m; python3 $E run --model $m --out $R/${m//[:\/]/_} ${=phase}; done
  done
  touch $R/GEN_DONE
else
  waitfor qwen2.5:14b-instruct
  while true; do
    fin=$([[ -f $R/GEN_DONE ]] && echo 1)
    for d in $R/*/; do python3 $E judge --judge qwen2.5:14b-instruct --out $d >/dev/null; done
    [[ -n $fin ]] && break; sleep 120
  done
  for d in $R/*/; do python3 $E judge --judge gemma3:12b --out $d >/dev/null; done
  touch $R/JUDGE_DONE
fi

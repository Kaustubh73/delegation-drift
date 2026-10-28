# delegation-drift

Does a "keep users engaged" system prompt make AI assistants take over decisions, turn by turn?

This is a v0 evaluation of delegation drift. Over a short conversation, does an assistant move from helping a user decide to deciding for them? It is not peer reviewed.

![Takeover rate by turn](results/fig1_takeover_by_turn.png)

## Key results

The setup is 5 open-weight models (4B to 12B parameters), 8 scenarios, and 3 system prompts (engagement, neutral, autonomy) with 2 paraphrases each. Each conversation has 4 turns. In 480 conversations the scripted user escalates (2 seeds per cell). In 240 control conversations the user never escalates (1 seed per cell). That gives 720 trajectories and 2,880 assistant turns. Two LLM judges (Qwen2.5-14B and Gemma3-12B) score each turn +1, 0 or -1 using [RUBRIC.md](RUBRIC.md). A takeover is a score of -1.

- With the neutral prompt, 28% to 72% of turn-4 replies are takeovers, depending on the model.
- The engagement prompt raises turn-4 takeover by 23 to 25 percentage points over the neutral prompt, pooled across models (95% CI roughly +11 to +39, scenario-cluster bootstrap). `analyze` prints the exact values per judge.
- Control conversations, where the user never escalates, show 0% takeover in every cell except one (Mistral-7B under the engagement prompt, 6% by the Gemma judge). The engagement prompt lowers the threshold for accepting a hand-over, but it does not cause drift by itself.
- The autonomy prompt cuts takeover for 4 of 5 models. Qwen2.5-7B takes over in 59% to 78% of turn-4 replies under every prompt, so neither the engagement nor the autonomy prompt moves it.
- Claims of real-world actions ("I sent the email") occur mostly under the engagement prompt. The stricter LLM tagger flagged 98 turns: 95 engagement, 3 neutral, 0 autonomy. No human has verified these tags.

![Turn-4 takeover by model](results/fig2_takeover_by_model.png)

### How good are the judges?

I labelled 52 turns by hand, blind to condition. The judges agree with me moderately: Qwen has exact agreement 0.65 and weighted kappa 0.52, Gemma 0.54 and 0.37. Neither judge called a takeover that I did not (precision 100%), but they miss many (recall 69% and 54%), so the reported takeover rates are probably underestimates. Re-running `calibrate` on 2026-10-15 reproduced Qwen exactly, but Gemma scored 0.56 and 0.39 because one turn changed, so even temperature-0 judging is not perfectly repeatable in Ollama. The figure and the numbers above use the stored judge outputs.

![Judge vs human](results/fig3_judge_vs_human.png)

I also tried `gpt-oss:20b` as a judge, with the same prompt and settings, on the same 52 turns. It agrees with me much more closely: exact agreement 0.79, weighted kappa 0.76, takeover precision 12/14 and recall 12/13, with no failed parses. This was one run on 52 turns, so the standard error is about 0.06. Its 11 disagreements are all at the 0 boundary, 8 of them where I scored 0 and it scored +1. It has not been used for any reported result. Per-turn outputs are in `data/gold/calibration_gptoss_20b.json`.

## Limitations

- User turns are scripted and do not react to the assistant's pushback.
- Small models only (4B to 12B parameters), and no closed frontier models.
- The judges are LLMs with the agreement above, and the gold set is 52 turns from one annotator.
- A single run, with only 32 escalating trajectories per model and prompt.

## Reproduce

Run everything from the repository root. The scripts use only the Python 3 standard library (tested with Python 3.13), except the figures, which need `pip install -r requirements.txt`.

Without any models, using the shipped data:

```bash
python3 src/drift_eval.py analyze --out data/runs/*/   # results tables
python3 src/make_figures.py                            # regenerates results/*.png
```

With [Ollama](https://ollama.com) running and these models pulled (`ollama pull <tag>`): `llama3.1:8b-instruct-q4_K_M`, `qwen2.5:7b-instruct`, `gemma3:12b`, `mistral:7b-instruct`, `gemma4:e4b`, plus the judge `qwen2.5:14b-instruct` (and `gpt-oss:20b`, 14 GB, if you want to try it as a judge):

```bash
python3 src/drift_eval.py calibrate --gold data/gold/gold_human_v2_52.json \
  --transcripts data/runs/*/transcripts.jsonl --judge qwen2.5:14b-instruct gemma3:12b   # judge vs my labels, about 15 minutes

RUNS=data/runs src/lanes.sh gen     # regenerate all trajectories, hours, resumable
RUNS=data/runs src/lanes.sh judge   # re-score them with both judges
```

The other subcommands are `run`, `judge`, `tag` and `tag_strict`; see `src/drift_eval.py`. Seeds, settings and model digests are in `data/runs/<model>/runs.jsonl`.

## Layout

| Path | Contents |
|---|---|
| `src/drift_eval.py` | harness: run, judge, calibrate, tag, analyze, survival |
| `src/make_figures.py` | builds the figures in `results/` |
| `src/lanes.sh` | runs all models and judges in sequence |
| `src/run_tierB_models*.sh`, `src/run_simsens.sh` | persistent (LLM-simulated user) runs |
| `src/tierB_analysis.py`, `src/fig_survival.py`, `src/overrefusal_t4.py` | analysis of the persistent runs and the over-refusal check |
| `src/judges/` | scripts that add a third judge (gpt-oss, Qwen2.5-32B on Modal) to the scripted runs |
| `src/implicit/` | scripts for an implicit-engagement condition (prompts, Modal generation, analysis); not run yet |
| `src/pilot_v0/` | the original pilot scripts |
| `data/runs/` | raw trajectories and judge outputs per model (read-only), including the extra judge files |
| `data/runs_adaptive/` | raw persistent-user runs and pilots; not yet analysed in this README |
| `data/pilot_v0/` | raw output of the first pilot |
| `data/gold/` | my 52 human labels, the items they refer to, judge-agreement summaries, and gpt-oss per-turn outputs |
| `results/` | figures |
| `RUBRIC.md` | score definitions and the decision procedure behind the judge prompt |
| `references.bib` | cited and model papers |

Paths inside the scripts were updated when the repository was reorganised. The new layout has not been re-run end to end, so report anything that fails.

## AI use

Research design, analysis and conclusions are mine. I used AI coding assistants (Claude Code and others) to help write and debug code and drafts, and I reviewed the results.

## License

MIT, see [LICENSE](LICENSE).

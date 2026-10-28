"""Over-refusal (refuses_no_help tag, qwen2.5-14b) vs qwen judge score, scripted escalating arm, turn 4. python3 -I overrefusal_t4.py"""
import json, glob, collections
c = collections.defaultdict(lambda: [0, 0, 0, 0])
for d in glob.glob('data/runs/*/'):
    T = {json.loads(l)['key']: json.loads(l) for l in open(d + 'transcripts.jsonl')}
    J = {(r['key'], r['turn']): r['score'] for r in map(json.loads, open(d + 'judge_qwen2.5_14b-instruct.jsonl'))}
    for t in map(json.loads, open(d + 'tags_qwen2.5_14b-instruct.jsonl')):
        r = T[t['key']]
        if r['script'] != 'escalating' or t['turn'] != 4 or 'refuses_no_help' not in t: continue  # tag file also holds false-action-only rows
        x = c[r['condition']]; s = J.get((t['key'], 4)); ref = bool(t['refuses_no_help'])
        x[0] += 1; x[1] += s == 1; x[2] += ref; x[3] += ref and s == 1
print("| prompt | T4 replies tagged | judge +1 | refusal tag | refusal and +1 |\n|---|---|---|---|---|")
for k in ("engagement", "neutral", "autonomy"): print(f"| {k} | {c[k][0]} | {c[k][1]} | {c[k][2]} | {c[k][3]} |")

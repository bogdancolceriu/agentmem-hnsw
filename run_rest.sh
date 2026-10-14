#!/bin/bash
# Remaining experiments, in sequence (timing-sensitive benchmarks run on a quiet machine).
cd "$(dirname "$0")"
python3 -m experiments.exp_town --burns 90 30 --backends rerank100_norecent --seeds 30 --workers 2 > results/town_norecent.log 2>&1
while pgrep -f "^python3 -m experiments.exp_hnsw_ablation" > /dev/null; do sleep 20; done
python3 -m experiments.exp_hnsw_ablation validate --glove-n 500000 > results/hnsw_validate.log 2>&1
python3 -m experiments.exp_static --dataset glove --sizes 10000 100000 1000000 > results/static_glove.log 2>&1
python3 -m experiments.exp_static --dataset town > results/static_town.log 2>&1
python3 -m experiments.exp_static --rel-sweep 100000 > results/static_relsweep.log 2>&1
echo done > results/run_rest.done

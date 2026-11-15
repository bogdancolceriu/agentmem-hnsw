# When the index shapes the agent: exact memory retrieval for generative agents with HNSW

Seminar extension for *Sisteme de agenți inteligenți* (UTCN, Inginerie Software, 2026-2027),
built on article 11: Y. A. Malkov and D. A. Yashunin, *Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*, IEEE TPAMI 2020.

Author: Colceriu Mihai Bogdan

## The project in three points

1. **The article.** HNSW written from the paper's pseudocode (`agentmem/paper_hnsw.py`,
   Algorithms 1-5, compiled with numba) matches the recall of `hnswlib`, the authors' C++ library,
   and reproduces the paper's parameter study: the hierarchy pays off in low dimensions, the
   neighbour-selection heuristic matters on clustered and real data.
2. **The extension.** Generative agents (Park et al., UIST 2023) retrieve memories by
   `recency + importance + relevance`. A vector index only knows similarity, so frameworks fetch
   the K most similar memories and re-rank them (LangChain's `TimeWeightedVectorStoreRetriever`
   uses K = 100), which silently drops recent or important memories. With exponential decay the
   full score is *exactly* an inner product of two augmented vectors, so one inner-product HNSW
   index searches for the true top-k directly (`agentmem/score.py`, `agentmem/stores.py`).
   Wei et al. (2026, TANGO, arXiv:2609.00548) published the same identity independently for
   similarity plus freshness. This store adds what agents need: importance, the paper's min-max
   normalization, refresh on access and bounded coordinates.
   - On the memories of simulated agents it returns 96% of the exact top-10, where re-ranking
     the 100 most similar memories returns 28%. With LangChain's own classes on single agents'
     memories: 100% against 52%.
   - In a stress test with one million memories whose metadata is independent of their content:
     97.5% against under 1%, about a hundred times faster than a linear scan.
   - For the few thousand memories of one agent an exact scan is faster than any index. The
     index matters from tens of thousands of memories on.
   - The same idea covers forgetting curves beyond the exponential (`agentmem/curves.py`,
     `agentmem/curve_stores.py`): ACT-R's power law over every use of a memory is within 0.3% of
     a sum of eight exponentials, which folds into the index in the same way.
3. **Why it matters.** Willowbrook (`town/`) is a closed-loop simulation of 25 rule-based agents,
   calibrated so that exact retrieval reproduces the party scenario of the paper (13 of 25 agents
   knew about the party, 5 of 12 invited attended). In towns older than a week, plain similarity
   search cuts the spread of the invitation by about 40%, and overfetch-and-re-rank shifts it in
   both directions. The forgetting curve matters too: when every retrieval counts as a use, as
   in ACT-R, the most retrieved memories crowd out new ones, unless a retrieval counts as a small
   fraction of a use (0.03), which still folds.

Secondary results, in the appendices of the report: ablations and the case in which relevance
dominates the score, a LangChain retriever backed by the folded index
(`agentmem/langchain_retriever.py`) and a check of the public Generative Agents code
(`experiments/ga_reference_check.py`), whose recency term is rank-based and sorted the wrong way.

The report is `report/main.pdf`. A guide for the presentation, in Romanian, is
`docs/ghid_aparare.md`.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests/                                     # 45 tests, ~40 s
python demo/tour.py                                            # guided tour on small data, in Romanian (~2 min)
python demo/tour.py --oras                                     # ... plus two short town runs

python -m experiments.exp_hnsw_ablation validate fig3 fig4 fig7 glove   # HNSW study (~2 h)
python -m experiments.exp_static --dataset glove --sizes 10000 100000 1000000
python -m experiments.exp_static --dataset town
python -m experiments.exp_static --rel-sweep 100000
python -m experiments.exp_normef                               # depth of the range searches (~15 min)
python -m experiments.exp_buildvar                             # build-to-build variation (~2 min)
python -m experiments.time_exact_knn                           # exact k-NN scan time on GloVe
python -m experiments.exp_town --burns 0 7 30 90 --seeds 30    # multi-agent experiment (~3 h, 2 cores)
python -m experiments.exp_curves --part fit                    # forgetting-curve fits
python -m experiments.exp_curves --part static --sizes 100000 1000000   # ACT-R score at scale (~40 min)
python -m experiments.exp_curves --part fronts --sizes 100000 1000000   # Pareto front sizes
python -m experiments.exp_curves --part ranges --sizes 100000           # estimated vs exact ranges
python -m experiments.exp_town_curve --burns 90 30 --seeds 30  # town with forgetting curves (~3 h)
python -m experiments.exp_town_curve --burns 90 30 --scores uses_log --backends exact --seeds 30
python -m experiments.town_curve_diag --seeds 5                # why counting every use locks memory in
python -m experiments.exp_town_curve --burns 90 30 --scores uses_w003 uses_w01 uses_w03 --backends exact --seeds 30
python -m experiments.exp_town_curve --burns 30 --scores uses_w003 --backends folded --seeds 30
python -m experiments.exp_langchain
python -m experiments.ga_reference_check --repo /path/to/generative_agents
python -m experiments.plots                                    # figures/ from results/
python -m experiments.tables                                   # report/generated/ from results/
```

The GloVe vectors are downloaded from the gensim-data GitHub releases on first use
(`data/raw/`, ~540 MB) and cached as `.npy` in `data/cache/`.

On a machine with a GPU, `python -m experiments.llm_validation --seeds 10` repeats the party
scenario with planning and conversation decisions taken by a local LLM through Ollama. Once
`results/llm_validation.csv` exists, `experiments/tables.py` and the report pick it up
(`docs/RULARE_PE_LAPTOP.md` has the steps, in Romanian).

## Layout

```
agentmem/   score.py (score, folding), stores.py (memory stores), curves.py (forgetting curves
            as sums of exponentials), curve_stores.py (stores for any curve), paper_hnsw.py
            (HNSW from the paper), embed.py (GloVe-SIF sentence embeddings), langchain_retriever.py
town/       world.py (residents, places, routines), sim.py (closed-loop agent simulation)
demo/       tour.py (guided tour of every idea on small data), willowbrook.html (step-by-step replay
            of the party with three retrieval methods, open it in a browser)
experiments/ one script per experiment, plots.py, tables.py
tests/      correctness tests (folding is exact, stores agree with brute force)
results/    CSV / JSON outputs        figures/  PDF + PNG        report/  LaTeX report
```

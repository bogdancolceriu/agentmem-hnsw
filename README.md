# When the index shapes the agent: exact memory retrieval for generative agents with HNSW

Seminar extension for *Sisteme de agenți inteligenți* (UTCN, Inginerie Software, 2026-2027),
built on article 11: Y. A. Malkov and D. A. Yashunin, *Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*, IEEE TPAMI 2020.

Author: Colceriu Mihai Bogdan

## What is here

1. **HNSW from scratch** (`agentmem/paper_hnsw.py`): Algorithms 1-5 of the paper, compiled
   with numba, counting every distance evaluation. Validated against `hnswlib` (the authors'
   C++ library) and used to reproduce the paper's parameter study (hierarchy `mL`, neighbour
   selection heuristic, clustered data, real data).
2. **The extension** (`agentmem/score.py`, `agentmem/stores.py`): generative agents
   (Park et al., UIST 2023) retrieve memories by `recency + importance + relevance`.
   Vector indexes only support similarity, so frameworks fetch the K most similar memories and
   re-rank them (LangChain's `TimeWeightedVectorStoreRetriever` uses K = 100), which silently
   drops recent or important memories. With exponential decay the full score is *exactly* an
   inner product of two augmented vectors (forward decay, Cormode et al. 2009), so a single
   inner-product HNSW index searches for the true top-k directly, with weights and time chosen per
   query (recall@10 of 0.975 with one million memories, against 0.006 for LangChain's re-ranking).
   The store handles the parts that make this practical: time anchors and segments with
   compaction (bounded coordinates), refresh-on-access through a write buffer and the paper's
   min-max normalization (a per-query rescaling of the weights).
3. **A LangChain retriever** (`agentmem/langchain_retriever.py`) with the scoring rule and the main fields
   of LangChain's `TimeWeightedVectorStoreRetriever`, backed by the folded index.
4. **Willowbrook** (`town/`): a closed-loop simulation of 25 rule-based generative agents
   (perception, conversations, planning through memory retrieval), calibrated so that exact
   retrieval reproduces the information diffusion reported in the paper (13 of 25 agents knew
   about the party, 5 of 12 invited attended). Agents can live for months before the party, so
   memory size grows and retrieval approximations start to matter.
5. **A check of the reference implementation** (`experiments/ga_reference_check.py`): the
   public Generative Agents code is run unmodified on mock personas. Its recency term is
   rank-based and sorted the wrong way: the memory accessed longest ago gets the highest
   recency score.

## Reproduce

```bash
pip install numpy scipy pandas matplotlib hnswlib faiss-cpu numba pytest langchain-core langchain-classic
python -m pytest -q tests/                                     # 26 tests, ~30 s

python -m experiments.exp_hnsw_ablation validate fig3 fig4 fig7 glove   # HNSW study (~2 h)
python -m experiments.exp_static --dataset glove --sizes 10000 100000 1000000
python -m experiments.exp_static --dataset town
python -m experiments.exp_static --rel-sweep 100000
python -m experiments.exp_normef                               # depth of the range searches (~15 min)
python -m experiments.exp_buildvar                             # build-to-build variation (~2 min)
python -m experiments.time_exact_knn                           # exact k-NN scan time on GloVe
python -m experiments.exp_town --burns 0 7 30 90 --seeds 30    # multi-agent experiment (~3 h, 2 cores)
python -m experiments.exp_langchain
python -m experiments.ga_reference_check --repo /path/to/generative_agents
python -m experiments.plots                                    # figures/ from results/
python -m experiments.tables                                   # report/generated/ from results/
```

The GloVe vectors are downloaded from the gensim-data GitHub releases on first use
(`data/raw/`, ~540 MB) and cached as `.npy` in `data/cache/`.

Optional, on a machine with a GPU: `python -m experiments.llm_validation` repeats the party
scenario with planning and conversation decisions taken by a local LLM through Ollama
(see the docstring of that file).

## Layout

```
agentmem/   score.py (score, folding), stores.py (memory stores), paper_hnsw.py (HNSW from
            the paper), embed.py (GloVe-SIF sentence embeddings), langchain_retriever.py
town/       world.py (residents, places, routines), sim.py (closed-loop agent simulation)
experiments/ one script per experiment, plots.py, tables.py
tests/      correctness tests (folding is exact, stores agree with brute force)
results/    CSV / JSON outputs        figures/  PDF + PNG        report/  LaTeX report
```

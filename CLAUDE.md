# agentmem-hnsw: notes for Claude Code

Seminar project (UTCN, *Sisteme de agenți inteligenți*). The owner presents it himself and must be
able to explain every part, so prefer explaining over rewriting. Talk to him in Romanian.

## The project in three points

1. **The article.** HNSW rewritten from the paper's pseudocode (`agentmem/paper_hnsw.py`) behaves
   as the paper says.
2. **The extension.** The Generative Agents score (recency + importance + relevance) is an inner
   product, so one HNSW index searches it directly (`agentmem/score.py`, `agentmem/stores.py`).
   The same holds for other forgetting curves (`agentmem/curves.py`, `agentmem/curve_stores.py`).
3. **Why it matters.** In a simulated town of 25 agents (`town/`), how agents retrieve and forget
   changes what they do.

Everything else (ablations, IVF, the reference-code check, the LangChain retriever) is secondary
and lives in the report's appendices.

## Where things are

- `experiments/` one script per experiment, raw results go to `results/`
- `experiments/plots.py` writes `figures/`, `experiments/tables.py` writes `report/generated/`
- `report/main.tex` (+ `town_section.tex`, `curves_section.tex`, `llm_section.tex`)
- `docs/ghid_aparare.md` study guide, `docs/PLAN_REVIZIE.md` plan and journal,
  `docs/RULARE_PE_LAPTOP.md` steps for the laptop run
- `demo/tour.py` two-minute guided tour, `tests/` 45 unit tests

## Rules

- No number in the report is typed by hand. Every figure comes from `experiments/tables.py`
  (macros in `report/generated/numbers.tex`).
- Do not rerun the large experiments (`exp_hnsw`, `exp_static`, `exp_town`, `exp_town_curve`).
  Their results are in `results/` and must stay as they are. After any change,
  `git diff report/generated/numbers.tex` should show only the lines you meant to add.
- The one experiment still to run is `experiments/llm_validation.py` (needs a GPU and Ollama).
  See `docs/RULARE_PE_LAPTOP.md`.
- Report text: British English, plain words, no em dashes, no semicolons, no comma before "and".
  Romanian texts: natural Romanian, the same punctuation rules.
- State results as they are, including the unfavourable ones. Do not claim more than a table shows.
- Ask before committing or pushing.

"""Run the reference Generative Agents retrieval code and compare it with the paper.

We execute the functions of reverie/backend_server/persona/cognitive_modules/
retrieve.py from the public repository (joonspk-research/generative_agents,
Apache-2.0, commit fe05a71) unmodified, with only the OpenAI embedding call
replaced by our sentence encoder, on mock personas built from our memory
streams. Three differences from Sec. 4.1 of the paper show up:

  1. recency is computed from the *rank* of a memory in a list sorted by last
     access time (decay 0.995 per position), not from game hours;
  2. the list is sorted in ascending order, so the memory accessed longest ago
     receives the highest recency score and the most recently accessed one the
     lowest (after min-max scaling: 1.0 and 0.0);
  3. the weights are gw = [0.5, 3, 2] for (recency, relevance, importance)
     instead of all ones.

    python -m experiments.ga_reference_check --repo /path/to/generative_agents
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import datetime as dt
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy import dot
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
RETRIEVE = "reverie/backend_server/persona/cognitive_modules/retrieve.py"
BASE = dt.datetime(2023, 3, 2)


def load_reference(repo: Path, encoder) -> dict:
    """Exec every function definition of the reference retrieve.py (imports are skipped)."""
    src = (repo / RETRIEVE).read_text()
    tree = ast.parse(src)
    funcs = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    ns = dict(dot=dot, norm=norm, get_embedding=lambda text: encoder.encode([text])[0].tolist())
    exec(compile(funcs, str(repo / RETRIEVE), "exec"), ns)
    return ns


def make_persona(texts, imps, t_last_hours, now_hours, encoder, decay=0.995):
    """Mock persona with the attributes new_retrieve() touches."""
    nodes, embeddings = [], {}
    for i, (txt, imp, t) in enumerate(zip(texts, imps, t_last_hours)):
        node = SimpleNamespace(node_id=f"node_{i}", embedding_key=txt, poignancy=int(round(imp * 10)),
                               last_accessed=BASE + dt.timedelta(hours=float(t)), hour=float(t))
        nodes.append(node)
        if txt not in embeddings:
            embeddings[txt] = encoder.encode([txt])[0].tolist()
    nodes = nodes[::-1]  # the reference inserts new nodes at index 0 (newest first)
    a_mem = SimpleNamespace(seq_event=nodes, seq_thought=[], embeddings=embeddings,
                            id_to_node={n.node_id: n for n in nodes})
    # values from base_the_ville_n25/personas/*/bootstrap_memory/scratch.json
    scratch = SimpleNamespace(recency_w=1, relevance_w=1, importance_w=1, recency_decay=decay,
                              curr_time=BASE + dt.timedelta(hours=float(now_hours)))
    return SimpleNamespace(a_mem=a_mem, scratch=scratch)


def reference_retrieve(ref: dict, persona, focal: str, k: int) -> list[int]:
    """Call the reference new_retrieve on a copy (it updates last_accessed of what it returns)."""
    nodes = [SimpleNamespace(**vars(n)) for n in persona.a_mem.seq_event]
    a_mem = SimpleNamespace(seq_event=nodes, seq_thought=[], embeddings=persona.a_mem.embeddings,
                            id_to_node={n.node_id: n for n in nodes})
    p = SimpleNamespace(a_mem=a_mem, scratch=copy.copy(persona.scratch))
    with contextlib.redirect_stdout(io.StringIO()):
        out = ref["new_retrieve"](p, [focal], k)[focal]
    return [int(n.node_id.split("_")[1]) for n in out]


def toy(ref: dict, encoder) -> dict:
    """Five memories that differ only in when they were last accessed."""
    hours = [0.0, 10.0, 20.0, 30.0, 40.0]
    # identical text -> identical relevance; identical importance; only recency differs
    texts_same = ["Klaus saw a red bird in the park"] * 5
    persona_same = make_persona(texts_same, [0.3] * 5, hours, 41.0, encoder)
    top_ref = reference_retrieve(ref, persona_same, "What did Klaus see?", 1)
    order_ref = reference_retrieve(ref, persona_same, "What did Klaus see?", 5)
    # the recency values the reference code assigns, normalized as in new_retrieve
    nodes = sorted([[n.last_accessed, n] for n in persona_same.a_mem.seq_event], key=lambda x: x[0])
    nodes = [n for _, n in nodes]
    rec = ref["normalize_dict_floats"](ref["extract_recency"](persona_same, nodes), 0, 1)
    rec_by_hour = {nodes[i].hour: round(v, 3) for i, v in enumerate(rec.values())}
    paper = {h: round(float(v), 3) for h, v in
             zip(hours, (0.995 ** (41.0 - np.array(hours)) - 0.995 ** 41.0) / (0.995 ** 1.0 - 0.995 ** 41.0))}
    return dict(last_access_hours=hours, now=41.0,
                reference_recency_by_last_access_hour=rec_by_hour,
                paper_recency_by_last_access_hour=paper,
                reference_top1_last_access_hour=hours[top_ref[0]],
                reference_ranking_last_access_hours=[hours[i] for i in order_ref])


def town_comparison(ref: dict, encoder, burn: int = 30, seed: int = 0, k: int = 30) -> list[dict]:
    from agentmem import Weights, exact_scores, topk
    from town.sim import SimConfig, Town
    from town.world import date_str

    town = Town(SimConfig(days=burn + 2, party_day=burn + 1, seed=seed, measure_recall=False),
                encoder=encoder)
    town.run()
    T = town.cfg.days * 24.0 - 0.25
    rows = []
    for a in town.agents:
        n = a.store.n
        texts, imp, t_last = a.text[:n], a.store.imp[:n], a.store.t_last[:n]
        persona = make_persona(texts, imp, t_last, T, encoder)
        for focal in (f"{a.name}'s plan for {date_str(town.cfg.days)}.",
                      f"Important recent events for {a.name}'s life."):
            q = encoder.encode([focal])[0]
            got_ref = reference_retrieve(ref, persona, focal, k)
            E = a.store.E[:n]
            paper = topk(exact_scores(E, imp, t_last, q, T, Weights(1, 1, 1, minmax=True)), k)
            code_w = topk(exact_scores(E, imp, t_last, q, T, Weights(0.5, 2, 3, minmax=True)), k)
            age = lambda ids: float(np.mean(T - t_last[np.asarray(ids)]))
            rows.append(dict(agent=a.name, focal=focal, memories=n,
                             age_reference=age(got_ref), age_paper=age(paper), age_code_weights=age(code_w),
                             overlap_reference_paper=len(set(got_ref) & set(paper.tolist())) / k,
                             overlap_reference_codeweights=len(set(got_ref) & set(code_w.tolist())) / k,
                             news_reference=sum(a.kind[i] == 2 for i in got_ref),
                             news_paper=sum(a.kind[i] == 2 for i in paper.tolist())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/claude/joonspk-research/generative_agents")
    ap.add_argument("--out", default=str(ROOT / "results" / "ga_reference_check.json"))
    args = ap.parse_args()
    from town.sim import load_encoder

    enc = load_encoder()
    ref = load_reference(Path(args.repo), enc)
    res = dict(toy=toy(ref, enc))
    print("toy example (5 memories, identical except last access hour; now = 41 h)")
    for key, val in res["toy"].items():
        print(f"  {key}: {val}")
    rows = town_comparison(ref, enc)
    res["town"] = rows
    m = lambda key: float(np.mean([r[key] for r in rows]))
    res["town_summary"] = {key: m(key) for key in ("memories", "age_reference", "age_paper",
                                                   "age_code_weights", "overlap_reference_paper",
                                                   "overlap_reference_codeweights", "news_reference",
                                                   "news_paper")}
    print("\ntown, 25 agents x 2 planning focal points, top-30 after a 30-day burn-in")
    for key, val in res["town_summary"].items():
        print(f"  {key}: {val:.2f}")
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

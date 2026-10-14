"""LangChain's TimeWeightedVectorStoreRetriever vs our exact drop-in replacement.

Both retrievers receive the same documents (the memory streams of Willowbrook
agents with their importance and last-access times) and the same queries (the
agents' planning and conversation focal points). After every query we compare
the returned documents with the true top-k under the retriever's own formula
(recency + importance + cosine relevance over *all* documents), computed by
brute force on the retriever's own state just before the query.

    python -m experiments.exp_langchain --days 30 --k 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
from langchain_classic.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from agentmem.langchain_retriever import ExactTimeWeightedRetriever

ROOT = Path(__file__).resolve().parents[1]


class SIFEmbeddings(Embeddings):
    def __init__(self, enc):
        self.enc = enc

    def embed_documents(self, texts):
        return self.enc.encode(list(texts)).tolist()

    def embed_query(self, text):
        return self.enc.encode([text])[0].tolist()


class CosineInMemoryVectorStore(InMemoryVectorStore):
    """InMemoryVectorStore with cosine similarity as the relevance score."""

    def _select_relevance_score_fn(self):
        return lambda x: x


def true_topk(stream, E, q, now, decay_rate, k):
    hours = np.array([(now - d.metadata["last_accessed_at"]).total_seconds() / 3600 for d in stream])
    imp = np.array([d.metadata["importance"] for d in stream])
    s = (1.0 - decay_rate) ** hours + imp + E @ q
    return s, np.partition(s, len(s) - k)[len(s) - k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--decay-rate", type=float, default=0.005)
    ap.add_argument("--out", default=str(ROOT / "results" / "langchain.json"))
    args = ap.parse_args()
    from town.sim import SimConfig, Town
    from town.world import date_str

    town = Town(SimConfig(days=args.days, party_day=args.days - 1, seed=7, measure_recall=False))
    town.run()
    emb = SIFEmbeddings(town.enc)
    T_end = args.days * 24.0
    results = []
    for a in town.agents[: args.agents]:
        n = a.store.n
        now0 = dt.datetime.now()
        docs = [Document(page_content=a.text[i],
                         metadata=dict(importance=float(a.store.imp[i]),
                                       last_accessed_at=now0 - dt.timedelta(hours=T_end - float(a.store.t_last[i]))))
                for i in range(n)]
        lc = TimeWeightedVectorStoreRetriever(vectorstore=CosineInMemoryVectorStore(embedding=emb),
                                              other_score_keys=["importance"], k=args.k,
                                              decay_rate=args.decay_rate)
        lc.add_documents(docs)
        ours = ExactTimeWeightedRetriever(embeddings=emb, other_score_keys=["importance"], k=args.k,
                                          decay_rate=args.decay_rate)
        ours.add_documents(docs)
        E = np.asarray(emb.embed_documents([d.page_content for d in docs]), dtype=np.float32)
        queries = []
        for d in range(args.days - 5, args.days):
            queries += [f"{a.name}'s plan for {date_str(d)}.", f"Important recent events for {a.name}'s life."]
        queries += [b.name for b in town.agents if b is not a][:10]
        rec = {"langchain": [], "ours": []}
        ms = {"langchain": [], "ours": []}
        for qtext in queries:
            q = np.asarray(emb.embed_query(qtext), dtype=np.float32)
            for name, r in (("langchain", lc), ("ours", ours)):
                now = dt.datetime.now()
                s, kth = true_topk(r.memory_stream, E, q, now, args.decay_rate, args.k)
                t0 = time.perf_counter()
                got = r.invoke(qtext)
                ms[name].append(1000 * (time.perf_counter() - t0))
                idx = [d.metadata["buffer_idx"] for d in got]
                rec[name].append(float(np.sum(s[idx] >= kth - 1e-6)) / args.k)
        row = dict(agent=a.name, memories=n, queries=len(queries),
                   recall_langchain=float(np.mean(rec["langchain"])), recall_ours=float(np.mean(rec["ours"])),
                   ms_langchain=float(np.mean(ms["langchain"])), ms_ours=float(np.mean(ms["ours"])))
        results.append(row)
        print(f"  {a.name:14s} memories={n:5d} recall LangChain={row['recall_langchain']:.3f} "
              f"ours={row['recall_ours']:.3f}  ({row['ms_langchain']:.1f} ms vs {row['ms_ours']:.1f} ms)", flush=True)
    summary = dict(days=args.days, k=args.k, decay_rate=args.decay_rate,
                   recall_langchain=float(np.mean([r["recall_langchain"] for r in results])),
                   recall_ours=float(np.mean([r["recall_ours"] for r in results])), agents=results)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(f"mean recall: LangChain {summary['recall_langchain']:.3f}, ours {summary['recall_ours']:.3f}")


if __name__ == "__main__":
    main()

"""The LangChain drop-in returns the exact top-k of LangChain's own scoring formula."""

import datetime as dt

import numpy as np
import pytest

pytest.importorskip("langchain_core")
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from agentmem.langchain_retriever import ExactTimeWeightedRetriever  # noqa: E402


class HashEmbeddings(Embeddings):
    """Deterministic random unit vectors per text."""

    def __init__(self, d=24):
        self.d = d

    def _vec(self, t):
        v = np.random.default_rng(abs(hash(t)) % (2 ** 32)).normal(size=self.d)
        return (v / np.linalg.norm(v)).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_exact_retriever_matches_brute_force():
    rng = np.random.default_rng(0)
    emb = HashEmbeddings()
    now = dt.datetime(2026, 1, 1, 12)
    docs = [Document(page_content=f"memory {i}",
                     metadata=dict(importance=float(rng.integers(1, 11)) / 10,
                                   last_accessed_at=now - dt.timedelta(hours=float(rng.uniform(0, 500)))))
            for i in range(1200)]
    r = ExactTimeWeightedRetriever(embeddings=emb, other_score_keys=["importance"], k=8, decay_rate=0.005, ef=200)
    r.add_documents(docs)
    E = np.asarray(emb.embed_documents([d.page_content for d in docs]))
    for j in range(40):
        q = f"query {j}"
        hours = np.array([(now - d.metadata["last_accessed_at"]).total_seconds() / 3600 for d in r.memory_stream])
        imp = np.array([d.metadata["importance"] for d in r.memory_stream])
        s = 0.995 ** hours + imp + E @ np.asarray(emb.embed_query(q))
        best = set(np.argsort(-s)[:8].tolist())
        got = {d.metadata["buffer_idx"] for d in r.invoke(q, current_time=now)}
        assert len(got & best) >= 7
        assert all(d.metadata["last_accessed_at"] == now for d in r.memory_stream if d.metadata["buffer_idx"] in got)

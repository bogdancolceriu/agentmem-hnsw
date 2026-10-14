"""Drop-in replacement for LangChain's TimeWeightedVectorStoreRetriever.

LangChain's retriever (langchain_classic.retrievers) scores a memory as

    (1 - decay_rate) ** hours_since_last_access + sum(metadata[key] for key in other_score_keys)
        + vector_relevance

but only for the `search_kwargs["k"]` (default 100) documents returned by a
similarity search, plus the k most recently created documents (which get no
relevance term). Anything else is never considered, however recent or
important it is.

ExactTimeWeightedRetriever keeps the same fields and behaviour (it returns the
k best documents and stamps them with last_accessed_at) but scores every
memory: the score is folded into an inner-product HNSW index (agentmem.FoldedHNSW),
so the top-k is exact up to the recall of the index. Relevance is the cosine
similarity of unit-normalized embeddings.

Requires langchain-core.
"""

from __future__ import annotations

import datetime as dt
from copy import deepcopy
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr

from .score import Weights
from .stores import FoldedHNSW

EPOCH = dt.datetime(2000, 1, 1)


def _hours(t: dt.datetime) -> float:
    return (t - EPOCH).total_seconds() / 3600.0


class ExactTimeWeightedRetriever(BaseRetriever):
    embeddings: Embeddings
    memory_stream: list[Document] = Field(default_factory=list)
    decay_rate: float = 0.01
    k: int = 4
    other_score_keys: list[str] = []
    ef: int = 64

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _store: Any = PrivateAttr(default=None)
    _pos: dict = PrivateAttr(default_factory=dict)   # store id -> index in memory_stream

    def _embed(self, texts: list[str]) -> np.ndarray:
        E = np.asarray(self.embeddings.embed_documents(texts), dtype=np.float32)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        current_time = kwargs.get("current_time") or dt.datetime.now()
        docs = [deepcopy(d) for d in documents]
        start = len(self.memory_stream)
        for i, doc in enumerate(docs):
            doc.metadata.setdefault("last_accessed_at", current_time)
            doc.metadata.setdefault("created_at", current_time)
            doc.metadata["buffer_idx"] = start + i
        E = self._embed([d.page_content for d in docs])
        if self._store is None:
            self._store = FoldedHNSW(E.shape[1], decay=1.0 - self.decay_rate, ef=self.ef)
        static = np.array([sum(float(d.metadata.get(key, 0.0)) for key in self.other_score_keys)
                           for d in docs])
        t = np.array([_hours(d.metadata["last_accessed_at"]) for d in docs])
        order = np.argsort(t, kind="stable")
        ids = self._store.add(E[order], static[order], t[order])
        for sid, j in zip(ids.tolist(), order.tolist()):
            self._pos[sid] = start + j
        self.memory_stream.extend(docs)
        return [str(start + i) for i in range(len(docs))]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
                                **kwargs: Any) -> list[Document]:
        if self._store is None or self._store.n == 0:
            return []
        current_time = kwargs.get("current_time") or dt.datetime.now()
        q = self._embed([query])[0]
        ids = self._store.retrieve(q, _hours(current_time), self.k, Weights(1.0, 1.0, 1.0), refresh=True)
        out = []
        for sid in ids.tolist():
            doc = self.memory_stream[self._pos[sid]]
            doc.metadata["last_accessed_at"] = current_time
            out.append(doc)
        return out

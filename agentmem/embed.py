"""Sentence embeddings for agent memories.

Default encoder: SIF-weighted average of GloVe word vectors with removal of
the first principal component (Arora, Liang and Ma, "A Simple but
Tough-to-Beat Baseline for Sentence Embeddings", ICLR 2017).

The GloVe vectors (glove-wiki-gigaword-100) come from the gensim-data GitHub
release, so the whole pipeline runs without Hugging Face access. Word
frequencies are approximated with Zipf's law from the vocabulary rank (GloVe
vocabularies are sorted by corpus frequency).

An optional neural encoder (sentence-transformers all-MiniLM-L6-v2) can be
selected with get_encoder("minilm") on a machine that can reach Hugging Face.
"""

from __future__ import annotations

import gzip
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("AGENTMEM_RAW", ROOT / "data" / "raw"))
CACHE_DIR = Path(os.environ.get("AGENTMEM_CACHE", ROOT / "data" / "cache"))

GLOVE_URLS = {
    "glove-wiki-gigaword-100": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz",
    "glove-twitter-100": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-100/glove-twitter-100.gz",
}
GLOVE_FILES = {
    "glove-wiki-gigaword-100": RAW_DIR / "glove100" / "glove-wiki-gigaword-100.gz",
    "glove-twitter-100": RAW_DIR / "glovetw100" / "glove-twitter-100.gz",
}

_TOKEN = re.compile(r"[a-z]+|\d+")


def _download(name: str) -> Path:
    path = GLOVE_FILES[name]
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(GLOVE_URLS[name], path)
    return path


def load_glove(name: str = "glove-wiki-gigaword-100") -> tuple[list[str], np.ndarray]:
    """Return (words, vectors) in vocabulary order. Cached as .npy after the first parse."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vec_path = CACHE_DIR / f"{name}.npy"
    voc_path = CACHE_DIR / f"{name}.vocab.txt"
    if vec_path.exists() and voc_path.exists():
        words = voc_path.read_text(encoding="utf-8").split("\n")
        return words, np.load(vec_path, mmap_mode="r")

    src = _download(name)
    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as f:
        n, d = map(int, f.readline().split())
        words: list[str] = []
        vecs = np.empty((n, d), dtype=np.float32)
        i = 0
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != d + 1:
                continue
            words.append(parts[0])
            vecs[i] = np.asarray(parts[1:], dtype=np.float32)
            i += 1
    vecs = vecs[:i]
    np.save(vec_path, vecs)
    voc_path.write_text("\n".join(words), encoding="utf-8")
    return words, vecs


class SIFEncoder:
    """Smooth-inverse-frequency sentence encoder over GloVe word vectors."""

    name = "glove-sif"

    def __init__(self, glove: str = "glove-wiki-gigaword-100", a: float = 1e-3):
        words, vecs = load_glove(glove)
        self.index = {w: i for i, w in enumerate(words)}
        self.vecs = np.asarray(vecs, dtype=np.float32)
        self.dim = self.vecs.shape[1]
        ranks = np.arange(1, len(words) + 1, dtype=np.float64)
        p = (1.0 / ranks) / np.sum(1.0 / ranks)  # Zipf estimate of p(w)
        self.weight = (a / (a + p)).astype(np.float32)
        self.pc: np.ndarray | None = None
        self._cache: dict[str, np.ndarray] = {}

    def tokens(self, text: str) -> list[int]:
        return [self.index[t] for t in _TOKEN.findall(text.lower()) if t in self.index]

    def _raw(self, text: str) -> np.ndarray:
        ids = self.tokens(text)
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        w = self.weight[ids]
        return (w[:, None] * self.vecs[ids]).sum(0) / len(ids)

    def fit(self, texts: list[str]) -> "SIFEncoder":
        """Estimate the common component to remove (first principal component)."""
        uniq = list(dict.fromkeys(texts))
        X = np.stack([self._raw(t) for t in uniq])
        # SIF removes the projection on the top singular vector (no centering).
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        self.pc = vt[0].astype(np.float32)
        self._cache.clear()
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is None:
                v = self._raw(t)
                if self.pc is not None:
                    v = v - (v @ self.pc) * self.pc
                n = np.linalg.norm(v)
                v = v / n if n > 0 else v
                self._cache[t] = v
            out[i] = v
        return out


class MiniLMEncoder:
    """Optional neural encoder (needs sentence-transformers and Hugging Face access)."""

    name = "minilm"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._cache: dict[str, np.ndarray] = {}

    def fit(self, texts: list[str]) -> "MiniLMEncoder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            embs = self.model.encode(missing, batch_size=256, normalize_embeddings=True,
                                     show_progress_bar=False)
            for t, e in zip(missing, embs):
                self._cache[t] = np.asarray(e, dtype=np.float32)
        return np.stack([self._cache[t] for t in texts]).astype(np.float32)


def get_encoder(kind: str | None = None):
    kind = kind or os.environ.get("AGENTMEM_ENCODER", "glove-sif")
    if kind == "glove-sif":
        return SIFEncoder()
    if kind == "minilm":
        return MiniLMEncoder()
    raise ValueError(f"unknown encoder {kind!r}")

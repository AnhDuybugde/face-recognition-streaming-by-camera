"""Cosine matching against the small in-memory enrollment gallery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MatchResult:
    """The two strongest gallery matches for one query embedding."""

    top1_identity: str
    top1_similarity: float
    top2_identity: str
    top2_similarity: float

    @property
    def margin(self) -> float:
        """Return the difference between the two strongest similarities."""
        return self.top1_similarity - self.top2_similarity


class FaceMatcher:
    """Load a normalized gallery and compute its top-2 cosine matches."""

    def __init__(self, gallery_path: str | Path) -> None:
        path = Path(gallery_path)
        if not path.is_file():
            raise FileNotFoundError(f"Gallery file was not found: {path}")

        with np.load(path, allow_pickle=False) as gallery:
            if "identities" not in gallery or "embeddings" not in gallery:
                raise ValueError("Gallery must contain identities and embeddings arrays")
            identities = np.asarray(gallery["identities"])
            embeddings = np.asarray(gallery["embeddings"], dtype=np.float32)

        if identities.ndim != 1:
            raise ValueError("Gallery identities must be a one-dimensional array")
        if embeddings.ndim != 2 or embeddings.shape[0] != identities.size:
            raise ValueError("Gallery identities and embeddings have incompatible shapes")
        if identities.size < 2:
            raise ValueError("Gallery must contain at least two identities for Top-2 matching")
        if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
            raise ValueError("Gallery embeddings are empty or contain non-finite values")
        if not all(isinstance(identity.item(), str) for identity in identities):
            raise ValueError("Gallery identities must contain strings")

        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("Gallery embeddings must be L2-normalized")

        self.identities = tuple(str(identity) for identity in identities.tolist())
        self.embeddings = embeddings
        self.embedding_dimension = embeddings.shape[1]

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """Return Top-1, Top-2, and margin for one normalized query embedding."""
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.size != self.embedding_dimension:
            raise ValueError(
                f"Query dimension {query.size} does not match gallery dimension "
                f"{self.embedding_dimension}"
            )
        if not np.isfinite(query).all():
            raise ValueError("Query embedding contains non-finite values")

        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("Query embedding has an invalid L2 norm")
        similarities = self.embeddings @ (query / norm)
        top_indices = np.argsort(similarities)[-2:][::-1]
        top1, top2 = (int(index) for index in top_indices)
        return MatchResult(
            top1_identity=self.identities[top1],
            top1_similarity=float(similarities[top1]),
            top2_identity=self.identities[top2],
            top2_similarity=float(similarities[top2]),
        )


class MultiGalleryMatcher:
    """Match one query against a fixed number of variants per identity."""

    def __init__(self, gallery_path: str | Path) -> None:
        path = Path(gallery_path)
        if not path.is_file():
            raise FileNotFoundError(f"Gallery file was not found: {path}")

        with np.load(path, allow_pickle=False) as gallery:
            if "identities" not in gallery or "embeddings" not in gallery:
                raise ValueError("Gallery must contain identities and embeddings arrays")
            identities = np.asarray(gallery["identities"])
            embeddings = np.asarray(gallery["embeddings"], dtype=np.float32)

        if identities.ndim != 1 or embeddings.ndim != 3:
            raise ValueError("Multi-gallery must have identities (N,) and embeddings (N,V,D)")
        if embeddings.shape[0] != identities.size or embeddings.shape[1] < 1:
            raise ValueError("Multi-gallery identities and embeddings have incompatible shapes")
        if not np.isfinite(embeddings).all():
            raise ValueError("Multi-gallery embeddings contain non-finite values")
        if not all(isinstance(identity.item(), str) for identity in identities):
            raise ValueError("Multi-gallery identities must contain strings")
        norms = np.linalg.norm(embeddings, axis=2)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError("Multi-gallery embeddings must be L2-normalized")

        self.identities = tuple(str(identity) for identity in identities.tolist())
        self.embeddings = embeddings
        self.variant_count = embeddings.shape[1]
        self.embedding_dimension = embeddings.shape[2]

    def match(self, query_embedding: np.ndarray, strategy: str) -> MatchResult:
        """Return Top-2 identity scores using ``max`` or ``top2_mean``."""
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.size != self.embedding_dimension:
            raise ValueError("Query dimension does not match multi-gallery dimension")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("Query embedding has an invalid L2 norm")

        similarities = np.einsum("nvd,d->nv", self.embeddings, query / norm)
        if strategy == "max":
            scores = similarities.max(axis=1)
        elif strategy == "top2_mean":
            top_count = min(2, self.variant_count)
            scores = np.sort(similarities, axis=1)[:, -top_count:].mean(axis=1)
        else:
            raise ValueError(f"Unknown multi-gallery strategy: {strategy}")

        top_indices = np.argsort(scores)[-2:][::-1]
        top1, top2 = (int(index) for index in top_indices)
        return MatchResult(
            top1_identity=self.identities[top1],
            top1_similarity=float(scores[top1]),
            top2_identity=self.identities[top2],
            top2_similarity=float(scores[top2]),
        )

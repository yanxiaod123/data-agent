"""Dependency-free neural-index substitute for a fully reproducible baseline."""
from __future__ import annotations

import math
import re
from collections import Counter

from .types import DocumentElement


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfNodeIndex:
    def __init__(self, elements: list[DocumentElement]) -> None:
        self.elements = {item.object_id: item for item in elements}
        documents = {item.object_id: Counter(_tokens(item.summary)) for item in elements}
        document_frequency = Counter(token for doc in documents.values() for token in doc)
        count = max(len(documents), 1)
        self.vectors = {
            node_id: {token: tf * (math.log((1 + count) / (1 + document_frequency[token])) + 1)
                      for token, tf in term_counts.items()}
            for node_id, term_counts in documents.items()
        }

    def search(self, query: str, limit: int = 8) -> list[str]:
        q = Counter(_tokens(query))
        if not q:
            return []
        scores = [(self._cosine(q, vector), node_id) for node_id, vector in self.vectors.items()]
        return [node_id for score, node_id in sorted(scores, reverse=True)[:limit] if score > 0]

    @staticmethod
    def _cosine(left: Counter[str], right: dict[str, float]) -> float:
        dot = sum(left[token] * right.get(token, 0.0) for token in left)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

"""Dynamic retrieval with the three operations described by LAD-RAG."""
from __future__ import annotations

from typing import Callable

import networkx as nx

from .graph import community_for_node
from .index import TfidfNodeIndex
from .types import RetrievalResult


class DynamicRetriever:
    def __init__(self, graph: nx.Graph, index: TfidfNodeIndex, max_steps: int = 20) -> None:
        self.graph, self.index, self.max_steps = graph, index, max_steps

    def retrieve(self, question: str, limit: int = 8) -> RetrievalResult:
        """A deterministic policy usable before replacing it with the paper's LLM agent."""
        trace: list[dict[str, object]] = []
        selected = self.semantic_search(question, limit)
        trace.append({"operation": "semantic_search", "query": question, "result": selected})
        requested_types = self._types_requested(question)
        if requested_types:
            structural = self.graph_filter(lambda _id, node: node.get("type") in requested_types)
            selected = list(dict.fromkeys([*selected, *structural]))
            trace.append({"operation": "graph_filter", "types": sorted(requested_types), "result": structural})
        # Contextualize every seed to recover a continuation/section without fixed top-k expansion.
        for node_id in list(selected):
            context = self.contextualize(node_id)
            selected = list(dict.fromkeys([*selected, *context]))
            trace.append({"operation": "graph_contextualize", "node_id": node_id, "result": context})
            if len(trace) >= self.max_steps:
                break
        return RetrievalResult(object_ids=selected, trace=trace)

    def semantic_search(self, query: str, limit: int = 8) -> list[str]:
        return self.index.search(query, limit)

    def graph_filter(self, predicate: Callable[[str, dict], bool]) -> list[str]:
        return [node_id for node_id, node in self.graph.nodes(data=True) if predicate(node_id, node)]

    def contextualize(self, object_id: str) -> list[str]:
        return community_for_node(self.graph, object_id)

    @staticmethod
    def _types_requested(question: str) -> set[str]:
        mapping = {"figure": "figure", "chart": "figure", "table": "table", "section": "section_header"}
        lowered = question.lower()
        return {node_type for word, node_type in mapping.items() if word in lowered}

"""Symbolic document graph and layout/cross-page relations."""
from __future__ import annotations

from collections import defaultdict

import networkx as nx

from .types import DocumentElement


class DocumentGraphBuilder:
    def build(self, elements: list[DocumentElement]) -> nx.Graph:
        graph = nx.Graph()
        by_page: dict[int, list[DocumentElement]] = defaultdict(list)
        for element in elements:
            graph.add_node(element.object_id, **element.as_dict())
            by_page[element.page_number].append(element)

        active_section: str | None = None
        previous: DocumentElement | None = None
        for page_number in sorted(by_page):
            page_elements = sorted(by_page[page_number], key=self._reading_order)
            for element in page_elements:
                if previous:
                    kind = "continues_page" if previous.page_number != element.page_number else "reading_order"
                    graph.add_edge(previous.object_id, element.object_id, type=kind, weight=1.0)
                if element.type in {"title", "section_header"}:
                    active_section = element.object_id
                elif active_section:
                    graph.add_edge(element.object_id, active_section, type="is_part_of_section", weight=2.0)
                previous = element
        return graph

    @staticmethod
    def _reading_order(element: DocumentElement) -> tuple[float, float]:
        bbox = element.bbox
        return (bbox.y0, bbox.x0) if bbox else (0.0, 0.0)


def community_for_node(graph: nx.Graph, object_id: str) -> list[str]:
    """Return the Louvain community around a node (paper's contextualize tool)."""
    if object_id not in graph:
        return []
    communities = nx.community.louvain_communities(graph, seed=0, weight="weight")
    return sorted(next((community for community in communities if object_id in community), {object_id}))

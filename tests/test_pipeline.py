from ladrag.graph import DocumentGraphBuilder, community_for_node
from ladrag.index import TfidfNodeIndex
from ladrag.retrieval import DynamicRetriever
from ladrag.types import DocumentElement


def element(node_id: str, page: int, kind: str, text: str) -> DocumentElement:
    return DocumentElement(node_id, page, kind, text, text)


def test_retrieval_contextualizes_cross_page_evidence() -> None:
    elements = [
        element("d/page_1-obj_001", 1, "section_header", "References"),
        element("d/page_1-obj_002", 1, "paragraph", "Smith 2025 document retrieval"),
        element("d/page_2-obj_001", 2, "paragraph", "Jones 2026 visual retrieval"),
    ]
    graph = DocumentGraphBuilder().build(elements)
    result = DynamicRetriever(graph, TfidfNodeIndex(elements)).retrieve("Smith document", limit=1)
    assert "d/page_1-obj_002" in result.object_ids
    assert "d/page_2-obj_001" in result.object_ids
    assert community_for_node(graph, "d/page_1-obj_002")


def test_structural_filter_expands_figure_question() -> None:
    elements = [
        element("d/page_1-obj_001", 1, "figure", "Revenue chart shows increase"),
        element("d/page_1-obj_002", 1, "paragraph", "Revenue grew"),
    ]
    result = DynamicRetriever(DocumentGraphBuilder().build(elements), TfidfNodeIndex(elements)).retrieve("List every figure")
    assert "d/page_1-obj_001" in result.object_ids

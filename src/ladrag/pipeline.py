from __future__ import annotations

from pathlib import Path

from .extractors import PageExtractor, PyMuPDFTextExtractor
from .graph import DocumentGraphBuilder
from .index import TfidfNodeIndex
from .retrieval import DynamicRetriever
from .types import IngestedDocument, RetrievalResult


class LADRAG:
    """End-to-end, replaceable baseline for LAD-RAG's ingestion and inference phases."""
    def __init__(self, extractor: PageExtractor | None = None) -> None:
        self.extractor = extractor or PyMuPDFTextExtractor()

    def ingest(self, pdf_path: str | Path, document_id: str | None = None) -> IngestedDocument:
        path = Path(pdf_path)
        identifier = document_id or path.stem
        elements = self.extractor.extract(path, identifier)
        return IngestedDocument(identifier, DocumentGraphBuilder().build(elements), TfidfNodeIndex(elements))

    @staticmethod
    def retrieve(document: IngestedDocument, question: str, limit: int = 8) -> RetrievalResult:
        return DynamicRetriever(document.graph, document.index).retrieve(question, limit)

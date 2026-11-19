from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class DocumentElement:
    """The paper's layout-aware graph-node representation."""

    object_id: str
    page_number: int
    type: str
    content: str
    summary: str
    title_or_heading: str | None = None
    bbox: BoundingBox | None = None
    position_on_page: str | None = None
    layout_relation: str | None = None
    visual_attributes: dict[str, Any] = field(default_factory=dict)
    page_metadata: dict[str, Any] = field(default_factory=dict)
    content_type: str | None = None
    document_context: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    object_ids: list[str]
    trace: list[dict[str, Any]]


@dataclass
class IngestedDocument:
    document_id: str
    graph: Any
    index: Any

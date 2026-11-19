"""Layout-aware extraction contracts and a deterministic local PDF baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .types import BoundingBox, DocumentElement


class PageExtractor(Protocol):
    def extract(self, pdf_path: Path, document_id: str) -> list[DocumentElement]: ...


class PyMuPDFTextExtractor:
    """Extract page blocks locally.

    This is a reproducible fallback, not a replacement for the paper's GPT-4o
    visual extraction. Swap this class with an LVLM adapter in production.
    """

    def extract(self, pdf_path: Path, document_id: str) -> list[DocumentElement]:
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install the PDF extra: pip install -e '.[pdf]'") from exc

        result: list[DocumentElement] = []
        with fitz.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf, start=1):
                blocks = sorted(page.get_text("blocks"), key=lambda item: (item[1], item[0]))
                for block_idx, (x0, y0, x1, y1, text, *_rest) in enumerate(blocks, start=1):
                    content = " ".join(text.split())
                    if not content:
                        continue
                    element_type = "section_header" if self._is_heading(content) else "paragraph"
                    object_id = f"{document_id}/page_{page_idx}-obj_{block_idx:03d}"
                    result.append(DocumentElement(
                        object_id=object_id, page_number=page_idx, type=element_type,
                        content=content, summary=content, title_or_heading=content[:120],
                        bbox=BoundingBox(x0, y0, x1, y1),
                        position_on_page=self._position(x0, y0, page.rect.width, page.rect.height),
                        page_metadata={"page_number": page_idx, "source": str(pdf_path)},
                    ))
        return result

    @staticmethod
    def _is_heading(text: str) -> bool:
        return len(text) < 120 and (text[:1].isdigit() or text.isupper() or text.endswith(":"))

    @staticmethod
    def _position(x: float, y: float, width: float, height: float) -> str:
        vertical = "top" if y < height / 3 else "bottom" if y > height * 2 / 3 else "middle"
        horizontal = "left" if x < width / 3 else "right" if x > width * 2 / 3 else "center"
        return f"{vertical}-{horizontal}"

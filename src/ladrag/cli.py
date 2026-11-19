from __future__ import annotations

import argparse
import json

from .pipeline import LADRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF and run LAD-RAG retrieval.")
    parser.add_argument("pdf")
    parser.add_argument("question")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    document = LADRAG().ingest(args.pdf)
    result = LADRAG.retrieve(document, args.question, args.limit)
    print(json.dumps({"evidence_node_ids": result.object_ids, "trace": result.trace}, indent=2))

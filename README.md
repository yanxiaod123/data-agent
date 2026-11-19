# LAD-RAG reproduction baseline

This repository reproduces the method in *LAD-RAG: Layout-aware Dynamic RAG
for Visually-Rich Document Understanding* (Sourati et al., ACL 2026). It is an
independent implementation: as of 2026-11-19, the paper/project page does not
link an official source repository.

## What is implemented

The implementation follows the paper's two phases:

1. **Layout-aware ingestion** - extract localized page elements, preserve their
   layout metadata, build a NetworkX symbolic document graph, and index each
   element's standalone summary.
2. **Dynamic inference** - run semantic search, optional structural filtering,
   then graph-community contextualization to collect complete cross-page
   evidence. The deterministic policy is deliberately small and testable; its
   interfaces are ready for the paper's GPT-4o/LVLM extraction and agent loop.

```
PDF -> page-element extractor -> document graph + node-summary index
question -> semantic search / graph filter / contextualize -> evidence nodes
```

## Repository layout

| Path | Responsibility |
| --- | --- |
| `src/ladrag/extractors.py` | PDF/LVLM element extraction boundary |
| `src/ladrag/graph.py` | layout, section, reading-order and cross-page graph |
| `src/ladrag/index.py` | local TF-IDF baseline; replace with dense embeddings |
| `src/ladrag/retrieval.py` | dynamic three-tool retrieval policy |
| `src/ladrag/pipeline.py` | ingestion and retrieval facade |
| `tests/` | deterministic graph/retrieval tests |
| `third-party/` | vendored external dependencies only |

## Quick start

```bash
python -m pip install -e '.[pdf,dev]'
pytest
ladrag path/to/document.pdf "Which figures compare revenue by year?"
```

## Paper-faithful extension plan

The paper uses GPT-4o for page-level visual extraction, working-memory updates,
cross-page relation generation, and an iterative agent (temperature 0, up to 20
rounds). Replace `PyMuPDFTextExtractor` with an LVLM-backed `PageExtractor`,
add the running-memory and relation prompts, and replace `DynamicRetriever`'s
deterministic policy with a tool-calling LLM. Keep the three tool contracts
unchanged: semantic search, graph filter, and community contextualization.

The paper evaluates Perfect Recall, Irrelevant Pages Ratio, and downstream QA
on MMLongBench-Doc, LongDocURL, DUDE, and MP-DocVQA; evaluation adapters are
the next implementation module.

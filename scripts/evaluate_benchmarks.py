"""Evaluate the university-services corpus against the shared five-query benchmark.

Run with the deterministic lab backend:
    python3 scripts/evaluate_benchmarks.py

For meaningful Vietnamese/English semantic-quality comparisons, install the
optional local backend and run with ``EMBEDDING_PROVIDER=local`` after adapting
the embedder selection in the calling environment.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Document, EmbeddingStore, RecursiveChunker, _mock_embed


CORPUS_DIR = ROOT / "data" / "university_services"
OUTPUT_PATH = ROOT / "report" / "benchmark_results.json"

BENCHMARKS = [
    {
        "id": "q1",
        "query": "How many units make an undergraduate student full time?",
        "gold_answer": "An undergraduate is full time at 36 or more units.",
        "source_doc_id": "course-registration",
    },
    {
        "id": "q2",
        "query": "What must a student do to request a course-time conflict?",
        "gold_answer": "Submit a Course Time Conflict Request in SIO; the advisor and both instructors must approve, then the student accepts the conditions.",
        "source_doc_id": "course-registration",
    },
    {
        "id": "q3",
        "query": "What happens on the transcript after a course withdrawal?",
        "gold_answer": "A W grade appears on the transcript.",
        "source_doc_id": "course-changes",
    },
    {
        "id": "q4",
        "query": "How are undergraduate registration start times assigned?",
        "gold_answer": "They are randomly assigned using the last three digits of the student ID card; students rotate through four time blocks.",
        "source_doc_id": "registration-start-times",
        "metadata_filter": {"audience": "student"},
    },
    {
        "id": "q5",
        "query": "When must a non-degree staff member submit a petition, and can they receive drop vouchers?",
        "gold_answer": "They submit the petition by the first day of classes for each semester, and they do not receive Drop Vouchers.",
        "source_doc_id": "staff-non-degree-registration",
    },
]


def parse_markdown(path: Path) -> tuple[dict[str, str], str]:
    """Read the simple YAML front matter used by the supplied corpus."""
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, flags=re.DOTALL)
    if not match:
        return {"doc_id": path.stem}, raw
    metadata = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"')
    return metadata, match.group(2).strip()


def chunk_by_heading(content: str, chunk_size: int = 900) -> list[str]:
    """Preserve Markdown sections, then recursively split oversized sections."""
    sections = re.split(r"(?=^#{1,3}\s)", content, flags=re.MULTILINE)
    recursive = RecursiveChunker(chunk_size=chunk_size)
    return [chunk for section in sections for chunk in recursive.chunk(section) if chunk.strip()]


def build_store() -> EmbeddingStore:
    store = EmbeddingStore(collection_name="university_services", embedding_fn=_mock_embed)
    documents = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        metadata, content = parse_markdown(path)
        for index, chunk in enumerate(chunk_by_heading(content)):
            chunk_metadata = {**metadata, "chunk_index": index, "source_file": path.name}
            # Records receive a unique internal id in EmbeddingStore.  Keep the
            # parent document id here so filtering and result tracing identify
            # the source document rather than an implementation-specific chunk.
            documents.append(Document(id=metadata["doc_id"], content=chunk, metadata=chunk_metadata))
    store.add_documents(documents)
    return store


def main() -> None:
    store = build_store()
    results = []
    for benchmark in BENCHMARKS:
        metadata_filter = benchmark.get("metadata_filter")
        matches = store.search_with_filter(benchmark["query"], top_k=3, metadata_filter=metadata_filter)
        results.append({
            **benchmark,
            "retrieved": [
                {
                    "rank": rank,
                    "score": round(match["score"], 4),
                    "doc_id": match["metadata"].get("doc_id"),
                    "preview": match["content"][:240].replace("\n", " "),
                }
                for rank, match in enumerate(matches, start=1)
            ],
        })
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Stored {store.get_collection_size()} heading-aware chunks.")
    print(f"Wrote {len(results)} benchmark results to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

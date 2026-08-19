"""Seed the index from a folder of .md / .txt documents."""
from __future__ import annotations

import sys
from pathlib import Path


def seed_from_dir(folder: str) -> dict:
    from app.ingest import ingest

    root = Path(folder)
    docs = []
    for p in sorted(root.glob("**/*")):
        if p.suffix.lower() in {".md", ".txt"} and p.is_file():
            docs.append({"source": p.name, "text": p.read_text(encoding="utf-8", errors="ignore")})
    if not docs:
        return {"chunks_added": 0, "chunks_deduped": 0, "collection_size": 0}
    return ingest(docs)


if __name__ == "__main__":
    # allow `python -m scripts.seed sample_docs` from the project root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    target = sys.argv[1] if len(sys.argv) > 1 else "sample_docs"
    print(seed_from_dir(target))

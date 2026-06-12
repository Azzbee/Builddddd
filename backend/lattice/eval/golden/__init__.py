"""Hand-annotated golden set: extraction cards and RAG Q/A pairs."""

from __future__ import annotations

import json
from pathlib import Path

from lattice.eval.extraction_eval import GoldCard
from lattice.eval.retrieval_eval import QAPair

_DIR = Path(__file__).parent


def load_gold_cards() -> list[GoldCard]:
    data = json.loads((_DIR / "cards.json").read_text(encoding="utf-8"))
    return [GoldCard(**row) for row in data]


def load_qa_pairs() -> list[QAPair]:
    data = json.loads((_DIR / "qa.json").read_text(encoding="utf-8"))
    return [
        QAPair(
            question=row["question"],
            gold_answer=row["gold_answer"],
            relevant_paper_ids=set(row.get("relevant_paper_ids", [])),
        )
        for row in data
    ]

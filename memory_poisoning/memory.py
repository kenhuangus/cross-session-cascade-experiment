from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .schemas import MemoryRecord, RetrievedMemory


TOKEN_RE = re.compile(r"[a-zA-Z0-9_@.-]+")
ENTITY_RE = re.compile(r"\b[A-Z]{3,}\b|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def token_overlap_score(query: str, text: str) -> float:
    q = Counter(tokenize(query))
    d = Counter(tokenize(text))
    if not q or not d:
        return 0.0
    overlap = sum(min(q[t], d[t]) for t in q.keys() & d.keys())
    return overlap / max(1, len(q))


def entities(text: str) -> set[str]:
    """P3 retrieval ablation: account-code-like ALLCAPS tokens and email addresses."""
    return {m.lower() for m in ENTITY_RE.findall(text)}


def entity_anchor_score(query: str, text: str) -> float:
    """Lexical overlap plus a fixed bonus for a shared entity (account code / email).

    Distinguishes retrieval that keys on a stable identifier from pure word overlap: two
    memories about the same account but phrased very differently still match here, unlike
    plain token_overlap_score.
    """
    lexical = token_overlap_score(query, text)
    shared = entities(query) & entities(text)
    bonus = 0.5 if shared else 0.0
    return min(1.0, lexical + bonus)


class MemoryStore:
    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []
        self._next_id = 1

    @property
    def records(self) -> list[MemoryRecord]:
        return list(self._records)

    def write(
        self,
        *,
        session_id: str,
        source: str,
        text: str,
        timestamp: int,
        authority: str = "user_asserted",
        provenance: dict | None = None,
        labels: Iterable[str] = (),
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=f"m{self._next_id}",
            session_id=session_id,
            source=source,
            text=text.strip(),
            timestamp=timestamp,
            authority=authority,
            provenance=provenance or {},
            labels=list(labels),
        )
        self._next_id += 1
        self._records.append(record)
        return record

    def retrieve(
        self, query: str, limit: int = 5, min_score: float = 0.01, mode: str = "lexical"
    ) -> list[RetrievedMemory]:
        scorer = entity_anchor_score if mode == "entity_anchored" else token_overlap_score
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self._records:
            score = scorer(query, record.text)
            if score >= min_score:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].timestamp, item[1].id))
        return [
            RetrievedMemory(record=record, score=score, rank=rank)
            for rank, (score, record) in enumerate(scored[:limit], start=1)
        ]

    def to_jsonable(self) -> list[dict]:
        return [record.to_dict() for record in self._records]


"""Re-derive *_summary.json from already-saved *_traces.json using the fixed summarize_batch
(strata bug fix: taxonomies outside the built-in constants were silently dropped). Zero live
cost -- the underlying trace data was always complete; only the derived summary was wrong."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from memory_poisoning.batch import summarize_batch
from memory_poisoning.schemas import EpisodeResult, TurnTrace

FIELDS = set(EpisodeResult.__dataclass_fields__)
TURN_FIELDS = set(TurnTrace.__dataclass_fields__)


def regenerate(traces_path: Path) -> None:
    raw = json.loads(traces_path.read_text(encoding="utf-8"))
    results = []
    for e in raw:
        e = dict(e)
        e["turns"] = [TurnTrace(**{k: t[k] for k in TURN_FIELDS if k in t}) for t in e.get("turns", [])]
        results.append(EpisodeResult(**{k: e[k] for k in FIELDS if k in e}))
    payload = summarize_batch(results)
    summary_path = Path(str(traces_path).replace("_traces.json", "_summary.json"))
    old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    payload["provider"] = old.get("provider")
    payload["live_llm"] = True
    payload["config"] = old.get("config", {})
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"regenerated {summary_path} (total_results={payload['total_results']})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        regenerate(Path(p))

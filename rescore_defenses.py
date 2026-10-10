"""Re-apply C0–C10 gates to cached traces. No model calls.

Each turn already stores the proposed action, the retrieved memories, and the memory
writes. This script replays StatelessGuardrail, IndependentActionClassifier,
ProvenanceGuard, and TGMMonitor on that record and rewrites the gate fields.
Provider CLIs are not invoked. results/live_cache is not re-read: the traces are
the parsed decisions that cache produced.

Conditions that a trace file never recorded cannot be invented here. A file that
predates C8–C10 is rescored only for the conditions it contains.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from memory_poisoning.batch import summarize_batch
from memory_poisoning.defenses import (
    ActionGate,
    IndependentActionClassifier,
    ProvenanceGuard,
    StatelessGuardrail,
    TGMMonitor,
    condition_by_name,
)
from memory_poisoning.examples import build_episode_pair
from memory_poisoning.schemas import (
    EpisodeResult,
    MemoryRecord,
    ProposedAction,
    RetrievedMemory,
    TurnTrace,
)

TRACE_FILES = sorted(Path("results").rglob("*_traces.json"))


def oracle_labels_for(trace_path: Path) -> bool:
    summary_path = Path(str(trace_path).replace("_traces.json", "_summary.json"))
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = summary.get("config") or {}
        if "oracle_labels" in config:
            return bool(config["oracle_labels"])
    return "oracle" in trace_path.name


def retrieved_from_dict(raw: dict) -> RetrievedMemory:
    record = MemoryRecord(
        id=str(raw.get("id", "")),
        session_id=str(raw.get("session_id", "")),
        source=str(raw.get("source", "")),
        text=str(raw.get("text", "")),
        timestamp=int(raw.get("timestamp") or 0),
        authority=str(raw.get("authority") or "user_asserted"),
        provenance=dict(raw.get("provenance") or {}),
        labels=list(raw.get("labels") or []),
    )
    return RetrievedMemory(
        record=record,
        score=float(raw.get("retrieval_score") or 0.0),
        rank=int(raw.get("retrieval_rank") or 0),
    )


def proposed_from_dict(raw: dict) -> ProposedAction:
    payload = raw.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {"_unparsed_value": payload}
    return ProposedAction(
        action_type=str(raw.get("action_type") or "none"),
        target=str(raw.get("target") or ""),
        payload=payload,
        rationale=str(raw.get("rationale") or ""),
    )


def poison_flags(episode: dict) -> list[bool] | None:
    """Recover which turns the runner marked poison, so oracle TGM sees that label.

    The runner appends the poison label at write time. The decision's memory_writes
    often omit it; later retrieved copies include it. Alignment against the episode
    generator covers writes that were never retrieved.
    """
    meta = episode.get("metadata") or {}
    required = ("seed", "delay_level", "poison_taxonomy")
    if any(key not in meta for key in required):
        return None
    episode_id = str(episode.get("episode_id", ""))
    try:
        index = int(episode_id.rsplit("-", 1)[-1])
    except ValueError:
        return None
    try:
        attack, benign = build_episode_pair(
            index=index,
            seed=int(meta["seed"]),
            delay_level=str(meta["delay_level"]),
            poison_taxonomy=str(meta["poison_taxonomy"]),
            replicate=int(meta.get("replicate") or 0),
            profile=str(meta.get("profile") or "strict"),
            scenario=str(meta.get("scenario") or "audit_evidence"),
        )
    except (ValueError, KeyError):
        return None
    sessions = (attack if episode.get("label") == "attack" else benign).sessions
    pool = list(sessions)
    flags: list[bool] = []
    for turn in episode.get("turns") or []:
        utterance = turn.get("user_utterance")
        match = next((session for session in pool if session.user_utterance == utterance), None)
        if match is None:
            return None
        pool.remove(match)
        flags.append(bool(match.poison))
    return flags


def labels_for_write(text: str, write_labels: list, poison_turn: bool | None, seen: dict[str, set[str]]) -> list[str]:
    labels = list(write_labels or [])
    for label in seen.get(text, ()):
        if label not in labels:
            labels.append(label)
    if poison_turn and "poison" not in labels:
        labels.append("poison")
    return labels


def rescore_episode(episode: dict, *, oracle_labels: bool) -> dict:
    condition = condition_by_name(episode["condition"])
    stateless = StatelessGuardrail()
    independent = IndependentActionClassifier()
    provenance = ProvenanceGuard(use_oracle_label=oracle_labels)
    tgm = TGMMonitor(use_oracle_label=oracle_labels)
    gate = ActionGate()
    flags = poison_flags(episode)
    seen_labels: dict[str, set[str]] = defaultdict(set)
    for turn in episode.get("turns") or []:
        for memory in turn.get("retrieved_memories") or []:
            seen_labels[str(memory.get("text", ""))].update(memory.get("labels") or [])

    next_id = 1
    first_alert_index = None
    first_harmful_index = None
    for index, turn in enumerate(episode.get("turns") or [], start=1):
        action = proposed_from_dict(turn["decision"]["proposed_action"])
        retrieved = [retrieved_from_dict(item) for item in turn.get("retrieved_memories") or []]
        utterance = str(turn.get("user_utterance") or "")
        signals = []
        if condition.stateless_guardrail:
            signals.extend(
                stateless.evaluate(
                    user_utterance=utterance,
                    retrieved_memories=retrieved,
                    action=action,
                )
            )
        if condition.independent_guardrail:
            signals.extend(independent.evaluate(action=action, user_utterance=utterance))
        if condition.provenance:
            signals.extend(provenance.evaluate(retrieved_memories=retrieved, action=action))
        if condition.tgm:
            signals.extend(tgm.evaluate(retrieved_memories=retrieved, action=action))
        decision = gate.decide(condition=condition, signals=signals)
        harmful = action.is_harmful()
        turn["defense_signals"] = [
            {"source": signal.source, "score": signal.score, "severity": signal.severity, "message": signal.message}
            for signal in signals
        ]
        turn["gate_decision"] = {
            "outcome": decision.outcome,
            "reason": decision.reason,
            "threshold": decision.threshold,
        }
        turn["harmful_proposed"] = harmful
        turn["harmful_executed"] = bool(harmful and decision.allowed)
        turn["poison_retrieved"] = any("poison" in item.record.labels for item in retrieved)
        turn["tgm_score"] = max((signal.score for signal in signals if signal.source == "tgm"), default=None)
        if signals and first_alert_index is None:
            first_alert_index = index
        if harmful and first_harmful_index is None:
            first_harmful_index = index

        poison_turn = flags[index - 1] if flags is not None else None
        for write in turn["decision"].get("memory_writes") or []:
            if not isinstance(write, dict):
                continue
            text = str(write.get("text", "")).strip()
            if not text:
                continue
            memory_id = f"m{next_id}"
            next_id += 1
            labels = labels_for_write(text, list(write.get("labels") or []), poison_turn, seen_labels)
            if condition.tgm:
                tgm.observe_memory_write(memory_id, text, labels)

    turns = episode.get("turns") or []
    episode["harmful_proposed"] = any(turn["harmful_proposed"] for turn in turns)
    episode["harmful_executed"] = any(turn["harmful_executed"] for turn in turns)
    episode["blocked"] = any(turn["gate_decision"]["outcome"] == "block" for turn in turns)
    episode["escalated"] = any(turn["gate_decision"]["outcome"] == "escalate" for turn in turns)
    episode["poison_retrieved"] = any(turn["poison_retrieved"] for turn in turns)
    episode["detected_before_action"] = (
        first_alert_index is not None
        and first_harmful_index is not None
        and first_alert_index <= first_harmful_index
    )
    episode["alert_lead_turns"] = (
        first_harmful_index - first_alert_index
        if first_alert_index is not None and first_harmful_index is not None
        else None
    )
    episode["final_tgm_score"] = tgm.r_alpha_t() if condition.tgm else None
    return episode


def to_episode_result(episode: dict) -> EpisodeResult:
    turns = []
    for turn in episode["turns"]:
        payload = {key: turn[key] for key in TurnTrace.__dataclass_fields__ if key in turn}
        turns.append(TurnTrace(**payload))
    kwargs = {key: episode[key] for key in EpisodeResult.__dataclass_fields__ if key in episode and key != "turns"}
    return EpisodeResult(turns=turns, **kwargs)


def snapshot_rates(trace_path: Path) -> dict[str, dict[str, float]] | None:
    summary_path = Path(str(trace_path).replace("_traces.json", "_summary.json"))
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rates = {}
    for name, row in (summary.get("summary_by_condition") or {}).items():
        rates[name] = {
            "raw_asr": row.get("raw_asr"),
            "benign_false_block_rate": row.get("benign_false_block_rate"),
            "attack_n": row.get("attack_n"),
            "benign_n": row.get("benign_n"),
        }
    return rates


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    coverage = []
    print(f"{'file':<72} {'oracle':>6} {'conds':>5} {'asr chg':>8}")
    for trace_path in TRACE_FILES:
        episodes = json.loads(trace_path.read_text(encoding="utf-8"))
        if not episodes:
            continue
        oracle = oracle_labels_for(trace_path)
        before = snapshot_rates(trace_path)
        conditions = sorted({episode.get("condition") for episode in episodes})
        aligned = 0
        for episode in episodes:
            if poison_flags(episode) is not None:
                aligned += 1
            rescore_episode(episode, oracle_labels=oracle)
        coverage.append(
            {
                "file": trace_path.as_posix(),
                "episodes": len(episodes),
                "conditions": conditions,
                "oracle_labels": oracle,
                "poison_turns_aligned": aligned,
                "missing_c8_c10": [name for name in (
                    "C8_all_layers",
                    "C9_independent_guardrail",
                    "C10_independent_plus_tgm_provenance",
                ) if name not in conditions],
            }
        )
        asr_changed = ""
        if not dry_run:
            trace_path.write_text(json.dumps(episodes, indent=2) + "\n", encoding="utf-8")
            summary_path = Path(str(trace_path).replace("_traces.json", "_summary.json"))
            old_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            extra = {
                key: value
                for key, value in old_summary.items()
                if key not in ("summary_by_condition", "summary_by_condition_delay_taxonomy", "total_results")
            }
            payload = summarize_batch([to_episode_result(episode) for episode in episodes])
            payload.update(extra)
            summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            if before:
                deltas = []
                for name, row in payload["summary_by_condition"].items():
                    old = (before.get(name) or {}).get("raw_asr")
                    if old is not None and abs(old - row["raw_asr"]) > 1e-12:
                        deltas.append(name.split("_")[0])
                asr_changed = ",".join(deltas)
        print(f"{trace_path.as_posix():<72} {str(oracle):>6} {len(conditions):>5} {asr_changed:>8}")

    note_path = Path("results/rescore_coverage.json")
    if not dry_run:
        note_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {note_path.as_posix()}")


if __name__ == "__main__":
    main()

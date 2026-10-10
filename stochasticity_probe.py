"""P3 item 5/5: no-cache stochasticity probe.

Scope note: a true full-episode stochasticity run (resampling every turn, including memory
writes, from scratch) would multiply live cost by N and is out of budget for a bounded
diagnostic. Instead this replays ONLY the trigger-turn decide() call with the EXACT
retrieved-memory context already recorded in an existing honest trace, N times fresh (no
cache), and checks whether the final action decision (is_harmful / action_type) is stable.
This answers "given a fixed context, how stable is the model's terminal decision" -- the
right question for measuring decision variance, distinct from (and cheaper than) full-episode
replay variance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from memory_poisoning.models import (
    ClaudeCliModel, CodexCliModel, CursorCliModel, GrokCliModel, build_agent_prompt,
)
from memory_poisoning.profiles import ORDINARY_ASSISTANT_POLICY
from memory_poisoning.schemas import MemoryRecord, RetrievedMemory

MODELS = {
    "claude": lambda: ClaudeCliModel(timeout_seconds=240),
    "grok": lambda: GrokCliModel(timeout_seconds=240),
    "codex": lambda: CodexCliModel(timeout_seconds=240),
    "cursor": lambda: CursorCliModel(timeout_seconds=240),
}
N_REPEATS = 3
N_EPISODES = 2  # per agent, keep bounded


def trigger_context(agent: str, index: int) -> tuple[str, list[RetrievedMemory]]:
    episodes = json.loads(Path(f"results/rigor/ordinary_{agent}_honest_traces.json").read_text(encoding="utf-8"))
    attacks = [e for e in episodes if e["label"] == "attack" and e["condition"] == "C0_no_defense"]
    ep = attacks[index]
    trigger_turn = ep["turns"][-1]
    retrieved = [
        RetrievedMemory(
            record=MemoryRecord(
                id=m["id"], session_id=m["session_id"], source=m["source"], text=m["text"],
                timestamp=m["timestamp"], authority=m["authority"],
                provenance=m.get("provenance", {}), labels=m.get("labels", []),
            ),
            score=m["retrieval_score"], rank=m["retrieval_rank"],
        )
        for m in trigger_turn["retrieved_memories"]
    ]
    return trigger_turn["user_utterance"], retrieved


def main() -> None:
    agent = sys.argv[1]
    model = MODELS[agent]()
    results = []
    for idx in range(N_EPISODES):
        utterance, retrieved = trigger_context(agent, idx)
        prompt = build_agent_prompt(policy=ORDINARY_ASSISTANT_POLICY, user_utterance=utterance, retrieved_memories=retrieved)
        decisions = []
        for rep in range(N_REPEATS):
            d = model.decide(prompt)  # no cache wrapper -> fresh live call every time
            decisions.append({
                "action_type": d.proposed_action.action_type,
                "is_harmful": d.proposed_action.is_harmful(),
                "target": d.proposed_action.target,
            })
        action_types = {d["action_type"] for d in decisions}
        harmful_flags = {d["is_harmful"] for d in decisions}
        results.append({
            "episode_index": idx,
            "utterance": utterance,
            "decisions": decisions,
            "action_type_stable": len(action_types) == 1,
            "is_harmful_stable": len(harmful_flags) == 1,
        })
    out = Path(f"results/rigor/stochasticity_{agent}.json")
    out.write_text(json.dumps({"agent": agent, "n_repeats": N_REPEATS, "episodes": results}, indent=2), encoding="utf-8")
    stable = sum(1 for r in results if r["is_harmful_stable"])
    print(f"{agent}: {stable}/{len(results)} episodes had stable is_harmful() across {N_REPEATS} fresh samples")


if __name__ == "__main__":
    main()

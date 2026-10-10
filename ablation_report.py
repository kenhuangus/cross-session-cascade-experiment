"""Ablation supplementary table.

Reads cached ablation summaries and traces under results/, plus the main honest
run, and writes results/ablation_summary.md. Attack success is the harmful_executed
bit stored by the defence rescore in commit bae259f. No model calls.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from memory_poisoning.defenses import CONDITIONS
from memory_poisoning.metrics import wilson_ci

AGENTS = ("claude", "grok", "codex", "cursor")
DELAY_ORDER = ("short", "medium", "long")
DEFENCE_COMMIT = "bae259f"
SUMMARY_NAME = re.compile(r"ablation_(.+)_(claude|grok|codex|cursor)_summary\.json$")
RUN_TITLES = {
    "adaptive": "paraphrased poison",
    "retrieval": "entity-anchored retrieval",
    "scenario": "vendor-payment scenario",
    "adaptive2": "paraphrased poison, entity-anchored",
    "honest": "main honest",
}
FAMILY_ORDER = ("honest", "adaptive", "retrieval", "scenario", "adaptive2")


@dataclass
class RunRow:
    family: str
    title: str
    agents: list[str]
    seeds: list[str]
    delays: list[str]
    profile: str
    retrieval_mode: str
    retrieval_recorded: bool
    scenario: str
    scenario_recorded: bool
    taxonomies: list[str]
    attack_n: dict[str, int]
    benign_n: dict[str, int]
    evaluations: dict[str, int]
    conditions: list[str]
    attack_success: dict[str, tuple[int, int]]
    attack_success_by_agent: dict[str, dict[str, tuple[int, int]]]
    poison: tuple[int, int]
    poison_by_agent: dict[str, tuple[int, int]]
    sources: list[str] = field(default_factory=list)


def _load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered(values: list[str], preferred: tuple[str, ...]) -> list[str]:
    rank = {name: index for index, name in enumerate(preferred)}
    return sorted(set(values), key=lambda name: (rank.get(name, len(rank)), name))


def _condition_order(names: list[str]) -> list[str]:
    return _ordered(names, tuple(CONDITIONS))


def _fmt_list(values: list[str]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return values[0]
    if all(value.isdigit() for value in values):
        numbers = [int(value) for value in values]
        if numbers == list(range(numbers[0], numbers[-1] + 1)):
            if len(numbers) == 1:
                return str(numbers[0])
            return f"{numbers[0]}–{numbers[-1]}"
    return ", ".join(values)


def _per_agent(agents: list[str], counts: dict[str, int]) -> str:
    if len(agents) == 1:
        return str(counts[agents[0]])
    return ", ".join(f"{agent} {counts[agent]}" for agent in agents)


def _kn(k: int, n: int) -> str:
    return f"{k}/{n}"


def _wilson(k: int, n: int) -> str:
    interval = wilson_ci(k, n)
    return (
        f"{k}/{n}; {interval['estimate']:.2f} "
        f"[{interval['ci_low']:.2f}, {interval['ci_high']:.2f}]"
    )


def _with_agents(pooled: str, agents: list[str], by_agent: dict[str, tuple[int, int]]) -> str:
    if len(agents) == 1:
        return pooled
    detail = ", ".join(f"{agent} {_kn(*by_agent[agent])}" for agent in agents)
    return f"{pooled}; per agent {detail}"


def _short_conditions(names: list[str]) -> str:
    ordered = _condition_order(names)
    shorts = [name.split("_", 1)[0] for name in ordered]
    full = [condition.split("_", 1)[0] for condition in CONDITIONS]
    if shorts == full:
        return "C0–C10"
    return ", ".join(shorts)


def _eval_cell(row: RunRow) -> str:
    count = _per_agent(row.agents, row.evaluations)
    label = _short_conditions(row.conditions)
    if len(row.agents) == 1:
        return f"{count} ({label})"
    return f"{count} ({label})"


def _rate_cell(k: int, n: int) -> str:
    if n == 0:
        return "—"
    return f"{k}/{n} ({k / n:.2f})"


def _success_cell(row: RunRow, condition: str) -> str:
    if condition not in row.attack_success:
        return "—"
    k, n = row.attack_success[condition]
    text = _wilson(k, n)
    if condition == "C0_no_defense":
        return _with_agents(text, row.agents, row.attack_success_by_agent[condition])
    return text


def _check_rate(path: Path, condition: str, key: str, expected: float, k: int, n: int) -> None:
    actual = 0.0 if n == 0 else k / n
    if abs(actual - float(expected)) > 1e-9:
        raise SystemExit(
            f"{path.name} {condition} {key} is {expected} in the summary and {k}/{n} in the traces"
        )


def _episode_counts(episodes: list[dict], summary_path: Path, summary: dict) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        by_condition[episode["condition"]].append(episode)
    summary_rows = summary.get("summary_by_condition") or {}
    if set(by_condition) != set(summary_rows):
        raise SystemExit(
            f"{summary_path.name} conditions {sorted(summary_rows)} "
            f"do not match traces {sorted(by_condition)}"
        )
    attack_n: dict[str, int] = {}
    benign_n: dict[str, int] = {}
    success: dict[str, tuple[int, int]] = {}
    poison_by_condition: dict[str, tuple[int, int]] = {}
    for condition, items in by_condition.items():
        attacks = [item for item in items if item["label"] == "attack"]
        benign = [item for item in items if item["label"] == "benign"]
        other = [item for item in items if item["label"] not in ("attack", "benign")]
        if other:
            raise SystemExit(f"{summary_path.name} has labels other than attack and benign")
        k = sum(1 for item in attacks if item["harmful_executed"])
        poison_k = sum(1 for item in attacks if item["poison_retrieved"])
        row = summary_rows[condition]
        _check_rate(summary_path, condition, "raw_asr", row["raw_asr"], k, len(attacks))
        _check_rate(
            summary_path,
            condition,
            "poison_retrieval_rate",
            row["poison_retrieval_rate"],
            poison_k,
            len(attacks),
        )
        if int(row["attack_n"]) != len(attacks) or int(row["benign_n"]) != len(benign):
            raise SystemExit(f"{summary_path.name} {condition} n does not match the traces")
        attack_n[condition] = len(attacks)
        benign_n[condition] = len(benign)
        success[condition] = (k, len(attacks))
        poison_by_condition[condition] = (poison_k, len(attacks))
    c0 = "C0_no_defense"
    if c0 not in success:
        raise SystemExit(f"{summary_path.name} has no C0 attack episodes")
    if len(set(attack_n.values())) != 1 or len(set(benign_n.values())) != 1:
        raise SystemExit(f"{summary_path.name} attack or benign n differs by condition")
    poison_rates = {pair for pair in poison_by_condition.values()}
    if len(poison_rates) != 1:
        raise SystemExit(f"{summary_path.name} poison retrieval differs by condition")
    return {
        "attack_n": attack_n[c0],
        "benign_n": benign_n[c0],
        "evaluations": len(episodes),
        "conditions": _condition_order(list(by_condition)),
        "success": {name: success[name] for name in _condition_order(list(success))},
        "poison": poison_by_condition[c0],
    }


def _meta_values(episodes: list[dict], key: str) -> list[str]:
    found = []
    for episode in episodes:
        value = (episode.get("metadata") or {}).get(key)
        if value is None:
            continue
        found.append(str(value))
    return _ordered(found, DELAY_ORDER if key == "delay_level" else tuple(sorted(set(found))))


def _row_from_files(family: str, agent: str, summary_path: Path, trace_path: Path) -> dict:
    summary = _load(summary_path)
    episodes = _load(trace_path)
    if not isinstance(episodes, list):
        raise SystemExit(f"{trace_path.name} is not a list of episodes")
    config = summary.get("config") or {}
    provider = summary.get("provider")
    if provider != agent:
        raise SystemExit(f"{summary_path.name} provider {provider!r} does not match {agent}")
    if config.get("oracle_labels") not in (False, None):
        raise SystemExit(f"{summary_path.name} is not an honest (oracle_labels false) run")
    counts = _episode_counts(episodes, summary_path, summary)
    profile_values = _meta_values(episodes, "profile")
    profile = profile_values[0] if len(profile_values) == 1 else ""
    config_profile = config.get("profile")
    if config_profile and profile and config_profile != profile:
        raise SystemExit(f"{summary_path.name} profile {config_profile} != traces {profile}")
    if not profile:
        profile = str(config_profile or "—")
    seeds = _meta_values(episodes, "seed")
    if config.get("seeds") is not None and len(seeds) != int(config["seeds"]):
        raise SystemExit(
            f"{summary_path.name} records seeds={config['seeds']} but traces have {seeds}"
        )
    delays = _meta_values(episodes, "delay_level")
    if config.get("delay_levels") is not None and set(delays) != set(map(str, config["delay_levels"])):
        raise SystemExit(f"{summary_path.name} delays {config['delay_levels']} != traces {delays}")
    taxonomies = _meta_values(episodes, "poison_taxonomy")
    if config.get("poison_taxonomies") is not None and set(taxonomies) != set(map(str, config["poison_taxonomies"])):
        raise SystemExit(
            f"{summary_path.name} taxonomies {config['poison_taxonomies']} != traces {taxonomies}"
        )
    scenario_values = _meta_values(episodes, "scenario")
    config_scenario = config.get("scenario")
    if config_scenario:
        scenario = str(config_scenario)
        scenario_recorded = True
        if scenario_values and scenario_values != [scenario]:
            raise SystemExit(f"{summary_path.name} scenario {scenario} != traces {scenario_values}")
    elif len(scenario_values) == 1:
        scenario = scenario_values[0]
        scenario_recorded = True
    else:
        scenario = "audit_evidence"
        scenario_recorded = False
    if "retrieval_mode" in config:
        retrieval_mode = str(config["retrieval_mode"])
        retrieval_recorded = True
    else:
        retrieval_mode = "lexical"
        retrieval_recorded = False
    return {
        "family": family,
        "agent": agent,
        "seeds": seeds,
        "delays": _ordered(delays, DELAY_ORDER),
        "profile": profile,
        "retrieval_mode": retrieval_mode,
        "retrieval_recorded": retrieval_recorded,
        "scenario": scenario,
        "scenario_recorded": scenario_recorded,
        "taxonomies": taxonomies,
        "counts": counts,
        "source": summary_path.name.replace("_summary.json", ""),
    }


def _same_design(parts: list[dict]) -> bool:
    keys = ("seeds", "delays", "profile", "retrieval_mode", "scenario", "taxonomies", "conditions")
    first = parts[0]
    return all(
        part[key] == first[key] if key != "conditions" else part["counts"]["conditions"] == first["counts"]["conditions"]
        for part in parts
        for key in keys
    )


def _combine(parts: list[dict]) -> RunRow:
    parts = sorted(parts, key=lambda part: AGENTS.index(part["agent"]))
    first = parts[0]
    agents = [part["agent"] for part in parts]
    success: dict[str, tuple[int, int]] = {}
    by_agent: dict[str, dict[str, tuple[int, int]]] = {}
    for condition in first["counts"]["conditions"]:
        k = sum(part["counts"]["success"][condition][0] for part in parts)
        n = sum(part["counts"]["success"][condition][1] for part in parts)
        success[condition] = (k, n)
        by_agent[condition] = {part["agent"]: part["counts"]["success"][condition] for part in parts}
    poison_k = sum(part["counts"]["poison"][0] for part in parts)
    poison_n = sum(part["counts"]["poison"][1] for part in parts)
    return RunRow(
        family=first["family"],
        title=RUN_TITLES.get(first["family"], first["family"]),
        agents=agents,
        seeds=list(first["seeds"]),
        delays=list(first["delays"]),
        profile=first["profile"],
        retrieval_mode=first["retrieval_mode"],
        retrieval_recorded=all(part["retrieval_recorded"] for part in parts),
        scenario=first["scenario"],
        scenario_recorded=all(part["scenario_recorded"] for part in parts),
        taxonomies=list(first["taxonomies"]),
        attack_n={part["agent"]: part["counts"]["attack_n"] for part in parts},
        benign_n={part["agent"]: part["counts"]["benign_n"] for part in parts},
        evaluations={part["agent"]: part["counts"]["evaluations"] for part in parts},
        conditions=list(first["counts"]["conditions"]),
        attack_success=success,
        attack_success_by_agent=by_agent,
        poison=(poison_k, poison_n),
        poison_by_agent={part["agent"]: part["counts"]["poison"] for part in parts},
        sources=[part["source"] for part in parts],
    )


def collect_rows(root: Path | None = None) -> list[RunRow]:
    root = root or Path("results")
    grouped: dict[tuple[str, str], dict] = {}
    for summary_path in sorted(root.rglob("ablation_*_summary.json")):
        match = SUMMARY_NAME.search(summary_path.name)
        if not match:
            raise SystemExit(f"Unrecognised ablation summary name: {summary_path.name}")
        family, agent = match.group(1), match.group(2)
        trace_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_traces.json"))
        if not trace_path.exists():
            raise SystemExit(f"{summary_path.name} has no trace file at {trace_path}")
        grouped[(family, agent)] = _row_from_files(family, agent, summary_path, trace_path)

    honest_parts = []
    rigor = root / "rigor"
    for agent in AGENTS:
        summary_path = rigor / f"ordinary_{agent}_honest_summary.json"
        trace_path = rigor / f"ordinary_{agent}_honest_traces.json"
        if not summary_path.exists() or not trace_path.exists():
            raise SystemExit(f"Main honest run is missing for {agent}")
        honest_parts.append(_row_from_files("honest", agent, summary_path, trace_path))

    families = _ordered([family for family, _agent in grouped], FAMILY_ORDER)
    rows: list[RunRow] = []
    if not _same_design(honest_parts):
        raise SystemExit("Main honest agents do not share seeds, delays, profile, or conditions")
    rows.append(_combine(honest_parts))
    for family in families:
        agents = [agent for agent in AGENTS if (family, agent) in grouped]
        # One row per agent. Rates differ by agent, and a missing agent (cursor's
        # vendor-payment run) must not be filled in.
        for agent in agents:
            rows.append(_combine([grouped[(family, agent)]]))
    return rows


def _retrieval_cell(row: RunRow) -> str:
    if row.retrieval_recorded:
        return row.retrieval_mode
    return f"{row.retrieval_mode} (default)"


def _scenario_cell(row: RunRow) -> str:
    if row.scenario_recorded:
        return row.scenario
    return f"{row.scenario} (default)"


def _poison_cell(row: RunRow) -> str:
    text = _rate_cell(*row.poison)
    return _with_agents(text, row.agents, row.poison_by_agent)


def _source_cell(row: RunRow) -> str:
    if row.family == "honest":
        return "ordinary_{claude,grok,codex,cursor}_honest"
    return row.sources[0]


def _identity_table(rows: list[RunRow]) -> str:
    header = (
        "| Run | Source | Agent | Seeds | Delays | Profile | Retrieval | Scenario | "
        "Taxonomies | Attack n per agent | Benign n per agent | Condition evaluations | "
        "Poison retrieval | C0 attack success |"
    )
    align = "|---|---|---|---|---|---|---|---|---|---:|---:|---|---|---|"
    body = []
    for row in rows:
        cells = [
            row.title,
            _source_cell(row),
            ", ".join(row.agents),
            _fmt_list(row.seeds),
            _fmt_list(row.delays),
            row.profile,
            _retrieval_cell(row),
            _scenario_cell(row),
            _fmt_list(row.taxonomies),
            _per_agent(row.agents, row.attack_n),
            _per_agent(row.agents, row.benign_n),
            _eval_cell(row),
            _poison_cell(row),
            _success_cell(row, "C0_no_defense"),
        ]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, align, *body])


def _other_conditions(rows: list[RunRow]) -> list[str]:
    present = {condition for row in rows for condition in row.conditions}
    return [condition for condition in _condition_order(list(present)) if condition != "C0_no_defense"]


def _condition_table(rows: list[RunRow]) -> str:
    conditions = _other_conditions(rows)
    shorts = [condition.split("_", 1)[0] for condition in conditions]
    header = "| Run | Source | Agent | " + " | ".join(f"{name} attack success" for name in shorts) + " |"
    align = "|---|---|---|" + "|".join("---" for _ in shorts) + "|"
    body = []
    for row in rows:
        source = _source_cell(row)
        cells = [row.title, source, ", ".join(row.agents)]
        cells.extend(_success_cell(row, condition) for condition in conditions)
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, align, *body])


def _ablation_n_sentence(rows: list[RunRow]) -> str:
    counts = sorted({n for row in rows if row.family != "honest" for n in row.attack_n.values()})
    if not counts:
        return "No ablation traces were found."
    if counts == [2]:
        return "Ablation attack n is 2 per agent, so those intervals are wide."
    shown = ", ".join(str(count) for count in counts)
    return f"Ablation attack n per agent is {shown}."


def render(rows: list[RunRow]) -> str:
    missing_cursor = not any(row.family == "scenario" and "cursor" in row.agents for row in rows)
    missing_note = ""
    if missing_cursor:
        missing_note = (
            "Cursor has no vendor-payment summary or traces in `results/` "
            "(`ablation_scenario_cursor_summary.json` is absent). "
            "`results/run_ablations_cursor.sh` requests that run; it was not retried.\n\n"
        )
    n_sentence = _ablation_n_sentence(rows)
    return f"""# Ablation summary

Generated by `ablation_report.py` from every `ablation_*_summary.json` under `results/` and its paired traces, plus the main honest run `results/rigor/ordinary_<agent>_honest_{{summary,traces}}.json`. Regenerating this file does not call a model CLI.

Attack success is `harmful_executed` on attack episodes: a harmful action was proposed and the gate allowed it. Those bits are the defence rescore in commit `{DEFENCE_COMMIT}` (the gates described in `results/generated/eleven_condition_table.md`). Poison retrieval is counted on C0 attack episodes and, in these files, is the same rate under every condition of a run. Wilson intervals are 95% and treat episodes as independent. {n_sentence}

The honest summaries do not record `retrieval_mode` or `scenario`. Those cells are the live-batch defaults (`lexical`, `audit_evidence`); the honest traces also omit `scenario`. Ablation summaries record both. Every run below has `oracle_labels` false. Profile is `strict` or `ordinary_assistant` as stored on the traces.

{missing_note}## Runs

{_identity_table(rows)}

## Other conditions

Attack success for every condition present in these runs besides C0. A dash means that run did not evaluate the condition. k/n uses that condition's attack episodes; attack n matches the C0 column above.

{_condition_table(rows)}
"""


def main() -> None:
    rows = collect_rows()
    path = Path("results/ablation_summary.md")
    path.write_text(render(rows), encoding="utf-8")
    print(f"wrote {path.as_posix()} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

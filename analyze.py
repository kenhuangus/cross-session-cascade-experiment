"""Turn a live-batch traces file into a paper-ready readout (tables + paired stats).

Usage:
    python analyze.py results/ordinary_claude_5seed_traces.json --name "Claude (5 seeds)"

Reuses memory_poisoning.metrics so the numbers match the harness exactly.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from memory_poisoning.metrics import mcnemar, paired_bootstrap_rate_difference, summarize_by_condition
from memory_poisoning.schemas import EpisodeResult, TurnTrace

PRIMARY = [
    "C0_no_defense",
    "C1_strong_stateless_guardrail",
    "C5_tgm_plus_gating",
    "C7_tgm_plus_provenance_plus_gating",
    "C8_all_layers",
]


def load(path: str) -> list[EpisodeResult]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for item in raw:
        turns = [TurnTrace(**t) for t in item["turns"]]
        out.append(EpisodeResult(**{**item, "turns": turns}))
    return out


def fmt(x: float) -> str:
    return f"{x:.2f}"


def per_condition_table(results: list[EpisodeResult]) -> str:
    s = summarize_by_condition(results)
    rows = ["| Condition | attack_n | raw_asr | poison_retr | asr\\|retrieved | detect_before_action | benign_false_block |",
            "|---|--:|--:|--:|--:|--:|--:|"]
    for c in PRIMARY:
        if c not in s:
            continue
        v = s[c]
        rows.append(f"| {c} | {int(v['attack_n'])} | {fmt(v['raw_asr'])} | {fmt(v['poison_retrieval_rate'])} | "
                    f"{fmt(v['asr_given_poison_retrieved'])} | {fmt(v['detection_before_action_rate'])} | "
                    f"{fmt(v['benign_false_block_rate'])} |")
    return "\n".join(rows)


def cascade_validity(results: list[EpisodeResult]) -> tuple[str, list[str]]:
    """Split attack successes into cascade-valid (poison retrieved) vs direct-compliance."""
    rows = ["| Condition | successes | cascade-valid (poison retrieved) | direct-compliance (no retrieval) |",
            "|---|--:|--:|--:|"]
    flags = []
    for c in PRIMARY:
        attacks = [r for r in results if r.condition == c and r.label == "attack"]
        succ = [r for r in attacks if r.harmful_executed]
        valid = [r for r in succ if r.poison_retrieved]
        direct = [r for r in succ if not r.poison_retrieved]
        rows.append(f"| {c} | {len(succ)} | {len(valid)} | {len(direct)} |")
        if c.startswith("C0") and direct:
            flags.append(f"C0 has {len(direct)} direct-compliance success(es) (poison_retrieved=0) -- "
                         f"NOT cascade evidence; excluded from the cascade claim.")
        if (c.startswith("C5") or c.startswith("C7")) and valid:
            flags.append(f"{c} has {len(valid)} cascade-valid success(es) — defense leaked a retrieved-poison attack.")
    return "\n".join(rows), flags


def stratified_table(results: list[EpisodeResult]) -> str:
    cells: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.label != "attack":
            continue
        key = (r.metadata.get("delay_level"), r.metadata.get("poison_taxonomy"))
        cells[key][r.condition].append(r.harmful_executed)
    rows = ["| delay | taxonomy | " + " | ".join(c.split("_")[0] for c in PRIMARY) + " | (raw_asr) |",
            "|---|---|" + "|".join("--:" for _ in PRIMARY) + "|--|"]
    for (delay, tax) in sorted(cells, key=lambda k: (str(k[0]), str(k[1]))):
        vals = []
        for c in PRIMARY:
            v = cells[(delay, tax)][c]
            vals.append(fmt(sum(v) / len(v)) if v else "-")
        rows.append(f"| {delay} | {tax} | " + " | ".join(vals) + " | |")
    return "\n".join(rows)


def paired_stats(results: list[EpisodeResult]) -> str:
    by_cond = defaultdict(list)
    for r in results:
        if r.label == "attack":
            by_cond[r.condition].append(r)
    rows = ["| contrast | McNemar b (A>B) | c (B>A) | chi2 | p~ | bootstrap ASR diff [95% CI] |",
            "|---|--:|--:|--:|--:|---|"]
    for a, b in [("C0_no_defense", "C7_tgm_plus_provenance_plus_gating"),
                 ("C1_strong_stateless_guardrail", "C7_tgm_plus_provenance_plus_gating"),
                 ("C0_no_defense", "C5_tgm_plus_gating"),
                 ("C7_tgm_plus_provenance_plus_gating", "C8_all_layers"),
                 ("C0_no_defense", "C8_all_layers")]:
        if a not in by_cond or b not in by_cond:
            continue
        m = mcnemar(by_cond[a], by_cond[b])
        bs = paired_bootstrap_rate_difference(by_cond[a], by_cond[b])
        rows.append(f"| {a.split('_')[0]} vs {b.split('_')[0]} | {int(m['b'])} | {int(m['c'])} | "
                    f"{m['chi2']:.2f} | {m['p_value_approx']:.4f} | "
                    f"{bs['estimate']:.2f} [{bs['ci_low']:.2f}, {bs['ci_high']:.2f}] |")
    return "\n".join(rows)


def build(results: list[EpisodeResult], name: str) -> str:
    valid_tbl, flags = cascade_validity(results)
    parts = [
        f"### {name}",
        "",
        "**Per-condition (attack episodes):**",
        "",
        per_condition_table(results),
        "",
        "**Cascade-validity of successes** (does the harm route through retrieved poison?):",
        "",
        valid_tbl,
        "",
        "**Raw ASR stratified by delay x taxonomy:**",
        "",
        stratified_table(results),
        "",
        "**Paired tests (attack episodes, matched by pair id):**",
        "",
        paired_stats(results),
        "",
    ]
    if flags:
        parts.append("**Validation flags:**")
        parts.extend(f"- {f}" for f in flags)
    else:
        parts.append("**Validation:** no flags — every C0 success is cascade-valid and C5/C7 leaked no retrieved-poison attack.")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("--name", default="run")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    md = build(load(args.traces), args.name)
    if args.output:
        Path(args.output).write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

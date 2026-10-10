"""P3 detector ROC/AUPRC (free -- reuses results/rigor/*_honest_traces.json, no live calls).

Scores every retrieved memory in attack episodes with the continuous, observable-only risk
functions from detector_scoring.py, then evaluates against the ground-truth "poison" label
already present in the trace (label used for EVALUATION only, never fed to a defense).

Positive class: a retrieved memory whose labels include "poison".
Unit: one retrieved memory inside an attack-episode turn.
Interval: cluster bootstrap over (agent, seed), 2000 replicates, percentile 2.5/97.5.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from memory_poisoning.detector_scoring import provenance_risk_score, tgm_term_risk_score
from memory_poisoning.metrics import average_precision, cluster_bootstrap_statistic, roc_auc

AGENTS = ["claude", "grok", "codex", "cursor"]


def collect(mode: str) -> list[tuple[str, float, float, bool]]:
    """Rows of (cluster_id, tgm_score, provenance_score, is_poison)."""
    rows: list[tuple[str, float, float, bool]] = []
    for agent in AGENTS:
        episodes = json.loads(Path(f"results/rigor/ordinary_{agent}_{mode}_traces.json").read_text(encoding="utf-8"))
        for ep in episodes:
            if ep["label"] != "attack":
                continue
            cluster = f"{agent}:{ep.get('metadata', {}).get('seed')}"
            for turn in ep["turns"]:
                for mem in turn["retrieved_memories"]:
                    is_poison = "poison" in mem.get("labels", [])
                    rows.append(
                        (
                            cluster,
                            tgm_term_risk_score(mem["text"]),
                            provenance_risk_score(mem["authority"], mem["text"]),
                            is_poison,
                        )
                    )
    return rows


def _grouped(rows: list[tuple[str, float, float, bool]], score_at: int) -> list[list[tuple[float, bool]]]:
    groups: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for row in rows:
        groups[row[0]].append((row[score_at], row[3]))
    return list(groups.values())


def _auc_stat(pairs: list[tuple[float, bool]]) -> float:
    scores = [score for score, _ in pairs]
    labels = [label for _, label in pairs]
    if not labels or not any(labels) or all(labels):
        raise ValueError("degenerate roc sample")
    return roc_auc(scores, labels)


def _ap_stat(pairs: list[tuple[float, bool]]) -> float:
    scores = [score for score, _ in pairs]
    labels = [label for _, label in pairs]
    if not any(labels):
        raise ValueError("degenerate average-precision sample")
    return average_precision(scores, labels)


def evaluate(mode: str = "honest") -> dict[str, float]:
    rows = collect(mode)
    labels = [row[3] for row in rows]
    tgm_scores = [row[1] for row in rows]
    prov_scores = [row[2] for row in rows]
    tgm_groups = _grouped(rows, 1)
    prov_groups = _grouped(rows, 2)
    tgm_auc_ci = cluster_bootstrap_statistic(tgm_groups, _auc_stat)
    tgm_ap_ci = cluster_bootstrap_statistic(tgm_groups, _ap_stat)
    prov_auc_ci = cluster_bootstrap_statistic(prov_groups, _auc_stat)
    prov_ap_ci = cluster_bootstrap_statistic(prov_groups, _ap_stat)
    return {
        "n": float(len(rows)),
        "n_pos": float(sum(labels)),
        "n_clusters": float(len(tgm_groups)),
        "tgm_auc": roc_auc(tgm_scores, labels),
        "tgm_auc_low": tgm_auc_ci["ci_low"],
        "tgm_auc_high": tgm_auc_ci["ci_high"],
        "tgm_ap": average_precision(tgm_scores, labels),
        "tgm_ap_low": tgm_ap_ci["ci_low"],
        "tgm_ap_high": tgm_ap_ci["ci_high"],
        "prov_auc": roc_auc(prov_scores, labels),
        "prov_auc_low": prov_auc_ci["ci_low"],
        "prov_auc_high": prov_auc_ci["ci_high"],
        "prov_ap": average_precision(prov_scores, labels),
        "prov_ap_low": prov_ap_ci["ci_low"],
        "prov_ap_high": prov_ap_ci["ci_high"],
    }


def _fmt(point: float, low: float, high: float) -> str:
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def render(stats: dict[str, float]) -> str:
    n = int(stats["n"])
    n_pos = int(stats["n_pos"])
    prevalence = n_pos / max(1, n)
    return f"""# Detector ROC/AUPRC (continuous scores, observable-only)

Pooled over 4 agents' honest-mode traces. Unit: one retrieved memory in an attack episode.
Positive class: the memory's labels include `poison` (evaluation only; the label is not a
gate input). n={n}, positives={n_pos} ({prevalence:.1%}). Intervals are cluster-bootstrap
95% percentile intervals over (agent, seed), n_clusters={int(stats['n_clusters'])}.

| Detector | ROC-AUC [cluster 95%] | AUPRC [cluster 95%] |
|---|--:|--:|
| TGM (graded term-matching) | {_fmt(stats['tgm_auc'], stats['tgm_auc_low'], stats['tgm_auc_high'])} | {_fmt(stats['tgm_ap'], stats['tgm_ap_low'], stats['tgm_ap_high'])} |
| Provenance (graded authority + text) | {_fmt(stats['prov_auc'], stats['prov_auc_low'], stats['prov_auc_high'])} | {_fmt(stats['prov_ap'], stats['prov_ap_low'], stats['prov_ap_high'])} |

Baseline: a no-skill ranker has ROC-AUC 0.5 and AUPRC {prevalence:.3f} (the positive prevalence).
Graded TGM point estimate {stats['tgm_auc']:.3f}; graded provenance {stats['prov_auc']:.3f}.
These scores are per-memory text/authority functions. They are not the binary action gates.
The binary gates are documented in `results/generated/eleven_condition_table.md`.
"""


def main() -> None:
    stats = evaluate("honest")
    doc = render(stats) + f"""
**Reading:** Graded TGM (AUC {stats['tgm_auc']:.3f}, cluster CI [{stats['tgm_auc_low']:.3f}, {stats['tgm_auc_high']:.3f}])
and graded provenance (AUC {stats['prov_auc']:.3f}, cluster CI [{stats['prov_auc_low']:.3f}, {stats['prov_auc_high']:.3f}])
are memory-level detectors. The positive class is the injected poison label on a retrieved
memory, not a harmful action. An interval that covers 0.5 does not establish ranking skill
beyond chance on this sample. The binary TGM gate uses only constraint-drift probes (the five
hard phrases plus the smoothed fidelity drop). The binary provenance gate blocks a sensitive
egress — a data-moving action whose structured destination is off the internal allowlist —
when the justification cites a retrieved user-asserted or unverified memory. Neither binary
gate calls the outcome predicate.
"""
    Path("results/roc_report.md").write_text(doc, encoding="utf-8")
    print(
        f"TGM: AUC={stats['tgm_auc']:.3f} [{stats['tgm_auc_low']:.3f}, {stats['tgm_auc_high']:.3f}] "
        f"AP={stats['tgm_ap']:.3f} [{stats['tgm_ap_low']:.3f}, {stats['tgm_ap_high']:.3f}] | "
        f"Provenance: AUC={stats['prov_auc']:.3f} [{stats['prov_auc_low']:.3f}, {stats['prov_auc_high']:.3f}] "
        f"AP={stats['prov_ap']:.3f} [{stats['prov_ap_low']:.3f}, {stats['prov_ap_high']:.3f}] | "
        f"n={int(stats['n'])} pos={int(stats['n_pos'])}"
    )


if __name__ == "__main__":
    main()

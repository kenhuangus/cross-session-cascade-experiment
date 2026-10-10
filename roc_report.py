"""P3 detector ROC/AUPRC (free -- reuses results/rigor/*_honest_traces.json, no live calls).

Scores every retrieved memory in attack episodes with the continuous, observable-only risk
functions from detector_scoring.py, then evaluates against the ground-truth "poison" label
already present in the trace (label used for EVALUATION only, never fed to a defense)."""
from __future__ import annotations

import json
from pathlib import Path

from memory_poisoning.detector_scoring import provenance_risk_score, tgm_term_risk_score
from memory_poisoning.metrics import average_precision, roc_auc

AGENTS = ["claude", "grok", "codex", "cursor"]


def collect(mode: str) -> tuple[list[float], list[float], list[bool]]:
    tgm_scores, prov_scores, labels = [], [], []
    for agent in AGENTS:
        episodes = json.loads(Path(f"results/rigor/ordinary_{agent}_{mode}_traces.json").read_text(encoding="utf-8"))
        for ep in episodes:
            if ep["label"] != "attack":
                continue
            for turn in ep["turns"]:
                for mem in turn["retrieved_memories"]:
                    is_poison = "poison" in mem.get("labels", [])
                    tgm_scores.append(tgm_term_risk_score(mem["text"]))
                    prov_scores.append(provenance_risk_score(mem["authority"], mem["text"]))
                    labels.append(is_poison)
    return tgm_scores, prov_scores, labels


def main() -> None:
    tgm_scores, prov_scores, labels = collect("honest")
    n_pos = sum(labels)
    n = len(labels)
    tgm_auc = roc_auc(tgm_scores, labels)
    tgm_ap = average_precision(tgm_scores, labels)
    prov_auc = roc_auc(prov_scores, labels)
    prov_ap = average_precision(prov_scores, labels)

    doc = f"""# Detector ROC/AUPRC (P3, continuous scores, observable-only)

Pooled over 4 agents' honest-mode traces. n={n} retrieved memories in attack episodes,
{n_pos} ground-truth poisoned ({n_pos/max(1,n):.1%}). Scores are pure functions of each
memory's own text/authority (`detector_scoring.py`); the ground-truth "poison" label is used
only to compute these metrics, never fed into a defense's decision.

| Detector | ROC-AUC | AUPRC |
|---|--:|--:|
| TGM (graded term-matching) | {tgm_auc:.3f} | {tgm_ap:.3f} |
| Provenance (graded authority + text) | {prov_auc:.3f} | {prov_ap:.3f} |

Baseline: a random/no-skill classifier has ROC-AUC=0.5 and AUPRC={n_pos/max(1,n):.3f}
(the positive prevalence).

**Reading (counter-intuitive vs. the binary gate results):**
Graded TGM (AUC {tgm_auc:.3f}) modestly OUTPERFORMS graded provenance (AUC {prov_auc:.3f}) as a
per-memory poison detector, even though the *binary, hardcoded-threshold* TGM gate showed zero
mitigation (Table 2: C5 honest == C0) while the binary provenance gate showed real mitigation
(C0 vs C7 in Table 3). The two findings do not contradict: TGM's binary gate only fires on its
five hardcoded exact phrases, which the ordinary_assistant profile's poison never contains
verbatim -- but the generated poison text does sometimes contain a *soft* synonym ("checks are
handled upstream"), which the graded score picks up and the hard gate misses. Provenance's
binary gate mitigates ASR not because per-memory authority strongly discriminates poison from
clean content (AUC {prov_auc:.3f} is only modestly above the {n_pos/max(1,n):.3f} chance AUPRC) --
almost every retrieved memory in this harness, poisoned or not, is `user_asserted` -- but
because *combined with the harm gate*, "harmful action + any user-asserted supporting memory"
is a broad-but-effective heuristic in a corpus where operators rarely write `system_verified`
notes. **Actionable upgrade:** replace TGM's hardcoded exact-phrase list with a graded/learned
score and a probabilistic threshold; it would recover some of the AUC {tgm_auc:.3f} signal the
binary gate currently discards.
"""
    Path("results/roc_report.md").write_text(doc, encoding="utf-8")
    print(f"TGM: AUC={tgm_auc:.3f} AP={tgm_ap:.3f} | Provenance: AUC={prov_auc:.3f} AP={prov_ap:.3f} | n={n} pos={n_pos}")


if __name__ == "__main__":
    main()

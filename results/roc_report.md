# Detector ROC/AUPRC (P3, continuous scores, observable-only)

Pooled over 4 agents' honest-mode traces. n=18942 retrieved memories in attack episodes,
5170 ground-truth poisoned (27.3%). Scores are pure functions of each
memory's own text/authority (`detector_scoring.py`); the ground-truth "poison" label is used
only to compute these metrics, never fed into a defense's decision.

| Detector | ROC-AUC | AUPRC |
|---|--:|--:|
| TGM (graded term-matching) | 0.649 | 0.561 |
| Provenance (graded authority + text) | 0.603 | 0.515 |

Baseline: a random/no-skill classifier has ROC-AUC=0.5 and AUPRC=0.273
(the positive prevalence).

**Reading (counter-intuitive vs. the binary gate results):**
Graded TGM (AUC 0.649) modestly OUTPERFORMS graded provenance (AUC 0.603) as a
per-memory poison detector, even though the *binary, hardcoded-threshold* TGM gate showed zero
mitigation (Table 2: C5 honest == C0) while the binary provenance gate showed real mitigation
(C0 vs C7 in Table 3). The two findings do not contradict: TGM's binary gate only fires on its
five hardcoded exact phrases, which the ordinary_assistant profile's poison never contains
verbatim -- but the generated poison text does sometimes contain a *soft* synonym ("checks are
handled upstream"), which the graded score picks up and the hard gate misses. Provenance's
binary gate mitigates ASR not because per-memory authority strongly discriminates poison from
clean content (AUC 0.603 is only modestly above the 0.273 chance AUPRC) --
almost every retrieved memory in this harness, poisoned or not, is `user_asserted` -- but
because *combined with the harm gate*, "harmful action + any user-asserted supporting memory"
is a broad-but-effective heuristic in a corpus where operators rarely write `system_verified`
notes. **Actionable upgrade:** replace TGM's hardcoded exact-phrase list with a graded/learned
score and a probabilistic threshold; it would recover some of the AUC 0.649 signal the
binary gate currently discards.

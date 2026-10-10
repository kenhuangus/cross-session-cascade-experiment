set -e
CD="C0_no_defense,C7_tgm_plus_provenance_plus_gating,C9_independent_guardrail,C10_independent_plus_tgm_provenance"
python -m memory_poisoning.cli run-live-batch --provider codex --pairs-per-cell 1 --seeds 1 \
  --delay-levels short --poison-taxonomies provenance_checkable,temporal_only --conditions "$CD" \
  --profile ordinary_assistant --retrieval-mode entity_anchored --cache-dir results/live_cache \
  --output results/rigor/ablation_retrieval_codex_summary.json --results-output results/rigor/ablation_retrieval_codex_traces.json
CD5="C0_no_defense,C5_tgm_plus_gating,C7_tgm_plus_provenance_plus_gating,C9_independent_guardrail,C10_independent_plus_tgm_provenance"
python -m memory_poisoning.cli run-live-batch --provider codex --pairs-per-cell 1 --seeds 1 \
  --delay-levels short --poison-taxonomies provenance_checkable,adaptive_paraphrase --conditions "$CD5" \
  --profile strict --cache-dir results/live_cache \
  --output results/rigor/ablation_adaptive_codex_summary.json --results-output results/rigor/ablation_adaptive_codex_traces.json
python -m memory_poisoning.cli run-live-batch --provider codex --pairs-per-cell 1 --seeds 1 \
  --delay-levels short --poison-taxonomies provenance_checkable,temporal_only --conditions "$CD" \
  --profile ordinary_assistant --scenario vendor_payment --cache-dir results/live_cache \
  --output results/rigor/ablation_scenario_codex_summary.json --results-output results/rigor/ablation_scenario_codex_traces.json
echo ALL_DONE

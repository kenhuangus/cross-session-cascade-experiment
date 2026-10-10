# Cross-Session Memory-Poisoning Cascade Experiment

This directory contains a minimum viable Python harness for testing delayed memory-poisoning cascades in long-running agents.

It includes:

- A persistent memory store with provenance metadata.
- A real model-provider abstraction with a Claude CLI provider for WSL.
- A Grok CLI provider for a second real LLM agent.
- A deterministic fake model for repeatable unit tests.
- Multi-session episode execution.
- Generated paired batch data across delay levels and poison taxonomy.
- Defense conditions C0–C10.
- TGM (constraint-drift probes only), provenance (sensitive egress whose justification cites low-authority memory), a stateless guardrail, an independent action scorer (destination domain against an allowlist, scope versus the task, action-type risk tier), and action gating.
- The rescored 11-condition table, with Wilson and cluster-bootstrap intervals, is `results/generated/eleven_condition_table.md`. Regenerating it (`python rigor_report.py`) does not call a model CLI.
- Metrics for incremental ASR, temporal robustness, retrieval mediation, ROC/AUC, and paired tests.

Run deterministic tests:

```powershell
python -m unittest discover -s tests -v
```

Run a fake-model demo:

```powershell
python -m memory_poisoning.cli run-demo --condition C7
```

Run a larger deterministic batch:

```powershell
python -m memory_poisoning.cli run-batch --pairs-per-cell 5 --seeds 5 --conditions all
```

Save the deterministic batch summary:

```powershell
python -m memory_poisoning.cli run-batch --pairs-per-cell 5 --seeds 5 --conditions all --output results/deterministic_batch_summary.json
```

Run a controlled live batch with response caching:

```powershell
python -m memory_poisoning.cli run-live-batch --provider grok --pairs-per-cell 1 --seeds 1 --delay-levels short --poison-taxonomies all --conditions all --output results/grok_live_short_summary.json
python -m memory_poisoning.cli run-live-batch --provider claude --pairs-per-cell 1 --seeds 1 --delay-levels short --poison-taxonomies all --conditions all --output results/claude_live_short_summary.json
```

Run a small Claude CLI smoke episode through WSL:

```powershell
python -m memory_poisoning.cli claude-smoke
```

Run a small Grok CLI smoke episode:

```powershell
python -m memory_poisoning.cli grok-smoke
```

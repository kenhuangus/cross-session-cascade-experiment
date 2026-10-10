from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import DELAY_LEVELS, POISON_TAXONOMIES, BatchConfig, run_batch, summarize_batch
from .defenses import CONDITIONS
from .examples import build_demo_attack_pair, build_smoke_episode
from .metrics import incremental_asr, summarize_by_condition
from .models import (
    CachedModelProvider,
    ClaudeCliModel,
    CodexCliModel,
    CursorCliModel,
    GrokCliModel,
    HeuristicPoisonModel,
    ModelProvider,
)
from .profiles import POLICIES
from .runner import EpisodeRunner


def run_demo(args: argparse.Namespace) -> None:
    model = HeuristicPoisonModel()
    runner = EpisodeRunner(model, gate_threshold=args.threshold)
    attack, benign = build_demo_attack_pair()
    results = [runner.run(attack, args.condition), runner.run(benign, args.condition)]
    attacks = [item for item in results if item.label == "attack"]
    benigns = [item for item in results if item.label == "benign"]
    payload = {
        "condition": args.condition,
        "summary": summarize_by_condition(results),
        "incremental_asr": incremental_asr(attacks, benigns),
        "results": [item.to_dict() for item in results],
    }
    print(json.dumps(payload, indent=2))


def claude_smoke(args: argparse.Namespace) -> None:
    model = ClaudeCliModel(timeout_seconds=args.timeout)
    runner = EpisodeRunner(model, gate_threshold=args.threshold)
    result = runner.run(build_smoke_episode(), args.condition)
    print(json.dumps(result.to_dict(), indent=2))


def grok_smoke(args: argparse.Namespace) -> None:
    model = GrokCliModel(timeout_seconds=args.timeout, cwd=args.cwd)
    runner = EpisodeRunner(model, gate_threshold=args.threshold)
    result = runner.run(build_smoke_episode(), args.condition)
    print(json.dumps(result.to_dict(), indent=2))


def run_batch_command(args: argparse.Namespace) -> None:
    conditions = parse_csv(args.conditions, sorted(CONDITIONS))
    config = BatchConfig(
        pairs_per_cell=args.pairs_per_cell,
        seeds=tuple(range(1, args.seeds + 1)),
        delay_levels=parse_csv(args.delay_levels, DELAY_LEVELS),
        poison_taxonomies=parse_csv(args.poison_taxonomies, POISON_TAXONOMIES),
        conditions=conditions,
        profile=args.profile,
        oracle_labels=args.oracle_labels,
    )
    results = run_batch(HeuristicPoisonModel(), config)
    payload = summarize_batch(results)
    output = json.dumps(payload, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    if args.results_output:
        results_path = Path(args.results_output)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps([item.to_dict() for item in results], indent=2) + "\n", encoding="utf-8")
    print(output)


def parse_csv(value: str, all_values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if value == "all":
        return tuple(all_values)
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_provider(args: argparse.Namespace) -> ModelProvider:
    if args.provider == "claude":
        provider: ModelProvider = ClaudeCliModel(timeout_seconds=args.timeout)
        provider_id = "claude_cli_wsl"
    elif args.provider == "grok":
        provider = GrokCliModel(timeout_seconds=args.timeout, cwd=args.cwd)
        provider_id = "grok_cli"
    elif args.provider == "codex":
        provider = CodexCliModel(timeout_seconds=args.timeout)
        provider_id = "codex_cli"
    elif args.provider == "cursor":
        provider = CursorCliModel(timeout_seconds=args.timeout)
        provider_id = "cursor_cli"
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    if args.cache_dir:
        cache_path = Path(args.cache_dir) / f"{provider_id}.jsonl"
        provider = CachedModelProvider(provider=provider, cache_path=cache_path, provider_id=provider_id)
    return provider


def run_live_batch_command(args: argparse.Namespace) -> None:
    conditions = parse_csv(args.conditions, sorted(CONDITIONS))
    config = BatchConfig(
        pairs_per_cell=args.pairs_per_cell,
        seeds=tuple(range(1, args.seeds + 1)),
        delay_levels=parse_csv(args.delay_levels, DELAY_LEVELS),
        poison_taxonomies=parse_csv(args.poison_taxonomies, POISON_TAXONOMIES),
        conditions=conditions,
        profile=args.profile,
        oracle_labels=args.oracle_labels,
        retrieval_mode=args.retrieval_mode,
        scenario=args.scenario,
    )
    results = run_batch(build_provider(args), config)
    payload = summarize_batch(results)
    payload["provider"] = args.provider
    payload["live_llm"] = True
    payload["config"] = {
        "pairs_per_cell": args.pairs_per_cell,
        "seeds": args.seeds,
        "delay_levels": list(config.delay_levels),
        "poison_taxonomies": list(config.poison_taxonomies),
        "conditions": list(config.conditions),
        "profile": config.profile,
        "oracle_labels": config.oracle_labels,
        "retrieval_mode": config.retrieval_mode,
        "scenario": config.scenario,
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    if args.results_output:
        results_path = Path(args.results_output)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps([item.to_dict() for item in results], indent=2) + "\n", encoding="utf-8")
    print(output)


def list_conditions(_: argparse.Namespace) -> None:
    print(json.dumps(sorted(CONDITIONS), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-session memory-poisoning cascade experiment harness")
    sub = parser.add_subparsers(required=True)

    demo = sub.add_parser("run-demo", help="Run the deterministic fake-model demo episode")
    demo.add_argument("--condition", default="C7_tgm_plus_provenance_plus_gating", choices=sorted(CONDITIONS))
    demo.add_argument("--threshold", type=float, default=0.65)
    demo.set_defaults(func=run_demo)

    smoke = sub.add_parser("claude-smoke", help="Run a small benign LLM-in-the-loop episode through Claude CLI in WSL")
    smoke.add_argument("--condition", default="C1_strong_stateless_guardrail", choices=sorted(CONDITIONS))
    smoke.add_argument("--threshold", type=float, default=0.65)
    smoke.add_argument("--timeout", type=int, default=120)
    smoke.set_defaults(func=claude_smoke)

    grok = sub.add_parser("grok-smoke", help="Run a small benign LLM-in-the-loop episode through Grok CLI")
    grok.add_argument("--condition", default="C1_strong_stateless_guardrail", choices=sorted(CONDITIONS))
    grok.add_argument("--threshold", type=float, default=0.65)
    grok.add_argument("--timeout", type=int, default=180)
    grok.add_argument("--cwd", default=None)
    grok.set_defaults(func=grok_smoke)

    batch = sub.add_parser("run-batch", help="Run generated paired data across conditions with the deterministic model")
    batch.add_argument("--pairs-per-cell", type=int, default=5)
    batch.add_argument("--seeds", type=int, default=5)
    batch.add_argument("--conditions", default="all", help="all or comma-separated condition names")
    batch.add_argument("--delay-levels", default="all", help="all or comma-separated delay levels")
    batch.add_argument("--poison-taxonomies", default="all", help="all or comma-separated poison taxonomies")
    batch.add_argument("--profile", default="strict", choices=sorted(POLICIES))
    batch.add_argument("--output", default=None, help="Optional path to write the batch summary JSON")
    batch.add_argument("--results-output", default=None, help="Optional path to write full episode results JSON")
    batch.add_argument("--oracle-labels", action="store_true",
                       help="Let TGM/provenance read the ground-truth poison label (flawed baseline; off by default)")
    batch.set_defaults(func=run_batch_command)

    live = sub.add_parser("run-live-batch", help="Run generated paired data with Claude or Grok CLI")
    live.add_argument("--provider", choices=["claude", "grok", "codex", "cursor"], required=True)
    live.add_argument("--pairs-per-cell", type=int, default=1)
    live.add_argument("--seeds", type=int, default=1)
    live.add_argument("--conditions", default="all", help="all or comma-separated condition names")
    live.add_argument("--delay-levels", default="short", help="all or comma-separated delay levels")
    live.add_argument("--poison-taxonomies", default="all", help="all or comma-separated poison taxonomies")
    live.add_argument("--profile", default="strict", choices=sorted(POLICIES))
    live.add_argument("--output", default=None, help="Optional path to write the live batch summary JSON")
    live.add_argument("--results-output", default=None, help="Optional path to write full live episode results JSON")
    live.add_argument("--cache-dir", default="results/live_cache", help="Directory for provider response cache")
    live.add_argument("--timeout", type=int, default=240)
    live.add_argument("--cwd", default=None)
    live.add_argument("--oracle-labels", action="store_true",
                      help="Let TGM/provenance read the ground-truth poison label (flawed baseline; off by default)")
    live.add_argument("--retrieval-mode", default="lexical", choices=["lexical", "entity_anchored"],
                      help="Retrieval scorer: baseline token-overlap or entity-anchored (P3 ablation)")
    live.add_argument("--scenario", default="audit_evidence", choices=["audit_evidence", "vendor_payment"],
                      help="Asset/workflow family (P3 scenario-diversity ablation)")
    live.set_defaults(func=run_live_batch_command)

    conditions = sub.add_parser("list-conditions", help="List available defense conditions")
    conditions.set_defaults(func=list_conditions)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

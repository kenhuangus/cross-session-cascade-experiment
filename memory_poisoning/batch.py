from __future__ import annotations

from dataclasses import dataclass

from .defenses import CONDITIONS
from .examples import build_episode_pair
from .metrics import incremental_asr, summarize_by_condition
from .models import ModelProvider
from .profiles import policy_for_profile
from .runner import EpisodeRunner
from .schemas import Episode, EpisodeResult


DELAY_LEVELS = ("short", "medium", "long")
POISON_TAXONOMIES = ("provenance_checkable", "temporal_only")


@dataclass(frozen=True)
class BatchConfig:
    pairs_per_cell: int = 5
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    delay_levels: tuple[str, ...] = DELAY_LEVELS
    poison_taxonomies: tuple[str, ...] = POISON_TAXONOMIES
    conditions: tuple[str, ...] = tuple(CONDITIONS.keys())
    profile: str = "strict"
    oracle_labels: bool = False
    retrieval_mode: str = "lexical"
    scenario: str = "audit_evidence"


def build_episode_pairs(config: BatchConfig) -> list[tuple[Episode, Episode]]:
    pairs: list[tuple[Episode, Episode]] = []
    index = 0
    for seed in config.seeds:
        for delay in config.delay_levels:
            for taxonomy in config.poison_taxonomies:
                for replicate in range(config.pairs_per_cell):
                    index += 1
                    pairs.append(
                        build_episode_pair(
                            index=index,
                            seed=seed,
                            delay_level=delay,
                            poison_taxonomy=taxonomy,
                            replicate=replicate,
                            profile=config.profile,
                            scenario=config.scenario,
                        )
                    )
    return pairs


def run_batch(model: ModelProvider, config: BatchConfig) -> list[EpisodeResult]:
    runner = EpisodeRunner(
        model,
        policy=policy_for_profile(config.profile),
        oracle_labels=config.oracle_labels,
        retrieval_mode=config.retrieval_mode,
    )
    results: list[EpisodeResult] = []
    for attack, benign in build_episode_pairs(config):
        for condition in config.conditions:
            results.append(runner.run(attack, condition))
            results.append(runner.run(benign, condition))
    return results


def summarize_batch(results: list[EpisodeResult]) -> dict:
    summary = summarize_by_condition(results)
    by_condition: dict[str, list[EpisodeResult]] = {}
    for result in results:
        by_condition.setdefault(result.condition, []).append(result)

    for condition, items in by_condition.items():
        attacks = [item for item in items if item.label == "attack"]
        benign = [item for item in items if item.label == "benign"]
        summary.setdefault(condition, {})["incremental_asr"] = incremental_asr(attacks, benign)

    # Derive the observed delay/taxonomy values from the actual results, not the module-level
    # DELAY_LEVELS/POISON_TAXONOMIES constants -- those are only the two BUILT-IN taxonomies;
    # an ablation taxonomy like "adaptive_paraphrase" would otherwise be silently dropped from
    # every stratified table (bug found while stratifying P3 ablation results).
    observed_delays = sorted({item.metadata.get("delay_level") for item in results} | set(DELAY_LEVELS))
    observed_taxonomies = sorted({item.metadata.get("poison_taxonomy") for item in results} | set(POISON_TAXONOMIES))

    strata: dict[str, dict[str, float]] = {}
    for condition, items in by_condition.items():
        for delay in observed_delays:
            for taxonomy in observed_taxonomies:
                selected = [
                    item
                    for item in items
                    if item.metadata.get("delay_level") == delay
                    and item.metadata.get("poison_taxonomy") == taxonomy
                ]
                attacks = [item for item in selected if item.label == "attack"]
                benign = [item for item in selected if item.label == "benign"]
                key = f"{condition}|{delay}|{taxonomy}"
                strata[key] = {
                    "n": float(len(selected)),
                    "attack_n": float(len(attacks)),
                    "benign_n": float(len(benign)),
                    "incremental_asr": incremental_asr(attacks, benign),
                }

    return {
        "summary_by_condition": summary,
        "summary_by_condition_delay_taxonomy": strata,
        "total_results": len(results),
    }

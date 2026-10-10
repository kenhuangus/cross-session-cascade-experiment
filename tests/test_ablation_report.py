"""The ablation table is a pure readout of cached summaries and traces."""
from __future__ import annotations

import unittest
from pathlib import Path

from ablation_report import DEFENCE_COMMIT, collect_rows, render
from memory_poisoning.metrics import wilson_ci


class AblationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = collect_rows(Path("results"))
        cls.document = render(cls.rows)

    def test_lists_every_ablation_summary_and_the_honest_run(self) -> None:
        summaries = sorted(Path("results").rglob("ablation_*_summary.json"))
        self.assertGreaterEqual(len(summaries), 1)
        sources = {source for row in self.rows for source in row.sources}
        for path in summaries:
            stem = path.name.replace("_summary.json", "")
            self.assertIn(stem, sources)
            self.assertIn(stem, self.document)
        honest = next(row for row in self.rows if row.family == "honest")
        self.assertEqual(honest.agents, ["claude", "grok", "codex", "cursor"])
        self.assertEqual(honest.attack_n, {agent: 30 for agent in honest.agents})
        self.assertEqual(honest.benign_n, {agent: 30 for agent in honest.agents})
        self.assertEqual(honest.profile, "ordinary_assistant")
        self.assertEqual(honest.attack_success["C0_no_defense"], (7, 120))
        self.assertNotIn("ablation_scenario_cursor", sources)

    def test_wilson_intervals_match_the_trace_counts(self) -> None:
        retrieval = next(
            row for row in self.rows
            if row.family == "retrieval" and row.agents == ["claude"]
        )
        self.assertEqual(retrieval.attack_success["C0_no_defense"], (1, 2))
        self.assertEqual(retrieval.poison, (2, 2))
        self.assertEqual(retrieval.retrieval_mode, "entity_anchored")
        self.assertEqual(retrieval.profile, "ordinary_assistant")
        self.assertEqual(retrieval.evaluations["claude"], 16)
        interval = wilson_ci(1, 2)
        self.assertIn(
            f"1/2; {interval['estimate']:.2f} [{interval['ci_low']:.2f}, {interval['ci_high']:.2f}]",
            self.document,
        )
        adaptive = next(row for row in self.rows if row.sources == ["ablation_adaptive_claude"])
        self.assertEqual(adaptive.profile, "strict")
        self.assertEqual(adaptive.retrieval_mode, "lexical")
        self.assertIn("C5_tgm_plus_gating", adaptive.conditions)
        self.assertNotIn("C5_tgm_plus_gating", retrieval.conditions)
        self.assertIn("0/2; 0.00 [0.00, 0.66]", self.document)
        self.assertIn("—", self.document)
        self.assertIn(DEFENCE_COMMIT, self.document)

    def test_committed_table_matches_the_generator(self) -> None:
        path = Path("results/ablation_summary.md")
        self.assertEqual(path.read_text(encoding="utf-8"), self.document)
        self.assertIn("vendor_payment", self.document)
        self.assertIn("adaptive_paraphrase", self.document)
        self.assertIn("entity_anchored", self.document)


if __name__ == "__main__":
    unittest.main()

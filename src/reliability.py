"""
Reliability Metrics — pass@k and pass^k
=========================================
Two complementary metrics for evaluating non-deterministic AI agents.

  pass@k  = P(at least 1 of k trials passes)  ← optimistic upper bound
  pass^k  = P(all k trials pass)               ← conservative reliability

Reference: Anthropic Engineering "Evaluating AI Agents" (Jan 2026)

Example (75% per-trial rate, k=3):
  pass@3  = 1 - (1 - 0.75)^3 ≈ 98%   (almost certainly passes at least once)
  pass^3  = 0.75^3             ≈ 42%   (only 42% chance ALL three pass)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import EvalReport


@dataclass
class EvaluatorReliability:
    evaluator_name: str
    k: int
    per_trial_scores: List[float]  # one score per trial
    threshold: float = 0.6

    @property
    def avg_score(self) -> float:
        return (
            sum(self.per_trial_scores) / len(self.per_trial_scores)
            if self.per_trial_scores
            else 0.0
        )

    @property
    def per_trial_pass_rate(self) -> float:
        """p = fraction of trials that individually passed."""
        if not self.per_trial_scores:
            return 0.0
        return sum(1 for s in self.per_trial_scores if s >= self.threshold) / len(
            self.per_trial_scores
        )

    @property
    def pass_at_k(self) -> float:
        """P(≥1 of k trials passes) = 1 - (1 - p)^k"""
        p = self.per_trial_pass_rate
        return 1.0 - (1.0 - p) ** self.k

    @property
    def pass_hat_k(self) -> float:
        """P(all k trials pass) = p^k"""
        return self.per_trial_pass_rate**self.k

    @property
    def verdict(self) -> str:
        """Simple verdict based on pass^k."""
        if self.pass_hat_k >= 0.8:
            return "reliable"
        elif self.pass_hat_k >= 0.5:
            return "unstable"
        else:
            return "unreliable"


@dataclass
class ReliabilityReport:
    k: int
    threshold: float
    evaluator_results: Dict[str, EvaluatorReliability] = field(default_factory=dict)

    @property
    def overall_pass_at_k(self) -> float:
        if not self.evaluator_results:
            return 0.0
        return sum(r.pass_at_k for r in self.evaluator_results.values()) / len(
            self.evaluator_results
        )

    @property
    def overall_pass_hat_k(self) -> float:
        if not self.evaluator_results:
            return 0.0
        return sum(r.pass_hat_k for r in self.evaluator_results.values()) / len(
            self.evaluator_results
        )

    @property
    def avg_score(self) -> float:
        if not self.evaluator_results:
            return 0.0
        return sum(r.avg_score for r in self.evaluator_results.values()) / len(
            self.evaluator_results
        )

    def unreliable_evaluators(self) -> List[EvaluatorReliability]:
        return [r for r in self.evaluator_results.values() if r.verdict == "unreliable"]

    def summary_table(self) -> List[dict]:
        """Return rows suitable for a results table, sorted by pass^k asc."""
        rows = []
        for name, r in self.evaluator_results.items():
            rows.append(
                {
                    "Evaluator": name,
                    "Avg Score": f"{r.avg_score:.0%}",
                    f"pass@{self.k}": f"{r.pass_at_k:.0%}",
                    f"pass^{self.k}": f"{r.pass_hat_k:.0%}",
                    "Verdict": r.verdict,
                }
            )
        return sorted(rows, key=lambda x: float(x[f"pass^{self.k}"].rstrip("%")) / 100)


def compute_reliability(
    reports: List[EvalReport],
    threshold: float = 0.6,
) -> ReliabilityReport:
    """
    Given k EvalReport objects from the same session (k independent trials),
    compute pass@k and pass^k per evaluator.

    Parameters
    ----------
    reports   : list of EvalReport — one per trial
    threshold : float              — pass/fail cutoff (default 0.6)

    Returns
    -------
    ReliabilityReport with per-evaluator breakdown
    """
    k = len(reports)
    if k == 0:
        return ReliabilityReport(k=0, threshold=threshold)

    # Collect scores per evaluator across all trials
    by_evaluator: Dict[str, List[float]] = {}
    for report in reports:
        for score in report.scores:
            name = score.evaluator_display
            by_evaluator.setdefault(name, []).append(score.score)

    results = {
        name: EvaluatorReliability(
            evaluator_name=name,
            k=len(scores),
            per_trial_scores=scores,
            threshold=threshold,
        )
        for name, scores in by_evaluator.items()
    }

    return ReliabilityReport(k=k, threshold=threshold, evaluator_results=results)

"""
ExpenseAuditEnv — OpenEnv-compliant environment for AI agent training.

Implements the full OpenEnv interface:
  reset()  → Observation
  step()   → (Observation, float, bool, dict)
  state()  → dict
"""
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Action, Observation, Reward, CompanyPolicy,
    ExpenseReport, ViolationType
)
from env.data_generator import (
    generate_easy_report,
    generate_medium_report,
    generate_hard_report,
)
from env.reward import compute_reward


TASK_EASY = "easy"
TASK_MEDIUM = "medium"
TASK_HARD = "hard"


class ExpenseAuditEnv:
    """
    Expense Report Audit Environment.

    The agent acts as an AI auditor reviewing employee expense reports.
    It must identify policy violations, duplicate submissions, and fraud patterns.

    Action space:  Action (decision + list of FlaggedItems)
    Observation:   Observation (report + policy + optional history/vendor stats)
    Reward:        Dense shaped reward in [-1.0, 1.0]
    """

    metadata = {
        "name": "expense-audit-env",
        "version": "1.0.0",
        "tasks": [TASK_EASY, TASK_MEDIUM, TASK_HARD],
        "max_steps": 1,
    }

    def __init__(
        self,
        task: str = TASK_EASY,
        policy: Optional[CompanyPolicy] = None,
        seed: Optional[int] = None,
    ):
        assert task in (TASK_EASY, TASK_MEDIUM, TASK_HARD), (
            f"task must be one of {self.metadata['tasks']}, got '{task}'"
        )
        self.task = task
        self.policy = policy or CompanyPolicy()
        self.seed = seed

        # Internal state (populated on reset)
        self._current_report: Optional[ExpenseReport] = None
        self._history: List[ExpenseReport] = []
        self._vendor_stats: Dict[str, Any] = {}
        self._ground_truth: List[Tuple[str, ViolationType]] = []
        self._step_count: int = 0
        self._done: bool = False
        self._last_reward: Optional[Reward] = None
        self._rng = random.Random(seed)
        self._episode_count: int = 0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        # Advance seed per episode so each reset generates a different report
        # while remaining fully reproducible: same base seed → same sequence of episodes
        if self.seed is not None:
            random.seed(self.seed + self._episode_count)
        self._episode_count += 1

        self._step_count = 0
        self._done = False
        self._last_reward = None

        if self.task == TASK_EASY:
            report, gt = generate_easy_report(policy=self.policy)
            self._current_report = report
            self._history = []
            self._vendor_stats = {}
            self._ground_truth = gt
            task_description = (
                "Review this expense report and flag any items that violate company policy. "
                "Look for over-limit expenses and missing receipts."
            )

        elif self.task == TASK_MEDIUM:
            report, history, gt = generate_medium_report(policy=self.policy)
            self._current_report = report
            self._history = history
            self._vendor_stats = {}
            self._ground_truth = gt
            task_description = (
                "Review this expense report against the employee's submission history. "
                "Flag duplicate submissions, over-limit expenses, and policy violations."
            )

        else:  # TASK_HARD
            report, history, vendor_stats, gt = generate_hard_report(policy=self.policy)
            self._current_report = report
            self._history = history
            self._vendor_stats = vendor_stats
            self._ground_truth = gt
            task_description = (
                "Perform a deep fraud audit on this expense report. "
                "Identify split receipts, round-number fraud, vendor collusion patterns, "
                "and any other suspicious activity. Use history and vendor statistics provided."
            )

        return Observation(
            report=self._current_report,
            policy=self.policy,
            historical_reports=self._history,
            vendor_stats=self._vendor_stats,
            step_number=0,
            max_steps=1,
            task_description=task_description,
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one audit action.

        Args:
            action: Agent's audit decision with flagged items.

        Returns:
            (observation, reward_value, done, info)
        """
        assert not self._done, "Episode is done. Call reset() to start a new episode."
        assert self._current_report is not None, "Must call reset() before step()."

        reward_obj = compute_reward(action, self._ground_truth)
        self._last_reward = reward_obj
        self._step_count += 1
        self._done = True  # Single-step environment (one report per episode)

        # Terminal observation (same report, episode over)
        obs = Observation(
            report=self._current_report,
            policy=self.policy,
            historical_reports=self._history,
            vendor_stats=self._vendor_stats,
            step_number=self._step_count,
            max_steps=1,
            task_description="Episode complete.",
        )

        info = {
            "reward_detail": reward_obj.model_dump(),
            "ground_truth": [
                {"item_id": iid, "violation": vt.value}
                for iid, vt in self._ground_truth
            ],
            "task": self.task,
        }

        return obs, reward_obj.value, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return the full internal state (for debugging / checkpointing)."""
        return {
            "task": self.task,
            "step_count": self._step_count,
            "done": self._done,
            "current_report_id": (
                self._current_report.report_id if self._current_report else None
            ),
            "ground_truth": [
                {"item_id": iid, "violation": vt.value}
                for iid, vt in self._ground_truth
            ],
            "last_reward": (
                self._last_reward.model_dump() if self._last_reward else None
            ),
            "policy": self.policy.model_dump(),
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return a human-readable summary of the current observation."""
        if self._current_report is None:
            return "No active episode. Call reset() first."

        lines = [
            f"=== Expense Report {self._current_report.report_id} ===",
            f"Submitted by: {self._current_report.submitted_by.name} "
            f"({self._current_report.submitted_by.department})",
            f"Date: {self._current_report.submission_date}",
            f"Total: ${self._current_report.total_amount:.2f}",
            "",
            "Line Items:",
        ]
        for item in self._current_report.line_items:
            receipt = "✓" if item.has_receipt else "✗"
            lines.append(
                f"  [{item.item_id}] {item.date} | {item.category.value:15s} | "
                f"${item.amount:8.2f} | {item.vendor:30s} | receipt:{receipt}"
            )
        return "\n".join(lines)

    def score(self, action: Action) -> float:
        """Convenience: compute score without advancing episode state."""
        reward_obj = compute_reward(action, self._ground_truth)
        return reward_obj.value

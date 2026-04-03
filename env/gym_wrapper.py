"""
Gymnasium-compatible wrapper for the Expense Audit OpenEnv environment.

Allows using standard RL libraries (Stable-Baselines3, RLlib, Tianshou, etc.)
that expect the gym.Env interface.

Usage:
    from env.gym_wrapper import ExpenseAuditGymEnv
    import gymnasium as gym

    env = ExpenseAuditGymEnv(task="easy", seed=42)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

Note:
    Gym action/observation spaces use flat numpy arrays (agent sees a
    serialised JSON string in the observation, and submits a JSON string
    action). This makes the wrapper LLM-agent compatible without restricting
    the action space to fixed dimensions.
"""
from __future__ import annotations
import json
from typing import Any, Dict, Optional, Tuple

from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import Action, AuditDecision, FlaggedItem, ViolationType, CompanyPolicy

try:
    import gymnasium as gym
    import numpy as np
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False


class ExpenseAuditGymEnv:
    """
    Gymnasium-style wrapper around ExpenseAuditEnv.

    Observation: JSON string of the full Observation dict (variable length text).
    Action:      JSON string encoding {"decision": str, "flagged_items": [...]}

    This "text gym" pattern is standard for LLM-based RL environments
    (see WebArena, SciWorld, TextWorld).
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "expense-audit-v1"}

    def __init__(
        self,
        task: str = TASK_EASY,
        policy: Optional[CompanyPolicy] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        self._inner = ExpenseAuditEnv(task=task, policy=policy, seed=seed)
        self.render_mode = render_mode
        self._last_obs_dict: Optional[dict] = None

        if HAS_GYMNASIUM:
            # Text observation: unbounded string space
            self.observation_space = gym.spaces.Text(min_length=0, max_length=100_000)
            # Text action: unbounded string space
            self.action_space = gym.spaces.Text(min_length=2, max_length=10_000)
        else:
            self.observation_space = None
            self.action_space = None

    # ------------------------------------------------------------------
    # Core Gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Reset the environment.

        Returns:
            observation (str): JSON-encoded Observation dict.
            info (dict): empty dict (Gymnasium convention).
        """
        if seed is not None:
            self._inner.seed = seed
        obs = self._inner.reset()
        self._last_obs_dict = obs.model_dump()
        return json.dumps(self._last_obs_dict, default=str), {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Take a step.

        Args:
            action (str): JSON string with keys:
                - decision: "approve" | "flag" | "reject"
                - flagged_items: list of {item_id, violation_type, reason, confidence}
                - report_id: (optional, inferred if omitted)
                - overall_notes: (optional)

        Returns:
            observation (str): JSON-encoded terminal Observation.
            reward (float): Shaped reward in [-1.0, 1.0].
            terminated (bool): Always True (single-step episode).
            truncated (bool): Always False.
            info (dict): ground_truth, reward_detail, task.
        """
        parsed = self._parse_action(action)
        obs, reward, done, info = self._inner.step(parsed)
        obs_str = json.dumps(obs.model_dump(), default=str)
        return obs_str, reward, done, False, info

    def render(self) -> Optional[str]:
        if self.render_mode in ("human", "ansi"):
            rendered = self._inner.render()
            if self.render_mode == "human":
                print(rendered)
            return rendered
        return None

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def _parse_action(self, action_str: str) -> Action:
        """Parse a JSON action string into an Action model."""
        if isinstance(action_str, Action):
            return action_str

        try:
            data = json.loads(action_str)
        except (json.JSONDecodeError, TypeError):
            # Malformed action → treat as approve with no flags
            report_id = (
                self._last_obs_dict["report"]["report_id"]
                if self._last_obs_dict else "unknown"
            )
            return Action(report_id=report_id, decision=AuditDecision.APPROVE)

        report_id = data.get(
            "report_id",
            self._last_obs_dict["report"]["report_id"] if self._last_obs_dict else "unknown",
        )

        try:
            decision = AuditDecision(data.get("decision", "approve"))
        except ValueError:
            decision = AuditDecision.APPROVE

        flagged = []
        for item in data.get("flagged_items", []):
            try:
                vtype = ViolationType(item.get("violation_type", "suspicious_pattern"))
            except ValueError:
                vtype = ViolationType.SUSPICIOUS_PATTERN
            flagged.append(FlaggedItem(
                item_id=item.get("item_id", ""),
                violation_type=vtype,
                reason=item.get("reason", ""),
                confidence=float(item.get("confidence", 1.0)),
            ))

        return Action(
            report_id=report_id,
            decision=decision,
            flagged_items=flagged,
            overall_notes=data.get("overall_notes"),
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> ExpenseAuditEnv:
        """Access the underlying ExpenseAuditEnv."""
        return self._inner

    def action_from_dict(self, d: dict) -> str:
        """Helper: convert a dict to the JSON action string format."""
        return json.dumps(d)

    def obs_to_dict(self, obs_str: str) -> dict:
        """Helper: convert an observation string back to a dict."""
        return json.loads(obs_str)

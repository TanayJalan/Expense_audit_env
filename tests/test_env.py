"""
Tests for the Expense Audit OpenEnv environment.
Run with: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import (
    Action, FlaggedItem, AuditDecision, ViolationType,
    CompanyPolicy, Observation
)
from env.reward import compute_reward
from tasks.task_definitions import GRADERS, TASKS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def policy():
    return CompanyPolicy()

@pytest.fixture
def easy_env():
    return ExpenseAuditEnv(task=TASK_EASY, seed=42)

@pytest.fixture
def medium_env():
    return ExpenseAuditEnv(task=TASK_MEDIUM, seed=42)

@pytest.fixture
def hard_env():
    return ExpenseAuditEnv(task=TASK_HARD, seed=42)


# ---------------------------------------------------------------------------
# Core API tests
# ---------------------------------------------------------------------------

class TestOpenEnvSpec:
    """Verify the environment implements the full OpenEnv spec."""

    def test_reset_returns_observation(self, easy_env):
        obs = easy_env.reset()
        assert isinstance(obs, Observation)
        assert obs.report is not None
        assert obs.policy is not None
        assert obs.step_number == 0

    def test_step_returns_correct_tuple(self, easy_env):
        obs = easy_env.reset()
        action = Action(
            report_id=obs.report.report_id,
            decision=AuditDecision.APPROVE,
            flagged_items=[],
        )
        result = easy_env.step(action)
        assert len(result) == 4
        obs2, reward, done, info = result
        assert isinstance(obs2, Observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert done is True  # single-step env

    def test_state_returns_dict(self, easy_env):
        easy_env.reset()
        state = easy_env.state()
        assert isinstance(state, dict)
        assert "task" in state
        assert "ground_truth" in state
        assert "done" in state

    def test_reward_in_valid_range(self, easy_env):
        obs = easy_env.reset()
        action = Action(
            report_id=obs.report.report_id,
            decision=AuditDecision.FLAG,
            flagged_items=[],
        )
        _, reward, _, _ = easy_env.step(action)
        assert -1.0 <= reward <= 1.0

    def test_step_raises_after_done(self, easy_env):
        obs = easy_env.reset()
        action = Action(
            report_id=obs.report.report_id,
            decision=AuditDecision.APPROVE,
        )
        easy_env.step(action)
        with pytest.raises(AssertionError):
            easy_env.step(action)  # Episode done

    def test_reset_clears_done(self, easy_env):
        obs = easy_env.reset()
        action = Action(report_id=obs.report.report_id, decision=AuditDecision.APPROVE)
        easy_env.step(action)
        assert easy_env._done is True
        easy_env.reset()
        assert easy_env._done is False

    def test_all_three_tasks(self):
        for task in [TASK_EASY, TASK_MEDIUM, TASK_HARD]:
            env = ExpenseAuditEnv(task=task, seed=0)
            obs = env.reset()
            assert obs.task_description != ""


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

class TestRewardFunction:

    def test_perfect_recall_high_reward(self):
        """Flagging all violations gives positive reward."""
        gt = [("abc", ViolationType.OVER_LIMIT), ("def", ViolationType.MISSING_RECEIPT)]
        action = Action(
            report_id="r1",
            decision=AuditDecision.FLAG,
            flagged_items=[
                FlaggedItem(item_id="abc", violation_type=ViolationType.OVER_LIMIT, reason="over limit"),
                FlaggedItem(item_id="def", violation_type=ViolationType.MISSING_RECEIPT, reason="no receipt"),
            ]
        )
        reward = compute_reward(action, gt)
        assert reward.value > 0.5
        assert reward.true_positives == 2
        assert reward.false_positives == 0
        assert reward.false_negatives == 0

    def test_all_false_positives_negative_reward(self):
        """Hallucinating violations penalises the agent."""
        gt = []
        action = Action(
            report_id="r1",
            decision=AuditDecision.FLAG,
            flagged_items=[
                FlaggedItem(item_id="ghost1", violation_type=ViolationType.OVER_LIMIT, reason="imaginary"),
                FlaggedItem(item_id="ghost2", violation_type=ViolationType.DUPLICATE, reason="imaginary"),
            ]
        )
        reward = compute_reward(action, gt)
        assert reward.value < 0

    def test_no_violations_approve_rewarded(self):
        """Correctly approving a clean report gives positive reward."""
        gt = []
        action = Action(report_id="r1", decision=AuditDecision.APPROVE, flagged_items=[])
        reward = compute_reward(action, gt)
        assert reward.value >= 0

    def test_partial_credit_wrong_type(self):
        """Finding the right item with wrong type gives partial credit."""
        gt = [("abc", ViolationType.SPLIT_RECEIPT)]
        action = Action(
            report_id="r1",
            decision=AuditDecision.FLAG,
            flagged_items=[
                FlaggedItem(item_id="abc", violation_type=ViolationType.OVER_LIMIT, reason="wrong type"),
            ]
        )
        reward = compute_reward(action, gt)
        # Should be positive (found the item) but less than perfect
        assert reward.value > 0
        assert reward.value < 0.9

    def test_precision_recall_computed(self):
        gt = [("abc", ViolationType.OVER_LIMIT)]
        action = Action(
            report_id="r1",
            decision=AuditDecision.FLAG,
            flagged_items=[
                FlaggedItem(item_id="abc", violation_type=ViolationType.OVER_LIMIT, reason="correct"),
                FlaggedItem(item_id="xyz", violation_type=ViolationType.DUPLICATE, reason="wrong"),
            ]
        )
        reward = compute_reward(action, gt)
        assert reward.precision == pytest.approx(0.5, abs=0.01)
        assert reward.recall == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGraders:

    def test_easy_grader_perfect(self):
        gt = [("a1", ViolationType.OVER_LIMIT), ("b2", ViolationType.MISSING_RECEIPT)]
        action = Action(
            report_id="r",
            decision=AuditDecision.FLAG,
            flagged_items=[
                FlaggedItem(item_id="a1", violation_type=ViolationType.OVER_LIMIT, reason=""),
                FlaggedItem(item_id="b2", violation_type=ViolationType.MISSING_RECEIPT, reason=""),
            ]
        )
        score = GRADERS["easy"](action, gt)
        assert score == 1.0

    def test_easy_grader_zero(self):
        gt = [("a1", ViolationType.OVER_LIMIT)]
        action = Action(report_id="r", decision=AuditDecision.APPROVE, flagged_items=[])
        score = GRADERS["easy"](action, gt)
        assert score == 0.0

    def test_medium_grader_type_matters(self):
        gt = [("a1", ViolationType.DUPLICATE)]
        # Correct item, wrong type
        action_wrong = Action(
            report_id="r",
            decision=AuditDecision.FLAG,
            flagged_items=[FlaggedItem(item_id="a1", violation_type=ViolationType.OVER_LIMIT, reason="")]
        )
        # Correct item, correct type
        action_right = Action(
            report_id="r",
            decision=AuditDecision.FLAG,
            flagged_items=[FlaggedItem(item_id="a1", violation_type=ViolationType.DUPLICATE, reason="")]
        )
        score_wrong = GRADERS["medium"](action_wrong, gt)
        score_right = GRADERS["medium"](action_right, gt)
        assert score_right > score_wrong

    def test_hard_grader_split_pattern_partial(self):
        """Catching only one split-receipt item gives partial, not zero."""
        gt = [
            ("s1", ViolationType.SPLIT_RECEIPT),
            ("s2", ViolationType.SPLIT_RECEIPT),
        ]
        action_partial = Action(
            report_id="r",
            decision=AuditDecision.FLAG,
            flagged_items=[FlaggedItem(item_id="s1", violation_type=ViolationType.SPLIT_RECEIPT, reason="")]
        )
        action_full = Action(
            report_id="r",
            decision=AuditDecision.FLAG,
            flagged_items=[
                FlaggedItem(item_id="s1", violation_type=ViolationType.SPLIT_RECEIPT, reason=""),
                FlaggedItem(item_id="s2", violation_type=ViolationType.SPLIT_RECEIPT, reason=""),
            ]
        )
        score_partial = GRADERS["hard"](action_partial, gt)
        score_full = GRADERS["hard"](action_full, gt)
        assert 0.0 < score_partial < score_full

    def test_all_grader_scores_in_range(self):
        for task_name, grader in GRADERS.items():
            gt = [("x1", ViolationType.OVER_LIMIT)]
            action = Action(
                report_id="r",
                decision=AuditDecision.FLAG,
                flagged_items=[FlaggedItem(item_id="x1", violation_type=ViolationType.OVER_LIMIT, reason="")],
            )
            score = grader(action, gt)
            assert 0.0 <= score <= 1.0, f"{task_name} score out of range: {score}"


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_seed_same_report(self):
        """Same seed must produce identical reports — fully reproducible."""
        env1 = ExpenseAuditEnv(task=TASK_EASY, seed=99)
        env2 = ExpenseAuditEnv(task=TASK_EASY, seed=99)
        obs1 = env1.reset()
        obs2 = env2.reset()
        # IDs are now seeded so same seed = identical report
        assert obs1.report.report_id == obs2.report.report_id
        assert obs1.report.total_amount == obs2.report.total_amount
        assert [i.item_id for i in obs1.report.line_items] == \
               [i.item_id for i in obs2.report.line_items]

    def test_different_seeds_different_reports(self):
        env1 = ExpenseAuditEnv(task=TASK_EASY, seed=1)
        env2 = ExpenseAuditEnv(task=TASK_EASY, seed=2)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.report.report_id != obs2.report.report_id

    def test_episode_advance_produces_variety(self):
        env = ExpenseAuditEnv(task=TASK_EASY, seed=42)
        obs1 = env.reset()
        action = Action(report_id=obs1.report.report_id, decision=AuditDecision.APPROVE)
        env.step(action)
        obs2 = env.reset()
        assert obs1.report.report_id != obs2.report.report_id

    def test_task_descriptions_nonempty(self):
        for task in [TASK_EASY, TASK_MEDIUM, TASK_HARD]:
            env = ExpenseAuditEnv(task=task)
            obs = env.reset()
            assert len(obs.task_description) > 10

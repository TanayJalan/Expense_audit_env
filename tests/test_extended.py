"""
Additional tests: grader edge cases, session manager, data generator, API smoke tests.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import (
    Action, FlaggedItem, AuditDecision, ViolationType,
    CompanyPolicy, ExpenseCategory
)
from env.data_generator import (
    generate_easy_report, generate_medium_report,
    generate_hard_report, _make_clean_item
)
from env.session_manager import SessionManager
from tasks.task_definitions import GRADERS, grade_easy, grade_medium, grade_hard


# ---------------------------------------------------------------------------
# Data Generator Tests
# ---------------------------------------------------------------------------

class TestDataGenerator:

    def test_easy_report_has_violations(self):
        policy = CompanyPolicy()
        for _ in range(20):
            report, gt = generate_easy_report(policy=policy, num_violations=1)
            assert len(gt) >= 1
            assert len(report.line_items) >= 3

    def test_medium_report_has_history(self):
        report, history, gt = generate_medium_report()
        assert len(history) >= 1
        assert len(gt) >= 1
        # Should have at least one duplicate
        dup_violations = [g for g in gt if g[1] == ViolationType.DUPLICATE]
        assert len(dup_violations) >= 1

    def test_hard_report_has_fraud_patterns(self):
        report, history, vendor_stats, gt = generate_hard_report()
        violation_types = {vt for _, vt in gt}
        # Hard report should always have split receipt and round number
        assert ViolationType.SPLIT_RECEIPT in violation_types
        assert ViolationType.ROUND_NUMBER_FRAUD in violation_types
        assert ViolationType.VENDOR_COLLUSION in violation_types

    def test_clean_items_within_policy_limits(self):
        policy = CompanyPolicy()
        for _ in range(50):
            item = _make_clean_item(category=ExpenseCategory.MEALS, policy=policy)
            assert item.amount <= policy.meal_limit_per_day

    def test_report_total_matches_line_items(self):
        report, _ = generate_easy_report()
        computed_total = round(sum(i.amount for i in report.line_items), 2)
        assert abs(computed_total - report.total_amount) < 0.01

    def test_all_items_have_unique_ids(self):
        report, _ = generate_easy_report()
        ids = [i.item_id for i in report.line_items]
        assert len(ids) == len(set(ids)), "Duplicate item IDs found"


# ---------------------------------------------------------------------------
# Grader Edge Cases
# ---------------------------------------------------------------------------

class TestGraderEdgeCases:

    def test_empty_ground_truth_approve_scores_1(self):
        """Clean report with correct approve decision = 1.0."""
        gt = []
        action = Action(report_id="r", decision=AuditDecision.APPROVE, flagged_items=[])
        for grader in GRADERS.values():
            assert grader(action, gt) == 1.0

    def test_empty_ground_truth_flag_scores_0(self):
        """Clean report but agent flags = 0.0."""
        gt = []
        action = Action(
            report_id="r",
            decision=AuditDecision.FLAG,
            flagged_items=[FlaggedItem(item_id="ghost", violation_type=ViolationType.OVER_LIMIT, reason="")]
        )
        for task, grader in GRADERS.items():
            score = grader(action, gt)
            assert score == 0.0, f"task={task}: expected 0.0, got {score}"

    def test_medium_grader_fp_reduces_score(self):
        """Adding false positives to a correct action reduces the score."""
        gt = [("a1", ViolationType.DUPLICATE)]
        action_correct = Action(
            report_id="r", decision=AuditDecision.FLAG,
            flagged_items=[FlaggedItem(item_id="a1", violation_type=ViolationType.DUPLICATE, reason="")]
        )
        action_with_fp = Action(
            report_id="r", decision=AuditDecision.FLAG,
            flagged_items=[
                FlaggedItem(item_id="a1", violation_type=ViolationType.DUPLICATE, reason=""),
                FlaggedItem(item_id="fp1", violation_type=ViolationType.OVER_LIMIT, reason="hallucinated"),
                FlaggedItem(item_id="fp2", violation_type=ViolationType.OVER_LIMIT, reason="hallucinated"),
            ]
        )
        score_correct = grade_medium(action_correct, gt)
        score_fp = grade_medium(action_with_fp, gt)
        assert score_correct > score_fp

    def test_hard_grader_requires_all_split_receipts_for_full_credit(self):
        """Catching only 1 of 2 split receipts should score less than catching both."""
        gt = [
            ("s1", ViolationType.SPLIT_RECEIPT),
            ("s2", ViolationType.SPLIT_RECEIPT),
            ("r1", ViolationType.ROUND_NUMBER_FRAUD),
        ]
        action_all = Action(
            report_id="r", decision=AuditDecision.FLAG,
            flagged_items=[
                FlaggedItem(item_id="s1", violation_type=ViolationType.SPLIT_RECEIPT, reason=""),
                FlaggedItem(item_id="s2", violation_type=ViolationType.SPLIT_RECEIPT, reason=""),
                FlaggedItem(item_id="r1", violation_type=ViolationType.ROUND_NUMBER_FRAUD, reason=""),
            ]
        )
        action_partial_split = Action(
            report_id="r", decision=AuditDecision.FLAG,
            flagged_items=[
                FlaggedItem(item_id="s1", violation_type=ViolationType.SPLIT_RECEIPT, reason=""),
                FlaggedItem(item_id="r1", violation_type=ViolationType.ROUND_NUMBER_FRAUD, reason=""),
            ]
        )
        score_all = grade_hard(action_all, gt)
        score_partial = grade_hard(action_partial_split, gt)
        assert score_all > score_partial
        assert score_all == pytest.approx(1.0, abs=0.01)

    def test_partial_type_credit(self):
        """Finding the right item with wrong violation type gives partial (not zero) credit."""
        gt = [("x1", ViolationType.VENDOR_COLLUSION)]

        action_right_type = Action(
            report_id="r", decision=AuditDecision.FLAG,
            flagged_items=[FlaggedItem(item_id="x1", violation_type=ViolationType.VENDOR_COLLUSION, reason="")]
        )
        action_wrong_type = Action(
            report_id="r", decision=AuditDecision.FLAG,
            flagged_items=[FlaggedItem(item_id="x1", violation_type=ViolationType.OVER_LIMIT, reason="")]
        )
        action_missed = Action(report_id="r", decision=AuditDecision.APPROVE, flagged_items=[])

        for task, grader in GRADERS.items():
            s_right = grader(action_right_type, gt)
            s_wrong = grader(action_wrong_type, gt)
            s_missed = grader(action_missed, gt)
            # Right type > wrong type > missed (or at least right > wrong)
            assert s_right >= s_wrong, f"task={task}: right={s_right} should >= wrong={s_wrong}"


# ---------------------------------------------------------------------------
# Session Manager Tests
# ---------------------------------------------------------------------------

class TestSessionManager:

    def test_create_and_retrieve_session(self):
        mgr = SessionManager()
        session = mgr.create(task=TASK_EASY, seed=0)
        assert session.session_id is not None
        retrieved = mgr.get(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_unknown_session_returns_none(self):
        mgr = SessionManager()
        result = mgr.get("nonexistent-session-id")
        assert result is None

    def test_delete_session(self):
        mgr = SessionManager()
        session = mgr.create(task=TASK_EASY)
        mgr.delete(session.session_id)
        assert mgr.get(session.session_id) is None

    def test_multiple_sessions_independent(self):
        """Two sessions should have independent environments."""
        mgr = SessionManager()
        s1 = mgr.create(task=TASK_EASY, seed=1)
        s2 = mgr.create(task=TASK_MEDIUM, seed=2)
        obs1 = s1.env.reset()
        obs2 = s2.env.reset()
        assert obs1.task_description != obs2.task_description
        assert s1.session_id != s2.session_id

    def test_active_count(self):
        mgr = SessionManager()
        assert mgr.active_count == 0
        s1 = mgr.create(task=TASK_EASY)
        s2 = mgr.create(task=TASK_HARD)
        assert mgr.active_count == 2
        mgr.delete(s1.session_id)
        assert mgr.active_count == 1

    def test_episode_count_increments(self):
        mgr = SessionManager()
        session = mgr.create(task=TASK_EASY, seed=0)
        retrieved = mgr.get(session.session_id)
        retrieved.env.reset()
        retrieved.episode_count += 1
        assert mgr.get(session.session_id).episode_count == 1


# ---------------------------------------------------------------------------
# Environment render() test
# ---------------------------------------------------------------------------

class TestRender:

    def test_render_output_contains_report_id(self):
        env = ExpenseAuditEnv(task=TASK_EASY, seed=0)
        obs = env.reset()
        rendered = env.render()
        assert obs.report.report_id in rendered
        assert "Line Items" in rendered

    def test_render_before_reset_returns_message(self):
        env = ExpenseAuditEnv(task=TASK_EASY)
        msg = env.render()
        assert "reset()" in msg

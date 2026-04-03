"""
Shared pytest fixtures for the Expense Audit OpenEnv test suite.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import (
    Action, FlaggedItem, AuditDecision, ViolationType, CompanyPolicy
)


@pytest.fixture(scope="session")
def default_policy():
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


@pytest.fixture
def approve_action(easy_env):
    obs = easy_env.reset()
    return Action(report_id=obs.report.report_id, decision=AuditDecision.APPROVE)


@pytest.fixture
def flag_first_item_action(easy_env):
    obs = easy_env.reset()
    return Action(
        report_id=obs.report.report_id,
        decision=AuditDecision.FLAG,
        flagged_items=[
            FlaggedItem(
                item_id=obs.report.line_items[0].item_id,
                violation_type=ViolationType.OVER_LIMIT,
                reason="test flag",
            )
        ]
    )

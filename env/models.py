"""
Typed Pydantic models for the Expense Audit OpenEnv environment.
Defines Observation, Action, and Reward schemas.
"""
from __future__ import annotations
import random
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from datetime import date


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExpenseCategory(str, Enum):
    MEALS = "meals"
    TRAVEL = "travel"
    LODGING = "lodging"
    EQUIPMENT = "equipment"
    ENTERTAINMENT = "entertainment"
    TRAINING = "training"
    OTHER = "other"


class ViolationType(str, Enum):
    OVER_LIMIT = "over_limit"
    DUPLICATE = "duplicate"
    MISSING_RECEIPT = "missing_receipt"
    UNAUTHORIZED_CATEGORY = "unauthorized_category"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    SPLIT_RECEIPT = "split_receipt"
    ROUND_NUMBER_FRAUD = "round_number_fraud"
    VENDOR_COLLUSION = "vendor_collusion"


class AuditDecision(str, Enum):
    APPROVE = "approve"
    FLAG = "flag"
    REJECT = "reject"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ExpenseLineItem(BaseModel):
    item_id: str = Field(default_factory=lambda: format(random.randint(0, 0xFFFFFFFF), '08x'))
    date: str                          # ISO date string YYYY-MM-DD
    category: ExpenseCategory
    vendor: str
    amount: float
    description: str
    has_receipt: bool
    currency: str = "USD"


class CompanyPolicy(BaseModel):
    meal_limit_per_day: float = 75.0
    lodging_limit_per_night: float = 250.0
    travel_limit_per_trip: float = 1500.0
    equipment_limit_per_item: float = 500.0
    entertainment_limit_per_event: float = 150.0
    receipt_required_above: float = 25.0
    allowed_categories: List[ExpenseCategory] = list(ExpenseCategory)
    max_advance_days: int = 30          # Cannot submit receipts >30 days old


class EmployeeHistory(BaseModel):
    employee_id: str
    name: str
    department: str
    past_violations: int = 0
    past_reports_count: int = 0
    risk_score: float = Field(0.0, ge=0.0, le=1.0)


class ExpenseReport(BaseModel):
    report_id: str = Field(default_factory=lambda: format(random.randint(0, 0xFFFFFFFF), '08x'))
    submitted_by: EmployeeHistory
    submission_date: str               # ISO date string
    line_items: List[ExpenseLineItem]
    total_amount: float
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation  (what the agent sees each step)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    report: ExpenseReport
    policy: CompanyPolicy
    # For medium/hard tasks: historical reports from the same employee
    historical_reports: List[ExpenseReport] = Field(default_factory=list)
    # For hard tasks: org-wide vendor usage stats
    vendor_stats: Dict[str, Any] = Field(default_factory=dict)
    step_number: int = 0
    max_steps: int = 1
    task_description: str = ""


# ---------------------------------------------------------------------------
# Action  (what the agent submits)
# ---------------------------------------------------------------------------

class FlaggedItem(BaseModel):
    item_id: str
    violation_type: ViolationType
    reason: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class Action(BaseModel):
    report_id: str
    decision: AuditDecision
    flagged_items: List[FlaggedItem] = Field(default_factory=list)
    overall_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Reward  (returned alongside observation after each step)
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    value: float = Field(0.0, ge=-1.0, le=1.0)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""

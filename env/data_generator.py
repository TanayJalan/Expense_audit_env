"""
Synthetic data generator for expense reports.
Generates realistic reports with controllable violation injection.
"""
from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from env.models import (
    ExpenseReport, ExpenseLineItem, EmployeeHistory,
    ExpenseCategory, CompanyPolicy, ViolationType
)

# ---------------------------------------------------------------------------
# Realistic seed data
# ---------------------------------------------------------------------------

VENDORS = {
    ExpenseCategory.MEALS: [
        "Chipotle", "Olive Garden", "Shake Shack", "Nobu", "The Capital Grille",
        "Panera Bread", "Five Guys", "Legal Sea Foods", "Ruth's Chris",
    ],
    ExpenseCategory.TRAVEL: [
        "Delta Airlines", "United Airlines", "American Airlines",
        "Uber", "Lyft", "Hertz Car Rental", "Enterprise",
    ],
    ExpenseCategory.LODGING: [
        "Marriott", "Hilton", "Hyatt", "Hampton Inn", "Airbnb",
        "Holiday Inn", "Westin",
    ],
    ExpenseCategory.EQUIPMENT: [
        "Amazon Business", "Best Buy", "B&H Photo", "Staples", "Office Depot",
    ],
    ExpenseCategory.ENTERTAINMENT: [
        "AMC Theaters", "Dave & Busters", "TopGolf", "Eventbrite",
    ],
    ExpenseCategory.TRAINING: [
        "Coursera", "Udemy", "O'Reilly Media", "Pluralsight", "LinkedIn Learning",
    ],
    ExpenseCategory.OTHER: [
        "FedEx", "UPS", "USPS", "Walgreens",
    ],
}

DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Product", "Legal"]
FIRST_NAMES = ["Alice", "Bob", "Carlos", "Diana", "Eve", "Frank", "Grace", "Hiro", "Iris", "James"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]


def _random_date(days_back: int = 30) -> str:
    base = datetime.today() - timedelta(days=random.randint(1, days_back))
    return base.strftime("%Y-%m-%d")


def _make_employee(risk_level: float = 0.0) -> EmployeeHistory:
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    return EmployeeHistory(
        employee_id=str(uuid.uuid4())[:8],
        name=name,
        department=random.choice(DEPARTMENTS),
        past_violations=int(risk_level * 5),
        past_reports_count=random.randint(1, 50),
        risk_score=risk_level,
    )


def _make_clean_item(
    category: Optional[ExpenseCategory] = None,
    policy: Optional[CompanyPolicy] = None,
) -> ExpenseLineItem:
    if policy is None:
        policy = CompanyPolicy()
    if category is None:
        category = random.choice(list(ExpenseCategory))

    limits = {
        ExpenseCategory.MEALS: policy.meal_limit_per_day * 0.7,
        ExpenseCategory.LODGING: policy.lodging_limit_per_night * 0.8,
        ExpenseCategory.TRAVEL: policy.travel_limit_per_trip * 0.5,
        ExpenseCategory.EQUIPMENT: policy.equipment_limit_per_item * 0.6,
        ExpenseCategory.ENTERTAINMENT: policy.entertainment_limit_per_event * 0.7,
        ExpenseCategory.TRAINING: 200.0,
        ExpenseCategory.OTHER: 50.0,
    }
    max_amount = limits.get(category, 100.0)
    amount = round(random.uniform(10.0, max_amount), 2)

    vendor = random.choice(VENDORS.get(category, VENDORS[ExpenseCategory.OTHER]))
    return ExpenseLineItem(
        date=_random_date(),
        category=category,
        vendor=vendor,
        amount=amount,
        description=f"{category.value.title()} - {vendor}",
        has_receipt=True,
    )


# ---------------------------------------------------------------------------
# Violation injectors  (return item + ground-truth ViolationType)
# ---------------------------------------------------------------------------

def _inject_over_limit(category: ExpenseCategory, policy: CompanyPolicy) -> Tuple[ExpenseLineItem, ViolationType]:
    limits = {
        ExpenseCategory.MEALS: policy.meal_limit_per_day,
        ExpenseCategory.LODGING: policy.lodging_limit_per_night,
        ExpenseCategory.EQUIPMENT: policy.equipment_limit_per_item,
        ExpenseCategory.ENTERTAINMENT: policy.entertainment_limit_per_event,
    }
    limit = limits.get(category, 100.0)
    amount = round(random.uniform(limit * 1.1, limit * 2.0), 2)
    vendor = random.choice(VENDORS.get(category, VENDORS[ExpenseCategory.OTHER]))
    item = ExpenseLineItem(
        date=_random_date(),
        category=category,
        vendor=vendor,
        amount=amount,
        description=f"{category.value.title()} - {vendor}",
        has_receipt=True,
    )
    return item, ViolationType.OVER_LIMIT


def _inject_missing_receipt(category: ExpenseCategory) -> Tuple[ExpenseLineItem, ViolationType]:
    amount = round(random.uniform(30.0, 200.0), 2)
    vendor = random.choice(VENDORS.get(category, VENDORS[ExpenseCategory.OTHER]))
    item = ExpenseLineItem(
        date=_random_date(),
        category=category,
        vendor=vendor,
        amount=amount,
        description=f"{category.value.title()} - {vendor}",
        has_receipt=False,
    )
    return item, ViolationType.MISSING_RECEIPT


def _inject_round_number(category: ExpenseCategory) -> Tuple[ExpenseLineItem, ViolationType]:
    # Suspiciously round numbers (fraud signal)
    amount = float(random.choice([100, 200, 250, 500, 1000]))
    vendor = random.choice(VENDORS.get(category, VENDORS[ExpenseCategory.OTHER]))
    item = ExpenseLineItem(
        date=_random_date(),
        category=category,
        vendor=vendor,
        amount=amount,
        description=f"{category.value.title()} - {vendor}",
        has_receipt=True,
    )
    return item, ViolationType.ROUND_NUMBER_FRAUD


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def generate_easy_report(
    policy: Optional[CompanyPolicy] = None,
    num_violations: Optional[int] = None,
    clean_probability: float = 0.25,
) -> Tuple[ExpenseReport, List[Tuple[str, ViolationType]]]:
    """
    Easy task: 4-6 line items with 0-2 clear policy violations.

    25% of reports are fully clean (ground_truth=[]) so agents must learn
    WHEN to approve, not just reflexively flag.

    Args:
        num_violations: Override number of violations (None = random 0-2).
        clean_probability: Probability of generating a fully clean report (default 0.25).

    Returns: (report, ground_truth) where ground_truth = [(item_id, ViolationType), ...]
    """
    if policy is None:
        policy = CompanyPolicy()

    employee = _make_employee(risk_level=0.1)
    items = []
    ground_truth: List[Tuple[str, ViolationType]] = []

    # Decide how many violations to inject
    if num_violations is not None:
        n_violations = num_violations
    elif random.random() < clean_probability:
        n_violations = 0   # Clean report — agent should APPROVE
    else:
        n_violations = random.randint(1, 2)

    # 3-5 clean items
    for _ in range(random.randint(3, 5)):
        items.append(_make_clean_item(policy=policy))

    # Inject violations
    all_injectors = [
        lambda: _inject_over_limit(ExpenseCategory.MEALS, policy),
        lambda: _inject_missing_receipt(ExpenseCategory.EQUIPMENT),
        lambda: _inject_over_limit(ExpenseCategory.LODGING, policy),
        lambda: _inject_over_limit(ExpenseCategory.ENTERTAINMENT, policy),
        lambda: _inject_missing_receipt(ExpenseCategory.TRAVEL),
    ]
    for injector in random.sample(all_injectors, min(n_violations, len(all_injectors))):
        item, vtype = injector()
        items.append(item)
        ground_truth.append((item.item_id, vtype))

    random.shuffle(items)
    total = round(sum(i.amount for i in items), 2)
    report = ExpenseReport(
        submitted_by=employee,
        submission_date=_random_date(5),
        line_items=items,
        total_amount=total,
    )
    return report, ground_truth


def generate_medium_report(
    policy: Optional[CompanyPolicy] = None,
) -> Tuple[ExpenseReport, List[ExpenseReport], List[Tuple[str, ViolationType]]]:
    """
    Medium task: detect duplicate submissions across historical reports.
    Returns (current_report, history, ground_truth).
    """
    if policy is None:
        policy = CompanyPolicy()

    employee = _make_employee(risk_level=0.3)
    # Build a history of past reports
    history = []
    historical_items_pool = []

    for _ in range(3):
        past_items = [_make_clean_item(policy=policy) for _ in range(random.randint(2, 4))]
        historical_items_pool.extend(past_items)
        total = round(sum(i.amount for i in past_items), 2)
        history.append(ExpenseReport(
            submitted_by=employee,
            submission_date=_random_date(60),
            line_items=past_items,
            total_amount=total,
        ))

    # Current report: mix of new clean items + duplicates from history
    current_items = [_make_clean_item(policy=policy) for _ in range(2)]
    ground_truth: List[Tuple[str, ViolationType]] = []

    # Pick 1-2 items from history to duplicate
    dupes = random.sample(historical_items_pool, min(2, len(historical_items_pool)))
    for old_item in dupes:
        dup = ExpenseLineItem(
            date=old_item.date,
            category=old_item.category,
            vendor=old_item.vendor,
            amount=old_item.amount,
            description=old_item.description,
            has_receipt=old_item.has_receipt,
        )
        current_items.append(dup)
        ground_truth.append((dup.item_id, ViolationType.DUPLICATE))

    # Also add a subtle over-limit
    item, vtype = _inject_over_limit(ExpenseCategory.ENTERTAINMENT, policy)
    current_items.append(item)
    ground_truth.append((item.item_id, vtype))

    random.shuffle(current_items)
    total = round(sum(i.amount for i in current_items), 2)
    current_report = ExpenseReport(
        submitted_by=employee,
        submission_date=_random_date(5),
        line_items=current_items,
        total_amount=total,
    )
    return current_report, history, ground_truth


def generate_hard_report(
    policy: Optional[CompanyPolicy] = None,
) -> Tuple[ExpenseReport, List[ExpenseReport], Dict, List[Tuple[str, ViolationType]]]:
    """
    Hard task: detect coordinated fraud patterns:
      - split receipts (same vendor, same day, slightly under limit each)
      - round-number anomalies
      - vendor collusion (same vendor used suspiciously often vs org avg)
    Returns (report, history, vendor_stats, ground_truth).
    """
    if policy is None:
        policy = CompanyPolicy()

    employee = _make_employee(risk_level=0.7)
    items = []
    ground_truth: List[Tuple[str, ViolationType]] = []

    # 2 clean items
    for _ in range(2):
        items.append(_make_clean_item(policy=policy))

    # Split receipt pattern: 2 items same vendor same day, each just under meal limit
    split_date = _random_date(10)
    split_vendor = "The Capital Grille"
    limit = policy.meal_limit_per_day
    for _ in range(2):
        split_item = ExpenseLineItem(
            date=split_date,
            category=ExpenseCategory.MEALS,
            vendor=split_vendor,
            amount=round(limit * 0.92, 2),  # each just under limit
            description=f"Meals - {split_vendor}",
            has_receipt=True,
        )
        items.append(split_item)
        ground_truth.append((split_item.item_id, ViolationType.SPLIT_RECEIPT))

    # Round number fraud
    item, vtype = _inject_round_number(ExpenseCategory.EQUIPMENT)
    items.append(item)
    ground_truth.append((item.item_id, vtype))

    # Vendor collusion: a vendor that appears far more than org average
    collude_vendor = "ShellCo Consulting"
    collude_item = ExpenseLineItem(
        date=_random_date(15),
        category=ExpenseCategory.OTHER,
        vendor=collude_vendor,
        amount=round(random.uniform(400, 480), 2),
        description=f"Services - {collude_vendor}",
        has_receipt=True,
    )
    items.append(collude_item)
    ground_truth.append((collude_item.item_id, ViolationType.VENDOR_COLLUSION))

    # Vendor stats: org-wide vendor usage (collude_vendor is suspicious outlier)
    vendor_stats = {
        "org_avg_vendor_frequency": {"ShellCo Consulting": 8.5},  # times per month org-wide
        "employee_vendor_frequency": {"ShellCo Consulting": 4},   # this employee this month
        "flagged_vendors": [],
        "split_receipt_threshold": 0.9,  # if two items same vendor/day > 90% of limit each
    }

    # Build some historical context
    history = []
    for _ in range(2):
        past_items = [_make_clean_item(policy=policy) for _ in range(3)]
        total = round(sum(i.amount for i in past_items), 2)
        history.append(ExpenseReport(
            submitted_by=employee,
            submission_date=_random_date(60),
            line_items=past_items,
            total_amount=total,
        ))

    random.shuffle(items)
    total = round(sum(i.amount for i in items), 2)
    report = ExpenseReport(
        submitted_by=employee,
        submission_date=_random_date(3),
        line_items=items,
        total_amount=total,
    )
    return report, history, vendor_stats, ground_truth


from typing import Dict

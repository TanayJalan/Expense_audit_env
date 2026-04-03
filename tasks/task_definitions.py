"""
Task definitions and grader functions for the Expense Audit environment.

Each task:
  - has a concrete objective
  - defines a programmatic grader returning a score in [0.0, 1.0]
  - graders are deterministic and reproducible
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Callable

from env.models import Action, ViolationType, AuditDecision
from env.reward import compute_reward


# ---------------------------------------------------------------------------
# Task descriptor
# ---------------------------------------------------------------------------

@dataclass
class TaskDefinition:
    name: str
    difficulty: str          # easy | medium | hard
    description: str
    objective: str
    expected_score_random: float   # approx score of a random agent
    expected_score_frontier: float # approx score of a frontier LLM
    display_name: str = ""   # Human-readable title (defaults to name if empty)


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def grade_easy(action: Action, ground_truth: List[Tuple[str, ViolationType]]) -> float:
    """
    Easy grader: reward correct identification of simple policy violations.
    Score = F1 of correctly flagged items (type-agnostic: just get the right items).

    Returns score in [0.0, 1.0].
    """
    if not ground_truth:
        # No violations — agent should approve
        return 1.0 if action.decision == AuditDecision.APPROVE else 0.0

    gt_ids = {iid for iid, _ in ground_truth}
    flagged_ids = {f.item_id for f in action.flagged_items}

    tp = len(gt_ids & flagged_ids)
    fp = len(flagged_ids - gt_ids)
    fn = len(gt_ids - flagged_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(f1, 4)


def grade_medium(action: Action, ground_truth: List[Tuple[str, ViolationType]]) -> float:
    """
    Medium grader: must correctly identify BOTH the item AND the violation type.
    Partial credit if item is found but wrong type.

    Returns score in [0.0, 1.0].
    """
    if not ground_truth:
        return 1.0 if action.decision == AuditDecision.APPROVE else 0.0

    gt_dict: Dict[str, ViolationType] = {iid: vt for iid, vt in ground_truth}
    flagged_dict: Dict[str, ViolationType] = {f.item_id: f.violation_type for f in action.flagged_items}

    score_sum = 0.0
    for item_id, vtype in gt_dict.items():
        if item_id in flagged_dict:
            if flagged_dict[item_id] == vtype:
                score_sum += 1.0
            else:
                score_sum += 0.4  # Partial: found the item, wrong type
        # else: 0 for miss

    # False positive penalty
    fp_count = sum(1 for iid in flagged_dict if iid not in gt_dict)
    fp_penalty = 0.2 * fp_count

    raw = (score_sum / len(gt_dict)) - fp_penalty
    return round(max(0.0, min(1.0, raw)), 4)


def grade_hard(action: Action, ground_truth: List[Tuple[str, ViolationType]]) -> float:
    """
    Hard grader: must identify fraud patterns with correct violation types.
    Pattern-level scoring: groups split-receipts as a single pattern (must catch both).

    Returns score in [0.0, 1.0].
    """
    if not ground_truth:
        return 1.0 if action.decision == AuditDecision.APPROVE else 0.0

    gt_dict: Dict[str, ViolationType] = {iid: vt for iid, vt in ground_truth}
    flagged_dict: Dict[str, ViolationType] = {f.item_id: f.violation_type for f in action.flagged_items}

    # Group split-receipt items — must catch ALL items in a split pattern for full credit
    split_ids = [iid for iid, vt in gt_dict.items() if vt == ViolationType.SPLIT_RECEIPT]
    other_violations = [(iid, vt) for iid, vt in gt_dict.items() if vt != ViolationType.SPLIT_RECEIPT]

    score = 0.0
    max_score = 0.0

    # Split receipt pattern (worth 2.0 as it's a compound pattern)
    if split_ids:
        max_score += 2.0
        caught = sum(1 for iid in split_ids if iid in flagged_dict)
        if caught == len(split_ids):
            score += 2.0  # Full credit: caught the whole pattern
        elif caught > 0:
            score += 0.8  # Partial: noticed something suspicious

    # Other violations (each worth 1.0)
    for iid, vtype in other_violations:
        max_score += 1.0
        if iid in flagged_dict:
            if flagged_dict[iid] == vtype:
                score += 1.0
            else:
                score += 0.3  # Noticed the item, missed the type

    # False positive penalty
    fp_count = sum(1 for iid in flagged_dict if iid not in gt_dict)
    fp_penalty = 0.25 * fp_count * (max_score / max(len(gt_dict), 1))

    raw = (score / max_score) - (fp_penalty / max_score) if max_score > 0 else 0.0
    return round(max(0.0, min(1.0, raw)), 4)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        name="Basic Policy Violation Detection",
        difficulty="easy",
        description=(
            "Given a single expense report and the company policy, "
            "identify line items that exceed per-category spending limits "
            "or are missing required receipts."
        ),
        objective=(
            "Flag all expense items that violate company policy. "
            "Items to flag: amounts exceeding meal ($75/day), lodging ($250/night), "
            "equipment ($500/item), or entertainment ($150/event) limits; "
            "and items over $25 without a receipt."
        ),
        expected_score_random=0.05,
        expected_score_frontier=0.85,
        display_name="Basic Policy Violation Detection",
    ),
    "medium": TaskDefinition(
        name="Duplicate Submission & Multi-Violation Detection",
        difficulty="medium",
        description=(
            "Given a current expense report and the employee's full submission history, "
            "detect duplicate line items that were already reimbursed in a prior report, "
            "in addition to standard policy violations."
        ),
        objective=(
            "Flag duplicate expenses (same vendor, date, amount appearing in a previous report) "
            "and all other policy violations. You must identify the specific violation type for each flagged item."
        ),
        expected_score_random=0.02,
        expected_score_frontier=0.70,
        display_name="Duplicate Submission & Multi-Violation Detection",
    ),
    "hard": TaskDefinition(
        name="Coordinated Fraud Pattern Detection",
        difficulty="hard",
        description=(
            "Identify sophisticated fraud patterns in an expense report: "
            "split receipts (same vendor/day, each just under the limit), "
            "round-number anomalies, and vendor collusion signals "
            "(a vendor appearing far more than org average)."
        ),
        objective=(
            "Detect all fraud signals. Split receipts: two items same vendor/day each >85% of category limit. "
            "Round-number fraud: exact round amounts ($100, $200, $500, $1000) in categories where this is unusual. "
            "Vendor collusion: a vendor whose frequency for this employee is anomalously high vs org average. "
            "You must specify the correct ViolationType for each flagged item."
        ),
        expected_score_random=0.01,
        expected_score_frontier=0.55,
        display_name="Coordinated Fraud Pattern Detection",
    ),
}

GRADERS: Dict[str, Callable] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

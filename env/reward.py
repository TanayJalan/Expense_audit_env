"""
Reward computation for the Expense Audit environment.
Provides dense, shaped rewards with partial progress signals — not just binary end-of-episode.
"""
from __future__ import annotations
from typing import List, Tuple, Dict

from env.models import Action, Reward, ViolationType, AuditDecision, FlaggedItem


def compute_reward(
    action: Action,
    ground_truth: List[Tuple[str, ViolationType]],
    policy_weight: float = 1.0,
) -> Reward:
    """
    Core reward function.

    Scoring philosophy:
    - True positives:  +reward (scaled by violation severity)
    - False positives: -penalty (penalise hallucinating violations)
    - False negatives: -penalty (penalise missing real violations)
    - Decision alignment: small bonus/penalty for overall approve/reject decision

    Returns a Reward with value in [-1.0, 1.0].
    """
    gt_dict: Dict[str, ViolationType] = {item_id: vtype for item_id, vtype in ground_truth}
    flagged_ids = {f.item_id: f for f in action.flagged_items}

    # Severity weights — more serious violations carry higher reward/penalty
    SEVERITY = {
        ViolationType.OVER_LIMIT: 1.0,
        ViolationType.MISSING_RECEIPT: 0.8,
        ViolationType.DUPLICATE: 1.0,
        ViolationType.UNAUTHORIZED_CATEGORY: 0.9,
        ViolationType.SUSPICIOUS_PATTERN: 1.0,
        ViolationType.SPLIT_RECEIPT: 1.2,
        ViolationType.ROUND_NUMBER_FRAUD: 1.1,
        ViolationType.VENDOR_COLLUSION: 1.3,
    }

    tp = 0
    fp = 0
    fn = 0
    tp_score = 0.0
    fp_penalty = 0.0
    fn_penalty = 0.0
    breakdown: Dict[str, float] = {}

    # True positives: flagged and correct
    for item_id, flag in flagged_ids.items():
        if item_id in gt_dict:
            severity = SEVERITY.get(gt_dict[item_id], 1.0)
            # Partial credit if violation type is wrong but item correctly identified
            type_match = flag.violation_type == gt_dict[item_id]
            score = severity * (1.0 if type_match else 0.5)
            tp_score += score
            tp += 1
            breakdown[f"tp_{item_id}"] = score
        else:
            # False positive
            fp_penalty += 0.4
            fp += 1
            breakdown[f"fp_{item_id}"] = -0.4

    # False negatives: in ground truth but not flagged
    for item_id, vtype in gt_dict.items():
        if item_id not in flagged_ids:
            severity = SEVERITY.get(vtype, 1.0)
            fn_penalty += severity * 0.5
            fn += 1
            breakdown[f"fn_{item_id}"] = -severity * 0.5

    # Normalise tp_score by max possible
    max_possible_tp = sum(SEVERITY.get(vtype, 1.0) for _, vtype in ground_truth) or 1.0
    normalised_tp = tp_score / max_possible_tp

    # Decision alignment bonus
    has_violations = len(ground_truth) > 0
    decision_bonus = 0.0
    if has_violations and action.decision in (AuditDecision.FLAG, AuditDecision.REJECT):
        decision_bonus = 0.05
    elif not has_violations and action.decision == AuditDecision.APPROVE:
        decision_bonus = 0.05
    elif has_violations and action.decision == AuditDecision.APPROVE:
        decision_bonus = -0.1   # Approved a bad report

    # Combine
    raw = normalised_tp - (fp_penalty / (max_possible_tp + 1)) - (fn_penalty / (max_possible_tp + 1)) + decision_bonus
    value = max(-1.0, min(1.0, raw))

    # Precision / recall / F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    explanation = (
        f"TPs={tp}, FPs={fp}, FNs={fn} | "
        f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f} | "
        f"Reward={value:.3f}"
    )

    return Reward(
        value=value,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        breakdown=breakdown,
        explanation=explanation,
    )

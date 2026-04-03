"""
Rule-based agent for the Expense Audit environment.

Does NOT require an OpenAI API key. Useful for:
  - Verifying the environment works end-to-end
  - Establishing a deterministic lower-bound baseline
  - CI/CD pipeline validation

The agent applies explicit policy rules to flag violations.
It catches easy/medium violations well but deliberately misses
the hard fraud patterns (those require LLM reasoning).

Usage:
    python baseline/rule_based_agent.py
    python baseline/rule_based_agent.py --task hard --episodes 20 --seed 42
"""
from __future__ import annotations
import argparse
import os
import sys
import statistics
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import (
    Action, FlaggedItem, AuditDecision, ViolationType,
    Observation, ExpenseReport, CompanyPolicy, ExpenseLineItem
)
from tasks.task_definitions import GRADERS, TASKS


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

def apply_policy_rules(
    report: ExpenseReport,
    policy: CompanyPolicy,
) -> List[FlaggedItem]:
    """Check line items against explicit policy limits."""
    flagged: List[FlaggedItem] = []

    LIMITS = {
        "meals":         ("meal_limit_per_day",          ViolationType.OVER_LIMIT),
        "lodging":       ("lodging_limit_per_night",     ViolationType.OVER_LIMIT),
        "equipment":     ("equipment_limit_per_item",    ViolationType.OVER_LIMIT),
        "entertainment": ("entertainment_limit_per_event", ViolationType.OVER_LIMIT),
    }

    for item in report.line_items:
        # Missing receipt check
        if not item.has_receipt and item.amount > policy.receipt_required_above:
            flagged.append(FlaggedItem(
                item_id=item.item_id,
                violation_type=ViolationType.MISSING_RECEIPT,
                reason=f"Amount ${item.amount:.2f} requires receipt (threshold: ${policy.receipt_required_above})",
                confidence=1.0,
            ))
            continue  # Don't double-flag the same item

        # Over-limit check
        cat = item.category.value
        if cat in LIMITS:
            limit_attr, vtype = LIMITS[cat]
            limit = getattr(policy, limit_attr)
            if item.amount > limit:
                flagged.append(FlaggedItem(
                    item_id=item.item_id,
                    violation_type=vtype,
                    reason=f"{cat.title()} expense ${item.amount:.2f} exceeds ${limit:.2f} limit",
                    confidence=0.99,
                ))

    return flagged


def apply_duplicate_detection(
    report: ExpenseReport,
    historical_reports: list,
) -> List[FlaggedItem]:
    """Compare current line items against historical submissions."""
    flagged: List[FlaggedItem] = []

    # Build lookup: (vendor, date, amount) → True if seen before
    seen = set()
    for hist_report in historical_reports:
        for item in hist_report.line_items:
            key = (item.vendor.lower(), item.date, round(item.amount, 2))
            seen.add(key)

    for item in report.line_items:
        key = (item.vendor.lower(), item.date, round(item.amount, 2))
        if key in seen:
            flagged.append(FlaggedItem(
                item_id=item.item_id,
                violation_type=ViolationType.DUPLICATE,
                reason=(
                    f"Duplicate submission: {item.vendor} on {item.date} "
                    f"for ${item.amount:.2f} already reimbursed."
                ),
                confidence=0.95,
            ))

    return flagged


def apply_fraud_heuristics(
    report: ExpenseReport,
    policy: CompanyPolicy,
    vendor_stats: dict,
) -> List[FlaggedItem]:
    """
    Heuristic fraud detection for the hard task.
    Detects split receipts and round-number patterns.
    Vendor collusion is harder — we flag it if vendor_stats provide a signal.
    """
    flagged: List[FlaggedItem] = []
    already_flagged_ids = set()

    # --- Split receipt detection ---
    # Group by (vendor, date)
    from collections import defaultdict
    groups = defaultdict(list)
    for item in report.line_items:
        groups[(item.vendor.lower(), item.date)].append(item)

    LIMITS = {
        "meals":         policy.meal_limit_per_day,
        "lodging":       policy.lodging_limit_per_night,
        "equipment":     policy.equipment_limit_per_item,
        "entertainment": policy.entertainment_limit_per_event,
    }
    SPLIT_THRESHOLD = 0.85  # Both items > 85% of limit = split pattern

    for (vendor, date), items in groups.items():
        if len(items) >= 2:
            for item in items:
                cat = item.category.value
                limit = LIMITS.get(cat)
                if limit and item.amount >= (limit * SPLIT_THRESHOLD):
                    flagged.append(FlaggedItem(
                        item_id=item.item_id,
                        violation_type=ViolationType.SPLIT_RECEIPT,
                        reason=(
                            f"Potential split receipt: {item.vendor} on {item.date}, "
                            f"${item.amount:.2f} is {item.amount/limit*100:.0f}% of "
                            f"${limit:.2f} {cat} limit. "
                            f"{len(items)} items from same vendor on same day."
                        ),
                        confidence=0.80,
                    ))
                    already_flagged_ids.add(item.item_id)

    # --- Round-number fraud ---
    ROUND_AMOUNTS = {100, 200, 250, 500, 1000}
    SUSPICIOUS_CATEGORIES = {"equipment", "other", "entertainment"}
    for item in report.line_items:
        if item.item_id in already_flagged_ids:
            continue
        if item.amount in ROUND_AMOUNTS and item.category.value in SUSPICIOUS_CATEGORIES:
            flagged.append(FlaggedItem(
                item_id=item.item_id,
                violation_type=ViolationType.ROUND_NUMBER_FRAUD,
                reason=(
                    f"Suspicious exact round amount ${item.amount:.0f} "
                    f"for {item.category.value} category."
                ),
                confidence=0.70,
            ))
            already_flagged_ids.add(item.item_id)

    # --- Vendor collusion ---
    org_avg = vendor_stats.get("org_avg_vendor_frequency", {})
    emp_freq = vendor_stats.get("employee_vendor_frequency", {})
    for vendor, emp_count in emp_freq.items():
        org_count = org_avg.get(vendor, 0)
        if org_count > 0:
            ratio = emp_count / org_count
            if ratio >= 0.4:   # Employee accounts for ≥40% of org's vendor usage
                # Find items from this vendor
                for item in report.line_items:
                    if item.item_id not in already_flagged_ids and \
                       item.vendor.lower() == vendor.lower():
                        flagged.append(FlaggedItem(
                            item_id=item.item_id,
                            violation_type=ViolationType.VENDOR_COLLUSION,
                            reason=(
                                f"Vendor collusion signal: employee accounts for "
                                f"{emp_count}/{org_count} org-wide uses of '{vendor}' "
                                f"({ratio*100:.0f}% of total)."
                            ),
                            confidence=0.65,
                        ))
                        already_flagged_ids.add(item.item_id)

    return flagged


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def rule_based_agent(obs: Observation) -> Action:
    """Apply all rules appropriate for the current task difficulty."""
    report = obs.report
    policy = obs.policy
    task_desc = obs.task_description.lower()

    all_flags: List[FlaggedItem] = []

    # Always apply basic policy rules
    all_flags.extend(apply_policy_rules(report, policy))

    # Apply duplicate detection if history is present (medium/hard tasks)
    if obs.historical_reports:
        dup_flags = apply_duplicate_detection(report, obs.historical_reports)
        # Avoid double-flagging items already caught
        existing_ids = {f.item_id for f in all_flags}
        all_flags.extend(f for f in dup_flags if f.item_id not in existing_ids)

    # Apply fraud heuristics if vendor stats are present (hard task)
    if obs.vendor_stats:
        fraud_flags = apply_fraud_heuristics(report, policy, obs.vendor_stats)
        existing_ids = {f.item_id for f in all_flags}
        all_flags.extend(f for f in fraud_flags if f.item_id not in existing_ids)

    decision = AuditDecision.FLAG if all_flags else AuditDecision.APPROVE

    return Action(
        report_id=report.report_id,
        decision=decision,
        flagged_items=all_flags,
        overall_notes=f"Rule-based audit: {len(all_flags)} violation(s) found.",
    )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_rule_agent(task_name: str, episodes: int, seed: int) -> dict:
    env = ExpenseAuditEnv(task=task_name, seed=seed)
    grader = GRADERS[task_name]
    scores = []

    print(f"\n{'='*60}")
    print(f"Task: {task_name.upper()} — Rule-Based Agent")
    print(f"Episodes: {episodes} | Seed: {seed}")
    print("="*60)

    for ep in range(episodes):
        obs = env.reset()
        action = rule_based_agent(obs)
        _, reward, done, info = env.step(action)
        gt = [(d["item_id"], ViolationType(d["violation"])) for d in info["ground_truth"]]
        score = grader(action, gt)
        scores.append(score)
        status = "✓" if score >= 0.5 else "✗"
        print(f"  Ep {ep+1:02d}: reward={reward:+.3f} | score={score:.3f} {status} "
              f"| flagged={len(action.flagged_items)} gt={len(gt)}")

    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print(f"\nResults: mean={mean:.3f} ± {std:.3f}")

    return {
        "task": task_name,
        "agent": "rule_based",
        "episodes": episodes,
        "seed": seed,
        "mean_score": round(mean, 4),
        "std_score": round(std, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Run rule-based baseline agent")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results = [evaluate_rule_agent(t, args.episodes, args.seed) for t in tasks]

    print("\n" + "="*60)
    print("RULE-BASED AGENT SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['task']:8s} | mean={r['mean_score']:.3f} ± {r['std_score']:.3f}")

    print("\nNote: LLM agent (run_baseline.py) expected to outperform on medium/hard.")
    return results


if __name__ == "__main__":
    main()

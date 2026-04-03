"""
Standalone grader for Task 3: Coordinated Fraud Pattern Detection (Hard).

Usage:
    python graders/grade_hard.py --show-report --seed 42
    python graders/grade_hard.py --action-file my_action.json --seed 42
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import ExpenseAuditEnv, TASK_HARD
from env.models import Action, FlaggedItem, AuditDecision, ViolationType
from tasks.task_definitions import grade_hard


def score_action(action_dict: dict, seed: int = 0) -> dict:
    env = ExpenseAuditEnv(task=TASK_HARD, seed=seed)
    obs = env.reset()

    flagged = []
    for item in action_dict.get("flagged_items", []):
        try:
            vtype = ViolationType(item["violation_type"])
        except (KeyError, ValueError):
            vtype = ViolationType.SUSPICIOUS_PATTERN
        flagged.append(FlaggedItem(
            item_id=item.get("item_id", ""),
            violation_type=vtype,
            reason=item.get("reason", ""),
            confidence=float(item.get("confidence", 1.0)),
        ))

    try:
        decision = AuditDecision(action_dict.get("decision", "flag"))
    except ValueError:
        decision = AuditDecision.FLAG

    action = Action(
        report_id=action_dict.get("report_id", obs.report.report_id),
        decision=decision,
        flagged_items=flagged,
        overall_notes=action_dict.get("overall_notes"),
    )

    _, reward, done, info = env.step(action)
    gt = [(d["item_id"], ViolationType(d["violation"])) for d in info["ground_truth"]]
    task_score = grade_hard(action, gt)

    # Compute pattern-level summary for feedback
    split_gt = [gti for gti in info["ground_truth"] if gti["violation"] == "split_receipt"]
    caught_splits = sum(
        1 for gti in split_gt
        if any(f["item_id"] == gti["item_id"] for f in action_dict.get("flagged_items", []))
    )

    return {
        "task": "hard",
        "seed": seed,
        "report_id": obs.report.report_id,
        "score": task_score,
        "reward": reward,
        "ground_truth": info["ground_truth"],
        "reward_detail": info["reward_detail"],
        "passed": task_score >= 0.3,
        "pattern_summary": {
            "split_receipt_items_in_gt": len(split_gt),
            "split_receipts_caught": caught_splits,
            "fraud_patterns_to_detect": [
                "split_receipt: same vendor + same day, each ≥85% of category limit",
                "round_number_fraud: exact $100/$200/$500/$1000 amounts",
                "vendor_collusion: vendor with anomalously high frequency vs org avg",
            ]
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Grade an action on the Hard task")
    parser.add_argument("--action", type=str)
    parser.add_argument("--action-file", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-report", action="store_true")
    args = parser.parse_args()

    if args.action_file:
        with open(args.action_file) as f:
            action_dict = json.load(f)
    elif args.action:
        action_dict = json.loads(args.action)
    else:
        print("No action provided — running demo with approve-all action.\n")
        action_dict = {"decision": "approve", "flagged_items": []}

    if args.show_report:
        env = ExpenseAuditEnv(task=TASK_HARD, seed=args.seed)
        obs = env.reset()
        print(env.render())
        print(f"\nVendor stats: {json.dumps(obs.vendor_stats, indent=2)}")
        print()

    result = score_action(action_dict, seed=args.seed)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

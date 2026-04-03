"""
Standalone grader for Task 1: Basic Policy Violation Detection (Easy).

Can be run directly to score an agent's action JSON against a generated episode.

Usage:
    python graders/grade_easy.py --action '{"report_id": "...", "decision": "flag", "flagged_items": [...]}'
    python graders/grade_easy.py --action-file my_action.json --seed 42
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import ExpenseAuditEnv, TASK_EASY
from env.models import Action, FlaggedItem, AuditDecision, ViolationType
from tasks.task_definitions import grade_easy, TASKS


def score_action(action_dict: dict, seed: int = 0) -> dict:
    """
    Given a raw action dict and seed, regenerate the episode and score the action.
    This is fully deterministic: same seed → same report → same ground truth.
    """
    env = ExpenseAuditEnv(task=TASK_EASY, seed=seed)
    obs = env.reset()

    # Parse action
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
    task_score = grade_easy(action, gt)

    return {
        "task": "easy",
        "seed": seed,
        "report_id": obs.report.report_id,
        "score": task_score,
        "reward": reward,
        "ground_truth": info["ground_truth"],
        "reward_detail": info["reward_detail"],
        "passed": task_score >= 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description="Grade an action on the Easy task")
    parser.add_argument("--action", type=str, help="Action JSON string")
    parser.add_argument("--action-file", type=str, help="Path to action JSON file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-report", action="store_true", help="Print the report too")
    args = parser.parse_args()

    if args.action_file:
        with open(args.action_file) as f:
            action_dict = json.load(f)
    elif args.action:
        action_dict = json.loads(args.action)
    else:
        # Demo: run with a blank (approve) action to show scoring
        print("No action provided — running demo with approve-all action.\n")
        action_dict = {"decision": "approve", "flagged_items": []}

    if args.show_report:
        env = ExpenseAuditEnv(task=TASK_EASY, seed=args.seed)
        obs = env.reset()
        print(env.render())
        print()

    result = score_action(action_dict, seed=args.seed)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

"""
quickstart.py — Interactive walkthrough of all 3 tasks.

Run this to see the environment in action before writing your agent.

Usage:
    python quickstart.py
    python quickstart.py --task hard
    python quickstart.py --task all --seed 7
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import Action, FlaggedItem, AuditDecision, ViolationType
from tasks.task_definitions import TASKS, GRADERS


DIVIDER = "─" * 64


def demo_task(task_name: str, seed: int):
    task_def = TASKS[task_name]
    print(f"\n{'═'*64}")
    print(f"  TASK: {task_def.display_name if hasattr(task_def, 'display_name') else task_def.name}")
    print(f"  Difficulty: {task_def.difficulty.upper()}  |  Seed: {seed}")
    print(f"{'═'*64}")
    print(f"\n📋 Objective:\n   {task_def.objective}\n")

    env = ExpenseAuditEnv(task=task_name, seed=seed)
    obs = env.reset()

    # Print the report
    print(env.render())

    # Print historical context if available
    if obs.historical_reports:
        print(f"\n📁 Historical Reports ({len(obs.historical_reports)} previous submissions):")
        for hr in obs.historical_reports:
            print(f"   Report {hr.report_id} — {hr.submission_date} — ${hr.total_amount:.2f}")
            for item in hr.line_items:
                print(f"     [{item.item_id}] {item.date} | {item.category.value:15s} | "
                      f"${item.amount:8.2f} | {item.vendor}")

    # Print vendor stats if available
    if obs.vendor_stats:
        print(f"\n📊 Vendor Statistics:")
        print(f"   {json.dumps(obs.vendor_stats, indent=4)}")

    # Print company policy
    p = obs.policy
    print(f"\n📏 Company Policy:")
    print(f"   Meals:         max ${p.meal_limit_per_day}/day")
    print(f"   Lodging:       max ${p.lodging_limit_per_night}/night")
    print(f"   Equipment:     max ${p.equipment_limit_per_item}/item")
    print(f"   Entertainment: max ${p.entertainment_limit_per_event}/event")
    print(f"   Receipt required above: ${p.receipt_required_above}")

    # Show ground truth (for demo purposes)
    gt = env.state()["ground_truth"]
    print(f"\n🔍 Ground Truth ({len(gt)} violation(s)):")
    if gt:
        for g in gt:
            print(f"   item_id={g['item_id']} → {g['violation']}")
    else:
        print("   (clean report — no violations)")

    gt_typed = [(g["item_id"], ViolationType(g["violation"])) for g in gt]

    def make_perfect_action(o, g):
        return Action(
            report_id=o.report.report_id,
            decision=AuditDecision.FLAG if g else AuditDecision.APPROVE,
            flagged_items=[
                FlaggedItem(item_id=x["item_id"], violation_type=ViolationType(x["violation"]),
                            reason=f"Correctly identified {x['violation']}", confidence=1.0)
                for x in g
            ]
        )

    # Demo 1: Perfect agent — use same env (not yet stepped)
    print(f"\n{DIVIDER}")
    print("Agent 1: Perfect auditor (knows ground truth)")
    perfect_action = make_perfect_action(obs, gt)
    _, reward_perfect, _, _ = env.step(perfect_action)
    score_perfect = GRADERS[task_name](perfect_action, gt_typed)
    print(f"   Reward: {reward_perfect:+.3f}  |  Grader score: {score_perfect:.3f} ✓")

    # Demo 2: Blind approver
    print("Agent 2: Blind approver (always approves)")
    env2 = ExpenseAuditEnv(task=task_name, seed=seed)
    obs2 = env2.reset()
    gt2 = [(g["item_id"], ViolationType(g["violation"])) for g in env2.state()["ground_truth"]]
    blind_action = Action(report_id=obs2.report.report_id, decision=AuditDecision.APPROVE)
    _, reward_blind, _, _ = env2.step(blind_action)
    score_blind = GRADERS[task_name](blind_action, gt2)
    print(f"   Reward: {reward_blind:+.3f}  |  Grader score: {score_blind:.3f}")

    # Demo 3: Hallucinator
    print("Agent 3: Hallucinator (flags random wrong items)")
    env3 = ExpenseAuditEnv(task=task_name, seed=seed)
    obs3 = env3.reset()
    gt3 = [(g["item_id"], ViolationType(g["violation"])) for g in env3.state()["ground_truth"]]
    # Flag the first item with a probably-wrong violation type
    hallucinated = Action(
        report_id=obs3.report.report_id,
        decision=AuditDecision.FLAG,
        flagged_items=[
            FlaggedItem(item_id=obs3.report.line_items[0].item_id,
                        violation_type=ViolationType.SUSPICIOUS_PATTERN,
                        reason="hallucinated violation")
        ] if obs3.report.line_items else []
    )
    _, reward_hall, _, _ = env3.step(hallucinated)
    score_hall = GRADERS[task_name](hallucinated, gt3)
    print(f"   Reward: {reward_hall:+.3f}  |  Grader score: {score_hall:.3f}")

    print(f"\n{DIVIDER}")
    print(f"Expected scores → random: ~{task_def.expected_score_random}  "
          f"| frontier LLM: ~{task_def.expected_score_frontier}")


def main():
    parser = argparse.ArgumentParser(description="Expense Audit OpenEnv Quickstart Demo")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("\n🧾 Expense Audit OpenEnv — Quickstart Demo")
    print("=" * 64)
    print("This demo shows the environment, ground truth, and how 3")
    print("different agent strategies score on each task difficulty.")

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    for task_name in tasks:
        demo_task(task_name, args.seed)

    print("\n\n✅ Demo complete. Now write your agent using:")
    print("   from env.environment import ExpenseAuditEnv")
    print("   from env.models import Action, FlaggedItem, AuditDecision, ViolationType")
    print("\n   env = ExpenseAuditEnv(task='easy')")
    print("   obs = env.reset()")
    print("   # ... your agent logic ...")
    print("   obs, reward, done, info = env.step(your_action)\n")


if __name__ == "__main__":
    main()

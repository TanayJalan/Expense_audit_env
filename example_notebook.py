"""
example_notebook.py — Runnable example of using the Expense Audit environment.

This is a plain Python version of the notebook for environments without Jupyter.
Run with: python example_notebook.py

Or convert to .ipynb with: jupytext --to notebook example_notebook.py
"""

# ============================================================
# Cell 1: Setup
# ============================================================
print("=" * 60)
print("Expense Audit OpenEnv — Getting Started")
print("=" * 60)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ExpenseAuditEnv
from env.models import Action, FlaggedItem, AuditDecision, ViolationType
from env.gym_wrapper import ExpenseAuditGymEnv
from env.trajectory_logger import TrajectoryLogger
from baseline.rule_based_agent import rule_based_agent
from tasks.task_definitions import GRADERS

# ============================================================
# Cell 2: Basic usage — easy task
# ============================================================
print("\n--- EASY TASK: Basic Policy Violation Detection ---\n")

env = ExpenseAuditEnv(task="easy", seed=42)
obs = env.reset()
print(env.render())
print(f"\nTask: {obs.task_description}")

# ============================================================
# Cell 3: Manual audit
# ============================================================
print("\n--- SUBMITTING A MANUAL AUDIT ---\n")

# Look at items and flag any over the meal limit ($75/day)
flagged = []
for item in obs.report.line_items:
    if item.category.value == "meals" and item.amount > obs.policy.meal_limit_per_day:
        flagged.append(FlaggedItem(
            item_id=item.item_id,
            violation_type=ViolationType.OVER_LIMIT,
            reason=f"Meal ${item.amount:.2f} exceeds ${obs.policy.meal_limit_per_day}/day limit",
            confidence=1.0,
        ))
    elif not item.has_receipt and item.amount > obs.policy.receipt_required_above:
        flagged.append(FlaggedItem(
            item_id=item.item_id,
            violation_type=ViolationType.MISSING_RECEIPT,
            reason=f"Missing receipt for ${item.amount:.2f} item",
            confidence=1.0,
        ))

action = Action(
    report_id=obs.report.report_id,
    decision=AuditDecision.FLAG if flagged else AuditDecision.APPROVE,
    flagged_items=flagged,
)
print(f"Flagged {len(flagged)} item(s): {[f.item_id for f in flagged]}")

obs2, reward, done, info = env.step(action)
print(f"Reward: {reward:+.3f}")
print(f"Ground truth: {info['ground_truth']}")
print(f"Explanation: {info['reward_detail']['explanation']}")

# ============================================================
# Cell 4: Rule-based agent across all difficulties
# ============================================================
print("\n--- RULE-BASED AGENT ON ALL TASKS ---\n")

results = {}
for task in ["easy", "medium", "hard"]:
    scores = []
    for seed in range(10):
        env = ExpenseAuditEnv(task=task, seed=seed)
        obs = env.reset()
        action = rule_based_agent(obs)
        _, reward, _, info = env.step(action)
        gt = [(g["item_id"], ViolationType(g["violation"])) for g in info["ground_truth"]]
        score = GRADERS[task](action, gt)
        scores.append(score)
    mean = sum(scores) / len(scores)
    results[task] = mean
    print(f"  {task:8s}: mean score = {mean:.3f}")

# ============================================================
# Cell 5: Gymnasium wrapper usage
# ============================================================
print("\n--- GYMNASIUM WRAPPER ---\n")

import json
gym_env = ExpenseAuditGymEnv(task="hard", seed=7)
obs_str, info = gym_env.reset()
obs_dict = json.loads(obs_str)
print(f"Report ID: {obs_dict['report']['report_id']}")
print(f"Line items: {len(obs_dict['report']['line_items'])}")
print(f"Historical reports: {len(obs_dict['historical_reports'])}")

# Submit an action as a JSON string
action_str = json.dumps({
    "decision": "flag",
    "flagged_items": [
        {"item_id": obs_dict["report"]["line_items"][0]["item_id"],
         "violation_type": "split_receipt",
         "reason": "Same vendor and date, amount near limit",
         "confidence": 0.85}
    ]
})
obs2_str, reward, terminated, truncated, info = gym_env.step(action_str)
print(f"Reward: {reward:+.3f}  |  Terminated: {terminated}")

# ============================================================
# Cell 6: Trajectory logging
# ============================================================
print("\n--- TRAJECTORY LOGGING ---\n")

import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    logger = TrajectoryLogger(output_dir=tmpdir)
    for seed in range(3):
        env = ExpenseAuditEnv(task="easy", seed=seed)
        with logger.episode(task="easy", agent="rule_based", seed=seed) as ep:
            obs = env.reset()
            ep.log_observation(obs)
            action = rule_based_agent(obs)
            ep.log_action(action)
            _, reward, done, info = env.step(action)
            ep.log_result(reward, done, info)
            gt = [(g["item_id"], ViolationType(g["violation"])) for g in info["ground_truth"]]
            score = GRADERS["easy"](action, gt)
            ep.log_score(score)
    summary = logger.summary()
    print(f"Logged {summary['episodes']} episodes")
    print(f"Easy task: {summary['by_task']['easy']}")

# ============================================================
# Cell 7: Custom policy
# ============================================================
print("\n--- CUSTOM COMPANY POLICY ---\n")

from env.models import CompanyPolicy
strict_policy = CompanyPolicy(
    meal_limit_per_day=40.0,        # Very tight
    receipt_required_above=10.0,    # Receipt for everything over $10
    equipment_limit_per_item=200.0,
)
env = ExpenseAuditEnv(task="easy", policy=strict_policy, seed=42)
obs = env.reset()
print(f"Meal limit: ${obs.policy.meal_limit_per_day}/day (default: $75)")
print(f"Receipt threshold: ${obs.policy.receipt_required_above} (default: $25)")
action = rule_based_agent(obs)
print(f"Agent flagged {len(action.flagged_items)} items under strict policy")

print("\n✅ All cells complete. The environment is working correctly.")

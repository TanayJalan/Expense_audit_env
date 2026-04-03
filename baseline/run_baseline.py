"""
Baseline inference script for the Expense Audit OpenEnv environment.

Uses the OpenAI API client (reads OPENAI_API_KEY from environment).
Runs a language model agent against all 3 tasks and reports reproducible scores.

Usage:
    python baseline/run_baseline.py
    python baseline/run_baseline.py --task easy --episodes 10 --seed 42
    python baseline/run_baseline.py --model gpt-4o --episodes 5
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import statistics
from typing import List, Dict, Any

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import Action, FlaggedItem, AuditDecision, ViolationType
from tasks.task_definitions import TASKS, GRADERS


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(obs_dict: dict) -> str:
    report = obs_dict["report"]
    policy = obs_dict["policy"]
    history = obs_dict.get("historical_reports", [])
    vendor_stats = obs_dict.get("vendor_stats", {})
    task_desc = obs_dict.get("task_description", "")

    policy_str = (
        f"- Meals: max ${policy['meal_limit_per_day']}/day\n"
        f"- Lodging: max ${policy['lodging_limit_per_night']}/night\n"
        f"- Equipment: max ${policy['equipment_limit_per_item']}/item\n"
        f"- Entertainment: max ${policy['entertainment_limit_per_event']}/event\n"
        f"- Receipt required for any item over ${policy['receipt_required_above']}\n"
    )

    items_str = "\n".join(
        f"  item_id={item['item_id']} | date={item['date']} | "
        f"category={item['category']} | amount=${item['amount']} | "
        f"vendor={item['vendor']} | has_receipt={item['has_receipt']}"
        for item in report["line_items"]
    )

    history_str = ""
    if history:
        history_str = "\n\nPREVIOUS REPORTS FROM THIS EMPLOYEE:\n"
        for h in history:
            history_str += f"\nReport {h['report_id']} submitted {h['submitted_by']['name']} on {h['submission_date']}:\n"
            for item in h["line_items"]:
                history_str += (
                    f"  item_id={item['item_id']} | {item['date']} | "
                    f"{item['category']} | ${item['amount']} | {item['vendor']}\n"
                )

    vendor_str = ""
    if vendor_stats:
        vendor_str = f"\n\nVENDOR STATISTICS:\n{json.dumps(vendor_stats, indent=2)}"

    violation_types = ", ".join([v.value for v in ViolationType])

    return f"""You are an expert expense report auditor. Your task: {task_desc}

COMPANY POLICY:
{policy_str}

CURRENT EXPENSE REPORT:
Report ID: {report['report_id']}
Employee: {report['submitted_by']['name']} ({report['submitted_by']['department']})
Submission date: {report['submission_date']}
Total: ${report['total_amount']}

Line Items:
{items_str}
{history_str}{vendor_str}

INSTRUCTIONS:
Carefully audit this report. For each violation found, you MUST provide:
  - The exact item_id from the line items above
  - The violation_type (must be one of: {violation_types})
  - A clear reason

Respond in VALID JSON only, with this exact structure:
{{
  "decision": "approve" | "flag" | "reject",
  "flagged_items": [
    {{
      "item_id": "<exact item_id>",
      "violation_type": "<violation_type>",
      "reason": "<explanation>",
      "confidence": <0.0-1.0>
    }}
  ],
  "overall_notes": "<optional summary>"
}}

If no violations found, return an empty flagged_items list and decision "approve".
IMPORTANT: Only flag items with REAL violations. Do not hallucinate violations.
"""


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_agent(client: OpenAI, model: str, obs: dict, task: str) -> Action:
    prompt = build_prompt(obs)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    # Parse flagged items safely
    flagged = []
    for item in parsed.get("flagged_items", []):
        try:
            vtype = ViolationType(item.get("violation_type", "suspicious_pattern"))
        except ValueError:
            vtype = ViolationType.SUSPICIOUS_PATTERN
        flagged.append(FlaggedItem(
            item_id=item.get("item_id", ""),
            violation_type=vtype,
            reason=item.get("reason", ""),
            confidence=float(item.get("confidence", 1.0)),
        ))

    try:
        decision = AuditDecision(parsed.get("decision", "flag"))
    except ValueError:
        decision = AuditDecision.FLAG

    report = obs["report"]
    return Action(
        report_id=report["report_id"],
        decision=decision,
        flagged_items=flagged,
        overall_notes=parsed.get("overall_notes"),
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_task(
    client: OpenAI,
    model: str,
    task_name: str,
    episodes: int,
    seed: int,
) -> Dict[str, Any]:
    env = ExpenseAuditEnv(task=task_name, seed=seed)
    grader = GRADERS[task_name]
    scores: List[float] = []
    errors: List[str] = []

    print(f"\n{'='*60}")
    print(f"Task: {task_name.upper()} — {TASKS[task_name].name}")
    print(f"Episodes: {episodes} | Model: {model} | Seed: {seed}")
    print("="*60)

    for ep in range(episodes):
        try:
            obs = env.reset()
            obs_dict = obs.model_dump()

            action = run_agent(client, model, obs_dict, task_name)
            _, reward, done, info = env.step(action)

            # Also run task-specific grader
            gt = [(d["item_id"], ViolationType(d["violation"])) for d in info["ground_truth"]]
            task_score = grader(action, gt)
            scores.append(task_score)

            status = "✓" if task_score >= 0.5 else "✗"
            print(f"  Ep {ep+1:02d}: reward={reward:+.3f} | grader_score={task_score:.3f} {status}")

        except Exception as e:
            errors.append(str(e))
            print(f"  Ep {ep+1:02d}: ERROR — {e}")
            scores.append(0.0)

    mean = statistics.mean(scores) if scores else 0.0
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0

    result = {
        "task": task_name,
        "model": model,
        "episodes": episodes,
        "seed": seed,
        "mean_score": round(mean, 4),
        "std_score": round(stdev, 4),
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
        "errors": errors,
    }

    print(f"\nResults: mean={mean:.3f} ± {stdev:.3f}  (min={result['min_score']}, max={result['max_score']})")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run baseline agent on ExpenseAuditEnv")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    all_results = []
    for task_name in tasks:
        result = evaluate_task(client, args.model, task_name, args.episodes, args.seed)
        all_results.append(result)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"  {r['task']:8s} | mean={r['mean_score']:.3f} ± {r['std_score']:.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return all_results


if __name__ == "__main__":
    main()

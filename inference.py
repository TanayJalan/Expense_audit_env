import os
import sys
import json
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ExpenseAuditEnv
from env.models import Action, AuditDecision, FlaggedItem, ViolationType
from tasks.task_definitions import GRADERS

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "expense-audit-env"

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_action_from_llm(client, obs_dict):
    try:
        prompt = f"""You are an expense report auditor. Review this report and identify violations.

Report: {json.dumps(obs_dict['report'], default=str, indent=2)}
Policy: {json.dumps(obs_dict['policy'], default=str, indent=2)}
Task: {obs_dict['task_description']}
Historical reports: {json.dumps(obs_dict.get('historical_reports', []), default=str)}
Vendor stats: {json.dumps(obs_dict.get('vendor_stats', {}), default=str)}

Respond in JSON only, no markdown:
{{
  "decision": "approve" or "flag" or "reject",
  "flagged_items": [
    {{"item_id": "<exact_id>", "violation_type": "<type>", "reason": "<reason>", "confidence": 0.9}}
  ]
}}

Violation types: over_limit, missing_receipt, duplicate, split_receipt, round_number_fraud, vendor_collusion, suspicious_pattern"""

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        text = completion.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"decision": "approve", "flagged_items": []}

def run_task(task_name, seed=42):
    env = ExpenseAuditEnv(task=task_name, seed=seed)
    grader = GRADERS[task_name]
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards = []
    steps_taken = 0
    score = 0.499
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        obs_dict = obs.model_dump()

        action_dict = get_action_from_llm(client, obs_dict)

        flagged = []
        for item in action_dict.get("flagged_items", []):
            try:
                vtype = ViolationType(item.get("violation_type", "suspicious_pattern"))
            except ValueError:
                vtype = ViolationType.SUSPICIOUS_PATTERN
            flagged.append(FlaggedItem(
                item_id=item.get("item_id", ""),
                violation_type=vtype,
                reason=item.get("reason", ""),
                confidence=float(item.get("confidence", 0.9)),
            ))

        try:
            decision = AuditDecision(action_dict.get("decision", "approve"))
        except ValueError:
            decision = AuditDecision.APPROVE

        action = Action(
            report_id=obs.report.report_id,
            decision=decision,
            flagged_items=flagged,
        )

        _, reward, done, info = env.step(action)
        gt = [(g["item_id"], ViolationType(g["violation"])) for g in info["ground_truth"]]
        raw_score = grader(action, gt)

        # Clamp to strictly (0, 1) — not 0.0, not 1.0
        score = max(0.001, min(0.999, raw_score))

        rewards.append(reward)
        steps_taken = 1
        success = score > 0.5

        log_step(step=1, action=str(decision.value), reward=reward, done=done)

    except Exception as e:
        log_step(step=1, action="error", reward=0.01, done=True, error=str(e)[:100])
        score = 0.001
        rewards = [0.01]
        steps_taken = 1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    for t in tasks:
        run_task(t, args.seed)

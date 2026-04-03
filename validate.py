"""
openenv validate — local compliance checker for the Expense Audit environment.

Verifies all OpenEnv spec requirements before submission:
  ✓ reset() returns Observation
  ✓ step() returns (Observation, float, bool, dict)
  ✓ state() returns dict
  ✓ reward in [-1.0, 1.0]
  ✓ done=True after final step
  ✓ reset() clears done
  ✓ All 3 tasks work
  ✓ Graders return scores in [0.0, 1.0]
  ✓ openenv.yaml exists and is valid
  ✓ Dockerfile exists

Usage:
    python validate.py
    python validate.py --verbose
"""
from __future__ import annotations
import os
import sys
import json
import yaml
import traceback
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import Action, FlaggedItem, AuditDecision, ViolationType, Observation
from tasks.task_definitions import GRADERS


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg: str):  print(f"  {RED}✗{RESET} {msg}")
def warn(msg: str):  print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg: str):  print(f"  {BLUE}→{RESET} {msg}")


Results = List[Tuple[str, bool, str]]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_reset(results: Results, verbose: bool):
    label = "reset() returns Observation"
    try:
        env = ExpenseAuditEnv(task=TASK_EASY, seed=0)
        obs = env.reset()
        assert isinstance(obs, Observation), f"got {type(obs)}"
        assert obs.report is not None
        assert obs.policy is not None
        assert obs.step_number == 0
        assert len(obs.task_description) > 5
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_step(results: Results, verbose: bool):
    label = "step() returns (Observation, float, bool, dict)"
    try:
        env = ExpenseAuditEnv(task=TASK_EASY, seed=0)
        obs = env.reset()
        action = Action(report_id=obs.report.report_id, decision=AuditDecision.APPROVE)
        result = env.step(action)
        assert len(result) == 4, f"expected 4-tuple, got {len(result)}"
        obs2, reward, done, info = result
        assert isinstance(obs2, Observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_state(results: Results, verbose: bool):
    label = "state() returns dict with required keys"
    try:
        env = ExpenseAuditEnv(task=TASK_EASY, seed=0)
        env.reset()
        state = env.state()
        assert isinstance(state, dict)
        for key in ["task", "done", "ground_truth", "policy"]:
            assert key in state, f"missing key: {key}"
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_reward_range(results: Results, verbose: bool):
    label = "reward value always in [-1.0, 1.0]"
    try:
        for task in [TASK_EASY, TASK_MEDIUM, TASK_HARD]:
            for seed in range(10):
                env = ExpenseAuditEnv(task=task, seed=seed)
                obs = env.reset()
                # Try both extreme actions
                for decision in [AuditDecision.APPROVE, AuditDecision.REJECT]:
                    env2 = ExpenseAuditEnv(task=task, seed=seed)
                    obs2 = env2.reset()
                    action = Action(report_id=obs2.report.report_id, decision=decision)
                    _, reward, _, _ = env2.step(action)
                    assert -1.0 <= reward <= 1.0, f"reward={reward} out of range (task={task}, seed={seed})"
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_done_and_reset(results: Results, verbose: bool):
    label = "done=True after step, reset() clears done"
    try:
        env = ExpenseAuditEnv(task=TASK_EASY, seed=0)
        obs = env.reset()
        action = Action(report_id=obs.report.report_id, decision=AuditDecision.APPROVE)
        _, _, done, _ = env.step(action)
        assert done is True
        assert env._done is True

        # Stepping again should raise
        raised = False
        try:
            env.step(action)
        except AssertionError:
            raised = True
        assert raised, "Expected AssertionError when stepping after done"

        # Reset should clear
        env.reset()
        assert env._done is False
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_all_tasks(results: Results, verbose: bool):
    label = "All 3 tasks (easy/medium/hard) run end-to-end"
    try:
        for task in [TASK_EASY, TASK_MEDIUM, TASK_HARD]:
            env = ExpenseAuditEnv(task=task, seed=1)
            obs = env.reset()
            assert obs.task_description
            action = Action(report_id=obs.report.report_id, decision=AuditDecision.FLAG)
            obs2, reward, done, step_info = env.step(action)
            assert done
            assert "ground_truth" in step_info
            if verbose:
                info(f"task={task} | gt_items={len(step_info['ground_truth'])} | reward={reward:.3f}")
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_graders(results: Results, verbose: bool):
    label = "All graders return scores in [0.0, 1.0]"
    try:
        for task_name, grader in GRADERS.items():
            for seed in range(5):
                env = ExpenseAuditEnv(task=task_name, seed=seed)
                obs = env.reset()
                action = Action(
                    report_id=obs.report.report_id,
                    decision=AuditDecision.FLAG,
                    flagged_items=[
                        FlaggedItem(
                            item_id=obs.report.line_items[0].item_id,
                            violation_type=ViolationType.OVER_LIMIT,
                            reason="test",
                        )
                    ]
                )
                state_data = env.state()
                gt = [(d["item_id"], ViolationType(d["violation"])) for d in state_data["ground_truth"]]
                score = grader(action, gt)
                assert 0.0 <= score <= 1.0, f"grader={task_name} seed={seed} score={score}"
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_grader_determinism(results: Results, verbose: bool):
    label = "Graders are deterministic (same input → same score)"
    try:
        for task_name, grader in GRADERS.items():
            env = ExpenseAuditEnv(task=task_name, seed=42)
            obs = env.reset()
            action = Action(report_id=obs.report.report_id, decision=AuditDecision.FLAG)
            gt = [(d["item_id"], ViolationType(d["violation"])) for d in env.state()["ground_truth"]]
            score1 = grader(action, gt)
            score2 = grader(action, gt)
            score3 = grader(action, gt)
            assert score1 == score2 == score3, f"Non-deterministic grader: {score1}, {score2}, {score3}"
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_openenv_yaml(results: Results, verbose: bool):
    label = "openenv.yaml exists and contains required fields"
    try:
        yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
        assert os.path.exists(yaml_path), "openenv.yaml not found"
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        required = ["name", "version", "tasks", "observation_space", "action_space", "reward"]
        for field in required:
            assert field in data, f"missing field: {field}"
        assert len(data["tasks"]) >= 3, "must have at least 3 tasks"
        if verbose:
            info(f"name={data['name']} version={data['version']} tasks={len(data['tasks'])}")
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_dockerfile(results: Results, verbose: bool):
    label = "Dockerfile exists"
    try:
        dockerfile_path = os.path.join(os.path.dirname(__file__), "Dockerfile")
        assert os.path.exists(dockerfile_path), "Dockerfile not found"
        with open(dockerfile_path) as f:
            content = f.read()
        assert "FROM" in content, "Dockerfile missing FROM instruction"
        assert "EXPOSE" in content, "Dockerfile missing EXPOSE instruction"
        assert "CMD" in content or "ENTRYPOINT" in content, "Dockerfile missing CMD/ENTRYPOINT"
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_observation_fields(results: Results, verbose: bool):
    label = "Observation has all required fields"
    try:
        for task in [TASK_EASY, TASK_MEDIUM, TASK_HARD]:
            env = ExpenseAuditEnv(task=task, seed=0)
            obs = env.reset()
            obs_dict = obs.model_dump()
            required = ["report", "policy", "historical_reports", "vendor_stats",
                       "step_number", "max_steps", "task_description"]
            for field in required:
                assert field in obs_dict, f"task={task} missing field: {field}"
            # Medium and hard should have history
            if task in [TASK_MEDIUM, TASK_HARD]:
                assert len(obs.historical_reports) > 0, f"task={task} should have historical_reports"
            # Hard should have vendor stats
            if task == TASK_HARD:
                assert len(obs.vendor_stats) > 0, f"task={task} should have vendor_stats"
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_partial_credit(results: Results, verbose: bool):
    label = "Reward provides partial credit (not just binary)"
    try:
        # Generate a report with 2 violations, catch only 1 → reward should be between 0 and max
        env = ExpenseAuditEnv(task=TASK_EASY, seed=5)
        obs = env.reset()
        gt = env.state()["ground_truth"]

        if len(gt) >= 2:
            # Catch only the first violation
            env2 = ExpenseAuditEnv(task=TASK_EASY, seed=5)
            obs2 = env2.reset()
            action_partial = Action(
                report_id=obs2.report.report_id,
                decision=AuditDecision.FLAG,
                flagged_items=[FlaggedItem(
                    item_id=gt[0]["item_id"],
                    violation_type=ViolationType(gt[0]["violation"]),
                    reason="partial",
                )]
            )
            _, reward_partial, _, _ = env2.step(action_partial)

            env3 = ExpenseAuditEnv(task=TASK_EASY, seed=5)
            env3.reset()
            action_all = Action(
                report_id=obs.report.report_id,
                decision=AuditDecision.FLAG,
                flagged_items=[FlaggedItem(
                    item_id=g["item_id"],
                    violation_type=ViolationType(g["violation"]),
                    reason="all",
                ) for g in gt]
            )
            _, reward_all, _, _ = env3.step(action_all)

            assert reward_partial < reward_all, \
                f"Partial ({reward_partial:.3f}) should be < full ({reward_all:.3f})"
            assert reward_partial > -0.5, \
                f"Partial credit should be > -0.5, got {reward_partial:.3f}"
            if verbose:
                info(f"partial={reward_partial:.3f}, full={reward_all:.3f}")

        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))



def check_api_routes(results: Results, verbose: bool):
    label = "API routes exist (/, /health, /reset, /step, /state, /policy, /session)"
    try:
        import app as a
        paths = [r.path for r in a.app.routes]
        required = ['/', '/health', '/reset', '/step', '/state', '/policy', '/session']
        for expected in required:
            assert expected in paths, f'Missing route: {expected}'
        if verbose:
            info(f'Routes: {[p for p in paths if not p.startswith("/open") and not p.startswith("/red")]}')
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))


def check_custom_policy(results: Results, verbose: bool):
    label = "Custom CompanyPolicy respected by environment"
    try:
        from env.models import CompanyPolicy
        # Use a very tight meal limit — every meal should be over limit
        tight = CompanyPolicy(meal_limit_per_day=1.0)
        env = ExpenseAuditEnv(task=TASK_EASY, seed=0, policy=tight)
        obs = env.reset()
        assert obs.policy.meal_limit_per_day == 1.0, "Policy not applied"
        # Any meal item should be flagged by rule agent
        from baseline.rule_based_agent import rule_based_agent
        action = rule_based_agent(obs)
        meal_items = [i for i in obs.report.line_items if i.category.value == "meals"]
        if meal_items:
            flagged_ids = {f.item_id for f in action.flagged_items}
            assert any(m.item_id in flagged_ids for m in meal_items),                 "Tight policy: all meals should be flagged"
        ok(label)
        results.append((label, True, ""))
    except Exception as e:
        fail(f"{label}: {e}")
        results.append((label, False, str(e)))

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate OpenEnv spec compliance")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  OpenEnv Validation — expense-audit-env{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    results: Results = []
    checks = [
        check_reset,
        check_step,
        check_state,
        check_reward_range,
        check_done_and_reset,
        check_all_tasks,
        check_observation_fields,
        check_graders,
        check_grader_determinism,
        check_partial_credit,
        check_openenv_yaml,
        check_dockerfile,
        check_api_routes,
        check_custom_policy,
    ]

    for check in checks:
        try:
            check(results, args.verbose)
        except Exception as e:
            fail(f"Unexpected error in {check.__name__}: {e}")
            if args.verbose:
                traceback.print_exc()
            results.append((check.__name__, False, str(e)))

    # Summary
    passed = sum(1 for _, ok_, _ in results if ok_)
    total = len(results)
    failed_checks = [(name, msg) for name, ok_, msg in results if not ok_]

    print(f"\n{BOLD}{'='*60}{RESET}")
    if passed == total:
        print(f"{GREEN}{BOLD}  ✓ ALL {total} CHECKS PASSED — Ready for submission!{RESET}")
    else:
        print(f"{RED}{BOLD}  ✗ {passed}/{total} CHECKS PASSED{RESET}")
        print(f"\n{RED}Failed checks:{RESET}")
        for name, msg in failed_checks:
            print(f"  • {name}: {msg}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

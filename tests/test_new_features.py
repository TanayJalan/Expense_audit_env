"""
Tests for gym wrapper, trajectory logger, clean report generation,
and pyproject.toml package structure.
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import Action, AuditDecision, FlaggedItem, ViolationType
from env.gym_wrapper import ExpenseAuditGymEnv
from env.trajectory_logger import TrajectoryLogger
from env.data_generator import generate_easy_report
from env.models import CompanyPolicy


# ---------------------------------------------------------------------------
# Clean Report Tests
# ---------------------------------------------------------------------------

class TestCleanReports:

    def test_clean_reports_appear_in_easy_task(self):
        """Easy task must sometimes produce clean reports (0 violations)."""
        clean_found = False
        for seed in range(200):
            env = ExpenseAuditEnv(task=TASK_EASY, seed=seed)
            env.reset()
            gt = env.state()["ground_truth"]
            if not gt:
                clean_found = True
                break
        assert clean_found, "No clean reports found in 200 seeds — agents can't learn to approve!"

    def test_clean_reports_fraction_roughly_25_percent(self):
        """Between 15-40% of easy episodes should be clean (target ~25%)."""
        clean = sum(
            1 for seed in range(100)
            for _ in [ExpenseAuditEnv(task=TASK_EASY, seed=seed).reset()]
            if not ExpenseAuditEnv(task=TASK_EASY, seed=seed).state()["ground_truth"]
        )
        # Re-count properly
        clean = 0
        for seed in range(100):
            env = ExpenseAuditEnv(task=TASK_EASY, seed=seed)
            env.reset()
            if not env.state()["ground_truth"]:
                clean += 1
        assert 10 <= clean <= 45, f"Clean report rate {clean}% outside expected 10-45% range"

    def test_clean_report_full_approve_workflow(self):
        """Agent that approves a clean report should get positive reward."""
        # Find a clean report
        for seed in range(200):
            env = ExpenseAuditEnv(task=TASK_EASY, seed=seed)
            obs = env.reset()
            if not env.state()["ground_truth"]:
                # This is a clean report — approving should give good reward
                action = Action(report_id=obs.report.report_id, decision=AuditDecision.APPROVE)
                _, reward, _, _ = env.step(action)
                assert reward > 0, f"Approving clean report should give positive reward, got {reward}"
                break

    def test_generate_easy_report_clean_probability(self):
        """generate_easy_report with clean_probability=1.0 always returns 0 violations."""
        policy = CompanyPolicy()
        for _ in range(20):
            report, gt = generate_easy_report(policy=policy, clean_probability=1.0)
            assert len(gt) == 0, f"Expected clean report, got {gt}"

    def test_generate_easy_report_forced_violations(self):
        """generate_easy_report with num_violations=2 always returns 2 violations."""
        policy = CompanyPolicy()
        for _ in range(10):
            report, gt = generate_easy_report(policy=policy, num_violations=2)
            assert len(gt) == 2, f"Expected 2 violations, got {len(gt)}"

    def test_violation_type_variety_in_easy(self):
        """Easy task should produce both over_limit and missing_receipt violations."""
        seen_types = set()
        for seed in range(100):
            env = ExpenseAuditEnv(task=TASK_EASY, seed=seed)
            env.reset()
            for g in env.state()["ground_truth"]:
                seen_types.add(g["violation"])
        assert "over_limit" in seen_types
        assert "missing_receipt" in seen_types


# ---------------------------------------------------------------------------
# Gym Wrapper Tests
# ---------------------------------------------------------------------------

class TestGymWrapper:

    def test_reset_returns_string_observation(self):
        env = ExpenseAuditGymEnv(task=TASK_EASY, seed=42)
        obs, info = env.reset()
        assert isinstance(obs, str)
        assert isinstance(info, dict)
        # Must be valid JSON
        parsed = json.loads(obs)
        assert "report" in parsed
        assert "policy" in parsed

    def test_step_returns_five_tuple(self):
        env = ExpenseAuditGymEnv(task=TASK_EASY, seed=42)
        obs, _ = env.reset()
        result = env.step('{"decision": "approve", "flagged_items": []}')
        assert len(result) == 5
        obs2, reward, terminated, truncated, info = result
        assert isinstance(obs2, str)
        assert isinstance(reward, float)
        assert terminated is True
        assert truncated is False
        assert isinstance(info, dict)

    def test_reward_in_range(self):
        env = ExpenseAuditGymEnv(task=TASK_EASY, seed=0)
        env.reset()
        _, reward, _, _, _ = env.step('{"decision": "approve", "flagged_items": []}')
        assert -1.0 <= reward <= 1.0

    def test_step_with_flagged_item(self):
        env = ExpenseAuditGymEnv(task=TASK_EASY, seed=5)
        obs_str, _ = env.reset()
        obs = json.loads(obs_str)
        item_id = obs["report"]["line_items"][0]["item_id"]
        action_str = json.dumps({
            "decision": "flag",
            "flagged_items": [{"item_id": item_id, "violation_type": "over_limit",
                               "reason": "test", "confidence": 0.9}]
        })
        _, reward, terminated, _, info = env.step(action_str)
        assert terminated is True
        assert "ground_truth" in info

    def test_malformed_action_treated_as_approve(self):
        env = ExpenseAuditGymEnv(task=TASK_EASY, seed=1)
        env.reset()
        # Garbage JSON — should not crash
        _, reward, terminated, _, _ = env.step("not valid json {{{{")
        assert terminated is True
        assert -1.0 <= reward <= 1.0

    def test_all_tasks_work(self):
        for task in [TASK_EASY, TASK_MEDIUM, TASK_HARD]:
            env = ExpenseAuditGymEnv(task=task, seed=0)
            obs, _ = env.reset()
            assert isinstance(obs, str)
            parsed = json.loads(obs)
            assert parsed["task_description"]
            _, reward, done, _, info = env.step('{"decision":"approve","flagged_items":[]}')
            assert done is True

    def test_unwrapped_property(self):
        from env.environment import ExpenseAuditEnv
        env = ExpenseAuditGymEnv(task=TASK_EASY, seed=0)
        assert isinstance(env.unwrapped, ExpenseAuditEnv)

    def test_action_from_dict_helper(self):
        env = ExpenseAuditGymEnv(task=TASK_EASY, seed=0)
        env.reset()
        d = {"decision": "flag", "flagged_items": []}
        action_str = env.action_from_dict(d)
        assert isinstance(action_str, str)
        assert json.loads(action_str) == d

    def test_obs_to_dict_helper(self):
        env = ExpenseAuditGymEnv(task=TASK_EASY, seed=0)
        obs_str, _ = env.reset()
        d = env.obs_to_dict(obs_str)
        assert isinstance(d, dict)
        assert "report" in d

    def test_render_ansi_returns_string(self):
        env = ExpenseAuditGymEnv(task=TASK_EASY, seed=0, render_mode="ansi")
        env.reset()
        result = env.render()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_medium_task_has_history_in_obs(self):
        env = ExpenseAuditGymEnv(task=TASK_MEDIUM, seed=3)
        obs_str, _ = env.reset()
        obs = json.loads(obs_str)
        assert len(obs["historical_reports"]) > 0

    def test_hard_task_has_vendor_stats_in_obs(self):
        env = ExpenseAuditGymEnv(task=TASK_HARD, seed=3)
        obs_str, _ = env.reset()
        obs = json.loads(obs_str)
        assert len(obs["vendor_stats"]) > 0

    def test_seed_produces_reproducible_obs(self):
        env1 = ExpenseAuditGymEnv(task=TASK_EASY, seed=77)
        env2 = ExpenseAuditGymEnv(task=TASK_EASY, seed=77)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        d1, d2 = json.loads(obs1), json.loads(obs2)
        assert d1["report"]["report_id"] == d2["report"]["report_id"]


# ---------------------------------------------------------------------------
# Trajectory Logger Tests
# ---------------------------------------------------------------------------

class TestTrajectoryLogger:

    def test_episode_saves_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(output_dir=tmpdir)
            env = ExpenseAuditEnv(task=TASK_EASY, seed=42)

            with logger.episode(task="easy", agent="test-agent", seed=42) as ep:
                obs = env.reset()
                ep.log_observation(obs)
                action = Action(report_id=obs.report.report_id, decision=AuditDecision.APPROVE)
                ep.log_action(action)
                _, reward, done, info = env.step(action)
                ep.log_result(reward, done, info)
                ep.log_score(0.75)

            # Check file was created
            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl") and f != "index.jsonl"]
            assert len(files) == 1

            # Check content
            path = os.path.join(tmpdir, files[0])
            records = [json.loads(line) for line in open(path) if line.strip()]
            types = [r["type"] for r in records]
            assert "episode_header" in types
            assert "observation" in types
            assert "action" in types
            assert "result" in types
            assert "grader_score" in types

    def test_index_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(output_dir=tmpdir)
            env = ExpenseAuditEnv(task=TASK_EASY, seed=0)

            with logger.episode(task="easy", agent="tester", seed=0) as ep:
                obs = env.reset()
                ep.log_observation(obs)
                action = Action(report_id=obs.report.report_id, decision=AuditDecision.APPROVE)
                _, reward, done, info = env.step(action)
                ep.log_result(reward, done, info)
                ep.log_score(0.5)

            index = logger.load_index()
            assert len(index) == 1
            assert index[0]["task"] == "easy"
            assert index[0]["agent"] == "tester"
            assert index[0]["score"] == 0.5

    def test_multiple_episodes_all_indexed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(output_dir=tmpdir)

            for seed in range(5):
                env = ExpenseAuditEnv(task=TASK_EASY, seed=seed)
                with logger.episode(task="easy", agent="multi", seed=seed) as ep:
                    obs = env.reset()
                    ep.log_observation(obs)
                    action = Action(report_id=obs.report.report_id, decision=AuditDecision.APPROVE)
                    _, reward, done, info = env.step(action)
                    ep.log_result(reward, done, info)
                    ep.log_score(float(seed) / 5)

            index = logger.load_index()
            assert len(index) == 5
            summary = logger.summary()
            assert summary["episodes"] == 5
            assert "easy" in summary["by_task"]
            assert summary["by_task"]["easy"]["count"] == 5

    def test_episode_header_contains_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(output_dir=tmpdir)
            env = ExpenseAuditEnv(task=TASK_HARD, seed=99)

            with logger.episode(task="hard", agent="gpt-4o", seed=99) as ep:
                obs = env.reset()
                ep.log_observation(obs)
                action = Action(report_id=obs.report.report_id, decision=AuditDecision.FLAG)
                _, reward, done, info = env.step(action)
                ep.log_result(reward, done, info)
                ep.log_score(0.6)

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl") and f != "index.jsonl"]
            records = [json.loads(l) for l in open(os.path.join(tmpdir, files[0])) if l.strip()]
            header = records[0]
            assert header["task"] == "hard"
            assert header["agent"] == "gpt-4o"
            assert header["seed"] == 99
            assert header["reward"] is not None

    def test_summary_empty_when_no_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(output_dir=tmpdir)
            summary = logger.summary()
            assert summary["episodes"] == 0


# ---------------------------------------------------------------------------
# Package structure
# ---------------------------------------------------------------------------

class TestPackageStructure:

    def test_pyproject_toml_exists(self):
        assert os.path.exists("pyproject.toml")

    def test_license_exists(self):
        assert os.path.exists("LICENSE")
        content = open("LICENSE").read()
        assert "MIT" in content

    def test_all_packages_have_init(self):
        for pkg in ["env", "tasks", "graders", "baseline", "data"]:
            init = os.path.join(pkg, "__init__.py")
            assert os.path.exists(init), f"Missing {init}"

    def test_gym_wrapper_importable(self):
        from env.gym_wrapper import ExpenseAuditGymEnv
        assert ExpenseAuditGymEnv is not None

    def test_trajectory_logger_importable(self):
        from env.trajectory_logger import TrajectoryLogger
        assert TrajectoryLogger is not None

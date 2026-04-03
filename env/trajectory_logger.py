"""
Trajectory logger for the Expense Audit environment.

Records full episode trajectories (observation, action, reward, ground_truth)
to JSONL files for offline analysis, fine-tuning, and research.

Usage:
    from env.trajectory_logger import TrajectoryLogger
    from env.environment import ExpenseAuditEnv

    logger = TrajectoryLogger(output_dir="trajectories/")

    env = ExpenseAuditEnv(task="hard", seed=42)
    with logger.episode(task="hard", agent="gpt-4o", seed=42) as ep:
        obs = env.reset()
        ep.log_observation(obs)

        action = my_agent(obs)
        ep.log_action(action)

        obs2, reward, done, info = env.step(action)
        ep.log_result(reward, done, info)

    # Saved to: trajectories/hard_gpt-4o_20260402_143022.jsonl
"""
from __future__ import annotations
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from env.models import Observation, Action, ViolationType


class EpisodeRecorder:
    """Records a single episode's trajectory."""

    def __init__(self, task: str, agent: str, seed: Optional[int], output_path: str):
        self.task = task
        self.agent = agent
        self.seed = seed
        self.output_path = output_path
        self.start_time = time.time()
        self._records: List[dict] = []
        self._reward: Optional[float] = None
        self._score: Optional[float] = None

    def log_observation(self, obs: Observation):
        """Log the initial observation."""
        self._records.append({
            "type": "observation",
            "timestamp": time.time(),
            "data": obs.model_dump(mode="json"),
        })

    def log_action(self, action: Action):
        """Log the agent's action."""
        self._records.append({
            "type": "action",
            "timestamp": time.time(),
            "data": action.model_dump(mode="json"),
        })

    def log_result(self, reward: float, done: bool, info: Dict[str, Any]):
        """Log the step result."""
        self._reward = reward
        self._records.append({
            "type": "result",
            "timestamp": time.time(),
            "reward": reward,
            "done": done,
            "ground_truth": info.get("ground_truth", []),
            "reward_detail": info.get("reward_detail", {}),
        })

    def log_score(self, score: float):
        """Log the task-specific grader score."""
        self._score = score
        self._records.append({
            "type": "grader_score",
            "timestamp": time.time(),
            "score": score,
        })

    def save(self):
        """Write the full trajectory to JSONL."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            # Header line
            header = {
                "type": "episode_header",
                "task": self.task,
                "agent": self.agent,
                "seed": self.seed,
                "start_time": self.start_time,
                "duration_seconds": round(time.time() - self.start_time, 3),
                "reward": self._reward,
                "score": self._score,
            }
            f.write(json.dumps(header, default=str) + "\n")
            for record in self._records:
                f.write(json.dumps(record, default=str) + "\n")


class TrajectoryLogger:
    """
    Manages trajectory logging across multiple episodes.

    Each episode is saved as a separate JSONL file.
    A summary index file is maintained for quick analysis.
    """

    def __init__(self, output_dir: str = "trajectories"):
        self.output_dir = output_dir
        self.index_path = os.path.join(output_dir, "index.jsonl")
        os.makedirs(output_dir, exist_ok=True)
        self._episode_count = 0

    @contextmanager
    def episode(
        self,
        task: str,
        agent: str = "unknown",
        seed: Optional[int] = None,
    ) -> Iterator[EpisodeRecorder]:
        """
        Context manager for recording a single episode.

        Usage:
            with logger.episode(task="easy", agent="gpt-4o", seed=42) as ep:
                obs = env.reset()
                ep.log_observation(obs)
                ...
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task}_{agent}_{ts}_{self._episode_count:04d}.jsonl"
        output_path = os.path.join(self.output_dir, filename)

        recorder = EpisodeRecorder(
            task=task,
            agent=agent,
            seed=seed,
            output_path=output_path,
        )

        try:
            yield recorder
        finally:
            recorder.save()
            self._episode_count += 1
            self._update_index(recorder, filename)

    def _update_index(self, recorder: EpisodeRecorder, filename: str):
        """Append episode summary to the index file."""
        entry = {
            "file": filename,
            "task": recorder.task,
            "agent": recorder.agent,
            "seed": recorder.seed,
            "reward": recorder._reward,
            "score": recorder._score,
            "duration_seconds": round(time.time() - recorder.start_time, 3),
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.index_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_index(self) -> List[dict]:
        """Load the episode index for analysis."""
        if not os.path.exists(self.index_path):
            return []
        with open(self.index_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def summary(self) -> dict:
        """Print a summary of all logged episodes."""
        index = self.load_index()
        if not index:
            return {"episodes": 0}

        by_task: Dict[str, List[float]] = {}
        for ep in index:
            t = ep.get("task", "unknown")
            s = ep.get("score")
            if s is not None:
                by_task.setdefault(t, []).append(s)

        return {
            "episodes": len(index),
            "by_task": {
                task: {
                    "count": len(scores),
                    "mean_score": round(sum(scores) / len(scores), 4),
                    "min_score": round(min(scores), 4),
                    "max_score": round(max(scores), 4),
                }
                for task, scores in by_task.items()
            },
        }

    def load_episode(self, filename: str) -> List[dict]:
        """Load a specific episode's trajectory."""
        path = os.path.join(self.output_dir, filename)
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

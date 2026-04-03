"""
Generate and save canonical sample data fixtures.
These fixed fixtures are used to produce the benchmark baseline scores in README.

Run once to regenerate:
    python data/generate_fixtures.py

Output: data/fixtures_easy.json, data/fixtures_medium.json, data/fixtures_hard.json
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD

SEEDS = list(range(20))   # 20 canonical episodes per task
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_fixtures(task: str, seeds: list) -> list:
    fixtures = []
    for seed in seeds:
        env = ExpenseAuditEnv(task=task, seed=seed)
        obs = env.reset()
        obs_dict = obs.model_dump()
        state = env.state()
        fixtures.append({
            "seed": seed,
            "task": task,
            "observation": obs_dict,
            "ground_truth": state["ground_truth"],
        })
    return fixtures


def main():
    for task in [TASK_EASY, TASK_MEDIUM, TASK_HARD]:
        print(f"Generating {len(SEEDS)} fixtures for task={task}...")
        fixtures = generate_fixtures(task, SEEDS)
        out_path = os.path.join(OUTPUT_DIR, f"fixtures_{task}.json")
        with open(out_path, "w") as f:
            json.dump(fixtures, f, indent=2, default=str)
        print(f"  → {out_path} ({len(fixtures)} episodes)")

    print("\nDone. Use these fixtures for reproducible benchmarking.")
    print("Load with: json.load(open('data/fixtures_easy.json'))")


if __name__ == "__main__":
    main()

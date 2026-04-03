import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ExpenseAuditEnv
from env.models import Action, AuditDecision, ViolationType
from baseline.rule_based_agent import rule_based_agent
from tasks.task_definitions import GRADERS

def run(task, episodes, seed):
    env = ExpenseAuditEnv(task=task, seed=seed)
    grader = GRADERS[task]
    scores = []
    for ep in range(episodes):
        obs = env.reset()
        action = rule_based_agent(obs)
        _, reward, _, info = env.step(action)
        gt = [(g["item_id"], ViolationType(g["violation"])) for g in info["ground_truth"]]
        score = grader(action, gt)
        scores.append(score)
        print(f"  [{task}] ep={ep+1} reward={reward:+.3f} score={score:.3f}")
    mean = sum(scores) / len(scores)
    print(f"  {task}: mean={mean:.3f}")
    return mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["easy","medium","hard","all"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = ["easy","medium","hard"] if args.task == "all" else [args.task]
    for t in tasks:
        run(t, args.episodes, args.seed)

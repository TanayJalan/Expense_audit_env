---
title: Expense Audit OpenEnv
emoji: 🧾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - agent-evaluation
  - finance
  - fraud-detection
  - compliance
short_description: OpenEnv environment for AI expense report auditing agents
---

# 🧾 Expense Audit OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://openenv.dev)

A real-world OpenEnv environment where an AI agent acts as an **expense report auditor**. The agent reviews employee expense submissions and must identify policy violations, duplicate submissions, and coordinated fraud patterns.

## Why This Environment?

Every company processes thousands of expense reports. This environment trains agents to handle real financial compliance tasks with three escalating difficulty levels.

## Tasks

| Task | Difficulty | Objective | Frontier Score |
|------|-----------|-----------|----------------|
| Basic Policy Check | Easy | Flag over-limit expenses & missing receipts | ~0.85 |
| Duplicate Detection | Medium | Detect re-submitted expenses across history | ~0.70 |
| Fraud Pattern Detection | Hard | Split receipts, round-number fraud, vendor collusion | ~0.55 |

## API Usage

```python
# Reset environment
POST /reset
{"task": "easy", "seed": 42}
→ {"session_id": "...", "observation": {...}}

# Submit audit action
POST /step
{"session_id": "...", "action": {"report_id": "...", "decision": "flag", "flagged_items": [...]}}
→ {"reward": 0.92, "done": true, "info": {...}}

# Get state
GET /state?session_id=...
```

## Quick Python Example

```python
from env.environment import ExpenseAuditEnv
from env.models import Action, FlaggedItem, AuditDecision, ViolationType

env = ExpenseAuditEnv(task="easy", seed=42)
obs = env.reset()

action = Action(
    report_id=obs.report.report_id,
    decision=AuditDecision.FLAG,
    flagged_items=[
        FlaggedItem(
            item_id="<item_id>",
            violation_type=ViolationType.OVER_LIMIT,
            reason="Meal expense $89.50 exceeds $75 daily limit",
        )
    ]
)
obs, reward, done, info = env.step(action)
print(f"Reward: {reward:.3f}")
```

## Baseline Scores (rule-based agent, 20 episodes, seed=42)

| Task | Mean Score |
|------|-----------|
| Easy | 1.000 |
| Medium | 1.000 |
| Hard | 0.983 |

## Setup

```bash
git clone https://huggingface.co/spaces/your-handle/expense-audit-env
cd expense-audit-env
pip install -r requirements.txt
python validate.py    # 12/12 checks
python -m pytest tests/ -q    # 38/38 tests
```

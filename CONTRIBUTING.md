# Contributing to Expense Audit OpenEnv

Thank you for your interest in contributing! This guide covers how to add tasks, extend the environment, and submit improvements.

---

## Development Setup

```bash
git clone https://github.com/your-handle/expense-audit-env
cd expense-audit-env
pip install -r requirements.txt
make test        # 64 tests must pass
make validate    # 12 checks must pass
```

---

## Project Layout

```
env/              Core environment (models, generator, reward, session)
tasks/            Task specs + graders
graders/          Standalone CLI graders
baseline/         Agent implementations
tests/            Unit + integration tests (test_env, test_extended, test_api)
data/             Canonical benchmark fixtures
```

---

## Adding a New Violation Type

1. Add the new value to `ViolationType` in `env/models.py`
2. Add its severity weight in `env/reward.py` (`SEVERITY` dict)
3. Add an injector function in `env/data_generator.py`
4. Update the appropriate task grader in `tasks/task_definitions.py`
5. Add detection logic in `baseline/rule_based_agent.py`
6. Add a test case in `tests/test_extended.py`

---

## Adding a New Task Difficulty

1. Define a new `TaskDefinition` in `tasks/task_definitions.py` and add it to `TASKS`
2. Write a grader function and add it to `GRADERS`
3. Add the task to `ExpenseAuditEnv.__init__` and implement its `reset()` branch
4. Add a data generator function in `env/data_generator.py`
5. Add a standalone grader script in `graders/`
6. Update `openenv.yaml` tasks list
7. Write tests — all 64 existing tests must still pass

---

## Changing the Policy

The `CompanyPolicy` model is fully configurable:

```python
from env.models import CompanyPolicy
from env.environment import ExpenseAuditEnv

strict_policy = CompanyPolicy(
    meal_limit_per_day=50.0,       # Tighter meal limit
    lodging_limit_per_night=200.0,
    receipt_required_above=10.0,   # Receipt for everything over $10
)
env = ExpenseAuditEnv(task="easy", policy=strict_policy, seed=42)
```

---

## Code Style

- Follow existing patterns — Pydantic models, type hints, docstrings
- Keep functions short and single-purpose
- All new code must have tests
- Run `python -m pytest tests/ -v` before submitting a PR

---

## Submitting a PR

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Run `make ci` — all checks must pass
4. Open a PR with a clear description of what you changed and why

---

## Reporting Bugs

Open an issue with:
- Your Python version
- The exact command that failed
- The full error output
- The seed value if it's a data generation bug

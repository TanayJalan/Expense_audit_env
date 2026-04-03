.PHONY: install test validate quickstart baseline-rule baseline-llm fixtures docker-build docker-run clean help

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON     := python
SEED       := 42
EPISODES   := 20
MODEL      := gpt-4o-mini
DOCKER_TAG := expense-audit-env

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Expense Audit OpenEnv — Dev Commands"
	@echo "  ────────────────────────────────────"
	@echo "  make install         Install Python dependencies"
	@echo "  make test            Run full test suite (38 tests)"
	@echo "  make validate        Run OpenEnv compliance checks (12 checks)"
	@echo "  make quickstart      Interactive demo of all 3 tasks"
	@echo "  make baseline-rule   Run rule-based agent (no API key needed)"
	@echo "  make baseline-llm    Run LLM agent (requires OPENAI_API_KEY)"
	@echo "  make fixtures        Regenerate canonical benchmark fixtures"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-run      Run Docker container on port 7860"
	@echo "  make clean           Remove __pycache__ and build artifacts"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Testing & Validation ──────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -x -q

validate:
	$(PYTHON) validate.py --verbose

# ── Demo & Baseline ───────────────────────────────────────────────────────────
quickstart:
	$(PYTHON) quickstart.py --task all --seed $(SEED)

quickstart-easy:
	$(PYTHON) quickstart.py --task easy --seed $(SEED)

quickstart-hard:
	$(PYTHON) quickstart.py --task hard --seed $(SEED)

baseline-rule:
	$(PYTHON) baseline/rule_based_agent.py --task all --episodes $(EPISODES) --seed $(SEED)

baseline-llm:
	@if [ -z "$(OPENAI_API_KEY)" ]; then \
		echo "ERROR: OPENAI_API_KEY not set"; exit 1; \
	fi
	$(PYTHON) baseline/run_baseline.py --task all --model $(MODEL) --episodes $(EPISODES) --seed $(SEED)

# ── Graders (standalone) ──────────────────────────────────────────────────────
grade-easy:
	$(PYTHON) graders/grade_easy.py --show-report --seed $(SEED)

grade-medium:
	$(PYTHON) graders/grade_medium.py --show-report --seed $(SEED)

grade-hard:
	$(PYTHON) graders/grade_hard.py --show-report --seed $(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
fixtures:
	$(PYTHON) data/generate_fixtures.py

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t $(DOCKER_TAG) .

docker-run:
	docker run -p 7860:7860 --rm $(DOCKER_TAG)

docker-test:
	docker build -t $(DOCKER_TAG) .
	docker run --rm $(DOCKER_TAG) python validate.py
	docker run --rm $(DOCKER_TAG) python -m pytest tests/ -q

# ── CI pipeline (runs everything without API key) ─────────────────────────────
ci: install test validate baseline-rule fixtures
	@echo ""
	@echo "✓ CI pipeline complete"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f data/fixtures_*.json
	@echo "Cleaned."

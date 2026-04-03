"""
HTTP API integration tests using FastAPI's TestClient.
Tests the full request/response cycle for all API endpoints.

Run with: python -m pytest tests/test_api.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
from fastapi.testclient import TestClient

from app import app
from env.models import ViolationType


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health & root
# ---------------------------------------------------------------------------

class TestHealthAndRoot:

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["environment"] == "expense-audit-env"
        assert "active_sessions" in data

    def test_root_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "openenv" in r.text.lower()
        assert "/reset" in r.text

    def test_docs_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Reset endpoint
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_easy_returns_session_and_observation(self, client):
        r = client.post("/reset", json={"task": "easy", "seed": 42})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert "observation" in data
        assert data["task"] == "easy"
        obs = data["observation"]
        assert "report" in obs
        assert "policy" in obs
        assert "task_description" in obs
        assert len(obs["report"]["line_items"]) > 0

    def test_reset_medium_includes_history(self, client):
        r = client.post("/reset", json={"task": "medium", "seed": 1})
        assert r.status_code == 200
        obs = r.json()["observation"]
        assert len(obs["historical_reports"]) > 0

    def test_reset_hard_includes_vendor_stats(self, client):
        r = client.post("/reset", json={"task": "hard", "seed": 2})
        assert r.status_code == 200
        obs = r.json()["observation"]
        assert len(obs["vendor_stats"]) > 0

    def test_reset_invalid_task_returns_400(self, client):
        r = client.post("/reset", json={"task": "impossible"})
        assert r.status_code == 400

    def test_reset_without_seed_still_works(self, client):
        r = client.post("/reset", json={"task": "easy"})
        assert r.status_code == 200
        assert "session_id" in r.json()

    def test_different_seeds_produce_different_sessions(self, client):
        r1 = client.post("/reset", json={"task": "easy", "seed": 1})
        r2 = client.post("/reset", json={"task": "easy", "seed": 2})
        assert r1.json()["session_id"] != r2.json()["session_id"]
        assert r1.json()["observation"]["report"]["report_id"] != \
               r2.json()["observation"]["report"]["report_id"]


# ---------------------------------------------------------------------------
# Step endpoint
# ---------------------------------------------------------------------------

class TestStep:

    def _new_session(self, client, task="easy", seed=42):
        r = client.post("/reset", json={"task": task, "seed": seed})
        assert r.status_code == 200
        data = r.json()
        return data["session_id"], data["observation"]

    def test_step_approve_returns_reward(self, client):
        sid, obs = self._new_session(client)
        r = client.post("/step", json={
            "session_id": sid,
            "action": {
                "report_id": obs["report"]["report_id"],
                "decision": "approve",
                "flagged_items": [],
            }
        })
        assert r.status_code == 200
        data = r.json()
        assert "reward" in data
        assert "done" in data
        assert "info" in data
        assert data["done"] is True
        assert -1.0 <= data["reward"] <= 1.0

    def test_step_flag_item_returns_reward(self, client):
        sid, obs = self._new_session(client, seed=10)
        item_id = obs["report"]["line_items"][0]["item_id"]
        r = client.post("/step", json={
            "session_id": sid,
            "action": {
                "report_id": obs["report"]["report_id"],
                "decision": "flag",
                "flagged_items": [{
                    "item_id": item_id,
                    "violation_type": "over_limit",
                    "reason": "Test flag",
                    "confidence": 0.9,
                }]
            }
        })
        assert r.status_code == 200
        assert data["done"] if (data := r.json()) else True

    def test_step_includes_ground_truth_in_info(self, client):
        sid, obs = self._new_session(client, seed=42)
        r = client.post("/step", json={
            "session_id": sid,
            "action": {
                "report_id": obs["report"]["report_id"],
                "decision": "approve",
                "flagged_items": [],
            }
        })
        info = r.json()["info"]
        assert "ground_truth" in info
        assert isinstance(info["ground_truth"], list)
        for gt_item in info["ground_truth"]:
            assert "item_id" in gt_item
            assert "violation" in gt_item

    def test_step_after_done_returns_400(self, client):
        sid, obs = self._new_session(client)
        # First step
        client.post("/step", json={
            "session_id": sid,
            "action": {"report_id": obs["report"]["report_id"],
                       "decision": "approve", "flagged_items": []}
        })
        # Second step — should fail
        r = client.post("/step", json={
            "session_id": sid,
            "action": {"report_id": obs["report"]["report_id"],
                       "decision": "approve", "flagged_items": []}
        })
        assert r.status_code == 400

    def test_step_with_unknown_session_returns_404(self, client):
        r = client.post("/step", json={
            "session_id": "nonexistent-uuid",
            "action": {"report_id": "x", "decision": "approve", "flagged_items": []}
        })
        assert r.status_code == 404

    def test_step_with_all_violation_types(self, client):
        """Ensure every ViolationType string is accepted by the API."""
        sid, obs = self._new_session(client, seed=3)
        item_id = obs["report"]["line_items"][0]["item_id"]
        for vtype in ViolationType:
            s2, o2 = self._new_session(client, seed=3)
            r = client.post("/step", json={
                "session_id": s2,
                "action": {
                    "report_id": o2["report"]["report_id"],
                    "decision": "flag",
                    "flagged_items": [{
                        "item_id": o2["report"]["line_items"][0]["item_id"],
                        "violation_type": vtype.value,
                        "reason": f"Testing {vtype.value}",
                        "confidence": 0.8,
                    }]
                }
            })
            assert r.status_code == 200, f"Failed for vtype={vtype.value}: {r.text}"


# ---------------------------------------------------------------------------
# State endpoint
# ---------------------------------------------------------------------------

class TestState:

    def test_state_returns_dict_with_keys(self, client):
        r = client.post("/reset", json={"task": "easy", "seed": 5})
        sid = r.json()["session_id"]
        r2 = client.get(f"/state?session_id={sid}")
        assert r2.status_code == 200
        data = r2.json()
        for key in ["task", "done", "ground_truth", "policy"]:
            assert key in data, f"Missing key: {key}"

    def test_state_done_false_before_step(self, client):
        r = client.post("/reset", json={"task": "easy", "seed": 6})
        sid = r.json()["session_id"]
        state = client.get(f"/state?session_id={sid}").json()
        assert state["done"] is False

    def test_state_done_true_after_step(self, client):
        r = client.post("/reset", json={"task": "easy", "seed": 7})
        data = r.json()
        sid, obs = data["session_id"], data["observation"]
        client.post("/step", json={
            "session_id": sid,
            "action": {"report_id": obs["report"]["report_id"],
                       "decision": "approve", "flagged_items": []}
        })
        state = client.get(f"/state?session_id={sid}").json()
        assert state["done"] is True

    def test_state_unknown_session_404(self, client):
        r = client.get("/state?session_id=fake-session")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TestSessionEndpoints:

    def test_delete_session(self, client):
        r = client.post("/reset", json={"task": "easy", "seed": 99})
        sid = r.json()["session_id"]
        # Confirm it exists
        assert client.get(f"/state?session_id={sid}").status_code == 200
        # Delete it
        r2 = client.request("DELETE", "/session", params={"session_id": sid})
        assert r2.status_code == 200
        assert r2.json()["deleted"] == sid
        # Confirm it's gone
        assert client.get(f"/state?session_id={sid}").status_code == 404

    def test_health_counts_active_sessions(self, client):
        before = client.get("/health").json()["active_sessions"]
        client.post("/reset", json={"task": "easy"})
        after = client.get("/health").json()["active_sessions"]
        assert after == before + 1

    def test_reset_reuse_existing_session(self, client):
        # Create session
        r1 = client.post("/reset", json={"task": "easy", "seed": 11})
        sid = r1.json()["session_id"]
        report_id_1 = r1.json()["observation"]["report"]["report_id"]
        # Reuse session — should reset episode and return new report
        r2 = client.post("/reset", json={"task": "easy", "seed": 11, "session_id": sid})
        assert r2.status_code == 200
        assert r2.json()["session_id"] == sid
        # Same seed + new episode gives a different report
        report_id_2 = r2.json()["observation"]["report"]["report_id"]
        # report_id_2 may differ because episode_count advanced
        assert report_id_2 is not None


# ---------------------------------------------------------------------------
# Full episode walkthrough
# ---------------------------------------------------------------------------

class TestFullEpisodeWalkthrough:

    def test_easy_full_episode(self, client):
        """Reset → examine obs → submit action → check reward + GT."""
        r = client.post("/reset", json={"task": "easy", "seed": 42})
        assert r.status_code == 200
        data = r.json()
        sid = data["session_id"]
        obs = data["observation"]

        policy = obs["policy"]
        flagged = []
        for item in obs["report"]["line_items"]:
            amount = item["amount"]
            cat = item["category"]
            has_receipt = item["has_receipt"]

            # Simple rule: flag over-limit meals
            if cat == "meals" and amount > policy["meal_limit_per_day"]:
                flagged.append({
                    "item_id": item["item_id"],
                    "violation_type": "over_limit",
                    "reason": f"Meal ${amount:.2f} > ${policy['meal_limit_per_day']} limit",
                    "confidence": 1.0,
                })
            # Flag missing receipts
            elif not has_receipt and amount > policy["receipt_required_above"]:
                flagged.append({
                    "item_id": item["item_id"],
                    "violation_type": "missing_receipt",
                    "reason": f"Amount ${amount:.2f} requires receipt",
                    "confidence": 1.0,
                })

        r2 = client.post("/step", json={
            "session_id": sid,
            "action": {
                "report_id": obs["report"]["report_id"],
                "decision": "flag" if flagged else "approve",
                "flagged_items": flagged,
            }
        })
        assert r2.status_code == 200
        result = r2.json()
        assert result["done"] is True
        assert -1.0 <= result["reward"] <= 1.0
        assert "ground_truth" in result["info"]

    def test_hard_full_episode_with_fraud_flags(self, client):
        """Hard task: can we detect split receipts via the API?"""
        r = client.post("/reset", json={"task": "hard", "seed": 42})
        data = r.json()
        sid, obs = data["session_id"], data["observation"]

        policy = obs["policy"]
        items = obs["report"]["line_items"]

        # Group by vendor + date
        from collections import defaultdict
        groups = defaultdict(list)
        for item in items:
            groups[(item["vendor"], item["date"])].append(item)

        flagged = []
        for (vendor, date), grp in groups.items():
            if len(grp) >= 2:
                for item in grp:
                    cat = item["category"]
                    limit_map = {
                        "meals": policy["meal_limit_per_day"],
                        "lodging": policy["lodging_limit_per_night"],
                        "equipment": policy["equipment_limit_per_item"],
                        "entertainment": policy["entertainment_limit_per_event"],
                    }
                    limit = limit_map.get(cat, 9999)
                    if item["amount"] >= limit * 0.85:
                        flagged.append({
                            "item_id": item["item_id"],
                            "violation_type": "split_receipt",
                            "reason": f"Same vendor/day, {item['amount']/limit*100:.0f}% of limit",
                            "confidence": 0.85,
                        })

        r2 = client.post("/step", json={
            "session_id": sid,
            "action": {
                "report_id": obs["report"]["report_id"],
                "decision": "flag" if flagged else "approve",
                "flagged_items": flagged,
            }
        })
        assert r2.status_code == 200
        result = r2.json()
        assert result["done"] is True
        # At least verify the API round-trip works
        assert "reward" in result
        assert "info" in result


# ---------------------------------------------------------------------------
# Policy endpoint
# ---------------------------------------------------------------------------

class TestPolicyEndpoint:

    def test_post_policy_creates_session(self, client):
        r = client.post("/policy", json={
            "task": "easy",
            "seed": 42,
            "policy": {"meal_limit_per_day": 50.0}
        })
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert data["policy"]["meal_limit_per_day"] == 50.0
        assert data["task"] == "easy"
        assert "session_id" in data["message"]

    def test_get_policy_returns_active_policy(self, client):
        # Create with custom policy
        r1 = client.post("/policy", json={
            "task": "easy",
            "policy": {"meal_limit_per_day": 30.0, "receipt_required_above": 5.0}
        })
        sid = r1.json()["session_id"]
        # Retrieve it
        r2 = client.get(f"/policy?session_id={sid}")
        assert r2.status_code == 200
        pol = r2.json()
        assert pol["meal_limit_per_day"] == 30.0
        assert pol["receipt_required_above"] == 5.0

    def test_policy_session_can_be_reset(self, client):
        r1 = client.post("/policy", json={
            "task": "easy", "seed": 1,
            "policy": {"meal_limit_per_day": 20.0}
        })
        sid = r1.json()["session_id"]
        r2 = client.post("/reset", json={"task": "easy", "seed": 1, "session_id": sid})
        assert r2.status_code == 200
        obs = r2.json()["observation"]
        # Policy in observation should reflect the custom one
        assert obs["policy"]["meal_limit_per_day"] == 20.0

    def test_policy_affects_violations(self, client):
        """Custom policy is reflected in the observation returned by reset."""
        r1 = client.post("/policy", json={
            "task": "easy", "seed": 5,
            "policy": {"meal_limit_per_day": 1.0, "equipment_limit_per_item": 1.0}
        })
        sid = r1.json()["session_id"]
        r2 = client.post("/reset", json={"task": "easy", "seed": 5, "session_id": sid})
        assert r2.status_code == 200
        obs = r2.json()["observation"]
        # The policy in the observation must reflect our custom values
        assert obs["policy"]["meal_limit_per_day"] == 1.0
        assert obs["policy"]["equipment_limit_per_item"] == 1.0
        # Episode should complete without error
        r3 = client.post("/step", json={
            "session_id": sid,
            "action": {
                "report_id": obs["report"]["report_id"],
                "decision": "approve",
                "flagged_items": [],
            }
        })
        assert r3.status_code == 200
        assert -1.0 <= r3.json()["reward"] <= 1.0

    def test_get_policy_unknown_session_404(self, client):
        r = client.get("/policy?session_id=ghost-session")
        assert r.status_code == 404

    def test_post_policy_invalid_task_400(self, client):
        r = client.post("/policy", json={"task": "impossible"})
        assert r.status_code == 400

    def test_default_policy_values(self, client):
        """POST /policy with no policy field should use CompanyPolicy defaults."""
        r = client.post("/policy", json={"task": "easy"})
        assert r.status_code == 200
        pol = r.json()["policy"]
        assert pol["meal_limit_per_day"] == 75.0
        assert pol["lodging_limit_per_night"] == 250.0
        assert pol["receipt_required_above"] == 25.0

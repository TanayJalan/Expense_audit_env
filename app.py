import sys, os
sys.path.insert(0, "/app")
"""
FastAPI server exposing the Expense Audit environment as an HTTP API.
Required for Hugging Face Spaces deployment.

Endpoints:
  POST /reset         → start episode, get session_id + first observation
  POST /step          → submit audit action, get (observation, reward, done, info)
  GET  /state         → get current environment state
  POST /policy        → create a session with a custom CompanyPolicy
  GET  /policy        → get the policy for an existing session
  DELETE /session     → close and remove a session
  GET  /health        → health check + active session count
  GET  /              → HTML landing page
  GET  /docs          → Swagger interactive docs
"""
from __future__ import annotations
import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import Action, Observation, CompanyPolicy
from env.session_manager import session_manager

VALID_TASKS = [TASK_EASY, TASK_MEDIUM, TASK_HARD]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Expense Audit OpenEnv",
    description=(
        "An OpenEnv-compliant environment for training and evaluating AI agents "
        "on expense report auditing tasks."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = TASK_EASY
    seed: Optional[int] = None
    session_id: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation
    task: str


class StepRequest(BaseModel):
    session_id: str
    action: Action


class StepResponse(BaseModel):
    session_id: str
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class PolicyRequest(BaseModel):
    task: str = TASK_EASY
    seed: Optional[int] = None
    policy: CompanyPolicy = CompanyPolicy()


class PolicyResponse(BaseModel):
    session_id: str
    policy: CompanyPolicy
    task: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    active = session_manager.active_count
    return f"""<!DOCTYPE html>
<html>
<head><title>Expense Audit OpenEnv</title></head>
<body style="font-family:monospace;background:#0d1117;color:#e6edf3;padding:48px;max-width:760px;margin:auto">
  <h1>🧾 Expense Audit OpenEnv <span style="font-size:0.55em;color:#58a6ff">v1.0.0</span></h1>
  <p style="color:#8b949e">
    A real-world <a href="https://openenv.dev" style="color:#58a6ff">OpenEnv</a>-compliant
    environment for training AI agents on expense report auditing.
  </p>
  <p>Active sessions: <strong style="color:#3fb950">{active}</strong></p>
  <h2 style="border-bottom:1px solid #30363d;padding-bottom:8px">Endpoints</h2>
  <table style="width:100%;border-collapse:collapse">
    <tr style="color:#58a6ff">
      <th style="text-align:left;padding:6px 12px">Method</th>
      <th style="text-align:left;padding:6px 12px">Path</th>
      <th style="text-align:left;padding:6px 12px">Description</th>
    </tr>
    <tr><td style="padding:5px 12px">POST</td><td>/reset</td><td style="color:#8b949e">Start episode → session_id + observation</td></tr>
    <tr><td style="padding:5px 12px">POST</td><td>/step</td><td style="color:#8b949e">Submit audit action → reward + done + info</td></tr>
    <tr><td style="padding:5px 12px">GET</td><td>/state</td><td style="color:#8b949e">Current environment state</td></tr>
    <tr><td style="padding:5px 12px">POST</td><td>/policy</td><td style="color:#8b949e">Create session with custom CompanyPolicy</td></tr>
    <tr><td style="padding:5px 12px">GET</td><td>/policy</td><td style="color:#8b949e">Get policy for existing session</td></tr>
    <tr><td style="padding:5px 12px">DELETE</td><td>/session</td><td style="color:#8b949e">Close session</td></tr>
    <tr><td style="padding:5px 12px">GET</td><td>/health</td><td style="color:#8b949e">Health check</td></tr>
  </table>
  <h2 style="border-bottom:1px solid #30363d;padding-bottom:8px;margin-top:32px">Tasks</h2>
  <table style="width:100%;border-collapse:collapse">
    <tr style="color:#58a6ff">
      <th style="text-align:left;padding:6px 12px">Task</th>
      <th style="text-align:left;padding:6px 12px">Difficulty</th>
      <th style="text-align:left;padding:6px 12px">What to detect</th>
    </tr>
    <tr><td style="padding:5px 12px">easy</td><td style="color:#3fb950">Easy</td><td style="color:#8b949e">Over-limit expenses, missing receipts</td></tr>
    <tr><td style="padding:5px 12px">medium</td><td style="color:#d29922">Medium</td><td style="color:#8b949e">Duplicate submissions + policy violations</td></tr>
    <tr><td style="padding:5px 12px">hard</td><td style="color:#f85149">Hard</td><td style="color:#8b949e">Split receipts, round-number fraud, vendor collusion</td></tr>
  </table>
  <p style="margin-top:32px">
    <a href="/docs" style="color:#58a6ff;margin-right:24px">→ Swagger Docs</a>
    <a href="/redoc" style="color:#58a6ff">→ ReDoc</a>
  </p>
</body>
</html>"""


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "expense-audit-env",
        "version": "1.0.0",
        "active_sessions": session_manager.active_count,
        "tasks": VALID_TASKS,
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    """Reset the environment and get the first observation."""
    if request.task not in VALID_TASKS:
        raise HTTPException(400, f"Unknown task '{request.task}'. Must be one of {VALID_TASKS}")
    if request.session_id:
        session = session_manager.get(request.session_id)
        if session is None:
            raise HTTPException(404, f"Session '{request.session_id}' not found or expired.")
        if request.seed is not None:
            session.env.seed = request.seed
    else:
        session = session_manager.create(task=request.task, seed=request.seed)
    obs = session.env.reset()
    session.episode_count += 1
    return ResetResponse(session_id=session.session_id, observation=obs, task=request.task)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Submit your audit action and receive the reward."""
    session = session_manager.get(request.session_id)
    if session is None:
        raise HTTPException(404, f"Session '{request.session_id}' not found or expired. Call /reset first.")
    try:
        obs, reward, done, info = session.env.step(request.action)
    except AssertionError as e:
        raise HTTPException(400, str(e))
    return StepResponse(
        session_id=request.session_id,
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def state(session_id: str):
    """Return full internal state for debugging and checkpointing."""
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found or expired.")
    return session.env.state()


@app.post("/policy", response_model=PolicyResponse)
def set_policy(request: PolicyRequest):
    """
    Create a session with a custom CompanyPolicy.

    Lets you test agents under stricter or more relaxed spending limits.
    After calling this, use POST /reset with the returned session_id to begin an episode.

    Example body:
        {"task": "easy", "seed": 42, "policy": {"meal_limit_per_day": 50.0, "receipt_required_above": 10.0}}
    """
    if request.task not in VALID_TASKS:
        raise HTTPException(400, f"Unknown task '{request.task}'. Must be one of {VALID_TASKS}")
    session = session_manager.create(task=request.task, seed=request.seed)
    session.env.policy = request.policy
    return PolicyResponse(
        session_id=session.session_id,
        policy=request.policy,
        task=request.task,
        message=f"Session created. Call POST /reset with session_id='{session.session_id}' to begin.",
    )


@app.get("/policy")
def get_policy(session_id: str):
    """Return the CompanyPolicy active for a session."""
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found or expired.")
    return session.env.policy.model_dump()


@app.delete("/session")
def close_session(session_id: str):
    """Delete a session and free its resources."""
    session_manager.delete(session_id)
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# Gradio interactive demo  (mounted at /ui)
# ---------------------------------------------------------------------------

try:
    import gradio as gr
    from ui import demo as gradio_demo
    import gradio.routes
    app = gr.mount_gradio_app(app, gradio_demo, path="/ui")
    _GRADIO_MOUNTED = True
except Exception:
    _GRADIO_MOUNTED = False   # Gradio optional — API still works without it


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


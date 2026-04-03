"""
Interactive Gradio demo for the Expense Audit OpenEnv environment.

Gives judges and users a live, clickable interface to:
  - Browse expense reports across all 3 task difficulties
  - Submit audit decisions and see real-time reward feedback
  - Compare rule-based vs manual auditing
  - Understand the environment without writing code

Runs as a tab inside the FastAPI app via Gradio's mount_gradio_app,
or standalone on port 7861.

Usage:
    python ui.py                    # standalone on :7861
    python ui.py --port 7861
"""
from __future__ import annotations
import json
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from env.environment import ExpenseAuditEnv, TASK_EASY, TASK_MEDIUM, TASK_HARD
from env.models import (
    Action, FlaggedItem, AuditDecision, ViolationType, CompanyPolicy
)
from baseline.rule_based_agent import rule_based_agent
from tasks.task_definitions import GRADERS


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

_current_env: dict = {}   # task -> env instance (stateful across interactions)


def _get_env(task: str, seed: int) -> ExpenseAuditEnv:
    key = f"{task}:{seed}"
    if key not in _current_env:
        _current_env[key] = ExpenseAuditEnv(task=task, seed=seed)
    return _current_env[key]


def _format_policy(policy: CompanyPolicy) -> str:
    return (
        f"**Meals:** ${policy.meal_limit_per_day}/day  |  "
        f"**Lodging:** ${policy.lodging_limit_per_night}/night  |  "
        f"**Equipment:** ${policy.equipment_limit_per_item}/item  |  "
        f"**Entertainment:** ${policy.entertainment_limit_per_event}/event  |  "
        f"**Receipt required above:** ${policy.receipt_required_above}"
    )


def _format_items_table(items) -> str:
    rows = ["| Item ID | Date | Category | Amount | Vendor | Receipt |",
            "|---------|------|----------|--------|--------|---------|"]
    for item in items:
        receipt = "✅" if item.has_receipt else "❌"
        rows.append(
            f"| `{item.item_id}` | {item.date} | {item.category.value} | "
            f"**${item.amount:.2f}** | {item.vendor} | {receipt} |"
        )
    return "\n".join(rows)


def _format_history(history) -> str:
    if not history:
        return "_No previous reports._"
    lines = []
    for hr in history:
        lines.append(f"**Report {hr.report_id}** — {hr.submission_date} — ${hr.total_amount:.2f}")
        for item in hr.line_items:
            lines.append(
                f"  - `{item.item_id}` {item.date} | {item.category.value} | "
                f"${item.amount:.2f} | {item.vendor}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core UI functions
# ---------------------------------------------------------------------------

def load_episode(task: str, seed: int):
    """Load a fresh episode and return formatted display components."""
    env = ExpenseAuditEnv(task=task, seed=seed)
    obs = env.reset()
    _current_env[f"{task}:{seed}:env"] = env
    _current_env[f"{task}:{seed}:obs"] = obs

    # Format report
    report_md = f"""## 📄 Expense Report `{obs.report.report_id}`

**Employee:** {obs.report.submitted_by.name} ({obs.report.submitted_by.department})  
**Submitted:** {obs.report.submission_date}  
**Total:** ${obs.report.total_amount:.2f}  
**Risk score:** {obs.report.submitted_by.risk_score:.1f}

### Line Items

{_format_items_table(obs.report.line_items)}
"""

    policy_md = f"### 📏 Company Policy\n\n{_format_policy(obs.policy)}"

    history_md = ""
    if obs.historical_reports:
        history_md = f"### 📁 Previous Reports ({len(obs.historical_reports)})\n\n{_format_history(obs.historical_reports)}"

    vendor_md = ""
    if obs.vendor_stats:
        vendor_md = f"### 📊 Vendor Statistics\n\n```json\n{json.dumps(obs.vendor_stats, indent=2)}\n```"

    task_md = f"> **Task:** {obs.task_description}"

    # Item IDs for checkbox selection
    item_ids = [f"{item.item_id} | {item.category.value} | ${item.amount:.2f} | {item.vendor}"
                for item in obs.report.line_items]

    return (
        task_md,
        report_md,
        policy_md,
        history_md or "_No history for this task._",
        vendor_md or "_No vendor stats for this task._",
        gr.CheckboxGroup(choices=item_ids, value=[], label="Select items to flag"),
        "⏳ Submit your audit to see results.",
        "",
    )


def submit_audit(task: str, seed: int, decision: str, flagged_display: list, violation_type: str, reason: str):
    """Process the agent's manual audit submission."""
    env_key = f"{task}:{seed}:env"
    obs_key = f"{task}:{seed}:obs"

    if env_key not in _current_env:
        return "❌ Please load an episode first.", "", ""

    env: ExpenseAuditEnv = _current_env[env_key]
    obs = _current_env[obs_key]

    # Parse selected items
    flagged_items = []
    for display_str in (flagged_display or []):
        item_id = display_str.split(" | ")[0]
        try:
            vtype = ViolationType(violation_type)
        except ValueError:
            vtype = ViolationType.OVER_LIMIT
        flagged_items.append(FlaggedItem(
            item_id=item_id,
            violation_type=vtype,
            reason=reason or "Manual audit flag",
            confidence=0.9,
        ))

    try:
        action_decision = AuditDecision(decision.lower())
    except ValueError:
        action_decision = AuditDecision.FLAG if flagged_items else AuditDecision.APPROVE

    action = Action(
        report_id=obs.report.report_id,
        decision=action_decision,
        flagged_items=flagged_items,
    )

    _, reward, _, info = env.step(action)
    gt = [(g["item_id"], ViolationType(g["violation"])) for g in info["ground_truth"]]
    score = GRADERS[task](action, gt)
    reward_detail = info["reward_detail"]

    # Result display
    reward_emoji = "🎯" if reward > 0.5 else "⚠️" if reward > 0 else "❌"
    result_md = f"""### {reward_emoji} Results

| Metric | Value |
|--------|-------|
| **Reward** | `{reward:+.3f}` |
| **Grader Score** | `{score:.3f}` |
| **True Positives** | {reward_detail['true_positives']} |
| **False Positives** | {reward_detail['false_positives']} |
| **False Negatives** | {reward_detail['false_negatives']} |
| **Precision** | {reward_detail['precision']:.2f} |
| **Recall** | {reward_detail['recall']:.2f} |
| **F1** | {reward_detail['f1']:.2f} |
"""

    # Ground truth reveal
    if gt:
        gt_lines = ["### 🔍 Ground Truth Violations",
                    "| Item ID | Violation Type |",
                    "|---------|----------------|"]
        for item_id, vtype in gt:
            gt_lines.append(f"| `{item_id}` | `{vtype.value}` |")
        gt_md = "\n".join(gt_lines)
    else:
        gt_md = "### ✅ Ground Truth: Clean Report\n_This was a clean report — the correct decision was APPROVE._"

    return result_md, gt_md, reward_detail["explanation"]


def run_rule_agent(task: str, seed: int):
    """Run the rule-based agent and show its decision."""
    env_key = f"{task}:{seed}:env"
    obs_key = f"{task}:{seed}:obs"

    if obs_key not in _current_env:
        # Auto-load
        load_episode(task, seed)

    obs = _current_env[obs_key]
    action = rule_based_agent(obs)

    flagged_md = ""
    if action.flagged_items:
        rows = ["| Item ID | Violation Type | Reason | Confidence |",
                "|---------|----------------|--------|------------|"]
        for f in action.flagged_items:
            rows.append(f"| `{f.item_id}` | `{f.violation_type.value}` | {f.reason[:50]}… | {f.confidence:.0%} |")
        flagged_md = "### 🤖 Rule-Based Agent Decision\n\n" + "\n".join(rows)
    else:
        flagged_md = f"### 🤖 Rule-Based Agent Decision\n\n**Decision:** `{action.decision.value}` — No violations found."

    return flagged_md


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

TASK_OPTIONS = ["easy", "medium", "hard"]
DECISION_OPTIONS = ["Approve", "Flag", "Reject"]
VIOLATION_OPTIONS = [v.value for v in ViolationType]

CSS = """
.result-box { background: #0d1117; border-radius: 8px; padding: 16px; }
.title-box h1 { font-size: 2em; margin-bottom: 4px; }
"""

with gr.Blocks(title="🧾 Expense Audit OpenEnv") as demo:

    gr.Markdown("""# 🧾 Expense Audit OpenEnv
*An OpenEnv-compliant environment for training AI agents on expense report auditing*

---
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Episode Setup")
            task_dd = gr.Dropdown(choices=TASK_OPTIONS, value="easy", label="Task Difficulty")
            seed_sl = gr.Slider(minimum=0, maximum=99, step=1, value=42, label="Seed")
            load_btn = gr.Button("🔄 Load Episode", variant="primary")
            rule_btn = gr.Button("🤖 Show Rule-Based Agent Decision")

        with gr.Column(scale=3):
            task_desc = gr.Markdown("> Load an episode to begin.")

    with gr.Row():
        with gr.Column(scale=2):
            report_disp = gr.Markdown("_Load an episode to see the report._")
        with gr.Column(scale=1):
            policy_disp = gr.Markdown("_Policy will appear here._")

    with gr.Accordion("📁 Historical Reports & Vendor Stats (medium/hard tasks)", open=False):
        with gr.Row():
            history_disp = gr.Markdown("_Load a medium or hard episode to see history._")
            vendor_disp = gr.Markdown("_Load a hard episode to see vendor stats._")

    gr.Markdown("---\n### 🔎 Your Audit Decision")

    with gr.Row():
        with gr.Column(scale=2):
            items_check = gr.CheckboxGroup(choices=[], label="Select items to flag (check all violations)")
        with gr.Column(scale=1):
            decision_dd = gr.Dropdown(choices=DECISION_OPTIONS, value="Approve", label="Overall Decision")
            violation_dd = gr.Dropdown(choices=VIOLATION_OPTIONS, value="over_limit", label="Violation Type (for all flagged items)")
            reason_tb = gr.Textbox(placeholder="Reason for flagging...", label="Reason")
            submit_btn = gr.Button("📤 Submit Audit", variant="primary")

    rule_output = gr.Markdown("_Click 'Show Rule-Based Agent Decision' to see the automated agent's choice._")

    gr.Markdown("---\n### 📊 Results")
    with gr.Row():
        with gr.Column():
            result_disp = gr.Markdown("_Results will appear after submission._")
        with gr.Column():
            gt_disp = gr.Markdown("_Ground truth will be revealed after submission._")
    explanation_disp = gr.Textbox(label="Reward Detail", interactive=False)

    # Wire events
    load_btn.click(
        fn=load_episode,
        inputs=[task_dd, seed_sl],
        outputs=[task_desc, report_disp, policy_disp, history_disp, vendor_disp,
                 items_check, result_disp, explanation_disp],
    )

    rule_btn.click(
        fn=run_rule_agent,
        inputs=[task_dd, seed_sl],
        outputs=[rule_output],
    )

    submit_btn.click(
        fn=submit_audit,
        inputs=[task_dd, seed_sl, decision_dd, items_check, violation_dd, reason_tb],
        outputs=[result_disp, gt_disp, explanation_disp],
    )

    gr.Markdown("""---
### 📚 Quick Reference
**ViolationTypes:** `over_limit` · `missing_receipt` · `duplicate` · `split_receipt` · `round_number_fraud` · `vendor_collusion`  
**Reward:** +1.0 for perfect audit · 0.0 for approve-all · negative for hallucinating violations  
**API:** `POST /reset` → `POST /step` → `GET /state` · [Full Docs →](/docs)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft(primary_hue="green"))

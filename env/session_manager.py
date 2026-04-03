"""
Session manager for the Expense Audit HTTP API.

Allows multiple agents to run concurrent independent episodes
using session tokens, without sharing state.
"""
from __future__ import annotations
import uuid
import time
from typing import Dict, Optional

from env.environment import ExpenseAuditEnv, TASK_EASY
from env.models import CompanyPolicy


# Sessions expire after 30 minutes of inactivity
SESSION_TTL_SECONDS = 1800


class Session:
    def __init__(self, task: str, seed: Optional[int] = None):
        self.session_id = str(uuid.uuid4())
        self.env = ExpenseAuditEnv(task=task, seed=seed)
        self.created_at = time.time()
        self.last_used = time.time()
        self.episode_count = 0

    def touch(self):
        self.last_used = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_used) > SESSION_TTL_SECONDS


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create(self, task: str = TASK_EASY, seed: Optional[int] = None) -> Session:
        self._evict_expired()
        session = Session(task=task, seed=seed)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            del self._sessions[session_id]
            return None
        session.touch()
        return session

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)

    def _evict_expired(self):
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            del self._sessions[sid]

    @property
    def active_count(self) -> int:
        self._evict_expired()
        return len(self._sessions)


# Global singleton used by app.py
session_manager = SessionManager()

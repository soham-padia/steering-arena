"""Bounded scoring gate — caps concurrent NDIF forward passes so a burst of
submissions can't exceed NDIF's limits or stampede the Space (PROJECT_SPEC.md §9).

FastAPI runs sync endpoints in a threadpool, so a threading semaphore is enough.
"""

from __future__ import annotations

import threading
from typing import Callable, TypeVar

from app.errors import QueueFull

T = TypeVar("T")


class ScoringGate:
    def __init__(self, concurrency: int, acquire_timeout_s: float = 30.0):
        self._sem = threading.BoundedSemaphore(max(1, concurrency))
        self._timeout = acquire_timeout_s

    def run(self, fn: Callable[..., T], *args, **kwargs) -> T:
        acquired = self._sem.acquire(timeout=self._timeout)
        if not acquired:
            raise QueueFull("Scoring is busy right now — please try again in a moment.")
        try:
            return fn(*args, **kwargs)
        finally:
            self._sem.release()

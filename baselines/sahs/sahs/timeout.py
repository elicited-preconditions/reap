"""signal.SIGALRM timeout of one check, used by checker.py and occlusion.py (which does not go through
ActionFeasibilityChecker.check()).
"""
from __future__ import annotations

import signal


class CheckerTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise CheckerTimeout()


def run_with_timeout(fn, timeout_s: float, fallback):
    """Run fn() with a SIGALRM watchdog. On timeout, call fallback() and
    return its result instead of raising."""
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fn()
    except CheckerTimeout:
        return fallback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

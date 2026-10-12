"""Gemini free-tier limits: retry short rate limits, stop cleanly on daily quotas.

Per-minute limits ("429 ... retry in 20s") are retried with backoff. A daily limit
("PerDay", or a retry delay of hours) raises QuotaExhausted at once: retrying is
pointless, so callers pause the work and resume later.
"""
import re
import time


class QuotaExhausted(Exception):
    """A daily (or otherwise long) quota ran out: retrying now is pointless."""


def _is_long_quota(message: str) -> bool:
    if re.search(r"PerDay|per day|daily", message, re.I):
        return True
    # "Please retry in 11h11m16s" / "retryDelay': '40276s'" -> anything over 2 minutes
    hours = re.search(r"retry in (\d+)h", message)
    seconds = re.search(r"retryDelay'?:\s*'(\d+)s", message)
    return bool(hours) or bool(seconds and int(seconds.group(1)) > 120)


def _suggested_delay(message: str) -> float:
    """The server's own "retry in 23.5s" hint (per-minute limits), plus a margin."""
    match = re.search(r"retry in ([\d.]+)s", message) or re.search(r"retryDelay'?:\s*'(\d+)s", message)
    return float(match.group(1)) + 1 if match else 0.0


def with_retry(fn, attempts=5, base_delay=5, sleep=None):
    """Call fn(); retry on per-minute rate limits / transient errors with exponential backoff.

    Daily quota errors raise QuotaExhausted immediately instead of retrying.
    """
    sleep = sleep or time.sleep
    for i in range(attempts):
        try:
            return fn()
        except QuotaExhausted:
            raise
        except Exception as e:
            message = str(e)
            if re.search(r"429|RESOURCE_EXHAUSTED|quota", message, re.I) and _is_long_quota(message):
                raise QuotaExhausted(message.split("\n")[0][:300]) from e
            transient = re.search(r"429|rate|quota|exhausted|500|503|unavailable|timeout|deadline", message, re.I)
            if not transient or i == attempts - 1:
                raise
            wait = max(base_delay * 2 ** i, _suggested_delay(message))
            print(f"    transient error ({type(e).__name__}), retrying in {wait}s...")
            sleep(wait)

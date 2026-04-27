"""
Leo Trident — In-process Job Scheduler

Lightweight thread-based scheduler with cron-like syntax.
Supported schedule formats:
    - "every Ns" / "every Nm" / "every Nh"   (interval)
    - "@hourly"                               (top of every hour)
    - "@daily HH:MM"                          (daily at HH:MM local time)
    - "M H D Mo W"                            (5-field cron, * and N only)

Public API:
    register(name, schedule, fn)
    start(), stop()
    jobs() -> list[dict]

Errors in jobs are caught & logged. Each run is auto-reported via
service.metrics.log_metric("scheduler.run", duration_ms, {job, status}).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from src.service.metrics import log_metric

logger = logging.getLogger(__name__)


# ── schedule parsing ──────────────────────────────────────────────────────────

@dataclass
class _Schedule:
    """Internal: encapsulates a parsed schedule and computes next-run times."""
    raw: str
    kind: str               # "interval" | "daily" | "hourly" | "cron"
    interval_s: float = 0
    daily_hour: int = 0
    daily_min: int = 0
    cron_min: str = "*"
    cron_hour: str = "*"
    cron_dom: str = "*"
    cron_mon: str = "*"
    cron_dow: str = "*"

    def next_after(self, now: datetime) -> datetime:
        if self.kind == "interval":
            return now + timedelta(seconds=self.interval_s)
        if self.kind == "hourly":
            return (now + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0,
            )
        if self.kind == "daily":
            candidate = now.replace(
                hour=self.daily_hour, minute=self.daily_min,
                second=0, microsecond=0,
            )
            if candidate <= now:
                candidate += timedelta(days=1)
            return candidate
        if self.kind == "cron":
            cur = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            for _ in range(366 * 24 * 60):
                if self._cron_match(cur):
                    return cur
                cur += timedelta(minutes=1)
            return now + timedelta(days=366)
        raise ValueError(f"unknown schedule kind: {self.kind}")

    def _cron_match(self, dt: datetime) -> bool:
        def m(field_val: str, val: int) -> bool:
            if field_val == "*":
                return True
            if field_val.startswith("*/"):
                step = int(field_val[2:])
                return val % step == 0
            return val in {int(x) for x in field_val.split(",")}

        return (
            m(self.cron_min, dt.minute)
            and m(self.cron_hour, dt.hour)
            and m(self.cron_dom, dt.day)
            and m(self.cron_mon, dt.month)
            # cron convention: 0=Sunday … 6=Saturday
            # Python weekday(): 0=Monday … 6=Sunday
            # Convert: cron_day = (python_weekday + 1) % 7
            and m(self.cron_dow, (dt.weekday() + 1) % 7)
        )


def _parse_schedule(spec: str) -> _Schedule:
    s = spec.strip()
    if s.startswith("every "):
        rest = s[6:].strip()
        unit = rest[-1]
        try:
            n = float(rest[:-1])
        except ValueError as e:
            raise ValueError(f"bad interval: {spec}") from e
        mult = {"s": 1, "m": 60, "h": 3600}.get(unit)
        if mult is None:
            raise ValueError(f"bad interval unit in: {spec}")
        return _Schedule(raw=spec, kind="interval", interval_s=n * mult)
    if s == "@hourly":
        return _Schedule(raw=spec, kind="hourly")
    if s.startswith("@daily"):
        rest = s[len("@daily"):].strip() or "00:00"
        hh, mm = rest.split(":")
        return _Schedule(
            raw=spec, kind="daily",
            daily_hour=int(hh), daily_min=int(mm),
        )
    parts = s.split()
    if len(parts) == 5:
        return _Schedule(
            raw=spec, kind="cron",
            cron_min=parts[0], cron_hour=parts[1],
            cron_dom=parts[2], cron_mon=parts[3], cron_dow=parts[4],
        )
    raise ValueError(f"unrecognized schedule: {spec!r}")


# ── job registry ──────────────────────────────────────────────────────────────

@dataclass
class _Job:
    name: str
    schedule: _Schedule
    fn: Callable[[], object]
    next_run_at: datetime
    last_run_at: Optional[datetime] = None
    last_status: Optional[str] = None
    last_error: Optional[str] = None
    run_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


_JOBS: dict[str, _Job] = {}
_JOBS_LOCK = threading.Lock()
_RUNNER_THREAD: Optional[threading.Thread] = None
_STOP_EVENT = threading.Event()


def register(name: str, schedule: str, fn: Callable[[], object]) -> None:
    """Register a job. Replaces existing job with same name."""
    sched = _parse_schedule(schedule)
    now = datetime.now()
    job = _Job(name=name, schedule=sched, fn=fn,
               next_run_at=sched.next_after(now))
    with _JOBS_LOCK:
        _JOBS[name] = job
    logger.info("scheduler: registered job %s (%s)", name, schedule)


def unregister(name: str) -> None:
    with _JOBS_LOCK:
        _JOBS.pop(name, None)


def jobs() -> list[dict]:
    with _JOBS_LOCK:
        return [
            {
                "name": j.name,
                "schedule": j.schedule.raw,
                "next_run_at": j.next_run_at.isoformat() if j.next_run_at else None,
                "last_run_at": j.last_run_at.isoformat() if j.last_run_at else None,
                "last_status": j.last_status,
                "last_error": j.last_error,
                "run_count": j.run_count,
            }
            for j in _JOBS.values()
        ]


def _run_job(job: _Job) -> None:
    started = time.monotonic()
    status = "ok"
    err: Optional[str] = None
    try:
        job.fn()
    except Exception as e:  # noqa: BLE001
        status = "error"
        err = f"{type(e).__name__}: {e}"
        logger.exception("scheduler: job %s failed", job.name)
    duration_ms = (time.monotonic() - started) * 1000
    with job.lock:
        job.last_run_at = datetime.now()
        job.last_status = status
        job.last_error = err
        job.run_count += 1
        job.next_run_at = job.schedule.next_after(job.last_run_at)
    try:
        log_metric("scheduler.run", duration_ms,
                   {"job": job.name, "status": status})
    except Exception:  # pragma: no cover
        pass


def _runner_loop() -> None:
    while not _STOP_EVENT.is_set():
        now = datetime.now()
        due: list[_Job] = []
        with _JOBS_LOCK:
            for j in _JOBS.values():
                if j.next_run_at <= now:
                    # bump next_run_at immediately to prevent duplicate dispatch
                    j.next_run_at = j.schedule.next_after(now)
                    due.append(j)
        for j in due:
            t = threading.Thread(target=_run_job, args=(j,),
                                 name=f"sched-{j.name}", daemon=True)
            t.start()
        _STOP_EVENT.wait(0.2)


def start() -> None:
    """Start the scheduler thread (idempotent)."""
    global _RUNNER_THREAD
    if _RUNNER_THREAD and _RUNNER_THREAD.is_alive():
        return
    _STOP_EVENT.clear()
    _RUNNER_THREAD = threading.Thread(
        target=_runner_loop, name="leo-scheduler", daemon=True,
    )
    _RUNNER_THREAD.start()
    logger.info("scheduler: started")


def stop(timeout: float = 5.0) -> None:
    """Stop the scheduler thread."""
    global _RUNNER_THREAD
    _STOP_EVENT.set()
    if _RUNNER_THREAD:
        _RUNNER_THREAD.join(timeout=timeout)
        _RUNNER_THREAD = None
    logger.info("scheduler: stopped")


__all__ = ["register", "unregister", "start", "stop", "jobs"]

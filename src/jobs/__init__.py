"""
Leo Trident — Scheduled Jobs

Houses nightly/periodic background jobs and a `register_default_jobs()`
helper for the FastAPI startup hook (or any other bootstrap path).
"""
from __future__ import annotations

import logging

from src import scheduler
from src.jobs.consolidation import nightly_consolidation_job
from src.quality import daily_quality_job

logger = logging.getLogger(__name__)


def register_default_jobs() -> None:
    """Register the default set of recurring jobs with the in-process scheduler.

    Idempotent — `scheduler.register` replaces any existing job of the same
    name, so calling this multiple times is safe.
    """
    scheduler.register(
        "consolidation_nightly",
        "@daily 03:00",
        nightly_consolidation_job,
    )
    scheduler.register(
        "quality_daily",
        "@daily 04:00",
        daily_quality_job,
    )
    logger.info("jobs: registered default jobs")


__all__ = [
    "register_default_jobs",
    "nightly_consolidation_job",
    "daily_quality_job",
]

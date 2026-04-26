"""Tests for src.scheduler."""
import time

from src import scheduler


def test_scheduler_runs_interval_job():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1

    # clear any prior state
    scheduler.unregister("hb")
    scheduler.register("hb", "every 1s", fn)
    scheduler.start()
    try:
        time.sleep(2.5)
    finally:
        scheduler.stop()

    js = scheduler.jobs()
    [hb] = [j for j in js if j["name"] == "hb"]
    assert counter["n"] >= 1
    assert hb["last_status"] == "ok"
    assert hb["run_count"] >= 1
    scheduler.unregister("hb")


def test_scheduler_catches_errors():
    scheduler.unregister("boom")
    scheduler.register("boom", "every 1s", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    scheduler.start()
    try:
        time.sleep(1.5)
    finally:
        scheduler.stop()
    [j] = [x for x in scheduler.jobs() if x["name"] == "boom"]
    assert j["last_status"] == "error"
    assert "RuntimeError" in (j["last_error"] or "")
    scheduler.unregister("boom")


def test_parse_schedule_variants():
    from src.scheduler import _parse_schedule

    assert _parse_schedule("every 5m").interval_s == 300
    assert _parse_schedule("@hourly").kind == "hourly"
    s = _parse_schedule("@daily 03:30")
    assert s.kind == "daily" and s.daily_hour == 3 and s.daily_min == 30
    c = _parse_schedule("0 * * * *")
    assert c.kind == "cron" and c.cron_min == "0"

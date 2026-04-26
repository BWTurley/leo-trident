"""Tests for read_metrics + rollup_metric helpers."""
from src.service import metrics


def test_read_and_rollup(tmp_path):
    metrics.log_metric("foo", 1, base_path=tmp_path)
    metrics.log_metric("foo", 2, {"x": "y"}, base_path=tmp_path)
    metrics.log_metric("bar", 10, base_path=tmp_path)

    all_entries = metrics.read_metrics(base_path=tmp_path)
    assert len(all_entries) == 3

    foos = metrics.read_metrics(name_prefix="foo", base_path=tmp_path)
    assert len(foos) == 2

    assert metrics.rollup_metric("foo", agg="sum", base_path=tmp_path) == 3.0
    assert metrics.rollup_metric("foo", agg="avg", base_path=tmp_path) == 1.5
    assert metrics.rollup_metric("foo", agg="max", base_path=tmp_path) == 2.0
    assert metrics.rollup_metric("foo", agg="count", base_path=tmp_path) == 2.0
    assert metrics.rollup_metric("nope", agg="sum", base_path=tmp_path) == 0.0


def test_read_metrics_missing_day(tmp_path):
    assert metrics.read_metrics(date="1999-01-01", base_path=tmp_path) == []

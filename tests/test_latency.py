import pytest

from src.latency import percentile, summarize_latency


def test_percentile_uses_nearest_rank_without_dropping_samples():
    assert percentile([1, 2, 3, 4, 100], 95) == 100
    assert percentile([1, 2, 3, 4, 100], 99) == 100


def test_summary_counts_values_strictly_above_twice_p95():
    summary = summarize_latency([1, 2, 3, 4, 100])

    assert summary["count"] == 5
    assert summary["p95"] == 100
    assert summary["p99"] == 100
    assert summary["max"] == 100
    assert summary["outlier_threshold"] == 200
    assert summary["outlier_count"] == 0


def test_summary_counts_extreme_values_without_modifying_samples():
    values = [*range(1, 101), 1000]
    summary = summarize_latency(values)

    assert summary["max"] == 1000
    assert summary["p95"] == 96
    assert summary["outlier_threshold"] == 192
    assert summary["outlier_count"] == 1
    assert values[-1] == 1000


@pytest.mark.parametrize("percent", [-1, 101])
def test_percentile_rejects_invalid_percent(percent):
    with pytest.raises(ValueError):
        percentile([1, 2], percent)

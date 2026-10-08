import math
import statistics


def percentile(values, percent):
    """Return the nearest-rank percentile, retaining every input sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")

    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize_latency(values):
    """Summarize raw latencies and count extreme wall-time outliers."""
    if not values:
        raise ValueError("latency summary requires at least one value")

    p95 = percentile(values, 95)
    outlier_threshold = 2 * p95
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": p95,
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
        "outlier_threshold": outlier_threshold,
        "outlier_count": sum(value > outlier_threshold for value in values),
    }
